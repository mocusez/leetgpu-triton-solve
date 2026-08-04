
这段代码实现的是 **Scaled Dot-Product Attention 的反向传播**，输入为：

* `Q ∈ ℝ^{M×d}`
* `K, V ∈ ℝ^{N×d}`
* 上游梯度 `dO ∈ ℝ^{M×d}`

输出写入：

* `dQ ∈ ℝ^{M×d}`
* `dK, dV ∈ ℝ^{N×d}`

代码将反向传播拆成三个 Triton kernel，并用较大的全局临时矩阵保存 `S = QKᵀ·scale` 和 `dP = dO·Vᵀ`，避免后续 kernel 重复计算这两个大矩阵乘法。

---

# 一、先理解数学公式

前向注意力为：

[
S = \frac{QK^T}{\sqrt d}
]

[
P = \operatorname{softmax}(S)
]

[
O = PV
]

令：

[
\text{scale} = \frac{1}{\sqrt d}
]

给定 `dO`，反向传播依次是：

[
dV = P^T dO
]

[
dP = dO V^T
]

softmax 的梯度不能直接写成逐元素乘法。对每一行 (i)：

[
\delta_i = \sum_j P_{ij}dP_{ij}
]

[
dS_{ij} = P_{ij}(dP_{ij}-\delta_i)
]

由于原始 logits 是：

[
S = QK^T\cdot\text{scale}
]

所以代码把 `scale` 合并进 `ds`：

[
\widetilde{dS}=P\odot(dP-\delta)\cdot\text{scale}
]

最终：

[
dQ = \widetilde{dS}K
]

[
dK = \widetilde{dS}^TQ
]

代码中的三个 kernel 分别负责：

1. 计算并保存 `S`、`Sᵀ`、`dP`、`dPᵀ`，同时计算 softmax 统计量；
2. 根据保存的数据计算 `dQ`；
3. 根据转置布局的数据计算 `dK` 和 `dV`。

---

# 二、导入部分

```python
import torch
import triton
import triton.language as tl
from triton.runtime.errors import OutOfResources
```

对应文件开头。

### `import torch`

Host 端使用 PyTorch：

* 管理 GPU tensor；
* 做 padding；
* 创建临时缓冲区；
* 做转置和连续化；
* 最后把结果复制回用户提供的输出 tensor。

### `import triton`

用于：

* `@triton.jit` 编译 kernel；
* `triton.cdiv` 计算向上取整的 grid 大小；
* `triton.next_power_of_2` 计算 block 维度。

### `import triton.language as tl`

`tl` 是 Triton kernel 内的 DSL，包括：

* `tl.program_id`
* `tl.arange`
* `tl.load`
* `tl.store`
* `tl.dot`
* `tl.exp`
* `tl.max`
* `tl.sum`

### `OutOfResources`

当某个 block 配置占用太多寄存器、共享内存或其他 GPU 资源时，kernel 编译或启动可能抛出该异常。Host 端捕获它后尝试更小的 tile 配置。

---

# 三、Kernel 1：`_attn_bwd_pre`

## 3.1 Kernel 的职责

```python
@triton.jit(do_not_specialize=["N_true"])
def _attn_bwd_pre(
    Q, KT, VT, dO,
    S, ST, DP, DPT,
    Mp, Lp, Deltap,
    stride_qm, stride_kt, stride_vt, stride_dom,
    stride_sm, stride_stn, stride_dpm, stride_dptn,
    M, N_true, D, sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
```



它一次处理一个 query 行块，并遍历全部 key 块。

输入布局大致为：

* `Q`: `[M, D]`
* `KT = Kᵀ`: `[D, N]`
* `VT = Vᵀ`: `[D, N]`
* `dO`: `[M, D]`

输出临时矩阵：

* `S`: `[M, N]`
* `ST`: `[N, M]`
* `DP`: `[M, N]`
* `DPT`: `[N, M]`

行统计量：

* `Mp`: 每行最大值 (m_i)
* `Lp`: 每行指数和 (l_i)
* `Deltap`: 每行 (\delta_i=\sum_jP_{ij}dP_{ij})

### 为什么同时保存自然布局和转置布局？

`dQ` 计算按 query 行遍历，需要连续读取：

```text
S[m, :]
DP[m, :]
```

所以使用 `S` 和 `DP`。

`dK/dV` 计算按 key 行遍历，需要连续读取：

```text
S[:, n]
DP[:, n]
```

如果直接读自然布局，会形成跨行、非连续访问。因此预先保存：

```text
ST[n, :]
DPT[n, :]
```

让第三个 kernel 也能合并访问显存。

### `do_not_specialize=["N_true"]`

它告诉 Triton 不要根据 `N_true` 的具体数值生成不同的 JIT 特化版本。

这里：

* `N_true` 是未 padding 的真实 key 数量；
* 实际分配和循环使用的是 padding 后的 `N2`；
* `N_true` 仅用于屏蔽 padding 列。

---

## 3.2 确定当前 program 处理的坐标

```python
pid = tl.program_id(0)
offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
offs_d = tl.arange(0, BLOCK_D)
mask_m = offs_m < M
mask_d = offs_d < D
mask_md = mask_m[:, None] & mask_d[None, :]
```



### `pid = tl.program_id(0)`

获取一维 launch grid 中当前 Triton program 的编号。

一个 Triton program 可以近似理解为一个 CUDA thread block，但两者并非完全相同的抽象。

### `offs_m`

```python
offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
```

当前 program 处理的 query 行号。

例如：

```text
BLOCK_M = 16
pid = 3
```

则：

```text
offs_m = [48, 49, ..., 63]
```

### `offs_d`

```python
offs_d = [0, 1, ..., BLOCK_D-1]
```

表示 head dimension 的列号。

当前 program 会把一块 `Q` 加载成：

```text
[BLOCK_M, BLOCK_D]
```

### `mask_m`

处理最后一个不完整 query block。

虽然 Host 端通常把 `M` padding 到 16 的倍数，但 kernel 仍保留边界检查。

### `mask_d`

`BLOCK_D` 不一定等于真实 `D`。

例如：

```text
D = 96
BLOCK_D = 128
```

则 `offs_d >= 96` 的位置必须屏蔽。

### `mask_md`

通过广播形成二维 mask：

```text
[BLOCK_M, 1] AND [1, BLOCK_D]
    ↓
[BLOCK_M, BLOCK_D]
```

---

## 3.3 加载 `Q` 和 `dO`

```python
q = tl.load(
    Q + offs_m[:, None] * stride_qm + offs_d[None, :],
    mask=mask_md,
    other=0.0
)
do = tl.load(
    dO + offs_m[:, None] * stride_dom + offs_d[None, :],
    mask=mask_md,
    other=0.0
)
```



对二维连续 tensor：

```text
地址 = 基地址 + 行号 × 行步长 + 列号
```

因此：

```python
Q + offs_m[:, None] * stride_qm + offs_d[None, :]
```

生成 `[BLOCK_M, BLOCK_D]` 的指针矩阵。

越界位置填 `0.0`，这样 padding 维度不会影响矩阵乘法。

---

## 3.4 初始化在线 softmax 统计量

```python
m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
l_i = tl.zeros([BLOCK_M], tl.float32)
delta = tl.zeros([BLOCK_M], tl.float32)
```



每个 query 行维护三个量。

### `m_i`

当前已经扫描过的 key 块中的最大 logit：

[
m_i=\max_j S_{ij}
]

初始化为负无穷。

### `l_i`

相对于当前最大值的指数和：

[
l_i=\sum_j e^{S_{ij}-m_i}
]

### `delta`

这里暂时存储未归一化的：

[
\sum_j e^{S_{ij}-m_i}dP_{ij}
]

循环结束后再除以 `l_i`，得到：

[
\frac{\sum_j e^{S_{ij}-m_i}dP_{ij}}
{\sum_j e^{S_{ij}-m_i}}
=======================

\sum_jP_{ij}dP_{ij}
]

---

## 3.5 遍历 key 块

```python
for start_n in range(0, stride_sm, BLOCK_N):
```



这里的 `stride_sm` 是临时矩阵 `S` 的行步长。

由于 `S` 是连续二维 tensor：

```python
S.shape == (m_chunk, N2)
S.stride(0) == N2
```

所以这里实际上是在做：

```python
for start_n in range(0, N2, BLOCK_N):
```

即扫描全部 padding 后的 key 列。

这种写法依赖 `S` 是连续分配的；Host 端确实通过 `torch.empty((m_chunk, N2))` 保证了这一点。

---

## 3.6 当前 key block 的索引和 mask

```python
offs_n = start_n + tl.arange(0, BLOCK_N)
mask_n = offs_n < stride_sm
mask_dn = mask_d[:, None] & mask_n[None, :]
```

`offs_n` 是当前 key block 的列号。

`mask_dn` 形状为：

```text
[BLOCK_D, BLOCK_N]
```

用于加载 `KT` 和 `VT`。

---

## 3.7 加载转置后的 `K` 和 `V`

```python
kt = tl.load(
    KT + offs_d[:, None] * stride_kt + offs_n[None, :],
    mask=mask_dn,
    other=0.0
)
vt = tl.load(
    VT + offs_d[:, None] * stride_vt + offs_n[None, :],
    mask=mask_dn,
    other=0.0
)
```

得到：

```text
kt: [BLOCK_D, BLOCK_N]
vt: [BLOCK_D, BLOCK_N]
```

因为 Host 端提前执行：

```python
KT = Kp.t().contiguous()
VT = Vp.t().contiguous()
```

所以这里读取的是连续的 `[D, N]` 布局，而不是对原始 `K/V` 做跨步读取。

---

## 3.8 计算 `S` 和 `dP`

```python
s = tl.dot(q, kt, input_precision="ieee") * sm_scale
s = tl.where(offs_n[None, :] < N_true, s, float("-inf"))
dp = tl.dot(do, vt, input_precision="ieee")
```



形状为：

```text
q   : [BLOCK_M, BLOCK_D]
kt  : [BLOCK_D, BLOCK_N]
s   : [BLOCK_M, BLOCK_N]
```

所以：

[
s=qk^T\cdot\text{scale}
]

同理：

```text
do : [BLOCK_M, BLOCK_D]
vt : [BLOCK_D, BLOCK_N]
dp : [BLOCK_M, BLOCK_N]
```

即：

[
dP=dOV^T
]

### `input_precision="ieee"`

在支持 TF32 的 NVIDIA GPU 上，Triton 的 float32 `tl.dot` 可能采用较低精度的硬件路径。

指定 `"ieee"` 表示要求更接近标准 IEEE float32 的乘法精度，通常精度更高，但可能比 TF32 路径慢。

### 为什么只对 `s` 屏蔽 `N_true`？

```python
s = tl.where(offs_n < N_true, s, -inf)
```

padding 出来的 key 不能进入 softmax。

设置为 `-inf` 后：

[
e^{-\infty}=0
]

所以它们的 softmax 概率严格为零。

`dp` 对 padding 列不必显式改成零，因为 padding 后的 `V` 本身为零，因此：

[
dO\cdot V_{\text{padding}}^T=0
]

不过即使 `dp` 有值，只要 `p=0`，最终也不会产生梯度贡献。

---

## 3.9 在线 softmax 更新

```python
m_new = tl.maximum(m_i, tl.max(s, 1))
alpha = tl.exp(m_i - m_new)
p = tl.exp(s - m_new[:, None])
l_i = l_i * alpha + tl.sum(p, 1)
delta = delta * alpha + tl.sum(p * dp, 1)
m_i = m_new
```



这是整个 kernel 最重要的部分。

假设之前已经扫描了一些 key block，旧统计量是：

[
m_{\text{old}}
]

[
l_{\text{old}}
=\sum_{\text{old }j}e^{s_j-m_{\text{old}}}
]

新 block 最大值为：

[
m_{\text{block}}=\max_{\text{new }j}s_j
]

新的全局最大值：

[
m_{\text{new}}=\max(m_{\text{old}},m_{\text{block}})
]

### `alpha`

```python
alpha = exp(m_i - m_new)
```

旧指数和是以旧最大值为基准的。改成新最大值后必须重新缩放：

[
e^{s-m_{\text{old}}}
\cdot e^{m_{\text{old}}-m_{\text{new}}}
=======================================

e^{s-m_{\text{new}}}
]

因此：

[
\alpha=e^{m_{\text{old}}-m_{\text{new}}}
]

### 当前 block 的指数

```python
p = exp(s - m_new)
```

注意这里的 `p` 还不是最终 softmax 概率，因为尚未除以 `l_i`。

它只是：

[
\tilde p=e^{s-m_{\text{new}}}
]

### 更新指数和

```python
l_i = l_i * alpha + sum(p)
```

即：

[
l_{\text{new}}
==============

l_{\text{old}}e^{m_{\text{old}}-m_{\text{new}}}
+
\sum_{\text{new }j}e^{s_j-m_{\text{new}}}
]

### 更新 `delta` 的分子

```python
delta = delta * alpha + sum(p * dp)
```

维护：

[
r_i=\sum_j e^{s_{ij}-m_i}dP_{ij}
]

最后：

[
\delta_i=\frac{r_i}{l_i}
=\sum_jP_{ij}dP_{ij}
]

这允许 kernel 在不保存完整 softmax `P` 的情况下得到 softmax 反向需要的行归约量。

---

## 3.10 保存自然布局和转置布局

```python
mask_mn = mask_m[:, None] & mask_n[None, :]
tl.store(S + ..., s, mask=mask_mn)
tl.store(DP + ..., dp, mask=mask_mn)

st = tl.trans(s)
dpt = tl.trans(dp)

mask_nm = mask_n[:, None] & mask_m[None, :]
tl.store(ST + ..., st, mask=mask_nm)
tl.store(DPT + ..., dpt, mask=mask_nm)
```



`s` 和 `dp` 是：

```text
[BLOCK_M, BLOCK_N]
```

转置后：

```text
[BLOCK_N, BLOCK_M]
```

于是同一批计算结果被写入四个缓冲区：

| 缓冲区   | 布局      | 后续用途       |
| ----- | ------- | ---------- |
| `S`   | `[M,N]` | 计算 `dQ`    |
| `DP`  | `[M,N]` | 计算 `dQ`    |
| `ST`  | `[N,M]` | 计算 `dK/dV` |
| `DPT` | `[N,M]` | 计算 `dK/dV` |

代价是占用大量显存，但好处是：

* 不重复计算 `QKᵀ`；
* 不重复计算 `dOVᵀ`；
* 两个后续 kernel 都可以连续访问显存。

---

## 3.11 保存行统计量

```python
l_safe = tl.where(l_i == 0.0, 1.0, l_i)
tl.store(Mp + offs_m, m_i, mask=mask_m)
tl.store(Lp + offs_m, l_i, mask=mask_m)
tl.store(Deltap + offs_m, delta / l_safe, mask=mask_m)
```



### `l_safe`

理论上，只要每行至少有一个有效 key，`l_i > 0`。

但为了避免异常输入或全屏蔽行导致除零，代码使用：

```python
l_safe = 1 if l_i == 0 else l_i
```

### `Mp`

保存：

[
m_i=\max_jS_{ij}
]

### `Lp`

保存：

[
l_i=\sum_je^{S_{ij}-m_i}
]

后续可以重建 softmax：

[
P_{ij}=\frac{e^{S_{ij}-m_i}}{l_i}
]

### `Deltap`

此前 `delta` 是未归一化分子，因此最终保存：

[
\frac{\sum_je^{S_{ij}-m_i}dP_{ij}}
{\sum_je^{S_{ij}-m_i}}
======================

\sum_jP_{ij}dP_{ij}
]

---

# 四、Kernel 2：计算 `dQ`

函数定义：

```python
@triton.jit
def _attn_bwd_dq(
    K, S, DP, Mp, Lp, Deltap, DQ,
    stride_kn, stride_sm, stride_dpm, stride_dqm,
    M, D, sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
```



一个 program 负责一个 query block，遍历全部 key block。

目标公式：

[
dQ =
\left[
P\odot(dP-\delta)
\right]K\cdot\text{scale}
]

---

## 4.1 当前 query block

```python
pid = tl.program_id(0)
offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
offs_d = tl.arange(0, BLOCK_D)
mask_m = offs_m < M
mask_d = offs_d < D
```



与第一个 kernel 相同：

* `offs_m`：当前 query 行；
* `offs_d`：head dimension；
* `mask_m/mask_d`：处理边界。

---

## 4.2 加载 softmax 行统计量

```python
m_i = tl.load(Mp + offs_m, mask=mask_m, other=0.0)
l_i = tl.load(Lp + offs_m, mask=mask_m, other=1.0)
delta = tl.load(Deltap + offs_m, mask=mask_m, other=0.0)
l_safe = tl.where(l_i == 0.0, 1.0, l_i)
```



这些量用于从保存的 `S` 重建 `P`：

[
P=\frac{e^{S-m}}{l}
]

并计算：

[
dS=P(dP-\delta)\cdot\text{scale}
]

越界 query 行的默认值只是为了避免非法数值；对应位置最终不会写回。

---

## 4.3 初始化 `dQ` 累加器

```python
acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
```



最终：

```text
acc.shape = [BLOCK_M, BLOCK_D]
```

对应当前 query block 的 `dQ`。

使用 float32 累加，即使输入 `Q/K/V/dO` 是 float16 或 bfloat16，也可减少长维度归约时的误差。

---

## 4.4 遍历 key block

```python
for start_n in range(0, stride_sm, BLOCK_N):
    offs_n = start_n + tl.arange(0, BLOCK_N)
    mask_n = offs_n < stride_sm
    mask_mn = mask_m[:, None] & mask_n[None, :]
```

这里仍然利用：

```text
stride_sm == N2
```

扫描全部 key 列。

---

## 4.5 加载 `S`、`dP` 和 `K`

```python
s = tl.load(
    S + offs_m[:, None] * stride_sm + offs_n[None, :],
    mask=mask_mn,
    other=float("-inf")
)
dp = tl.load(
    DP + offs_m[:, None] * stride_dpm + offs_n[None, :],
    mask=mask_mn,
    other=0.0
)
k = tl.load(
    K + offs_n[:, None] * stride_kn + offs_d[None, :],
    mask=mask_n[:, None] & mask_d[None, :],
    other=0.0
)
```



形状：

```text
s  : [BLOCK_M, BLOCK_N]
dp : [BLOCK_M, BLOCK_N]
k  : [BLOCK_N, BLOCK_D]
```

因此接下来可以做：

```text
[BLOCK_M, BLOCK_N] @ [BLOCK_N, BLOCK_D]
→ [BLOCK_M, BLOCK_D]
```

### 越界 `s` 为什么用 `-inf`？

因为后面要执行：

```python
exp(s - m)
```

若填零，越界位置可能产生非零概率；填 `-inf` 后指数严格为零。

---

## 4.6 重建 softmax 并计算 `dS`

```python
p = tl.exp(s - m_i[:, None]) / l_safe[:, None]
ds = p * (dp - delta[:, None]) * sm_scale
```



第一行重建：

[
P_{ij}=\frac{e^{S_{ij}-m_i}}{l_i}
]

第二行计算：

[
ds_{ij}
=======

P_{ij}
\left(
dP_{ij}-\sum_kP_{ik}dP_{ik}
\right)
\cdot\text{scale}
]

这里的 `ds` 实际上已经包含从 `QKᵀ·scale` 对 `QKᵀ` 求导产生的 `scale`。

---

## 4.7 累加 `dQ`

```python
acc = tl.dot(ds, k, acc, input_precision="ieee")
```



等价于：

```python
acc = acc + ds @ k
```

每次只处理一个 key block，遍历所有 block 后：

[
acc=\sum_{\text{key blocks}}ds_{\text{block}}K_{\text{block}}
=dQ
]

Triton 的三参数 `tl.dot(a, b, acc)` 形式允许直接把矩阵乘结果加入已有累加器。

---

## 4.8 写回 `dQ`

```python
tl.store(
    DQ + offs_m[:, None] * stride_dqm + offs_d[None, :],
    acc,
    mask=mask_m[:, None] & mask_d[None, :]
)
```



仅写有效的 query 行和 head dimension。

---

# 五、Kernel 3：计算 `dK` 和 `dV`

函数定义：

```python
@triton.jit
def _attn_bwd_dkdv(
    Q, dO, ST, DPT, Mp, Lp, Deltap, DK, DV,
    stride_qm, stride_dom, stride_stn, stride_dptn,
    stride_dkn, stride_dvn,
    M, N, D, sm_scale, accumulate,
    BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
):
```



一个 program 负责一个 key block，遍历全部 query block。

目标公式：

[
dV=P^TdO
]

[
dK=dS^TQ
]

这里使用 `ST` 和 `DPT`，即：

```text
ST  = Sᵀ
DPT = dPᵀ
```

所以当前 tile 直接按 `[BLOCK_N, BLOCK_M]` 读取。

---

## 5.1 当前 key block

```python
pid = tl.program_id(0)
offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
offs_d = tl.arange(0, BLOCK_D)
mask_n = offs_n < N
mask_d = offs_d < D
```



与 `dQ` kernel 相反：

* `dQ`：一个 program 对应 query block；
* `dK/dV`：一个 program 对应 key block。

因为 `dK` 和 `dV` 的输出形状都是 `[N,D]`。

---

## 5.2 初始化两个累加器

```python
acc_dk = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
acc_dv = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
```



每个 program 同时计算当前 key block 的：

```text
dK: [BLOCK_N, BLOCK_D]
dV: [BLOCK_N, BLOCK_D]
```

这样可以共享：

* `P`
* `dS`
* `Q`
* `dO`
* softmax 统计量

减少重复显存访问。

---

## 5.3 遍历 query block

```python
for start_m in range(0, M, BLOCK_M):
    offs_m = start_m + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    mask_nm = mask_n[:, None] & mask_m[None, :]
```



每次处理：

```text
BLOCK_N 个 key
×
BLOCK_M 个 query
```

对应 `Sᵀ` 中的一个 tile。

---

## 5.4 加载转置后的 `S` 和 `dP`

```python
st = tl.load(
    ST + offs_n[:, None] * stride_stn + offs_m[None, :],
    mask=mask_nm,
    other=float("-inf")
)
dpt = tl.load(
    DPT + offs_n[:, None] * stride_dptn + offs_m[None, :],
    mask=mask_nm,
    other=0.0
)
```



形状：

```text
st  : [BLOCK_N, BLOCK_M]
dpt : [BLOCK_N, BLOCK_M]
```

它们分别对应：

[
S^T_{n,m}=S_{m,n}
]

[
dP^T_{n,m}=dP_{m,n}
]

因为第二维 `offs_m` 连续，所以读取是 coalesced 的。

---

## 5.5 加载每个 query 行的统计量

```python
m_i = tl.load(Mp + offs_m, mask=mask_m, other=0.0)
l_i = tl.load(Lp + offs_m, mask=mask_m, other=1.0)
delta = tl.load(Deltap + offs_m, mask=mask_m, other=0.0)
l_safe = tl.where(l_i == 0.0, 1.0, l_i)
```



虽然 `st` 是转置布局，但 softmax 归一化仍然按原始 query 行进行。

因此广播方向与 `dQ` kernel 不同：

* `dQ` 中统计量形状扩成 `[BLOCK_M, 1]`
* 这里扩成 `[1, BLOCK_M]`

---

## 5.6 加载 `Q` 和 `dO`

```python
q = tl.load(
    Q + offs_m[:, None] * stride_qm + offs_d[None, :],
    mask=mask_m[:, None] & mask_d[None, :],
    other=0.0
)
do = tl.load(
    dO + offs_m[:, None] * stride_dom + offs_d[None, :],
    mask=mask_m[:, None] & mask_d[None, :],
    other=0.0
)
```



形状：

```text
q  : [BLOCK_M, BLOCK_D]
do : [BLOCK_M, BLOCK_D]
```

---

## 5.7 重建转置形式的 `P` 和 `dS`

```python
p = tl.exp(st - m_i[None, :]) / l_safe[None, :]
ds = p * (dpt - delta[None, :]) * sm_scale
```



这里 `p` 的形状是：

```text
[BLOCK_N, BLOCK_M]
```

它实际上是当前 tile 的 (P^T)：

[
p_{n,m}=P_{m,n}
]

同理：

[
ds_{n,m}=
P_{m,n}(dP_{m,n}-\delta_m)\cdot\text{scale}
]

也就是当前 tile 的 (dS^T)。

---

## 5.8 同时累加 `dV` 和 `dK`

```python
acc_dv = tl.dot(p, do, acc_dv, input_precision="ieee")
acc_dk = tl.dot(ds, q, acc_dk, input_precision="ieee")
```



形状：

```text
p  : [BLOCK_N, BLOCK_M]
do : [BLOCK_M, BLOCK_D]
```

因此：

[
acc_dv += P^TdO
]

即：

[
dV=P^TdO
]

同理：

```text
ds : [BLOCK_N, BLOCK_M]
q  : [BLOCK_M, BLOCK_D]
```

所以：

[
acc_dk += dS^TQ
]

即：

[
dK=dS^TQ
]

---

## 5.9 多 chunk 时累加已有结果

```python
mask_nd = mask_n[:, None] & mask_d[None, :]
if accumulate != 0:
    prev_k = tl.load(DK + ..., mask=mask_nd, other=0.0)
    prev_v = tl.load(DV + ..., mask=mask_nd, other=0.0)
    acc_dk += prev_k
    acc_dv += prev_v
```



Host 端可能把 `M` 切成多个 chunk。

对每个 chunk：

[
dK_{\text{total}}
=================

\sum_c dK_c
]

[
dV_{\text{total}}
=================

\sum_c dV_c
]

第一个 chunk 直接写结果。

从第二个 chunk 开始：

1. 加载之前的 `dK/dV`；
2. 加入当前 chunk 的贡献；
3. 写回。

这里不需要 atomic，因为同一轮 kernel launch 中：

* 每个 program 负责不同的 key block；
* 不同 program 不会写同一行 `dK/dV`。

而不同 chunk 的 kernel launch 是按顺序执行的。

---

## 5.10 写回

```python
tl.store(DK + ..., acc_dk, mask=mask_nd)
tl.store(DV + ..., acc_dv, mask=mask_nd)
```



写入当前 key block 的梯度。

---

# 六、Host 端常量与辅助函数

## 6.1 Scratch 空间预算

```python
_SCRATCH_ELEMS = 1 << 27
_PRE_CONFIGS = [(16, 32), (16, 16)]
_DQ_CONFIGS = [(32, 32), (16, 32)]
_DKDV_CONFIGS = [(32, 32), (32, 16)]
```



### `_SCRATCH_ELEMS`

[
1 \ll 27 = 134,217,728
]

个 float32 元素。

每个 float32 是 4 字节，因此单个矩阵最多：

[
134,217,728\times4
=536,870,912\text{ bytes}
=512\text{ MiB}
]

注意代码同时分配四个同样形状的矩阵：

* `S`
* `ST`
* `DP`
* `DPT`

因此若真的达到预算上限，仅这四个矩阵合计约为：

[
4\times512\text{ MiB}=2\text{ GiB}
]

另外还有 `Q/K/V`、梯度和统计缓冲区。

### 配置列表

例如：

```python
_PRE_CONFIGS = [(16, 32), (16, 16)]
```

分别表示尝试：

```text
BLOCK_M=16, BLOCK_N=32
BLOCK_M=16, BLOCK_N=16
```

优先尝试较大的 tile；资源不足时退回较小 tile。

---

## 6.2 `_pad16`

```python
def _pad16(n):
    return (n + 15) // 16 * 16
```



把 `n` 向上取整到 16 的倍数。

例如：

```text
1  → 16
16 → 16
17 → 32
31 → 32
```

公式是：

[
\left\lceil\frac n{16}\right\rceil\times16
]

---

# 七、三个 launch 包装器

以 `_launch_pre` 为例：

```python
def _launch_pre(grid, args, M, N_true, D, sm_scale, BLOCK_D):
    for bm, bn in _PRE_CONFIGS:
        try:
            _attn_bwd_pre[grid(bm)](
                *args, M, N_true, D, sm_scale,
                BLOCK_M=bm,
                BLOCK_N=bn,
                BLOCK_D=BLOCK_D,
                num_warps=8,
                num_stages=1
            )
            return
        except OutOfResources:
            continue
```



逻辑是：

1. 遍历候选 tile 配置；
2. 尝试编译或启动；
3. 成功后立即 `return`；
4. 若资源不足，尝试下一组。

### `grid(bm)`

调用处传入：

```python
lambda bm: (triton.cdiv(mc, bm),)
```

所以 grid 是：

[
\left(
\left\lceil\frac{mc}{BLOCK_M}\right\rceil,
\right)
]

即每个 query block 启动一个 program。

### `num_warps=8`

每个 Triton program 使用 8 个 warp，也就是 NVIDIA GPU 上通常对应 256 个线程的执行资源。

这不是说程序中显式存在 256 个 Python/Triton 线程；Triton 编译器会把 tile 运算映射到这些 warp 上。

### `num_stages=1`

控制软件流水线阶段数量。

这里使用 1，通常是为了降低共享内存或寄存器压力，而不是追求更深的加载—计算流水线。

`_launch_dq` 和 `_launch_dkdv` 使用完全相同的回退思想。

一个值得注意的边界情况是：如果所有配置都触发 `OutOfResources`，这些函数会直接走到末尾而不抛出新的异常，也不会显式报告失败。更稳健的实现通常会在循环结束后重新抛出错误。

---

# 八、`solve` 主函数

```python
def solve(
    Q, K, V, dO,
    dQ, dK, dV,
    M, N, d
):
```



这里的输出 tensor `dQ/dK/dV` 由调用方预先创建，函数负责原地写入。

---

## 8.1 保证输入连续

```python
Q = Q.contiguous()
K = K.contiguous()
V = V.contiguous()
dO = dO.contiguous()
```



kernel 的地址计算只显式传入了行 stride，没有传入列 stride。

也就是说它默认：

```text
stride(1) == 1
```

因此必须保证最后一维连续。

如果输入原本已经连续，`.contiguous()` 通常不会复制；否则会新建连续副本。

---

## 8.2 缩放系数与 `BLOCK_D`

```python
sm_scale = 1.0 / (d ** 0.5)
BLOCK_D = 128 if d <= 128 else triton.next_power_of_2(d)
dev = Q.device
f32 = torch.float32
```



### `sm_scale`

[
\text{sm_scale}=\frac1{\sqrt d}
]

### `BLOCK_D`

如果：

```text
d <= 128
```

统一使用：

```text
BLOCK_D = 128
```

否则使用大于等于 `d` 的最小 2 的幂。

例如：

```text
d=64  → BLOCK_D=128
d=128 → BLOCK_D=128
d=160 → BLOCK_D=256
```

这样有利于生成规则的矩阵 tile，但当 `d` 很小时可能产生较多无效 lane。例如 `d=32` 仍会使用 128 列的 tile，其中后 96 列依靠 mask 屏蔽。

### 输出中间结果使用 float32

`dQp/dKp/dVp` 和 scratch 均使用 float32，提高累加精度。

---

## 8.3 Padding

```python
M2, N2, d2 = _pad16(M), _pad16(N), _pad16(d)
padded = (M2, N2, d2) != (M, N, d)
```



将三个维度全部补到 16 的倍数。

这简化了：

* tile 边界；
* JIT 形状管理；
* 矩阵乘硬件对齐；
* scratch 布局。

---

## 8.4 创建 padding 后的张量

```python
if padded:
    Qp = Q.new_zeros((M2, d2)); Qp[:M, :d].copy_(Q)
    Kp = K.new_zeros((N2, d2)); Kp[:N, :d].copy_(K)
    Vp = V.new_zeros((N2, d2)); Vp[:N, :d].copy_(V)
    dOp = dO.new_zeros((M2, d2)); dOp[:M, :d].copy_(dO)

    dQp = torch.empty((M2, d2), device=dev, dtype=f32)
    dKp = torch.empty((N2, d2), device=dev, dtype=f32)
    dVp = torch.empty((N2, d2), device=dev, dtype=f32)
```



padding 输入全部初始化为零。

这非常关键：

* padding 的 `K/V` 行为零；
* padding 的 `Q/dO` 行为零；
* padding 的 head dimension 也为零。

因此即使部分 kernel 对 padding 区域做计算，其贡献通常也是零。

真实数据复制到左上角：

```text
Qp[:M, :d] = Q
Kp[:N, :d] = K
...
```

梯度临时张量使用 float32。

---

## 8.5 不需要 padding 时复用输出

```python
else:
    Qp, Kp, Vp, dOp = Q, K, V, dO
    dQp = dQ if dQ.is_contiguous() else torch.empty(...)
    dKp = dK if dK.is_contiguous() else torch.empty(...)
    dVp = dV if dV.is_contiguous() else torch.empty(...)
```



若输出 tensor 连续，kernel 直接写用户提供的输出。

若输出不连续：

1. 创建连续临时 tensor；
2. kernel 写临时 tensor；
3. 最后通过切片 `copy_` 写回原输出。

同样是因为 kernel 默认列 stride 为 1。

---

## 8.6 显式转置 `K` 和 `V`

```python
KT = Kp.t().contiguous()
VT = Vp.t().contiguous()
```



原始：

```text
Kp: [N2, d2]
Vp: [N2, d2]
```

转置并连续化后：

```text
KT: [d2, N2]
VT: [d2, N2]
```

这样第一个 kernel 可直接计算：

```text
Q tile @ KT tile
dO tile @ VT tile
```

并获得连续内存读取。

代价是：

* 额外两份显存；
* 两次转置复制。

---

# 九、按 `M` 分 chunk

```python
m_chunk = (
    M2
    if M2 * N2 <= _SCRATCH_ELEMS
    else max(16, (_SCRATCH_ELEMS // N2) // 16 * 16)
)
multi = m_chunk < M2
```



scratch 的每个矩阵需要：

[
m_{\text{chunk}}\times N2
]

个元素。

如果：

[
M2\times N2\le_SCRATCH_ELEMS
]

则一次处理全部 query。

否则选择满足预算的最大 16 倍数：

[
m_{\text{chunk}}
================

\left\lfloor
\frac{_SCRATCH_ELEMS/N2}{16}
\right\rfloor\times16
]

并保证至少为 16。

### 为什么只切 `M`？

因为：

* `dQ` 天然按 query 行独立；
* `S/DP` 的 scratch 大小与 query 数量线性相关；
* 每个 query chunk 都必须遍历完整的 `N`；
* `dK/dV` 需要把不同 query chunk 的贡献相加。

---

# 十、分配临时缓冲区

```python
S = torch.empty((m_chunk, N2), ...)
ST = torch.empty((N2, m_chunk), ...)
DP = torch.empty((m_chunk, N2), ...)
DPT = torch.empty((N2, m_chunk), ...)
Mp = torch.empty(m_chunk, ...)
Lp = torch.empty(m_chunk, ...)
Deltap = torch.empty(m_chunk, ...)
```



scratch 只按最大 chunk 分配一次，然后每轮复用。

它们不需要初始化，因为每个有效位置都会在 `_attn_bwd_pre` 中被覆盖。

---

## 10.1 多 chunk 时清零 `dK/dV`

```python
if multi:
    dKp.zero_()
    dVp.zero_()
```



从严格的数据流看，第一个 chunk 的 `_attn_bwd_dkdv` 使用 `accumulate=0`，会完整覆盖有效的 `dK/dV`，因此清零并不是计算正确性的必要条件。

但清零可以：

* 避免未覆盖 padding 区域含随机数据；
* 提供更明确的初始状态；
* 防止未来修改累加逻辑时出现问题。

---

# 十一、主 chunk 循环

```python
for m0 in range(0, M2, m_chunk):
    mc = min(m_chunk, M2 - m0)
    q_c = Qp[m0:m0 + mc]
    do_c = dOp[m0:m0 + mc]
    dq_c = dQp[m0:m0 + mc]
```



* `m0`：当前 chunk 的起始 query 行；
* `mc`：当前 chunk 实际行数；
* `q_c/do_c/dq_c`：当前 chunk 的视图。

由于这是对连续 tensor 第一维做切片，列仍然连续。

---

# 十二、启动预处理 kernel

```python
pre_args = (
    q_c, KT, VT, do_c,
    S, ST, DP, DPT,
    Mp, Lp, Deltap,
    q_c.stride(0),
    KT.stride(0),
    VT.stride(0),
    do_c.stride(0),
    S.stride(0),
    ST.stride(0),
    DP.stride(0),
    DPT.stride(0)
)
```



这里打包所有指针和行 stride。

随后：

```python
_launch_pre(
    lambda bm: (triton.cdiv(mc, bm),),
    pre_args,
    mc,
    N,
    d2,
    sm_scale,
    BLOCK_D
)
```



几个参数需要特别区分：

* `M = mc`：当前 chunk 的行数；
* `N_true = N`：未 padding 的真实 key 数；
* `D = d2`：padding 后的 head dimension；
* `stride_sm = N2`：循环遍历的是 padding 后的 key 数。

因此：

* 计算和存储覆盖 `N2`；
* softmax 统计只包含前 `N` 个真实 key。

---

# 十三、启动 `dQ` kernel

```python
dq_args = (
    Kp, S, DP, Mp, Lp, Deltap, dq_c,
    Kp.stride(0),
    S.stride(0),
    DP.stride(0),
    dq_c.stride(0)
)
```

随后：

```python
_launch_dq(
    lambda bm: (triton.cdiv(mc, bm),),
    dq_args,
    mc,
    d2,
    sm_scale,
    BLOCK_D
)
```



每个 program 负责当前 chunk 中的一个 query tile。

虽然循环会覆盖 padding 后的 `N2` 个 key，但 padding key 的：

* `S = -inf`
* `P = 0`
* `K = 0`

因此不会影响 `dQ`。

---

# 十四、启动 `dK/dV` kernel

```python
dkdv_args = (
    q_c, do_c,
    ST, DPT,
    Mp, Lp, Deltap,
    dKp, dVp,
    q_c.stride(0),
    do_c.stride(0),
    ST.stride(0),
    DPT.stride(0),
    dKp.stride(0),
    dVp.stride(0)
)
```

随后：

```python
_launch_dkdv(
    lambda bn: (triton.cdiv(N2, bn),),
    dkdv_args,
    mc,
    N2,
    d2,
    sm_scale,
    1 if (multi and m0 > 0) else 0,
    BLOCK_D
)
```



### Grid

[
\left\lceil\frac{N2}{BLOCK_N}\right\rceil
]

每个 program 负责一个 key block。

### `accumulate`

```python
1 if (multi and m0 > 0) else 0
```

含义：

* 单 chunk：不读旧结果；
* 多 chunk 的第一个 chunk：不读旧结果；
* 多 chunk 的第二个及以后 chunk：加载旧 `dK/dV` 并累加。

### 为什么传入 `N2` 而不是 `N`？

kernel 会计算 padding 后的全部 key 行。

真实输出阶段只复制：

```text
[:N, :d]
```

所以 padding 行最终被丢弃。

---

# 十五、复制回用户输出

```python
if padded or dQp is not dQ:
    dQ[:M, :d].copy_(dQp[:M, :d])
if padded or dKp is not dK:
    dK[:N, :d].copy_(dKp[:N, :d])
if padded or dVp is not dV:
    dV[:N, :d].copy_(dVp[:N, :d])
```



存在两种需要复制的情况：

1. 做过 padding；
2. 用户输出 tensor 不连续，kernel 写入了连续临时 tensor。

只复制真实区域：

```text
dQ: 前 M 行、前 d 列
dK: 前 N 行、前 d 列
dV: 前 N 行、前 d 列
```

padding 区域被舍弃。

---

# 十六、完整数据流

整个过程可以概括为：

```text
Q [M,d] ─────┐
              ├─ Q @ Kᵀ × scale ─→ S  [M,N]
K [N,d] ─Kᵀ──┘                    └→ ST [N,M]

dO [M,d] ────┐
              ├─ dO @ Vᵀ ───────→ DP  [M,N]
V [N,d] ─Vᵀ──┘                    └→ DPT [N,M]
```

同时 Kernel 1 计算：

```text
Mp[i]    = max_j S[i,j]
Lp[i]    = Σ_j exp(S[i,j] - Mp[i])
Delta[i] = Σ_j P[i,j] · DP[i,j]
```

Kernel 2：

```text
P  = exp(S - Mp) / Lp
DS = P · (DP - Delta) · scale
dQ = DS @ K
```

Kernel 3：

```text
Pᵀ  = exp(ST - Mp) / Lp
DSᵀ = Pᵀ · (DPT - Delta) · scale

dV = Pᵀ  @ dO
dK = DSᵀ @ Q
```

---

# 十七、这份实现的核心设计取舍

## 优点

### 1. 避免重复大矩阵乘

`S=QKᵀ` 和 `dP=dOVᵀ` 只计算一次。

后续 `dQ/dK/dV` 都直接读取 scratch。

### 2. `dK/dV` 使用转置 scratch

通过额外保存 `ST/DPT`，第三个 kernel 可以连续读取数据，避免低效的跨步访问。

### 3. 在线 softmax 数值稳定

通过每行最大值和指数和重建 softmax，避免直接计算：

[
e^{S}
]

造成上溢。

### 4. float32 累加

三个梯度均在 float32 中累加。

### 5. 支持任意非 16 倍数形状

Host 端统一 padding，并在最后切回真实范围。

### 6. 支持大型 `M×N`

当 scratch 太大时，沿 `M` 切 chunk，避免单次临时矩阵超过预算。

---

## 代价与局限

### 1. scratch 显存开销非常大

需要四个 float32 的 `M×N` 矩阵：

[
4\times M\times N\times4\text{ bytes}
=16MN\text{ bytes}
]

例如：

```text
M=8192
N=4096
```

单个矩阵大小：

[
8192\times4096\times4
=128\text{ MiB}
]

四个合计：

[
512\text{ MiB}
]

这还没有计算 `KT/VT`、输入和梯度。

### 2. 显存带宽压力大

`S/DP` 都需要：

* 在 Kernel 1 中写入；
* 在 Kernel 2 中读取；
* `ST/DPT` 又需要在 Kernel 1 写入、Kernel 3 读取。

因此它用更多显存流量换取更少计算。

FlashAttention 风格实现通常倾向于重算部分量，以减少 `O(MN)` 的全局存储。

### 3. 显式转置也有额外成本

```python
Kp.t().contiguous()
Vp.t().contiguous()
```

需要分配和复制。

### 4. 不支持通用二维 stride

只传行 stride，默认最后一维 stride 为 1，因此输入必须连续。

### 5. 没有 batch/head 维度

代码只处理单个二维 attention：

```text
Q: [M,d]
K: [N,d]
V: [N,d]
```

实际多头注意力通常是：

```text
[B,H,M,d]
```

若扩展，需要：

* 增加 batch/head program 维度；
* 增加相应 stride；
* 为每个 `(batch, head)` 独立执行 softmax 统计。

### 6. 回退配置全部失败时缺乏显式错误

三个 `_launch_*` 函数捕获 `OutOfResources` 后仅继续尝试；如果所有候选配置都失败，函数没有主动抛出清晰异常。

---

# 十八、最容易混淆的五个点

1. **Kernel 1 中的临时 `p` 不是最终 softmax。**
   它只是 `exp(s-m_new)`，尚未除以总和。

2. **`delta` 在循环中也是未归一化的。**
   循环结束后执行 `delta / l_safe` 才得到
   (\sum_jP_{ij}dP_{ij})。

3. **代码中的 `ds` 已经乘了 `sm_scale`。**
   因此后续计算 `dQ/dK` 时不需要再乘一次。

4. **`stride_sm` 在循环中被当作 `N2` 使用。**
   这是因为 `S` 连续，`S.stride(0)==N2`，不是一般意义上所有 stride 都可当作维度长度。

5. **同时保存 `S` 和 `ST` 不是数学需要，而是内存访问优化。**
   它牺牲显存容量和写带宽，换取后续 kernel 的连续读取。
