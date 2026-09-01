# 逐行解析：Triton 多头交叉注意力

## 整体思路

这是一个 **Flash-Attention 风格的单 kernel 实现**：不物化完整的 `M×N` 注意力矩阵，而是把 N 维切成块，一边扫描一边用"在线 softmax"累积结果。每个 GPU program 负责 **一个 head 下的 BLOCK_M 个 query**，循环扫过全部 N 个 encoder 位置。

数学上，对每行 query 我们要算：

$$\text{out} = \frac{\sum_j e^{s_j - \max_k s_k}\, v_j}{\sum_j e^{s_j - \max_k s_k}}, \quad s_j = \frac{q \cdot k_j}{\sqrt{D}}$$

在线 softmax 的技巧是：每处理一块就更新 running max `m_i`，并用因子 `e^{m_{旧} - m_{新}}` 把之前累积的值"修正"到新基准上，这样分块处理和一次性算完全等价。

---

## 第一部分：kernel 签名（第 6–17 行）

```python
@triton.jit                                                    # 6
def _cross_attention_kernel(                                   # 7
    Q, K, V, O,                                                # 8
    stride_qm, stride_qh,                                      # 9
    stride_kn, stride_kh,                                      # 10
    stride_vn, stride_vh,                                      # 11
    stride_om, stride_oh,                                      # 12
    M, N, D,                                                   # 13
    sm_scale,                                                  # 14
    BLOCK_M: tl.constexpr,                                     # 15
    BLOCK_N: tl.constexpr,                                     # 16
    BLOCK_D: tl.constexpr,                                     # 17
):
```

- **第 6 行** `@triton.jit`：告诉 Triton 这是要编译成 GPU kernel 的函数，不是普通 Python。调用时按给定的 grid 启动成千上万个 program 实例并行执行。
- **第 8 行** `Q, K, V, O`：四个张量的**基地址指针**。Triton kernel 收到的不是张量对象，而是指向显存的标量指针，后面的寻址都是"指针 + 整数偏移"。
- **第 9–12 行** strides：张量布局是 `(position, head, dim)`，所以在 head `h`、位置 `p`、维度 `d` 处的元素地址 = `base + p*stride_m + h*stride_h + d`（dim 维连续，stride 为 1，故省略）。只传行/头两个 stride 即可。
- **第 13 行** `M, N, D`：真实尺寸，用于边界 mask（M/N/D 都可以不是 block 的整数倍）。
- **第 14 行** `sm_scale`：softmax 缩放因子 `1/√D`，在 host 端算好传入。
- **第 15–17 行** `tl.constexpr`：**编译期常量**。BLOCK 尺寸决定 tile 形状和 shared memory 分配，编译时必须已知。每组不同取值会触发一次独立编译（Triton 自动缓存）。

## 第二部分：确定本 program 的职责范围（第 21–26 行）

```python
pid_m = tl.program_id(0)      # 21
h = tl.program_id(1)          # 22
```

- grid 是二维的 `(⌈M/BLOCK_M⌉, H)`。`program_id(0)` 是本 program 负责的 **query 块编号**，`program_id(1)` 是 **head 编号**。

```python
offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # 24
offs_d = tl.arange(0, BLOCK_D)                     # 25
d_mask = offs_d < D                                # 26
```

- **第 24 行**：本块负责的 query 行号向量，形如 `[pid_m*64, pid_m*64+1, ..., pid_m*64+63]`。`tl.arange` 类似 `np.arange`，生成 tile 内索引。
- **第 25 行**：维度索引 `[0, 1, ..., BLOCK_D-1]`。
- **第 26 行**：`D` 可能不是 2 的幂（如 D=33 → BLOCK_D=64），`d_mask` 标记哪些列是有效的，越界列在加载时填 0（填 0 不影响点积结果）。

## 第三部分：加载 Q 块（第 29–31 行）

```python
q_ptrs = Q + offs_m[:, None] * stride_qm + h * stride_qh + offs_d[None, :]   # 29
q_mask = (offs_m[:, None] < M) & d_mask[None, :]                             # 30
q = tl.load(q_ptrs, mask=q_mask, other=0.0)                                  # 31
```

- **第 29 行**是 Triton 的核心惯用法：`offs_m[:, None]` 是 `(BLOCK_M, 1)` 列向量，`offs_d[None, :]` 是 `(1, BLOCK_D)` 行向量，广播相加后 `q_ptrs` 是 `(BLOCK_M, BLOCK_D)` 的**指针矩阵**——每个元素是该 tile 单元对应的显存地址。
- **第 30 行**：两个条件——行号不能超出 M（M 不是 64 的倍数时尾块有哑行）；列不能超出 D。
- **第 31 行**：按指针矩阵批量加载成寄存器 tile。`mask` 指定无效位置，`other=0.0` 填零。Q 在整个 kernel 中只加载这一次（对 K/V 循环不变）。

## 第四部分：在线 softmax 状态（第 34–36 行）

```python
m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)   # 34: running max
l_i = tl.zeros([BLOCK_M], dtype=tl.float32)                 # 35: running Σexp
acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)        # 36: running Σp·v
```

- `m_i[j]`：到目前**为止所有已扫描 key 中，第 j 个 query 见过的最大 score**（未缩放前的稳定性基准）。
- `l_i[j]`：以当前 `m_i` 为基准的指数和 `Σ e^{s - m_i}`。
- `acc[j, :]`：以当前 `m_i` 为基准的加权和 `Σ e^{s - m_i} · v`。
- 三者都以 `-inf` / 0 初始化；最后再统一除以 `l_i` 得到真正的 softmax 输出。

## 第五部分：主循环——扫过所有 encoder 位置（第 38–62 行）

```python
for start_n in range(0, N, BLOCK_N):     # 38
    offs_n = start_n + tl.arange(0, BLOCK_N)   # 39
    n_mask = offs_n < N                        # 40
```

每次取 BLOCK_N 个 key/value。`n_mask` 处理 N 不是 BLOCK_N 整数倍的尾块。

```python
kt_ptrs = K + offs_n[None, :] * stride_kn + h * stride_kh + offs_d[:, None]  # 45
kt = tl.load(kt_ptrs, mask=n_mask[None, :] & d_mask[:, None], other=0.0)     # 46
```

- **K 直接按转置布局加载成 `(BLOCK_D, BLOCK_N)`**：行索引是 d、列索引是 n。这样下一步 `tl.dot(q, kt)` 就是合法的 `(M,D)×(D,N)` 矩阵乘，**不需要 `tl.trans()`**。
- 这正是上次报错的修复点：`tl.trans` 在 Triton 内部要做一次 shared memory 布局转换，额外占 32KB，导致 T4 的 64KB 超限。这里只是指针运算的写法不同，数学上完全等价（显存读取的 coalescing 稍差，但正确性和容量优先）。

```python
s = tl.dot(q, kt) * sm_scale                      # 49
s = tl.where(n_mask[None, :], s, float("-inf"))   # 50
```

- **第 49 行**：`(BLOCK_M, BLOCK_D) @ (BLOCK_D, BLOCK_N)` → score 矩阵 `(BLOCK_M, BLOCK_N)`，即这批 query 对这 32 个 key 的原始注意力分数，乘上 `1/√D`。`tl.dot` 在 fp32 下于 T4 走 FMA 路径。
- **第 50 行**：把越界 key 列的 score 置为 `-inf`。这样下一步 `exp(-inf - m) = 0`，尾块的哑 key 对 softmax 贡献为零——这就是"mask 掉不存在的 key"的机制。

```python
m_new = tl.maximum(m_i, tl.max(s, 1))    # 53
alpha = tl.exp(m_i - m_new)              # 54
p = tl.exp(s - m_new[:, None])           # 55
l_i = l_i * alpha + tl.sum(p, 1)         # 56
acc = acc * alpha[:, None]               # 57
```

在线 softmax 的五步更新（对每一行 query 独立进行）：

- **53**：新基准 = 历史最大与当前块逐行最大（`tl.max(s, 1)` 沿 key 维归约）的较大者。
- **54**：**修正因子** `α = e^{m_旧 - m_新}`。之前的 `l_i`、`acc` 都是以旧基准算的，换基准后要整体乘 α。首轮 `m_i=-inf` 时 `α=0`，正确清零初始值。
- **55**：当前块的未归一化注意力权重 `p = e^{s - m_新}`（数值稳定：指数参数 ≤ 0，不会溢出）。
- **56**：分母累积：`l_i ← l_i·α + Σ_j p_j`。
- **57**：分子累积的第一步：把旧的 `acc` 修正到新基准。

```python
v_ptrs = V + offs_n[:, None] * stride_vn + h * stride_vh + offs_d[None, :]   # 59
v = tl.load(v_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)       # 60
acc += tl.dot(p, v)          # 61
m_i = m_new                  # 62
```

- **59–60**：按正常 `(BLOCK_N, BLOCK_D)` 布局加载 V tile。
- **61**：`(BLOCK_M, BLOCK_N) @ (BLOCK_N, BLOCK_D)`，把当前块的 `p·v` 加进 `acc`。此时 `acc = Σ_{已扫描} e^{s - m_新} v`。
- **62**：更新 running max，进入下一块。

循环结束后：`acc[j] = Σ_all e^{s_j - m*} v_j`，`l_i[j] = Σ_all e^{s_j - m*}`，其中 `m*` 是全 N 上的真实最大值——与一次性计算严格等价。

## 第六部分：归一化与写回（第 64–67 行）

```python
acc = acc / l_i[:, None]     # 64
o_ptrs = O + offs_m[:, None] * stride_om + h * stride_oh + offs_d[None, :]   # 66
tl.store(o_ptrs, acc, mask=q_mask)   # 67
```

- **64**：除以分母，得到真正的 softmax 加权平均。
- **66–67**：按与加载 Q 相同的寻址方式写回 `(M, H, D)` 的 output。`mask=q_mask` 保证尾块的哑行不写显存（哑行因 `l_i=0` 会算出 NaN，但被 mask 挡住，无害）。

## 第七部分：host 端 `solve`（第 71–108 行）

```python
BLOCK_D = max(16, triton.next_power_of_2(D))   # 81
```

- `tl.dot` 要求 tile 各维 ≥ 16 且为 2 的幂，所以 D=33 → BLOCK_D=64，D=1 → BLOCK_D=16（多余部分 mask 填 0）。

```python
if BLOCK_D <= 64:    BLOCK_M, BLOCK_N = 64, 64     # 87-88
elif BLOCK_D == 128: BLOCK_M, BLOCK_N = 64, 32     # 89-90
else:                BLOCK_M, BLOCK_N = 32, 16     # 91-92
```

- **T4 共享内存约束的产物**：Triton 在 `tl.dot` 前要把操作数 tile 经 shared memory 做布局转换，fp32 每元素 4B。D 越大 tile 越大，只能缩小 M/N 方向的块。这三档都经过 sm_75 离线编译实测（49152 / 57344 / 51200 B），均低于 65536 B 上限。性能评测点 D=128 落在第二档。

```python
grid = (triton.cdiv(M, BLOCK_M), H)   # 94
```

- `cdiv` 向上取整。grid 第一维覆盖所有 query，第二维每个 head 一份。评测尺寸下为 `(16, 16)` = 256 个 program，T4 有 40 个 SM，并行度充足。

```python
_cross_attention_kernel[grid](
    Q, K, V, output,
    Q.stride(0), Q.stride(1), ...      # 97-100: 只传 position 和 head 两维 stride
    M, N, D,
    1.0 / (D ** 0.5),                  # 102: 缩放因子 1/√D
    BLOCK_M=..., BLOCK_N=..., BLOCK_D=...,   # constexpr 参数
    num_warps=4,                       # 106: 每个 program 4 个 warp (128 线程)
    num_stages=1,                      # 107: 不做软件流水双缓冲，省 shared memory
)
```

- **97–100**：从张量对象读出真实 stride 传入，不假设布局之外的任何东西。
- **`num_stages=1`**：Triton 默认会对循环里的加载做多级流水（预取下一块），每多一级就多一份 tile 的 shared memory。T4 的 64KB 容不下双缓冲，设为 1 牺牲一点延迟隐藏换取能跑。

**一句话总结**：kernel 用二维 grid 把"query 块 × head"分给各 program，每个 program 内部用 flash 在线 softmax 单趟扫完 N 个 key/value，全程 fp32，通过转置加载 K 和分档 block 尺寸把 shared memory 压进 T4 的 64KB 限制内。