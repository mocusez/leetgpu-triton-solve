
## 1. 这段代码整体在做什么

这段 Triton 内核实现的是一个简化版的 **滑动窗口自注意力前向计算**：

[
S_{m,n}=\frac{Q_mK_n^T}{\sqrt d}
]

只保留满足下面条件的注意力位置：

[
|m-n|\leq \text{window_size}
]

然后在允许的 (n) 维度上做 softmax：

[
P_{m,n}
=======

\frac{\exp(S_{m,n})}
{\sum_{j:,|m-j|\leq w}\exp(S_{m,j})}
]

最后计算：

[
O_m=\sum_nP_{m,n}V_n
]

因此它在语义上等价于：

```python
scores = Q @ K.T / sqrt(d)
scores[abs(row_index - col_index) > window_size] = -inf
probs = softmax(scores, dim=-1)
output = probs @ V
```

但 Triton 版本不会把完整的 (M\times N) 注意力矩阵写入显存，而是分块读取 K、V，并利用在线 softmax 累积结果。这与 FlashAttention 中使用的分块、在线归一化思路相似。Triton 官方 fused-attention 教程也采用了逐块更新最大值、归一化和输出累加器的结构。([Triton Language][1])

这里没有 batch 维度，也没有多头维度。根据 `solve()`：

```python
attention[grid](Q, K, V, output, M, M, ...)
```

内核里的 `N` 被设置成 `M`，所以实际假设：

```text
Q:      [M, d]
K:      [M, d]
V:      [M, d]
output: [M, d]
```

并且这些张量必须是按行连续存储的。

---

# 2. 导入部分

```python
import torch
import triton
import triton.language as tl
```

### `torch`

用于管理 GPU Tensor、形状、数据类型和显存。

### `triton`

提供：

* `@triton.jit`
* `triton.cdiv`
* `triton.next_power_of_2`
* 内核启动接口 `kernel[grid](...)`

### `triton.language as tl`

这是 Triton 内核内部使用的 DSL，包括：

* `tl.program_id`
* `tl.arange`
* `tl.load`
* `tl.store`
* `tl.dot`
* `tl.exp`
* `tl.sum`
* `tl.max`

---

# 3. 内核声明

```python
@triton.jit
def attention(
    Q, K, V,
    output,
    M, N, d, window_size,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr
):
```

## `@triton.jit`

表示这个 Python 函数不是普通 Python 函数，而是要由 Triton 编译成 GPU 内核。

## 普通参数

```python
Q, K, V, output
```

是 GPU 全局内存指针。

```python
M, N, d, window_size
```

是运行时标量：

* `M`：query 序列长度
* `N`：key/value 序列长度
* `d`：每个 token 的特征维度
* `window_size`：滑动窗口半径

在当前 `solve()` 中：

```python
N = M
```

因此是自注意力。

## `tl.constexpr` 参数

```python
BLOCK_M: tl.constexpr
BLOCK_N: tl.constexpr
BLOCK_D: tl.constexpr
```

这些值在编译时已知，Triton 会针对具体值生成专门的内核。

含义如下：

| 参数        | 含义                            |
| --------- | ----------------------------- |
| `BLOCK_M` | 一个 Triton program 处理多少个 query |
| `BLOCK_N` | 每次循环读取多少个 key/value           |
| `BLOCK_D` | 内部使用的特征维度，通常为不小于 `d` 的 2 的幂   |

---

# 4. 确定当前 program 处理哪些 query

```python
pid = tl.program_id(0)
```

Triton 会启动多个 program instance。`tl.program_id(0)` 返回当前 program 在启动网格第 0 维上的编号。([Triton Language][2])

假设：

```text
BLOCK_M = 16
```

那么：

```text
pid = 0 → query 0～15
pid = 1 → query 16～31
pid = 2 → query 32～47
```

---

```python
offset_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
```

`tl.arange(0, BLOCK_M)` 生成：

```text
[0, 1, 2, ..., BLOCK_M-1]
```

所以 `offset_m` 表示当前 program 处理的全局 query 行号。

例如：

```text
pid = 2
BLOCK_M = 16
```

则：

```text
offset_m =
[32, 33, ..., 47]
```

`tl.arange` 的长度通常需要使用适合编译器块表示的 2 的幂，因此代码会把特征维度补齐到 2 的幂。([Triton Language][3])

---

```python
offset_d = tl.arange(0, BLOCK_D)
```

生成特征维度索引：

```text
[0, 1, 2, ..., BLOCK_D-1]
```

假设真实：

```text
d = 50
```

而：

```text
BLOCK_D = 64
```

则 `offset_d` 是 `0～63`，其中 `50～63` 属于补齐区域。

---

# 5. 创建边界 mask

```python
mask_m = offset_m < M
```

形状：

```text
[BLOCK_M]
```

用于处理最后一个 query block 越界的问题。

例如：

```text
M = 35
BLOCK_M = 16
pid = 2
```

此时：

```text
offset_m = [32, 33, 34, 35, ..., 47]
mask_m   = [1,  1,  1,  0,  ..., 0]
```

---

```python
mask_d = offset_d < d
```

形状：

```text
[BLOCK_D]
```

用于屏蔽补齐的特征维度。

例如：

```text
d = 50
BLOCK_D = 64
```

则：

```text
mask_d[0:50]  = True
mask_d[50:64] = False
```

---

# 6. 加载 Q block

```python
vals_q = tl.load(
    Q + offset_m[:, None] * d + offset_d[None, :],
    mask_m[:, None] & mask_d[None, :],
    other=0.0
)
```

这是非常关键的一行。

## 6.1 地址计算

假设 Q 是连续的二维矩阵：

```text
Q.shape = [M, d]
```

那么：

```text
Q[m, k] 的线性地址 = Q + m*d + k
```

代码：

```python
offset_m[:, None] * d
```

形状是：

```text
[BLOCK_M, 1]
```

表示每一行的起始地址。

```python
offset_d[None, :]
```

形状是：

```text
[1, BLOCK_D]
```

表示列偏移。

二者广播相加后：

```text
Q + offset_m[:, None] * d + offset_d[None, :]
```

得到一个：

```text
[BLOCK_M, BLOCK_D]
```

的指针矩阵。

---

## 6.2 加载 mask

```python
mask_m[:, None] & mask_d[None, :]
```

广播后形状为：

```text
[BLOCK_M, BLOCK_D]
```

只有同时满足下面条件才会真正读取：

```text
query 行号 < M
特征列号 < d
```

越界位置返回：

```python
other=0.0
```

Triton 的 `tl.load` 在 mask 为 false 时不会读取该地址，而是返回 `other`。([Triton Language][4])

最终：

```text
vals_q.shape = [BLOCK_M, BLOCK_D]
```

真实的前 `d` 列是 Q，补齐列为 0。

---

# 7. 注意力缩放因子

```python
scale = tl.sqrt(d.to(tl.float32))
```

将 `d` 转为 float32，然后计算：

[
\sqrt d
]

后面执行：

```python
QK^T / scale
```

也就是标准 scaled dot-product attention：

[
\frac{QK^T}{\sqrt d}
]

缩放可以避免特征维度增大时，点积幅度过大，导致 softmax 过于尖锐。

---

# 8. 初始化在线 softmax 状态

```python
out_vals = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
```

形状：

```text
[BLOCK_M, BLOCK_D]
```

它保存尚未归一化的：

[
\sum_n e^{s_{m,n}-m_m}V_n
]

即输出分子累加器。

用 float32 累积可以提高精度。

---

```python
ma = tl.full(
    (BLOCK_M,),
    float("-inf"),
    dtype=tl.float32
)
```

形状：

```text
[BLOCK_M]
```

`ma` 表示每个 query 到目前为止见过的最大 attention score：

[
m_i=\max_{j\text{ 已处理}}S_{i,j}
]

初始还没有处理任何 key，所以设为负无穷。

变量名 `ma` 可以理解为 `max accumulator`。

---

```python
sum = tl.full(
    (BLOCK_M,),
    0.0,
    dtype=tl.float32
)
```

形状：

```text
[BLOCK_M]
```

保存当前归一化基准下的 softmax 分母：

[
l_i=\sum_{j\text{ 已处理}}e^{S_{i,j}-m_i}
]

初始为 0。

这里把变量命名为 `sum` 会遮蔽 Python 内置的 `sum`，但在这个内核中没有直接问题。更清晰的名字通常是：

```python
denom
```

或：

```python
l_i
```

---

# 9. 遍历所有 K/V block

```python
for step in range(0, tl.cdiv(N, BLOCK_N)):
```

`tl.cdiv(N, BLOCK_N)` 是向上取整除法：

[
\left\lceil\frac{N}{BLOCK_N}\right\rceil
]

假设：

```text
N = 35
BLOCK_N = 16
```

那么：

```text
tl.cdiv(35, 16) = 3
```

循环处理：

```text
step = 0 → key 0～15
step = 1 → key 16～31
step = 2 → key 32～47，其中 35～47 越界
```

官方 Triton 矩阵乘法教程也使用了 `range(0, tl.cdiv(...))` 来遍历运行时长度的分块。([Triton Language][5])

---

# 10. 当前 key block 的位置

```python
offset_n = step * BLOCK_N + tl.arange(0, BLOCK_N)
```

形状：

```text
[BLOCK_N]
```

表示当前 K/V block 的全局 token 下标。

例如：

```text
step = 2
BLOCK_N = 16
```

则：

```text
offset_n = [32, 33, ..., 47]
```

---

```python
mask_n = offset_n < N
```

形状：

```text
[BLOCK_N]
```

用于屏蔽最后一个 K/V block 的越界 token。

---

# 11. 加载 K block

```python
vals_k = tl.load(
    K + offset_n[:, None] * d + offset_d[None, :],
    mask_n[:, None] & mask_d[None, :],
    other=0.0
)
```

与加载 Q 基本相同。

最终：

```text
vals_k.shape = [BLOCK_N, BLOCK_D]
```

可以理解为：

```python
vals_k = K[offset_n, :]
```

越界行和补齐列填 0。

---

# 12. 计算 QKᵀ

```python
vals_qk = tl.dot(
    vals_q,
    tl.permute(vals_k, (1, 0)),
    allow_tf32=False
) / scale
```

## 12.1 转置 K

原始：

```text
vals_k.shape = [BLOCK_N, BLOCK_D]
```

经过：

```python
tl.permute(vals_k, (1, 0))
```

变成：

```text
[BLOCK_D, BLOCK_N]
```

`tl.permute` 用于重新排列张量维度；对二维张量使用 `(1, 0)` 相当于转置。([Triton Language][6])

---

## 12.2 矩阵乘法

```text
vals_q:          [BLOCK_M, BLOCK_D]
transpose(vals_k): [BLOCK_D, BLOCK_N]
```

因此：

```text
vals_qk: [BLOCK_M, BLOCK_N]
```

每个元素是：

[
\text{vals_qk}[i,j]
===================

\frac{
Q_{\text{offset_m}[i]}
\cdot
K_{\text{offset_n}[j]}
}{
\sqrt d
}
]

`tl.dot` 对二维 block 执行矩阵乘法，并要求内维度兼容。([Triton Language][7])

---

## 12.3 `allow_tf32=False`

这表示当输入是 float32 时，不希望使用较低尾数精度的 TF32 点积。

不过在当前 Triton API 中，`allow_tf32` 已经被标为弃用，较新的写法是使用：

```python
input_precision="ieee"
```

如果 Q、K 是 float16 或 bfloat16，该参数通常不决定它们的输入精度路径。([Triton Language][8])

---

# 13. 创建滑动窗口 mask

```python
mask = (
    tl.abs(offset_m[:, None] - offset_n[None, :])
    <= window_size
)
```

形状：

```text
[BLOCK_M, BLOCK_N]
```

其中：

```python
offset_m[:, None]
```

形状：

```text
[BLOCK_M, 1]
```

而：

```python
offset_n[None, :]
```

形状：

```text
[1, BLOCK_N]
```

广播相减后得到每一对 query/key 的距离：

[
|m-n|
]

当：

[
|m-n|\leq \text{window_size}
]

时允许注意力。

例如 `window_size=2`，query 位置 `m=5` 可以关注：

```text
n = 3, 4, 5, 6, 7
```

不能关注其他位置。

这是双向窗口，不是因果窗口。它既允许看左边，也允许看右边。

如果想改为因果滑动窗口，条件应类似：

```python
(offset_n[None, :] <= offset_m[:, None]) & (
    offset_n[None, :] >= offset_m[:, None] - window_size
)
```

---

# 14. 计算当前 block 的最大值

```python
vals_qk_ma = tl.where(mask, vals_qk, float(-100))
```

窗口内：

```text
使用真实 score
```

窗口外：

```text
替换为 -100
```

这样后面的最大值主要来自窗口内位置。

随后：

```python
ma_now = tl.maximum(
    tl.max(vals_qk_ma, axis=1),
    ma
)
```

## `tl.max(..., axis=1)`

对每个 query 行，在当前 key block 中求最大 score：

```text
[BLOCK_M, BLOCK_N]
        ↓ axis=1
[BLOCK_M]
```

记当前 block 最大值为：

[
b_i=\max_{j\in\text{当前块}}S_{i,j}
]

然后与历史最大值 `ma` 合并：

[
m_i^{new}=\max(m_i^{old},b_i)
]

这就是在线 softmax 的核心。

---

# 15. 计算当前 block 的指数权重

```python
vals_exp = tl.where(
    mask_m[:, None] & mask_n[None, :],
    tl.exp(vals_qk - ma_now[:, None]),
    0.0
)
```

先计算：

[
e^{S_{i,j}-m_i^{new}}
]

减去最大值可以避免指数溢出。softmax 对所有 score 同时减去同一个常数结果不变，这是标准的数值稳定 softmax 技巧。官方 Triton softmax 教程同样先减去每行最大值再计算指数。([Triton Language][9])

然后通过：

```python
mask_m[:, None] & mask_n[None, :]
```

屏蔽越界 query 和越界 key。

---

```python
vals_exp = tl.where(mask, vals_exp, 0.0)
```

再次应用窗口 mask。

所以最终：

[
\text{vals_exp}[i,j]
====================

\begin{cases}
e^{S_{i,j}-m_i^{new}},
& \text{位置有效且在窗口内}\
0,
& \text{其他情况}
\end{cases}
]

这两次 `tl.where` 可以合成一个完整 mask：

```python
valid = (
    mask_m[:, None]
    & mask_n[None, :]
    & mask
)
```

---

# 16. 当前 block 的分母贡献

```python
sum_now = tl.sum(vals_exp, axis=1)
```

对当前 key block 求和：

[
l_i^{block}
===========

\sum_{j\in\text{当前块}}
e^{S_{i,j}-m_i^{new}}
]

形状从：

```text
[BLOCK_M, BLOCK_N]
```

变成：

```text
[BLOCK_M]
```

---

# 17. 加载 V block

```python
vals_v = tl.load(
    V + offset_n[:, None] * d + offset_d[None, :],
    mask_n[:, None] & mask_d[None, :],
    other=0.0
)
```

结果：

```text
vals_v.shape = [BLOCK_N, BLOCK_D]
```

相当于：

```python
vals_v = V[offset_n, :]
```

---

# 18. 更新输出累加器

```python
out_vals = (
    out_vals * tl.exp(ma - ma_now)[:, None]
    + tl.dot(vals_exp, vals_v, allow_tf32=False)
)
```

这是整个内核中最重要的一行。

## 18.1 为什么旧结果需要重新缩放

旧累加器使用旧最大值 `ma`：

[
O_i^{old}
=========

\sum_{\text{旧块}}
e^{S_{i,j}-m_i^{old}}V_j
]

现在最大值变成：

[
m_i^{new}
]

为了让旧结果与当前 block 使用同一个指数基准，需要乘：

[
e^{m_i^{old}-m_i^{new}}
]

因为：

[
e^{S-m^{old}}
e^{m^{old}-m^{new}}
===================

e^{S-m^{new}}
]

所以：

```python
out_vals * tl.exp(ma - ma_now)[:, None]
```

就是把旧累加器转换到新的最大值基准下。

---

## 18.2 当前 block 的输出贡献

```python
tl.dot(vals_exp, vals_v)
```

形状：

```text
vals_exp: [BLOCK_M, BLOCK_N]
vals_v:   [BLOCK_N, BLOCK_D]
结果:      [BLOCK_M, BLOCK_D]
```

数学上是：

[
\sum_{j\in\text{当前块}}
e^{S_{i,j}-m_i^{new}}V_j
]

---

## 18.3 合并结果

因此整个更新式是：

[
O_i^{new}
=========

O_i^{old}e^{m_i^{old}-m_i^{new}}
+
\sum_{j\in\text{当前块}}
e^{S_{i,j}-m_i^{new}}V_j
]

这使得内核可以逐块处理 K/V，而无需保存完整注意力矩阵。

---

# 19. 更新 softmax 分母

```python
sum = (
    sum * tl.exp(ma - ma_now)
    + sum_now
)
```

与输出累加器使用完全相同的缩放逻辑：

[
l_i^{new}
=========

l_i^{old}e^{m_i^{old}-m_i^{new}}
+
\sum_{j\in\text{当前块}}
e^{S_{i,j}-m_i^{new}}
]

这样，处理完全部 key block 后：

```text
out_vals = softmax 分子
sum      = softmax 分母
```

---

```python
ma = ma_now
```

保存新的历史最大值，供下一轮使用。

---

# 20. 最终归一化并写回

```python
tl.store(
    output + offset_m[:, None] * d + offset_d[None, :],
    out_vals / sum[:, None],
    mask=mask_m[:, None] & mask_d[None, :]
)
```

循环结束后：

[
\frac{\text{out_vals}_i}{\text{sum}_i}
======================================

\frac{
\sum_j e^{S_{i,j}-m_i}V_j
}{
\sum_j e^{S_{i,j}-m_i}
}
]

公共因子 (e^{-m_i}) 会在分子和分母中抵消，因此等价于：

[
\sum_j \operatorname{softmax}(S_i)_jV_j
]

`sum[:, None]` 把：

```text
[BLOCK_M]
```

扩展为：

```text
[BLOCK_M, 1]
```

以便除到每一个特征维度。

写回 mask 保证：

* 不写越界 query
* 不写补齐的特征维度

---

# 21. `solve()` 启动函数

```python
def solve(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    output: torch.Tensor,
    M: int,
    d: int,
    window_size: int,
):
```

这是 CPU 端的 Python 包装函数，负责选择 block 大小并启动 Triton 内核。

---

```python
BLOCK_M = 16
```

每个 Triton program 处理 16 个 query token。

---

```python
BLOCK_N = 16
```

每次循环处理 16 个 key/value token。

---

```python
BLOCK_D = max(16, triton.next_power_of_2(d))
```

选择不小于 `d` 的 2 的幂，并且至少为 16。

例如：

| `d` | `next_power_of_2(d)` | `BLOCK_D` |
| --: | -------------------: | --------: |
|   8 |                    8 |        16 |
|  16 |                   16 |        16 |
|  48 |                   64 |        64 |
|  64 |                   64 |        64 |
|  96 |                  128 |       128 |

这样做是为了生成规则的编译期 block。

如果：

```text
d = 48
BLOCK_D = 64
```

多出来的 16 列通过：

```python
mask_d = offset_d < d
```

填 0 并禁止写回。

---

```python
grid = (triton.cdiv(M, BLOCK_M), )
```

创建一维启动网格。

假设：

```text
M = 35
BLOCK_M = 16
```

则：

```text
grid = (3,)
```

启动 3 个 program：

```text
program 0 → query 0～15
program 1 → query 16～31
program 2 → query 32～47，35～47 被屏蔽
```

---

```python
attention[grid](
    Q, K, V, output,
    M, M, d, window_size,
    BLOCK_M, BLOCK_N, BLOCK_D
)
```

启动内核。

这里传递：

```text
M = query 数量
N = M = key/value 数量
```

因此是长度为 M 的自注意力。

更常见、更清晰的写法是把编译期参数写成关键字：

```python
attention[grid](
    Q, K, V, output,
    M, M, d, window_size,
    BLOCK_M=BLOCK_M,
    BLOCK_N=BLOCK_N,
    BLOCK_D=BLOCK_D,
)
```

---

# 22. 各变量形状汇总

| 变量         | 形状                   | 含义              |
| ---------- | -------------------- | --------------- |
| `offset_m` | `[BLOCK_M]`          | 当前 query 行号     |
| `offset_n` | `[BLOCK_N]`          | 当前 key/value 行号 |
| `offset_d` | `[BLOCK_D]`          | 特征维度编号          |
| `vals_q`   | `[BLOCK_M, BLOCK_D]` | query block     |
| `vals_k`   | `[BLOCK_N, BLOCK_D]` | key block       |
| `vals_v`   | `[BLOCK_N, BLOCK_D]` | value block     |
| `vals_qk`  | `[BLOCK_M, BLOCK_N]` | 当前 score block  |
| `mask`     | `[BLOCK_M, BLOCK_N]` | 滑动窗口 mask       |
| `vals_exp` | `[BLOCK_M, BLOCK_N]` | 未归一化 softmax 权重 |
| `ma`       | `[BLOCK_M]`          | 每行历史最大值         |
| `sum`      | `[BLOCK_M]`          | 每行 softmax 分母   |
| `out_vals` | `[BLOCK_M, BLOCK_D]` | 输出分子累加器         |

---

# 23. 这段代码的重要限制和潜在问题

## 23.1 它没有真正降低滑窗注意力的计算复杂度

虽然代码使用：

```python
mask = abs(m - n) <= window_size
```

但 mask 是在：

```python
vals_qk = tl.dot(...)
```

之后才应用的。

也就是说，每个 query block 仍会遍历所有 K block，并计算全部 QK 点积。

因此计算复杂度仍然接近：

[
O(MNd)
]

当 `N=M` 时是：

[
O(M^2d)
]

而不是真正局部注意力希望达到的：

[
O(Mwd)
]

它节省的是完整注意力矩阵的显存，而不是 QK 乘法量。

真正的滑窗优化应该让每个 query block 只遍历附近的 K/V block，例如大致遍历：

```text
pid*BLOCK_M - window_size
到
(pid+1)*BLOCK_M + window_size
```

范围内的 key block。

---

## 23.2 最大值 mask 不完整

当前代码：

```python
vals_qk_ma = tl.where(mask, vals_qk, float(-100))
```

这里只使用了窗口 mask，没有加入：

```python
mask_m
mask_n
```

最后一个 K block 的越界 key 是用全 0 的 K 加载的，所以对应 score 可能是 0。

如果这些越界位置恰好满足窗口距离条件，它们可能参与：

```python
tl.max(vals_qk_ma, axis=1)
```

从而让 `ma_now` 被不存在的 key 影响。

更合理的有效 mask 是：

```python
valid_mask = (
    mask_m[:, None]
    & mask_n[None, :]
    & (
        tl.abs(offset_m[:, None] - offset_n[None, :])
        <= window_size
    )
)
```

然后最大值和指数都基于这个 mask。

---

## 23.3 `-100` 不是严格的负无穷

```python
float(-100)
```

是假定真实 attention score 不会远小于 -100。

如果真实 score 小于 -100，那么窗口外位置可能影响最大值基准，或者造成额外的指数下溢风险。

通常会使用：

```python
-float("inf")
```

或足够小的值，例如：

```python
-1.0e6
```

官方 attention 教程在 mask score 时使用了很大的负数，并通过在线最大值维护数值稳定性。([Triton Language][1])

不过直接换成 `-inf` 时，还需要谨慎处理“当前整块没有任何有效 key”的情况，避免出现：

```text
-inf - (-inf) = NaN
```

---

## 23.4 可能出现除以零

最终：

```python
out_vals / sum[:, None]
```

如果某个 query 没有任何有效 key，则：

```text
sum = 0
```

最终产生 NaN。

在当前 `solve()` 中，只要：

```text
N = M
window_size >= 0
```

每个 query 至少能关注自己，所以通常不会出现这个问题。

但建议显式检查：

```python
assert window_size >= 0
assert M > 0
```

---

## 23.5 默认假设张量连续

地址计算写死为：

```python
row * d + column
```

这等价于假设：

```python
Q.stride() == (d, 1)
K.stride() == (d, 1)
V.stride() == (d, 1)
output.stride() == (d, 1)
```

如果输入经过转置、切片，变成非连续张量，结果会错误。

包装函数应加入：

```python
assert Q.is_contiguous()
assert K.is_contiguous()
assert V.is_contiguous()
assert output.is_contiguous()
```

或者把 stride 显式传入内核。

---

## 23.6 `vals_exp` 与 `vals_v` 的数据类型值得检查

`vals_exp` 通常经过 `tl.exp` 后是 float32，而 `vals_v` 可能是 float16 或 bfloat16。

官方 fused-attention 实现会在执行概率与 V 的点积前，把概率转换到适合矩阵乘法的数据类型：

```python
p = p.to(dtype)
acc = tl.dot(p, v, acc)
```

同时保持累加器为 float32。([Triton Language][1])

当前代码：

```python
tl.dot(vals_exp, vals_v)
```

可能导致：

* 某些版本或类型组合不兼容
* 使用较慢的 float32 点积路径
* 无法充分使用 Tensor Core

具体表现取决于 Triton 版本、GPU 和 Q/K/V 数据类型。

---

## 23.7 block 参数非常保守

当前：

```text
BLOCK_M = 16
BLOCK_N = 16
```

适合教学和小尺寸测试，但通常不是高性能 attention 的最佳选择。

官方 fused-attention 示例会在更大的 block，例如 64、128，以及不同 `num_warps`、`num_stages` 之间调优。([Triton Language][1])

实际代码一般使用：

```python
@triton.autotune(...)
```

针对序列长度、head dimension 和 GPU 自动选择配置。

---

# 24. 一句话总结执行过程

每个 Triton program：

```text
1. 读取 16 个 query
2. 每次读取 16 个 key 和 value
3. 计算一个 16×16 的 QK score block
4. 应用滑动窗口 mask
5. 在线更新每个 query 的最大值和 softmax 分母
6. 在线累积 softmax(score) × V
7. 遍历完所有 key 后归一化
8. 写出 16×d 的结果
```

核心在线更新公式是：

[
m^{new}=\max(m^{old},m^{block})
]

[
l^{new}
=======

l^{old}e^{m^{old}-m^{new}}
+
\sum_{\text{block}}e^{s-m^{new}}
]

[
O^{new}
=======

O^{old}e^{m^{old}-m^{new}}
+
\sum_{\text{block}}e^{s-m^{new}}V
]

最后：

[
\text{output}=\frac{O}{l}
]

因此，这段代码本质上是一个 **单头、无 batch、双向滑动窗口、分块在线 softmax 的 attention 前向内核**。

[1]: https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html "Fused Attention — Triton documentation"
[2]: https://triton-lang.org/main/python-api/generated/triton.language.program_id.html?utm_source=chatgpt.com "triton.language.program_id"
[3]: https://triton-lang.org/main/python-api/generated/triton.language.arange.html?utm_source=chatgpt.com "triton.language.arange"
[4]: https://triton-lang.org/main/python-api/generated/triton.language.load.html "triton.language.load — Triton documentation"
[5]: https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html "Matrix Multiplication — Triton documentation"
[6]: https://triton-lang.org/main/python-api/generated/triton.language.permute.html "triton.language.permute — Triton documentation"
[7]: https://triton-lang.org/main/python-api/generated/triton.language.dot.html "triton.language.dot — Triton documentation"
[8]: https://triton-lang.org/main/python-api/generated/triton.language.dot.html?utm_source=chatgpt.com "triton.language.dot"
[9]: https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html "Fused Softmax — Triton documentation"
