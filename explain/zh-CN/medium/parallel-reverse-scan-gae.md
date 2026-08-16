下面按代码结构逐行解释。整体思想是：

\[
\delta_t=r_t+\gamma V_{t+1}-V_t
\]

\[
A_t=\delta_t+(\gamma\lambda)A_{t+1}
\]

把时间维度切成若干块。每个块先并行计算“不考虑右侧块 carry 的局部 GAE”，再计算每个块的 carry，最后将 carry 加回块内每个位置。

---

# 一、导入部分

```python
import torch
import triton
import triton.language as tl
```

- `import torch`：导入 PyTorch，用于张量类型、创建临时 GPU workspace、获取张量 stride。
- `import triton`：导入 Triton 主模块，用于启动 kernel、计算 grid 大小等。
- `import triton.language as tl`：导入 Triton 的 kernel 编程语言接口，例如：
  - `tl.load`
  - `tl.store`
  - `tl.arange`
  - `tl.where`
  - `tl.associative_scan`

---

# 二、仿射变换组合函数

```python
@triton.jit
def _compose_affine(a_left, b_left, a_right, b_right):
```

### `@triton.jit`

表示这个函数会被 Triton JIT 编译成 GPU 代码。

### 函数参数

GAE 的递推为：

\[
A_t=\delta_t+cA_{t+1}
\]

可以把它看成一个仿射变换：

\[
f_t(x)=\delta_t+cx
\]

统一写成：

\[
f(x)=ax+b
\]

其中：

- `a_left`、`b_left`：左侧位置的仿射变换参数；
- `a_right`、`b_right`：右侧已经扫描完成的区间的仿射变换参数。

---

```python
    return a_right * a_left, b_right + a_right * b_left
```

左侧变换：

\[
L(x)=a_Lx+b_L
\]

右侧变换：

\[
R(x)=a_Rx+b_R
\]

先经过左侧，再经过右侧：

\[
R(L(x))=a_R(a_Lx+b_L)+b_R
\]

整理得：

\[
R(L(x))=(a_Ra_L)x+(a_Rb_L+b_R)
\]

所以新的参数是：

```python
a_new = a_right * a_left
b_new = b_right + a_right * b_left
```

函数同时返回这两个值。

---

# 三、乘法组合函数

```python
@triton.jit
def _multiply(left, right):
    return left * right
```

这是一个非常简单的二元结合函数，用于计算后缀乘积。

在第三个 kernel 中，需要得到：

\[
c^{block\_end-t}
\]

这可以通过对块内从右到左的 `c` 做后缀乘积得到。

---

# 四、第一个 kernel：计算块内局部 GAE

```python
@triton.jit
def _gae_local_kernel(
```

定义第一个 Triton kernel。它负责：

1. 计算每个位置的 TD error：`delta`；
2. 在当前时间块内做反向 scan；
3. 将局部 GAE 暂时写入 `advantages`；
4. 保存每个块的摘要和块系数。

---

## 4.1 kernel 参数

```python
    rewards_ptr,
    values_ptr,
    advantages_ptr,
    work_ptr,
```

这四个参数在传入 PyTorch 张量后，会成为 Triton 中的数据指针：

- `rewards_ptr`：奖励张量地址；
- `values_ptr`：价值张量地址；
- `advantages_ptr`：输出张量地址；
- `work_ptr`：块级临时数据地址。

---

```python
    gamma,
    c,
```

- `gamma`：折扣因子；
- `c`：预先计算好的 `gamma * lam`。

之所以单独传入 `c`，是因为 GAE 反向递推只依赖：

\[
c=\gamma\lambda
\]

而不需要反复在 kernel 内计算乘法。

---

```python
    S,
    num_blocks,
```

- `S`：每个序列的时间长度；
- `num_blocks`：每个 batch 被划分成多少个时间块。

例如：

```text
S = 4096
BLOCK_SIZE = 256
num_blocks = 16
```

---

```python
    stride_rb,
    stride_rs,
    stride_vb,
    stride_vs,
    stride_ab,
    stride_as,
```

这些是张量的 stride，单位是“元素个数”，不是字节数：

- `stride_rb`：`rewards` 的 batch 维 stride；
- `stride_rs`：`rewards` 的时间维 stride；
- `stride_vb`：`values` 的 batch 维 stride；
- `stride_vs`：`values` 的时间维 stride；
- `stride_ab`：`advantages` 的 batch 维 stride；
- `stride_as`：`advantages` 的时间维 stride。

对于普通连续张量 `[B, S]`：

```text
stride_batch = S
stride_time = 1
```

但显式传 stride 可以支持非连续张量视图。

---

```python
    BLOCK_SIZE: tl.constexpr,
):
```

`BLOCK_SIZE` 是编译期常量。

加上 `tl.constexpr` 后，Triton 会为这个固定值生成专门编译的 kernel。这里 `BLOCK_SIZE=256`，因此 `tl.arange(0, BLOCK_SIZE)` 等操作在编译时就能确定大小。

---

## 4.2 将 program ID 映射到 batch 和 block

```python
    pid = tl.program_id(0)
```

获取当前 GPU program 的编号。

这个 kernel 的 grid 是：

```python
grid = (B * num_blocks,)
```

所以每个 program 负责：

- 一个 batch；
- 该 batch 中的一个时间块。

---

```python
    batch = pid // num_blocks
```

整数除法，得到当前 program 对应的 batch 编号。

例如：

```text
num_blocks = 16
pid = 35
batch = 35 // 16 = 2
```

---

```python
    block = pid % num_blocks
```

取余，得到当前 batch 内的块编号。

继续上面的例子：

```text
block = 35 % 16 = 3
```

因此 `pid=35` 处理：

```text
batch 2，block 3
```

---

## 4.3 生成块内时间下标

```python
    offsets = tl.arange(0, BLOCK_SIZE)
```

生成一个向量：

```text
[0, 1, 2, ..., 255]
```

这是当前块内部的偏移量。

---

```python
    t = block * BLOCK_SIZE + offsets
```

计算当前 program 要处理的全局时间下标。

例如：

```text
block = 2
BLOCK_SIZE = 256
```

那么：

```text
t = [512, 513, ..., 767]
```

---

```python
    valid = t < S
```

判断每个时间位置是否有效。

当 `S` 不是 `BLOCK_SIZE` 的整数倍时，最后一个块会越界。例如：

```text
S = 300
BLOCK_SIZE = 256
```

第二个块理论上有：

```text
t = 256 ... 511
```

但只有：

```text
t = 256 ... 299
```

是有效的。

`valid` 是一个布尔向量。

---

## 4.4 使用 64 位下标

```python
    b64 = batch.to(tl.int64)
    t64 = t.to(tl.int64)
```

把 batch 和时间下标转换成 64 位整数，用于指针偏移。

这样可以避免张量很大时发生 32 位整数溢出。虽然本题最大 `B*S` 不超过 16,777,216，通常 32 位也够用，但使用 `int64` 更稳妥。

---

## 4.5 加载 rewards

```python
    reward = tl.load(
        rewards_ptr + b64 * stride_rb + t64 * stride_rs,
        mask=valid,
        other=0.0,
    )
```

这里一次加载 `BLOCK_SIZE` 个奖励值。

地址计算为：

\[
address = rewards\_ptr + batch \times stride_{rb} + t \times stride_{rs}
\]

对应 PyTorch 中的：

```python
rewards[batch, t]
```

参数含义：

- `mask=valid`：只加载 `t < S` 的位置；
- `other=0.0`：无效位置填 0。

因此 `reward` 是一个长度为 `BLOCK_SIZE` 的向量。

---

## 4.6 加载当前 values

```python
    value = tl.load(
        values_ptr + b64 * stride_vb + t64 * stride_vs,
        mask=valid,
        other=0.0,
    )
```

加载：

```python
values[batch, t]
```

即当前时间步的 \(V_t\)。

---

## 4.7 加载下一时刻 values

```python
    next_value = tl.load(
        values_ptr + b64 * stride_vb + (t64 + 1) * stride_vs,
        mask=(t + 1) < S,
        other=0.0,
    )
```

加载：

```python
values[batch, t + 1]
```

即 \(V_{t+1}\)。

特别注意 mask：

```python
mask=(t + 1) < S
```

当 `t == S - 1` 时，`t + 1 == S`，已经越界，因此不会加载，而是使用 `other=0.0`。

这就实现了题目要求：

\[
V_S=0
\]

---

## 4.8 计算 TD error

```python
    delta = reward + gamma * next_value - value
```

对应公式：

\[
\delta_t=r_t+\gamma V_{t+1}-V_t
\]

这里 `reward`、`next_value` 和 `value` 都是向量，因此这一行会并行计算当前块内 256 个位置的 TD error。

对于最后一个位置：

```text
next_value = 0
```

因此：

\[
\delta_{S-1}=r_{S-1}-V_{S-1}
\]

---

## 4.9 构造仿射变换

```python
    trans_scale = tl.where(valid, c, 1.0)
```

对有效位置：

\[
a=c
\]

对无效位置：

\[
a=1
\]

即：

```python
trans_scale = [c, c, ..., c, 1, 1, ...]
```

无效位置使用 1，是为了让它们成为恒等变换的一部分。

---

```python
    trans_shift = tl.where(valid, delta, 0.0)
```

对有效位置：

\[
b=\delta_t
\]

对无效位置：

\[
b=0
\]

所以每个位置表示的仿射变换为：

- 有效位置：

\[
x\rightarrow cx+\delta_t
\]

- 无效位置：

\[
x\rightarrow x
\]

无效位置相当于恒等变换：

\[
x\rightarrow 1x+0
\]

这样最后一个不完整块也能使用相同逻辑。

---

## 4.10 块内反向 scan

```python
    block_scale, local_advantage = tl.associative_scan(
        (trans_scale, trans_shift),
        axis=0,
        combine_fn=_compose_affine,
        reverse=True,
    )
```

这是第一个 kernel 最核心的一行。

### 输入

同时扫描两个向量：

```python
(trans_scale, trans_shift)
```

每个元素表示：

\[
x\rightarrow ax+b
\]

### 扫描轴

```python
axis=0
```

表示沿着当前向量的时间维扫描。

### 组合函数

```python
combine_fn=_compose_affine
```

指定用前面定义的仿射变换组合规则。

### 扫描方向

```python
reverse=True
```

表示从右向左扫描，也就是：

```text
t = 块尾 -> 块头
```

这正好符合 GAE 的依赖方向：

\[
A_t=\delta_t+cA_{t+1}
\]

---

### 两个返回值

#### `local_advantage`

表示假设当前块右边界的 advantage 为 0 时，当前位置的局部 GAE：

\[
L_t=\delta_t+c\delta_{t+1}+c^2\delta_{t+2}+\cdots
\]

但只累加到当前块末尾。

#### `block_scale`

表示右侧块 carry 对当前位置的乘性影响：

\[
P_t=c^{block\_end-t}
\]

因此块内任意位置可以写成：

\[
A_t=L_t+P_tA_{block\_end}
\]

其中：

- \(L_t\) 是 `local_advantage`；
- \(P_t\) 是 `block_scale`；
- \(A_{block\_end}\) 是当前块右侧边界的 advantage，目前还不知道。

---

## 4.11 暂时保存局部 GAE

```python
    tl.store(
        advantages_ptr + b64 * stride_ab + t64 * stride_as,
        local_advantage,
        mask=valid,
    )
```

将 `local_advantage` 写入输出张量。

对应：

```python
advantages[batch, t] = local_advantage
```

注意：此时保存的还不是最终答案，因为它还没有加上右侧块的 carry：

\[
A_t=L_t+P_tA_{block\_end}
\]

第三个 kernel 会补上：

```python
block_scale * carry
```

---

## 4.12 计算 work 的起始偏移

```python
    work_base = b64 * (2 * num_blocks)
```

`work` 的逻辑形状是：

```text
[B, 2, num_blocks]
```

展开成一维后，每个 batch 占据：

```text
2 * num_blocks
```

个 float32。

因此当前 batch 的起始位置是：

\[
batch\times 2\times num\_blocks
\]

---

## 4.13 找到第 0 个 lane

```python
    first_lane = offsets == 0
```

生成布尔向量，只有块内第一个位置为 `True`：

```text
[True, False, False, ...]
```

这里关心块起点，因为块起点的局部 GAE 和系数可以概括整个块。

---

## 4.14 提取块起点的局部 GAE

```python
    block_local = tl.sum(
        tl.where(first_lane, local_advantage, 0.0),
        axis=0,
    )
```

逐部分解释：

```python
tl.where(first_lane, local_advantage, 0.0)
```

只保留第 0 个 lane 的 `local_advantage`，其他位置变成 0。

然后：

```python
tl.sum(..., axis=0)
```

把所有 lane 加起来。

因为只有一个非零值，所以结果就是：

```python
local_advantage[0]
```

这是 Triton 中提取向量某个 lane 的常用写法。

`block_local` 表示当前块起点的局部 GAE。

---

## 4.15 提取整个块的系数

```python
    block_coeff = tl.sum(
        tl.where(first_lane, block_scale, 0.0),
        axis=0,
    )
```

同理，提取：

```python
block_scale[0]
```

对于完整块，它的值是：

\[
c^{256}
\]

对于最后一个不完整块，假设实际长度为 \(L\)，它的值是：

\[
c^L
\]

这个系数表示：

\[
A_{block\_start}
=
local_{block\_start}
+
block\_coeff
\times
A_{block\_end}
\]

---

## 4.16 保存块摘要

```python
    tl.store(work_ptr + work_base + block, block_local)
```

将当前块起点的局部 GAE 保存到 `work` 的 plane 0。

位置是：

```text
work[batch, 0, block]
```

---

```python
    tl.store(
        work_ptr + work_base + num_blocks + block,
        block_coeff,
    )
```

将当前块的系数保存到 plane 1。

位置是：

```text
work[batch, 1, block]
```

---

# 五、第二个 kernel：计算每个块的 carry

```python
@triton.jit
def _gae_carry_kernel(work_ptr, num_blocks):
```

这个 kernel 每个 program 处理一个 batch。

它要做的事情是从右往左扫描块摘要，计算每个块右边界处的 advantage，也就是进入该块的 carry。

---

```python
    batch = tl.program_id(0)
```

当前 program 对应一个 batch。

grid 是：

```python
(B,)
```

所以 `program_id(0)` 就是 batch 编号。

---

```python
    work_base = batch.to(tl.int64) * (2 * num_blocks)
```

计算当前 batch 在 `work` 中的起始偏移。

---

```python
    carry = tl.zeros((), dtype=tl.float32)
```

初始化 carry 为标量 0。

含义是：

\[
A_S=0
\]

因为最后一个时间块右边界的 advantage 为 0。

---

```python
    for i in range(num_blocks):
```

遍历当前 batch 的所有块。

---

```python
        block = num_blocks - 1 - i
```

将正序循环转换成从右往左的块编号。

例如有 4 个块：

```text
i = 0 -> block = 3
i = 1 -> block = 2
i = 2 -> block = 1
i = 3 -> block = 0
```

GAE 必须从右向左传播 carry。

---

```python
        block_local = tl.load(work_ptr + work_base + block)
```

读取当前块起点的局部 GAE：

\[
L_{block\_start}
\]

---

```python
        block_coeff = tl.load(
            work_ptr + work_base + num_blocks + block
        )
```

读取当前块的乘性系数：

\[
P_{block}=c^{block\_length}
\]

---

```python
        tl.store(work_ptr + work_base + block, carry)
```

将当前 `carry` 写回 plane 0。

此时 plane 0 中的内容不再是局部摘要，而是“进入当前块的 carry”。

当前块中任意位置最终需要：

\[
A_t=L_t+P_tA_{block\_end}
\]

这里的 `carry` 就是：

\[
A_{block\_end}
\]

---

```python
        carry = block_local + block_coeff * carry
```

更新 carry，使其变成当前块起点的完整 advantage：

\[
A_{block\_start}
=
L_{block\_start}
+
P_{block}A_{block\_end}
\]

下一轮处理左边一个块时，这个值就是它的 `A_block_end`。

---

# 六、第三个 kernel：将 carry 加回块内每个位置

```python
@triton.jit
def _gae_apply_carry_kernel(
```

定义第三个 kernel。

它负责：

\[
A_t=L_t+P_tA_{block\_end}
\]

其中：

- \(L_t\)：第一个 kernel 已经写入 `advantages`；
- \(A_{block\_end}\)：第二个 kernel 已经写入 `work`；
- \(P_t\)：本 kernel 通过后缀乘积重新计算。

---

## 6.1 参数

```python
    advantages_ptr,
    work_ptr,
    c,
    S,
    num_blocks,
```

- `advantages_ptr`：暂存局部 GAE、最终保存完整 GAE；
- `work_ptr`：读取每个块的 carry；
- `c`：等于 `gamma * lam`；
- `S`：序列长度；
- `num_blocks`：每个 batch 的块数。

---

```python
    stride_ab,
    stride_as,
```

输出张量 `advantages` 的两个 stride。

---

```python
    BLOCK_SIZE: tl.constexpr,
):
```

块大小，编译期常量。

---

## 6.2 映射 program ID

```python
    pid = tl.program_id(0)
    batch = pid // num_blocks
    block = pid % num_blocks
```

和第一个 kernel 一样：

- 一个 program 处理一个 batch 中的一个时间块；
- `batch` 是 batch 编号；
- `block` 是时间块编号。

---

## 6.3 计算时间下标

```python
    offsets = tl.arange(0, BLOCK_SIZE)
    t = block * BLOCK_SIZE + offsets
    valid = t < S
```

与第一个 kernel 相同：

- `offsets` 是块内偏移；
- `t` 是全局时间位置；
- `valid` 防止最后一个块越界。

---

## 6.4 转换下标类型

```python
    b64 = batch.to(tl.int64)
    t64 = t.to(tl.int64)
```

转换为 64 位整数，用于安全的地址计算。

---

## 6.5 读取局部 GAE

```python
    local_advantage = tl.load(
        advantages_ptr + b64 * stride_ab + t64 * stride_as,
        mask=valid,
        other=0.0,
    )
```

从 `advantages` 中读取第一个 kernel 暂存的局部 GAE：

\[
L_t
\]

此时 `advantages` 里还不是最终结果。

---

## 6.6 构造后缀乘积输入

```python
    trans_scale = tl.where(valid, c, 1.0)
```

有效位置填入 `c`，无效位置填入 `1`。

例如一个有效长度为 4 的块：

```text
[c, c, c, c, 1, 1, ...]
```

---

## 6.7 计算 carry 系数

```python
    carry_scale = tl.associative_scan(
        trans_scale,
        axis=0,
        combine_fn=_multiply,
        reverse=True,
    )
```

对 `trans_scale` 从右向左做乘法 scan。

对于一个完整块，结果是：

```text
位置 t=block_start:     c^256
位置 t=block_start+1:   c^255
...
位置 t=block_end-2:     c^2
位置 t=block_end-1:     c
```

也就是：

\[
carry\_scale_t=c^{block\_end-t}
\]

这正是当前位置乘以右侧块 carry 时需要的系数。

---

## 6.8 读取当前块的 carry

```python
    work_base = b64 * (2 * num_blocks)
    carry = tl.load(work_ptr + work_base + block)
```

先计算当前 batch 在 `work` 中的偏移，然后读取：

```text
work[batch, 0, block]
```

第二个 kernel 已经把 plane 0 改成了 carry，因此这里得到：

\[
A_{block\_end}
\]

---

## 6.9 得到最终 advantage

```python
    result = local_advantage + carry_scale * carry
```

对应公式：

\[
A_t=L_t+c^{block\_end-t}A_{block\_end}
\]

其中：

- `local_advantage`：\(L_t\)；
- `carry_scale`：\(c^{block\_end-t}\)；
- `carry`：\(A_{block\_end}\)。

---

## 6.10 写回最终结果

```python
    tl.store(
        advantages_ptr + b64 * stride_ab + t64 * stride_as,
        result,
        mask=valid,
    )
```

将最终 GAE 写入预先分配的 `advantages` 张量。

到这里，`advantages` 中的临时局部结果被完整的最终结果覆盖。

---

# 七、Python 入口函数 `solve`

```python
# rewards, values, advantages are tensors on the GPU
def solve(
```

题目要求所有输入张量都已经在 GPU 上。

---

## 7.1 函数参数

```python
    rewards: torch.Tensor,
    values: torch.Tensor,
    advantages: torch.Tensor,
    gamma: float,
    lam: float,
    B: int,
    S: int,
):
```

参数分别是：

- `rewards`：形状 `[B, S]` 的奖励；
- `values`：形状 `[B, S]` 的价值；
- `advantages`：预先分配的输出；
- `gamma`：折扣因子；
- `lam`：GAE 的 λ；
- `B`：batch 大小；
- `S`：序列长度。

---

## 7.2 设置块大小

```python
    BLOCK_SIZE = 256
```

每个 program 处理 256 个连续时间步。

在性能测试尺寸：

```text
B = 64
S = 4096
```

下：

\[
num\_blocks=4096/256=16
\]

总 program 数为：

\[
64\times16=1024
\]

这比每个 batch 只使用一个 program 的串行反向循环更适合 Tesla T4。

---

## 7.3 计算块数

```python
    num_blocks = triton.cdiv(S, BLOCK_SIZE)
```

等价于：

\[
num\_blocks=\left\lceil\frac{S}{BLOCK\_SIZE}\right\rceil
\]

例如：

```text
S = 300
BLOCK_SIZE = 256
num_blocks = 2
```

---

## 7.4 计算 GAE 衰减系数

```python
    c = float(gamma) * float(lam)
```

对应：

\[
c=\gamma\lambda
\]

后面的反向递推使用：

\[
A_t=\delta_t+cA_{t+1}
\]

---

## 7.5 创建临时 workspace

```python
    work = torch.empty(
        (B, 2, num_blocks),
        device=advantages.device,
        dtype=torch.float32,
    )
```

创建一个 GPU 临时张量，逻辑形状：

```text
[B, 2, num_blocks]
```

其中：

- `work[b, 0, k]`：先保存块摘要，后保存进入块的 carry；
- `work[b, 1, k]`：保存块系数 \(c^{block\_length}\)。

对于性能测试尺寸：

```text
B = 64
num_blocks = 16
```

`work` 只有：

```text
64 × 2 × 16 = 2048
```

个 float32，约 8 KB，非常小。

---

## 7.6 设置 grid

```python
    grid = (B * num_blocks,)
```

第一和第三个 kernel 都使用这个 grid。

每个 program 处理：

```text
一个 batch 中的一个 BLOCK_SIZE 时间块
```

---

## 7.7 启动第一个 kernel

```python
    _gae_local_kernel[grid](
```

使用 `grid` 启动 `_gae_local_kernel`。

---

```python
        rewards,
        values,
        advantages,
        work,
```

将四个张量传入 kernel。Triton 会把它们转换成设备指针。

---

```python
        float(gamma),
        c,
        S,
        num_blocks,
```

传入：

- `gamma`；
- `c = gamma * lam`；
- 序列长度；
- 每个 batch 的块数。

---

```python
        rewards.stride(0),
        rewards.stride(1),
        values.stride(0),
        values.stride(1),
        advantages.stride(0),
        advantages.stride(1),
```

传入三个张量的 batch 维和时间维 stride。

对于连续张量 `[B, S]`：

```text
stride(0) = S
stride(1) = 1
```

使用 stride 让 kernel 可以正确处理非连续二维视图。

---

```python
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
```

- `BLOCK_SIZE=256`：传给 kernel 的编译期常量；
- `num_warps=4`：每个 Triton program 使用 4 个 warp，即 128 个线程。

---

## 7.8 启动第二个 kernel

```python
    _gae_carry_kernel[(B,)](
        work,
        num_blocks,
        num_warps=1,
    )
```

第二个 kernel 的 grid 是：

```python
(B,)
```

也就是每个 batch 一个 program。

`num_warps=1` 的原因：

- 每个 program 只做 `num_blocks` 次很小的标量循环；
- 没有必要使用多个 warp；
- 单 warp 更节省资源。

---

## 7.9 启动第三个 kernel

```python
    _gae_apply_carry_kernel[grid](
        advantages,
        work,
        c,
        S,
        num_blocks,
```

传入：

- 暂存局部 GAE 的 `advantages`；
- 保存 carry 的 `work`；
- 衰减系数 `c`；
- 序列长度 `S`；
- 块数 `num_blocks`。

---

```python
        advantages.stride(0),
        advantages.stride(1),
```

传入输出张量的 stride。

---

```python
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
```

第三个 kernel 与第一个 kernel 一样，每个 program 处理 256 个时间位置，因此使用 4 个 warp。

---

# 八、三个 kernel 的完整数据流

## 第一个 kernel

计算：

\[
\delta_t=r_t+\gamma V_{t+1}-V_t
\]

并得到：

\[
L_t=\delta_t+c\delta_{t+1}+\cdots+c^{block\_end-1-t}\delta_{block\_end-1}
\]

暂存：

```text
advantages[b, t] = L_t
```

同时保存：

```text
work[b, 0, block] = L_block_start
work[b, 1, block] = c^block_length
```

---

## 第二个 kernel

从最右侧块开始：

```text
carry = 0
```

反向计算：

\[
A_{block\_start}
=
L_{block\_start}
+
c^{block\_length}
A_{block\_end}
\]

并把旧的 `A_block_end` 存回：

```text
work[b, 0, block] = carry
```

---

## 第三个 kernel

读取：

```text
L_t = advantages[b, t]
carry = work[b, 0, block]
```

计算：

\[
A_t=L_t+c^{block\_end-t}carry
\]

最后写回：

```text
advantages[b, t] = A_t
```

---

# 九、为什么这种写法比单个反向循环更适合 T4

如果每个 batch 只启动一个 program，那么：

```text
B = 64
```

最多只有 64 个 program，而且每个 program 要串行处理 4096 个时间步。

这份实现将每个序列拆成 16 个块：

```text
64 × 16 = 1024 个 program
```

块内的 256 个位置又可以并行 scan，因此更适合 Tesla T4 这类具有多个 SM 的 GPU。

整个算法仍然保持数学上的严格反向依赖，只是把递推拆成了：

```text
块内局部 scan -> 块间 carry scan -> 应用 carry
```

三个阶段。