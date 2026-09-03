下面按代码顺序逐行讲解。

## Kernel 部分

```python
@triton.jit
```
Triton 的装饰器，标记这是一个 JIT 编译的 GPU kernel。第一次调用时 Triton 会根据参数类型和 `tl.constexpr` 的值把这段 Python 编译成 GPU 机器码（在 T4 上就是 sm_75 的 SASS），之后复用编译缓存。

```python
def _dpo_loss_kernel(
    chosen_ptr, rejected_ptr, chosen_ref_ptr, rejected_ref_ptr,
    out_ptr,
    beta, inv_b,
    B,
    BLOCK: tl.constexpr,
):
```
Kernel 的参数。传入 torch 张量时 Triton 会自动取它的显存地址（指针），所以前五个参数在 kernel 内部当指针用。`beta`、`inv_b` 是标量浮点数，`B` 是标量整数。`BLOCK: tl.constexpr` 表示它是**编译期常量**——每个不同的 `BLOCK` 值会单独编译一份 kernel，常量化的好处是循环展开、寄存器分配都能做死。

```python
    pid = tl.program_id(0)
```
取当前 program（类似 CUDA 的 block）在第 0 维的编号。B=65536、BLOCK=1024 时，grid 有 64 个 program，`pid` 取值 0~63，各自负责一段 1024 元素的数据。

```python
    offs = pid * BLOCK + tl.arange(0, BLOCK)
```
`tl.arange(0, BLOCK)` 生成 `[0, 1, ..., 1023]` 的向量，加上 `pid * BLOCK` 的偏移后，`offs` 就是这个 program 负责的 1024 个全局下标。Triton 的编程模型是**块级向量**：一条语句同时操作 BLOCK 个元素，由编译器映射到线程。

```python
    mask = offs < B
```
越界掩码。B 不一定是 1024 的整数倍（比如 B=1000），最后一个 program 的部分下标会超出数组范围，`mask` 标记哪些 lane 是合法的。

```python
    l_plus    = tl.load(chosen_ptr + offs,       mask=mask, other=0.0)
    l_minus   = tl.load(rejected_ptr + offs,     mask=mask, other=0.0)
    l_plus_r  = tl.load(chosen_ref_ptr + offs,   mask=mask, other=0.0)
    l_minus_r = tl.load(rejected_ref_ptr + offs, mask=mask, other=0.0)
```
从显存加载四组 log probability。`chosen_ptr + offs` 是指针加法，得到每个 lane 的地址；`mask=mask` 保证越界 lane 不真正访存（避免非法地址错误），`other=0.0` 给这些 lane 填 0。四条都是连续 1024 个 fp32 的合并访存（coalesced），这是 GPU 上最高效的访存模式。

```python
    z = beta * ((l_plus - l_minus) - (l_plus_r - l_minus_r))
```
按公式算 preference logit：`z = β·[(l⁺ − l⁻) − (l⁺ref − l⁻ref)]`，即当前策略的 chosen/rejected 对数概率差，减去参考策略的对应差，再乘 β。整条是逐元素向量运算，1024 个元素一条语句完成。

```python
    x = -z
    sp = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(x)))
```
计算 `softplus(-z)`，用的是数值稳定的等价形式：

$$\operatorname{softplus}(x) = \max(x, 0) + \log\!\left(1 + e^{-|x|}\right)$$

为什么这样写：如果直接算 `log(1 + exp(x))`，当 `x` 是很大的正数时 `exp(x)` 会溢出成 inf。改写后 `exp(-|x|)` 的参数永远 ≤ 0，结果落在 (0, 1]，绝不溢出；而 `x` 很大时 `max(x,0)` 就是 x 本身，`log(1+exp(-|x|))` 趋近 0，正好逼近 `softplus(x) ≈ x` 的渐近行为。这正对应题目"remains finite for very large logits"的要求。

```python
    sp = tl.where(mask, sp, 0.0)
```
把越界 lane 的贡献清零。这一步必须有：虽然加载时越界 lane 填了 0，但那会算出 `z=0 → softplus(0)=ln2 ≈ 0.693`，如果留着会被加进总和，结果就偏了。

```python
    partial = tl.sum(sp, axis=0)
```
块内归约：把 1024 个元素的 `sp` 求和成一个标量。Triton 会用 warp shuffle / 共享内存做树形归约（在 T4 上是标准操作），不用手写。

```python
    tl.atomic_add(out_ptr, partial * inv_b)
```
把这个 program 的部分和乘上 `1/B`（即 `inv_b`），原子加到 `output[0]`。64 个 program 并发执行，`atomic_add` 保证加法不丢数据；每个 program 加的是自己那份的 `partial/B`，全部加完正好等于均值。先乘 `inv_b` 再加是因为均值 = Σ(sp)/B，把它摊到每个块里，就不需要第二次 kernel launch 来做除法了。

## 封装函数部分

```python
def solve(
    chosen_logps: torch.Tensor,
    rejected_logps: torch.Tensor,
    chosen_ref_logps: torch.Tensor,
    rejected_ref_logps: torch.Tensor,
    output: torch.Tensor,
    beta: float,
    B: int,
):
```
按题目要求的签名，四个输入和输出都是已在 GPU 上的张量，结果写进 `output[0]`。

```python
    output.zero_()
```
把 output 清零。因为 kernel 用的是 `atomic_add` **累加**，如果 output 里有残留值，结果就错了。末尾的下划线表示原地修改。

```python
    BLOCK = 1024
    grid = (triton.cdiv(B, BLOCK),)
```
`triton.cdiv(B, BLOCK)` 是向上取整除法 `ceil(B/1024)`，决定启动多少个 program。B=65536 时是 64；B=1000 时是 1（靠 mask 保护）。

```python
    _dpo_loss_kernel[grid](
        chosen_logps, rejected_logps, chosen_ref_logps, rejected_ref_logps,
        output,
        beta, 1.0 / B,
        B,
        BLOCK=BLOCK,
        num_warps=4,
    )
```
启动 kernel。`kernel[grid](...)` 是 Triton 的启动语法；torch 张量在这里被自动转成指针传入。`1.0 / B` 在 Python 侧算好倒数，kernel 里 64 个 program 就都不用各自再做除法了。`num_warps=4` 表示每个 program 用 4 个 warp（128 线程）处理 1024 个元素，每线程 8 个元素，对 T4 是合理的占用配置。

## 整体数据流

```
四个 [B] 张量
   │  64 个 program 并行，每个读 1024×4 个 fp32
   ▼
z = β·[(l⁺−l⁻) − (l⁺ref−l⁻ref)]        ← 逐元素
   ▼
softplus(−z)（稳定形式），掩码清零越界 lane
   ▼
tl.sum 块内归约 → 每块一个标量
   ▼
×(1/B) 后 atomic_add → output[0] 得到均值
```

计算量本身极小（65536 个元素、每个十几次浮点运算），瓶颈在显存带宽（共读 1MB），所以这个 kernel 在 T4 上基本一次读取就完成，耗时是微秒级。