这是上一版的**完全融合升级版**：只用一个 kernel 完成整个 SwiGLU MLP，不再产生 `hidden` 中间矩阵。它采用的是 **split-F（沿 d_ffn 切分）** 思路：每个 program 只算 hidden 的一小块 `h_tile`，立刻用它乘 `W_down` 的对应行块，把部分结果**原子加**到 output 上。下面逐行解释，重点说与上一版的差异。

---

## Kernel 签名（第 5–16 行）

```python
@triton.jit
def _swiglu_fused_kernel(
    x_ptr, wg_ptr, wu_ptr, wd_ptr, c_ptr,
    M, K, F, N,                       # M, d_model, d_ffn, d_model
    stride_xm, stride_xk,
    stride_wgk, stride_wgn,
    stride_wuk, stride_wun,
    stride_wdf, stride_wdn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_F: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
):
```
- 比上一版多了 `wd_ptr`（W_down）和 `c_ptr`（最终输出），现在一个 kernel 吃进全部四个张量。
- 四个维度参数：`M`=batch 行数，`K`=d_model（第一次投影的归约维），`F`=d_ffn（hidden 维，也是第二次乘法的归约维），`N`=d_model（最终输出宽度）。
- 多了一个 tile 尺寸 `BLOCK_F`：hidden 维的切分粒度，这是融合的关键。

## Program 坐标与索引（第 18–27 行）

```python
    pid_m = tl.program_id(0)
    pid_f = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rf = pid_f * BLOCK_F + tl.arange(0, BLOCK_F)
    rk = tl.arange(0, BLOCK_K)

    mask_m = rm < M
    mask_f = rf < F
```
- grid 的第 1 维从「输出列块」变成了 **hidden 维的块**（`pid_f`）。每个 program 负责：行块 `rm` × hidden 块 `rf`。
- `mask_f` 对应上一版的 `mask_n`，只是含义从输出列边界变成 hidden 维边界。

## 第一阶段：算 hidden 块（第 29–56 行）

```python
    x_ptrs = x_ptr + rm[:, None] * stride_xm + rk[None, :] * stride_xk
    wg_ptrs = wg_ptr + rk[:, None] * stride_wgk + rf[None, :] * stride_wgn
    wu_ptrs = wu_ptr + rk[:, None] * stride_wuk + rf[None, :] * stride_wun

    acc_g = tl.zeros((BLOCK_M, BLOCK_F), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_M, BLOCK_F), dtype=tl.float32)
```
- 与上一版相同的指针构造，只是累加器形状是 `(BLOCK_M, BLOCK_F)`——这个 program 只产出 hidden 的一个窄条。

```python
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem = K - k * BLOCK_K
        x_tile = tl.load(x_ptrs,
                         mask=mask_m[:, None] & (rk[None, :] < k_rem),
                         other=0.0)
        wg_tile = tl.load(wg_ptrs, ...)
        wu_tile = tl.load(wu_ptrs, ...)
        acc_g += tl.dot(x_tile, wg_tile, allow_tf32=False)
        acc_u += tl.dot(x_tile, wu_tile, allow_tf32=False)

        x_ptrs += BLOCK_K * stride_xk
        wg_ptrs += BLOCK_K * stride_wgk
        wu_ptrs += BLOCK_K * stride_wuk

    h = acc_g * tl.sigmoid(acc_g) * acc_u   # SiLU(gate) * up, in registers
```
- 这部分和上一版逐行一致：沿 K=d_model 分块循环，双路矩阵乘累加，尾部用 `k_rem` 掩码处理，最后寄存器内做 SwiGLU。**关键差别是：这里得到的 `h` 是 `(BLOCK_M, BLOCK_F)` 的一小条，而且根本不写显存，直接留在寄存器里进入下一阶段。** 这就是「融合」省掉的显存往返。

## 第二阶段：hidden 块 × W_down 行块，原子累加（第 58–68 行）

```python
    # h @ W_down[rf, :] accumulated into output[rm, :] via atomics
    for n0 in range(0, N, BLOCK_N):
        rn = n0 + tl.arange(0, BLOCK_N)
        mask_n = rn < N
```
- 矩阵乘法 `hidden @ W_down` 里，hidden 的第 `rf` 列块只和 W_down 的第 `rf` **行块**相乘，贡献到 output 的**所有列**。所以这里沿输出宽度 N 循环，每次处理一个 `BLOCK_N` 宽的列条。

```python
        wd_tile = tl.load(
            wd_ptr + rf[:, None] * stride_wdf + rn[None, :] * stride_wdn,
            mask=mask_f[:, None] & mask_n[None, :], other=0.0)
```
- 加载 W_down 的 `(BLOCK_F, BLOCK_N)` 块：行对应 hidden 块 `rf`，列对应当前输出列条 `rn`。带边界掩码，越界填 0。

```python
        part = tl.dot(h, wd_tile, allow_tf32=False)
```
- 本 program 对 output 这个列条的**部分贡献**：`(BLOCK_M, BLOCK_F) @ (BLOCK_F, BLOCK_N) → (BLOCK_M, BLOCK_N)`。注意完整结果需要所有 `pid_f` 的贡献加起来。

```python
        c_ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        tl.atomic_add(c_ptrs, part, mask=mask_m[:, None] & mask_n[None, :])
```
- `tl.atomic_add`：把部分结果**原子地加**到 output 上。因为多个 `pid_f` 的 program 会写同一块 output 区域，普通 `tl.store` 会互相覆盖，必须用原子加来保证「求和」语义。这是 split-K/split-F 类 kernel 的标志性写法。

## 启动函数（第 72–96 行）

```python
    BLOCK_M, BLOCK_F, BLOCK_K, BLOCK_N = 64, 64, 32, 64

    output.zero_()  # atomics accumulate, so start from zero
```
- **`output.zero_()` 是必须的**：原子加是「读-改-写」累加，如果 output 里是垃圾值，结果就错了。上一版用 `tl.store` 直接覆盖所以不需要这步。

```python
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(d_ffn, BLOCK_F))
    _swiglu_fused_kernel[grid](
        x, W_gate, W_up, W_down, output,
        M, d_model, d_ffn, d_model,
        ...,
        num_warps=4, num_stages=2,
    )
```
- grid 是 `(⌈M/64⌉, ⌈d_ffn/64⌉)`：每个 program 算 hidden 的一列条。注意 grid 规模跟上一版的 kernel1 相同，但每个 program 干的活更多了（还要顺路扫一遍 output 的全部 N 列）。

---

## 两版对比与权衡

| | 上一版（两个 kernel） | 这一版（融合 kernel） |
|---|---|---|
| hidden 矩阵 | 分配 M×d_ffn 显存，写一次读一次 | **完全在寄存器里，零显存开销** |
| kernel 启动 | 2 次 | 1 次 |
| 输出写入 | `tl.store`，无需清零 | `tl.atomic_add`，必须先 `zero_()` |
| W_down 读取 | 每行块读一遍（共 ⌈M/BLOCK_M⌉ 遍） | 每个 (行块, hidden 块) 组合读一遍，次数相同但局部性更差 |
| x / W_gate / W_up 读取 | 每列块读一遍 | **每个 hidden 块独立重读一遍 x 行块**（⌈F/BLOCK_F⌉ 遍，与上版 kernel1 一致） |
| 并行度 | 受 M×N 限制 | grid 第二维是 F，更大时并行度更高 |
| 数值 | 顺序归约，确定性 | 原子加的求和顺序不确定，**多次运行结果可能有微小浮点差异** |

**适用场景**：d_ffn 很大（比如 11008、16384）时收益最明显——省掉一个巨大的中间张量，显存带宽压力小很多。代价是原子加的带宽开销（fp32 原子加在 T4 上没有硬件加速，走 L2 原子操作，N=d_model 越大原子写流量越大）和运行间的非确定性。如果 M 很小（推理 decode 场景 M=1~几十），这版通常更划算；如果 M 很大，两个 kernel 的经典形态往往吞吐更高。
