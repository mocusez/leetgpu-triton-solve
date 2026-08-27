下面按代码顺序逐段解释 `fused_add_rmsnorm.py`。

## 1. 导入

```python
import torch
import triton
import triton.language as tl
```

- `torch`：张量对象和启动 kernel 的宿主环境。
- `triton`：JIT 编译器入口（`@triton.jit`、`triton.next_power_of_2` 等）。
- `triton.language as tl`：kernel 内部使用的编程语言，提供 `tl.load`、`tl.store`、`tl.sum` 等在 GPU 上并行执行的算子。

## 2. 单块快路径 kernel（C ≤ 8192）

```python
@triton.jit
def _add_rmsnorm_fused_kernel(
    x_ptr, res_ptr, w_ptr, out_ptr,
    C, eps,
    BLOCK: tl.constexpr,
):
```

- `@triton.jit`：告诉 Triton 把函数编译成 GPU kernel（对每个 BLOCK 取值生成一份机器码）。
- `x_ptr, res_ptr, w_ptr, out_ptr`：传入的其实是张量，Triton 取其数据指针，在 kernel 内按 C 指针方式寻址。
- `C, eps`：运行时标量参数。
- `BLOCK: tl.constexpr`：**编译期常量**。它是 2 的幂且 ≥ C。`constexpr` 让编译器知道块大小，从而确定寄存器分配和展开循环。

```python
    row = tl.program_id(0)
```

- 当前 program（类似 CUDA 的 block）的编号。启动网格是 `(N,)`，所以每个 program 负责**一行**（一个 token）。

```python
    cols = tl.arange(0, BLOCK)
    mask = cols < C
```

- `cols`：生成 `[0, 1, ..., BLOCK-1]` 的列索引向量，代表本 program 内各线程负责的特征维度。
- `mask`：因为 `BLOCK ≥ C` 且是 2 的幂（例如 C=4000 时 BLOCK=4096），越界的列用 mask 屏蔽，防止读到下一行的数据。

```python
    x = tl.load(x_ptr + row * C + cols, mask=mask, other=0.0)
    r = tl.load(res_ptr + row * C + cols, mask=mask, other=0.0)
    z = x + r
```

- `x_ptr + row * C + cols`：第 `row` 行、各 `cols` 列的地址（行主序，行偏移 = 行号 × 行宽）。这是向量化加载：一条指令由整个 program 的线程协作把整行读进寄存器。
- `mask=mask, other=0.0`：越界位置填 0.0——0 的平方还是 0，不会污染后面的平方和。
- `z = x + r`：向量加法。**关键点**：`z` 只存在于寄存器中，永远不会写回全局内存，这就是"融合"省下的那一轮读写。

```python
    ms = tl.sum(z * z, axis=0) / C
    rstd = 1.0 / tl.sqrt(ms + eps)
```

- `z * z`：逐元素平方；`tl.sum(..., axis=0)`：把整个向量归约成一个标量（Triton 自动生成 warp 内 shuffle 归约 + 少量共享内存的跨 warp 归约）。
- `ms`：行内平方的均值，即 $\frac{1}{C}\sum z_j^2$。注意除的是真实列数 `C` 而不是 `BLOCK`，因为越界位置贡献的是 0。
- `rstd`：均方根的倒数 $1/\sqrt{ms + \varepsilon}$。用倒数是为了后面把除法变成乘法，GPU 上乘法更快。

```python
    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    tl.store(out_ptr + row * C + cols, z * rstd * w, mask=mask)
```

- 加载 per-feature 权重向量 `weight`。
- `z * rstd * w`：先归一化（乘 rstd）再乘权重，等价于题目的 $\frac{z_{i,j}}{rms_i}\cdot w_j$，一次乘完直接写回 `out` 的对应位置。

## 3. 循环回退路径 kernel（C > 8192）

签名相同，区别在函数体——一行太大放不下单个块，分块循环：

```python
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(0, C, BLOCK):
        offs = start + cols
        mask = offs < C
        x = tl.load(x_ptr + row * C + offs, mask=mask, other=0.0)
        r = tl.load(res_ptr + row * C + offs, mask=mask, other=0.0)
        z = x + r
        acc += z * z
```

- **第一遍循环**：以 `BLOCK=4096` 为步长扫过整行，`acc` 逐块累加平方（注意是向量累加，循环结束后才 `tl.sum` 归约一次，比每块归约更高效）。

```python
    ms = tl.sum(acc, axis=0) / C
    rstd = 1.0 / tl.sqrt(ms + eps)
```

- 与单块路径相同，得到该行 rstd。

```python
    for start in range(0, C, BLOCK):
        ...
        tl.store(out_ptr + row * C + offs, (x + r) * rstd * w, mask=mask)
```

- **第二遍循环**：重新读 `x`/`residual` 算出 `(x + r)`，归一化后写出。重读的代价很低——数据刚读过，还热在 L2 缓存里。虽然比单块路径多一轮读取，但依然满足"不把 z 写到全局内存"的融合要求。

## 4. 启动逻辑

```python
_MAX_SINGLE_BLOCK = 8192
```

- 单块路径的上限。BLOCK=8192 时每个 program 要驻留约 3 个 8192 宽的向量，寄存器压力仍在 T4 单 block 预算内；再大就切循环路径更稳。

```python
    block = triton.next_power_of_2(C)
    if block <= _MAX_SINGLE_BLOCK:
        num_warps = max(1, min(32, block // 512))
        _add_rmsnorm_fused_kernel[(N,)](
            x, residual, weight, out, C, eps,
            BLOCK=block, num_warps=num_warps,
        )
```

- `next_power_of_2(C)`：Triton 的 `tl.arange` 要求长度是 2 的幂，所以块大小向上取整。
- `num_warps = block // 512`（夹在 1~32）：每个 warp 处理 512 个元素。评测规模 C=4096 → 8 个 warp = 256 线程，每线程 16 个元素，是 T4 上延迟隐藏和寄存器占用之间比较平衡的配置。
- `kernel[(N,)](...)`：启动网格为 N 个 program，每行一个；`BLOCK=...` 作为 constexpr 传入触发 JIT 编译（同一 BLOCK 值只编译一次，后续直接复用缓存）。

```python
    else:
        _add_rmsnorm_fused_loop_kernel[(N,)](
            x, residual, weight, out, C, eps,
            BLOCK=4096, num_warps=8,
        )
```

- C 超过 8192 时走循环路径，固定 4096 块、8 warps。

## 设计要点回顾

| 决策 | 原因 |
|---|---|
| 每 program 处理一行 | RMS 是行内归约，天然无跨 program 通信，无需原子操作或同步 |
| z 留在寄存器 | 融合的核心收益：省掉一次完整的 (N,C) 全局内存读写 |
| `other=0.0` 填充 | 0² = 0，不影响平方和 |
| 求 rstd 用倒数 | 把 N·C 次除法换成乘法 |
| BLOCK ≤ 8192 单块 / 否则循环 | 评测规模 C=4096 走最快的单次遍历路径，同时覆盖题目 65536 的上限 |