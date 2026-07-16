
---

### 1. Kernel 签名与参数

```python
@triton.jit
def selective_scan_fwd_kernel(
    u_ptr, delta_ptr, A_ptr, B_ptr, C_ptr, skip_ptr, y_ptr,
    batch, seq_len, d_model, d_state,
    u_stride_b, u_stride_t, u_stride_d,
    ...
    BLOCK_DSTATE: tl.constexpr,
):
```

- `@triton.jit`：将 Python 函数编译为 GPU kernel。所有参数在编译时分为两类：
  - **运行时参数**（大写以外的）：调用时传入的实际值（指针、整数）。
  - **编译时常量**（`tl.constexpr`）：编译期确定，用于生成静态优化的机器码。这里 `BLOCK_DSTATE` 决定寄存器分配和循环展开方式。

- **指针参数**：`u_ptr` 等是 `tl.pointer_type` 的底层地址，不是 PyTorch Tensor 对象。必须通过 `stride` 手动计算偏移。

- **Stride 参数**：每个张量传入 3 个 stride（如 `u_stride_b`, `u_stride_t`, `u_stride_d`），对应 PyTorch Tensor 的 `stride()[0]`, `stride()[1]`, `stride()[2]`。这允许 kernel 处理**非连续内存布局**（如转置、切片后的 Tensor），而不假设 `stride_d == 1`。

---

### 2. 并行粒度：从 Program ID 到 (batch, d_model)

```python
    pid = tl.program_id(0)
    b = pid // d_model
    d = pid % d_model
```

- Triton 的 `program_id` 对应 CUDA 的 **blockIdx**。这里使用 1D grid。
- 将线性 `pid` 映射到二维逻辑坐标 `(b, d)`：
  - `b`：batch 索引
  - `d`：`d_model`（通道）索引
- **关键设计**：每个 Program（block）只负责**一个** `(b, d)` 对。这意味着 `batch × d_model` 个 block 可以完全并行启动。在测试配置（4×512=2048）下，能充分填满 T4 的 40 个 SM。

---

### 3. 线程索引与 Mask

```python
    tid = tl.arange(0, BLOCK_DSTATE)
    mask = tid < d_state
```

- `tl.arange(0, BLOCK_DSTATE)`：生成 `[0, 1, ..., BLOCK_DSTATE-1]` 的向量。这对应 CUDA block 内线程的“向量化”操作。
- `mask`：当 `d_state` 不是 2 的幂时（如 16、48），用于屏蔽越界线程。所有后续的 `tl.load` 和 `tl.where` 都使用此 mask，确保不会访问非法内存或引入错误计算。

---

### 4. 预加载静态权重 A 与 skip

```python
    A_d = tl.load(A_ptr + d * A_stride_d + tid * A_stride_n, mask=mask, other=0.0)
    skip_d = tl.load(skip_ptr + d * skip_stride_d)
```

- **A[d, :] 的加载**：`A` 是 `[d_model, d_state]`。对于固定的 `d`，整行 `A_d` 是**只读常量**，在 `seq_len` 循环中复用。通过 `tid` 向量化一次加载整行到寄存器。
- `other=0.0`：对 mask 为 False 的位置填充 0，避免 NaN 传播。
- **skip[d] 的加载**：标量，`skip` 是 `[d_model]`。每个通道一个标量，同样在整个序列中复用。

---

### 5. 隐藏状态初始化

```python
    h = tl.zeros((BLOCK_DSTATE,), dtype=tl.float32)
```

- `h` 是 shape `(BLOCK_DSTATE,)` 的**寄存器向量**。
- 初始化为 0，对应题目条件 $h_{b,-1,d,n} = 0$。
- 注意：这是**每个线程私有的寄存器**，不是 shared memory。`d_state` 最大 64，寄存器压力极低。

---

### 6. 顺序扫描：seq_len 循环

```python
    for t in range(seq_len):
```

- 这是整个 kernel 中**唯一串行**的部分。由于递推关系 $h_t = \bar{A}_t h_{t-1} + \bar{B}_t u_t$ 存在数据依赖，无法在 `seq_len` 维度并行。
- 但 `batch × d_model` 的 2048 个 block 并行，以及 block 内 `d_state` 的向量化并行，已经充分利用了 GPU。

---

### 7. 加载标量 u 与 delta

```python
        u_btd = tl.load(u_ptr + b * u_stride_b + t * u_stride_t + d * u_stride_d)
        delta_btd = tl.load(delta_ptr + b * delta_stride_b + t * delta_stride_t + d * delta_stride_d)
```

- `u[b, t, d]` 和 `delta[b, t, d]` 是**标量**（单个 float32）。
- 计算地址时显式使用 3 个 stride，支持任意内存布局。若 `u` 是连续张量，`u_stride_d` 通常为 1。
- 这两个值在每个时间步变化，必须在循环内加载。

---

### 8. 离散化：计算 A_bar

```python
        A_bar = tl.exp(delta_btd * A_d)
```

- **数学**：$\bar{A}_{b,t,d,n} = \exp(\Delta_{b,t,d} \cdot A_{d,n})$
- `delta_btd` 是标量，`A_d` 是向量。Triton 自动广播（标量 × 向量 = 向量）。
- 结果 `A_bar` 是 shape `(BLOCK_DSTATE,)` 的寄存器向量。

---

### 9. 加载 B 与 C

```python
        B_bt = tl.load(B_ptr + b * B_stride_b + t * B_stride_t + tid * B_stride_n, mask=mask, other=0.0)
        C_bt = tl.load(C_ptr + b * C_stride_b + t * C_stride_t + tid * C_stride_n, mask=mask, other=0.0)
```

- `B[b, t, :]` 和 `C[b, t, :]` 是长度为 `d_state` 的向量。
- 注意：题目说明**所有通道共享相同的 B 和 C**。这意味着对于固定的 `(b, t)`，所有 `d` 读取的 `B` 和 `C` 内容相同。但在 1D block 方案中，每个 block 独立加载，依赖 L2 Cache 广播来避免全局内存带宽爆炸。

---

### 10. 计算 B_bar

```python
        B_bar = delta_btd * B_bt
```

- **数学**：$\bar{B}_{b,t,d,n} = \Delta_{b,t,d} \cdot B_{b,t,n}$
- 同样是标量 × 向量广播。

---

### 11. 状态更新：核心递推

```python
        h = A_bar * h + B_bar * u_btd
```

- **数学**：$h_{b,t,d,n} = \bar{A}_{b,t,d,n} \cdot h_{b,t-1,d,n} + \bar{B}_{b,t,d,n} \cdot u_{b,t,d}$
- `A_bar * h`：向量逐元素乘（Hadamard 积）。
- `B_bar * u_btd`：标量 `u_btd` 广播到向量。
- 整个 `h` 更新完全在**寄存器**内完成，无 shared memory 读写延迟。

---

### 12. 计算输出 y

```python
        y_val = tl.sum(tl.where(mask, C_bt * h, 0.0)) + skip_d * u_btd
```

- **数学**：$y_{b,t,d} = \sum_n (C_{b,t,n} \cdot h_{b,t,d,n}) + \text{skip}_d \cdot u_{b,t,d}$
- `C_bt * h`：向量逐元素乘。
- `tl.where(mask, ..., 0.0)`：将 mask 外的 `d_state` 位置清零，避免它们参与累加。
- `tl.sum(...)`：**block-level reduce**。将 `(BLOCK_DSTATE,)` 向量归约为标量。这是 Triton 隐式的 warp shuffle + shared memory reduce，无需手动同步。
- `skip_d * u_btd`：残差连接。

---

### 13. 写回全局内存

```python
        tl.store(y_ptr + b * y_stride_b + t * y_stride_t + d * y_stride_d, y_val)
```

- `y_val` 是标量（`tl.sum` 的 reduce 结果）。
- 所有线程（`tid`）持有相同的 `y_val` 值，同时写入 `y[b, t, d]` 的同一地址。
- **Triton 语义**：虽然多个线程写同地址，但由于值相同，结果正确。Triton 编译器会将其处理为有效的 store 操作。

---

### 14. solve 函数：动态配置与启动

```python
def solve(...):
    BLOCK_DSTATE = 1
    while BLOCK_DSTATE < d_state:
        BLOCK_DSTATE *= 2
    BLOCK_DSTATE = min(BLOCK_DSTATE, 64)
```

- **动态选择 block 大小**：`d_state` 可能为 1~64。选择**最小 2 的幂**（如 16→16，48→64），确保：
  - `tl.arange` 和 `tl.sum` 的 reduce 效率最高。
  - warp 内线程数对齐（32 的倍数时无 diverge）。
- 上限 64，因为题目约束 `d_state ≤ 64`。

```python
    num_warps = max(1, (BLOCK_DSTATE + 31) // 32)
```

- 计算所需 warp 数。例如 `BLOCK_DSTATE=16` 时只需 1 个 warp；`BLOCK_DSTATE=64` 时需 2 个 warps。
- `max(1, ...)` 保证至少 1 个 warp。

```python
    grid = (batch * d_model,)
```

- 1D grid，总 block 数 = `batch × d_model`。这是高并行度的关键。

```python
    selective_scan_fwd_kernel[grid](
        u, delta, A, B, C, skip, y,
        batch, seq_len, d_model, d_state,
        u.stride(0), u.stride(1), u.stride(2),
        ...
        BLOCK_DSTATE=BLOCK_DSTATE,
        num_warps=num_warps,
    )
```

- `kernel[grid]`：Triton 的启动语法。`grid` 是 tuple，定义 block 数量。
- 所有 PyTorch Tensor 直接传入，Triton 自动提取 `.data_ptr()`。
- `BLOCK_DSTATE` 作为 `constexpr` 传入，触发编译期特化。若多次调用遇到不同 `d_state`，Triton 会自动编译多个版本（cache）。

---

### 整体执行流程图

```
Grid: 2048 blocks (batch=4, d_model=512)
├─ Block 0:   (b=0, d=0)  ── 处理 seq_len=4096, d_state=16
├─ Block 1:   (b=0, d=1)  ── 处理 seq_len=4096, d_state=16
...
├─ Block 511: (b=0, d=511)
├─ Block 512: (b=1, d=0)
...
└─ Block 2047:(b=3, d=511)

每个 Block 内部:
  Warp 0 (32 threads):
    tid = [0..15] 有效, [16..31] 被 mask 屏蔽
    循环 t=0..4095:
      加载 u, delta, B, C
      向量计算 A_bar, B_bar
      更新 h (寄存器)
      reduce 得到 y_val
      store y
```

这种设计在 **T4 上最大化 Occupancy**，通过海量轻量 block 掩盖内存延迟，是 Memory-bound 扫描类问题的最优策略。
