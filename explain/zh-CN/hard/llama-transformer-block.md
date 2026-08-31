下面按文件顺序，对 `/mnt/agents/output/triton_llama_block.py` 逐段逐行讲解。

---

## 1. 模块 docstring（第 1–41 行）

```python
# -*- coding: utf-8 -*-
"""
Single Llama-style decoder block (pre-norm, GQA 8Q/2KV, RoPE, SwiGLU) in OpenAI Triton.
...
"""
```

- 第 1 行：声明源码编码为 UTF-8，保险起见（文件里有中文注释也能安全读取）。
- docstring 记录了：目标硬件是 Tesla T4（sm_75，图灵架构，fp32，无 TF32 张量核）；**官方签名是 `solve(x, output, weights, cos, sin, seq_len)`，output 是第 2 个参数**——这正是之前所有 illegal memory access 的根因；以及 weights 一维缓冲区的完整偏移表和 9 个 kernel 的流水线顺序。这些信息是给"下次改代码的人"（包括我自己）看的，防止再犯参数顺序错误。

---

## 2. 导入与常量（第 43–74 行）

```python
import math
from contextlib import nullcontext

import torch
import triton
import triton.language as tl
```

- `math`：算注意力缩放因子 `1/√64`。
- `nullcontext`：一个什么都不做的上下文管理器，用于"在 CUDA 上才进 `torch.cuda.device` 上下文、在 CPU（解释器验证）上就跳过"的统一写法。
- `torch` / `triton` / `triton.language as tl`：PyTorch 负责张量分配与启动，Triton 写 GPU kernel。

```python
D_MODEL = 512
N_Q_HEADS = 8
N_KV_HEADS = 2
HEAD_DIM = 64
FFN_HIDDEN = 1408
QKV_DIM = 768                      # W_Q|W_K|W_V 从偏移 512 起连续存放
GU_DIM = 2 * FFN_HIDDEN            # 2816，W_gate|W_up 从 656384 起连续存放
EPS = 1e-5
WEIGHTS_NUMEL = 2_819_072
```

题目给定的架构超参数。两个"合成"常量是关键优化点：

- `QKV_DIM = 512+128+128 = 768`：因为 W_Q、W_K、W_V 在 weights 缓冲区里**物理上首尾相接**，等价于一个 (768, 512) 的大矩阵，所以 Q/K/V 三个投影可以用**一次 GEMM** 完成。
- `GU_DIM = 2816`：同理，W_gate 和 W_up 连续，拼成 (2816, 512)，gate/up 两个投影也合并成一次 GEMM。

```python
OFF_W1 = 0
OFF_WQKV = 512
OFF_WO = 393728
OFF_W2 = 655872
OFF_WGU = 656384
OFF_WD = 2098176
```

权重在一维缓冲区里的起始元素下标，与官方 `challenge.py` 逐一核对过：

| 偏移 | 内容 | 形状 |
|---|---|---|
| 0 | RMSNorm-1 增益 | (512,) |
| 512 | W_Q | (512,512)，注意 W_K、W_V 紧跟其后 |
| 393728 | W_O | (512,512) |
| 655872 | RMSNorm-2 增益 | (512,) |
| 656384 | W_gate | (1408,512)，W_up 紧跟其后 |
| 2098176 | W_down | (512,1408) |

```python
_SYNC_MAX_T = 256
```

只有 `seq_len ≤ 256`（官方功能测试的最大规模）才在每个阶段后 `synchronize`。这样如果小测试挂了，报错能精确到是哪一步；而 T=2048 的性能测试完全异步发射，不被同步拖慢。

---

## 3. `_stage`：带错误定位的启动器（第 77–83 行）

```python
def _stage(name, sync, launch):
    try:
        launch()
        if sync:
            torch.cuda.synchronize()
    except RuntimeError as e:
        raise RuntimeError(f"[llama_block] fault at stage <{name}>: {e}") from e
```

- `launch` 是一个零参数 lambda，封装一次（或一组）kernel 启动；调用它才真正发射。
- `sync` 为真时立刻 `torch.cuda.synchronize()` 等 GPU 跑完——CUDA 错误是**异步上报**的，不同步的话错误会在后面某个无辜的调用处才冒出来（之前`_silu_mul_kernel` 背锅就是这个机制）。
- 捕获到异常就包一层 `[llama_block] fault at stage <名字>` 再抛出，让报错直接指明故障阶段。`from e` 保留原始异常链。

---

## 4. `_rmsnorm_kernel`（第 86–95 行）

```python
@triton.jit
def _rmsnorm_kernel(x_ptr, w_ptr, w_off, y_ptr,
                    N: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + row * N + offs, mask=mask, other=0.0)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / N + EPS)
    w = tl.load(w_ptr + w_off + offs, mask=mask, other=0.0)
    tl.store(y_ptr + row * N + offs, x * rstd * w, mask=mask)
```

逐行：

- `@triton.jit`：声明这是 Triton JIT kernel，首次以某组 `constexpr` 组合调用时现场编译成 PTX。
- 参数：`x_ptr`/`y_ptr` 是输入/输出矩阵首地址；`w_ptr` 是**整个 weights 缓冲区**首地址；`w_off` 是增益向量在其中的整数偏移（不传切片后的 tensor，只传偏移，避免任何指针误算）；`N=512` 是行宽；三个 `tl.constexpr` 参数在编译期固定，参与常量折叠。
- `row = tl.program_id(0)`：启动网格是 `(T,)`，一个 program（CTA）负责一行。
- `offs = tl.arange(0, BLOCK)`：生成 `[0,1,...,511]` 的向量下标（BLOCK=512，正好一行）。
- `mask = offs < N`：这里 BLOCK==N 恒真，但保留掩码写法是防御性的通用模式。
- `x = tl.load(...)`：一条指令向量加载整行 512 个 fp32；`row * N` 是行首偏移（假设行连续——host 端已 `.contiguous()` 保证）。
- `rstd = 1/sqrt(mean(x²)+EPS)`：RMSNorm 的倒数标准差；`tl.sum(x*x, axis=0)` 把 512 维向量归约成标量。
- `w = tl.load(w_ptr + w_off + offs, ...)`：从大缓冲区偏移处读出增益向量。
- `tl.store(..., x * rstd * w, ...)`：归一化 × 增益，写回对应行。

公式即 `y = x · rsqrt(mean(x²)+ε) · w`，无 bias、无 mean 减法（RMSNorm 与 LayerNorm 的区别）。

---

## 5. `_gemm_nt_kernel`：核心矩阵乘（第 98–114 行）

计算 `C(M,N) = A(M,K) @ B(N,K)ᵀ`，其中 B 是 weights 缓冲区里按"（输出维， 输入维）"行主序存放的投影矩阵——PyTorch 的 `x @ W.T` 正是这个布局。

```python
@triton.jit
def _gemm_nt_kernel(a_ptr, b_ptr, b_off, c_ptr,
                    M, N, K,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
```

- 二维网格：`(ceil(M/64), N/64)`，每个 program 负责输出 C 的一个 64×64 分块。

```python
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
```

- `rm`：本块负责的 64 个行号；`rn`：64 个列号；`rk`：K 方向每次迭代的 32 个下标。

```python
    a_ptrs = a_ptr + rm[:, None] * K + rk[None, :]
    b_ptrs = b_ptr + b_off + rn[:, None] * K + rk[None, :]
```

- 广播构造两个 64×32 的指针矩阵：`a_ptrs[i,j] = A[rm[i], rk[j]]`（A 行主序，行 stride=K）；`b_ptrs[i,j] = weights[b_off + rn[i]*K + rk[j]]`，即 B 的第 rn[i] 行。**注意 B 不转置取数据**，而是按原布局读、后面用 `tl.trans` 在寄存器里转——这就是 "NT"（A 不转置、B 转置）的含义，也避免了任何物化的转置拷贝。

```python
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(rm[:, None] < M) & (rk[None, :] < K - k0), other=0.0)
        b = tl.load(b_ptrs, mask=(rn[:, None] < N) & (rk[None, :] < K - k0), other=0.0)
        acc = tl.dot(a, tl.trans(b), acc, input_precision="ieee")
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K
```

- `acc`：64×64 的 fp32 累加器，常驻寄存器。
- K 循环每次推进 32：`mask` 同时防两类越界——行/列越界（M、N 不被块整除时）和 K 尾块越界；`other=0.0` 让越界位置填 0，**填 0 对乘加结果无影响**，这是掩码安全性的关键。
- `tl.dot(a, tl.trans(b), acc, input_precision="ieee")`：64×32 乘 32×64 的矩阵乘并累加进 acc。**`input_precision="ieee"` 是本题在 T4 上的命门**——T4（sm_75）没有 TF32 张量核，强制 IEEE fp32 让编译器生成 FFMA 指令，避免精度损失或非法指令；第三个位置参数 `acc` 表示累加而非覆盖。
- 指针随循环右移 32 列，进入下一次迭代。

```python
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc, mask=c_mask)
```

- 带掩码写回 C 的 64×64 分块（行 stride=N），越界的行/列不写。

同一个 kernel 被四处复用：QKV（N=768, K=512）、out-proj（N=512, K=512）、gate|up（N=2816, K=512）、down（N=512, K=1408）——只换指针、偏移和形状参数。

---

## 6. `_add_kernel`：残差相加（第 117–124 行）

```python
@triton.jit
def _add_kernel(a_ptr, b_ptr, out_ptr, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < numel
    a = tl.load(a_ptr + offs, mask=m, other=0.0)
    b = tl.load(b_ptr + offs, mask=m, other=0.0)
    tl.store(out_ptr + offs, a + b, mask=m)
```

最简单的逐元素 kernel：把 (T,512) 拉平成 `T*512` 个元素，每个 program 处理 1024 个。`residual = x + proj` 这类操作算力极低、纯粹吃带宽，单独一个 kernel 比融进 GEMM 更稳妥（之前尝试融合进 GEMM 反而增加复杂度，后来拆出来了）。

---

## 7. `_rope_kernel`：旋转位置编码（第 127–143 行）

```python
@triton.jit
def _rope_kernel(x_ptr, cos_ptr, sin_ptr, T, row_stride, base_off,
                 HALF: tl.constexpr, BLOCK_T: tl.constexpr):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
```

- 网格 `(ceil(T/32), n_heads)`：每个 program 处理某个头的 32 个 token。对 Q 启动时 `n_heads=8, base_off=0`；对 K 启动时 `n_heads=2, base_off=512`（K 在 qkv 缓冲区的第 512 列起）。

```python
    rows = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs = tl.arange(0, HALF)
    m = rows < T
    c = tl.load(cos_ptr + rows[:, None] * HALF + offs[None, :], mask=m[:, None], other=0.0)
    s = tl.load(sin_ptr + rows[:, None] * HALF + offs[None, :], mask=m[:, None], other=0.0)
```

- `rows`：32 个 token 序号；`offs`：头内前半维度的 32 个下标（HALF=32）。
- `c`/`s`：读 cos/sin 表，形状 (32 token, 32 频率)，表的行 stride 是 HALF=32。掩码挡掉 T 尾部的越界行。

```python
    base = x_ptr + base_off + pid_h * (2 * HALF) + rows[:, None] * row_stride
    x1 = tl.load(base + offs[None, :], mask=m[:, None], other=0.0)
    x2 = tl.load(base + HALF + offs[None, :], mask=m[:, None], other=0.0)
```

- `base` 指向 qkv 缓冲区中"该 token、该头"的 64 维向量的起点：缓冲区列偏移（Q=0/K=512）+ 头内偏移（每头 64）+ 行偏移（行 stride=768）。
- `x1` = 前 32 维，`x2` = 后 32 维（题目定义的半分切法，不是 Llama 原版的交错切法）。

```python
    tl.store(base + offs[None, :], x1 * c - x2 * s, mask=m[:, None])
    tl.store(base + HALF + offs[None, :], x1 * s + x2 * c, mask=m[:, None])
```

- 原地写回旋转结果：`[x1|x2] → [x1·cos − x2·sin | x1·sin + x2·cos]`。因为 `x1`、`x2` 都已经读进寄存器，原地写不会自己踩自己。

---

## 8. `_attn_kernel`：Flash 式因果 GQA 注意力（第 146–186 行）

这是最复杂的一个 kernel。网格 `(ceil(T/64), 8)`：每个 program 负责**一个 query 头的 64 行 query**，内部对允许看见的 key 分块流式扫描，用 online softmax 避免物化 T×T 的分数矩阵（T=2048 时单头就要 16 MB，8 个头 128 MB，Flash 方式几乎不占额外显存）。

```python
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    kv_h = pid_h // 4                       # GQA: q 头 h 使用 kv 头 h//4
```

- `pid_h // 4` 就是 GQA 的全部实现：8 个 query 头两两共享 2 个 KV 头（头 0–3→KV 0，头 4–7→KV 1）。逻辑上等价于官方参考的 `repeat_interleave(4)`，但**不做物理复制**，读 K/V 时直接用 `kv_h` 寻址，省显存省带宽。

```python
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    m_ok = offs_m < T

    q_base = qkv_ptr + pid_h * D            # Q 列   0:512
    k_base = qkv_ptr + 512 + kv_h * D       # K 列 512:640
    v_base = qkv_ptr + 640 + kv_h * D       # V 列 640:768
```

- `qkv` 缓冲区每行 768 列 = [Q 的 8 头×64 | K 的 2 头×64 | V 的 2 头×64]，三个基址指针直接按列段定位。

```python
    q = tl.load(q_base + offs_m[:, None] * qkv_stride + offs_d[None, :],
                mask=m_ok[:, None], other=0.0)
    m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, D), tl.float32)
```

- `q`：本块的 Q 分块 (64, 64)，整个 key 循环期间常驻寄存器。
- online softmax 的三个状态：`m_i` = 到目前为止见过的最大分数（初始 −∞）；`l_i` = 归一化分母（指数和）；`acc` = 未归一化的加权和 Σ exp(s−m)·V，(64 行， 64 维）。

```python
    hi = pid_m * BLOCK_M + BLOCK_M          # 因果：key 只需扫到本块最后一行
    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_ok = offs_n < T
        k = tl.load(k_base + offs_n[:, None] * qkv_stride + offs_d[None, :],
                    mask=n_ok[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
        s = tl.where((offs_m[:, None] >= offs_n[None, :]) & n_ok[None, :], s, float("-inf"))
```

- **因果剪枝**：第 `pid_m` 块的最后一行是 `pid_m*64+63`，它最多只看到同下标的 key，所以循环上界 `hi` 就是本块末尾——比扫全部 T 列省约一半计算。
- 每次取 32 个 key（BLOCK_N=32）：载入 K 分块 (32,64)，`tl.dot(q, kᵀ)` 得 (64,32) 的原始分数，乘 `1/√64`。
- `tl.where` 施加两层掩码：`offs_m >= offs_n` 是因果（query 行只能看不晚于自己的 key），`n_ok` 防 T 尾部越界；被禁位置写成 −∞，softmax 后权重为 0。

```python
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(v_base + offs_n[:, None] * qkv_stride + offs_d[None, :],
                    mask=n_ok[:, None], other=0.0)
        acc = tl.dot(p, v, acc, input_precision="ieee")
        m_i = m_new
```

online softmax 的标准更新（每行独立）：

1. `m_new`：新旧最大值取大（`tl.max(s, 1)` 对每行 32 个分数归约）。
2. `alpha = exp(m_old − m_new)`：旧累加值的"贬值因子"——最大值变大后，之前按旧最大值缩放的和要重新对齐。
3. `p = exp(s − m_new)`：本块注意力权重（未归一化）；被掩的 −∞ 位置 `exp` 后得 0。减最大值是数值稳定的关键，防止大分数 `exp` 溢出。
4. `l_i = l_i·alpha + Σp`：分母同步 rescale 再累加。
5. `acc = acc·alpha`：分子也 rescale。
6. 载入对应的 V 分块，`acc += p @ V`（64×32 乘 32×64）。
7. 更新 `m_i` 进入下一块。

> 全零输入（官方 `zero_x` 测试）时：q=0 → s 全 0 → m_new=0、p=1 → l_i = 可见 key 数、acc=0 → 输出 0，无 NaN。首块时 `m_i=−∞`，`alpha=exp(−∞−有限值)=0`，`0*0+Σp` 也安全。

```python
    acc = acc / l_i[:, None]
    tl.store(out_ptr + pid_h * D + offs_m[:, None] * 512 + offs_d[None, :],
             acc, mask=m_ok[:, None])
```

- 循环结束后除以分母完成 softmax 归一化，把 (64,64) 结果写到输出矩阵的"该头、该批行"位置（输出行 stride=512，头内列偏移 `pid_h*64`）。

共享内存账：Q 分块 16 KB + K/V 分块各 8 KB = 32 KB，低于 T4 的 48 KB 默认上限（已用离线编译核实）。

---

## 9. `_silu_mul_kernel`：SwiGLU 激活（第 189–198 行）

```python
@triton.jit
def _silu_mul_kernel(gu_ptr, out_ptr, N: tl.constexpr, gu_stride,
                     BLOCK_N: tl.constexpr):
    pid_t = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N
    g = tl.load(gu_ptr + pid_t * gu_stride + offs, mask=mask, other=0.0)
    u = tl.load(gu_ptr + pid_t * gu_stride + N + offs, mask=mask, other=0.0)
    tl.store(out_ptr + pid_t * N + offs, g * tl.sigmoid(g) * u, mask=mask)
```

- 上游 gate|up 融合 GEMM 产出的 `gu` 缓冲区每行 2816 列：**前 1408 列是 gate，后 1408 列是 up**。所以 `u` 的地址就是 `g` 的地址 `+ N`（+1408）——一次读两个位置，不用分开启动两个 GEMM。
- 网格 `(T, ceil(1408/512)=3)`，每个 program 处理一行的 512 个通道（第三块掩码掉 1408 之后的 128 个）。
- `g * tl.sigmoid(g) * u` 即 `silu(gate) · up`——SwiGLU 的定义。

---

## 10. `solve()`：host 端总装（第 201–293 行）

```python
def solve(x, output, weights, cos, sin, seq_len):
    """LeetGPU 签名：x, output, weights, cos, sin, seq_len（output 在第 2 位！）"""
    T = int(seq_len)
```

- **这就是本次修复的核心**。官方 harness 按 `(x, output, weights, cos, sin, seq_len)` 传参；之前的版本假设 weights 在第 2 位，把 (T,512) 的输出张量当 281 万元素的权重读，越界数十万元素 → illegal memory access。

```python
    f32 = torch.float32
    dev = x.device if x.device.type == "cuda" else (
        torch.device("cuda", 0) if torch.cuda.is_available() else x.device)
    dev_ctx = torch.cuda.device(dev) if dev.type == "cuda" else nullcontext()

    with dev_ctx:
```

- 确定运行设备：正常走 `x.device`；万一传入 CPU 张量（比如本地解释器验证）且有 GPU，则挪到 cuda:0。`torch.cuda.device(dev)` 上下文把后续所有 CUDA 操作钉在该设备上；CPU 路径用 `nullcontext()` 空上下文，同一份代码两种环境都能跑。

```python
        sync = (dev.type == "cuda" and T <= _SYNC_MAX_T)
```

- 小规模 + CUDA 才开启逐阶段同步（错误定位模式）；T=2048 基准测试关闭。

```python
        x = x.to(device=dev, dtype=f32).contiguous()
        w = weights.to(device=dev, dtype=f32).contiguous().view(-1)
        cos = cos.to(device=dev, dtype=f32).contiguous()
        sin = sin.to(device=dev, dtype=f32).contiguous()
```

- 四个输入统一成"本设备、fp32、行主序连续"。**在 harness 上这些全是零开销的 no-op**（`to` 在 dtype/device 相同、`contiguous` 在连续时直接返回原 tensor），只是防御性规范化。`w.view(-1)` 拉平成一维，因为 kernel 全部用"基址 + 整数偏移"寻址。

```python
        if w.numel() < WEIGHTS_NUMEL:
            raise RuntimeError(
                f"[llama_block] weights buffer too short: numel={w.numel()}, ...")
        if x.numel() < T * D_MODEL or cos.numel() < T * (HEAD_DIM // 2):
            raise RuntimeError(...)
```

- 两道廉价的前置校验。第一条就是当初破案的那条——如果参数顺序再搞错，会立刻得到一条明说"检查 solve() 参数顺序"的报错，而不是玄学 illegal memory access。

```python
        out_view = output.to(device=dev, dtype=f32)
        if out_view.is_contiguous() and out_view.shape == (T, D_MODEL):
            out_buf = out_view
            need_copy = out_view.data_ptr() != output.data_ptr()
        else:
            out_buf = torch.empty((T, D_MODEL), device=dev, dtype=f32)
            need_copy = True
```

- 结果写回策略：harness 上 `output` 本来就是 fp32 连续 CUDA 张量 → `out_buf` **直接就是调用方的 output**，最后一个 kernel 直接写进去，零拷贝（`data_ptr` 相同 → `need_copy=False`）。只有在异常情形（dtype/设备被转换过、非连续）才另开缓冲区、最后 `copy_` 回去。

```python
        norm1 = torch.empty((T, D_MODEL), device=dev, dtype=f32)
        qkv = torch.empty((T, QKV_DIM), device=dev, dtype=f32)
        attn = torch.empty((T, D_MODEL), device=dev, dtype=f32)
        proj = torch.empty((T, D_MODEL), device=dev, dtype=f32)
        x1 = torch.empty((T, D_MODEL), device=dev, dtype=f32)
        norm2 = torch.empty((T, D_MODEL), device=dev, dtype=f32)
        gu = torch.empty((T, GU_DIM), device=dev, dtype=f32)
        ffn = torch.empty((T, FFN_HIDDEN), device=dev, dtype=f32)
        ffn_out = torch.empty((T, D_MODEL), device=dev, dtype=f32)
```

- 九个中间缓冲：归一化 1 结果、融合 QKV、注意力输出、out-proj 结果、残差 1 后的隐状态 `x1`、归一化 2 结果、融合 gate|up、SwiGLU 输出、down-proj 结果。`torch.empty` 走 PyTorch 缓存分配器，每次调用开销仅微秒级。

```python
        BM, BN, BK, NW = 64, 64, 32, 4
        add_grid = (triton.cdiv(T * D_MODEL, 1024),)
```

- GEMM 分块参数与 add kernel 的网格（`cdiv` = 向上取整除）。

然后是 9 个阶段（每个都经 `_stage` 包装）：

```python
        # 1) RMSNorm 1
        _stage("1-rmsnorm1", sync, lambda: _rmsnorm_kernel[(T,)](
            x, w, OFF_W1, norm1, N=D_MODEL, EPS=EPS, BLOCK=512, num_warps=4))
```

- 网格 `(T,)` 一行一个 program；从 weights 偏移 0 读增益；输出 `norm1 = RMSNorm(x)`。

```python
        # 2) QKV 投影  (T,512) @ (768,512)^T -> (T,768)
        _stage("2-qkv-gemm", sync, lambda: _gemm_nt_kernel[(triton.cdiv(T, BM), QKV_DIM // BN)](
            norm1, w, OFF_WQKV, qkv, T, QKV_DIM, D_MODEL, ...))
```

- 网格 `(ceil(T/64), 768/64=12)`；B 指针 = weights + 偏移 512，一次 GEMM 同时算出 Q、K、V 写进 `qkv`。

```python
        # 3) RoPE 作用于 Q（8 头，列偏移 0）和 K（2 头，列偏移 512），原地
        def rope():
            _rope_kernel[(triton.cdiv(T, 32), N_Q_HEADS)](
                qkv, cos, sin, T, QKV_DIM, 0, ...)
            _rope_kernel[(triton.cdiv(T, 32), N_KV_HEADS)](
                qkv, cos, sin, T, QKV_DIM, D_MODEL, ...)
        _stage("3-rope", sync, rope)
```

- 注意 RoPE **只旋 Q 和 K，不碰 V**；两次启动分别覆盖 qkv 缓冲区的 Q 段（列 0 起 8 头）和 K 段（列 512 起 2 头）。`rope` 是普通函数（内含两次启动），整体作为一个阶段。

```python
        # 4) 因果 GQA 注意力 -> (T,512)
        _stage("4-attention", sync, lambda: _attn_kernel[(triton.cdiv(T, 64), N_Q_HEADS)](
            qkv, attn, T, QKV_DIM, 1.0 / math.sqrt(HEAD_DIM),
            BLOCK_M=64, BLOCK_N=32, D=HEAD_DIM, num_warps=4, num_stages=1))
```

- 缩放因子 `1/√64`；`num_stages=1` 关闭软件流水以压低共享内存（T4 上求稳）。

```python
        # 5) 输出投影 + 残差: x1 = x + attn @ W_O^T
        _stage("5-outproj-gemm", ...)
        _stage("5b-residual-add", sync, lambda: _add_kernel[add_grid](
            x, proj, x1, T * D_MODEL, BLOCK=1024, num_warps=4))
```

- 先 GEMM 出 `proj = attn @ W_Oᵀ`（B 偏移 393728），再逐元素 `x1 = x + proj`——完成 `x' = x + Attn(RMSNorm1(x))`。

```python
        # 6) RMSNorm 2        -> norm2 = RMSNorm2(x1)
        # 7) 融合 gate|up GEMM -> gu (T,2816)，B 偏移 656384
        # 8) SwiGLU            -> ffn = silu(gate)*up (T,1408)
        # 9) down GEMM + 残差  -> out_buf = x1 + ffn @ W_down^T（B 偏移 2098176，K=1408）
```

- 6–9 与前面完全同构，完成 `output = x' + FFN(RMSNorm2(x'))`。注意第 9 步的 K=1408（down 投影的输入维），且最后一个 add 的输出是 `out_buf`——harness 情形下它就是调用方的 `output` 本体。

```python
        if need_copy:
            output.copy_(out_buf)
```

- 仅当结果落在临时缓冲区时才拷贝回 `output`（harness 上不触发）。

---

## 11. `forward()` 便捷封装（第 296–299 行）

```python
def forward(x, weights, cos, sin):
    out = torch.empty_like(x)
    solve(x, out, weights, cos, sin, x.shape[0])
    return out
```

- 按官方参数顺序调用 `solve` 的小帮手，方便本地自测/计时。

---

## 12. `__main__` 自测（第 302 行起）

```python
    def make_case(T, zero_x=False):
        ...torch.cat([rms1, W_Q, W_K, W_V, W_O, rms2, W_gate, W_up, W_down])
        freqs = 1.0 / (10000.0 ** (torch.arange(0, HEAD_DIM, 2, device=dev).float() / HEAD_DIM))
```

- **逐字段复刻官方 `challenge.py` 的数据生成**：rms 增益 ~ U(0.8,1.2)，矩阵 ~ N(0, 0.02²)，x ~ U(−1,1)，RoPE 表用 θ=10000、`freqs = θ^(−2i/64)`（i=0..31）。

```python
    def reference(x, weights, cos, sin):
        ...rms_norm / apply_rope / repeat_interleave(4) / causal softmax / SwiGLU...
```

- 纯 PyTorch 参考实现，数学上与官方 `reference_impl` 一致（einsum 形式和官方 matmul 形式等价）。

```python
    cases = [(1, False), (4, True), (2, False), (4, False), (16, False),
             (64, False), (30, False), (100, False), (128, False), (256, False)]
    for T, zx in cases:
        ...assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)
```
