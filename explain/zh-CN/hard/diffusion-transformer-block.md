---

## 0. 文件头与常量

```python
import torch
import triton
import triton.language as tl
```
`torch` 只用来分配显存 buffer;`triton.language as tl` 是写 kernel 用的 DSL。

```python
OFF_W_ADA = 0          # W_ada (3072, 512)
OFF_B_ADA = 1572864    # b_ada (3072,)
OFF_W_QKV = 1575936    # W_qkv (1536, 512)
...
```
打包权重 buffer 里每个参数的**起始下标（单位：float 个数）**，直接照题目给的偏移表抄的。后面用 `weights[OFF_W_QKV:]` 这种切片拿到一个 `data_ptr` 恰好落在该参数起始处的视图，零拷贝。

```python
D_MODEL, MOD_DIM, QKV_DIM, MLP_DIM = 512, 3072, 1536, 2048
N_HEADS, HEAD_DIM = 8, 64
SHIFT_MSA, GATE_MSA = 0, 1024
SHIFT_MLP, GATE_MLP = 1536, 2560
```
调制向量是 `(B, 3072)`，内部布局为 `[shift_msa | scale_msa | gate_msa | shift_mlp | scale_mlp | gate_mlp]`，每段 512。所以 `scale_msa` 的偏移 = `SHIFT_MSA + 512`，以此类推。kernel 里用这些常量定位各段。

---

## 1. `_gelu_tanh` — GELU 的 tanh 近似

```python
@triton.jit
def _gelu_tanh(x):
    z = 0.7978845608028654 * (x + 0.044715 * x * x * x)
```
`@triton.jit` 表示这是编译到 GPU 的函数（可作为子函数被其他 kernel 内联）。`0.7978845608028654 = √(2/π)`，这一行算 tanh 的内部参数 `z = √(2/π)(x + 0.044715x³)`。

```python
    e = tl.exp(-2.0 * tl.abs(z))
    t = (1.0 - e) / (1.0 + e)
    t = tl.where(z >= 0.0, t, -t)
```
恒等式 `tanh(z) = sign(z) · (1 − e^{−2|z|}) / (1 + e^{−2|z|})`。不用 `tl.math.tanh` 是为了跨 Triton 版本稳定；用 `|z|` 是为了让指数永远是 `e^{负数} ≤ 1`，**数值上不会溢出**（直接 `e^{2z}` 在 `z > 44` 时变 inf,`inf/inf = NaN`)。

```python
    return 0.5 * x * (1.0 + t)
```
拼回 `0.5x(1 + tanh(...))`，与 PyTorch 的 `gelu(approximate='tanh')` 误差 ~5e-7。

---

## 2. `_mod_gemm_kernel` — 调制向量 GEMM(SiLU 融合在输入端）

功能：`mod(B,3072) = SiLU(c)(B,512) @ W_ada(3072,512)ᵀ + b_ada`。

```python
def _mod_gemm_kernel(c_ptr, w_ptr, b_ptr, o_ptr, B,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    K: tl.constexpr = 512
```
指针 + 运行时维度 `B` + 编译期常量块尺寸（`tl.constexpr`，编译时定死，参与循环展开）。`K=512` 直接写死。

```python
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
```
二维 grid：这个 program 负责输出 tile 的第 `pid_m` 行块、第 `pid_n` 列块。`rm/rn/rk` 是本 tile 覆盖的**全局行号/列号/K 下标向量**（如 `rm = [64,65,...,127]`)。

```python
    a_ptrs = c_ptr + rm[:, None] * K + rk[None, :]
    w_ptrs = w_ptr + rn[None, :] * K + rk[:, None]
```
构造二维指针矩阵。`rm[:,None]*K` 是 `(BLOCK_M,1)` 的行起始偏移，加 `rk[None,:]` 广播成 `(BLOCK_M,BLOCK_K)`——A tile 每个元素的地址。W 按 `(out,in)` 行主序存储，用 `rn`（输出维）乘行 stride `K`、`rk` 当列，相当于**边读边转置**，直接得到 `Wᵀ` 的 tile `(BLOCK_K, BLOCK_N)`。

```python
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=rm[:, None] < B, other=0.0)
        a = a / (1.0 + tl.exp(-a))
        w = tl.load(w_ptrs, mask=rn[None, :] < 3072, other=0.0)
        acc = tl.dot(a, w, acc, input_precision="ieee")
        a_ptrs += BLOCK_K
        w_ptrs += BLOCK_K
```
- 累加器初始化为 0。
- 沿 K 维循环：装入 A tile，**立刻在寄存器里算 SiLU**(`x·σ(x)`，写成 `x/(1+e^{-x})`)——这就是融合点，省掉单独逐元素 kernel 和中间 buffer。
- `mask=rm<B`：因为 `B ≤ 16` 而 `BLOCK_M=16`(`tl.dot` 最小维度要求），多出的行是"假行"，load 填 0、最后不存。
- `tl.dot(a, w, acc)` 即 `acc += a @ w`;`input_precision="ieee"` 强制 fp32 IEEE 乘加——**T4 没有 TF32 单元**，默认 tf32 在 Turing 上会出问题，这行是 T4 兼容的关键。
- 指针整体前进一个 `BLOCK_K`，进入下一 K 分块（`K=512` 能被 64 整除，无需 K 方向 mask)。

```python
    acc = acc + tl.load(b_ptr + rn, mask=rn < 3072, other=0.0)[None, :]
    tl.store(o_ptr + rm[:, None] * 3072 + rn[None, :], acc,
             mask=(rm[:, None] < B) & (rn[None, :] < 3072))
```
加 bias（广播到每行），写回，mask 掉假行。

---

## 3. `_gemm_ln_kernel` — LayerNorm + adaLN 调制融合进 GEMM（本版核心）

功能：`C(M,N) = act( (LN(A)·(1+scale[batch]) + shift[batch]) @ W(N,512)ᵀ + bias )`。它替代了"先跑 LN kernel 写 `h`、再跑 GEMM 读 `h`"两步。

```python
    K: tl.constexpr = 512
    ...
    row_mask = rm < M
    mod_row = mod_ptr + (rm // T) * 3072 + SHIFT_OFF
```
`rm // T` 把全局行号（= `b·T + t`）换算成 **batch 下标**——即使一个 tile 跨 batch 边界，逐行 gather 也对。`mod_row` 是每行调制向量的起始地址（指向 `shift` 段）。越界假行的地址会被后面的 load mask 挡住，不会真的访存。

**第一遍：求均值**
```python
    s = tl.zeros((BLOCK_M,), dtype=tl.float32)
    a_ptrs = a_ptr + rm[:, None] * K + rk[None, :]
    for _ in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=row_mask[:, None], other=0.0)
        s += tl.sum(a, 1)
        a_ptrs += BLOCK_K
    mean = s / K
```
沿 K 分块扫完这一行块的全部 512 个特征，逐行累加得 `mean`。LN 必须看到整行才能归一化，而 GEMM 的 K 循环是逐块的，所以要在 dot 之前**预扫两遍**。

**第二遍：求（中心化）方差**
```python
    v = tl.zeros((BLOCK_M,), dtype=tl.float32)
    a_ptrs = a_ptr + rm[:, None] * K + rk[None, :]   # 指针复位
    for ...:
        d = tl.where(row_mask[:, None], a - mean[:, None], 0.0)
        v += tl.sum(d * d, 1)
    rstd = 1.0 / tl.sqrt(v / K + EPS)
```
用 `(x−μ)²` 而不是 `E[x²]−μ²`，避免大数相消（数值更稳）；`tl.where` 把假行的贡献清零。`EPS=1e-6` 按题目要求。得逐行 `rstd = 1/√(σ²+ε)`。

**第三遍：边归一化边调制边 dot**
```python
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    a_ptrs = ...; w_ptrs = ...                       # 双双复位
    for k0 in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=row_mask[:, None], other=0.0)
        sh = tl.load(mod_row[:, None] + (k0 + rk)[None, :], ...)
        sc = tl.load(mod_row[:, None] + K + (k0 + rk)[None, :], ...)
        a = (a - mean[:, None]) * rstd[:, None] * (1.0 + sc) + sh
        w = tl.load(w_ptrs, mask=rn[None, :] < N, other=0.0)
        acc = tl.dot(a, w, acc, input_precision="ieee")
        a_ptrs += BLOCK_K; w_ptrs += BLOCK_K
```
- `sh/sc` 是当前 K 分块对应的 shift/scale 片段，逐行 gather(`(BLOCK_M,BLOCK_K)` 小gather,mod 总共才几十 KB，全在 L1/L2 里）。`sc` 的 `+ K` 即 `+512`：scale 段紧跟 shift 段。
- `a = (a−μ)·rstd·(1+scale) + shift` —— 注意是 **`1 + scale`**（题目明确要求），这就是 adaLN 调制，LN 本身无可学仿射参数。
- 调制完的 tile 直接进 `tl.dot`,**`h` 从不落显存**——这就是省掉一个 kernel + 32MB 流量的地方。
- A 行块只有 128KB，三遍扫描中后两遍基本全命中 L2，代价很小。

```python
    acc = acc + tl.load(b_ptr + rn, mask=rn < N, other=0.0)[None, :]
    if ACT == 1:
        acc = _gelu_tanh(acc)
    tl.store(...)
```
bias;`ACT` 是编译期开关，用于 FC1 时（`ACT=1`）把 GELU 融在出口；用于 QKV 时（`ACT=0`）直接写回。`if` 在编译期就被裁剪，无运行时分支开销。

> 冗余说明：N 方向有 `N/64` 个 program 共用同一行块，LN 会被重算这么多次——拿廉价计算换显存流量，划算。

---

## 4. `_attn_kernel` — flash 式在线 softmax 注意力

功能：每个 program 处理**一个 (batch, head) 的一段 query 行**，双向（无 causal mask),scale=1/√64=0.125。

```python
    pid_m, pid_bh = tl.program_id(0), tl.program_id(1)
    b, h = pid_bh // 8, pid_bh % 8
    base = qkv_ptr + b * T * 1536 + h * 64
```
grid 第二维摊平 `(B×8)` 个头。`base` 指向该 (b,h) 在打包 qkv 里的 Q 切片起点——回忆 qkv 每行 1536 维 = `[Q(512) | K(512) | V(512)]`，第 h 头的 Q 在 `[h·64, h·64+64)`。

```python
    q = tl.load(base + rm[:, None] * 1536 + rd[None, :], mask=m_mask[:, None], other=0.0)
    q = q * 0.125
```
装入本 tile 的 Q `(BLOCK_M, 64)`，行 stride 是 1536。**提前把 scale 乘进 Q**，和分数矩阵算完再乘数学等价，省得每轮乘整个 S tile。

```python
    m_i = tl.full((BLOCK_M,), float("-inf"), ...)   # 逐行 running max
    l_i = tl.zeros((BLOCK_M,), ...)                 # 逐行 running 分母
    acc = tl.zeros((BLOCK_M, D), ...)               # 逐行 running 加权和
    for n0 in range(0, T, BLOCK_N):
        k = tl.load(base + 512 + rn[:, None]*1536 + rd[None, :], ...)
        s = tl.dot(q, tl.trans(k), input_precision="ieee")
        s = tl.where(n_mask[None, :], s, float("-inf"))
```
扫描 K/V(`base+512` 是 K 切片）。`s = Q·Kᵀ` 得 `(BLOCK_M, BLOCK_N)` 分数；T 不是块整数倍时，把越界列置 `-inf`，softmax 后权重自然为 0。**不加 causal mask**（题目要求双向）。

```python
        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.exp(s - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(base + 1024 + ...)
        acc = acc * alpha[:, None] + tl.dot(p, v, input_precision="ieee")
        m_i = m_new
```
**在线 softmax(Milakov–Gimelshein 技巧）**：每来一块新分数，用新老 max 的较大者 `m_new` 为基准算 `p = e^{s−m_new}`（防溢出）；历史累积量乘以修正因子 `alpha = e^{m_old−m_new}` 重新对齐基准，再累加。这样**永不物化 `T×T` 分数矩阵**(T=4096 时单头就要 64MB)。首轮 `m_i=-inf` → `alpha=0`，历史清零，数学自洽。

```python
    acc = acc / l_i[:, None]
    tl.store(out_ptr + (b * T + rm[:, None]) * 512 + h * 64 + rd[None, :], acc, mask=...)
```
最后统一除以分母得到真正的 softmax 加权输出，写回 `(B,T,512)` 布局中该头的 64 列。

---

## 5. `_gemm_gate_res_kernel` — GEMM + 门控 + 残差（epilogue 融合）

功能：`out = res + gate[batch(row)] ⊙ (A @ W(N,K)ᵀ + bias)`。用于两处：输出投影（`A=attn, res=x, gate=g_msa`）和 FC2(`A=ff, res=x1, gate=g_mlp`)。

主体循环与普通 GEMM 完全相同（不再赘述）。差异在收尾：

```python
    msk = (rm[:, None] < M) & (rn[None, :] < N)
    gate = tl.load(mod_ptr + (rm // T)[:, None] * 3072 + GATE_OFF + rn[None, :], mask=msk, other=0.0)
    res = tl.load(res_ptr + rm[:, None] * N + rn[None, :], mask=msk, other=0.0)
    tl.store(out_ptr + ..., res + gate * acc, mask=msk)
```
- `gate` 按行所属 batch 从 mod 的 `GATE_OFF` 段（1024=g_msa 或 2560=g_mlp）逐行 gather，列方向与输出 tile 对齐——正好是 adaLN-Zero 的逐样本门控。
- 读残差、`res + gate⊙acc`、写回，全在同一个 epilogue 里完成，省掉了独立的"乘门控+加残差"逐元素 kernel。
- K 维为 512（投影）或 2048(FC2)，都能被 `BLOCK_K=32` 整除，无需 K mask。

---

## 6. `solve` — 驱动（6 次 launch)

```python
    x, c, weights = x.contiguous(), c.contiguous(), weights.contiguous()
```
防御性保证内存连续（已是连续则零开销），kernel 的指针算术依赖行主序连续。

```python
    mod  = torch.empty((B, MOD_DIM), ...)
    qkv  = torch.empty((M, QKV_DIM), ...)
    attn = torch.empty((M, D_MODEL), ...)
    x1   = torch.empty((M, D_MODEL), ...)
    ff   = torch.empty((M, MLP_DIM), ...)
```
中间 buffer。注意融合版已经没有 `h` 和 `tmp_c` 了。`(M, …)` 把 `(B,T)` 摊平成逐 token 行。

```python
    _mod_gemm_kernel[(triton.cdiv(B, 16), triton.cdiv(MOD_DIM, 64))](..., BLOCK_M=16, ...)
```
**①** 调制 GEMM。grid 按输出 `(B,3072)` 切块；`BLOCK_M=16` 是 `tl.dot` 允许的最小 M(B<16 的部分靠 mask)。

```python
    _gemm_ln_kernel[(triton.cdiv(M, 64), triton.cdiv(QKV_DIM, 64))](
        x, ..., qkv, mod, M, QKV_DIM, T, SHIFT_OFF=SHIFT_MSA, ..., ACT=0, ...)
```
**②** 直接吃 `x`，内部完成 LN+msa 调制，输出 qkv。`SHIFT_OFF=0` 指向 shift_msa/scale_msa 段。

```python
    _attn_kernel[(triton.cdiv(T, 64), B * N_HEADS)](qkv, attn, T, ..., num_stages=1)
```
**③** 注意力。`num_stages=1` 是有意的：循环里 K、V 两个 16KB tile，若 `num_stages=2` 软件流水双缓冲会逼近 T4 的 64KB 共享内存上限，保守取 1。

```python
    _gemm_gate_res_kernel[...](attn, W_o, b_o, x, mod, x1, ..., GATE_OFF=GATE_MSA, ...)
```
**④** `x1 = x + g_msa ⊙ (attn·W_oᵀ + b_o)`，残差是原始 `x`。

```python
    _gemm_ln_kernel[...](x1, W_fc1, b_fc1, ff, mod, ..., SHIFT_OFF=SHIFT_MLP, ACT=1, ...)
```
**⑤** 对 `x1` 做 LN+mlp 调制 → FC1 → GELU(tanh) 出口，`ACT=1`。

```python
    _gemm_gate_res_kernel[...](ff, W_fc2, b_fc2, x1, mod, output, ..., GATE_OFF=GATE_MLP, ...)
```
**⑥** `output = x1 + g_mlp ⊙ (ff·W_fc2ᵀ + b_fc2)`，残差是 `x1`，结果直接写进题目给的 `output`。

```python
    return output
```

---

## 整体记忆图

```
c ──①(SiLU+GEMM)──────────────► mod(B,3072)
                                 │ sh/sc/g 广播到每个 token
x ──②(LN+mod+GEMM)► qkv ──③(flash attn)► attn ──④(GEMM+gate+res x)► x1
x1 ─⑤(LN+mod+GEMM+GELU)► ff ──⑥(GEMM+gate+res x1)► output
```

两条 GEMM 主链上，②⑤ 把"前"操作（LN/调制）融进 prologue,④⑥ 把"后"操作（门控/残差）融进 epilogue,③ 用在线 softmax 避免物化分数矩阵——所有中间结果只落显存一次，这就是这版融合设计的全貌。

有哪一段想再往深挖（比如在线 softmax 的数学推导、Triton 的 shared memory/寄存器分配、或者 grid 占用率估算），可以单独展开。