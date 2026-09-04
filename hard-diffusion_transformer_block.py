# Single DiT (Diffusion Transformer) block in OpenAI Triton, fp32 — fused version.
# Tesla T4 (sm_75, Turing) compatible:
#   * all tl.dot calls use input_precision="ieee" (Turing has no TF32 units)
#   * shared-memory usage per kernel <= 32 KB (T4 has 64 KB/SM)
#
# Fused pipeline (6 kernel launches, down from 9):
#   1) mod = SiLU(c) @ W_ada^T + b_ada          SiLU fused into GEMM prologue
#   2) qkv = (LN(x)*(1+sc1)+sh1) @ W_qkv^T + b_qkv    LN+modulate fused into GEMM prologue
#   3) attn = softmax(QK^T/sqrt(64)) V          flash-style online softmax, no causal mask
#   4) x1  = x + g_msa * (attn @ W_o^T + b_o)   gate+residual fused into GEMM epilogue
#   5) ff  = GELU_tanh((LN(x1)*(1+sc2)+sh2) @ W_fc1^T + b_fc1)   LN fused, GELU epilogue
#   6) out = x1 + g_mlp * (ff @ W_fc2^T + b_fc2)        gate+residual fused into epilogue
#
# Deliberately NOT fused (correctness/perf on T4):
#   * attention + output projection: needs cross-head fp32 atomic_add (nondeterministic)
#   * FC1 + FC2 in one kernel: register-resident hidden tile spills to local memory on T4

import torch
import triton
import triton.language as tl

# ---------------- packed weight buffer offsets (in floats) ----------------
OFF_W_ADA = 0          # W_ada (3072, 512)
OFF_B_ADA = 1572864    # b_ada (3072,)
OFF_W_QKV = 1575936    # W_qkv (1536, 512)
OFF_B_QKV = 2362368    # b_qkv (1536,)
OFF_W_O   = 2363904    # W_o   (512, 512)
OFF_B_O   = 2626048    # b_o   (512,)
OFF_W_FC1 = 2626560    # W_fc1 (2048, 512)
OFF_B_FC1 = 3675136    # b_fc1 (2048,)
OFF_W_FC2 = 3677184    # W_fc2 (512, 2048)
OFF_B_FC2 = 4725760    # b_fc2 (512,)

D_MODEL, MOD_DIM, QKV_DIM, MLP_DIM = 512, 3072, 1536, 2048
N_HEADS, HEAD_DIM = 8, 64
# mod layout: [shift_msa | scale_msa | gate_msa | shift_mlp | scale_mlp | gate_mlp]
SHIFT_MSA, GATE_MSA = 0, 1024        # scale_msa = SHIFT_MSA + 512
SHIFT_MLP, GATE_MLP = 1536, 2560     # scale_mlp = SHIFT_MLP + 512


# ------------------------------- kernels -----------------------------------

@triton.jit
def _gelu_tanh(x):
    # 0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715 x^3))), tanh via exp (stable for large |z|)
    z = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    e = tl.exp(-2.0 * tl.abs(z))
    t = (1.0 - e) / (1.0 + e)
    t = tl.where(z >= 0.0, t, -t)
    return 0.5 * x * (1.0 + t)


@triton.jit
def _mod_gemm_kernel(c_ptr, w_ptr, b_ptr, o_ptr, B,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # out(B,3072) = SiLU(c)(B,512) @ W_ada(3072,512)^T + b_ada ; SiLU fused on the A-load
    K: tl.constexpr = 512
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    a_ptrs = c_ptr + rm[:, None] * K + rk[None, :]
    w_ptrs = w_ptr + rn[None, :] * K + rk[:, None]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=rm[:, None] < B, other=0.0)
        a = a / (1.0 + tl.exp(-a))                        # SiLU, fused
        w = tl.load(w_ptrs, mask=rn[None, :] < 3072, other=0.0)
        acc = tl.dot(a, w, acc, input_precision="ieee")
        a_ptrs += BLOCK_K
        w_ptrs += BLOCK_K
    acc = acc + tl.load(b_ptr + rn, mask=rn < 3072, other=0.0)[None, :]
    tl.store(o_ptr + rm[:, None] * 3072 + rn[None, :], acc,
             mask=(rm[:, None] < B) & (rn[None, :] < 3072))


@triton.jit
def _gemm_ln_kernel(a_ptr, w_ptr, b_ptr, c_ptr, mod_ptr, M, N, T,
                    SHIFT_OFF: tl.constexpr, EPS: tl.constexpr,
                    ACT: tl.constexpr,                    # 0 = none, 1 = GELU(tanh)
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # C(M,N) = act( (LN(A)*(1+scale[batch])+shift[batch]) @ W(N,512)^T + bias )
    # LayerNorm (no affine, over the K=512 feature dim) + adaLN modulate fused as a
    # 3-pass prologue over K: mean -> centered variance -> normalize+modulate+dot.
    # The A row-block (<=128 KB) stays in L2 across the passes; redundant LN math
    # across the N-tile programs is negligible vs the GEMM itself.
    K: tl.constexpr = 512
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    row_mask = rm < M
    mod_row = mod_ptr + (rm // T) * 3072 + SHIFT_OFF       # per-row modulation base

    s = tl.zeros((BLOCK_M,), dtype=tl.float32)             # pass 1: mean
    a_ptrs = a_ptr + rm[:, None] * K + rk[None, :]
    for _ in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=row_mask[:, None], other=0.0)
        s += tl.sum(a, 1)
        a_ptrs += BLOCK_K
    mean = s / K

    v = tl.zeros((BLOCK_M,), dtype=tl.float32)             # pass 2: centered variance
    a_ptrs = a_ptr + rm[:, None] * K + rk[None, :]
    for _ in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=row_mask[:, None], other=0.0)
        d = tl.where(row_mask[:, None], a - mean[:, None], 0.0)
        v += tl.sum(d * d, 1)
        a_ptrs += BLOCK_K
    rstd = 1.0 / tl.sqrt(v / K + EPS)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)   # pass 3: fused dot
    a_ptrs = a_ptr + rm[:, None] * K + rk[None, :]
    w_ptrs = w_ptr + rn[None, :] * K + rk[:, None]
    for k0 in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=row_mask[:, None], other=0.0)
        sh = tl.load(mod_row[:, None] + (k0 + rk)[None, :], mask=row_mask[:, None], other=0.0)
        sc = tl.load(mod_row[:, None] + K + (k0 + rk)[None, :], mask=row_mask[:, None], other=0.0)
        a = (a - mean[:, None]) * rstd[:, None] * (1.0 + sc) + sh
        w = tl.load(w_ptrs, mask=rn[None, :] < N, other=0.0)
        acc = tl.dot(a, w, acc, input_precision="ieee")    # fp32 FMA, T4-safe
        a_ptrs += BLOCK_K
        w_ptrs += BLOCK_K
    acc = acc + tl.load(b_ptr + rn, mask=rn < N, other=0.0)[None, :]
    if ACT == 1:
        acc = _gelu_tanh(acc)
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc,
             mask=row_mask[:, None] & (rn[None, :] < N))


@triton.jit
def _attn_kernel(qkv_ptr, out_ptr, T,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # flash-attention style, fp32, bidirectional (no causal mask), scale = 1/sqrt(64)
    D: tl.constexpr = 64
    pid_m, pid_bh = tl.program_id(0), tl.program_id(1)
    b, h = pid_bh // 8, pid_bh % 8
    base = qkv_ptr + b * T * 1536 + h * 64                 # Q slice of this (sample, head)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rd = tl.arange(0, D)
    m_mask = rm < T
    q = tl.load(base + rm[:, None] * 1536 + rd[None, :], mask=m_mask[:, None], other=0.0)
    q = q * 0.125                                          # 1/sqrt(64)
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)
    for n0 in range(0, T, BLOCK_N):
        rn = n0 + tl.arange(0, BLOCK_N)
        n_mask = rn < T
        k = tl.load(base + 512 + rn[:, None] * 1536 + rd[None, :], mask=n_mask[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k), input_precision="ieee")
        s = tl.where(n_mask[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.exp(s - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(base + 1024 + rn[:, None] * 1536 + rd[None, :], mask=n_mask[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p, v, input_precision="ieee")
        m_i = m_new
    acc = acc / l_i[:, None]
    tl.store(out_ptr + (b * T + rm[:, None]) * 512 + h * 64 + rd[None, :],
             acc, mask=m_mask[:, None])


@triton.jit
def _gemm_gate_res_kernel(a_ptr, w_ptr, b_ptr, res_ptr, mod_ptr, out_ptr,
                          M, N, K, T, GATE_OFF: tl.constexpr,
                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # out = res + gate[batch(row)] * (A @ W(N,K)^T + bias) ; gate+residual fused epilogue
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + rm[:, None] * K + rk[None, :]
    w_ptrs = w_ptr + rn[None, :] * K + rk[:, None]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K, BLOCK_K):                         # K % BLOCK_K == 0 always
        a = tl.load(a_ptrs, mask=rm[:, None] < M, other=0.0)
        w = tl.load(w_ptrs, mask=rn[None, :] < N, other=0.0)
        acc = tl.dot(a, w, acc, input_precision="ieee")
        a_ptrs += BLOCK_K
        w_ptrs += BLOCK_K
    acc = acc + tl.load(b_ptr + rn, mask=rn < N, other=0.0)[None, :]
    msk = (rm[:, None] < M) & (rn[None, :] < N)
    gate = tl.load(mod_ptr + (rm // T)[:, None] * 3072 + GATE_OFF + rn[None, :],
                   mask=msk, other=0.0)
    res = tl.load(res_ptr + rm[:, None] * N + rn[None, :], mask=msk, other=0.0)
    tl.store(out_ptr + rm[:, None] * N + rn[None, :], res + gate * acc, mask=msk)


# ------------------------------- driver ------------------------------------

# x, c, output, weights are tensors on the GPU
def solve(
    x: torch.Tensor,
    c: torch.Tensor,
    output: torch.Tensor,
    weights: torch.Tensor,
    batch_size: int,
    seq_len: int,
):
    B, T = batch_size, seq_len
    M = B * T
    dev, f32 = x.device, torch.float32
    x, c, weights = x.contiguous(), c.contiguous(), weights.contiguous()

    mod  = torch.empty((B, MOD_DIM), device=dev, dtype=f32)
    qkv  = torch.empty((M, QKV_DIM), device=dev, dtype=f32)
    attn = torch.empty((M, D_MODEL), device=dev, dtype=f32)
    x1   = torch.empty((M, D_MODEL), device=dev, dtype=f32)
    ff   = torch.empty((M, MLP_DIM), device=dev, dtype=f32)

    # 1) mod = SiLU(c) @ W_ada^T + b_ada                       (SiLU fused)
    _mod_gemm_kernel[(triton.cdiv(B, 16), triton.cdiv(MOD_DIM, 64))](
        c, weights[OFF_W_ADA:], weights[OFF_B_ADA:], mod, B,
        BLOCK_M=16, BLOCK_N=64, BLOCK_K=64, num_warps=4, num_stages=2)

    # 2) qkv = (LN(x)*(1+scale_msa)+shift_msa) @ W_qkv^T + b_qkv   (LN+modulate fused)
    _gemm_ln_kernel[(triton.cdiv(M, 64), triton.cdiv(QKV_DIM, 64))](
        x, weights[OFF_W_QKV:], weights[OFF_B_QKV:], qkv, mod,
        M, QKV_DIM, T, SHIFT_OFF=SHIFT_MSA, EPS=1e-6, ACT=0,
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, num_warps=4, num_stages=2)

    # 3) self-attention (8 heads, bidirectional, scale 1/sqrt(64))
    _attn_kernel[(triton.cdiv(T, 64), B * N_HEADS)](
        qkv, attn, T, BLOCK_M=64, BLOCK_N=64, num_warps=4, num_stages=1)

    # 4) x1 = x + gate_msa * (attn @ W_o^T + b_o)              (gate+residual fused)
    _gemm_gate_res_kernel[(triton.cdiv(M, 64), triton.cdiv(D_MODEL, 64))](
        attn, weights[OFF_W_O:], weights[OFF_B_O:], x, mod, x1,
        M, D_MODEL, D_MODEL, T, GATE_OFF=GATE_MSA,
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, num_warps=4, num_stages=2)

    # 5) ff = GELU_tanh((LN(x1)*(1+scale_mlp)+shift_mlp) @ W_fc1^T + b_fc1)
    _gemm_ln_kernel[(triton.cdiv(M, 64), triton.cdiv(MLP_DIM, 64))](
        x1, weights[OFF_W_FC1:], weights[OFF_B_FC1:], ff, mod,
        M, MLP_DIM, T, SHIFT_OFF=SHIFT_MLP, EPS=1e-6, ACT=1,
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, num_warps=4, num_stages=2)

    # 6) output = x1 + gate_mlp * (ff @ W_fc2^T + b_fc2)       (gate+residual fused)
    _gemm_gate_res_kernel[(triton.cdiv(M, 64), triton.cdiv(D_MODEL, 64))](
        ff, weights[OFF_W_FC2:], weights[OFF_B_FC2:], x1, mod, output,
        M, D_MODEL, MLP_DIM, T, GATE_OFF=GATE_MLP,
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, num_warps=4, num_stages=2)

    return output
