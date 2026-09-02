import torch
import triton
import triton.language as tl

@triton.jit
def _swiglu_gate_up_kernel(
    x_ptr, wg_ptr, wu_ptr, h_ptr,
    M, K, N,
    stride_xm, stride_xk,
    stride_wgk, stride_wgn,
    stride_wuk, stride_wun,
    stride_hm, stride_hn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    mask_m = rm < M
    mask_n = rn < N

    x_ptrs = x_ptr + rm[:, None] * stride_xm + rk[None, :] * stride_xk
    wg_ptrs = wg_ptr + rk[:, None] * stride_wgk + rn[None, :] * stride_wgn
    wu_ptrs = wu_ptr + rk[:, None] * stride_wuk + rn[None, :] * stride_wun

    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype = tl.float32)
    acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype = tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem = K - k * BLOCK_K
        x_tile = tl.load(x_ptrs,
                        mask = mask_m[:, None] & (rk[None, :] < k_rem),
                        other = 0.0)
        wg_tile = tl.load(wg_ptrs,
                        mask = (rk[:, None] < k_rem) & mask_n[None, :],
                        other = 0.0)
        wu_tile = tl.load(wu_ptrs,
                        mask = (rk[:, None] < k_rem) & mask_n[None, :],
                        other = 0.0)

        acc_g += tl.dot(x_tile, wg_tile, allow_tf32 = False)
        acc_u += tl.dot(x_tile, wu_tile, allow_tf32 = False)

        x_ptrs += BLOCK_K * stride_xk
        wg_ptrs += BLOCK_K * stride_wgk
        wu_ptrs += BLOCK_K * stride_wuk

    h = acc_g * tl.sigmoid(acc_g) * acc_u

    h_ptrs = h_ptr + rm[:, None] * stride_hm + rn[None, :] * stride_hn
    tl.store(h_ptrs, h, mask = mask_m[:, None] & mask_n[None, :])

@triton.jit
def _matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, K, N,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    mask_m = rm < M
    mask_n = rn < N

    a_ptrs = a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
    b_ptrs = b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype = tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem = K - k * BLOCK_K
        a_tile = tl.load(a_ptrs,
                        mask = mask_m[:, None] & (rk[None, :] < k_rem),
                        other = 0.0)
        b_tile = tl.load(b_ptrs,
                        mask = (rk[:, None] < k_rem) & mask_n[None, :],
                        other = 0.0)
        acc += tl.dot(a_tile, b_tile, allow_tf32 = False)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    c_ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask = mask_m[:, None] & mask_n[None, :])

def solve(
    x: torch.Tensor,
    W_gate: torch.Tensor,
    W_up: torch.Tensor,
    W_down: torch.Tensor,
    output: torch.Tensor,
    M: int,
    d_model: int,
    d_ffn: int,
):
    # Block sizes are chosen so that shared-memory usage stays within the
    # 64 KB per-block limit of sm_75 (Tesla T4) with num_stages=2:
    #   kernel1: 2 * (64*32 + 2*32*64) * 4 B = 48 KB
    #   kernel2: 2 * (64*32 + 32*64) * 4 B   = 32 KB
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32

    hidden = torch.empty((M, d_ffn), device=x.device, dtype=torch.float32)

    grid1 = (triton.cdiv(M, BLOCK_M), triton.cdiv(d_ffn, BLOCK_N))
    _swiglu_gate_up_kernel[grid1](
        x, W_gate, W_up, hidden,
        M, d_model, d_ffn,
        x.stride(0), x.stride(1),
        W_gate.stride(0), W_gate.stride(1),
        W_up.stride(0), W_up.stride(1),
        hidden.stride(0), hidden.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )

    grid2 = (triton.cdiv(M, BLOCK_M), triton.cdiv(d_model, BLOCK_N))
    _matmul_kernel[grid2](
        hidden, W_down, output,
        M, d_ffn, d_model,
        hidden.stride(0), hidden.stride(1),
        W_down.stride(0), W_down.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
