import torch
import triton
import triton.language as tl

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
    pid_m = tl.program_id(0)
    pid_f = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rf = pid_f * BLOCK_F + tl.arange(0, BLOCK_F)
    rk = tl.arange(0, BLOCK_K)

    mask_m = rm < M
    mask_f = rf < F

    x_ptrs = x_ptr + rm[:, None] * stride_xm + rk[None, :] * stride_xk
    wg_ptrs = wg_ptr + rk[:, None] * stride_wgk + rf[None, :] * stride_wgn
    wu_ptrs = wu_ptr + rk[:, None] * stride_wuk + rf[None, :] * stride_wun

    acc_g = tl.zeros((BLOCK_M, BLOCK_F), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_M, BLOCK_F), dtype=tl.float32)

    # hidden tile: gate/up projections, K = d_model reduced in BLOCK_K chunks
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem = K - k * BLOCK_K
        x_tile = tl.load(x_ptrs,
                         mask=mask_m[:, None] & (rk[None, :] < k_rem),
                         other=0.0)
        wg_tile = tl.load(wg_ptrs,
                          mask=(rk[:, None] < k_rem) & mask_f[None, :],
                          other=0.0)
        wu_tile = tl.load(wu_ptrs,
                          mask=(rk[:, None] < k_rem) & mask_f[None, :],
                          other=0.0)
        # fp32 dot; TF32 is unsupported on sm_75 (T4) anyway
        acc_g += tl.dot(x_tile, wg_tile, allow_tf32=False)
        acc_u += tl.dot(x_tile, wu_tile, allow_tf32=False)

        x_ptrs += BLOCK_K * stride_xk
        wg_ptrs += BLOCK_K * stride_wgk
        wu_ptrs += BLOCK_K * stride_wuk

    h = acc_g * tl.sigmoid(acc_g) * acc_u   # SiLU(gate) * up, in registers

    # h @ W_down[rf, :] accumulated into output[rm, :] via atomics
    for n0 in range(0, N, BLOCK_N):
        rn = n0 + tl.arange(0, BLOCK_N)
        mask_n = rn < N
        wd_tile = tl.load(
            wd_ptr + rf[:, None] * stride_wdf + rn[None, :] * stride_wdn,
            mask=mask_f[:, None] & mask_n[None, :], other=0.0)
        part = tl.dot(h, wd_tile, allow_tf32=False)
        c_ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        tl.atomic_add(c_ptrs, part, mask=mask_m[:, None] & mask_n[None, :])


# x, W_gate, W_up, W_down, output are tensors on the GPU
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
    BLOCK_M, BLOCK_F, BLOCK_K, BLOCK_N = 64, 64, 32, 64

    output.zero_()  # atomics accumulate, so start from zero

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(d_ffn, BLOCK_F))
    _swiglu_fused_kernel[grid](
        x, W_gate, W_up, W_down, output,
        M, d_model, d_ffn, d_model,
        x.stride(0), x.stride(1),
        W_gate.stride(0), W_gate.stride(1),
        W_up.stride(0), W_up.stride(1),
        W_down.stride(0), W_down.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_F=BLOCK_F, BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=2,
    )