import torch
import triton
import triton.language as tl


@triton.jit
def _cross_attention_kernel(
    Q, K, V, O,
    stride_qm, stride_qh,
    stride_kn, stride_kh,
    stride_vn, stride_vh,
    stride_om, stride_oh,
    M, N, D,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # One program: BLOCK_M decoder queries x one head.
    # Loops over all N encoder positions with online (flash) softmax.
    pid_m = tl.program_id(0)
    h = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D

    # Load this block of queries for head h: (BLOCK_M, BLOCK_D)
    q_ptrs = Q + offs_m[:, None] * stride_qm + h * stride_qh + offs_d[None, :]
    q_mask = (offs_m[:, None] < M) & d_mask[None, :]
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Online-softmax state
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)  # running max
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)                # running sum of exp
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)       # running weighted V sum

    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N

        # Load K already transposed: (BLOCK_D, BLOCK_N).
        # Avoids tl.trans(), whose layout conversion allocates an extra
        # shared-memory buffer (which blows the 64 KB limit on Tesla T4).
        kt_ptrs = K + offs_n[None, :] * stride_kn + h * stride_kh + offs_d[:, None]
        kt = tl.load(kt_ptrs, mask=n_mask[None, :] & d_mask[:, None], other=0.0)

        # Scaled scores (BLOCK_M, BLOCK_N)
        s = tl.dot(q, kt) * sm_scale
        s = tl.where(n_mask[None, :], s, float("-inf"))

        # Online softmax rescale
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]

        v_ptrs = V + offs_n[:, None] * stride_vn + h * stride_vh + offs_d[None, :]
        v = tl.load(v_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
        acc += tl.dot(p, v)
        m_i = m_new

    acc = acc / l_i[:, None]

    o_ptrs = O + offs_m[:, None] * stride_om + h * stride_oh + offs_d[None, :]
    tl.store(o_ptrs, acc, mask=q_mask)


# Q, K, V, output are tensors on the GPU
def solve(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    output: torch.Tensor,
    M: int,
    N: int,
    H: int,
    D: int,
):
    BLOCK_D = max(16, triton.next_power_of_2(D))

    # Tesla T4 (sm_75) has only 64 KB of shared memory per SM. fp32 tiles are
    # 4 B/elem, and Triton stages Q/K/V/score tiles through shared memory for
    # tl.dot operand-layout conversion, so block sizes must shrink as D grows.
    # These combinations were verified offline to fit within 64 KB on sm_75.
    if BLOCK_D <= 64:
        BLOCK_M, BLOCK_N = 64, 64
    elif BLOCK_D == 128:
        BLOCK_M, BLOCK_N = 64, 32
    else:  # BLOCK_D == 256
        BLOCK_M, BLOCK_N = 32, 16

    grid = (triton.cdiv(M, BLOCK_M), H)
    _cross_attention_kernel[grid](
        Q, K, V, output,
        Q.stride(0), Q.stride(1),
        K.stride(0), K.stride(1),
        V.stride(0), V.stride(1),
        output.stride(0), output.stride(1),
        M, N, D,
        1.0 / (D ** 0.5),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=1,
    )