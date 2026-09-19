import torch
import triton
import triton.language as tl


# q, kv_cache, W_UK, W_UV, output are tensors on the GPU
def solve(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    W_UK: torch.Tensor,
    W_UV: torch.Tensor,
    output: torch.Tensor,
    num_heads: int,
    seq_len: int,
    kv_lora_rank: int,
    head_dim: int,
    rope_dim: int,
):
    H = num_heads
    S = seq_len
    R = kv_lora_rank
    D = head_dim
    P = rope_dim

    # Inputs follow the shapes/dtypes/ranges in the problem statement.
    # Actual strides are passed to the kernels; contiguous inputs are not required.
    BH = 16
    BN = min(128, max(32, triton.next_power_of_2(S)))
    BK = min(64, max(32, triton.next_power_of_2(R)))
    BP = max(32, triton.next_power_of_2(P))
    NS = triton.cdiv(S, BN)

    # Change to "ieee" to disable TF32-based dot products.
    precision = "tf32x3"
    scale_log2 = 1.4426950408889634 * ((D + P) ** -0.5)

    q_latent = torch.empty((H, R), device=q.device, dtype=torch.float32)
    partial = torch.empty((H, NS, R), device=q.device, dtype=torch.float32)
    stats = torch.empty((H, NS, 2), device=q.device, dtype=torch.float32)
    latent = torch.empty((H, R), device=q.device, dtype=torch.float32)

    _mla_absorb_query[(H, triton.cdiv(R, 64))](
        q, W_UK, q_latent,
        R, D,
        *q.stride(), *W_UK.stride(),
        BD=triton.next_power_of_2(D), BR=64,
        num_warps=4, num_stages=1,
    )

    _mla_split_attention[(triton.cdiv(H, BH), NS)](
        q, q_latent, kv_cache, partial, stats,
        H, S, R, D, P, NS, scale_log2,
        *q.stride(), *kv_cache.stride(),
        BH=BH, BN=BN, BK=BK, BP=BP,
        PRECISION=precision,
        num_warps=8, num_stages=1,
    )

    _mla_merge_segments[(H, triton.cdiv(R, 128))](
        partial, stats, latent,
        R, NS,
        BS=triton.next_power_of_2(NS), BR=128,
        num_warps=4, num_stages=1,
    )

    _mla_project_output[(H, triton.cdiv(D, 32))](
        latent, W_UV, output,
        R, D,
        *W_UV.stride(), *output.stride(),
        BR=triton.next_power_of_2(R), BD=32,
        num_warps=8 if R >= 256 else 4,
        num_stages=1,
    )


@triton.jit
def _mla_absorb_query(
    Q, WUK, QL,
    R: tl.constexpr, D: tl.constexpr,
    qh: tl.constexpr, qd: tl.constexpr,
    wh: tl.constexpr, wd: tl.constexpr, wr: tl.constexpr,
    BD: tl.constexpr, BR: tl.constexpr,
):
    h = tl.program_id(0)
    r = tl.program_id(1) * BR + tl.arange(0, BR)
    d = tl.arange(0, BD)

    q_nope = tl.load(Q + h * qh + d * qd, d < D, other=0.0)
    w = tl.load(
        WUK + h * wh + d[:, None] * wd + r[None, :] * wr,
        (d[:, None] < D) & (r[None, :] < R),
        other=0.0,
    )
    q_latent = tl.sum(q_nope[:, None] * w, axis=0)
    tl.store(QL + h * R + r, q_latent, r < R)


@triton.jit
def _mla_split_attention(
    Q, QL, CACHE, PART, STATS,
    H: tl.constexpr, S: tl.constexpr,
    R: tl.constexpr, D: tl.constexpr, P: tl.constexpr,
    NS: tl.constexpr, SCALE_LOG2: tl.constexpr,
    qh: tl.constexpr, qd: tl.constexpr,
    cs: tl.constexpr, cd: tl.constexpr,
    BH: tl.constexpr, BN: tl.constexpr,
    BK: tl.constexpr, BP: tl.constexpr,
    PRECISION: tl.constexpr,
):
    h = tl.program_id(0) * BH + tl.arange(0, BH)
    split = tl.program_id(1)
    t = split * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)

    scores = tl.zeros((BH, BN), dtype=tl.float32)

    # q_latent @ c.T; cache tiles are shared by the BH heads.
    for block in range(0, tl.cdiv(R, BK)):
        r = block * BK + k
        q_latent = tl.load(
            QL + h[:, None] * R + r[None, :],
            (h[:, None] < H) & (r[None, :] < R),
            other=0.0,
        )
        c = tl.load(
            CACHE + t[:, None] * cs + r[None, :] * cd,
            (t[:, None] < S) & (r[None, :] < R),
            other=0.0,
        )
        scores = tl.dot(
            q_latent, tl.trans(c), scores,
            input_precision=PRECISION,
        )

    # q_pe @ k_pe.T; there is no head dimension in cached k_pe.
    pidx = tl.arange(0, BP)
    q_pe = tl.load(
        Q + h[:, None] * qh + (D + pidx[None, :]) * qd,
        (h[:, None] < H) & (pidx[None, :] < P),
        other=0.0,
    )
    k_pe = tl.load(
        CACHE + t[:, None] * cs + (R + pidx[None, :]) * cd,
        (t[:, None] < S) & (pidx[None, :] < P),
        other=0.0,
    )
    scores = tl.dot(
        q_pe, tl.trans(k_pe), scores,
        input_precision=PRECISION,
    )

    # exp2(x * log2(e)) == exp(x).
    scores = scores * SCALE_LOG2
    scores = tl.where(t[None, :] < S, scores, -float("inf"))

    # Every segment contains at least one valid token.
    # Padded heads are suppressed by masks on loads/stores.
    m = tl.max(scores, axis=1)
    p = tl.exp2(scores - m[:, None])
    l = tl.sum(p, axis=1)

    stat_offsets = (h * NS + split) * 2
    tl.store(STATS + stat_offsets, m, h < H)
    tl.store(STATS + stat_offsets + 1, l, h < H)

    # Store unnormalized weighted LATENT sums; p remains float32.
    for block in range(0, tl.cdiv(R, BK)):
        r = block * BK + k
        c = tl.load(
            CACHE + t[:, None] * cs + r[None, :] * cd,
            (t[:, None] < S) & (r[None, :] < R),
            other=0.0,
        )
        u = tl.dot(p, c, input_precision=PRECISION)
        offsets = (h[:, None] * NS + split) * R + r[None, :]
        tl.store(
            PART + offsets, u,
            (h[:, None] < H) & (r[None, :] < R),
        )


@triton.jit
def _mla_merge_segments(
    PART, STATS, LATENT,
    R: tl.constexpr, NS: tl.constexpr,
    BS: tl.constexpr, BR: tl.constexpr,
):
    h = tl.program_id(0)
    r = tl.program_id(1) * BR + tl.arange(0, BR)
    split = tl.arange(0, BS)

    stat_offsets = (h * NS + split) * 2
    m = tl.load(
        STATS + stat_offsets, split < NS, other=-float("inf")
    )
    l = tl.load(
        STATS + stat_offsets + 1, split < NS, other=0.0
    )

    global_m = tl.max(m, axis=0)
    alpha = tl.exp2(m - global_m)
    denominator = tl.sum(alpha * l, axis=0)

    u = tl.load(
        PART + (h * NS + split[:, None]) * R + r[None, :],
        (split[:, None] < NS) & (r[None, :] < R),
        other=0.0,
    )
    latent = tl.sum(u * alpha[:, None], axis=0) / denominator
    tl.store(LATENT + h * R + r, latent, r < R)


@triton.jit
def _mla_project_output(
    LATENT, WUV, OUT,
    R: tl.constexpr, D: tl.constexpr,
    wh: tl.constexpr, wr: tl.constexpr, wd: tl.constexpr,
    oh: tl.constexpr, od: tl.constexpr,
    BR: tl.constexpr, BD: tl.constexpr,
):
    h = tl.program_id(0)
    d = tl.program_id(1) * BD + tl.arange(0, BD)
    r = tl.arange(0, BR)

    latent = tl.load(LATENT + h * R + r, r < R, other=0.0)
    w = tl.load(
        WUV + h * wh + r[:, None] * wr + d[None, :] * wd,
        (r[:, None] < R) & (d[None, :] < D),
        other=0.0,
    )
    result = tl.sum(latent[:, None] * w, axis=0)
    tl.store(OUT + h * oh + d * od, result, d < D)