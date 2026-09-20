import torch
import triton
import triton.language as tl

_DOT_PRECISION = "tf32x3"

@triton.autotune(
    configs = [
        triton.Config(
            {"BM": 32, "BD": 64, "BK": 32},
            num_warps = 4,
            num_stages = 3,
        ),
        triton.Config(
            {"BM": 64, "BD": 64, "BK": 32},
            num_warps = 4,
            num_stages = 3,
        ),
        triton.Config(
            {"BM": 64, "BD": 128, "BK": 32},
            num_warps = 8,
            num_stages = 3,
        ),
        triton.Config(
            {"BM": 64, "BD": 128, "BK": 64},
            num_warps = 8,
            num_stages = 3,
        ),
    ],
    key = ["B", "C", "H", "W", "P", "D", "PRECISION"]
)
@triton.jit
def _patch_embed_kernel(
    Images, Weight, Bias, Cls, Pos, Out,
    B: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    P: tl.constexpr,
    D: tl.constexpr,
    PRECISION: tl.constexpr,
    BM: tl.constexpr,
    BD: tl.constexpr,
    BK: tl.constexpr,
    GROUP_M: tl.constexpr
):
    GW: tl.constexpr = W // P
    N: tl.constexpr = (H // P) * GW
    T: tl.constexpr = N + 1
    M: tl.constexpr = B * T
    K: tl.constexpr = C * P * P

    pid = tl.program_id(0)

    num_m = tl.cdiv(M, BM)
    num_d = tl.cdiv(D, BD)

    group_size = GROUP_M * num_d
    group_id = pid // group_size
    first_m = group_id * GROUP_M
    actual_m = tl.minimum(num_m - first_m, GROUP_M)

    in_group = pid % group_size
    pid_m = first_m + in_group % actual_m
    pid_d = in_group // actual_m 

    m = pid_m * BM + tl.arange(0, BM)
    d = pid_d * BD + tl.arange(0, BD)

    batch = m // T
    row = m % T

    patch = tl.maximum(row - 1, 0)
    is_patch = (m < M) & (row != 0)

    image_base= (
        batch * (C * H * W)
        + (patch // GW) * (P * W)
        + (patch % GW) * P
    )

    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BD), dtype = tl.float32)
    for block_k in range(0, tl.cdiv(K, BK)):
        k = block_k * BK + rk

        image_delta = (
            (k //(P * P)) * (H * W)
            + ((k // P) % P) * W
            + k % P
        )

        a = tl.load(
            Images + image_base[:, None] + image_delta[None, :],
            mask = is_patch[:, None] & (k[None, :] < K),
            other = 0.0
        )

        w = tl.load(
            Weight + d[None, :] * K + k[:, None],
            mask = (k[: ,None] < K) & (d[None, :] < D),
            other = 0.0,
        )
        acc = tl.dot(a, w, acc, input_precision = PRECISION)

    bias = tl.load(Bias + d, mask = d < D, other = 0.0)
    cls = tl.load(Cls + d, mask = d < D, other = 0.0)

    valid = (m[: ,None] < M) & (d[None, :] < D)

    pos = tl.load(
        Pos + row[: ,None].to(tl.int64) * D + d[None, :],
        mask = valid,
        other = 0.0
    )

    result = tl.where(
        row[:, None] == 0,
        cls[None, :],
        acc + bias[None, :],
    )
    result = result + pos

    tl.store(
        Out + m[:, None].to(tl.int64) * D + d[None, :],
        result,
        mask = valid,
    )


# images, patch_weight, patch_bias, cls_token, pos_embed, output are tensors on the GPU
def solve(
    images: torch.Tensor,
    patch_weight: torch.Tensor,
    patch_bias: torch.Tensor,
    cls_token: torch.Tensor,
    pos_embed: torch.Tensor,
    output: torch.Tensor,
    B: int,
    C: int,
    H: int,
    W: int,
    P: int,
    D: int,
):
    N = (H // P) * (W // P)

    grid = lambda meta: (
        triton.cdiv(B * (N + 1), meta["BM"])
        * triton.cdiv(D, meta["BD"]),
    )

    _patch_embed_kernel[grid](
        images,
        patch_weight,
        patch_bias,
        cls_token,
        pos_embed,
        output,
        B,
        C,
        H,
        W,
        P,
        D,
        PRECISION = _DOT_PRECISION,
        GROUP_M = 8
    )
