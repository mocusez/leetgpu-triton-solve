import torch
import triton
import triton.language as tl

@triton.jit
def _dpo_loss_kernel(
    chosen_ptr, rejected_ptr, chosen_ref_ptr, rejected_ref_ptr,
    out_ptr,
    beta, inv_b,
    B,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < B

    l_plus = tl.load(chosen_ptr + offs, mask = mask, other = 0.0)
    l_minus = tl.load(rejected_ptr + offs, mask = mask, other = 0.0)
    l_plus_r = tl.load(chosen_ref_ptr + offs, mask = mask, other = 0.0)
    l_minus_r = tl.load(rejected_ref_ptr + offs, mask = mask, other = 0.0)

    z = beta * ((l_plus - l_minus) - (l_plus_r - l_minus_r))

    x = -z
    sp = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(x)))
    sp = tl.where(mask, sp, 0.0)

    partial = tl.sum(sp, axis = 0)
    tl.atomic_add(out_ptr, partial * inv_b)

# chosen_logps, rejected_logps, chosen_ref_logps, rejected_ref_logps, output are tensors on the GPU
def solve(
    chosen_logps: torch.Tensor,
    rejected_logps: torch.Tensor,
    chosen_ref_logps: torch.Tensor,
    rejected_ref_logps: torch.Tensor,
    output: torch.Tensor,
    beta: float,
    B: int,
):
    BLOCK = 1024
    grid = (triton.cdiv(B, BLOCK),)
    _dpo_loss_kernel[grid](
        chosen_logps, rejected_logps, chosen_ref_logps, rejected_ref_logps,
        output,
        beta, 1.0 / B,
        B,
        BLOCK = BLOCK,
        num_warps = 4,
    )
