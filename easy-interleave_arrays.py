import torch
import triton
import triton.language as tl


@triton.jit
def interleave_kernel(A_ptr, B_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE * 2 + tl.arange(0, BLOCK_SIZE * 2)
    mask = offsets < N * 2
    halved_offsets = offsets >> 1

    a_vals = tl.load(A_ptr + halved_offsets, mask=mask)
    b_vals = tl.load(B_ptr + halved_offsets, mask=mask)

    result = tl.where(offsets % 2, b_vals, a_vals)
    tl.store(output_ptr + offsets, result, mask=mask)


# A, B, output are tensors on the GPU
def solve(A: torch.Tensor, B: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 256

    def grid(meta):
        return (triton.cdiv(N, meta["BLOCK_SIZE"]),)

    interleave_kernel[grid](A, B, output, N, BLOCK_SIZE=BLOCK_SIZE)
