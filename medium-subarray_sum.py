import torch
import triton
import triton.language as tl

@triton.jit
def subarray_sum_kernel(input, output, S:tl.constexpr, E: tl.constexpr, BLOCK_SIZE:  tl.constexpr):
    pid = tl.program_id(axis = 0)
    offsets = pid * BLOCK_SIZE + S + tl.arange(0, BLOCK_SIZE)
    mask = offsets <= E

    local_input = tl.load(input+offsets, mask, other = 0)
    local_sum = tl.sum(local_input)

    tl.atomic_add(output, local_sum) a

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int, S: int, E: int):
    total = E - S + 1
    BLOCK_SIZE = 1024

    grid = (triton.cdiv(total, BLOCK_SIZE),)
    subarray_sum_kernel[grid](input, output, S, E, BLOCK_SIZE)
