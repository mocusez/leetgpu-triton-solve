import torch
import triton
import triton.language as tl


@triton.jit
def geglu(input, output, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis = 0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x1 = tl.load(input + offset, mask = offset < N // 2, other = 0.0)
    x2 = tl.load(input + offset + N // 2, mask = offset < N // 2, other = 0.0)
    gelu_x2 = x2 * (1 + tl.erf(x2 * tl.sqrt(2.0) * 0.5)) * 0.5
    output_ = x1 * gelu_x2
    tl.store(output + offset, output_, mask = offset < N // 2)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N // 2, BLOCK_SIZE),)
    geglu[grid](input, output, N, BLOCK_SIZE=BLOCK_SIZE)
