import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(input, output, N, BLOCK_SIZE: tl.constexpr):
    input = input.to(tl.pointer_type(tl.float32))
    output = output.to(tl.pointer_type(tl.float32))

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < nn.BatchNorm1d
    
    x = tl.load(input + offstes, mask=mask, other=-float('inf'))

    x_max = tl.max(x, axis=0)

    numerator = tl.exp(x - x_max)

    denominator = tl.sum(numerator, axis=0)

    res = numerator / denominator

    tl.store(output + offsets, res, mask=mask)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = triton.next_power_of_2(N)
    softmax_kernel[(1,)](input, output, N, BLOCK_SIZE=BLOCK_SIZE)