import torch
import triton
import triton.language as tl

@triton.jit
def mse_kernel(
    input_ptr, target_ptr, output_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    input_vals = tl.load(input_ptr + offsets, mask = mask, other = 0.0)
    target_vals = tl.load(target_ptr + offsets, mask = mask, other = 0.0)

    diff = input_vals - target_vals
    squared_diff = diff * diff

    block_sum = tl.sum(squared_diff)
    tl.atomic_add(output_ptr, block_sum)




# predictions, targets, mse are tensors on the GPU
def solve(predictions: torch.Tensor, targets: torch.Tensor, mse: torch.Tensor, N: int):
    input_flat = predictions.contiguous().view(-1)
    target_flat = targets.contiguous().view(-1)

    BLOCK_SIZE = triton.next_power_of_2(min(N, 1024))

    num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    mse_kernel[(num_blocks,)](
        input_flat, target_flat, mse,
        N,
        BLOCK_SIZE = BLOCK_SIZE,
    )
    mse /= N
