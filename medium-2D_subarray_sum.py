import torch
import triton
import triton.language as tl


@triton.jit
def subarray_sum_kernel(input_ptr,  output_ptr,
                        N, M,
                        S_ROW,E_ROW, S_COL, E_COL,
                        BLOCK_SIZE: tl.constexpr,
                        BLOCK_SIZE_COL: tl.constexpr):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    offset_row = pid0 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) + S_ROW
    offset_col = pid1 * BLOCK_SIZE_COL + tl.arange(0, BLOCK_SIZE_COL) + S_COL

    mask_row = offset_row <= E_ROW
    mask_col = offset_col <= E_COL

    offset = offset_row[:,None] * M + offset_col[None,:]
    mask = mask_row[:,None] & mask_col[None,:]

    input_data = tl.load(input_ptr + offset, mask=mask)
    input_data_sum = input_data.sum()

    if input_data_sum > 0:
        tl.atomic_add(output_ptr, input_data.sum())


# input, output are tensors on the GPU
def solve(
    input: torch.Tensor,
    output: torch.Tensor,
    N: int,
    M: int,
    S_ROW: int,
    E_ROW: int,
    S_COL: int,
    E_COL: int,
):
    BLOCK_SIZE = 1
    BLOCK_SIZE_COL = 1024

    grid = (triton.cdiv(E_ROW - S_ROW + 1, BLOCK_SIZE), triton.cdiv(E_COL - S_COL + 1, BLOCK_SIZE_COL))
    subarray_sum_kernel[grid](input, output, N, M, S_ROW, E_ROW, S_COL, E_COL, BLOCK_SIZE, BLOCK_SIZE_COL, num_warps = 4)

