import torch
import triton
import triton.language as tl

@triton.jit
def subarray_sum_kernel(input_ptr, output_ptr,
                        N, M, K,
                        S_DEP, E_DEP,
                        S_ROW, E_ROW,
                        S_COL, E_COL,
                        BLOCK_SIZE_N: tl.constexpr,
                        BLOCK_SIZE_M: tl.constexpr,
                        BLOCK_SIZE_K: tl.constexpr):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    pid2 = tl.program_id(2)

    offset_0 = pid0 * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N) + S_DEP
    offset_1 = pid1 * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M) + S_ROW
    offset_2 = pid2 * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K) + S_COL

    mask_0 = offset_0 <= E_DEP
    mask_1 = offset_1 <= E_ROW
    mask_2 = offset_2 <= E_COL

    offset = offset_0[:, None, None] * M * K + offset_1[None,:,None] * K + offset_2[None, None, :]
    mask = mask_0[:,None, None] & mask_1[None,:,None] & mask_2[None,None,:]

    input_data = tl.load(input_ptr + offset, mask=mask)
    input_data_sum = input_data.sum()

    if input_data_sum != 0:
        tl.atomic_add(output_ptr, input_data_sum)


# input, output are tensors on the GPU
def solve(
    input: torch.Tensor,
    output: torch.Tensor,
    N: int,
    M: int,
    K: int,
    S_DEP: int,
    E_DEP: int,
    S_ROW: int,
    E_ROW: int,
    S_COL: int,
    E_COL: int,
):
    BLOCK_SIZE_N = 1
    BLOCK_SIZE_M = 1
    BLOCK_SIZE_K = 1024

    grid = (triton.cdiv(E_DEP - S_DEP + 1, BLOCK_SIZE_N), triton.cdiv(E_ROW - S_ROW + 1, BLOCK_SIZE_M), triton.cdiv(E_COL - S_COL + 1, BLOCK_SIZE_K))
    subarray_sum_kernel[grid](input, output,
                            N, M, K,
                            S_DEP, E_DEP,
                            S_ROW, E_ROW,
                            S_COL, E_COL,
                            BLOCK_SIZE_N,
                            BLOCK_SIZE_M,
                            BLOCK_SIZE_K,
                            num_warps = 4)
