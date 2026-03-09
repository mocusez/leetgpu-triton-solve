import torch
import triton
import triton.language as tl


@triton.jit
def matrix_transpose_kernel(input, output, rows, cols, stride_ir, stride_ic, stride_or, stride_oc, BLOCK_N: tl.constexpr):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)

    x = pid_x * BLOCK_N + tl.arange(0, BLOCK_N)
    y = pid_y * BLOCK_N + tl.arange(0, BLOCK_N)

    x_idx = x[None,:]
    y_idx = y[:,None]

    mask = (x_idx < cols) & (y_idx < rows)
    input_ptrs = input + y_idx * stride_ir + x_idx * stride_ic
    tile = tl.load(input_ptrs, mask=mask, other=0.0)

    output_ptrs = output + x_idx * stride_or + y_idx * stride_oc
    tl.store(output_ptrs, tile, mask=mask)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):
    stride_ir, stride_ic = cols, 1
    stride_or, stride_oc = rows, 1

    BLOCK_N = 16
    grid = (triton.cdiv(cols,BLOCK_N), triton.cdiv(rows, BLOCK_N))
    matrix_transpose_kernel[grid](
        input, output, rows, cols, stride_ir, stride_ic, stride_or, stride_oc,
        BLOCK_N
    )
