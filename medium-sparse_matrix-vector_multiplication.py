import torch
import triton
import triton.language as tl


@triton.jit
def cal(A,x,y,M,N,BLOCK_M:tl.constexpr, BLOCK_N:tl.constexpr):
    pid_m = tl.program_id(0)
    offset_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offset_m < M
    sum = tl.zeros((BLOCK_M,),dtype=tl.float32)

    for pid_n in range(0, tl.cdiv(N,BLOCK_N)):
        offset_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offset_n < N
        vals_a = tl.load(A + offset_m[:,None] * N + offset_n[None,:],
                        mask = (mask_m[:,None] & mask_n[None,:]), other=0.0)
        vals_x = tl.load(x + offset_n, mask = mask_n, other=0.0)
        sum += tl.sum(vals_a * vals_x[None,:], axis=1)
    tl.store(y + offset_m, sum, mask=mask_m)


# A, x, y are tensors on the GPU
def solve(A: torch.Tensor, x: torch.Tensor, y: torch.Tensor, M: int, N: int, nnz: int):
    BLOCK_M = 1
    BLOCK_N = 1024
    grid = (triton.cdiv(M, BLOCK_M),)
    cal[grid](A,x,y,M,N,BLOCK_M,BLOCK_N)
