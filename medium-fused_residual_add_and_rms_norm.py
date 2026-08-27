import torch
import triton
import triton.language as tl

@triton.jit
def _add_rmsnorm_fused_kernel(
    x_ptr, res_ptr, w_ptr, out_ptr,
    C, eps,
    BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < C

    x = tl.load(x_ptr + row * C + cols, mask = mask, other = 0.0)
    r = tl.load(res_ptr + row * C + cols, mask = mask, other = 0.0)
    z = x + r

    ms = tl.sum(z * z, axis = 0) / C
    rstd = 1.0 / tl.sqrt(ms + eps)
    w = tl.load(w_ptr + cols, mask = mask, other = 0.0)
    tl.store(out_ptr + row * C + cols, z * rstd * w, mask = mask)

@triton.jit
def _add_rmsnorm_fused_loop_kernel(
    x_ptr, res_ptr, w_ptr, out_ptr,
    C, eps,
    BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    cols = tl.arrange(0, BLOCK)

    acc = tl.zeros([BLOCK], dtype = tl.float32)
    for start in range(0, C, BLOCK):
        offs = start + cols
        mask = offs < C
        x = tl.load(x_ptr + row * C + offs, mask = mask, other = 0.0)
        r = tl.load(res_ptr, row * C + offs, mask = mask, other = 0.0)
        z = x + r
        acc += z * z
    
    ms = tl.sum(acc, axis = 0) / C
    rstd = 1.0 / tl.sqrt(ms + eps)
    for start in range(0, C, BLOCK):
        offs= start + cols 
        mask = offs < C
        x = tl.load(x_ptr + row * C + offs, mask = mask, other = 0.0)
        r = tl.load(res_ptr + row * C + offs, mask = mask, other = 0.0)
        w = tl.load(w_ptr + offs, mask = mask, other = 0.0)
        tl.store(out_ptr + row * C + offs,(x + r) * rstd * w, mask = mask)

_MAX_SINGLE_BLOCK = 8192

# x, residual, weight, out are tensors on the GPU
def solve(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    N: int,
    C: int,
    eps: float,
):
    block = triton.next_power_of_2(C)
    if block <= _MAX_SINGLE_BLOCK:
        num_warps = max(1, min(32, block // 512))
        _add_rmsnorm_fused_kernel[(N, )](
            x, residual, weight, out,
            C, eps,
            BLOCK=block,
            num_warps=num_warps,
        )
    else:
        _add_rmsnorm_fused_loop_kernel[(N,)](
            x, residual, weight, out,
            C, eps,
            BLOCK=4096,
            num_warps = 8
        )
