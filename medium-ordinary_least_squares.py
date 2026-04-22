import torch
import triton
import triton.language as tl


@triton.jit
def gram_kernel(X, y, XtX, Xty, n_samples, n_features, stride_x_0, stride_x_1, BLOCK_M:tl.constexpr,BLOCK_N:tl.constexpr):
    pid = tl.program_id(0)
    offset_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offset_n < n_features

    acc_xtx = tl.zeros((BLOCK_N, BLOCK_N), dtype = tl.float32)
    acc_xty = tl.zeros((BLOCK_N, ), dtype = tl.float32)

    for i in range(0, n_samples, BLOCK_M):
        offset_m = i + tl.arange(0, BLOCK_M)
        mask_m = offset_m < n_samples

        x_mn = tl.load(X + offset_m[:, None] * stride_x_0 + offset_n[None,:] * stride_x_1, mask = (mask_m[:, None] & mask_n[None, :]), other = 0.0)
        x_nm = tl.load(X + offset_n[:, None] * stride_x_1 + offset_m[None,:] * stride_x_0, mask = (mask_n[:, None] & mask_m[None, :]), other = 0.0)

        acc_xtx += tl.dot(x_nm, x_mn)
        y_m = tl.load(y + offset_m, mask = mask_m, other = 0.0)

        acc_xty += tl.sum(x_nm * y_m, axis = 1)

    tl.store(XtX + offset_n[:, None] * n_features + offset_n[None, :], acc_xtx, mask = (mask_n[:,None] & mask_n[None,:]))
    tl.store(Xty + offset_n, acc_xty, mask = mask_n)




# X, y, beta are tensors on the GPU
def solve(X: torch.Tensor, y: torch.Tensor, beta: torch.Tensor, n_samples: int, n_features: int):
    BLOCK_M = 32
    BLOCK_N = 32

    X = X.view(n_samples, n_features).contiguous()
    y = y.view(n_samples).contiguous()
    beta = beta.view(n_features).contiguous()

    XtX = torch.empty((n_features, n_features), device=X.device, dtype = torch.float32)
    Xty = torch.empty(n_features, device=X.device, dtype=torch.float32)

    grid = (triton.cdiv(n_features, BLOCK_N),)
    gram_kernel[grid](
        X, y, XtX, Xty,
        n_samples, n_features,
        X.stride(0), X.stride(1),
        BLOCK_M,
        BLOCK_N
    )
    beta[:] = torch.linalg.solve(XtX, Xty)
