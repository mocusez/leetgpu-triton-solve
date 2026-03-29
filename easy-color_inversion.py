import torch
import triton
import triton.language as tl


@triton.jit
def invert_kernel(image_ptr, width, height, BLOCK_SIZE: tl.constexpr):
    image_ptr = image_ptr.to(tl.pointer_type(tl.uint8))
    pid = tl.program_id(axis = 0)
    offsets = pid * BLOCK_SIZE * 4 + tl.arange(0, BLOCK_SIZE * 4)
    mask = (offsets < width * height * 4) & (offsets % 4 != 3)
    image = tl.load(image_ptr + offsets, mask=mask)
    image = 255 - image
    tl.store(image_ptr + offsets, image, mask=mask)


# image is a tensor on the GPU
def solve(image: torch.Tensor, width: int, height: int):
    BLOCK_SIZE = 1024
    n_pixels = width * height
    grid = (triton.cdiv(n_pixels, BLOCK_SIZE),)

    invert_kernel[grid](image, width, height, BLOCK_SIZE)
