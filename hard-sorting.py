import math
import triton
import triton.language as tl


# ============================================================
# 1. Tile-local sort
# ============================================================

@triton.jit
def local_sort_kernel(
    input_ptr,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    input_ptr = input_ptr.to(tl.pointer_type(tl.float32))

    pid = tl.program_id(0)

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    # Padding with +inf means the tail of a non-power-of-two input
    # naturally moves to the end.
    x = tl.load(
        input_ptr + offsets,
        mask=mask,
        other=float("inf"),
    )

    # Sort the entire tile inside one Triton program.
    x = tl.sort(x)

    tl.store(
        input_ptr + offsets,
        x,
        mask=mask,
    )


# ============================================================
# 2. First stage of a bitonic merge
#
# Example for 8 elements:
#
# x0 <-> x7
# x1 <-> x6
# x2 <-> x5
# x3 <-> x4
#
# STEP is compile-time constant.
# ============================================================

@triton.jit
def global_reverse_compare_kernel(
    input_ptr,
    n,
    STEP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    input_ptr = input_ptr.to(tl.pointer_type(tl.float32))

    pid = tl.program_id(0)

    comparator_id = (
        pid * BLOCK_SIZE
        + tl.arange(0, BLOCK_SIZE)
    )

    half = 1 << STEP
    stride = 1 << (STEP + 1)

    # Equivalent to:
    #
    # block_id = comparator_id // half
    # lane     = comparator_id % half
    #
    # Since half is power of two, explicitly use bit operations.
    block_id = comparator_id >> STEP
    lane = comparator_id & (half - 1)

    block_start = block_id << (STEP + 1)

    off_x = block_start + lane
    off_y = block_start + stride - 1 - lane

    off_x = off_x.to(tl.int32)
    off_y = off_y.to(tl.int32)

    x_valid = off_x < n
    y_valid = off_y < n

    x = tl.load(
        input_ptr + off_x,
        mask=x_valid,
        other=float("inf"),
    )

    y = tl.load(
        input_ptr + off_y,
        mask=y_valid,
        other=float("inf"),
    )

    lo = tl.minimum(x, y)
    hi = tl.maximum(x, y)

    # Always write the compare-exchange result.
    # This generally gives simpler generated code than write_msk.
    tl.store(
        input_ptr + off_x,
        lo,
        mask=x_valid,
    )

    tl.store(
        input_ptr + off_y,
        hi,
        mask=y_valid,
    )


# ============================================================
# 3. Normal bitonic merge compare
#
# Example:
#
# x0 <-> x4
# x1 <-> x5
# x2 <-> x6
# x3 <-> x7
# ============================================================

@triton.jit
def global_stride_compare_kernel(
    input_ptr,
    n,
    STEP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    input_ptr = input_ptr.to(tl.pointer_type(tl.float32))

    pid = tl.program_id(0)

    comparator_id = (
        pid * BLOCK_SIZE
        + tl.arange(0, BLOCK_SIZE)
    )

    half = 1 << STEP

    block_id = comparator_id >> STEP
    lane = comparator_id & (half - 1)

    off_x = (block_id << (STEP + 1)) + lane
    off_y = off_x + half

    off_x = off_x.to(tl.int32)
    off_y = off_y.to(tl.int32)

    x_valid = off_x < n
    y_valid = off_y < n

    x = tl.load(
        input_ptr + off_x,
        mask=x_valid,
        other=float("inf"),
    )

    y = tl.load(
        input_ptr + off_y,
        mask=y_valid,
        other=float("inf"),
    )

    lo = tl.minimum(x, y)
    hi = tl.maximum(x, y)

    tl.store(
        input_ptr + off_x,
        lo,
        mask=x_valid,
    )

    tl.store(
        input_ptr + off_y,
        hi,
        mask=y_valid,
    )


# ============================================================
# Host entry point
#
# data_ptr: raw CUDA device pointer
# N:        number of float32 elements
# ============================================================

def solve(data_ptr: int, N: int):
    if N <= 1:
        return

    # --------------------------------------------------------
    # T4 tuning parameters
    # --------------------------------------------------------
    #
    # LOCAL_BLOCK determines how much work is fused into one
    # local Triton sort.
    #
    # Recommended sweep on T4:
    #
    #   LOCAL_BLOCK = 256
    #   LOCAL_BLOCK = 512
    #   LOCAL_BLOCK = 1024
    #
    # 512 is a relatively conservative default for T4.
    #
    LOCAL_BLOCK = 512

    # Number of comparator IDs processed by one global program.
    GLOBAL_BLOCK = 1024

    local_log2 = int(math.log2(LOCAL_BLOCK))

    n_pow2 = triton.next_power_of_2(N)
    total_log2 = int(math.log2(n_pow2))

    # --------------------------------------------------------
    # Case 1:
    # Entire array fits inside one local sorting tile.
    # --------------------------------------------------------
    if n_pow2 <= LOCAL_BLOCK:
        block = n_pow2

        local_sort_kernel[(1,)](
            data_ptr,
            N,
            BLOCK_SIZE=block,
            num_warps=8 if block >= 512 else 4,
        )

        return

    # --------------------------------------------------------
    # Phase A:
    # Sort every LOCAL_BLOCK independently.
    #
    # Instead of running:
    #
    #   1 + 2 + ... + log2(LOCAL_BLOCK)
    #
    # global kernels, this requires only ONE kernel launch.
    # --------------------------------------------------------

    local_grid = (triton.cdiv(N, LOCAL_BLOCK),)

    local_sort_kernel[local_grid](
        data_ptr,
        N,
        BLOCK_SIZE=LOCAL_BLOCK,
        num_warps=8,
    )

    # Number of compare-exchanges in each global stage.
    #
    # Use n_pow2 here rather than N because we're conceptually
    # sorting:
    #
    #   [input..., +inf, +inf, ...]
    #
    num_comparators = n_pow2 // 2

    global_grid = (
        triton.cdiv(num_comparators, GLOBAL_BLOCK),
    )

    # --------------------------------------------------------
    # Phase B:
    # Hierarchical global merges.
    #
    # We already have sorted LOCAL_BLOCK-size chunks.
    #
    # For each larger merge:
    #
    #   1. reverse compare across the whole merge region
    #   2. perform only comparisons that CROSS local blocks
    #   3. once distance < LOCAL_BLOCK, finish all remaining
    #      stages with one local_sort_kernel
    #
    # This is the main optimization over the original code.
    # --------------------------------------------------------

    for merge_step in range(local_log2, total_log2):

        # ----------------------------------------------------
        # First / mirrored compare of bitonic merge.
        # ----------------------------------------------------

        global_reverse_compare_kernel[global_grid](
            data_ptr,
            N,
            STEP=merge_step,
            BLOCK_SIZE=GLOBAL_BLOCK,
            num_warps=8,
        )

        # ----------------------------------------------------
        # Only execute global stages whose distance is at least
        # LOCAL_BLOCK.
        #
        # Once comparison distance is smaller than LOCAL_BLOCK,
        # no comparator crosses a local tile boundary anymore.
        # ----------------------------------------------------

        for step in range(
            merge_step - 1,
            local_log2 - 1,
            -1,
        ):
            global_stride_compare_kernel[global_grid](
                data_ptr,
                N,
                STEP=step,
                BLOCK_SIZE=GLOBAL_BLOCK,
                num_warps=8,
            )

        # ----------------------------------------------------
        # Fuse all remaining:
        #
        #   step = local_log2 - 1
        #   ...
        #   step = 0
        #
        # into a single local sorting operation.
        #
        # At this point every LOCAL_BLOCK chunk owns the correct
        # set of elements; it only needs to be locally ordered.
        # ----------------------------------------------------

        local_sort_kernel[local_grid](
            data_ptr,
            N,
            BLOCK_SIZE=LOCAL_BLOCK,
            num_warps=8,
        )