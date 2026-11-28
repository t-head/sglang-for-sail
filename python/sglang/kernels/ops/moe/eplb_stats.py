from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _add_to_row_kernel(
    data_ptr,
    src_ptr,
    layer_idx: int,
    num_experts: int,
    row_stride: int,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel to add src vector to a specific row of data tensor.

    Args:
        data_ptr: Pointer to the 2D data tensor (num_layers, num_experts)
        src_ptr: Pointer to the 1D source tensor (num_experts,)
        layer_idx: Index of the row to add to
        num_experts: Number of experts (columns)
        row_stride: Stride between rows
        BLOCK_SIZE: Number of elements processed per program instance
    """
    # Get program ID
    pid = tl.program_id(0)

    # Compute the starting offset for this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Create mask for bounds checking
    mask = offsets < num_experts

    # Compute pointers to data row and source
    row_ptr = data_ptr + layer_idx * row_stride
    data_ptrs = row_ptr + offsets
    src_ptrs = src_ptr + offsets

    # Load data and source with mask
    data_vals = tl.load(data_ptrs, mask=mask, other=0)
    src_vals = tl.load(src_ptrs, mask=mask, other=0)

    # Perform addition
    result = data_vals + src_vals

    # Store result with mask
    tl.store(data_ptrs, result, mask=mask)


def add_to_row(data: torch.Tensor, src: torch.Tensor, layer_idx: int) -> None:
    """Accumulate one contiguous count vector into a strided layer row."""
    n = src.numel()
    assert data.ndim == 2 and src.ndim == 1 and data.shape[1] >= n
    assert data.stride(1) == 1 and src.stride(0) == 1
    assert 0 <= layer_idx < data.shape[0]
    if n:
        _add_to_row_kernel[(triton.cdiv(n, 128),)](
            data, src, layer_idx, n, data.stride(0), BLOCK_SIZE=128, num_warps=4
        )
