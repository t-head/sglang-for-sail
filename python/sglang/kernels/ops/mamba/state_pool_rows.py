"""Masked indexed state writes, including slot zero and strided matrix entries."""

import torch
import triton
import triton.language as tl


@triton.jit
def _scatter_state_pool_rows(
    dst,
    src,
    indices,
    index_stride,
    dst_slot_stride,
    dst_head_stride,
    dst_v_stride,
    dst_k_stride,
    slots,
    ROW_SIZE: tl.constexpr,
    V: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.load(indices + row * index_stride).to(tl.int64)
    # Slot zero is valid in the release/v0.5.23 state-pool contract.
    if slot < 0:
        return
    if slot >= slots:
        return
    offset = (tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    head = offset // (V * K)
    v = (offset // K) % V
    k = offset % K
    value = tl.load(src + row * ROW_SIZE + offset, offset < ROW_SIZE, other=0.0)
    target = (
        slot * dst_slot_stride
        + head * dst_head_stride
        + v * dst_v_stride
        + k * dst_k_stride
    )
    tl.store(dst + target, value, offset < ROW_SIZE)


def scatter_state_pool_rows_(
    dst: torch.Tensor, src: torch.Tensor, indices: torch.Tensor
):
    """Write each dense scratch row to its nonnegative pool index on device.

    Live destination indices belong to the caller's pool and are unique. Only
    negative padding rows are skipped; no boolean compaction or host read occurs.
    The pool's logical [slot,H,V,K] entries may have arbitrary positive strides.
    """
    if dst.ndim != 4 or src.ndim != 4 or dst.shape[1:] != src.shape[1:]:
        raise ValueError("state rows must have matching [H,V,K] dimensions")
    if indices.ndim != 1 or indices.numel() != src.shape[0]:
        raise ValueError("one destination index is required for every source row")
    if indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("state indices must be int32 or int64")
    if not dst.is_cuda or dst.device != src.device or dst.device != indices.device:
        raise ValueError("state rows and indices must share a CUDA/SAIL device")
    if dst.dtype != src.dtype or not src.is_contiguous():
        raise ValueError("scratch rows must be contiguous and match pool dtype")
    rows = src.shape[0]
    size = src.shape[1] * src.shape[2] * src.shape[3]
    if rows == 0 or size == 0:
        return
    _scatter_state_pool_rows[(rows, triton.cdiv(size, 1024))](
        dst,
        src,
        indices,
        indices.stride(0),
        *dst.stride(),
        dst.shape[0],
        ROW_SIZE=size,
        V=src.shape[2],
        K=src.shape[3],
        BLOCK=1024,
        num_warps=4,
    )
