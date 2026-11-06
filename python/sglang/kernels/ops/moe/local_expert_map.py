"""Fused logical->local expert-id remap for standard MoE dispatch.

``StandardDispatcher`` translates the router's global expert ids through a
per-rank lookup table so its runners only see local expert indices. The table
is int32, so writing that as ``table[topk_ids]`` costs two launches: ATen first
materialises an int64 copy of the int32 index tensor, then runs the index
kernel. One Triton program does both halves in place.
"""

import torch
import triton
import triton.language as tl

_BLOCK = 1024


@triton.jit
def _local_expert_lookup_kernel(
    ids_ptr,
    table_ptr,
    out_ptr,
    numel,
    table_numel,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    idx = tl.load(ids_ptr + offs, mask=mask, other=0)
    # ATen wraps a negative index by the table length, which is what makes the
    # -1 drop sentinel read the table's spare trailing slot.
    idx = tl.where(idx < 0, idx + table_numel, idx)
    in_range = (idx >= 0) & (idx < table_numel)
    val = tl.load(table_ptr + idx, mask=mask & in_range, other=-1)
    tl.store(out_ptr + offs, val, mask=mask)


def local_expert_lookup(topk_ids: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """Return ``table[topk_ids]`` as one launch, keeping ``table``'s dtype.

    ``table`` is indexed flat, so its element type decides the result: int32 in
    the dispatcher's case, matching what the eager gather produced. The kernel
    folds in ATen's int32->int64 index promotion, so the caller only pays for
    the gather itself.
    """
    if topk_ids.device != table.device:
        raise ValueError("topk_ids and the expert table must share a device")
    if topk_ids.dtype.is_floating_point or table.dtype.is_floating_point:
        raise TypeError("topk_ids and the expert table must be integer tensors")
    if not topk_ids.is_contiguous() or not table.is_contiguous():
        raise ValueError("topk_ids and the expert table must be contiguous")
    out = torch.empty_like(topk_ids, dtype=table.dtype)
    numel = topk_ids.numel()
    if numel == 0:
        return out
    table_numel = table.numel()
    if table_numel == 0:
        raise ValueError("the expert table must not be empty")
    # A fixed block keeps the decode shape from recompiling: only the grid
    # follows numel, and a CUDA-graph capture bakes in whichever grid it saw.
    _local_expert_lookup_kernel[(triton.cdiv(numel, _BLOCK),)](
        topk_ids,
        table,
        out,
        numel,
        table_numel,
        BLOCK=_BLOCK,
    )
    return out
