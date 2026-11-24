from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _pack_ragged_scores(
    scores,
    lengths,
    row_starts,
    packed,
    SCORE_ROW_STRIDE: tl.constexpr,
    SCORE_COL_STRIDE: tl.constexpr,
    PACKED_ROW_STRIDE: tl.constexpr,
    HAS_STARTS: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    length = tl.load(lengths + row)
    # Indices-only DeepSelect does not read scores for rows shorter than topk.
    if length > TOPK:
        start = tl.load(row_starts + row) if HAS_STARTS else 0
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        values = tl.load(
            scores + row * SCORE_ROW_STRIDE + (start + cols) * SCORE_COL_STRIDE,
            mask=cols < length,
            other=0,
        )
        tl.store(packed + row * PACKED_ROW_STRIDE + cols, values, mask=cols < length)


@triton.jit(do_not_specialize=["SCORE_ROW_STRIDE", "PACKED_ROW_STRIDE"])
def _pack_ragged_scores_runtime(
    scores,
    lengths,
    row_starts,
    packed,
    SCORE_ROW_STRIDE,
    SCORE_COL_STRIDE: tl.constexpr,
    PACKED_ROW_STRIDE,
    HAS_STARTS: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
    ALIGNED_SCORE_ROWS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    length = tl.load(lengths + row)
    if length > TOPK:
        start = tl.load(row_starts + row) if HAS_STARTS else 0
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        score_stride = SCORE_ROW_STRIDE
        if ALIGNED_SCORE_ROWS:
            score_stride = tl.multiple_of(score_stride, 512)
        packed_stride = tl.multiple_of(PACKED_ROW_STRIDE, 512)
        values = tl.load(
            scores + row * score_stride + (start + cols) * SCORE_COL_STRIDE,
            mask=cols < length,
            other=0,
        )
        tl.store(packed + row * packed_stride + cols, values, mask=cols < length)


_WARMED_RAGGED_PACKERS: set[tuple[torch.device, int]] = set()


def _warm_ragged_packers(scores: torch.Tensor, topk: int) -> None:
    key = (scores.device, topk)
    if key in _WARMED_RAGGED_PACKERS:
        return
    # Seed the finite generic variants before the first dynamic business shape.
    # Run normal warmup before CUDA Graph capture, as for the selector itself.
    source = scores.new_zeros((4, 8192))
    destination = torch.empty_like(source)
    lengths = torch.full((4,), 8192, device=scores.device, dtype=torch.int32)
    starts = torch.zeros((4,), device=scores.device, dtype=torch.int32)
    for block in (1024, 2048):
        for has_starts in (False, True):
            _pack_ragged_scores_runtime[(4, triton.cdiv(8192, block))](
                source,
                lengths,
                starts if has_starts else None,
                destination,
                8192,
                1,
                8192,
                has_starts,
                topk,
                block,
                True,
                num_warps=4,
            )
    _WARMED_RAGGED_PACKERS.add(key)


def topk_transform_ragged(
    scores: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    out_offsets: torch.Tensor,
    row_starts: Optional[torch.Tensor] = None,
    *,
    max_seq_len: Optional[int] = None,
) -> torch.Tensor:
    """PPU DeepSelect over row windows, returning rebased int32 indices and -1 padding.

    The wheel supports end lengths but not begin offsets. Pack nontrivial
    windows into BF16 storage when needed; short rows need only their indices.
    No FP32 score matrix is materialized and the source scores are unchanged.
    ``max_seq_len`` is an optional host-known upper bound on per-row lengths,
    allowing packed storage to omit the other requests' columns.
    """
    import deep_select

    rows, cols = scores.shape
    if rows == 0:
        return scores.new_empty((0, topk), dtype=torch.int32)

    _warm_ragged_packers(scores, topk)
    lengths = lengths.contiguous()
    out_offsets = out_offsets.contiguous()
    alignment = deep_select.get_stride_requirement()[0]
    needs_pack = (
        (row_starts is not None and cols > topk)
        or scores.stride(1) != 1
        or scores.stride(0) * scores.element_size() % alignment != 0
        or scores.data_ptr() % 16 != 0
    )
    if needs_pack:
        packed_cols = cols if max_seq_len is None else min(cols, max_seq_len)
        aligned_cols = triton.cdiv(packed_cols, alignment // scores.element_size()) * (
            alignment // scores.element_size()
        )
        packed = scores.new_empty((rows, aligned_cols))
        if packed_cols > topk:
            starts = row_starts.contiguous() if row_starts is not None else None
            block = 2048 if max_seq_len is not None and packed_cols >= 4096 else 1024
            if packed.stride(0) % 512 == 0:
                _pack_ragged_scores_runtime[(rows, triton.cdiv(packed_cols, block))](
                    scores,
                    lengths,
                    starts,
                    packed,
                    scores.stride(0),
                    scores.stride(1),
                    packed.stride(0),
                    starts is not None,
                    topk,
                    block,
                    scores.stride(0) % 512 == 0,
                    num_warps=4,
                )
            else:
                # Keep the general layout path if a future wheel changes alignment.
                _pack_ragged_scores[(rows, triton.cdiv(packed_cols, block))](
                    scores,
                    lengths,
                    starts,
                    packed,
                    scores.stride(0),
                    scores.stride(1),
                    packed.stride(0),
                    starts is not None,
                    topk,
                    block,
                    num_warps=4,
                )
        scores = packed[:, :packed_cols]

    _, indices = deep_select.topk(
        scores,
        topk,
        end=lengths,
        indices_type=torch.int32,
        output_idx_offset=out_offsets,
        idx_oob_fill_value=-1,
        return_value=False,
    )
    return indices.contiguous()
