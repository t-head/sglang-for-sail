# SPDX-License-Identifier: Apache-2.0
"""Contiguous-K scoring for valid request spans from step5_paged_scores on 890P."""

import triton
import triton.language as tl

from sglang.kernels.ops.attention.step5.scoring_triton import (
    _decode_e4m3,
    _e4m3_bytes_to_fp8,
    _early_exit_worthwhile,
    _live_extent,
    _load_keys_tile,
    _plus_zero,
    _wipe_region_scores,
)


@triton.jit
def _pack_keys(
    Keys, Map, RowToReq, Positions, RequestLastRows, Compact,
    MAP_STRIDE: tl.constexpr, REGIONS: tl.constexpr, FP8_INDEXER: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    compact_request = tl.program_id(0)
    row = tl.load(RequestLastRows + compact_request).to(tl.int64)
    request = tl.load(RowToReq + row).to(tl.int64)
    live = tl.maximum(tl.load(Positions + row) // 8, 0)
    regions = tl.program_id(1) * BLOCK_R + tl.arange(0, BLOCK_R)
    valid = (request > 0) & (regions < live) & (regions < REGIONS)
    packed = _load_keys_tile(Keys, Map, request, regions, valid, MAP_STRIDE, BLOCK_R)
    if FP8_INDEXER:
        values = packed
    else:
        values = _decode_e4m3(packed, True)
    channel = tl.arange(0, 256)
    # compact_request is forward-local, not a potentially large request-pool slot.
    tl.store(
        Compact + (compact_request * REGIONS + regions[:, None]) * 256
        + channel[None, :],
        tl.trans(values), regions[:, None] < REGIONS,
    )


@triton.jit
def _compact_products(
    query, Compact, compact_request, regions, valid,
    REGIONS: tl.constexpr, BLOCK_R: tl.constexpr, FP8_INDEXER: tl.constexpr,
):
    channel = tl.arange(0, 256)
    values = tl.load(
        Compact + (compact_request * REGIONS + regions[:, None]) * 256
        + channel[None, :],
        valid[:, None], other=0,
    )
    if FP8_INDEXER:
        keys = _e4m3_bytes_to_fp8(tl.trans(values), True)
    else:
        keys = tl.trans(values)
    return tl.dot(query, keys)


@triton.jit
def _score_compact(
    Q, W, Compact, RowToReq, Positions, Out, SpanStart, SpanEnd, SpanRequest,
    HEADS: tl.constexpr, REGIONS: tl.constexpr, BLOCK_R: tl.constexpr,
    GRID_TILES: tl.constexpr, GROUP_ROWS: tl.constexpr,
    EARLY_EXIT: tl.constexpr, FP8_INDEXER: tl.constexpr, WEIGHT_SCALE: tl.constexpr,
):
    group = tl.program_id(0)
    first_tile = tl.program_id(1)
    row_base = tl.load(SpanStart + group).to(tl.int64)
    row_limit = tl.load(SpanEnd + group).to(tl.int64)
    compact_request = tl.load(SpanRequest + group).to(tl.int64)
    rows = row_base + tl.arange(0, GROUP_ROWS)
    row_valid = rows < row_limit
    request = tl.load(RowToReq + rows, row_valid, other=0)
    position = tl.load(Positions + rows, row_valid, other=0)
    base_request = tl.max(request, axis=0)
    mixed = (request != base_request) & (request != 0) & row_valid
    uniform = tl.sum(mixed.to(tl.int32), axis=0) == 0
    if EARLY_EXIT:
        live, max_live, last_tile = _live_extent(
            position, row_valid, request, BLOCK_R, REGIONS, True,
        )
        if first_tile >= last_tile:
            return
    matrix_row = tl.arange(0, GROUP_ROWS * 4)
    channel = tl.arange(0, 256)
    query_row = row_base + matrix_row // 4
    head = matrix_row % 4
    bits = tl.load(
        Q + (query_row[:, None] * HEADS + head[:, None]) * 256 + channel[None, :],
        (query_row[:, None] < row_limit) & (head[:, None] < HEADS), other=0,
    )
    if FP8_INDEXER:
        query = _e4m3_bytes_to_fp8(bits, True)
    else:
        query = _decode_e4m3(bits, True)
    weights = tl.load(
        W + rows[:, None] * HEADS + tl.arange(0, 4)[None, :],
        row_valid[:, None] & (tl.arange(0, 4)[None, :] < HEADS)
        & (request[:, None] > 0), other=0.0,
    ).to(tl.float32) * WEIGHT_SCALE
    if not EARLY_EXIT:
        live, max_live, last_tile = _live_extent(
            position, row_valid, request, BLOCK_R, REGIONS, True,
        )
    for tile in range(first_tile, last_tile, GRID_TILES):
        regions = tile * BLOCK_R + tl.arange(0, BLOCK_R)
        in_range = regions < REGIONS
        if uniform & (base_request > 0) & (tile * BLOCK_R < max_live):
            products = _compact_products(
                query, Compact, compact_request, regions,
                in_range & (regions < max_live), REGIONS, BLOCK_R, FP8_INDEXER,
            )
            weighted = tl.maximum(products, 0.0) * tl.reshape(
                weights, (GROUP_ROWS * 4, 1),
            )
            score = tl.sum(
                tl.reshape(weighted, (GROUP_ROWS, 4, BLOCK_R)), axis=1,
            )
            score = tl.where(regions[None, :] < live[:, None], score, -float("inf"))
            tl.store(
                Out + rows[:, None] * REGIONS + regions[None, :],
                _plus_zero(score), row_valid[:, None] & in_range[None, :],
            )


def launch(
    q, weights, compressed_index_keys, req_to_token, row_to_req, positions, out,
    *, workspace, weight_scale,
):
    regions = out.shape[1]
    block_regions = 32
    grid_tiles = min(triton.cdiv(regions, block_regions), 32)
    _wipe_region_scores(out)
    # CSA has published this layer's K; keep packing inside the layer loop and graph.
    _pack_keys[(workspace.keys.shape[0], triton.cdiv(regions, block_regions))](
        compressed_index_keys, req_to_token, row_to_req, positions,
        workspace.request_last_rows, workspace.keys, req_to_token.stride(0),
        regions, workspace.fp8, block_regions, num_warps=4,
    )
    _score_compact[(workspace.span_start.numel(), grid_tiles)](
        q, weights, workspace.keys, row_to_req, positions, out,
        workspace.span_start, workspace.span_end, workspace.span_requests,
        q.shape[1], regions, block_regions, grid_tiles, workspace.group_rows,
        _early_exit_worthwhile(q.shape[0], block_regions, grid_tiles),
        workspace.fp8, weight_scale, num_warps=2,
    )
