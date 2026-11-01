# SPDX-License-Identifier: Apache-2.0
"""Paged Step5 scoring over existing SW128 E4M3 compressed_index_keys.
"""

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs


@triton.jit
def _e4m3_to_bf16(bits):
    """Decode E4M3FN bytes to BF16 with integer math (no native fp8 type).

    ``tl.float8e4nv`` is unavailable before sm89, so the portable Triton
    fallback reinterprets the SW128 compressed-index-key / query byte manually.
    The widening is exact: E4M3FN's 3 mantissa bits fit inside BF16's 7. The
    two NaN encodings (0x7F/0xFF) decode to +/-480, which never occurs for
    real index keys.
    """
    b = bits.to(tl.int32)
    sign = (b >> 7) & 1
    exp = (b >> 3) & 0xF
    man = (b & 0x7).to(tl.float32)
    mantissa = tl.where(exp == 0, man * 0.125, 1.0 + man * 0.125)
    exponent = tl.where(exp == 0, -6, exp - 7)
    scale = ((exponent + 127) << 23).to(tl.float32, bitcast=True)
    value = mantissa * scale
    value = tl.where(sign == 1, -value, value)
    return value.to(tl.bfloat16)


@triton.jit
def _decode_e4m3(bits, NATIVE_FP8: tl.constexpr):
    """E4M3FN byte to BF16: hardware widen on sm89+, integer math below it."""
    if NATIVE_FP8:
        return bits.to(tl.float8e4nv, bitcast=True).to(tl.bfloat16)
    return _e4m3_to_bf16(bits)


def _native_fp8_decode(device: torch.device) -> bool:
    """sm89+ widens E4M3 in hardware; sm80 PPU keeps the integer decoder."""
    return (
        device.type == "cuda"
        and torch.version.hip is None
        and torch.cuda.get_device_capability(device) >= (8, 9)
    )


@triton.jit
def _plus_zero(score):
    # FlashInfer's radix key orders floats by bit pattern, where -0.0 sorts below
    # +0.0; folding here keeps the tie-break on the region id without a copy pass.
    bits = score.to(tl.uint32, bitcast=True)
    bits = tl.where((bits & 0x7FFFFFFF) == 0, 0, bits)
    return bits.to(tl.float32, bitcast=True)


@triton.jit(
    do_not_specialize=("REGIONS",),
    do_not_specialize_on_alignment=("REGIONS",),
)
def _score_regions(
    Q,
    W,
    CompressedIndexKeys,
    ReqToToken,
    RowToReq,
    Positions,
    Out,
    HEADS: tl.constexpr,
    MAP_STRIDE: tl.constexpr,
    REGIONS,
    BLOCK_R: tl.constexpr,
    GRID_TILES: tl.constexpr,
    CLEAR_FUTURE: tl.constexpr,
    WEIGHT_SCALE: tl.constexpr,
    NATIVE_FP8: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    first_tile = tl.program_id(1)
    head = tl.arange(0, 16)
    channel = tl.arange(0, 256)
    request = tl.load(RowToReq + row).to(tl.int64)
    position = tl.load(Positions + row)
    query = _decode_e4m3(
        tl.load(
            Q + (row * HEADS + head[:, None]) * 256 + channel[None, :],
            (request > 0) & (head[:, None] < HEADS),
            other=0,
        ),
        NATIVE_FP8,
    )
    weights = (
        tl.load(W + row * HEADS + head, (request > 0) & (head < HEADS), other=0).to(
            tl.float32
        )
        * WEIGHT_SCALE
    )
    # A 1M graph capture has 4096 capacity tiles while a 60K request has about
    # 235 live tiles. The SM10x selector consumes positions directly, so its
    # path can stop at live history. The portable FlashInfer fallback still
    # overwrites future tiles to make a full-width top-k replay safe.
    last_tile = tl.cdiv(REGIONS, BLOCK_R)
    if not CLEAR_FUTURE:
        live_regions = tl.maximum(position // 8, 0)
        last_tile = tl.minimum(tl.cdiv(live_regions, BLOCK_R), last_tile)
        # Selecting every historical region does not consume region_scores.
        last_tile = tl.where(live_regions <= 512, 0, last_tile)
        last_tile = tl.where(request > 0, last_tile, 0)
    for tile in range(first_tile, last_tile, GRID_TILES):
        regions = tile * BLOCK_R + tl.arange(0, BLOCK_R)
        if (request > 0) & (tile * BLOCK_R < position // 8):
            valid = regions < position // 8
            token = tl.load(
                ReqToToken + request * MAP_STRIDE + regions * 8,
                valid,
                other=0,
            ).to(tl.int64)
            page, slot = token // 64, token % 64 // 8
            # Each page contains two 128-byte channel atoms, each with eight
            # swizzled region rows. XOR uses the physical row, never logical ID.
            offset = (
                page[None, :] * 2048
                + (channel[:, None] // 128) * 1024
                + slot[None, :] * 128
                + (((channel[:, None] % 128) // 16) ^ slot[None, :]) * 16
                + channel[:, None] % 16
            )
            packed = tl.load(CompressedIndexKeys + offset, valid[None, :], other=0)
            keys = _decode_e4m3(packed, NATIVE_FP8)
            products = tl.dot(query, keys)
            score = tl.sum(tl.maximum(products, 0.0) * weights[:, None], axis=0)
            score = tl.where(valid, score, -float("inf"))
        else:
            score = tl.full((BLOCK_R,), -float("inf"), tl.float32)
        tl.store(Out + row * REGIONS + regions, _plus_zero(score), regions < REGIONS)


# E4M3FN has exponent bias 7 while fp8e4b15, which this backend supports
# natively, has bias 15: the same byte read as b15 is exactly 2^-8 of its E4M3FN
# value (identical exponent field, 3-bit mantissa, denormal convention). So the
# raw bytes can be bitcast to fp8e4b15 and fed to the MMA, and the accumulator is
# rescaled by 2^16 to restore the value. The map is a power of two, hence exact,
# and it removes ~7 FP32 decode instructions per element.
FP8_RESCALE = tl.constexpr(65536.0)


@triton.jit
def _e4m3_bytes_to_fp8(bits):
    """E4M3FN raw bytes, read as fp8e4b15 worth 2^-8 of the original value."""
    return bits.to(tl.float8e4b15, bitcast=True)


@triton.jit
def _load_keys_tile(
    CompressedIndexKeys,
    ReqToToken,
    request,
    regions,
    valid,
    MAP_STRIDE: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """Load one (256, BLOCK_R) E4M3FN key tile from the SW128-swizzled layout.

    The swizzle splits the 256 channels into 16 segments of 16 bytes, and the
    bytes inside one segment are physically contiguous. Addressing them as
    (segment, region, byte-in-segment) lets Triton fuse each segment into a
    single vector load. Computing a flat (256, BLOCK_R) offset instead makes
    every lane fetch one byte: measured 4.5 bytes per sector, 3.87x load
    wavefront over-issue and 41% of the issued instructions on IALU, while the
    MMA itself is 2.5% of the 235us this takes at rows=32 / 16384 live regions.
    The 3-D form reaches 57 bytes per sector and a 1.91x wavefront.
    """
    token = tl.load(
        ReqToToken + request * MAP_STRIDE + regions * 8, valid, other=0
    ).to(tl.int32)
    page, slot = token >> 6, (token & 63) >> 3
    # Each page contains two 128-byte channel atoms, each with eight swizzled
    # region rows; the XOR uses the physical row, never the logical ID.
    seg = tl.arange(0, 16)
    base = (
        page[None, :] * 2048
        + (seg[:, None] >> 3) * 1024
        + slot[None, :] * 128
        + (((seg[:, None] & 7) ^ slot[None, :]) * 16)
    )
    packed = tl.load(
        CompressedIndexKeys + base[:, :, None] + tl.arange(0, 16)[None, None, :],
        valid[None, :, None],
        other=0,
    )
    # (segment, region, byte-in-segment) -> channel = segment * 16 + byte
    return tl.reshape(tl.permute(packed, (0, 2, 1)), (256, BLOCK_R))


@triton.jit(
    do_not_specialize=("ROWS", "REGIONS"),
    do_not_specialize_on_alignment=("ROWS", "REGIONS"),
)
def _score_regions_ppu_opt(
    Q,
    W,
    CompressedIndexKeys,
    ReqToToken,
    RowToReq,
    Positions,
    Out,
    SpanStart,
    SpanEnd,
    HEADS: tl.constexpr,
    ROWS,
    MAP_STRIDE: tl.constexpr,
    REGIONS,
    BLOCK_R: tl.constexpr,
    GRID_TILES: tl.constexpr,
    CLEAR_FUTURE: tl.constexpr,
    WEIGHT_SCALE: tl.constexpr,
    GROUP_ROWS: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    MIXED: tl.constexpr,
    HAS_SPANS: tl.constexpr,
):
    """Score GROUP_ROWS rows together so they share one key-tile gather.

    Three differences from the original ``_score_regions``:
    1. key/query bytes go straight into the fp8 MMA instead of an element-wise
       software E4M3 decode (half of the instructions on the old path);
    2. M = GROUP_ROWS * HEAD_BLOCK is filled with adjacent rows of the same
       request, so the key tile is reused inside the group and the gather count
       drops to 1/GROUP_ROWS;
    3. a tile that lies entirely past live skips gather/MMA; under CLEAR_FUTURE
       its -inf columns are pre-filled by ``_wipe_scores`` instead of being
       written full width by every tile.

    When the group spans several requests (a request boundary falls inside it)
    the kernel takes the per-row cold path. That path must store Out as soon as
    a row is done and must not accumulate a group-level (GROUP_ROWS, BLOCK_R)
    result in registers: doing so evicts the hot-path fp8 query tile and was
    measured to push the capture case from 5.9us to over 20us.
    """
    group = tl.program_id(0).to(tl.int64)
    first_tile = tl.program_id(1)
    row_index = tl.arange(0, GROUP_ROWS)
    if HAS_SPANS:
        # Spans are cut from the per-request packed lengths, so every span holds
        # rows of one request and its length never exceeds GROUP_ROWS.
        row_base = tl.load(SpanStart + group).to(tl.int64)
        row_limit = tl.load(SpanEnd + group).to(tl.int64)
    else:
        row_base = group * GROUP_ROWS
        row_limit = ROWS
    rows = row_base + row_index
    row_valid = rows < row_limit
    request = tl.load(RowToReq + rows, row_valid, other=0)
    position = tl.load(Positions + rows, row_valid, other=0)
    base_request = tl.max(request, axis=0)
    # Padding rows carry request slot zero, so they neither gather keys nor write
    # a live score; only the owned rows have to agree for the hot path.
    mixed = (request != base_request) & (request != 0) & row_valid
    uniform = tl.sum(mixed.to(tl.int32), axis=0) == 0

    matrix_row = tl.arange(0, GROUP_ROWS * HEAD_BLOCK)
    channel = tl.arange(0, 256)
    head_index = tl.arange(0, HEAD_BLOCK)
    query_row = row_base + matrix_row // HEAD_BLOCK
    head = matrix_row % HEAD_BLOCK
    query = _e4m3_bytes_to_fp8(
        tl.load(
            Q + (query_row[:, None] * HEADS + head[:, None]) * 256 + channel[None, :],
            (query_row[:, None] < row_limit) & (head[:, None] < HEADS),
            other=0,
        )
    )
    weights = (
        tl.load(
            W + rows[:, None] * HEADS + tl.arange(0, HEAD_BLOCK)[None, :],
            row_valid[:, None]
            & (tl.arange(0, HEAD_BLOCK)[None, :] < HEADS)
            & (request[:, None] > 0),
            other=0.0,
        ).to(tl.float32)
        * WEIGHT_SCALE
    )

    live = tl.where(row_valid & (request > 0), tl.maximum(position // 8, 0), 0)
    max_live = tl.max(live, axis=0)
    # A 1M graph capture has 4096 capacity tiles while a 60K request has about
    # 235 live tiles, so both modes stop at live history. The portable
    # FlashInfer fallback (CLEAR_FUTURE) needs -inf past live to make a
    # full-width top-k replay safe; ``launch`` covers that part of every row
    # with the flat `_wipe_scores` pass instead of a per-tile full-width store.
    last_tile = tl.minimum(tl.cdiv(max_live, BLOCK_R), tl.cdiv(REGIONS, BLOCK_R))
    if not CLEAR_FUTURE:
        # Selecting every historical region does not consume region_scores.
        last_tile = tl.where(max_live <= 512, 0, last_tile)
    for tile in range(first_tile, last_tile, GRID_TILES):
        regions = tile * BLOCK_R + tl.arange(0, BLOCK_R)
        in_range = regions < REGIONS
        # Under CLEAR_FUTURE the whole tile is written: lanes past live get -inf,
        # the same value ``_wipe_scores`` already laid down, so the duplicate
        # store is idempotent. Otherwise stop at the tile edge that holds this
        # row's live range, matching the original "never touch past live" extent
        # (live<=512 writes nothing).
        if CLEAR_FUTURE:
            written = in_range[None, :]
        else:
            row_tiles = tl.minimum(tl.cdiv(live, BLOCK_R), tl.cdiv(REGIONS, BLOCK_R))
            row_tiles = tl.where(live <= 512, 0, row_tiles)
            written = regions[None, :] < (row_tiles * BLOCK_R)[:, None]
        if uniform & (base_request > 0) & (tile * BLOCK_R < max_live):
            packed = _load_keys_tile(
                CompressedIndexKeys,
                ReqToToken,
                base_request,
                regions,
                in_range & (regions < max_live),
                MAP_STRIDE,
                BLOCK_R,
            )
            products = tl.dot(query, _e4m3_bytes_to_fp8(packed)) * FP8_RESCALE
            weighted = tl.maximum(products, 0.0) * tl.reshape(
                weights, (GROUP_ROWS * HEAD_BLOCK, 1)
            )
            score = tl.sum(tl.reshape(weighted, (GROUP_ROWS, HEAD_BLOCK, BLOCK_R)), axis=1)
            score = tl.where(regions[None, :] < live[:, None], score, -float("inf"))
            tl.store(
                Out + rows[:, None] * REGIONS + regions[None, :],
                _plus_zero(score),
                row_valid[:, None] & written,
            )
        elif MIXED:
            # The group spans several requests (a request boundary falls inside
            # it): the key tile cannot be shared, so do the work per row. Each
            # row stores Out as soon as it is done instead of accumulating a
            # group-level (G, B) result, which would evict the hot-path fp8
            # query tile and was measured to push the capture case from 5.9us to
            # over 20us. The cold path only keeps one row's live values alive.
            for r in tl.range(0, GROUP_ROWS, loop_unroll_factor=1):
                row = row_base + r
                row_ok = row < row_limit
                request_r = tl.load(RowToReq + row, row_ok, other=0)
                position_r = tl.load(Positions + row, row_ok, other=0)
                live_r = tl.maximum(position_r // 8, 0)
                if CLEAR_FUTURE:
                    row_written = in_range
                else:
                    row_tiles_r = tl.minimum(
                        tl.cdiv(live_r, BLOCK_R), tl.cdiv(REGIONS, BLOCK_R)
                    )
                    row_written = (
                        regions < tl.where(live_r <= 512, 0, row_tiles_r) * BLOCK_R
                    )
                score_r = tl.full((BLOCK_R,), -float("inf"), tl.float32)
                if (request_r > 0) & (tile * BLOCK_R < live_r):
                    query_r = _e4m3_bytes_to_fp8(
                        tl.load(
                            Q + (row * HEADS + head_index[:, None]) * 256 + channel[None, :],
                            row_ok & (head_index[:, None] < HEADS),
                            other=0,
                        )
                    )
                    weight_r = (
                        tl.load(
                            W + row * HEADS + head_index,
                            row_ok & (head_index < HEADS),
                            other=0.0,
                        )
                        * WEIGHT_SCALE
                    ).to(tl.float32)
                    packed_r = _load_keys_tile(
                        CompressedIndexKeys,
                        ReqToToken,
                        request_r,
                        regions,
                        in_range & (regions < live_r),
                        MAP_STRIDE,
                        BLOCK_R,
                    )
                    products_r = tl.dot(query_r, _e4m3_bytes_to_fp8(packed_r)) * FP8_RESCALE
                    score_r = tl.sum(tl.maximum(products_r, 0.0) * weight_r[:, None], axis=0)
                    score_r = tl.where(regions < live_r, score_r, -float("inf"))
                tl.store(
                    Out + row * REGIONS + regions,
                    _plus_zero(score_r),
                    row_ok & row_written,
                )
        else:
            score = tl.zeros((GROUP_ROWS, BLOCK_R), tl.float32)
            score = tl.where(regions[None, :] < live[:, None], score, -float("inf"))
            tl.store(
                Out + rows[:, None] * REGIONS + regions[None, :],
                _plus_zero(score),
                row_valid[:, None] & written,
            )


@triton.jit(
    do_not_specialize=("ROWS", "REGIONS"),
    do_not_specialize_on_alignment=("ROWS", "REGIONS"),
)
def _score_four_rows_per_request(
    Q,
    W,
    CompressedIndexKeys,
    ReqToToken,
    RowToReq,
    Positions,
    Out,
    HEADS: tl.constexpr,
    ROWS,
    MAP_STRIDE: tl.constexpr,
    REGIONS,
    BLOCK_R: tl.constexpr,
    GRID_TILES: tl.constexpr,
    WEIGHT_SCALE: tl.constexpr,
    NATIVE_FP8: tl.constexpr,
):
    """Fill four MTP verifier rows with one fully occupied 16x256 dot."""
    group = tl.program_id(0).to(tl.int64)
    first_tile = tl.program_id(1)
    group_rows = group * 4 + tl.arange(0, 4)
    matrix_row = tl.arange(0, 16)
    query_row = group * 4 + matrix_row // 4
    head = matrix_row % 4
    channel = tl.arange(0, 256)
    row_valid = group_rows < ROWS
    request = tl.load(RowToReq + group_rows, row_valid, other=0).to(tl.int64)
    position = tl.load(Positions + group_rows, row_valid, other=0)
    # Verify metadata guarantees four consecutive rows for one request. Padding
    # groups carry request slot zero in all four rows.
    base_request = tl.load(RowToReq + group * 4).to(tl.int64)
    query = _decode_e4m3(
        tl.load(
            Q + (query_row[:, None] * HEADS + head[:, None]) * 256 + channel[None, :],
            (query_row[:, None] < ROWS) & (head[:, None] < HEADS) & (base_request > 0),
            other=0,
        ),
        NATIVE_FP8,
    )
    weights = (
        tl.load(
            W + group_rows[:, None] * HEADS + tl.arange(0, 4)[None, :],
            row_valid[:, None]
            & (tl.arange(0, 4)[None, :] < HEADS)
            & (request[:, None] > 0),
            other=0.0,
        ).to(tl.float32)
        * WEIGHT_SCALE
    )
    live = tl.where(row_valid & (request > 0), tl.maximum(position // 8, 0), 0)
    max_live = tl.max(live, axis=0)
    last_tile = tl.minimum(tl.cdiv(max_live, BLOCK_R), tl.cdiv(REGIONS, BLOCK_R))
    last_tile = tl.where(max_live <= 512, 0, last_tile)
    for tile in range(first_tile, last_tile, GRID_TILES):
        regions = tile * BLOCK_R + tl.arange(0, BLOCK_R)
        valid = regions < max_live
        token = tl.load(
            ReqToToken + base_request * MAP_STRIDE + regions * 8,
            valid & (base_request > 0),
            other=0,
        ).to(tl.int64)
        page, slot = token // 64, token % 64 // 8
        offset = (
            page[None, :] * 2048
            + (channel[:, None] // 128) * 1024
            + slot[None, :] * 128
            + (((channel[:, None] % 128) // 16) ^ slot[None, :]) * 16
            + channel[:, None] % 16
        )
        packed = tl.load(CompressedIndexKeys + offset, valid[None, :], other=0)
        keys = _decode_e4m3(packed, NATIVE_FP8)
        products = tl.dot(query, keys)
        weighted = tl.maximum(products, 0.0) * tl.reshape(weights, (16, 1))
        region_scores = tl.sum(tl.reshape(weighted, (4, 4, BLOCK_R)), axis=1)
        output_valid = row_valid[:, None] & (regions[None, :] < live[:, None])
        tl.store(
            Out + group_rows[:, None] * REGIONS + regions[None, :],
            _plus_zero(region_scores),
            output_valid,
        )


@triton.jit(
    do_not_specialize=("ROWS", "REGIONS"),
    do_not_specialize_on_alignment=("ROWS", "REGIONS"),
)
def _score_thirty_two_rows_per_request(
    Q,
    W,
    CompressedIndexKeys,
    ReqToToken,
    RowToReq,
    Positions,
    Out,
    HEADS: tl.constexpr,
    ROWS,
    MAP_STRIDE: tl.constexpr,
    REGIONS,
    BLOCK_R: tl.constexpr,
    GRID_TILES: tl.constexpr,
    WEIGHT_SCALE: tl.constexpr,
    NATIVE_FP8: tl.constexpr,
):
    group = tl.program_id(0).to(tl.int64)
    first_tile = tl.program_id(1)
    group_base = group * 32
    channel = tl.arange(0, 256)
    matrix_row = tl.arange(0, 32)
    head = matrix_row % 4
    row_in_subgroup = tl.arange(0, 8)
    for subgroup in tl.static_range(0, 4):
        rows = group_base + subgroup * 8 + row_in_subgroup
        row_valid = rows < ROWS
        request = tl.load(RowToReq + rows, row_valid, other=0).to(tl.int64)
        position = tl.load(Positions + rows, row_valid, other=0)
        base_request = tl.load(RowToReq + group_base).to(tl.int64)
        query_row = group_base + subgroup * 8 + matrix_row // 4
        query = _decode_e4m3(
            tl.load(
                Q + (query_row[:, None] * HEADS + head[:, None]) * 256 + channel[None, :],
                (query_row[:, None] < ROWS) & (head[:, None] < HEADS) & (base_request > 0),
                other=0,
            ),
            NATIVE_FP8,
        )
        weights = (
            tl.load(
                W + rows[:, None] * HEADS + tl.arange(0, 4)[None, :],
                row_valid[:, None]
                & (tl.arange(0, 4)[None, :] < HEADS)
                & (request[:, None] > 0),
                other=0.0,
            ).to(tl.float32)
            * WEIGHT_SCALE
        )
        live = tl.where(row_valid & (request > 0), tl.maximum(position // 8, 0), 0)
        max_live = tl.max(live, axis=0)
        last_tile = tl.minimum(tl.cdiv(max_live, BLOCK_R), tl.cdiv(REGIONS, BLOCK_R))
        last_tile = tl.where(max_live <= 512, 0, last_tile)
        for tile in range(first_tile, last_tile, GRID_TILES):
            regions = tile * BLOCK_R + tl.arange(0, BLOCK_R)
            valid = regions < max_live
            token = tl.load(
                ReqToToken + base_request * MAP_STRIDE + regions * 8,
                valid & (base_request > 0),
                other=0,
            ).to(tl.int64)
            page, slot = token // 64, token % 64 // 8
            offset = (
                page[None, :] * 2048
                + (channel[:, None] // 128) * 1024
                + slot[None, :] * 128
                + (((channel[:, None] % 128) // 16) ^ slot[None, :]) * 16
                + channel[:, None] % 16
            )
            packed = tl.load(CompressedIndexKeys + offset, valid[None, :], other=0)
            keys = _decode_e4m3(packed, NATIVE_FP8)
            products = tl.dot(query, keys)
            weighted = tl.maximum(products, 0.0) * tl.reshape(weights, (32, 1))
            # Preserve group1's 16-lane reduction; compact reduction changes rounding.
            heads = tl.permute(tl.reshape(weighted, (8, 4, BLOCK_R)), 0, 2, 1)
            padded = tl.join(heads, tl.zeros_like(heads))
            padded = tl.reshape(tl.permute(padded, 0, 1, 3, 2), (8, BLOCK_R, 8))
            padded = tl.join(padded, tl.zeros_like(padded))
            padded = tl.reshape(tl.permute(padded, 0, 1, 3, 2), (8, BLOCK_R, 16))
            region_scores = tl.sum(padded, axis=2)
            output_valid = row_valid[:, None] & (regions[None, :] < live[:, None])
            tl.store(
                Out + rows[:, None] * REGIONS + regions[None, :],
                _plus_zero(region_scores),
                output_valid,
            )


def _group_plan(rows, group_rows):
    """How many adjacent rows to fold into one MMA.

    ``group_rows`` is the caller's hint for how many rows one request owns (a
    prefill chunk, the 4 rows of an MTP verify); only a divisor of it keeps a
    group inside a single request. A group may not be wider than the row count,
    but there is no need to collect N full groups: grouping only narrows the
    first grid dimension, while the second one, ``grid_tiles`` (capped at 128),
    already fills the machine. Measured: decode at batch 1~3 (rows=4/8/12) is
    faster with the full group=4 (147us -> 47us at 16384 live regions) because
    the key-tile gather count drops to a quarter.
    """
    want = 1
    if group_rows and group_rows > 1:
        for cand in (8, 4, 2):
            if group_rows % cand == 0:
                want = cand
                break
    while want > 1 and rows < want:
        want //= 2
    return want


# Upper bound on the second grid dimension: when REGIONS is huge (a 1M graph
# capture has 4096 region tiles) one CTA per tile is unnecessary; 128 of them
# already fill the machine and the remaining tiles are looped inside the kernel.
_MAX_GRID_TILES = 128


# region_scores comes from torch.empty and the portable selector runs top-k over
# the whole row, so columns past live must be -inf. Keeping that sweep inside the
# scoring kernel forces stores no wider than BLOCK_R elements: measured 425us out
# of the 441us it takes for (16, 131080) went into laying down -inf. As its own
# flat kernel the same 8MB write costs ~4us, because every store is a 16B vector
# (BLOCK=4096 / 4 warps, measured 3.9us, 2.2TB/s, on par with torch fill_).
_WIPE_BLOCK = 4096


@triton.jit(
    do_not_specialize=("TOTAL",),
    do_not_specialize_on_alignment=("TOTAL",),
)
def _wipe_scores(Out, TOTAL, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(
        Out + offsets,
        tl.full((BLOCK,), -float("inf"), tl.float32),
        offsets < TOTAL,
    )


def _wipe_region_scores(out):
    """Fill region_scores with -inf; the live part is overwritten afterwards."""
    total = out.numel()
    _wipe_scores[(triton.cdiv(total, _WIPE_BLOCK),)](
        out, total, _WIPE_BLOCK, num_warps=4
    )


def _launch_reference(
    q,
    weights,
    compressed_index_keys,
    req_to_token,
    row_to_req,
    positions,
    out,
    *,
    clear_future,
    group_rows,
    weight_scale,
):
    """ `SGLANG_STEP5_SCORING_REGION_OPT=0` fallback to score_region"""
    # D256 gathers already consume substantial registers. A 32-region tile
    # reduces that pressure and gives small batches more independent CTAs.
    block_regions = 32
    grid_tiles = min(triton.cdiv(out.shape[1], block_regions), 128)
    native_fp8 = _native_fp8_decode(q.device)
    if group_rows == 32 and not clear_future:
        _score_thirty_two_rows_per_request[(triton.cdiv(q.shape[0], 32), grid_tiles)](
            q,
            weights,
            compressed_index_keys,
            req_to_token,
            row_to_req,
            positions,
            out,
            q.shape[1],
            q.shape[0],
            req_to_token.stride(0),
            out.shape[1],
            block_regions,
            grid_tiles,
            weight_scale,
            native_fp8,
            num_warps=4,
        )
        return
    if group_rows == 4 and not clear_future:
        if q.shape[0] <= 16 and torch.cuda.get_device_capability(q.device) == (10, 3):
            if q.shape[0] == 4:
                block_regions = 16
                grid_tiles = min(triton.cdiv(out.shape[1], block_regions), 1024)
            else:
                grid_tiles = min(triton.cdiv(out.shape[1], block_regions), 512)

        _score_four_rows_per_request[(triton.cdiv(q.shape[0], 4), grid_tiles)](
            q,
            weights,
            compressed_index_keys,
            req_to_token,
            row_to_req,
            positions,
            out,
            q.shape[1],
            q.shape[0],
            req_to_token.stride(0),
            out.shape[1],
            block_regions,
            grid_tiles,
            weight_scale,
            native_fp8,
            num_warps=4,
        )
        return
    _score_regions[(q.shape[0], grid_tiles)](
        q,
        weights,
        compressed_index_keys,
        req_to_token,
        row_to_req,
        positions,
        out,
        q.shape[1],
        req_to_token.stride(0),
        out.shape[1],
        block_regions,
        grid_tiles,
        clear_future,
        weight_scale,
        native_fp8,
        num_warps=4,
    )


def _launch_spans(
    q,
    weights,
    compressed_index_keys,
    req_to_token,
    row_to_req,
    positions,
    out,
    *,
    score_groups,
    clear_future,
    weight_scale,
):
    """Score one request's row span per CTA so ragged batches group as well.

    Spans are cut from the packed lengths, so the hot path always applies; the
    per-row cold path stays compiled as the guard against a mis-cut span.
    """
    starts, ends, span_rows = score_groups
    regions = out.shape[1]
    heads = q.shape[1]
    block_regions = 8
    grid_tiles = min(triton.cdiv(regions, block_regions), _MAX_GRID_TILES)
    _score_regions_ppu_opt[(starts.numel(), grid_tiles)](
        q,
        weights,
        compressed_index_keys,
        req_to_token,
        row_to_req,
        positions,
        out,
        starts,
        ends,
        heads,
        q.shape[0],
        req_to_token.stride(0),
        regions,
        block_regions,
        grid_tiles,
        clear_future,
        weight_scale,
        span_rows,
        max(4, heads + (-heads % 4)),
        True,
        True,
        num_warps=2,
    )


def launch(
    q,
    weights,
    compressed_index_keys,
    req_to_token,
    row_to_req,
    positions,
    out,
    *,
    clear_future,
    group_rows=1,
    weight_scale=1.0,
    score_groups=None,
):
    if not envs.SGLANG_STEP5_SCORING_REGION_OPT.get():
        _launch_reference(
            q,
            weights,
            compressed_index_keys,
            req_to_token,
            row_to_req,
            positions,
            out,
            clear_future=clear_future,
            group_rows=group_rows,
            weight_scale=weight_scale,
        )
        return
    rows = q.shape[0]
    regions = out.shape[1]
    heads = q.shape[1]
    # ``clear_future`` means the portable selector reads the whole row, so future
    # columns must be -inf. One flat fill replaces the per-tile full-width store
    # of the scoring kernel (the values written are identical).
    if clear_future and rows:
        _wipe_region_scores(out)
    if score_groups is not None:
        _launch_spans(
            q,
            weights,
            compressed_index_keys,
            req_to_token,
            row_to_req,
            positions,
            out,
            score_groups=score_groups,
            clear_future=clear_future,
            weight_scale=weight_scale,
        )
        return
    group = _group_plan(rows, group_rows)
    # Now that the key tile is addressed as (segment, region, byte-in-segment),
    # widening BLOCK_R no longer inflates the address math; instead it lifts the
    # per-row score store from 32B to 128B and cuts the tile count by 4x (16384
    # live regions: rows=32 goes from 223us to 85us of kernel time).
    # A single scoring grid does not fill the CUs with one warp per CTA, and two
    # warps are never slower across the rows/group combinations (a tie at
    # group=1); four warps lose to register spilling.
    block_regions = 32
    num_warps = 2
    grid_tiles = min(triton.cdiv(regions, block_regions), _MAX_GRID_TILES)
    head_block = max(4, heads + (-heads % 4))
    _score_regions_ppu_opt[(triton.cdiv(rows, group), grid_tiles)](
        q,
        weights,
        compressed_index_keys,
        req_to_token,
        row_to_req,
        positions,
        out,
        # HAS_SPANS=False: the span pointers are never loaded, so the position
        # tensor stands in for both and keeps the argument type int32*.
        positions,
        positions,
        heads,
        rows,
        req_to_token.stride(0),
        regions,
        block_regions,
        grid_tiles,
        clear_future,
        weight_scale,
        group,
        head_block,
        True,
        False,
        num_warps=num_warps,
    )
