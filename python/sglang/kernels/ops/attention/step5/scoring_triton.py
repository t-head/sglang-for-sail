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
        tl.store(Out + row * REGIONS + regions, score, regions < REGIONS)


# E4M3FN 的指数偏置是 7，本后端原生支持的 fp8e4b15 偏置是 15 —— 同一个字节在
# b15 下的数值恰好是 E4M3FN 的 2^-8（阶码、3 位尾数、非规格化约定完全一致）。
# 于是可以直接把原始字节 bitcast 成 fp8e4b15 送 MMA，再把累加结果乘 2^16 还原，
# 数值恒等（map 是 2 的幂，不引入任何舍入），却省掉了每个元素 7 条 FP32 解码指令。
FP8_RESCALE = tl.constexpr(65536.0)


@triton.jit
def _e4m3_bytes_to_fp8(bits):
    """E4M3FN 原始字节 → 数值等于原值的 2^-8 的 fp8e4b15。"""
    return bits.to(tl.float8e4b15, bitcast=True)


@triton.jit
def _load_keys_tile(
    CompressedIndexKeys,
    ReqToToken,
    request,
    regions,
    channel,
    valid,
    MAP_STRIDE: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """取一个 (256, BLOCK_R) 的 E4M3FN 键 tile（SW128 swizzle 布局）。"""
    token = tl.load(
        ReqToToken + request * MAP_STRIDE + regions * 8, valid, other=0
    ).to(tl.int32)
    # int32 偏移：tile 内地址只跨 page，2^31 字节以内，省掉一半整型指令。
    page, slot = token >> 6, (token & 63) >> 3
    ch = channel[:, None]
    # Each page contains two 128-byte channel atoms, each with eight
    # swizzled region rows. XOR uses the physical row, never logical ID.
    offset = (
        page[None, :] * 2048
        + (ch >> 7) * 1024
        + slot[None, :] * 128
        + (((ch & 127) >> 4) ^ slot[None, :]) * 16
        + (ch & 15)
    )
    return tl.load(CompressedIndexKeys + offset, valid[None, :], other=0)


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
    """GROUP_ROWS 行一起算：同一个 request 下的多行共用一次键 tile gather。

    与原 ``_score_regions`` 的三个差别：
    1. 键/查询字节直接送 fp8 MMA，不再逐元素软件解码 E4M3（原路径一半的指令）；
    2. M = GROUP_ROWS * HEAD_BLOCK 由同一 request 的相邻行拼满，键 tile 在组内
       复用，gather 次数降到 1/GROUP_ROWS；
    3. 整个 tile 落在 live 之外时直接跳过 gather/MMA，只写 -inf。

    组内 request 不一致（request 边界落在组内）时走逐行冷路径。冷路径必须「每行
    算完立刻写 Out」，不能在寄存器里累积组级 (GROUP_ROWS, BLOCK_R) 结果：那样会把
    热路径的 fp8 query tile 挤出寄存器，实测让 capture 用例从 5.9us 退化到 20us 以上。
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
    # 235 live tiles. The SM10x selector consumes positions directly, so its
    # path can stop at live history. The portable FlashInfer fallback still
    # overwrites future tiles to make a full-width top-k replay safe.
    if CLEAR_FUTURE:
        last_tile = tl.cdiv(REGIONS, BLOCK_R)
    else:
        last_tile = tl.minimum(tl.cdiv(max_live, BLOCK_R), tl.cdiv(REGIONS, BLOCK_R))
        # Selecting every historical region does not consume region_scores.
        last_tile = tl.where(max_live <= 512, 0, last_tile)
    for tile in range(first_tile, last_tile, GRID_TILES):
        regions = tile * BLOCK_R + tl.arange(0, BLOCK_R)
        in_range = regions < REGIONS
        # CLEAR_FUTURE 下要覆盖满宽；否则只写到本行 live 所在的 tile 边界为止，
        # 与原实现「不碰 live 之后」的写入范围保持一致（live<=512 一行都不写）。
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
                channel,
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
                score,
                row_valid[:, None] & written,
            )
        elif MIXED:
            # 组内跨 request（request 边界落在组内）：键 tile 没法在组内复用，只能
            # 按行做。这里刻意让每行算完立刻写 Out，不在 kernel 里累积组级 (G,B)
            # 结果：那样会把热路径的 fp8 query tile 挤出寄存器，实测 capture 从
            # 5.9us 退化到 20us 以上。冷路径只占用单行的活跃值。
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
                        channel,
                        in_range & (regions < live_r),
                        MAP_STRIDE,
                        BLOCK_R,
                    )
                    products_r = tl.dot(query_r, _e4m3_bytes_to_fp8(packed_r)) * FP8_RESCALE
                    score_r = tl.sum(tl.maximum(products_r, 0.0) * weight_r[:, None], axis=0)
                    score_r = tl.where(regions < live_r, score_r, -float("inf"))
                tl.store(Out + row * REGIONS + regions, score_r, row_ok & row_written)
        else:
            score = tl.zeros((GROUP_ROWS, BLOCK_R), tl.float32)
            score = tl.where(regions[None, :] < live[:, None], score, -float("inf"))
            tl.store(
                Out + rows[:, None] * REGIONS + regions[None, :],
                score,
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
            region_scores,
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
                region_scores,
                output_valid,
            )


def _group_plan(rows, group_rows):
    """每多少个相邻行拼一次 MMA。

    ``group_rows`` 是调用方给出的「一个 request 占多少行」提示（prefill 的 chunk、
    MTP 校验的 4 行），只有它是候选值的整数倍时才能保证组内同 request。行数太少时
    分组会带来大量空转的 CTA，所以要求至少能凑够 8 个满组。
    """
    want = 1
    if group_rows and group_rows > 1:
        for cand in (8, 4, 2):
            if group_rows % cand == 0:
                want = cand
                break
    while want > 1 and rows // want < 8:
        want //= 2
    return want


# grid 第二维上限：REGIONS 很大时（1M graph capture 有 4096 个 region）不需要每 tile
# 一个 CTA，留下 128 个足以填满机器，其余 tile 在 kernel 内循环。
_MAX_GRID_TILES = 128


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
    # 分组时 M = group*head_block 已经够宽，BLOCK_R 取 8 摊薄 gather 的固定开销；
    # 每组只有一行时 gather 无法复用，BLOCK_R=4 + 单 warp 的 CTA 吞吐更好。
    block_regions = 4 if group == 1 else 8
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
        num_warps=1 if group == 1 else 2,
    )
