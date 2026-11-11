# SPDX-License-Identifier: Apache-2.0
"""Step5 scoring 的测试夹具与独立实验；不接入生产派发。"""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.step5.scoring_triton import (
    _decode_e4m3,
    _e4m3_bytes_to_fp8,
    _early_exit_worthwhile,
    _live_extent,
    _load_keys_tile,
    _native_fp8,
    _plus_zero,
    _score_regions_ppu_opt,
    _wipe_region_scores,
)

VARIANTS = ("baseline", "tile", "compact", "compact_tile")


@triton.jit
def _pack_keys(
    Keys, Map, MaxPosition, Compact,
    MAP_STRIDE: tl.constexpr, REGIONS: tl.constexpr,
    BLOCK_R: tl.constexpr, NATIVE_FP8: tl.constexpr, FP8_INDEXER: tl.constexpr,
):
    request = tl.program_id(0)
    regions = tl.program_id(1) * BLOCK_R + tl.arange(0, BLOCK_R)
    live = tl.load(MaxPosition + request) // 8
    valid = (request > 0) & (regions < live) & (regions < REGIONS)
    packed = _load_keys_tile(Keys, Map, request, regions, valid, MAP_STRIDE, BLOCK_R)
    if FP8_INDEXER:
        values = packed
    else:
        values = _decode_e4m3(packed, NATIVE_FP8)
    channel = tl.arange(0, 256)
    tl.store(
        Compact + (request * REGIONS + regions[:, None]) * 256 + channel[None, :],
        tl.trans(values), (regions[:, None] < REGIONS),
    )


@triton.jit
def _compact_products(
    query, Compact, request, regions, valid,
    REGIONS: tl.constexpr, BLOCK_R: tl.constexpr,
    NATIVE_FP8: tl.constexpr, FP8_INDEXER: tl.constexpr,
):
    channel = tl.arange(0, 256)
    values = tl.load(
        Compact + (request * REGIONS + regions[:, None]) * 256 + channel[None, :],
        valid[:, None], other=0,
    )
    if FP8_INDEXER:
        keys = _e4m3_bytes_to_fp8(tl.trans(values), NATIVE_FP8)
    else:
        keys = tl.trans(values)
    rescale = 1.0
    if FP8_INDEXER and not NATIVE_FP8:
        rescale = 65536.0
    return tl.dot(query, keys) * rescale


@triton.jit
def _score_compact(
    Q, W, Compact, RowToReq, Positions, Out, SpanStart, SpanEnd,
    HEADS: tl.constexpr, REGIONS: tl.constexpr, BLOCK_R: tl.constexpr,
    GRID_TILES: tl.constexpr, GROUP_ROWS: tl.constexpr, HEAD_BLOCK: tl.constexpr,
    EARLY_EXIT: tl.constexpr, NATIVE_FP8: tl.constexpr, FP8_INDEXER: tl.constexpr,
):
    group = tl.program_id(0)
    first_tile = tl.program_id(1)
    row_base = tl.load(SpanStart + group).to(tl.int64)
    row_limit = tl.load(SpanEnd + group).to(tl.int64)
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
    matrix_row = tl.arange(0, GROUP_ROWS * HEAD_BLOCK)
    channel = tl.arange(0, 256)
    query_row = row_base + matrix_row // HEAD_BLOCK
    head = matrix_row % HEAD_BLOCK
    bits = tl.load(
        Q + (query_row[:, None] * HEADS + head[:, None]) * 256 + channel[None, :],
        (query_row[:, None] < row_limit) & (head[:, None] < HEADS), other=0,
    )
    if FP8_INDEXER:
        query = _e4m3_bytes_to_fp8(bits, NATIVE_FP8)
    else:
        query = _decode_e4m3(bits, NATIVE_FP8)
    weights = tl.load(
        W + rows[:, None] * HEADS + tl.arange(0, HEAD_BLOCK)[None, :],
        row_valid[:, None] & (tl.arange(0, HEAD_BLOCK)[None, :] < HEADS)
        & (request[:, None] > 0), other=0.0,
    ).to(tl.float32)
    if not EARLY_EXIT:
        live, max_live, last_tile = _live_extent(
            position, row_valid, request, BLOCK_R, REGIONS, True,
        )
    for tile in range(first_tile, last_tile, GRID_TILES):
        regions = tile * BLOCK_R + tl.arange(0, BLOCK_R)
        in_range = regions < REGIONS
        if uniform & (base_request > 0) & (tile * BLOCK_R < max_live):
            products = _compact_products(
                query, Compact, base_request, regions,
                in_range & (regions < max_live), REGIONS, BLOCK_R,
                NATIVE_FP8, FP8_INDEXER,
            )
            weighted = tl.maximum(products, 0.0) * tl.reshape(
                weights, (GROUP_ROWS * HEAD_BLOCK, 1),
            )
            score = tl.sum(
                tl.reshape(weighted, (GROUP_ROWS, HEAD_BLOCK, BLOCK_R)), axis=1,
            )
            score = tl.where(regions[None, :] < live[:, None], score, -float("inf"))
            tl.store(
                Out + rows[:, None] * REGIONS + regions[None, :],
                _plus_zero(score), row_valid[:, None] & in_range[None, :],
            )


def make_case(*, lengths, prefixes, heads=4, width=None, seed=0, easy=False, padding=0):
    """每个 logical page 映射到随机且不重叠的 physical page。"""
    if len(lengths) != len(prefixes) or not lengths or min(lengths) <= 0:
        raise ValueError("lengths/prefixes 必须描述非空请求")
    generator = torch.Generator().manual_seed(seed)
    context = max(p + n for p, n in zip(prefixes, lengths))
    map_cols = triton.cdiv(context, 64) * 64
    regions = width if width is not None else triton.cdiv(context, 8)
    if regions < triton.cdiv(context, 8):
        raise ValueError("score width 不能截断请求")
    requests = len(lengths)
    pages_per_request = map_cols // 64
    pages = requests * pages_per_request
    permutation = torch.randperm(pages, generator=generator)
    mapping = torch.zeros((requests + 1, map_cols), dtype=torch.int32)
    physical = permutation.reshape(requests, pages_per_request)
    slots = (physical[:, :, None] * 64 + torch.arange(64)).reshape(requests, map_cols)
    mapping[1:] = slots.to(torch.int32)
    shape = (pages, 8, 256)
    if easy:
        physical_keys = torch.randint(-1, 2, shape, generator=generator).float()
        q = torch.randint(-1, 2, (sum(lengths) + padding, heads, 256), generator=generator).float()
        weights = torch.ones((q.shape[0], heads), dtype=torch.float32)
    else:
        physical_keys = torch.randn(shape, generator=generator)
        q = torch.randn((sum(lengths) + padding, heads, 256), generator=generator)
        weights = torch.rand((q.shape[0], heads), generator=generator) * 0.5
    key_bytes = physical_keys.to(torch.float8_e4m3fn).view(torch.uint8)
    channel = torch.arange(256)[None, None, :]
    slot = torch.arange(8)[None, :, None]
    page = torch.arange(pages)[:, None, None]
    offsets = (page * 2048 + (channel // 128) * 1024 + slot * 128
               + (((channel % 128) // 16) ^ slot) * 16 + channel % 16)
    swizzled = torch.empty(pages * 2048, dtype=torch.uint8)
    swizzled[offsets.reshape(-1)] = key_bytes.reshape(-1)
    positions = torch.cat([
        torch.arange(p, p + n, dtype=torch.int32) for p, n in zip(prefixes, lengths)
    ] + [torch.zeros(padding, dtype=torch.int32)])
    owners = torch.cat([
        torch.full((n,), i + 1, dtype=torch.int32) for i, n in enumerate(lengths)
    ] + [torch.zeros(padding, dtype=torch.int32)])
    max_positions = torch.tensor([0] + [p + n - 1 for p, n in zip(prefixes, lengths)], dtype=torch.int32)
    return dict(
        q=q.to(torch.float8_e4m3fn).view(torch.uint8).cuda(), weights=weights.cuda(),
        keys=swizzled.view(pages, 8, 1, 256).cuda(), mapping=mapping.cuda(),
        owners=owners.cuda(), positions=positions.cuda(), max_positions=max_positions.cuda(),
        lengths=tuple(lengths) + ((padding,) if padding else ()), regions=regions,
        physical_keys=key_bytes.view(torch.float8_e4m3fn).float(), physical_pages=physical,
    )


def spans(case, group_rows):
    starts, ends = [], []
    base = 0
    for length in case["lengths"]:
        for start in range(base, base + length, group_rows):
            starts.append(start)
            ends.append(min(start + group_rows, base + length))
        base += length
    return (torch.tensor(starts, dtype=torch.int32, device="cuda"),
            torch.tensor(ends, dtype=torch.int32, device="cuda"))


def make_runner(case, variant, *, fp8=True, group_rows=8, block_regions=32, grid_tiles=32):
    if variant not in VARIANTS:
        raise ValueError(variant)
    tuned = variant in ("tile", "compact_tile")
    compact = variant in ("compact", "compact_tile")
    group = group_rows if tuned else 8
    block = block_regions if tuned else 8
    grid = min(triton.cdiv(case["regions"], block), grid_tiles if tuned else 128)
    starts, ends = spans(case, group)
    rows, heads, _ = case["q"].shape
    native = _native_fp8(case["q"].device)
    early = _early_exit_worthwhile(rows, block, grid)
    out = torch.empty((rows, case["regions"]), dtype=torch.float32, device="cuda")
    compact_keys = torch.empty(
        (case["mapping"].shape[0], case["regions"], 256),
        dtype=torch.uint8 if fp8 else torch.bfloat16, device="cuda",
    ) if compact else None

    def pack():
        if compact:
            _pack_keys[(case["mapping"].shape[0], triton.cdiv(case["regions"], 32))](
                case["keys"], case["mapping"], case["max_positions"], compact_keys,
                case["mapping"].stride(0), case["regions"], 32, native, fp8, num_warps=4,
            )

    def score():
        if compact:
            return _score_compact[(starts.numel(), grid)](
                case["q"], case["weights"], compact_keys, case["owners"],
                case["positions"], out, starts, ends,
                heads, case["regions"], block, grid, group, 4, early, native, fp8,
                num_warps=2,
            )
        else:
            return _score_regions_ppu_opt[(starts.numel(), grid)](
                case["q"], case["weights"], case["keys"], case["mapping"],
                case["owners"], case["positions"], out, starts, ends,
                heads, rows, case["mapping"].stride(0), case["regions"], block,
                grid, True, early, 1.0, group, 4, True, True, native, fp8, num_warps=2,
            )

    def run():
        _wipe_region_scores(out)
        pack()
        score()
        return out

    return dict(run=run, pack=pack, score=score, out=out, compact_keys=compact_keys,
                config=(group, block, grid), starts=starts, ends=ends)


def make_production_runner(case, *, compact=True, weights=None, q=None, weight_scale=None):
    """真实公共 scoring/selection 入口；调用者在外层固定灰度开关和 FP8 模式。"""
    from sglang.kernels.ops.attention.step5.scoring import step5_paged_scores
    from sglang.srt.layers.attention.step5.region_selection import (
        allocate_selection_workspace,
        select_step5_regions,
    )

    q = case["q"].view(torch.float8_e4m3fn) if q is None else q
    weights = case["weights"] if weights is None else weights
    workspace = allocate_selection_workspace(
        rows=q.shape[0], score_width=case["regions"], score_group_rows=1,
        device=q.device, prefill=True, score_lengths=case["lengths"],
        compact_allowed=compact,
    )

    def score(**overrides):
        return step5_paged_scores(
            q, weights, compressed_index_keys=case["keys"],
            req_to_token=case["mapping"], row_to_req=case["owners"],
            positions=case["positions"], out=workspace.region_scores,
            score_groups=workspace.score_groups, compact_workspace=workspace.compact_scoring,
            weight_scale=weight_scale, **overrides,
        )

    def run():
        return select_step5_regions(
            index_q=q, index_weights=weights, compressed_index_keys=case["keys"],
            req_to_token=case["mapping"], row_to_req=case["owners"],
            positions=case["positions"], score_width=case["regions"],
            scoring_weight_scale=weight_scale, workspace=workspace,
        )

    return dict(run=run, score=score, out=workspace.region_scores, workspace=workspace)


def sparse_request_slots(case, slots=(37, 103), capacity=160):
    """请求池很大时，紧凑 K 的 request 维仍只包含本 forward 的请求段。"""
    mapping = torch.zeros(
        (capacity, case["mapping"].shape[1]), dtype=torch.int32, device="cuda",
    )
    owners = case["owners"].clone()
    for original, slot in enumerate(slots, 1):
        mapping[slot].copy_(case["mapping"][original])
        owners[case["owners"] == original] = slot
    case["mapping"], case["owners"] = mapping, owners


def selection_snapshot(runner):
    packed, counts, lengths = runner["run"]()
    columns = torch.arange(packed.shape[1], device=packed.device)
    valid = columns[None, :] < counts[:, None]
    # PPU atomic staging 的槽序不稳定；只有有效项集合属于输出契约。
    selected = torch.where(valid, packed, -1).sort(dim=1).values
    return selected.clone(), counts.clone(), lengths.clone()


def reference_scores(case, rows=None):
    """独立 Torch FP32 oracle；先逐 head ReLU，再求加权和。"""
    q = case["q"].cpu().view(torch.float8_e4m3fn).float()
    weights = case["weights"].cpu()
    owners, positions = case["owners"].cpu(), case["positions"].cpu()
    rows = list(range(q.shape[0])) if rows is None else list(rows)
    out = torch.full((len(rows), case["regions"]), -float("inf"))
    for i, row in enumerate(rows):
        request = int(owners[row])
        live = int(positions[row]) // 8 if request else 0
        if not live:
            continue
        pages = case["physical_pages"][request - 1]
        keys = case["physical_keys"][pages].reshape(-1, 256)[:live]
        products = q[row] @ keys.T
        values = (products.relu() * weights[row, :, None]).sum(dim=0)
        out[i, :live] = torch.where(values == 0, 0.0, values)
    return out


def compare_scores(actual, expected, sample_rows=32):
    same = actual == expected
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    delta = torch.where(finite, (actual - expected).abs(), 0.0)
    indices = torch.linspace(0, actual.shape[0] - 1, min(sample_rows, actual.shape[0])).long().cuda()
    a, b = actual[indices].cpu(), expected[indices].cpu()
    k = min(512, actual.shape[1])
    top_a = torch.argsort(a, dim=1, descending=True, stable=True)[:, :k]
    top_b = torch.argsort(b, dim=1, descending=True, stable=True)[:, :k]
    return dict(bitwise_equal=torch.equal(actual.view(torch.int32), expected.view(torch.int32)),
                mismatches=int((~same).sum()), max_abs=float(delta.max()),
                top512_equal=torch.equal(top_a, top_b))
