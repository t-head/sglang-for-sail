# SPDX-License-Identifier: Apache-2.0
"""Numerical contracts for the SAIL JIT Step5 radix region selector.

The selector replaces the FlashInfer ragged top-k only when the JIT radix
selector module can build on this machine; everything after selection still
runs through ``step5_pack_decode_regions``. The cases below pin what that
hand-off must keep:

1. the ranked history is the top-k of the live prefix with ties toward the
   smaller region id, even though the output is ordered by score;
2. the current region always lands in the slot after the selected history, so
   the pack step consumes the same input layout on both paths;
3. the streaming chunk path (width above 16384) selects the same set as one CTA
   per row;
4. the packed address of every kept region matches ``req_to_token``;
5. repeated calls are bit-identical, so a captured graph replays the same ids.

Run standalone on one PPU device::

    python test/ppu/unit/test_step5_selection_ppu_jit.py
    python test/ppu/unit/test_step5_selection_ppu_jit.py --bench
"""

import sys

import torch

from sglang.kernels.ops.attention.step5.selection import step5_select_regions
from sglang.kernels.ops.attention.step5.selection_ppu import (
    allocate_step5_ppu_selector_workspace,
    can_use_step5_selection_ppu,
)

DEV = "cuda"
TOPK = 512
# One past the selector's single-CTA limit; above it the streaming chunk path runs.
STREAMING_WIDTH = 40000


def _scores(rows, width, live, *, seed=0):
    """FP32 rows with a finite live prefix and -inf afterwards."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    live = [live] * rows if isinstance(live, int) else list(live)
    out = torch.full((rows, width), float("-inf"), dtype=torch.float32, device=DEV)
    for row, count in enumerate(live):
        count = min(max(count, 0), width)
        if count:
            out[row, :count] = torch.randn(count, generator=gen)
    return out


def _select(scores, positions, req_to_token, row_to_req):
    """Run selection on the AOT selector and return ids, counts and packed rows."""
    rows = scores.shape[0]
    workspace = allocate_step5_ppu_selector_workspace(
        rows=rows, width=scores.shape[1], device=scores.device
    )
    region_ids = torch.empty((rows, 513), dtype=torch.int32, device=DEV)
    counts = torch.empty(rows, dtype=torch.int32, device=DEV)
    step5_select_regions(
        scores,
        positions=positions,
        row_to_req=row_to_req,
        region_ids=region_ids,
        counts=counts,
        use_cute=False,
        ppu_workspace=workspace,
    )
    packed = torch.empty((rows, 513), dtype=torch.int64, device=DEV)
    lengths = torch.empty(rows, dtype=torch.int32, device=DEV)
    from sglang.kernels.ops.attention.step5.sparse_metadata import (
        step5_pack_decode_regions,
    )

    step5_pack_decode_regions(
        region_ids,
        counts,
        req_to_token=req_to_token,
        row_to_req=row_to_req,
        positions=positions,
        packed_regions=packed,
        causal_lengths=lengths,
    )
    torch.cuda.synchronize()
    return region_ids.clone(), counts.clone(), packed.clone(), lengths.clone()


def _metadata(rows, live, *, int64_positions=False):
    live = [live] * rows if isinstance(live, int) else list(live)
    dtype = torch.int64 if int64_positions else torch.int32
    positions = torch.tensor([count * 8 for count in live], dtype=dtype, device=DEV)
    row_to_req = torch.arange(1, rows + 1, dtype=torch.int32, device=DEV)
    # Aligned physical mapping: slot zero is the padding request row.
    context = max(max(live) * 8 + 64, TOPK * 8 + 64)
    base = torch.arange(0, rows + 1, dtype=torch.int64, device=DEV) * context
    req_to_token = (base.view(-1, 1) + torch.arange(context, device=DEV)).to(
        torch.int32
    )
    req_to_token[0] = 0
    return positions, row_to_req, req_to_token


def _kept(region_ids, positions, row, count):
    causal = int(positions[row]) // 8
    ids = region_ids[row, :count].long()
    assert (ids >= 0).all() and (ids < causal).all(), "kept ids must be live"
    return torch.sort(ids).values


def _expect_kept(scores, row, count):
    return torch.sort(
        torch.argsort(-scores[row, :count], stable=True)[: min(count, TOPK)]
    ).values


def run_selected_set(counts_under_test=(300, 511, 512, 513, 700, 2048)):
    """The selected set is top-k of the live prefix, ties toward the smaller id."""
    width = 4096
    for live in counts_under_test:
        scores = _scores(1, width, live)
        positions, row_to_req, req_to_token = _metadata(1, live)
        region_ids, counts, _, lengths = _select(
            scores, positions, req_to_token, row_to_req
        )
        causal = live
        keep = min(live, TOPK)
        assert torch.equal(
            _kept(region_ids, positions, 0, keep), _expect_kept(scores, 0, live)
        ), f"live={live}: selected set differs from stable top-k"
        assert int(region_ids[0, keep]) == causal, (
            f"live={live}: the current region must follow the history"
        )
        assert (region_ids[0, keep + 1 :] == -1).all(), f"live={live}: tail must be -1"
        assert int(counts[0]) == keep + 1, f"live={live}: wrong count"
        assert int(lengths[0]) == live * 8 + 1, f"live={live}: wrong causal length"
        print(f"PASS selected set live={live}")


def run_tie_break():
    """Four score levels force the cut inside one tie level.

    The selector folds the region id into the score as ``score + id * 1e-9``,
    so ties resolve toward the larger id; that bias is only representable while
    the score magnitude leaves room for a 1e-9 step in FP32. Only the selected
    score multiset is invariant across tie-breaks, so it is what this case pins.
    """
    width, live = 4096, 900
    gen = torch.Generator(device="cpu").manual_seed(7)
    scores = torch.full((1, width), float("-inf"), device=DEV)
    scores[0, :live] = torch.randint(0, 4, (live,), generator=gen).float().to(DEV)
    positions, row_to_req, req_to_token = _metadata(1, live)
    region_ids, _, _, _ = _select(scores, positions, req_to_token, row_to_req)
    kept = region_ids[0, :TOPK].long()
    desc = torch.sort(scores[0, :live], descending=True).values
    assert torch.equal(
        torch.sort(scores[0, kept]).values, torch.sort(desc[:TOPK]).values
    ), "the selected score multiset differs from top-k"
    cutoff = desc[TOPK - 1]
    tied = (scores[0, :live] == cutoff).nonzero().flatten()
    need = TOPK - int((scores[0, :live] > cutoff).sum())
    picked = torch.sort(kept[scores[0, kept] == cutoff]).values
    assert torch.equal(picked, tied[-need:]), (
        f"ties must resolve toward the larger region id: {need} of {tied.numel()}"
    )
    print(f"PASS tie break (level cut {need}/{tied.numel()} toward the larger id)")


def run_stale_tail():
    """Finite scores past positions // 8 must never be selected."""
    width, live = 4096, 700
    scores = _scores(1, width, live, seed=3)
    scores[0, live:] = torch.linspace(1e4, 2e4, width - live, device=DEV)
    positions, row_to_req, req_to_token = _metadata(1, live)
    region_ids, counts, _, _ = _select(scores, positions, req_to_token, row_to_req)
    assert torch.equal(
        _kept(region_ids, positions, 0, TOPK), _expect_kept(scores, 0, live)
    ), "a stale tail value won a slot"
    assert int(counts[0]) == TOPK + 1
    print("PASS stale tail isolation")


def run_streaming_chunk_path():
    """Above 16384 regions the chunk path must select the single-CTA set."""
    live = 30000
    scores = _scores(2, STREAMING_WIDTH, live, seed=11)
    positions, row_to_req, req_to_token = _metadata(2, live)
    region_ids, counts, _, _ = _select(scores, positions, req_to_token, row_to_req)
    for row in range(2):
        assert torch.equal(
            _kept(region_ids, positions, row, TOPK), _expect_kept(scores, row, live)
        ), f"row={row}: chunk path selected a different set"
        assert int(region_ids[row, TOPK]) == live
        assert int(counts[row]) == TOPK + 1
    print(f"PASS streaming chunk path live={live}")


def run_packed_mapping():
    """Every kept region packs to its aligned req_to_token address."""
    width, live = 4096, 700
    scores = _scores(3, width, live)
    positions, row_to_req, req_to_token = _metadata(3, live)
    region_ids, counts, packed, _ = _select(scores, positions, req_to_token, row_to_req)
    for row in range(3):
        request = int(row_to_req[row])
        count = int(counts[row])
        for slot in range(count):
            logical = int(region_ids[row, slot]) * 8
            physical = int(req_to_token[request, logical])
            value = int(packed[row, slot])
            assert value >> 32 == physical // 8, (
                f"row={row} slot={slot}: physical word differs from req_to_token"
            )
            assert value & 0xFFFFFFFF == logical, (
                f"row={row} slot={slot}: logical word differs"
            )
    print("PASS packed physical mapping")


def run_repeatable():
    """Repeated selection keeps the same set; slot order is not promised.

    The staging pass hands out output slots with atomics, so the order inside
    [0, topk) follows warp timing. The selected set, the counts and the packed
    addresses as a set are what a captured graph must reproduce.
    """
    width, live = 8192, 1024
    scores = _scores(4, width, live, seed=5)
    positions, row_to_req, req_to_token = _metadata(4, live)
    first = _select(scores, positions, req_to_token, row_to_req)
    second = _select(scores, positions, req_to_token, row_to_req)
    assert torch.equal(
        torch.sort(first[0]).values, torch.sort(second[0]).values
    ), "repeated selection changed the selected set"
    assert torch.equal(first[1], second[1]), "repeated selection changed the counts"
    assert torch.equal(
        torch.sort(first[2]).values, torch.sort(second[2]).values
    ), "repeated selection changed the packed rows as a set"
    assert torch.equal(
        first[3], second[3]
    ), "repeated selection changed the causal lengths"
    print("PASS repeatability (same set, slot order unspecified)")


def _bench(fn, iters=20, reps=5):
    """Per-call device time via CUDA graph replay, host enqueue excluded."""
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(iters):
            fn()
    best = float("inf")
    for _ in range(reps):
        begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
        begin.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        best = min(best, begin.elapsed_time(end) * 1e3 / iters)
    return best


def report_cost():
    """Sweep the widths a decode and a long-context prefill actually scan."""
    rows = 16
    settings = {
        "decode-1k": 1024,
        "decode-8k": 8192,
        "decode-64k": 65536,
        "prefill-1M": 131080,
    }
    print("\n== rows=16 selector cost (us)")
    for tag, width in settings.items():
        scores = _scores(rows, width, width - 8, seed=2)
        positions = torch.full((rows,), (width - 8) * 8, dtype=torch.int32, device=DEV)
        row_to_req = torch.arange(1, rows + 1, dtype=torch.int32, device=DEV)
        workspace = allocate_step5_ppu_selector_workspace(
            rows=rows, width=width, device=scores.device
        )
        region_ids = torch.empty((rows, 513), dtype=torch.int32, device=DEV)
        counts = torch.empty(rows, dtype=torch.int32, device=DEV)

        def call():
            step5_select_regions(
                scores,
                positions=positions,
                row_to_req=row_to_req,
                region_ids=region_ids,
                counts=counts,
                use_cute=False,
                ppu_workspace=workspace,
            )

        print(f"  {tag:11s} width={width:6d}  select {_bench(call):8.1f} us")


def main():
    assert torch.cuda.is_available(), "needs a CUDA/PPU device"
    print(f"device={torch.cuda.get_device_name()}")
    probe = torch.zeros((1, 512), dtype=torch.float32, device=DEV)
    if not can_use_step5_selection_ppu(probe):
        print("SKIP: the JIT radix selector is not available")
        return
    run_selected_set()
    run_tie_break()
    run_stale_tail()
    run_streaming_chunk_path()
    run_packed_mapping()
    run_repeatable()
    print("ALL PASS")
    if "--bench" in sys.argv:
        report_cost()


if __name__ == "__main__":
    main()
