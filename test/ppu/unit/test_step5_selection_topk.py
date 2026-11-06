# SPDX-License-Identifier: Apache-2.0
"""Contract and cost tests for the Step5 region-selection top-k.

On PPU (sm80) ``selection.py`` always takes the FlashInfer branch: the CuTe
selector and the DSA ``deep_select_topk`` path are both gated to sm100. The
scores tensor is ``[T, (position_capacity + 7) // 8]`` -- 131080 columns for a
1M-token pool -- but a decode step only has ``position // 8`` live regions, so
the selection passes that length per row and the kernel scans the live prefix
instead of the whole width. A full-width scan costs ~562 us at 511 live regions
(grid=16, block=1024, 128 KiB dynamic shared memory: one CTA per row on 16 of
64 SMs); the length-aware kernel is ~2 us and the whole call ~12 us.

The cases below pin the four properties that any width- or length-aware rewrite
has to keep, and ``--bench`` reports where the time actually goes:

1. the selected set is "top-k by value, ties toward the smaller region id", so
   dropping ``deterministic``/``tie_break`` for speed changes which equally
   scored regions win;
2. ids at or past the row's causal boundary (``position // 8``) are discarded by
   ``step5_pack_decode_regions``, so scanning only the live prefix is
   output-equivalent even though the padding ids differ;
3. nothing past that boundary is read at all, so stale finite scores there (a
   warm pool holds the previous step's) cannot displace a live region;
4. the producer folds -0.0 into +0.0, because the radix key orders by bit
   pattern and would otherwise rank a signed zero ahead of a smaller region id.

Run standalone on one PPU device::

    python test/ppu/unit/test_step5_selection_topk.py
    python test/ppu/unit/test_step5_selection_topk.py --bench
"""

import sys

import flashinfer
import torch

from sglang.kernels.ops.attention.step5.scoring import step5_paged_scores
from sglang.kernels.ops.attention.step5.selection import step5_select_regions
from sglang.kernels.ops.attention.step5.selection_triton import (
    canonicalize_zero,
    launch as pack_regions,
)
from sglang.kernels.ops.attention.step5.sparse_metadata import (
    step5_pack_decode_regions,
)

DEV = "cuda"
TOPK = 512
# One past the largest pool this backend sizes scores for: 131080 regions is
# what the profile recorded (position_capacity ~ 1M tokens).
FULL_WIDTH = 131080


def _scores(rows, width, live, *, ties=False, seed=0):
    """FP32 score rows: finite prefix of ``live[i]`` regions, -inf afterwards."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    live = [live] * rows if isinstance(live, int) else list(live)
    out = torch.full((rows, width), float("-inf"), dtype=torch.float32, device=DEV)
    for r, n in enumerate(live):
        n = min(max(n, 0), width)
        if n == 0:
            continue
        part = torch.randn(n, generator=gen)
        if ties:
            # Four distinct levels: with >= 512 live regions the cut falls inside
            # a level, so the deterministic tie collection decides the winners.
            part = torch.randint(0, 4, (n,), generator=gen).float()
        out[r, :n] = part
    return out


def _metadata(rows, live):
    live = [live] * rows if isinstance(live, int) else list(live)
    assert len(live) == rows
    positions = torch.tensor([n * 8 for n in live], dtype=torch.int32, device=DEV)
    row_to_req = torch.arange(1, rows + 1, dtype=torch.int32, device=DEV)
    region_ids = torch.empty((rows, 513), dtype=torch.int32, device=DEV)
    counts = torch.empty(rows, dtype=torch.int32, device=DEV)
    return positions, row_to_req, region_ids, counts


def _select(scores, positions, row_to_req, region_ids, counts):
    step5_select_regions(
        scores,
        positions=positions,
        row_to_req=row_to_req,
        region_ids=region_ids,
        counts=counts,
        use_cute=False,
    )
    torch.cuda.synchronize()
    return region_ids.clone(), counts.clone()


def _kept_ids(region_ids, positions):
    """Mirror ``_pack_regions``: ids outside ``[0, position // 8)`` are dropped.

    The surviving ids are the only ones that reach the sparse kernels, so two
    selections are interchangeable exactly when these match.
    """
    causal = (positions // 8).view(-1, 1).long()
    ids = region_ids[:, :TOPK].long()
    kept = torch.where((ids >= 0) & (ids < causal), ids, torch.iinfo(torch.int32).max)
    return torch.sort(kept, dim=-1).values


def run_tie_break_semantics():
    """The selected set is top-k by value with ties resolved toward the smaller id."""
    width, rows = 2048, 4
    for live in (300, 700, width):
        scores = _scores(rows, width, live, ties=True)
        positions, row_to_req, region_ids, counts = _metadata(rows, live)
        got, got_counts = _select(scores, positions, row_to_req, region_ids, counts)
        got_again, counts_again = _select(
            scores, positions, row_to_req, region_ids, counts
        )
        assert torch.equal(got, got_again) and torch.equal(got_counts, counts_again), (
            f"live={live}: repeated selection differs (graph replay would diverge)"
        )

        # Layout contract of _pack_regions: sorted ids, then the current region,
        # then -1. Lanes past ``count + 1`` are the discarded tail.
        causal = (positions // 8).long()
        count = torch.minimum(causal, torch.tensor(TOPK, device=DEV))
        for r in range(rows):
            c = int(count[r])
            # Packing no longer sorts: the deterministic top-k already returns
            # ascending ids, and this fails if that contract ever changes.
            assert bool((got[r, 1 : c + 1].diff() > 0).all()), (
                f"live={live}: ids must ascend before the current region"
            )
            assert int(got[r, c]) == int(causal[r]), f"live={live}: current region slot"
            assert (got[r, c + 1 :] == -1).all(), f"live={live}: tail must be -1"

        ref = torch.argsort(-scores, dim=-1, stable=True)[:, :TOPK].long()
        assert torch.equal(_kept_ids(got, positions), _kept_ids(ref, positions)), (
            f"live={live}: kept ids are not top-{TOPK} with small-id tie break"
        )
        assert torch.equal(got_counts, count + 1)
        print(f"PASS tie-break semantics live={live}")


def run_short_context_padding():
    """Below 512 live regions the -inf padding fills the remaining output slots."""
    width, live = 4096, 300
    scores = _scores(1, width, live)
    positions, row_to_req, region_ids, counts = _metadata(1, live)
    got, got_counts = _select(scores, positions, row_to_req, region_ids, counts)
    causal = int(positions[0]) // 8
    assert torch.equal(
        torch.sort(got[0, :live].long()).values,
        torch.arange(live, dtype=torch.int64, device=DEV),
    ), "every live region must be selected when live < topk"
    # The top-k still spent its -inf tie-break on ids >= causal; those slots are
    # overwritten by the current region and -1, so they never reach attention.
    assert int(got[0, live]) == causal, "the current region follows the live ids"
    assert (got[0, live + 1 :] == -1).all(), "discarded padding must be -1"
    assert int(got_counts[0]) == causal + 1
    print(f"PASS short-context padding live={live}")


def run_live_prefix_isolation():
    """The scan stops at positions // 8, so the stale tail cannot win."""
    width, rows = 4096, 6
    live = [300, 511, 512, 513, 4000, 0]
    scores = _scores(rows, width, live, seed=2)
    # A real pool reuses its score buffer, so the tail still holds finite scores
    # from an earlier step. Only the live prefix may be read.
    for r, n in enumerate(live):
        scores[r, n:] = torch.linspace(1e4, 2e4, width - n, device=DEV)
    positions, row_to_req, region_ids, counts = _metadata(rows, live)
    row_to_req[-1] = 0  # unused request slot: this row scans nothing
    got, got_counts = _select(scores, positions, row_to_req, region_ids, counts)

    for r, n in enumerate(live):
        if n == 0:
            assert (got[r] == -1).all() and int(got_counts[r]) == 0, (
                "a padding row must select nothing"
            )
            continue
        count = min(n, TOPK)
        expect = torch.argsort(-scores[r, :n], stable=True)[:count].long()
        assert torch.equal(got[r, :count].long(), torch.sort(expect).values), (
            f"live={n}: stale ids past positions//8 were selected"
        )
        assert int(got[r, count]) == n, f"live={n}: current region slot"
        assert (got[r, count + 1 :] == -1).all(), f"live={n}: tail must be -1"
        assert int(got_counts[r]) == count + 1, f"live={n}: wrong count"
    print(f"PASS live-prefix isolation live={live}")


def run_width_narrowing():
    """Restricting the scanned width to max(position // 8) keeps the kept ids."""
    settings = {
        "short": [256] * 16,
        "boundary": [511, 512, 513, 700] * 4,
        "mixed": [256, 512, 1024, 2048, 4096, 8192, 16384, 32768] * 2,
        "full": [FULL_WIDTH - 8] * 16,
    }
    for tag, live in settings.items():
        scores = _scores(len(live), FULL_WIDTH, live, seed=1)
        positions, row_to_req, region_ids, counts = _metadata(len(live), live)
        full_ids, full_counts = _select(
            scores, positions, row_to_req, region_ids, counts
        )

        narrow = scores[:, : max(live)].contiguous()
        positions_n, row_to_req_n, ids_n, counts_n = _metadata(len(live), live)
        narrow_ids, narrow_counts = _select(
            narrow, positions_n, row_to_req_n, ids_n, counts_n
        )
        assert torch.equal(
            _kept_ids(full_ids, positions), _kept_ids(narrow_ids, positions)
        ), f"{tag}: narrowing the width changed kept region ids"
        assert torch.equal(full_counts, narrow_counts), f"{tag}: counts changed"
        print(f"PASS width narrowing {tag} ({FULL_WIDTH} -> {narrow.shape[1]})")


def run_producer_zero_contract():
    """Scoring hands the selector +0.0, never -0.0, for tied rows.

    The radix key ranks by bit pattern, where -0.0 sorts below +0.0. A zero query
    with negative head weights would emit -0.0 on every live region.
    """
    rows, heads, live, width = 4, 4, 800, 1024
    q = torch.zeros((rows, heads, 256), dtype=torch.bfloat16, device=DEV)
    weights = torch.full((rows, heads), -1.0, dtype=torch.float32, device=DEV)
    keys = torch.zeros((1, 8, 1, 256), dtype=torch.uint8, device=DEV)
    # Requests are 1..rows, so the pool view needs a row per request plus the
    # padding slot; a 2-row fixture read past the tensor once rows reached 4.
    req_to_token = torch.zeros((rows + 1, 8192), dtype=torch.int32, device=DEV)
    scores = torch.empty((rows, width), dtype=torch.float32, device=DEV)
    step5_paged_scores(
        q,
        weights,
        compressed_index_keys=keys,
        req_to_token=req_to_token,
        row_to_req=torch.arange(1, rows + 1, dtype=torch.int32, device=DEV),
        positions=torch.full((rows,), live * 8, dtype=torch.int32, device=DEV),
        out=scores,
        weight_scale=1.0,
    )
    torch.cuda.synchronize()
    history = scores[:, :live]
    assert (history == 0).all(), "a zero query must score exactly zero"
    assert not torch.signbit(history).any(), "scoring emitted -0.0"
    assert (scores[:, live:] == float("-inf")).all(), "future columns must be -inf"

    positions, row_to_req, region_ids, counts = _metadata(rows, live)
    got, got_counts = _select(scores, positions, row_to_req, region_ids, counts)
    want = torch.arange(TOPK, dtype=torch.int64, device=DEV)
    for r in range(rows):
        assert torch.equal(got[r, :TOPK].long(), want), (
            "a fully tied row must keep the 512 smallest ids"
        )
        assert int(got[r, TOPK]) == live, "the current region follows the live ids"
        assert int(got_counts[r]) == TOPK + 1
    print("PASS producer +0.0 contract")


def _req_to_token(rows, context):
    """Aligned [rows+1, context] pool view; request slot 0 stays padding."""
    base = torch.arange(rows + 1, dtype=torch.int32, device=DEV).mul_(8 * context)
    cols = torch.arange(context, dtype=torch.int32, device=DEV)
    return (base.unsqueeze(1) + cols.unsqueeze(0)).contiguous()


def run_fused_packing_equivalence():
    """Folding the pack into selection must match the separate pass bit for bit.

    ``step5_pack_decode_regions`` re-read the IDs selection had just written, so
    the only thing to prove is that the fused ``lane <= count`` predicate plus
    the slot-512 tail cover exactly what ``lane < count`` plus the 1024-lane
    sweep over the whole width covered before.
    """
    for tag, width, live in (
        ("single-tile", 512, [0, 1, 300, 512]),
        ("ragged", 4096, [0, 1, 511, 512, 513, 700]),
    ):
        rows = len(live)
        scores = _scores(rows, width, live, seed=3)
        positions, row_to_req, _, _ = _metadata(rows, live)
        row_to_req[0] = 0  # a padding row selects nothing
        req_to_token = _req_to_token(rows, 8 * width)
        for packed_width in (513, 514, 516):
            ids_split = torch.empty((rows, 513), dtype=torch.int32, device=DEV)
            counts_split = torch.empty(rows, dtype=torch.int32, device=DEV)
            packed_split = torch.full(
                (rows, packed_width), -5, dtype=torch.int64, device=DEV
            )
            lengths_split = torch.full((rows,), -5, dtype=torch.int32, device=DEV)
            step5_select_regions(
                scores,
                positions=positions,
                row_to_req=row_to_req,
                region_ids=ids_split,
                counts=counts_split,
                use_cute=False,
            )
            step5_pack_decode_regions(
                ids_split,
                counts_split,
                req_to_token=req_to_token,
                row_to_req=row_to_req,
                positions=positions,
                packed_regions=packed_split,
                causal_lengths=lengths_split,
            )

            ids_fused = torch.empty((rows, 513), dtype=torch.int32, device=DEV)
            counts_fused = torch.empty(rows, dtype=torch.int32, device=DEV)
            packed_fused = torch.full(
                (rows, packed_width), -5, dtype=torch.int64, device=DEV
            )
            lengths_fused = torch.full((rows,), -5, dtype=torch.int32, device=DEV)
            step5_select_regions(
                scores,
                positions=positions,
                row_to_req=row_to_req,
                region_ids=ids_fused,
                counts=counts_fused,
                use_cute=False,
                req_to_token=req_to_token,
                packed=packed_fused,
                causal_lengths=lengths_fused,
            )
            torch.cuda.synchronize()

            assert torch.equal(ids_split, ids_fused), f"{tag}/{packed_width}: ids"
            assert torch.equal(counts_split, counts_fused), f"{tag}/{packed_width}: counts"
            assert torch.equal(packed_split, packed_fused), (
                f"{tag}/{packed_width}: fused pack differs from the separate pass"
            )
            assert torch.equal(lengths_split, lengths_fused), (
                f"{tag}/{packed_width}: causal lengths differ"
            )
            assert (packed_fused >= -1).all(), "tail slots must be -1, not uninitialised"
        print(f"PASS fused packing equivalence {tag} (width={width})")


def _bench(fn, iters=20, reps=5):
    """Per-call device time via CUDA graph replay (host enqueue excluded)."""
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


def _topk(scores, dsa_graph_safe=True):
    return flashinfer.top_k(
        scores,
        TOPK,
        sorted=False,
        deterministic=True,
        tie_break=1,
        dsa_graph_safe=dsa_graph_safe,
    )


def report_cost():
    """Print the per-kernel attribution and the gain of the length-aware scan."""
    rows = 16
    live_sets = (256, 511, 513, 4096, 16384, FULL_WIDTH - 8)
    print(f"\n== rows={rows} width={FULL_WIDTH}: old chain vs select (us)")
    for live in live_sets:
        scores = _scores(rows, FULL_WIDTH, live)
        ordered = torch.empty_like(scores)
        positions, row_to_req, region_ids, counts = _metadata(rows, live)
        ids = torch.empty((rows, 513), dtype=torch.int32, device=DEV)
        packed = torch.empty((rows, 513), dtype=torch.int64, device=DEV)
        t_canon = _bench(lambda: canonicalize_zero(scores, ordered))
        t_topk = _bench(lambda: _topk(ordered))
        t_pack = _bench(
            lambda: pack_regions(
                ids, positions, row_to_req, packed, counts, selected_k=TOPK
            )
        )
        t_select = _bench(
            lambda: step5_select_regions(
                scores,
                positions=positions,
                row_to_req=row_to_req,
                region_ids=region_ids,
                counts=counts,
                use_cute=False,
            )
        )
        print(
            f"  live={live:6d}  canon {t_canon:6.1f}  topk-full {t_topk:7.1f}  "
            f"pack {t_pack:6.1f}  |  select {t_select:7.1f}  "
            f"({(t_canon + t_topk + t_pack) / t_select:5.1f}x)"
        )

    print(f"\n== rows={rows}: cost follows the longest live row, not the width")
    for tag, live in (
        ("all short", [256] * 16),
        ("one long", [256] * 15 + [FULL_WIDTH - 8]),
        ("half long", [256] * 8 + [FULL_WIDTH - 8] * 8),
        ("all long", [FULL_WIDTH - 8] * 16),
    ):
        scores = _scores(rows, FULL_WIDTH, live)
        positions, row_to_req, region_ids, counts = _metadata(rows, live)
        t_select = _bench(
            lambda: step5_select_regions(
                scores,
                positions=positions,
                row_to_req=row_to_req,
                region_ids=region_ids,
                counts=counts,
                use_cute=False,
            )
        )
        print(f"  {tag:11s} max_live={max(live):6d}  select {t_select:7.1f} us")

    print("\n== rows=16: tie-break / vec variants at live=511 (pivot lands on -inf)")
    scores = _scores(rows, FULL_WIDTH, 511)
    variants = {
        "VEC=1 det=T tie=SMALL (default)": lambda: _topk(scores),
        "VEC=4 det=T tie=SMALL": lambda: _topk(scores, dsa_graph_safe=False),
        "VEC=1 det=T tie=NONE": lambda: flashinfer.top_k(
            scores, TOPK, sorted=False, deterministic=True, tie_break=0,
            dsa_graph_safe=True,
        ),
        "torch.topk": lambda: torch.topk(scores, TOPK, dim=-1, sorted=False).indices,
    }
    for name, fn in variants.items():
        print(f"  {name:34s} {_bench(fn):7.1f} us")

    print("\n== rows sweep at live=full (one CTA per row, 64 SMs available)")
    for n in (1, 16, 32, 64):
        scores = _scores(n, FULL_WIDTH, FULL_WIDTH - 8)
        print(
            f"  rows={n:3d}  VEC=1 {_bench(lambda: _topk(scores)):7.1f} us"
            f"   VEC=4 {_bench(lambda: _topk(scores, dsa_graph_safe=False)):7.1f} us"
        )


def main():
    assert torch.cuda.is_available(), "needs a CUDA/PPU device"
    print(f"device={torch.cuda.get_device_name()}")
    run_tie_break_semantics()
    run_short_context_padding()
    run_live_prefix_isolation()
    run_width_narrowing()
    run_fused_packing_equivalence()
    run_producer_zero_contract()
    print("ALL PASS")
    if "--bench" in sys.argv:
        report_cost()


if __name__ == "__main__":
    main()
