# SPDX-License-Identifier: Apache-2.0
"""Contract and cost tests for the Step5 MoE router top-k.

The leaf runs one CTA per token row on a single warp, so its duration is the
serial chain of eight dependent reductions, each followed by a dependent reload
of the winning logit. Every round therefore reduces an int64 key that packs the
score's order-preserving bit pattern above the inverted expert id; one plain
``tl.max`` then returns the winner *and* its tie-broken index, where the obvious
``return_indices`` reduction computes the same pair far slower. That is a
"looks equivalent" rewrite, so the cases below pin the three properties a
further rewrite has to keep:

1. bitwise equality with the ``return_indices`` formulation, on random, tied and
   NaN rows and with the padded-row path active;
2. the ids are the top-8 by score with ties toward the smaller expert id, and
   the denominator adds the eight unbiased probabilities in *selection* order
   (a tree sum rounds differently) after multiplying by the routing scale;
3. a row holding a NaN logit emits +0.0 weights and id 0, while a padded row
   emits id -1 but still writes its weights.

``--bench`` prints the rows sweep next to the frozen ``return_indices``
formulation, which is also the cost the shape was chosen for.

Run standalone on one PPU device::

    python test/ppu/unit/test_step5_router_topk.py
    python test/ppu/unit/test_step5_router_topk.py --bench
"""

import sys

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.moe.step5_router_topk import Step5RouterTopKOp

DEV = "cuda"
NUM_EXPERTS = 352
TOPK = 8


@triton.jit
def _argmax_reference_kernel(
    logits,
    bias,
    weights_out,
    ids_out,
    logits_stride,
    weights_stride,
    ids_stride,
    num_token_non_padded,
    HAS_PADDING: tl.constexpr,
):
    """The ``return_indices`` formulation the bit-packed key replaced.

    Kept device-side so the equivalence is checked on the same arithmetic: a
    host reference cannot be bitwise here because torch's sigmoid is not
    bit-identical to ``tl.sigmoid``.
    """
    row = tl.program_id(0).to(tl.int64)
    expert = tl.arange(0, 512)
    valid = expert < 352
    gating = tl.load(logits + row * logits_stride + expert, mask=valid, other=0)
    probability = tl.sigmoid(gating.to(tl.float32))
    correction = tl.load(bias + expert, mask=valid, other=0).to(tl.float32)
    correction = tl.where(valid, correction, -float("inf"))
    scores = tl.where(valid, probability + correction, -float("inf"))
    has_bad = tl.max((gating != gating).to(tl.int32), axis=0) > 0

    weights = tl.zeros((8,), dtype=tl.float32)
    indices = tl.zeros((8,), dtype=tl.int32)
    total = 0.0
    slot = tl.arange(0, 8)
    for k in tl.static_range(8):
        _, index = tl.max(
            scores, axis=0, return_indices=True, return_indices_tie_break_left=True
        )
        index = index.to(tl.int32)
        selected_logit = tl.load(
            logits + row * logits_stride + index,
            mask=(index >= 0) & (index < 352),
            other=0,
        ).to(tl.float32)
        selected_probability = tl.sigmoid(selected_logit)
        weights = tl.where(slot == k, selected_probability, weights)
        indices = tl.where(slot == k, index, indices)
        total += selected_probability
        scores = tl.where(expert == index, -float("inf"), scores)

    weights = weights / (total + 1e-20)
    weights = weights * 3.0
    weights = tl.where(has_bad, 0.0, weights)
    indices = tl.where(has_bad, 0, indices)
    tl.store(weights_out + row * weights_stride + slot, weights)
    if HAS_PADDING:
        indices = tl.where(row < tl.load(num_token_non_padded), indices, -1)
    tl.store(ids_out + row * ids_stride + slot, indices)


def _grid_permutations(rows, seed):
    """Per-row expert permutation on a 0.01 grid.

    Two distinct experts then differ by at least ~1e-3 after the sigmoid, three
    orders of magnitude above the rounding difference between torch's sigmoid
    and ``tl.sigmoid``, so the id comparison cannot hinge on rounding.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    picks = [torch.randperm(NUM_EXPERTS, generator=gen) for _ in range(rows)]
    return torch.stack(picks).float() * 0.01 - 1.76


def _inputs(rows, *, seed=0, tied=False, with_nan=False):
    if tied:
        logits = torch.zeros((rows, NUM_EXPERTS), dtype=torch.float32)
        bias = torch.zeros(NUM_EXPERTS, dtype=torch.float32)
    else:
        logits = _grid_permutations(rows, seed)
        bias = torch.zeros(NUM_EXPERTS, dtype=torch.float32)
    if with_nan:
        logits[0, 0] = float("nan")
        logits[-1, NUM_EXPERTS - 1] = float("nan")
    return logits.to(DEV), bias.to(DEV)


def _reference(logits, bias, num_token_non_padded=None):
    """Stable-sort formulation of the routing contract."""
    probability = torch.sigmoid(logits)
    order = torch.argsort(-(probability + bias), dim=-1, stable=True)[:, :TOPK]
    selected = torch.sigmoid(torch.gather(logits, 1, order))
    total = torch.zeros(logits.shape[0], dtype=torch.float32, device=logits.device)
    for k in range(TOPK):
        total = total + selected[:, k]
    weights = selected / (total[:, None] + 1e-20) * 3.0
    ids = order.to(torch.int32)
    bad = torch.isnan(logits).any(dim=-1, keepdim=True)
    weights = weights.masked_fill(bad, 0.0)
    ids = ids.masked_fill(bad, 0)
    if num_token_non_padded is not None:
        rows = torch.arange(logits.shape[0], device=logits.device)[:, None]
        ids = ids.masked_fill(rows >= num_token_non_padded, -1)
    return weights, ids


def _launch(op, logits, bias, num_token_non_padded=None):
    return op.forward_triton(logits, bias, num_token_non_padded=num_token_non_padded)


def run_bitwise_equal_to_argmax_reference():
    """The bit-packed key selects and normalises exactly like ``return_indices``."""
    op = Step5RouterTopKOp()
    cases = [("random", {}), ("tied", {"tied": True}), ("nan", {"with_nan": True})]
    for tag, kwargs in cases:
        for rows in (1, 2, 7, 33, 64, 257):
            logits, bias = _inputs(rows, seed=rows + 11, **kwargs)
            weights = torch.empty((rows, TOPK), dtype=torch.float32, device=DEV)
            ids = torch.empty((rows, TOPK), dtype=torch.int32, device=DEV)
            _argmax_reference_kernel[(rows,)](
                logits,
                bias,
                weights,
                ids,
                logits.stride(0),
                weights.stride(0),
                ids.stride(0),
                None,
                HAS_PADDING=False,
                num_warps=4,
            )
            got_weights, got_ids = _launch(op, logits, bias)
            assert torch.equal(got_ids, ids), f"{tag} rows={rows}: ids diverged"
            assert torch.equal(got_weights, weights), (
                f"{tag} rows={rows}: weights are not bit-identical"
            )
        print(f"PASS bitwise vs return_indices, {tag} rows")

    rows, live = 17, 5
    logits, bias = _inputs(rows, seed=3)
    count = torch.tensor([live], dtype=torch.int32, device=DEV)
    weights = torch.empty((rows, TOPK), dtype=torch.float32, device=DEV)
    ids = torch.empty((rows, TOPK), dtype=torch.int32, device=DEV)
    _argmax_reference_kernel[(rows,)](
        logits,
        bias,
        weights,
        ids,
        logits.stride(0),
        weights.stride(0),
        ids.stride(0),
        count,
        HAS_PADDING=True,
        num_warps=4,
    )
    got_weights, got_ids = _launch(op, logits, bias, count)
    assert torch.equal(got_ids, ids) and torch.equal(got_weights, weights), (
        "the padded-row path diverged from the reference"
    )
    print(f"PASS bitwise vs return_indices, padding live={live}")


def run_selection_contract():
    """Top-8 by score, ties toward the smaller id, denominator in selection order."""
    op = Step5RouterTopKOp()
    rows = 12
    logits, bias = _inputs(rows, seed=5)
    scores = torch.sigmoid(logits) + bias
    top = scores.sort(dim=-1, descending=True).values[:, : TOPK + 1]
    assert float((top[:, :-1] - top[:, 1:]).min()) > 1e-4, (
        "case is degenerate: a score gap is too small to pin the selection"
    )

    got_weights, got_ids = _launch(op, logits, bias)
    want_weights, want_ids = _reference(logits, bias)
    assert torch.equal(got_ids, want_ids), "selected experts are not top-8 by score"
    assert torch.allclose(got_weights, want_weights, atol=1e-6, rtol=1e-5), (
        "normalisation diverged from the ordered-probability contract"
    )
    assert torch.allclose(
        got_weights.sum(dim=-1), torch.full((rows,), 3.0, device=DEV), atol=1e-6
    ), "the routing scale must be applied after normalisation"

    # Bias reorders the experts but must not reach the weights: identical
    # logits give identical probabilities, so every slot normalises to 3/8 even
    # though the scores differ.
    gen = torch.Generator(device="cpu").manual_seed(17)
    rank = torch.randperm(NUM_EXPERTS, generator=gen).float() * 0.01 - 1.76
    flat = torch.zeros((4, NUM_EXPERTS), dtype=torch.float32, device=DEV)
    weights, ids = _launch(op, flat, rank.to(DEV))
    want = torch.argsort(-rank, stable=True)[:TOPK].to(torch.int32)
    assert torch.equal(ids, want.to(DEV).repeat(4, 1)), (
        "bias must drive selection"
    )
    assert torch.allclose(weights, torch.full((4, TOPK), 3.0 / TOPK, device=DEV)), (
        "bias leaked into the routing weights"
    )

    _, tied_ids = _launch(op, flat, torch.zeros(NUM_EXPERTS, device=DEV))
    assert torch.equal(
        tied_ids, torch.arange(TOPK, device=DEV, dtype=torch.int32).repeat(4, 1)
    ), "a fully tied row must keep the 8 smallest expert ids in order"
    print("PASS selection contract")


def run_degenerate_rows():
    """NaN rows emit +0.0/id 0; padded rows emit id -1 but still write weights."""
    op = Step5RouterTopKOp()
    rows = 9
    logits, bias = _inputs(rows, seed=7, with_nan=True)
    weights, ids = _launch(op, logits, bias)

    for row in (0, rows - 1):
        assert (ids[row] == 0).all(), f"row {row}: a NaN row must emit id 0"
        assert (weights[row] == 0).all(), f"row {row}: a NaN row must emit zero weights"
        assert not torch.signbit(weights[row]).any(), (
            f"row {row}: NaN rows must emit +0.0, never -0.0"
        )

    # The padding mask is applied last, so it wins over the NaN rewrite.
    live = 4
    clean_logits, clean_bias = _inputs(rows, seed=8)
    count = torch.tensor([live], dtype=torch.int32, device=DEV)
    weights, ids = _launch(op, clean_logits, clean_bias, count)
    assert (ids[live:] == -1).all(), "padded rows must emit id -1"
    assert (ids[:live] >= 0).all(), "rows inside the padding count must keep their ids"
    assert torch.isfinite(weights).all(), "every row still writes finite weights"
    print(f"PASS degenerate rows, padding live={live}")


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


def report_cost():
    op = Step5RouterTopKOp()
    print(f"\n== per-call device time (us), one CTA per row, one warp")
    print(f"{'rows':>6} {'shipped':>9} {'argmax':>9} {'speedup':>8}")
    for rows in (1, 4, 16, 64, 128, 256, 512, 1024, 4096, 8192):
        logits, bias = _inputs(rows, seed=1)
        weights = torch.empty((rows, TOPK), dtype=torch.float32, device=DEV)
        ids = torch.empty((rows, TOPK), dtype=torch.int32, device=DEV)
        t_new = _bench(lambda: _launch(op, logits, bias))
        t_old = _bench(
            lambda: _argmax_reference_kernel[(rows,)](
                logits,
                bias,
                weights,
                ids,
                logits.stride(0),
                weights.stride(0),
                ids.stride(0),
                None,
                HAS_PADDING=False,
                num_warps=4,
            )
        )
        print(f"{rows:>6} {t_new:9.2f} {t_old:9.2f} {t_old / t_new:7.2f}x")


def main():
    assert torch.cuda.is_available(), "needs a CUDA/PPU device"
    print(f"device={torch.cuda.get_device_name()}")
    run_bitwise_equal_to_argmax_reference()
    run_selection_contract()
    run_degenerate_rows()
    print("ALL PASS")
    if "--bench" in sys.argv:
        report_cost()


if __name__ == "__main__":
    main()
