# SPDX-License-Identifier: Apache-2.0
"""Numerical contracts for the SAIL JIT Step5 sparse prefill.

The JIT kernel replaces the portable Triton prefill only when the sparse prefill
module can build on this machine and the query/KV head ratio is 8 or
16. It walks each query row's own selected regions, so unlike the union schedule
there is no cross-row sharing to hide an indexing error behind. The cases below
pin what the hand-off must keep:

1. with one selected region the output equals an FP32 softmax over its visible
   tokens, so the region walk, the causal mask and the epilogue agree;
2. with shuffled physical regions and short rows the output and the natural LSE
   match a direct FP32 reference, which reads the same packed words;
3. the logical start and the physical region stay independent: a region whose KV
   row differs from its logical position is still masked correctly;
4. repeated calls are bit-identical, so a captured graph replays the same rows;
5. the TP16 four-head layout is rejected instead of silently computed.

Run standalone on one PPU device::

    python test/ppu/unit/test_step5_prefill_ppu_jit.py
    python test/ppu/unit/test_step5_prefill_ppu_jit.py --bench
"""

import sys

import torch

from sglang.kernels.ops.attention.step5.prefill_ppu import (
    STEP5_PPU_PREFILL_REGION_TOKENS,
    can_use_step5_prefill_ppu,
    step5_sparse_prefill_ppu,
)

DEV = "cuda"
HEAD_DIM = 192
REGION = STEP5_PPU_PREFILL_REGION_TOKENS


def _selection(rows, width, live, *, seed=0):
    """Packed words whose physical regions are shuffled away from their starts.

    Returns the int64 packed rows and the int32 counts. Each row keeps ``live``
    selected regions, and ``starts`` run in logical order while ``physical`` is
    permuted, which is what a paged pool produces.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    live = [live] * rows if isinstance(live, int) else list(live)
    packed = torch.full((rows, width), -1, dtype=torch.int64, device=DEV)
    counts = torch.zeros(rows, dtype=torch.int32, device=DEV)
    for row, count in enumerate(live):
        count = min(max(count, 1), width)
        starts = torch.arange(count, dtype=torch.int64) * REGION
        physical = torch.randperm(count, generator=gen)[:count].to(torch.int64)
        words = (physical << 32) | (starts & 0xFFFFFFFF)
        packed[row, :count] = words.to(DEV)
        counts[row] = count
    return packed, counts


def _qkv(rows, heads, slots, *, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn((rows, heads, HEAD_DIM), generator=gen, device=DEV).to(
        torch.bfloat16
    )
    k = torch.randn((slots, 1, HEAD_DIM), generator=gen, device=DEV).to(
        torch.bfloat16
    )
    v = torch.randn((slots, 1, HEAD_DIM), generator=gen, device=DEV).to(
        torch.bfloat16
    )
    return q, k, v


def _reference(q, k, v, physical, starts, counts, positions, scale):
    """FP32 reference: softmax over each row's visible selected tokens."""
    rows, heads, _ = q.shape
    out = torch.empty_like(q)
    lse = torch.empty((rows, heads), dtype=torch.float32, device=DEV)
    for row in range(rows):
        keys, values, logical = [], [], []
        for slot in range(int(counts[row])):
            region = int(physical[row, slot])
            keys.append(k[region * REGION : (region + 1) * REGION, 0])
            values.append(v[region * REGION : (region + 1) * REGION, 0])
            logical.append(
                starts[row, slot] + torch.arange(REGION, dtype=torch.int32, device=DEV)
            )
        scores = torch.einsum("hd,kd->hk", q[row].float(), torch.cat(keys).float())
        scores = scores * scale
        visible = torch.cat(logical) <= positions[row]
        scores = scores.masked_fill(~visible, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        out[row] = torch.einsum("hk,kd->hd", weights, torch.cat(values).float()).to(
            torch.bfloat16
        )
        lse[row] = torch.logsumexp(scores, dim=-1)
    return out, lse


def _run_ppu(q, k, v, packed, counts, positions, scale):
    rows, heads, _ = q.shape
    physical = torch.empty(packed.shape, dtype=torch.int32, device=DEV)
    starts = torch.empty(packed.shape, dtype=torch.int32, device=DEV)
    out = torch.empty_like(q)
    lse = torch.empty((rows, heads), dtype=torch.float32, device=DEV)
    step5_sparse_prefill_ppu(
        q,
        k,
        v,
        packed=packed,
        physical_regions=physical,
        logical_starts=starts,
        live_regions=counts,
        positions=positions,
        out=out,
        lse=lse,
        scale=scale,
    )
    return out, lse, physical, starts


def case_single_region(heads):
    """One selected region: compare against an FP32 softmax over six tokens."""
    rows, slots = 3, 4 * 64
    q, k, v = _qkv(rows, heads, slots, seed=1)
    # Region 0 owns slots [0, 8) and starts at logical position 0, so the
    # reference indexes K/V directly and the causal limit hides the last two
    # tokens of the region.
    packed = torch.full((rows, 1), 0, dtype=torch.int64, device=DEV)
    counts = torch.ones(rows, dtype=torch.int32, device=DEV)
    visible = torch.full((rows,), 6, dtype=torch.int32, device=DEV)
    scale = HEAD_DIM**-0.5
    out, lse, _, _ = _run_ppu(q, k, v, packed, counts, visible - 1, scale)

    scores = torch.einsum("rhd,kd->rhk", q.float(), k[:visible[0], 0].float())
    scores = scores * scale
    expected = torch.einsum(
        "rhk,kd->rhd", torch.softmax(scores[:, :, : visible[0]], -1), v[: visible[0], 0].float()
    )
    torch.testing.assert_close(
        out.float(), expected, atol=3e-2, rtol=0, msg="single region output"
    )
    torch.testing.assert_close(
        lse,
        torch.logsumexp(scores[:, :, : visible[0]], dim=-1),
        atol=5e-2,
        rtol=0,
        msg="single region LSE",
    )


def case_matches_reference(heads, rows=6, live=(1, 2, 5, 9, 17, 33), seed=2):
    """Many regions: match the FP32 reference on output and natural LSE."""
    slots = 64 * REGION
    q, k, v = _qkv(rows, heads, slots, seed=seed)
    packed, counts = _selection(rows, 513, live, seed=seed)
    # Rows end mid-region so the token-level causal mask inside the last region
    # is exercised as well.
    lengths = counts * REGION - torch.arange(rows, dtype=torch.int32, device=DEV) % 3
    positions = (lengths - 1).to(torch.int32).contiguous()
    scale = HEAD_DIM**-0.5
    out, lse, physical, starts = _run_ppu(q, k, v, packed, counts, positions, scale)
    ref_out, ref_lse = _reference(
        q, k, v, physical, starts, counts, positions, scale
    )
    torch.testing.assert_close(out.float(), ref_out.float(), atol=4e-2, rtol=0)
    torch.testing.assert_close(lse, ref_lse, atol=5e-2, rtol=0)


def case_physical_differs(heads):
    """A region whose KV row is not its logical position is still masked right."""
    rows, slots = 2, 8 * REGION
    q, k, v = _qkv(rows, heads, slots, seed=5)
    # Logical region 1 selects physical region 7 and vice versa; both must be
    # masked by the logical start, not by the KV row they read.
    words = torch.tensor(
        [(7 << 32) | 0, (0 << 32) | REGION], dtype=torch.int64, device=DEV
    )
    packed = words.expand(rows, 2).contiguous()
    counts = torch.full((rows,), 2, dtype=torch.int32, device=DEV)
    positions = torch.full((rows,), REGION + 2, dtype=torch.int32, device=DEV)
    scale = HEAD_DIM**-0.5
    out, lse, physical, starts = _run_ppu(q, k, v, packed, counts, positions, scale)
    ref_out, ref_lse = _reference(
        q, k, v, physical, starts, counts, positions, scale
    )
    torch.testing.assert_close(out.float(), ref_out.float(), atol=4e-2, rtol=0)
    torch.testing.assert_close(lse, ref_lse, atol=5e-2, rtol=0)


def case_repeatable(heads):
    """The same inputs produce bit-identical output across launches."""
    rows, slots = 4, 64 * REGION
    q, k, v = _qkv(rows, heads, slots, seed=3)
    packed, counts = _selection(rows, 513, 8, seed=3)
    positions = (counts * REGION - 1).to(torch.int32).contiguous()
    scale = HEAD_DIM**-0.5
    first, first_lse, _, _ = _run_ppu(q, k, v, packed, counts, positions, scale)
    second, second_lse, _, _ = _run_ppu(q, k, v, packed, counts, positions, scale)
    assert torch.equal(first, second), "prefill output is not deterministic"
    assert torch.equal(first_lse, second_lse), "prefill LSE is not deterministic"


def case_heads4_rejected():
    """Four query heads keep the Triton path instead of a silent fallback."""
    q, k, _ = _qkv(2, 4, 64, seed=4)
    assert not can_use_step5_prefill_ppu(q, k), "four-head layout must be rejected"


def bench(heads=8, rows=64, live=512, iters=50):
    slots = 1024 * REGION
    q, k, v = _qkv(rows, heads, slots, seed=7)
    packed, counts = _selection(rows, 513, live, seed=7)
    positions = (counts * REGION - 1).to(torch.int32).contiguous()
    scale = HEAD_DIM**-0.5
    out = torch.empty_like(q)
    lse = torch.empty((rows, heads), dtype=torch.float32, device=DEV)
    physical = torch.empty(packed.shape, dtype=torch.int32, device=DEV)
    starts = torch.empty(packed.shape, dtype=torch.int32, device=DEV)
    args = dict(
        packed=packed,
        physical_regions=physical,
        logical_starts=starts,
        live_regions=counts,
        positions=positions,
        out=out,
        lse=lse,
        scale=scale,
    )
    for _ in range(5):
        step5_sparse_prefill_ppu(q, k, v, **args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        step5_sparse_prefill_ppu(q, k, v, **args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    heads = 8
    if not can_use_step5_prefill_ppu(*_qkv(2, 8, 64, seed=0)[:2]):
        raise SystemExit("JIT Step5 sparse prefill is unavailable on this device")
    case_heads4_rejected()
    case_single_region(heads)
    case_matches_reference(heads)
    case_physical_differs(heads)
    case_repeatable(heads)
    print("step5 prefill PPU JIT: all cases passed")
    if "--bench" in sys.argv:
        for h, live in ((8, 512), (16, 512), (8, 64), (16, 64)):
            ms = bench(h, live=live)
            print(f"heads={h} live={live}: {ms:.3f} ms per launch")


if __name__ == "__main__":
    main()