# SPDX-License-Identifier: Apache-2.0
"""Numerical contracts for the SAIL JIT Step5 sparse paged decode.

The JIT kernel replaces the portable Triton decode only when the paged decode
module can build on this machine and the query/KV head ratio is 8
or 16. The cases below pin what that hand-off must keep:

1. with one selected region the output equals an FP32 softmax over its eight
   tokens, so the region walk, the causal mask and the epilogue agree;
2. with many regions the output and the natural LSE match the Triton reference,
   which reads the same packed words through a different schedule;
3. the logical start and the physical region stay independent: a region whose
   KV row differs from its logical position is still masked correctly;
4. repeated calls are bit-identical, so a captured graph replays the same rows;
5. the TP16 four-head layout is rejected instead of silently computed.

Run standalone on one PPU device::

    python test/ppu/unit/test_step5_decode_ppu_jit.py
    python test/ppu/unit/test_step5_decode_ppu_jit.py --bench
"""

import sys

import torch

from sglang.kernels.ops.attention.step5.decode_ppu import (
    STEP5_PPU_DECODE_PAGE_SIZE,
    STEP5_PPU_DECODE_REGIONS,
    allocate_step5_ppu_decode_workspace,
    can_use_step5_decode_ppu,
    step5_sparse_decode_ppu,
)

DEV = "cuda"
HEAD_DIM = 192
REGION_TOKENS = 8


def _selection(rows, width, live, *, seed=0):
    """Packed words whose physical regions are shuffled away from their starts.

    Returns the int64 packed rows, the int32 counts and the int32 causal lengths.
    Each row keeps ``live`` history regions plus the current one, matching what
    the selection/pack pair hands to attention.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    live = [live] * rows if isinstance(live, int) else list(live)
    packed = torch.full((rows, width), -1, dtype=torch.int64, device=DEV)
    counts = torch.zeros(rows, dtype=torch.int32, device=DEV)
    lengths = torch.zeros(rows, dtype=torch.int32, device=DEV)
    for row, count in enumerate(live):
        count = min(max(count, 1), width)
        starts = torch.arange(count, dtype=torch.int64) * REGION_TOKENS
        # Physical regions run backwards so a row's KV order never mirrors its
        # logical order; the causal mask must follow the starts, not the rows.
        physical = torch.arange(count - 1, -1, -1, dtype=torch.int64)
        physical = physical[torch.randperm(count, generator=gen)]
        words = (physical << 32) | (starts & 0xFFFFFFFF)
        packed[row, :count] = words.to(DEV)
        counts[row] = count
        lengths[row] = count * REGION_TOKENS
    return packed, counts, lengths


def _qkv(rows, heads, slots, *, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn((rows, heads, HEAD_DIM), generator=gen, device="cuda").to(
        torch.bfloat16
    )
    k = torch.randn((slots, 1, HEAD_DIM), generator=gen, device="cuda").to(
        torch.bfloat16
    )
    v = torch.randn((slots, 1, HEAD_DIM), generator=gen, device="cuda").to(
        torch.bfloat16
    )
    return q, k, v


def _reference(q, k, v, packed, counts, lengths, scale):
    """Triton reference: one pass over each row's packed regions."""
    rows, heads, _ = q.shape
    valid_rows = torch.full((1,), rows, dtype=torch.int32, device=DEV)
    query_start_loc = torch.arange(rows + 1, dtype=torch.int32, device=DEV)
    return step5_sparse_decode_partials_ref(
        q, k, v, packed, counts, lengths, valid_rows, query_start_loc, scale
    )


def step5_sparse_decode_partials_ref(
    q, k, v, packed, counts, lengths, valid_rows, query_start_loc, scale
):
    from sglang.kernels.ops.attention.step5.sparse_decode_triton import launch

    out = torch.empty_like(q)
    lse = torch.empty((q.shape[0], q.shape[1]), dtype=torch.float32, device=DEV)
    launch(
        q,
        k,
        v,
        packed,
        counts,
        lengths,
        valid_rows,
        out,
        lse,
        softmax_scale=scale,
    )
    return out, lse


def _run_ppu(q, k, v, packed, positions, scale):
    rows, heads, _ = q.shape
    workspace = allocate_step5_ppu_decode_workspace(
        rows=rows,
        heads=heads,
        kv_heads=k.shape[1],
        regions=packed.shape[1],
        device=q.device,
    )
    out = torch.empty_like(q)
    lse = torch.empty((rows, heads), dtype=torch.float32, device=DEV)
    step5_sparse_decode_ppu(
        q,
        k,
        v,
        packed=packed,
        query_positions=positions,
        out=out,
        lse=lse,
        scale=scale,
        workspace=workspace,
    )
    return out, lse


def _positions(lengths):
    return (lengths.to(torch.int64) - 1).contiguous()


def case_single_region(heads):
    """One selected region: compare against an FP32 softmax over eight tokens."""
    rows, slots = 3, 4 * STEP5_PPU_DECODE_PAGE_SIZE
    q, k, v = _qkv(rows, heads, slots, seed=1)
    # Region 0 owns tokens [0, 8) and starts at logical position 0, so the
    # reference can index K/V directly without any selection indirection.
    packed = torch.full(
        (rows, STEP5_PPU_DECODE_REGIONS), -1, dtype=torch.int64, device=DEV
    )
    packed[:, 0] = 0
    counts = torch.ones(rows, dtype=torch.int32, device=DEV)
    lengths = torch.full((rows,), REGION_TOKENS, dtype=torch.int32, device=DEV)
    scale = HEAD_DIM**-0.5
    out, lse = _run_ppu(q, k, v, packed, _positions(lengths), scale)

    ref_k = k[:REGION_TOKENS, 0].float()
    ref_v = v[:REGION_TOKENS, 0].float()
    scores = torch.einsum("rhd,kd->rhk", q.float(), ref_k) * scale
    weights = torch.softmax(scores, dim=-1)
    expected = torch.einsum("rhk,kd->rhd", weights, ref_v)
    torch.testing.assert_close(
        out.float(), expected, atol=3e-2, rtol=0, msg="single region output"
    )
    torch.testing.assert_close(
        lse, torch.logsumexp(scores, dim=-1), atol=5e-2, rtol=0, msg="single region LSE"
    )
    del counts


def case_matches_triton(heads, rows=8, live=(1, 4, 16, 33), seed=2):
    """Several regions: match the Triton reference on output and natural LSE."""
    rows, slots = 8, 64 * STEP5_PPU_DECODE_PAGE_SIZE
    q, k, v = _qkv(rows, heads, slots, seed=seed)
    packed, counts, lengths = _selection(rows, STEP5_PPU_DECODE_REGIONS, live, seed=seed)
    scale = HEAD_DIM**-0.5
    positions = _positions(lengths)
    ref_out, ref_lse = _reference(q, k, v, packed, counts, lengths, scale)
    out, lse = _run_ppu(q, k, v, packed, positions, scale)
    torch.testing.assert_close(out.float(), ref_out.float(), atol=4e-2, rtol=0)
    torch.testing.assert_close(lse, ref_lse, atol=5e-2, rtol=0)


def case_repeatable(heads):
    """The same inputs produce bit-identical output across launches."""
    rows = 4
    slots = 64 * STEP5_PPU_DECODE_PAGE_SIZE
    q, k, v = _qkv(rows, heads, slots, seed=3)
    packed, _, lengths = _selection(rows, STEP5_PPU_DECODE_REGIONS, 8, seed=3)
    scale = HEAD_DIM**-0.5
    positions = _positions(lengths)
    first, first_lse = _run_ppu(q, k, v, packed, positions, scale)
    second, second_lse = _run_ppu(q, k, v, packed, positions, scale)
    assert torch.equal(first, second), "decode output is not deterministic"
    assert torch.equal(first_lse, second_lse), "decode LSE is not deterministic"


def case_heads4_rejected():
    """Four query heads keep the Triton path instead of a silent fallback."""
    q, k, v = _qkv(2, 4, 64, seed=4)
    assert not can_use_step5_decode_ppu(q, k), "four-head layout must be rejected"


def bench(heads=8, rows=64, live=512, iters=100):
    slots = 1024 * STEP5_PPU_DECODE_PAGE_SIZE
    q, k, v = _qkv(rows, heads, slots, seed=7)
    packed, counts, lengths = _selection(rows, STEP5_PPU_DECODE_REGIONS, live, seed=7)
    scale = HEAD_DIM**-0.5
    positions = _positions(lengths)
    workspace = allocate_step5_ppu_decode_workspace(
        rows=rows,
        heads=heads,
        kv_heads=k.shape[1],
        regions=packed.shape[1],
        device=q.device,
    )
    out = torch.empty_like(q)
    lse = torch.empty((rows, heads), dtype=torch.float32, device=DEV)
    for _ in range(10):
        step5_sparse_decode_ppu(
            q,
            k,
            v,
            packed=packed,
            query_positions=positions,
            out=out,
            lse=lse,
            scale=scale,
            workspace=workspace,
        )
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        step5_sparse_decode_ppu(
            q,
            k,
            v,
            packed=packed,
            query_positions=positions,
            out=out,
            lse=lse,
            scale=scale,
            workspace=workspace,
        )
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    heads = 8
    if not can_use_step5_decode_ppu(*_qkv(2, 8, 64, seed=0)[:2]):
        raise SystemExit("JIT Step5 paged decode is unavailable on this device")
    case_heads4_rejected()
    case_single_region(heads)
    case_matches_triton(heads)
    case_repeatable(heads)
    print("step5 decode PPU JIT: all cases passed")
    if "--bench" in sys.argv:
        for h, live in ((8, 512), (16, 512), (8, 64), (16, 64)):
            ms = bench(h, live=live)
            print(f"heads={h} live={live}: {ms:.3f} ms per launch")


if __name__ == "__main__":
    main()
