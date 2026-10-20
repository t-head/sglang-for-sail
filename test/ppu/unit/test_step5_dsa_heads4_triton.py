# SPDX-License-Identifier: Apache-2.0
"""Synthetic heads=4 validation for the Step5 DSA sparse Triton kernels.

TP16 shards the 64 attention heads into ``heads = 64 // 16 = 4`` per rank. The
CuTe sparse prefill/decode kernels are gated to sm90/sm100, so PPU (sm80) always
runs the portable Triton path. This test drives the two dispatchers directly --
``step5_sparse_prefill`` and ``step5_sparse_decode_partials`` -- so it exercises
both the ``q.shape[1] in (4, 8, 16)`` guard and the Triton routing, and compares
the BF16 attention / FP32 natural LSE against a plain torch reference.

``heads`` is parametrized over (4, 8, 16): 8 and 16 are the pre-existing TP8/TP4
configs and act as controls that prove the reference is correct, while 4 is the
new TP16 config under validation. The metadata (groups, unions, membership,
packed regions, causal limits) is head-independent, so every head count replays
the exact same selection pattern against per-head Q.

Run standalone on one idle PPU device::

    python test/ppu/unit/test_step5_dsa_heads4_triton.py
"""

import math

import torch

from sglang.kernels.ops.attention.step5.sparse_decode import (
    step5_sparse_decode_partials,
)
from sglang.kernels.ops.attention.step5.sparse_prefill import step5_sparse_prefill

DEV = "cuda"
SCALE = 1.0 / math.sqrt(192)


def _pack(phys_region: int, start_tok: int) -> int:
    """Decode packed region word: high 32 = physical region, low 32 = start token."""
    return (phys_region << 32) | start_tok


def _ref_decode(q, k, v, counts, packed, causal, rows, heads, max_phys_regions):
    """Mirror ``sparse_decode_triton._sparse_decode``: per (row, head) online softmax.

    A token is kept only when its region word is non-negative, its physical slot
    is in range, and its logical token is strictly below the row's causal length.
    """
    out = torch.zeros(rows, heads, 192, dtype=torch.float32, device=q.device)
    lse = torch.full((rows, heads), float("-inf"), dtype=torch.float32, device=q.device)
    for r in range(rows):
        cnt = int(counts[r])
        cl = int(causal[r])
        ks, vs = [], []
        for win in range(cnt):
            word = int(packed[r, win])
            if word < 0:
                continue
            phys_region = word >> 32
            start_tok = word & 0xFFFFFFFF
            if not 0 <= phys_region < max_phys_regions:
                continue
            for lane in range(8):
                if start_tok + lane < cl:
                    phys_tok = phys_region * 8 + lane
                    ks.append(k[phys_tok, 0].float())
                    vs.append(v[phys_tok, 0].float())
        for h in range(heads):
            if not ks:
                continue
            K = torch.stack(ks)
            V = torch.stack(vs)
            s = (K @ q[r, h].float()) * SCALE
            m = s.max()
            p = torch.exp(s - m)
            l = p.sum()
            out[r, h] = (p[:, None] * V).sum(0) / l
            lse[r, h] = m + torch.log(l)
    return out, lse


def _ref_prefill(
    q, k, v, uphys, ulog, ucounts, memb, gstart, gend, qpos, rows, heads,
    slots_div8, q_group_cap,
):
    """Mirror ``sparse_prefill_triton._sparse_prefill``: per group/head union softmax.

    Each group walks its ``[gstart, gend)`` query rows (clamped to ``q_group_cap
    = 128 // heads``); a union region is attended by a query only when the
    matching membership bit is set, the physical slot is in range, the logical
    region is non-negative, and the logical token is ``<= query_positions[row]``.
    """
    out = torch.zeros(rows, heads, 192, dtype=torch.float32, device=q.device)
    lse = torch.full((rows, heads), float("-inf"), dtype=torch.float32, device=q.device)
    groups = uphys.shape[0]
    capacity = uphys.shape[1]
    for g in range(groups):
        q_base = int(gstart[g])
        q_end = int(gend[g])
        q_tile = min(max(q_end - q_base, 0), q_group_cap)
        union_count = min(int(ucounts[g]), capacity)
        for q_slot in range(q_tile):
            row = q_base + q_slot
            if row >= rows:
                continue
            q_pos = int(qpos[row])
            ks, vs = [], []
            for win in range(union_count):
                pr = int(uphys[g, win])
                lr = int(ulog[g, win])
                if not ((int(memb[g, win]) >> q_slot) & 1):
                    continue
                if not 0 <= pr < slots_div8 or lr < 0:
                    continue
                for lane in range(8):
                    if lr * 8 + lane <= q_pos:
                        phys_tok = pr * 8 + lane
                        ks.append(k[phys_tok, 0].float())
                        vs.append(v[phys_tok, 0].float())
            for h in range(heads):
                if not ks:
                    continue
                K = torch.stack(ks)
                V = torch.stack(vs)
                s = (K @ q[row, h].float()) * SCALE
                m = s.max()
                p = torch.exp(s - m)
                l = p.sum()
                out[row, h] = (p[:, None] * V).sum(0) / l
                lse[row, h] = m + torch.log(l)
    return out, lse


def run_decode(heads: int):
    torch.manual_seed(0)
    rows, slots = 3, 128  # slots % 64 == 0; 16 physical regions
    max_phys_regions = slots // 8
    q = torch.randn(rows, heads, 192, dtype=torch.bfloat16, device=DEV)
    k = torch.randn(slots, 1, 192, dtype=torch.bfloat16, device=DEV)
    v = torch.randn(slots, 1, 192, dtype=torch.bfloat16, device=DEV)
    counts = torch.tensor([2, 1, 3], dtype=torch.int32, device=DEV)
    packed = torch.full((rows, 513), -1, dtype=torch.int64, device=DEV)
    packed[0, 0] = _pack(1, 8)
    packed[0, 1] = _pack(2, 16)
    packed[1, 0] = _pack(3, 0)
    packed[2, 0] = _pack(0, 0)
    packed[2, 1] = _pack(4, 24)
    packed[2, 2] = _pack(5, 40)
    causal = torch.tensor([20, 8, 100], dtype=torch.int32, device=DEV)
    query_start_loc = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=DEV)
    valid_rows = torch.tensor([rows], dtype=torch.int32, device=DEV)
    out = torch.zeros(rows, heads, 192, dtype=torch.bfloat16, device=DEV)
    lse = torch.zeros(rows, heads, dtype=torch.float32, device=DEV)

    # No-split schedule: the Triton decode path requires work == rows, mtp_q_len 1.
    step5_sparse_decode_partials(
        q, k, v,
        counts=counts, packed_regions=packed, causal_lengths=causal,
        query_start_loc=query_start_loc, valid_rows=valid_rows,
        out=out, lse=lse, softmax_scale=SCALE,
    )
    torch.cuda.synchronize()

    ref_out, ref_lse = _ref_decode(
        q, k, v, counts, packed, causal, rows, heads, max_phys_regions
    )
    assert torch.allclose(out.float(), ref_out, atol=2e-2, rtol=2e-2), (
        f"decode heads={heads} out mismatch: "
        f"max|d|={(out.float() - ref_out).abs().max().item():.4f}"
    )
    assert torch.allclose(lse, ref_lse, atol=1e-2, rtol=1e-2), (
        f"decode heads={heads} lse mismatch: "
        f"max|d|={(lse - ref_lse).abs().max().item():.4f}"
    )
    print(f"PASS decode heads={heads}")


def run_prefill(heads: int):
    torch.manual_seed(0)
    rows, slots, groups, capacity = 7, 256, 2, 8  # slots % 64 == 0; 32 regions
    slots_div8 = slots // 8
    q_group_cap = 128 // heads  # Triton kernel's Q_GROUP upper bound
    q = torch.randn(rows, heads, 192, dtype=torch.bfloat16, device=DEV)
    k = torch.randn(slots, 1, 192, dtype=torch.bfloat16, device=DEV)
    v = torch.randn(slots, 1, 192, dtype=torch.bfloat16, device=DEV)
    # group 0 -> rows 0..3 (positions 10..13); group 1 -> rows 4..6 (20..22).
    gstart = torch.tensor([0, 4], dtype=torch.int32, device=DEV)
    gend = torch.tensor([4, 7], dtype=torch.int32, device=DEV)
    qpos = torch.tensor([10, 11, 12, 13, 20, 21, 22], dtype=torch.int32, device=DEV)
    uphys = torch.zeros(groups, capacity, dtype=torch.int32, device=DEV)
    ulog = torch.zeros(groups, capacity, dtype=torch.int32, device=DEV)
    uphys[0, 0], ulog[0, 0] = 1, 1  # tokens 8..15
    uphys[0, 1], ulog[0, 1] = 2, 2  # tokens 16..23
    uphys[1, 0], ulog[1, 0] = 0, 0  # tokens 0..7
    uphys[1, 1], ulog[1, 1] = 3, 3  # tokens 24..31
    ucounts = torch.tensor([2, 2], dtype=torch.int32, device=DEV)
    memb = torch.zeros(groups, capacity, dtype=torch.int32, device=DEV)
    memb[0, 0], memb[0, 1] = 0b1111, 0b0011  # g0: all 4 rows; then rows 0,1
    memb[1, 0], memb[1, 1] = 0b0111, 0b0101  # g1: all 3 rows; then slots 0,2
    out = torch.zeros(rows, heads, 192, dtype=torch.bfloat16, device=DEV)
    lse = torch.zeros(rows, heads, dtype=torch.float32, device=DEV)

    step5_sparse_prefill(
        q, k, v,
        union_physical=uphys, union_logical=ulog, union_counts=ucounts,
        membership=memb, group_query_start=gstart, group_query_end=gend,
        query_positions=qpos, out=out, lse=lse, softmax_scale=SCALE,
    )
    torch.cuda.synchronize()

    ref_out, ref_lse = _ref_prefill(
        q, k, v, uphys, ulog, ucounts, memb, gstart, gend, qpos, rows, heads,
        slots_div8, q_group_cap,
    )
    assert torch.allclose(out.float(), ref_out, atol=2e-2, rtol=2e-2), (
        f"prefill heads={heads} out mismatch: "
        f"max|d|={(out.float() - ref_out).abs().max().item():.4f}"
    )
    assert torch.allclose(lse, ref_lse, atol=1e-2, rtol=1e-2), (
        f"prefill heads={heads} lse mismatch: "
        f"max|d|={(lse - ref_lse).abs().max().item():.4f}"
    )
    print(f"PASS prefill heads={heads}")


def main():
    assert torch.cuda.is_available(), "needs a CUDA/PPU device"
    cap = torch.cuda.get_device_capability()
    print(f"device={torch.cuda.get_device_name()} capability={cap}")
    # 8 and 16 are the pre-existing TP8/TP4 controls; 4 is the new TP16 target.
    for heads in (8, 16, 4):
        run_decode(heads)
        run_prefill(heads)
    print("ALL PASS")


if __name__ == "__main__":
    main()
