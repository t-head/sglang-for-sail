# SPDX-License-Identifier: Apache-2.0
"""Precision tests for the Step5 DSA sparse prefill Triton kernel.

``_sparse_prefill`` flattens each group's query rows and all logical heads into
one MMA M tile (``q_slot * HEADS + head``), so a single program walks a group's
union of eight-token regions once and applies the same K/V to every head. These
cases pin the contract that decomposition must not break, each against a plain
float32 torch reference derived from the documented semantics (not from the
kernel body):

* exact membership -- bit ``j`` of ``membership[group, window]`` selects the
  ``j``-th query *slot* of that group, so different slots of one group must see
  different token sets even though they share an M tile and the K/V loads.
* inclusive causality -- a key whose ``logical_tok`` equals the row's
  ``query_positions`` is attended; an exclusive rewrite leaves that row empty.
* union bookkeeping -- only the first ``union_counts[group]`` windows exist; the
  capacity tail may hold arbitrary garbage and must never be read.
* predicate drops -- ``phys_region < 0``, ``phys_region >= slots // 8`` and
  ``logical_region < 0`` entries contribute nothing.
* row ownership -- only rows inside a group's ``[start, end)`` range are written.
* empty rows -- zero output and ``-inf`` LSE, never NaN.

Group sizes, union lengths (including tails that are not a multiple of the
kernel's ``BLOCK_N``) and head counts 4/8/16 -- i.e. the head blocks the
launcher derives from them -- are covered by the case matrix.

Run standalone on one idle PPU device::

    python test/ppu/unit/test_step5_sparse_prefill_triton.py
"""

import math
import unittest

import torch

from sglang.kernels.ops.attention.step5.sparse_prefill import step5_sparse_prefill

REGION_TOKENS = 8
HEAD_DIM = 192
SCALE = 1.0 / math.sqrt(HEAD_DIM)


def _q_group(heads: int) -> int:
    """Max query rows packed in one group, as the backend builds them."""
    return min(16, 128 // heads)


def _build_metadata(
    *,
    heads: int,
    group_rows,
    capacity: int,
    union_counts,
    slots_div8: int,
    empty_groups=(),
    seed: int = 0,
):
    """Synthetic prefill metadata for the given per-group row counts.

    The union is a sorted set of distinct logical regions whose physical region
    is a fixed permutation of the logical one (a logical region always maps to
    one physical region), and each group's query positions sit near the end of
    its union span so that the causal predicate keeps a large but per-row
    different prefix. Live windows are additionally sprinkled with the three
    dead forms the kernel must drop.
    """
    gen = torch.Generator(device="cuda").manual_seed(seed)
    dev = "cuda"
    rows = int(sum(group_rows))
    slots = slots_div8 * REGION_TOKENS
    assert capacity <= slots_div8

    q = torch.randn(rows, heads, HEAD_DIM, dtype=torch.bfloat16, device=dev, generator=gen)
    k = torch.randn(slots, 1, HEAD_DIM, dtype=torch.bfloat16, device=dev, generator=gen)
    v = torch.randn(slots, 1, HEAD_DIM, dtype=torch.bfloat16, device=dev, generator=gen)

    gstart = torch.tensor(
        [sum(group_rows[:i]) for i in range(len(group_rows))], dtype=torch.int32, device=dev
    )
    gend = (gstart + torch.tensor(group_rows, dtype=torch.int32, device=dev)).contiguous()

    groups = len(group_rows)
    ulog = torch.sort(
        torch.stack(
            [torch.randperm(slots_div8, device=dev, generator=gen)[:capacity] for _ in range(groups)]
        ).to(torch.int32),
        dim=1,
    ).values
    uphys = ((ulog.to(torch.int64) * 5 + 1) % slots_div8).to(torch.int32)
    memb = (torch.rand(groups, capacity, device=dev, generator=gen) < 0.7).to(torch.int32)
    qpos = torch.zeros(rows, dtype=torch.int32, device=dev)

    for g in range(groups):
        count = int(union_counts[g])
        n_rows = int(group_rows[g])
        # The workspace beyond union_counts is ``torch.empty`` in production, so
        # fill it with entries that would be attended if read: a valid slot, a
        # causal logical region and every membership bit set. A kernel that
        # walks CAPACITY instead of union_counts folds them into the softmax.
        tail = torch.arange(count, capacity, device=dev)
        ulog[g, count:] = (tail % slots_div8).to(torch.int32)
        uphys[g, count:] = ((tail * 5 + 1) % slots_div8).to(torch.int32)
        memb[g, count:] = -1
        for _ in range(min(2, count // 3)):
            pick = int(torch.randint(0, count, (1,), device=dev, generator=gen))
            kind = int(torch.randint(0, 3, (1,), device=dev, generator=gen))
            memb[g, pick] = 0x1
            if kind == 0:
                uphys[g, pick] = -1
            elif kind == 1:
                uphys[g, pick] = slots_div8
            else:
                ulog[g, pick] = -1
                uphys[g, pick] = -1
        # Query positions trail the union span so causality keeps a prefix.
        span = int(ulog[g, :count].clamp_min(0).max()) * REGION_TOKENS + REGION_TOKENS if count else 0
        base = int(gstart[g])
        qpos[base : base + n_rows] = torch.arange(
            span - (n_rows - 1), span + 1, dtype=torch.int32, device=dev
        ).clamp_min(0)
    for g in empty_groups:
        memb[g].zero_()

    out = torch.zeros(rows, heads, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    lse = torch.zeros(rows, heads, dtype=torch.float32, device=dev)
    return dict(
        q=q, k=k, v=v, uphys=uphys, ulog=ulog,
        ucounts=torch.tensor(union_counts, dtype=torch.int32, device=dev),
        memb=memb, gstart=gstart, gend=gend, qpos=qpos, out=out, lse=lse,
        rows=rows, heads=heads, capacity=capacity, slots_div8=slots_div8,
    )


def _reference(case):
    """Float32 reference straight from the documented union semantics."""
    q, k, v = case["q"], case["k"], case["v"]
    uphys, ulog, memb = case["uphys"], case["ulog"], case["memb"]
    q_group = _q_group(case["heads"])
    rows, heads = case["rows"], case["heads"]
    slots = k.shape[0]

    out = torch.zeros(rows, heads, HEAD_DIM, dtype=torch.float32, device=q.device)
    lse = torch.full((rows, heads), float("-inf"), dtype=torch.float32, device=q.device)
    for g in range(uphys.shape[0]):
        base = int(case["gstart"][g])
        end = int(case["gend"][g])
        n_rows = min(max(end - base, 0), q_group)
        count = min(int(case["ucounts"][g]), case["capacity"])
        if n_rows <= 0 or count <= 0:
            continue
        row = base + torch.arange(n_rows, device=q.device)
        owned = row < rows
        if not bool(owned.any()):
            continue
        row, slot = row[owned], torch.arange(n_rows, device=q.device)[owned]

        n_tok = count * REGION_TOKENS
        tok = torch.arange(n_tok, device=q.device)
        win, lane = tok // REGION_TOKENS, tok % REGION_TOKENS
        phys, log = uphys[g, win], ulog[g, win]
        live = (phys >= 0) & (phys < case["slots_div8"]) & (log >= 0)
        phys_tok = (phys * REGION_TOKENS + lane).clamp(0, slots - 1).long()
        logical_tok = log * REGION_TOKENS + lane

        selected = ((memb[g, win].to(torch.int64)[None, :] >> slot[:, None]) & 1).bool()
        keep = selected & (logical_tok[None, :] <= case["qpos"][row][:, None]) & live[None, :]

        k_g = k.index_select(0, phys_tok)[:, 0].float()
        v_g = v.index_select(0, phys_tok)[:, 0].float()
        score = (q[row].float() @ k_g.T) * SCALE
        score = score.masked_fill(~keep[:, None, :], float("-inf"))
        m = score.max(dim=2).values
        has = torch.isfinite(m)
        m_safe = torch.where(has, m, torch.zeros_like(m))
        p = torch.where(torch.isfinite(score), torch.exp(score - m_safe[:, :, None]), 0.0)
        denom = p.sum(dim=2)
        ok = has & (denom > 0)
        # Index the (row, head) pair: heads of one row are independent, and a
        # row may be live for some heads only after masking anomalies.
        pair_row = row[:, None].expand(-1, heads)[ok]
        pair_head = torch.arange(heads, device=q.device)[None, :].expand_as(ok)[ok]
        out[pair_row, pair_head] = (p @ v_g)[ok] / denom[ok][..., None]
        lse[pair_row, pair_head] = m[ok] + torch.log(denom[ok])
    return out, lse


class TestStep5SparsePrefillTriton(unittest.TestCase):
    def _run(self, case):
        step5_sparse_prefill(
            case["q"], case["k"], case["v"],
            union_physical=case["uphys"], union_logical=case["ulog"],
            union_counts=case["ucounts"], membership=case["memb"],
            group_query_start=case["gstart"], group_query_end=case["gend"],
            query_positions=case["qpos"], out=case["out"], lse=case["lse"],
            softmax_scale=SCALE,
        )
        torch.cuda.synchronize()
        return case["out"], case["lse"]

    def _assert_matches_reference(self, name, case):
        got_out, got_lse = self._run(case)
        ref_out, ref_lse = _reference(case)
        self._assert_close(name, got_out, got_lse, ref_out, ref_lse)

    def _assert_close(self, name, got_out, got_lse, ref_out, ref_lse):
        self.assertFalse(torch.isnan(got_out.float()).any(), f"{name}: NaN in out")
        self.assertFalse(torch.isnan(got_lse).any(), f"{name}: NaN in LSE")

        # BF16 output against an fp32 reference: no element may be off by more
        # than ~1 bf16 ulp of full scale (the PV product rounds probabilities to
        # bf16), and the mean error stays far below that so a systematic drift --
        # a wrong mask, a shifted head or slot, a duplicated window -- cannot
        # hide behind per-element rounding.
        err = (got_out.float() - ref_out).abs()
        tol = 4e-3 * max(1.0, ref_out.abs().max().item())
        self.assertLessEqual(
            err.max().item(), tol, f"{name}: out max|d|={err.max().item():.5f} > tol={tol:.5f}"
        )
        self.assertLessEqual(
            err.mean().item(), 0.1 * tol,
            f"{name}: out mean|d|={err.mean().item():.6f} > {0.1 * tol:.6f}",
        )

        # LSE: the -inf pattern is part of the contract (empty rows), values are
        # compared only where the reference produced one.
        self.assertTrue(
            torch.equal(torch.isfinite(got_lse), torch.isfinite(ref_lse)),
            f"{name}: LSE finiteness pattern differs",
        )
        finite = torch.isfinite(ref_lse)
        if bool(finite.any()):
            d = (got_lse[finite] - ref_lse[finite]).abs()
            self.assertLessEqual(
                d.max().item(),
                2e-2 + 2e-3 * ref_lse[finite].abs().max().item(),
                f"{name}: LSE max|d|={d.max().item():.5f}",
            )

    def test_matches_fp32_reference(self):
        """Case matrix over head counts, group sizes and union tails."""
        cases = [
            # name, heads, per-group row counts, capacity, union counts, slots//8
            ("heads4-tail-groups", 4, [16, 16, 3], 9, [9, 9, 9], 64),
            ("heads4-cap512-tails", 4, [16, 16, 16, 16], 512, [512, 511, 3, 9], 4096),
            ("heads4-union-one-block", 4, [16, 15], 9, [8, 8], 128),
            ("heads8-partial-groups", 8, [16, 8, 5], 33, [33, 3, 17], 256),
            ("heads16-qgroup8", 16, [8, 8, 1], 64, [64, 1, 2], 512),
            ("heads4-empty-groups", 4, [4, 16, 16, 2], 8, [8, 8, 8, 8], 64),
            ("heads4-scale-256rows", 4, [16] * 16, 512, [512] * 16, 8192),
        ]
        for name, heads, group_rows, capacity, union_counts, slots_div8 in cases:
            with self.subTest(name):
                empty = (3,) if name == "heads4-empty-groups" else ()
                case = _build_metadata(
                    heads=heads, group_rows=group_rows, capacity=capacity,
                    union_counts=union_counts, slots_div8=slots_div8,
                    empty_groups=empty, seed=len(name),
                )
                self._assert_matches_reference(name, case)

    def test_zero_union_count_writes_empty_rows(self):
        """A group whose union is empty must write zeros and -inf, not garbage."""
        case = _build_metadata(
            heads=4, group_rows=[4, 4], capacity=8, union_counts=[0, 0], slots_div8=64
        )
        out, lse = self._run(case)
        self.assertFalse(bool(out.any()), "empty unions produced non-zero output")
        self.assertTrue(bool(torch.isinf(lse).all()) and bool((lse < 0).all()), "expected -inf LSE")

    def test_causal_boundary_is_inclusive(self):
        """The key at ``logical_tok == query_positions[row]`` must be attended."""
        case = _build_metadata(
            heads=4, group_rows=[1], capacity=4, union_counts=[1], slots_div8=64, seed=7
        )
        case["uphys"][0, 0] = 2
        case["ulog"][0, 0] = 0  # tokens 0..7
        case["qpos"][0] = 5  # keep tokens 0..5, drop 6..7
        case["memb"][0, 0] = 1
        out, lse = self._run(case)
        ref_out, ref_lse = _reference(case)
        self.assertTrue(torch.isfinite(lse[0]).all(), "boundary row treated as empty")
        # One row of six attended tokens: too few elements for the matrix's
        # mean-error bar, so only bound every element by one bf16 ulp here.
        err = (out.float() - ref_out).abs()
        self.assertLessEqual(err.max().item(), 4e-3 * max(1.0, ref_out.abs().max().item()))
        self.assertAlmostEqual(lse[0, 0].item(), ref_lse[0, 0].item(), places=2)

        # One token past the boundary and the same row must be empty.
        case2 = _build_metadata(
            heads=4, group_rows=[1], capacity=4, union_counts=[1], slots_div8=64, seed=7
        )
        case2["uphys"][0, 0] = 2
        case2["ulog"][0, 0] = 1  # tokens 8..15
        case2["qpos"][0] = 7  # 8 > 7: nothing is causal
        case2["memb"][0, 0] = 1
        out2, lse2 = self._run(case2)
        self.assertTrue(bool(torch.isinf(lse2).all()), "non-causal row was attended")

    def test_rows_outside_groups_untouched(self):
        """Rows no group owns keep whatever the caller left in the buffers."""
        rows, heads = 10, 4
        dev = "cuda"
        out = torch.full((rows, heads, HEAD_DIM), 3.0, dtype=torch.bfloat16, device=dev)
        lse = torch.full((rows, heads), 42.0, dtype=torch.float32, device=dev)
        gen = torch.Generator(device=dev).manual_seed(1)
        step5_sparse_prefill(
            torch.randn(rows, heads, HEAD_DIM, dtype=torch.bfloat16, device=dev, generator=gen),
            torch.randn(256, 1, HEAD_DIM, dtype=torch.bfloat16, device=dev, generator=gen),
            torch.randn(256, 1, HEAD_DIM, dtype=torch.bfloat16, device=dev, generator=gen),
            union_physical=torch.zeros(2, 4, dtype=torch.int32, device=dev),
            union_logical=torch.zeros(2, 4, dtype=torch.int32, device=dev),
            union_counts=torch.tensor([1, 1], dtype=torch.int32, device=dev),
            membership=torch.full((2, 4), 0xFFFF, dtype=torch.int32, device=dev),
            group_query_start=torch.tensor([0, 4], dtype=torch.int32, device=dev),
            group_query_end=torch.tensor([4, 6], dtype=torch.int32, device=dev),
            query_positions=torch.arange(rows, dtype=torch.int32, device=dev),
            out=out, lse=lse, softmax_scale=SCALE,
        )
        torch.cuda.synchronize()
        self.assertTrue(bool((out[6:] == 3.0).all()), "kernel wrote rows outside every group")
        self.assertTrue(bool((lse[6:] == 42.0).all()), "kernel wrote LSE outside every group")
        self.assertTrue(bool(torch.isfinite(lse[:6]).all()), "owned rows were not written")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "needs a CUDA/PPU device"
    unittest.main(verbosity=2)
