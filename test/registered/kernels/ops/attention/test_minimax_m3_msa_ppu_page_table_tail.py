"""Regression for the prefill "invalid PA on page-walk" UMD fault (packed page table tail).

A prefill chunk sizes the indexer/top-k tile window from the aligned tile
count, so short requests select padding tiles whose resolution reads the
capacity-sized packed page table past sum_pages. The table is zero-filled
and sized to cover that overshoot; these tests pin both properties.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.minimax_sparse_ops.msa_ppu import (
    _aligned_max_k_tiles,
    msa_ppu_forward_extend,
    ppu_msa_available,
)
from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.test_utils import CustomTestCase

register_ppu_ci(est_time=120, suite="stage-a-test-1-gpu-ppu")

PAGE = 128
HEAD_DIM = 128

_HAS_MODULE = True
HAS_GPU = torch.cuda.is_available()


def _table_capacity(bs: int, max_ctx: int) -> int:
    return (bs - 1) * ((max_ctx + PAGE - 1) // PAGE) + _aligned_max_k_tiles(max_ctx)


def _run_mixed_length_extend(lens, exts, max_ctx):
    """Run msa_ppu_forward_extend on a mixed-length batch; return (out, state, pages, sum_pages, capacity)."""
    device = torch.device("cuda")
    h_idx, h_q, h_kv, topk = 2, 32, 2, 16

    pages = [(kv + PAGE - 1) // PAGE for kv in lens]
    page_starts = [sum(pages[:i]) for i in range(len(pages))]
    sum_pages = sum(pages)
    pool_slots = sum_pages * PAGE
    req_to_token = torch.zeros(len(lens), max_ctx, dtype=torch.int64, device=device)
    for i, n in enumerate(pages):
        req_to_token[i, : n * PAGE] = (
            torch.arange(n * PAGE, dtype=torch.int64, device=device)
            + page_starts[i] * PAGE
        )

    # Poison the size class the capacity-sized page table is allocated from
    # so an undersized or uninitialized table would hand back stale non-zero
    # page ids, as a long-lived server allocator does.
    capacity = _table_capacity(len(lens), max_ctx)
    poison = [
        torch.empty(capacity, dtype=torch.int32, device=device) for _ in range(16)
    ]
    for buf in poison:
        buf.fill_(0x7FFFFFFF)
    del poison

    backend = SimpleNamespace(
        _ppu_msa_state=None,
        _ppu_msa_attend=True,
        _ppu_msa_use_fp8_kvcache=False,
        _ppu_msa_indexer_fp8=True,
        _ppu_msa_score_dtype=torch.bfloat16,
        _ppu_msa_num_idx_heads=h_idx,
        _ppu_msa_num_q_heads=h_q,
        num_kv_heads=h_kv,
        topk_blocks=topk,
        max_context_len=max_ctx,
        init_blocks=0,
        local_blocks=1,
        req_to_token=req_to_token,
        kv_pool=SimpleNamespace(main_pool=SimpleNamespace(size=pool_slots)),
        _ppu_msa_idx_scale=HEAD_DIM**-0.5,
        _ppu_msa_scale=HEAD_DIM**-0.5,
    )
    total_q = sum(exts)
    forward_batch = SimpleNamespace(
        seq_lens=torch.tensor(lens, dtype=torch.int32, device=device),
        seq_lens_cpu=torch.tensor(lens, dtype=torch.int64),
        req_pool_indices=torch.arange(len(lens), dtype=torch.int64, device=device),
    )
    q = torch.randn(total_q, h_q, HEAD_DIM, dtype=torch.bfloat16, device=device)
    idx_q = torch.randn(total_q, h_idx, HEAD_DIM, dtype=torch.bfloat16, device=device)
    idx_k_cache = torch.randn(
        pool_slots, 1, HEAD_DIM, dtype=torch.bfloat16, device=device
    ).to(torch.float8_e4m3fn)
    k_cache = torch.randn(
        pool_slots, h_kv, HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    v_cache = torch.randn_like(k_cache)
    cu_seqlens = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(exts), 0).tolist()),
        dtype=torch.int32,
        device=device,
    )
    prefix_lens = torch.tensor(
        [kv - e for kv, e in zip(lens, exts)], dtype=torch.int32, device=device
    )

    _, out = msa_ppu_forward_extend(
        backend,
        q,
        idx_q,
        idx_k_cache,
        k_cache,
        v_cache,
        forward_batch,
        cu_seqlens,
        forward_batch.seq_lens,
        prefix_lens,
        torch.tensor(exts, dtype=torch.int32, device=device),
        exts,
        max_seqlen_q=max(exts),
        layer_id=0,
    )
    torch.cuda.synchronize()
    return out, backend._ppu_msa_state, pages, sum_pages, capacity


@unittest.skipUnless(
    _HAS_MODULE and HAS_GPU and ppu_msa_available(),
    "msa_ppu module, msa_for_sail, or CUDA/PPU device unavailable",
)
class TestPrefillPackedPageTableTail(CustomTestCase):
    def test_mixed_length_batch_resolves_padding_pages_to_page_zero(self):
        out, state, pages, sum_pages, capacity = _run_mixed_length_extend(
            [8192, 1536], [512, 1536], 1 << 20
        )
        self.assertTrue(torch.isfinite(out.float()).all())
        # Premise: the short request really selected padding tiles past its
        # own page count, i.e. the tail of the table is reachable.
        short_rows = state["topk_buf"][512:]
        self.assertGreaterEqual(int(short_rows.max()), pages[1])
        # The unwritten tail must read as physical page 0, not stale garbage.
        table = state["kv_indices"]
        self.assertEqual(table.numel(), capacity)
        self.assertEqual(int(table[sum_pages:].abs().max()), 0)

    def test_small_context_aligned_candidates_stay_in_table(self):
        # One 1024-token request: 8 pages, but the selector window is the
        # aligned tile count (128), so padding ids far exceed the request's
        # page count. The table must be sized and zero-filled for that.
        out, state, pages, sum_pages, capacity = _run_mixed_length_extend(
            [1024], [1024], 1024
        )
        self.assertTrue(torch.isfinite(out.float()).all())
        topk_rows = state["topk_buf"]
        self.assertGreaterEqual(int(topk_rows.max()), pages[0])
        table = state["kv_indices"]
        self.assertEqual(table.numel(), capacity)
        self.assertEqual(int(table[sum_pages:].abs().max()), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
