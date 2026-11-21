"""Row-contract tests for the kpool indexer under breakable prefill CUDA graphs.

Runs the real producers (_get_topk_ragged_kpool / _get_topk_ragged_kpool_plan)
with only the kernel and cache boundaries mocked, so the graph-context guards
themselves are exercised. Card-free CPU module.

Run:
    python3 -m pytest -q test/registered/unit/layers/test_kpool_bcg_prefill_row_contract.py
Env: none required (no device needed; do NOT set PLA_BYPASS_EXEC /
PPU_LIB_SHOW_PARAMS — irrelevant here but kept neutral for CI parity).
"""

import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.dsa import dsa_indexer_kpool as kip
from sglang.srt.layers.attention.dsa.dsa_indexer_kpool import IndexerKPool
from sglang.srt.layers.attention.dsa.dsa_topk_backend import TopkTransformMethod
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


# ---------------------------------------------------------------- fixtures


@dataclass(frozen=True)
class FrozenAttn:
    """Mirrors the real capture metadata: plain assignment must fail."""

    kpool_extend_plan: object = None
    kpool_extend_pooled_page_tables: object = None
    kpool_extend_zero_starts: object = None
    kpool_seqlens_expanded: object = None
    topk_indices_offset: object = None
    page_size: int = 64


class _Meta:
    def __init__(self, attn_metadata, topk_transform_method, seqlens_expanded):
        self.attn_metadata = attn_metadata
        self.topk_transform_method = topk_transform_method
        self._seqlens_expanded = seqlens_expanded

    def get_seqlens_expanded(self):
        return self._seqlens_expanded

    def get_page_table_64(self):
        return torch.zeros(8, 8, dtype=torch.int32)


def _plan(rows, marker=None):
    pool = (
        torch.full((rows,), 777, dtype=torch.int32)
        if marker
        else torch.arange(rows, dtype=torch.int32)
    )
    return SimpleNamespace(
        seq_lens_expanded=torch.zeros(rows, dtype=torch.int32),
        pooled_seq_lens_expanded=pool,
        ragged_q_ks=torch.zeros(rows, dtype=torch.int32),
        ragged_q_ke=torch.zeros(rows, dtype=torch.int32),
        ragged_total_k_rows=0,
        ragged_k_u8=None,
        ragged_k_scale=None,
        ragged_concat_page_table=None,
        ragged_paged_page_table=None,
        ragged_paged_page_table_row_index=None,
    )


def _fb(bs, extend_lens, seq_lens_list, mode_name="extend"):
    total = sum(extend_lens)

    class _Mode:
        def is_idle(self):
            return False

        def is_extend_without_speculative(self):
            return mode_name == "extend"

        def is_target_verify(self):
            return mode_name == "target_verify"

        def is_draft_extend_v2(self):
            return mode_name == "draft_extend_v2"

        def is_decode_or_idle(self):
            return mode_name in ("decode", "idle")

    return SimpleNamespace(
        forward_mode=_Mode(),
        batch_size=bs,
        extend_num_tokens=total,
        extend_seq_lens_cpu=list(extend_lens),
        seq_lens_cpu=torch.tensor(seq_lens_list, dtype=torch.int64),
        seq_lens=torch.tensor(seq_lens_list, dtype=torch.int64),
        req_pool_indices=torch.arange(bs, dtype=torch.int64),
    )


class _LogitsStub:
    def __init__(self):
        self.calls = []

    def __call__(self, *, q_fp8, k_fp8, k_scale, weights, starts, ends, clean_logits):
        self.calls.append(
            {
                "rows": q_fp8.shape[0],
                "starts": starts.clone(),
                "ends": ends.clone(),
            }
        )
        return torch.zeros(q_fp8.shape[0], 4, dtype=torch.float32)


class _TopkStub:
    def __init__(self, index_topk, index_kpool):
        self.calls = []
        self.width = index_topk + index_kpool - 1

    def __call__(
        self,
        logits,
        pool_lens,
        seq_lens=None,
        page_table=None,
        topk_offsets=None,
        row_starts=None,
        out_rows=None,
        page_table_row_index=None,
        topk_v2_plan=None,
    ):
        self.calls.append(
            {
                "n_rows": logits.shape[0],
                "pool_lens": pool_lens.clone() if pool_lens is not None else None,
                "seq_lens": None if seq_lens is None else seq_lens.clone(),
                "topk_offsets": (
                    None if topk_offsets is None else topk_offsets.clone()
                ),
            }
        )
        return torch.full((logits.shape[0], self.width), -1, dtype=torch.int32)


def _make_indexer(index_topk=32, index_kpool=4):
    idx = IndexerKPool.__new__(IndexerKPool)
    idx.index_topk = index_topk
    idx.index_kpool = index_kpool
    idx.use_int8 = True
    idx.alt_stream = None
    idx.compress_gate_stream = None
    idx.block_size = 128
    idx.scale_fmt = None
    idx.index_kpool_compress = False
    idx.dsa_enable_prefill_cp = False
    return idx


class _Pool:
    page_size = 64
    index_page_size = 64

    def get_index_k_scale_buffer(self, layer_id, seq_len_t, table, a, b):
        n = int(seq_len_t[0])
        return (
            torch.zeros((n, 128), dtype=torch.uint8),
            torch.zeros((n, 2), dtype=torch.float16),
        )

    def get_index_k_scale_prefix_into(self, *a, **k):
        return None


def _single_chunk(num_rows, num_cols, device):
    return (slice(0, num_rows),)


class TestKPoolBcgPrefillRowContract(CustomTestCase):
    # ------------------------------------------------ producer-level guards

    def _run_producer(self, *, in_bcg, fb, metadata, q_rows):
        logits_stub = _LogitsStub()
        topk_stub = _TopkStub(32, 4)
        idx = _make_indexer()
        q_fp8 = torch.zeros(q_rows, 64)
        weights = torch.zeros(q_rows, 1)
        with (
            patch.object(kip, "is_in_breakable_cuda_graph", return_value=in_bcg),
            patch.object(kip, "get_token_to_kv_pool", return_value=_Pool()),
            patch.object(
                kip.envs.SGLANG_DSA_FUSE_TOPK,
                "get",
                return_value=metadata.topk_transform_method
                == TopkTransformMethod.RAGGED,
            ),
            patch.object(IndexerKPool, "_fp8_mqa_logits", logits_stub),
            patch.object(IndexerKPool, "_topk_from_kpool_logits", topk_stub),
            patch.object(kip, "_mqa_logits_row_chunks", _single_chunk),
        ):
            out = idx._get_topk_ragged_kpool(fb, 3, q_fp8, weights, metadata)
        return out, logits_stub.calls, topk_stub.calls

    def _extend_metadata(self, attn, method, expanded):
        return _Meta(attn, method, expanded)

    def test_bcg_multi_request_rebuilds_seqlens_and_offsets(self):
        """C1 + C3 (rebuild) + L7: mismatched expanded seqlens and stale
        short offsets must both be replaced from the live batch."""
        fb = _fb(2, [2, 4], [5, 7])
        attn = FrozenAttn(
            kpool_extend_pooled_page_tables=[
                torch.zeros(2, dtype=torch.int32),
                torch.zeros(2, dtype=torch.int32),
            ],
            topk_indices_offset=torch.tensor([0, 0], dtype=torch.int32),
        )
        # Verify-family expanded lengths (wrong row family on purpose).
        expanded = torch.tensor([10, 10], dtype=torch.int32)
        metadata = self._extend_metadata(attn, TopkTransformMethod.RAGGED, expanded)
        out, logits_calls, topk_calls = self._run_producer(
            in_bcg=True, fb=fb, metadata=metadata, q_rows=6
        )

        # Per-request MQA logits starts/ends follow the arange rebuild:
        # request 0: seq 5 over 2 rows -> [4, 5]; request 1: seq 7 over 4
        # rows -> [4, 5, 6, 7]; pool lens floor-div by kpool 4.
        self.assertEqual(logits_calls[0]["ends"].tolist(), [1, 1])
        self.assertEqual(logits_calls[1]["ends"].tolist(), [1, 1, 1, 1])

        # Fused offsets are rebuilt to concatenated-KV global starts:
        # request 0 rows -> 0; request 1 rows -> 5 (= seq_lens[0]).
        self.assertEqual(topk_calls[0]["topk_offsets"].tolist(), [0, 0])
        self.assertEqual(topk_calls[1]["topk_offsets"].tolist(), [5, 5, 5, 5])
        self.assertEqual(out.shape[0], 6)  # bridge invariant rows >= n

    def test_eager_offsets_pass_through_unchanged(self):
        """Regression: outside graph context the live offsets are sliced,
        never rebuilt."""
        fb = _fb(2, [2, 4], [5, 7])
        offsets = torch.tensor([0, 0, 3, 3, 3, 3], dtype=torch.int32)
        attn = FrozenAttn(
            kpool_extend_pooled_page_tables=[
                torch.zeros(2, dtype=torch.int32),
                torch.zeros(2, dtype=torch.int32),
            ],
            topk_indices_offset=offsets,
        )
        expanded = torch.tensor([4, 5, 4, 5, 6, 7], dtype=torch.int32)
        metadata = self._extend_metadata(attn, TopkTransformMethod.RAGGED, expanded)
        _, _, topk_calls = self._run_producer(
            in_bcg=False, fb=fb, metadata=metadata, q_rows=6
        )
        self.assertEqual(topk_calls[0]["topk_offsets"].tolist(), [0, 0])
        self.assertEqual(topk_calls[1]["topk_offsets"].tolist(), [3, 3, 3, 3])

    def test_zero_starts_wrong_shape_falls_back(self):
        """C2: per-request shaped zero_starts only; otherwise zeros."""
        fb = _fb(2, [2, 4], [5, 7])
        attn = FrozenAttn(
            kpool_extend_pooled_page_tables=[
                torch.zeros(2, dtype=torch.int32),
                torch.zeros(2, dtype=torch.int32),
            ],
            kpool_extend_zero_starts=torch.tensor([9, 9, 9], dtype=torch.int32),
        )
        metadata = self._extend_metadata(
            attn, TopkTransformMethod.RAGGED, torch.zeros(6, dtype=torch.int32)
        )
        _, logits_calls, _ = self._run_producer(
            in_bcg=True, fb=fb, metadata=metadata, q_rows=6
        )
        self.assertEqual(logits_calls[0]["starts"].tolist(), [0, 0])
        self.assertEqual(logits_calls[1]["starts"].tolist(), [0, 0, 0, 0])

    # ------------------------------------------------ impl-level plan guard

    def _run_impl(self, *, in_bcg, fb, attn, method, expanded, x_rows):
        logits_stub = _LogitsStub()
        topk_stub = _TopkStub(32, 4)
        skip_spy = []
        idx = _make_indexer()
        metadata = _Meta(attn, method, expanded)

        def _skip(*a, **k):
            skip_spy.append("skip")
            return torch.zeros(1, 1)

        with (
            patch.object(kip, "is_in_breakable_cuda_graph", return_value=in_bcg),
            patch.object(kip, "get_token_to_kv_pool", return_value=_Pool()),
            patch.object(
                kip,
                "get_attn_backend",
                lambda: SimpleNamespace(
                    get_indexer_metadata=lambda layer_id, fb_: metadata
                ),
            ),
            patch.object(kip, "get_is_capture_mode", return_value=False),
            patch.object(kip, "is_cuda", return_value=True),
            patch.object(kip, "is_hip", return_value=False),
            patch.object(kip, "is_npu", return_value=False),
            patch.object(kip, "_kpool_act_quant_int8", lambda q, b, s: (q, None)),
            patch.object(
                kip.envs.SGLANG_DSA_FUSE_TOPK,
                "get",
                return_value=method == TopkTransformMethod.RAGGED,
            ),
            patch.object(IndexerKPool, "_fp8_mqa_logits", logits_stub),
            patch.object(IndexerKPool, "_topk_from_kpool_logits", topk_stub),
            patch.object(kip, "_mqa_logits_row_chunks", _single_chunk),
            patch.object(
                IndexerKPool,
                "_get_q_k_bf16",
                lambda self, q_lora, x, positions, *a, **k: (
                    torch.zeros(x.shape[0], 1, 64),
                    torch.zeros(1, 64),
                    None,
                    None,
                ),
            ),
            patch.object(IndexerKPool, "_compress_write", lambda self, **k: None),
            patch.object(IndexerKPool, "_can_overlap_prefill", lambda self, *a: False),
            patch.object(
                IndexerKPool,
                "_resolve_head_gate_weights",
                lambda self, x, q_scale, hw: torch.zeros(x.shape[0], 1, 1),
            ),
            patch.object(IndexerKPool, "_forward_cuda_skip_logits", _skip),
            patch.object(
                IndexerKPool,
                "_forward_cuda_target_verify",
                lambda self, **k: skip_spy.append("verify") and torch.zeros(1, 1),
            ),
            patch.object(
                IndexerKPool,
                "_get_topk_paged",
                lambda self, *a, **k: skip_spy.append("paged") and torch.zeros(1, 1),
            ),
            patch(
                "sglang.srt.layers.cp.interleave.is_interleave_extend",
                return_value=self.interleave,
            ),
        ):
            if getattr(self, "cp_rows", None) is not None:
                ctx = patch(
                    "sglang.srt.layers.cp.bcg.interleave_bcg"
                    ".cp_interleave_indexer_rows",
                    return_value=self.cp_rows,
                )
                ctx.start()
                self.addCleanup(ctx.stop)
            out = idx._forward_cuda_impl(
                x=torch.zeros(x_rows, 16),
                q_lora=torch.zeros(x_rows, 8),
                positions=torch.arange(x_rows),
                forward_batch=fb,
                layer_id=3,
                return_indices=True,
            )
        return out, topk_stub.calls, skip_spy, attn

    def setUp(self):
        self.interleave = False
        self.cp_rows = None

    def test_padded_batch_fresh_plan_is_retained(self):
        """H2 (DP/bucket padding): plan rows == logical contract, x rows
        larger -> plan must survive and be consumed."""
        plan = _plan(4, marker=True)
        attn = FrozenAttn(kpool_extend_plan=plan)
        fb = _fb(2, [2, 2], [3000, 3000])
        self.interleave = False
        out, calls, skip, attn_after = self._run_impl(
            in_bcg=True,
            fb=fb,
            attn=attn,
            method=TopkTransformMethod.RAGGED,
            expanded=torch.zeros(4, dtype=torch.int32),
            x_rows=6,
        )
        self.assertIs(attn_after.kpool_extend_plan, plan)
        self.assertEqual(skip, [])
        # The plan producer fed the kernel its pooled lengths (777 marker).
        self.assertTrue(
            any(
                c["pool_lens"] is not None and (c["pool_lens"] == 777).any()
                for c in calls
            )
        )

    def test_interleave_padded_plan_is_retained(self):
        """H2 (CP interleave): plan rows follow the interleaved logical
        count, not physical rows; must survive."""
        plan = _plan(3, marker=True)
        attn = FrozenAttn(kpool_extend_plan=plan)
        fb = _fb(2, [1, 1], [3000, 3000])
        self.interleave = True
        self.cp_rows = (4, 3)
        _, calls, skip, attn_after = self._run_impl(
            in_bcg=True,
            fb=fb,
            attn=attn,
            method=TopkTransformMethod.RAGGED,
            expanded=torch.zeros(3, dtype=torch.int32),
            x_rows=4,
        )
        self.assertIs(attn_after.kpool_extend_plan, plan)
        self.assertTrue(
            any(
                c["pool_lens"] is not None and (c["pool_lens"] == 777).any()
                for c in calls
            )
        )

    def test_stale_bucket_plan_dropped_despite_matching_physical_rows(self):
        """H2 opposite direction: a capture-time plan whose rows match the
        padded x but not the live logical batch is stale and must drop to
        the live per-token rebuild path."""
        plan = _plan(6, marker=True)
        attn = FrozenAttn(
            kpool_extend_plan=plan,
            kpool_extend_pooled_page_tables=[
                torch.zeros(2, dtype=torch.int32),
                torch.zeros(2, dtype=torch.int32),
            ],
        )
        fb = _fb(2, [2, 2], [3000, 3000])  # logical rows 4, x rows 6
        self.interleave = False
        _, calls, skip, attn_after = self._run_impl(
            in_bcg=True,
            fb=fb,
            attn=attn,
            method=TopkTransformMethod.RAGGED,
            expanded=torch.zeros(4, dtype=torch.int32),
            x_rows=6,
        )
        self.assertIsNone(attn_after.kpool_extend_plan)
        self.assertFalse(
            any(
                c["pool_lens"] is not None and (c["pool_lens"] == 777).any()
                for c in calls
            )
        )

    # ------------------------------------------------ dispatch gates

    def test_bcg_extend_never_uses_skip_logits(self):
        fb = _fb(1, [6], [24])  # max_kv 24 <= topk 32: skip-eligible
        attn = FrozenAttn(
            kpool_extend_pooled_page_tables=[torch.zeros(2, dtype=torch.int32)]
        )
        self.interleave = False
        _, _, skip, _ = self._run_impl(
            in_bcg=True,
            fb=fb,
            attn=attn,
            method=TopkTransformMethod.PAGED,
            expanded=torch.zeros(6, dtype=torch.int32),
            x_rows=6,
        )
        self.assertNotIn("skip", skip)

    def test_eager_short_seq_keeps_skip_logits(self):
        fb = _fb(1, [6], [24])
        attn = FrozenAttn()
        _, _, skip, _ = self._run_impl(
            in_bcg=False,
            fb=fb,
            attn=attn,
            method=TopkTransformMethod.PAGED,
            expanded=torch.zeros(6, dtype=torch.int32),
            x_rows=6,
        )
        self.assertIn("skip", skip)

    def test_decode_verify_dispatch_zero_touch(self):
        for mode, marker in (
            ("target_verify", "verify"),
            ("draft_extend_v2", "verify"),
            ("decode", "paged"),
        ):
            with self.subTest(mode=mode):
                fb = _fb(1, [6], [100], mode_name=mode)
                attn = FrozenAttn()
                self.interleave = False
                _, _, skip, _ = self._run_impl(
                    in_bcg=True,
                    fb=fb,
                    attn=attn,
                    method=TopkTransformMethod.PAGED,
                    expanded=torch.zeros(6, dtype=torch.int32),
                    x_rows=6,
                )
                self.assertIn(marker, skip)
                self.assertNotIn("skip", skip)

    def test_cp_interleave_bridge_translation_untouched(self):
        import inspect

        from sglang.srt.layers.attention.dsa import kpool_prefill_cuda_graph

        src = inspect.getsource(
            kpool_prefill_cuda_graph._kpool_indexer_prefill_with_output
        )
        self.assertIn("cp_interleave_indexer_rows", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
