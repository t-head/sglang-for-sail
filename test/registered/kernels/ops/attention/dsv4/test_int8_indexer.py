"""Unit tests for the DSv4.1 PPU INT8 indexer cache and E4M3FN emulation.

Covers the three hardware-independent pieces of the INT8/dense-FP4 indexer
support: the packed index-K cache (``store_int8_index_k``/
``gather_int8_index_k`` addressing, FP32 scale byte packing, the
``slot > 0`` reservation), ``quantize_query``'s per-head Q-scale folding into
the head weights, and the software E4M3FN helpers that replace the
``float8e4nv`` casts Triton rejects on PPU below SM89 (bit-exact against
torch's E4M3FN cast/decode). The DeepGEMM ``int8_*_mqa_logits`` scoring
itself is covered by E2E accuracy runs.

Runs on any CUDA device; no DSv4 checkpoint required.
"""

import unittest

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsv4.fp4_rope_fake_quant import (
    _round_positive_e4m3_scale,
)
from sglang.kernels.ops.attention.dsv4.int8_indexer import (
    _rope_tail_pairs,
    gather_int8_index_k,
    index_q_rope_int8_quant_weights,
    int8_logits_rows_per_chunk,
    quantize_query,
    store_index_k_norm_rope_int8,
    store_int8_index_k,
)
from sglang.kernels.ops.embeddings.engram_gather import _e4m3fn_to_float
from sglang.kernels.ops.quantization.int8_kernel import per_token_quant_int8
from sglang.srt.layers.attention.dsv4.dsv41_sparse import RMSNorm, rope_tail
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

PAGE_SIZE = 64


@triton.jit
def _write_rope_pairs(out, RD: tl.constexpr):
    offsets = tl.arange(0, 128)
    _, pair = _rope_tail_pairs(offsets, RD)
    tl.store(out + offsets, pair)


@triton.jit
def _drive_round_scale(x_ptr, y_ptr, N, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(y_ptr + offs, _round_positive_e4m3_scale(x), mask=mask)


@triton.jit
def _drive_e4m3_decode(b_ptr, y_ptr, N, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    payload = tl.load(b_ptr + offs, mask=mask)
    tl.store(y_ptr + offs, _e4m3fn_to_float(payload), mask=mask)


class TestInt8IndexerCache(CustomTestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        self.device = "cuda"

    def _zeros_cache(self, pages):
        return torch.zeros(
            pages, PAGE_SIZE * 132, dtype=torch.uint8, device=self.device
        )

    def test_rope_pair_offsets_for_masked_lanes(self):
        """The non-RoPE lanes must keep the first frequency row's offsets in bounds."""
        got = torch.empty(128, device=self.device, dtype=torch.int32)
        _write_rope_pairs[(1,)](got, RD=64)

        expected = torch.zeros_like(got)
        expected[64:] = torch.arange(32, device=self.device).repeat_interleave(2)
        torch.testing.assert_close(got, expected, rtol=0, atol=0)

    def test_store_gather_roundtrip(self):
        torch.manual_seed(0)
        rows = 5
        loc = torch.tensor([0, 1, 63, 64, 255], dtype=torch.int64, device=self.device)
        key = torch.randn(rows, 128, dtype=torch.bfloat16, device=self.device)
        cache = self._zeros_cache(4)

        store_int8_index_k(key, cache, loc, PAGE_SIZE)
        k, scales = gather_int8_index_k(cache, loc, PAGE_SIZE)
        torch.cuda.synchronize()

        ref_q, ref_s = per_token_quant_int8(key.contiguous())
        self.assertTrue(torch.equal(k[1:], ref_q.view(rows, 128)[1:]))
        self.assertTrue(torch.equal(scales[1:], ref_s.view(rows).to(torch.float32)[1:]))
        # Slot 0 is reserved: nothing was written and its row stays zero.
        self.assertTrue(bool((k[0] == 0).all()) and scales[0].item() == 0.0)

    def test_scale_bytes_match_fp32_bitcast(self):
        torch.manual_seed(1)
        loc = torch.tensor([130], dtype=torch.int64, device=self.device)
        key = torch.randn(1, 128, dtype=torch.bfloat16, device=self.device)
        cache = self._zeros_cache(3)

        store_int8_index_k(key, cache, loc, PAGE_SIZE)
        torch.cuda.synchronize()

        ref_q, ref_s = per_token_quant_int8(key.contiguous())
        page, off = int(loc[0]) // PAGE_SIZE, int(loc[0]) % PAGE_SIZE
        scale_region = cache[:, PAGE_SIZE * 128 :].view(torch.float32)
        self.assertEqual(scale_region[page, off].item(), ref_s.view(-1)[0].item())
        payload = cache[:, : PAGE_SIZE * 128].view(torch.int8)
        payload = payload.reshape(cache.shape[0], PAGE_SIZE, 128)
        self.assertTrue(torch.equal(payload[page, off], ref_q.view(128)))

    def test_rows_per_chunk_matches_budget_arithmetic(self):
        for width in (1, 128, 257, 1024):
            for budget_mb in (1, 16, 4096):
                budget = budget_mb * 1024 * 1024
                row_bytes = (width * 4 + 1023) // 1024 * 1024
                expect = max(1, budget // row_bytes // 4 * 4)
                self.assertEqual(int8_logits_rows_per_chunk(width, budget), expect)

    def test_quantize_query_folds_q_scales_into_weights(self):
        torch.manual_seed(2)
        q = torch.randn(3, 4, 128, dtype=torch.bfloat16, device=self.device)
        weights = torch.rand(3, 4, dtype=torch.float32, device=self.device)

        q_int8, folded = quantize_query(q, weights)
        ref_q, ref_s = per_token_quant_int8(q.flatten(0, 1).contiguous())
        self.assertEqual(tuple(q_int8.shape), (3, 4, 128))
        self.assertTrue(torch.equal(q_int8, ref_q.view(3, 4, 128)))
        self.assertTrue(torch.equal(folded, weights * ref_s.view(3, 4)))
        # Dequantized error stays within one rounding step of the per-token scale.
        dequant = q_int8.float() * ref_s.view(3, 4, 1)
        err = (dequant - q.float()).abs()
        self.assertTrue((err <= ref_s.view(3, 4, 1)).all())

    def test_store_index_k_norm_rope_int8_matches_unfused(self):
        torch.manual_seed(3)
        eps = 1e-6
        for page_size in (64, 128):
            for rope_dim in (0, 64, 128):
                for weight_dtype in (torch.bfloat16, torch.float32):
                    with self.subTest(
                        page_size=page_size,
                        rope_dim=rope_dim,
                        weight_dtype=weight_dtype,
                    ):
                        rows = 9
                        x = torch.randn(
                            rows, 128, dtype=torch.bfloat16, device=self.device
                        )
                        norm = RMSNorm(128, eps).to(
                            device=self.device, dtype=weight_dtype
                        )
                        with torch.no_grad():
                            norm.weight.copy_(
                                torch.randn(128, device=self.device, dtype=weight_dtype)
                            )
                        angles = torch.randn(
                            rows, max(rope_dim // 2, 1), device=self.device
                        )
                        freqs = torch.polar(torch.ones_like(angles), angles)
                        if rope_dim == 0:
                            freqs = freqs[:, :0]

                        k_normed = norm(x)
                        k_rot = (
                            rope_tail(k_normed, freqs, rope_dim)
                            if rope_dim > 0
                            else k_normed
                        )
                        ref_q, ref_s = per_token_quant_int8(k_rot.contiguous())
                        ref_s = ref_s.view(rows)

                        got_q, got_s = store_index_k_norm_rope_int8(
                            x, norm.weight, eps, freqs, rope_dim
                        )
                        self.assertTrue(torch.equal(got_q, ref_q))
                        self.assertTrue(torch.equal(got_s, ref_s))

                        cache = torch.zeros(
                            4, page_size * 132, dtype=torch.uint8, device=self.device
                        )
                        loc = torch.tensor(
                            [
                                -1,
                                0,
                                1,
                                2,
                                page_size - 1,
                                page_size,
                                page_size + 5,
                                2 * page_size + 3,
                                3 * page_size + 1,
                            ],
                            dtype=torch.int64,
                            device=self.device,
                        )
                        store_index_k_norm_rope_int8(
                            x, norm.weight, eps, freqs, rope_dim, cache=cache, loc=loc
                        )
                        valid_loc = loc[2:]
                        gathered_k, gathered_s = gather_int8_index_k(
                            cache, valid_loc, page_size
                        )
                        self.assertTrue(torch.equal(gathered_k, ref_q[2:]))
                        self.assertTrue(torch.equal(gathered_s, ref_s[2:]))
                        # Reserved slot 0 must stay zeroed despite loc <= 0 rows.
                        slot0_k, slot0_s = gather_int8_index_k(
                            cache,
                            torch.zeros(1, dtype=torch.int64, device=self.device),
                            page_size,
                        )
                        self.assertTrue(
                            bool((slot0_k == 0).all()) and slot0_s[0].item() == 0.0
                        )

    def test_index_q_rope_int8_quant_weights_matches_unfused(self):
        torch.manual_seed(4)
        max_pos, heads = 64, 32
        weight_scale = (128**-0.5) * (heads**-0.5)
        for rope_dim in (0, 64, 128):
            for weights_dtype in (torch.float32, torch.bfloat16):
                for pos_dtype in (torch.int32, torch.int64):
                    with self.subTest(
                        rope_dim=rope_dim,
                        weights_dtype=weights_dtype,
                        pos_dtype=pos_dtype,
                    ):
                        rows = 7
                        q_raw = torch.randn(
                            rows, heads, 128, dtype=torch.bfloat16, device=self.device
                        )
                        w_raw = torch.randn(
                            rows, heads, dtype=torch.bfloat16, device=self.device
                        )
                        positions = torch.randint(
                            0, max_pos, (rows,), dtype=pos_dtype, device=self.device
                        )
                        angles = torch.randn(
                            max_pos, max(rope_dim // 2, 1), device=self.device
                        )
                        freqs_cis = torch.polar(torch.ones_like(angles), angles)
                        if rope_dim == 0:
                            freqs_cis = freqs_cis[:, :0]

                        q_rot = (
                            rope_tail(q_raw, freqs_cis[positions], rope_dim)
                            if rope_dim > 0
                            else q_raw
                        )
                        ref_q, ref_w = quantize_query(q_rot, w_raw * weight_scale)
                        ref_w = ref_w.to(weights_dtype)

                        got_q, got_w = index_q_rope_int8_quant_weights(
                            q_raw,
                            freqs_cis,
                            positions,
                            rope_dim,
                            w_raw,
                            weight_scale,
                            weights_dtype=weights_dtype,
                        )
                        self.assertEqual(got_q.shape, (rows, heads, 128))
                        self.assertEqual(got_w.dtype, weights_dtype)
                        self.assertTrue(torch.equal(got_q, ref_q))
                        self.assertTrue(torch.equal(got_w, ref_w))

        # Empty batch edge case.
        empty_q, empty_w = index_q_rope_int8_quant_weights(
            torch.empty(0, heads, 128, dtype=torch.bfloat16, device=self.device),
            torch.empty(max_pos, 32, dtype=torch.complex64, device=self.device),
            torch.empty(0, dtype=torch.int64, device=self.device),
            64,
            torch.empty(0, heads, dtype=torch.bfloat16, device=self.device),
            weight_scale,
        )
        self.assertEqual(empty_q.shape, (0, heads, 128))
        self.assertEqual(empty_w.shape, (0, heads))

    def test_gather_int8_empty_slots(self):
        cache = torch.zeros(2, 128 * 132, dtype=torch.uint8, device=self.device)
        k, s = gather_int8_index_k(
            cache, torch.empty(0, dtype=torch.int64, device=self.device), 128
        )
        self.assertEqual(k.shape, (0, 128))
        self.assertEqual(k.dtype, torch.int8)
        self.assertEqual(s.shape, (0,))
        self.assertEqual(s.dtype, torch.float32)

    def test_c2_target_verify_position0_and_position1_slots(self):
        """Target-verify batch with position=0 and position=1 for C2 must keep
        FlashMLA lengths clamped to 1 while top-k returns 0 and 1 valid slots."""
        from sglang.kernels.ops.attention.dsv4.metadata_kernel import (
            build_low_ratio_metadata,
        )
        from sglang.kernels.ops.attention.dsv4.topk import topk_transform_paged

        seq_lens = torch.tensor([1, 2], dtype=torch.int32, device=self.device)
        out_loc = torch.tensor([10, 11], dtype=torch.int64, device=self.device)
        topk = 64
        low = build_low_ratio_metadata(seq_lens, out_loc, topk)

        self.assertEqual(low.c2_out_loc.tolist(), [-1, 5])
        self.assertEqual(low.c2_seq_lens_raw.tolist(), [0, 1])
        self.assertEqual(low.c2_seq_lens.tolist(), [1, 1])
        self.assertEqual(low.c2_sparse_lens.tolist(), [1, 1])

        page_size = 128
        page_table = torch.tensor([[3], [3]], dtype=torch.int32, device=self.device)
        scores = torch.ones((2, page_size), dtype=torch.float32, device=self.device)
        out_page_indices = torch.full(
            (2, topk), -99, dtype=torch.int32, device=self.device
        )
        topk_transform_paged(
            scores, low.c2_seq_lens_raw, page_table, out_page_indices, page_size
        )
        torch.cuda.synchronize()

        self.assertEqual(int((out_page_indices[0] >= 0).sum().item()), 0)
        self.assertTrue(bool((out_page_indices[0] == -1).all()))
        self.assertEqual(int((out_page_indices[1] >= 0).sum().item()), 1)
        self.assertEqual(out_page_indices[1, 0].item(), 3 * page_size)
        self.assertTrue(bool((out_page_indices[1, 1:] == -1).all()))


class TestE4M3Emulation(CustomTestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        self.device = "cuda"

    def _drive(self, kernel, inp, out_dtype):
        n = inp.numel()
        out = torch.empty(n, dtype=out_dtype, device=self.device)
        kernel[(triton.cdiv(n, 256),)](inp, out, n, BLOCK=256)
        torch.cuda.synchronize()
        return out

    def test_round_scale_matches_torch_fp8_cast(self):
        values = torch.cat(
            [
                torch.linspace(2.0**-9, 2.0**-6 * 1.98, 512),
                torch.linspace(2.0**-6, 448.0, 1024),
                torch.tensor([2.0**-9, 2.0**-7, 1.0, 2.0, 448.0]),
            ]
        ).to(self.device)
        got = self._drive(_drive_round_scale, values, torch.float32)
        expect = values.to(torch.float8_e4m3fn).to(torch.float32)
        self.assertTrue(torch.equal(got, expect))

    def test_decode_matches_torch_fp8_view(self):
        payload = torch.arange(256, dtype=torch.uint8, device=self.device)
        got = self._drive(_drive_e4m3_decode, payload, torch.float32)
        expect = payload.view(torch.float8_e4m3fn).to(torch.float32)
        nan = torch.isnan(expect)
        self.assertTrue(bool(nan[0x7F]) and bool(nan[0xFF]))
        self.assertTrue(bool(got[nan].isnan().all()))
        self.assertTrue(torch.equal(got[~nan], expect[~nan]))


if __name__ == "__main__":
    unittest.main()
