"""PPU NoPE precision, KPool tails and sampled long-row oracles with full finiteness checks."""

from __future__ import annotations

import itertools
import math
import unittest

import torch

from sglang.srt.utils import is_sail
from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.test_utils import CustomTestCase

register_ppu_ci(est_time=120, suite="stage-a-test-1-gpu-ppu")

FP8 = torch.float8_e4m3fn
DIM = 512
SCALE = 1.0 / math.sqrt(256)
POOL_ROWS = 4099


def _make_backend(kpool, capacity, max_rows=0):
    from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend

    # Only the real method's buffer state is needed; no model/server or mocks.
    backend = DeepseekSparseAttnBackend.__new__(DeepseekSparseAttnBackend)
    backend.dsa_index_kpool = kpool
    backend.flashmla_padding = 64
    backend._ppu_q8_topk_capacity = capacity
    backend._ppu_q8_max_num_tokens = max_rows
    backend._ppu_q8_buffers = {}
    backend._ppu_q8_buffers_eager = {}
    return backend


def _make_case(rows, heads, width, seed, *, strided=False):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    q = (torch.randn(rows, heads, DIM, generator=gen, device="cuda") * 0.5).to(
        torch.bfloat16
    )
    kv = (torch.randn(POOL_ROWS, 1, DIM, generator=gen, device="cuda") * 0.5).to(
        torch.bfloat16
    )
    # A masked -1 must not become a contribution from reserved physical slot 0.
    kv[0].fill_(1.5)
    table = torch.randint(
        1, POOL_ROWS, (rows, width), generator=gen, dtype=torch.int32, device="cuda"
    )
    kind = torch.arange(rows, device="cuda") % 6
    table[kind != 0] = -1
    for column in (0, 1, 2):
        table[kind == 1, column] = 11 + column
    table[kind == 3, -1] = 19
    table[kind == 4, :2] = 23  # Duplicate IDs count twice, not once.
    for column in (31, 129, 2047):
        table[kind == 5, column] = 29 + column
    table[(kind == 1) | (kind == 4) | (kind == 5), -1] = 37
    if strided:
        storage = table.new_full((rows, width * 2), -1)
        storage[:, ::2] = table
        table = storage[:, ::2]
        q = q.transpose(0, 1).contiguous().transpose(0, 1)
    return q, kv, table


def _oracle(
    q,
    kv,
    table,
    lengths=None,
    sample_rows=None,
    *,
    probability_fp8=False,
    absolute_values=False,
):
    """Independent softmax/PV reference, including the native E4M3 probability ABI."""
    selected = list(range(q.shape[0])) if sample_rows is None else sample_rows
    q, kv = q.float(), kv[:, 0].float()
    results = []
    for start in range(0, len(selected), 8):
        rows = selected[start : start + 8]
        ids = table[rows].long()
        valid = ids >= 0
        if lengths is not None:
            valid &= (
                torch.arange(ids.shape[1], device=ids.device)[None, :]
                < lengths[rows, None]
            )
        keys = kv[ids.clamp_min(0)]
        logits = torch.bmm(q[rows], keys.transpose(1, 2)) * SCALE
        logits.masked_fill_(~valid[:, None, :], -torch.inf)
        # Empty selections have zero output, not softmax(-inf,...,-inf) NaNs.
        logits = torch.where(valid.any(-1)[:, None, None], logits, 0.0)
        if probability_fp8:
            logits_log2 = logits * math.log2(math.e)
            maximum = logits_log2.amax(-1, keepdim=True).ceil()
            probabilities = (torch.exp2(logits_log2 - maximum) * 448.0).masked_fill(
                ~valid[:, None, :], 0.0
            )
            probs = probabilities.to(FP8).float() / probabilities.sum(
                -1, keepdim=True
            ).clamp_min(1e-20)
        else:
            probs = torch.softmax(logits, dim=-1).masked_fill(~valid[:, None, :], 0.0)
        results.append(torch.bmm(probs, keys.abs() if absolute_values else keys))
    return torch.cat(results)


def _forward(backend, q, kv, table, lengths=None, **kwargs):
    return backend._forward_ppu_native_q8(
        q_nope=q,
        q_rope=q.new_empty((*q.shape[:2], 0)),
        kv_cache=kv,
        page_table_1=table,
        sm_scale=SCALE,
        v_head_dim=DIM,
        topk_length=lengths,
        **kwargs,
    )


class TestDSANativeQ8PPU(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not is_sail() or torch.cuda.get_device_capability() != (8, 9):
            raise unittest.SkipTest("Native Q8 numerical tests require PPU0015 (SM89)")
        from sglang.kernels.ops.attention.ppu_flash_mla import (
            require_sparse_fp8_attention,
        )

        # Missing native FP8 support on the designated CI runner is a failure.
        require_sparse_fp8_attention()

    def _check(self, out, q, kv_bf16, table, *, lengths=None, sample_rows=None, label):
        self.assertEqual(tuple(out.shape), tuple(q.shape))
        self.assertTrue(torch.isfinite(out).all().item(), label)
        ref_q8 = _oracle(q.to(FP8), kv_bf16.to(FP8), table, lengths, sample_rows)
        ref_bf16 = _oracle(q, kv_bf16, table, lengths, sample_rows)
        ref_p8 = _oracle(
            q.to(FP8),
            kv_bf16.to(FP8),
            table,
            lengths,
            sample_rows,
            probability_fp8=True,
        )
        weighted_abs = _oracle(
            q.to(FP8),
            kv_bf16.to(FP8),
            table,
            lengths,
            sample_rows,
            absolute_values=True,
        )
        actual = out if sample_rows is None else out[sample_rows]
        error = (actual.float() - ref_q8).abs()
        numerical_error = (actual.float() - ref_p8).abs()
        loss = (ref_q8 - ref_bf16).abs()
        print(
            f"{label}: fp8_probability_vs_fp32_softmax max={error.max().item():.6g} "
            f"mae={error.mean().item():.6g}; "
            f"native_vs_fp8_probability_oracle max={numerical_error.max().item():.6g} "
            f"mae={numerical_error.mean().item():.6g}; "
            f"qkv_quantization_vs_original_bf16 max={loss.max().item():.6g} "
            f"mae={loss.mean().item():.6g}"
        )
        # E4M3 P rounding has relative unit 1/16; BF16 output rounding has unit 1/256.
        budget = weighted_abs / 16 + actual.float().abs() / 256 + 2e-5
        self.assertTrue((error <= budget).all().item(), label)
        self.assertLess(numerical_error.mean().item(), 1e-3, label)

    def test_heads_rows_topk_matrix(self):
        """Head padding, odd row counts, and 2051 -> 2176 table alignment."""
        for heads, rows, width in itertools.product(
            (1, 8, 16, 32, 64), (1, 17, 437), (2048, 2051, 2176)
        ):
            with self.subTest(heads=heads, rows=rows, width=width):
                kpool = width - 2048 + 1
                capacity = ((width + 127) // 128) * 128
                backend = _make_backend(kpool, capacity)
                q, kv, table = _make_case(
                    rows,
                    heads,
                    width,
                    seed=335 + heads + rows + width,
                    strided=heads in (8, 32),
                )
                # KPool must ignore a sequence-length vector that trims valid tails.
                lengths = torch.ones(rows, dtype=torch.int32, device="cuda")
                out = _forward(
                    backend,
                    q,
                    kv.to(FP8),
                    table,
                    lengths if kpool > 1 else None,
                )
                self._check(out, q, kv, table, label=f"H={heads} R={rows} K={width}")

    def test_single_selected_key_zero_query_is_exact(self):
        q, kv, table = _make_case(17, 8, 2051, seed=340)
        q.zero_()
        table.fill_(-1)
        table[:, -1] = 19
        kv_fp8 = kv.to(FP8)
        backend = _make_backend(4, 2176)
        out = _forward(backend, q, kv_fp8, table)
        expected = kv_fp8[19, 0].to(torch.bfloat16).expand_as(out)
        torch.testing.assert_close(out, expected, rtol=0, atol=0)

    def test_short_length_vectors(self):
        """Missing metadata rows must be inferred; supplied rows must still clip."""
        q, kv, table = _make_case(17, 16, 2048, seed=336, strided=True)
        for values in ([], [1], [1, 64, 129], [2048] * 20):
            with self.subTest(lengths=values):
                # Also exercise noncontiguous metadata and excess metadata rows.
                storage = torch.zeros(2 * len(values), dtype=torch.int32, device="cuda")
                lengths = storage[::2]
                lengths.copy_(torch.tensor(values, dtype=torch.int32, device="cuda"))
                effective = torch.full((17,), 2048, dtype=torch.int32, device="cuda")
                count = min(len(values), 17)
                effective[:count] = lengths[:count]
                backend = _make_backend(1, 2048)
                out = _forward(backend, q, kv.to(FP8), table, lengths)
                self._check(
                    out,
                    q,
                    kv,
                    table,
                    lengths=effective,
                    label=f"short_length_count={len(values)}",
                )

    def test_graph_replay_and_changed_input_buffer_reuse(self):
        """Captured addresses survive replay; new Q/KV/holes/tails replace stale data."""
        rows, heads, width = 17, 8, 2051
        backend = _make_backend(4, 2176, max_rows=rows)
        q, kv_bf16, table = _make_case(rows, heads, width, seed=337, strided=True)
        kv = kv_bf16.to(FP8)

        eager_backend = _make_backend(4, 2176, max_rows=0)
        _forward(eager_backend, q, kv, table, reuse_buffers=True)
        eager_addresses = {
            key: buf.data_ptr()
            for key, buf in eager_backend._ppu_q8_buffers_eager.items()
        }
        eager_q, eager_kv, eager_table = _make_case(rows, heads, width, seed=341)
        eager_out = _forward(
            eager_backend,
            eager_q,
            eager_kv.to(FP8),
            eager_table,
            reuse_buffers=True,
        )
        self._check(
            eager_out,
            eager_q,
            eager_kv,
            eager_table,
            label="eager_decode_buffer_reuse",
        )
        self.assertEqual(eager_backend._ppu_q8_buffers, {})
        self.assertEqual(
            {
                key: buf.data_ptr()
                for key, buf in eager_backend._ppu_q8_buffers_eager.items()
            },
            eager_addresses,
        )

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream):
            for _ in range(3):
                _forward(backend, q, kv, table, reuse_buffers=True)
            stream.synchronize()
            addresses = {
                key: buf.data_ptr() for key, buf in backend._ppu_q8_buffers.items()
            }
            with torch.cuda.graph(graph, stream=stream):
                captured_out = _forward(backend, q, kv, table, reuse_buffers=True)
        torch.cuda.current_stream().wait_stream(stream)
        first = None
        for seed in (338, 339, 340):
            new_q, new_kv, new_table = _make_case(rows, heads, width, seed)
            # Alternate an empty row and a tail-only row at the same address.
            new_table[0].fill_(-1)
            if seed % 2:
                new_table[0, -1] = 41
            q.copy_(new_q)
            kv_bf16.copy_(new_kv)
            kv.copy_(new_kv.to(FP8))
            table.copy_(new_table)
            graph.replay()
            torch.cuda.synchronize()
            self._check(
                captured_out, q, kv_bf16, table, label=f"graph_replay_seed={seed}"
            )
            if first is None:
                first = captured_out.clone()
            else:
                self.assertFalse(torch.equal(first, captured_out))

        # Same stream and padded width, but narrower real widths: stale columns
        # must be overwritten with -1, including after a wider, populated call.
        with torch.cuda.stream(stream):
            for real_width, real_rows in (
                (2176, 17),
                (2051, 1),
                (2051, 17),
                (2048, 17),
            ):
                q2, kv2, ids2 = _make_case(
                    real_rows, heads, real_width, seed=341 + real_width, strided=True
                )
                out = _forward(backend, q2, kv2.to(FP8), ids2, reuse_buffers=True)
                self._check(
                    out,
                    q2,
                    kv2,
                    ids2,
                    label=f"reuse_R={real_rows}_K={real_width}",
                )
            for key, address in addresses.items():
                self.assertEqual(backend._ppu_q8_buffers[key].data_ptr(), address)
            query = next(
                buf
                for key, buf in backend._ppu_q8_buffers.items()
                if key[0] == ("query", heads)
            )
            self.assertTrue(
                torch.equal(
                    query[:, :heads].contiguous().view(torch.uint8),
                    q2.to(FP8).contiguous().view(torch.uint8),
                )
            )
            self.assertTrue((query[:, heads:].view(torch.uint8) == 0).all().item())
        torch.cuda.current_stream().wait_stream(stream)

    def test_first_chunk_scan_and_query_cast_boundaries(self):
        """4096 scan threshold and 65536-row cast chunks must not lose any rows."""
        for rows in (4095, 4096, 65535, 65536, 65537):
            with self.subTest(rows=rows):
                gen = torch.Generator(device="cuda").manual_seed(335 + rows)
                q = (torch.randn(rows, 1, DIM, device="cuda", generator=gen) * 0.5).to(
                    torch.bfloat16
                )
                kv = (torch.randn(43, 1, DIM, device="cuda", generator=gen) * 0.5).to(
                    torch.bfloat16
                )
                # Broadcast a short selection to bound runtime and input memory.
                # Holes before IDs still distinguish count from last-valid-slot.
                one_row = torch.full((1, 2051), -1, dtype=torch.int32, device="cuda")
                one_row[0, [0, 3, 7]] = torch.tensor(
                    [11, 19, 37], dtype=torch.int32, device="cuda"
                )
                table = one_row.expand(rows, -1)
                backend = _make_backend(4, 2176)
                out = _forward(
                    backend,
                    q,
                    kv.to(FP8),
                    table,
                    first_prefill_chunk=True,
                )
                samples = sorted(
                    {
                        0,
                        1,
                        rows // 2,
                        rows - 2,
                        rows - 1,
                        *[
                            i
                            for i in (4094, 4095, 4096, 65534, 65535, 65536)
                            if i < rows
                        ],
                    }
                )
                self._check(
                    out, q, kv, table, sample_rows=samples, label=f"cast_rows={rows}"
                )
                del out, backend, q, kv, table

    def test_raw_fp8_nope_scatter_preserves_reserved_and_unwritten_slots(self):
        """512+0 is raw FP8, not packed 528/656-byte KV; empty RoPE is never read."""
        from sglang.kernels.ops.kvcache.mla_buffer import (
            set_mla_kv_buffer_triton_fp8_quant,
        )

        for reserved in (0, 7, -1):
            with self.subTest(reserved=reserved):
                gen = torch.Generator(device="cuda").manual_seed(342)
                source = (
                    torch.randn(9, 1, DIM + 32, generator=gen, device="cuda") * 8
                ).to(torch.bfloat16)
                nope = source[..., :DIM]  # Nontrivial row stride, contiguous channels.
                rope = source.new_empty((9, 1, 0))
                loc = torch.tensor([0, 7, 3, 19, 5, 11, 23, 2, 13], device="cuda")
                for dtype in (FP8, torch.uint8):
                    pool = torch.empty(29, 1, DIM, dtype=dtype, device="cuda")
                    pool.view(torch.uint8).fill_(0x35)
                    expected = pool.view(torch.uint8).clone()
                    converted = nope.to(FP8).contiguous().view(torch.uint8)
                    valid = loc != reserved
                    expected[loc[valid]] = converted[valid]
                    set_mla_kv_buffer_triton_fp8_quant(
                        pool, loc, nope, rope, FP8, reserved_skip_index=reserved
                    )
                    self.assertTrue(torch.equal(pool.view(torch.uint8), expected))
                    self.assertEqual(pool[0].numel() * pool.element_size(), DIM)


if __name__ == "__main__":
    unittest.main()
