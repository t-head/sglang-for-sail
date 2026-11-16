"""Device regression for the DSA K-only writer, without model checkpoints.

The fixture supplies projections and scheduler metadata; K processing, cache
writes, TopK transforms and CUDA graph capture use the production implementations.
Run with PYTHONPATH=python CUDA_VISIBLE_DEVICES=0 python -m unittest discover \
    -s test/registered/unit/layers/attention -p test_dsa_k_only_ppu.py -v
"""

import math
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.srt.utils import is_sail
from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.test_utils import CustomTestCase

register_ppu_ci(est_time=180, stage="stage-a", runner_config="1-gpu-ppu")


class _Projection(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.register_buffer("weight", weight)

    def forward(self, x):
        return F.linear(x, self.weight), None


class _Pool:
    index_page_size = page_size = 64
    index_head_dim = quant_block_size = 128
    index_kpool = 1
    device = "cuda"

    def __init__(self, dtype):
        self.dtype = dtype
        self.packed_bytes_per_token = 64 if dtype == "mxfp4" else 128
        self.scale_bytes_per_token = 4
        self.buffer = torch.zeros(
            256,
            64 * (128 if dtype == "bf16" else self.packed_bytes_per_token + 4),
            device="cuda",
            dtype=torch.bfloat16 if dtype == "bf16" else torch.uint8,
        )

    def get_index_k_with_scale_buffer(self, layer_id):
        return self.buffer

    def set_index_k_scale_buffer(self, layer_id, loc, index_k, index_k_scale):
        from sglang.kernels.ops.kvcache.dsa_index_k_cache import SetKAndS

        SetKAndS.execute(
            pool=self,
            buf=self.buffer,
            loc=loc,
            index_k=index_k,
            index_k_scale=index_k_scale,
        )


@unittest.skipUnless(torch.cuda.is_available() and is_sail(), "PPU device required")
class TestDsaKOnlyPPU(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        import sglang.srt.layers.attention.dsa.dsa_indexer as dsa
        from sglang.kernels.ops.quantization.act_quant_triton import act_quant
        from sglang.srt.layers.attention.dsa.dsa_indexer_metadata import (
            DSAIndexerMetadata,
        )
        from sglang.srt.layers.attention.dsa.dsa_topk_backend import (
            DSATopKBackend,
            TopkTransformMethod,
        )
        from sglang.srt.layers.rotary_embedding import RotaryEmbedding
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        cls.dsa = dsa
        cls.act_quant = staticmethod(act_quant)
        cls.metadata_cls = DSAIndexerMetadata
        cls.topk_backend = DSATopKBackend.SGL_KERNEL
        cls.transform = TopkTransformMethod
        cls.mode = ForwardMode
        cls.rope_cls = RotaryEmbedding
        major, minor = torch.cuda.get_device_capability()
        cls.supports_topk_v2 = major * 10 + minor >= 89
        cls.cache_dtypes = (
            ("fp8", "int8", "mxfp4") if cls.supports_topk_v2 else ("int8",)
        )

    def setUp(self):
        torch.manual_seed(923)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(torch.inference_mode())
        self.stack.enter_context(
            patch.object(self.dsa, "is_cp_active", return_value=False)
        )
        self.stack.enter_context(
            patch.object(self.dsa, "dsa_shards_index_k", return_value=False)
        )
        self.stack.enter_context(
            patch.object(self.dsa, "_index_k_dcp", return_value=(1, 0))
        )
        from sglang.srt.environ import envs

        self.stack.enter_context(envs.SGLANG_OPT_USE_TOPK_V2.override(False))
        self.stack.enter_context(envs.SGLANG_DSA_FUSE_TOPK.override(True))

    def _indexer(self, dtype, fusion, *, skip_rope=False):
        idx = self.dsa.Indexer.__new__(self.dsa.Indexer)
        torch.nn.Module.__init__(idx)
        idx.head_dim, idx.rope_head_dim = 128, 64
        idx.n_heads, idx.hidden_size, idx.q_lora_rank = 32, 256, 128
        idx.index_topk, idx.block_size, idx.scale_fmt = 2048, 128, None
        idx.softmax_scale = 128**-0.5
        idx.use_fp4 = dtype == "mxfp4"
        idx.use_int8 = dtype == "int8"
        idx.use_fp8 = dtype == "fp8"
        idx.bf16_indexer = dtype == "bf16"
        idx.use_dsa_indexer_fusion = fusion
        idx.use_aiter_fused_fp8_writer = False
        idx.use_gfx950_fused_indexer = False
        idx.dsa_enable_prefill_cp = False
        idx.skip_rope, idx.apply_hadamard = skip_rope, not skip_rope
        idx.alt_stream = None
        from sglang.srt.layers.layernorm import LayerNorm

        idx.k_norm = LayerNorm(128).cuda()
        idx.rotary_emb = self.rope_cls(
            64, 64, 8192, 10000, False, torch.bfloat16
        ).cuda()

        def weight(out, inp):
            return torch.randn(
                out, inp, device="cuda", dtype=torch.bfloat16
            ) / math.sqrt(inp)

        wk, gate = weight(128, 256), weight(32, 256)
        if fusion:
            idx.wk_weights_proj = _Projection(torch.cat((wk, gate)))
        else:
            idx.wk = _Projection(wk)
            idx.weights_proj = _Projection(gate)
        idx.wq_b = _Projection(weight(32 * 128, 128))
        return idx

    def _batch(self, prefill, *, padded=0):
        if prefill:
            lengths = torch.arange(1, 66, device="cuda", dtype=torch.int32).repeat(2)
            cu_q = torch.tensor([0, 65, 130], device="cuda", dtype=torch.int32)
            offsets = torch.tensor(
                [0] * 65 + [65] * 65, device="cuda", dtype=torch.int32
            )
            seq_cpu = torch.tensor([65, 65], dtype=torch.int32)
            method = self.transform.RAGGED
        else:
            lengths = torch.tensor(
                [1, 63, 64, 65, 127, 2048], device="cuda", dtype=torch.int32
            )
            cu_q = torch.arange(7, device="cuda", dtype=torch.int32)
            offsets, seq_cpu, method = None, lengths.cpu(), self.transform.PAGED
        rows = lengths.numel()
        page_table = (
            torch.arange(192, device="cuda", dtype=torch.int32).flip(0).reshape(6, 32)
        )
        page_table_1 = (
            page_table[:, :, None] * 64
            + torch.arange(64, device="cuda", dtype=torch.int32)
        ).reshape(6, -1)
        md = self.metadata_cls(
            SimpleNamespace(
                cache_seqlens_int32=lengths,
                dsa_seqlens_expanded=lengths,
                cu_seqlens_q=cu_q,
                topk_indices_offset=offsets,
                page_table_1=page_table_1,
                real_page_table=page_table,
                page_size=64,
                topk_v2_plan=None,
                token_to_batch_idx=None,
            ),
            method,
            self.topk_backend,
        )
        # Sparse physical writes cross page boundaries and do not touch page 0.
        loc = torch.arange(rows + padded, device="cuda", dtype=torch.int64) * 3 + 63
        batch = SimpleNamespace(
            forward_mode=self.mode.EXTEND if prefill else self.mode.DECODE,
            out_cache_loc=loc,
            seq_lens_cpu=seq_cpu,
            attn_cp_metadata=None,
        )
        positions = torch.arange(rows + padded, device="cuda", dtype=torch.int64) + 37
        x = torch.randn(rows + padded, 256, device="cuda", dtype=torch.bfloat16)
        q_lora = torch.randn(rows + padded, 128, device="cuda", dtype=torch.bfloat16)
        return batch, md, x, q_lora, positions

    def _full_writer(self, idx, pool, batch, x, q_lora, positions):
        with patch.object(self.dsa, "get_token_to_kv_pool", return_value=pool):
            if idx.use_dsa_indexer_fusion and not idx.bf16_indexer:
                idx._fused_q_prepare_and_store(
                    x,
                    q_lora,
                    positions,
                    batch,
                    0,
                    self.act_quant,
                    enable_dual_stream=False,
                )
            else:
                _, key, _ = idx._get_q_k_bf16(q_lora, x, positions, False, batch)
                idx._store_index_k_cache(batch, 0, key, act_quant=self.act_quant)

    def _expected_indices(self, metadata):
        lengths = metadata.get_seqlens_expanded()
        columns = torch.arange(2048, device="cuda", dtype=torch.int32)[None, :]
        if metadata.topk_transform_method == self.transform.RAGGED:
            indices = columns + metadata.attn_metadata.topk_indices_offset[:, None]
        else:
            indices = metadata.attn_metadata.page_table_1
        return torch.where(columns < lengths[:, None], indices, -1)

    def test_k_only_matches_full_cache_and_topk(self):
        for dtype in self.cache_dtypes:
            for fusion in (False, True):
                for prefill in (False, True):
                    with self.subTest(dtype=dtype, fusion=fusion, prefill=prefill):
                        idx = self._indexer(dtype, fusion)
                        batch, md, x, q_lora, pos = self._batch(prefill)
                        full, short = _Pool(dtype), _Pool(dtype)
                        self._full_writer(idx, full, batch, x, q_lora, pos)
                        with patch.object(
                            self.dsa, "get_token_to_kv_pool", return_value=short
                        ):
                            result = idx._forward_cuda_k_only(
                                x, pos, batch, 0, self.act_quant, md
                            )
                        torch.cuda.synchronize()
                        torch.testing.assert_close(
                            short.buffer, full.buffer, rtol=0, atol=0
                        )
                        torch.testing.assert_close(
                            result, self._expected_indices(md), rtol=0, atol=0
                        )
                        # Non-constant logits select the same visible indices.
                        logits = torch.randn(
                            result.shape, device="cuda", dtype=torch.float32
                        )
                        torch.testing.assert_close(
                            md.topk_transform(logits, 2048), result, rtol=0, atol=0
                        )

    def test_no_rope_no_hadamard_kimi_geometry(self):
        for dtype in self.cache_dtypes:
            with self.subTest(dtype=dtype):
                idx = self._indexer(dtype, False, skip_rope=True)
                batch, md, x, q_lora, pos = self._batch(False)
                full, short = _Pool(dtype), _Pool(dtype)
                self._full_writer(idx, full, batch, x, q_lora, pos)
                with patch.object(self.dsa, "get_token_to_kv_pool", return_value=short):
                    result = idx._forward_cuda_k_only(
                        x, pos, batch, 0, self.act_quant, md
                    )
                torch.cuda.synchronize()
                torch.testing.assert_close(short.buffer, full.buffer, rtol=0, atol=0)
                torch.testing.assert_close(
                    result, self._expected_indices(md), rtol=0, atol=0
                )

    def test_forward_cuda_short_prefill_and_decode(self):
        for dtype in self.cache_dtypes:
            for fusion in (False, True):
                for prefill in (False, True):
                    with self.subTest(dtype=dtype, fusion=fusion, prefill=prefill):
                        idx = self._indexer(dtype, fusion)
                        batch, md, x, q_lora, pos = self._batch(prefill)
                        pool = _Pool(dtype)
                        with ExitStack() as stack:
                            stack.enter_context(
                                patch.object(
                                    self.dsa, "get_is_capture_mode", return_value=False
                                )
                            )
                            stack.enter_context(
                                patch.object(
                                    self.dsa, "get_token_to_kv_pool", return_value=pool
                                )
                            )
                            stack.enter_context(
                                patch.object(
                                    self.dsa,
                                    "get_attn_backend",
                                    return_value=SimpleNamespace(
                                        get_indexer_metadata=lambda *args: md
                                    ),
                                )
                            )
                            stack.enter_context(
                                patch.object(
                                    self.dsa,
                                    "maybe_capture_indexer_topk",
                                    side_effect=lambda layer, result: result,
                                )
                            )
                            result = idx.forward_cuda(
                                x, q_lora, pos, batch, 0, use_cuda_graph=False
                            )
                        torch.cuda.synchronize()
                        torch.testing.assert_close(
                            result, self._expected_indices(md), rtol=0, atol=0
                        )

    def test_forward_cuda_dense_graph(self):
        from sglang.srt.model_executor.runner_utils import capture_mode

        for dtype in self.cache_dtypes:
            for fusion in (False, True):
                with self.subTest(dtype=dtype, fusion=fusion):
                    idx = self._indexer(dtype, fusion)
                    batch, md, x, q_lora, pos = self._batch(False)
                    pool = _Pool(dtype)
                    with ExitStack() as stack:
                        stack.enter_context(
                            patch.object(
                                self.dsa, "get_is_capture_mode", return_value=True
                            )
                        )
                        stack.enter_context(
                            patch.object(
                                capture_mode, "_capture_attention_variant", "dense"
                            )
                        )
                        stack.enter_context(
                            patch.object(
                                self.dsa, "get_token_to_kv_pool", return_value=pool
                            )
                        )
                        stack.enter_context(
                            patch.object(
                                self.dsa,
                                "get_attn_backend",
                                return_value=SimpleNamespace(
                                    get_indexer_metadata=lambda *args: md
                                ),
                            )
                        )
                        stack.enter_context(
                            patch.object(
                                self.dsa,
                                "maybe_capture_indexer_topk",
                                side_effect=lambda layer, result: result,
                            )
                        )
                        stream = torch.cuda.Stream()
                        stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(stream):
                            for _ in range(3):
                                idx.forward_cuda(
                                    x, q_lora, pos, batch, 0, use_cuda_graph=False
                                )
                        torch.cuda.current_stream().wait_stream(stream)
                        torch.cuda.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            output = idx.forward_cuda(
                                x, q_lora, pos, batch, 0, use_cuda_graph=False
                            )
                    for _ in range(3):
                        x.normal_()
                        pos.add_(1)
                        graph.replay()
                        torch.cuda.synchronize()
                        torch.testing.assert_close(
                            output, self._expected_indices(md), rtol=0, atol=0
                        )

    def test_topk_v2_compact_only_decode_and_graph_replay(self):
        """Dense K-only decode must work without the legacy wide page table."""
        if not self.supports_topk_v2:
            self.skipTest("TopK v2 device coverage requires an SM89+ PPU")
        from sglang.kernels.ops.attention.dsv4.topk import plan_topk_v2
        from sglang.srt.environ import envs

        for dtype in self.cache_dtypes:
            with self.subTest(dtype=dtype), envs.SGLANG_OPT_USE_TOPK_V2.override(True):
                idx = self._indexer(dtype, False)
                batch, metadata, x, _, positions = self._batch(False)
                expected = self._expected_indices(metadata)
                metadata.attn_metadata.page_table_1 = None
                metadata.attn_metadata.topk_v2_plan = plan_topk_v2(
                    metadata.get_seqlens_int32()
                )
                pool = _Pool(dtype)
                with patch.object(self.dsa, "get_token_to_kv_pool", return_value=pool):
                    eager = idx._forward_cuda_k_only(
                        x, positions, batch, 0, self.act_quant, metadata
                    )
                    torch.testing.assert_close(eager, expected, rtol=0, atol=0)
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            idx._forward_cuda_k_only(
                                x, positions, batch, 0, self.act_quant, metadata
                            )
                    torch.cuda.current_stream().wait_stream(stream)
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        output = idx._forward_cuda_k_only(
                            x, positions, batch, 0, self.act_quant, metadata
                        )
                    for _ in range(3):
                        x.normal_()
                        positions.add_(1)
                        graph.replay()
                        torch.cuda.synchronize()
                        torch.testing.assert_close(output, expected, rtol=0, atol=0)

    def test_k_only_graph_replay_and_padding(self):
        for dtype in self.cache_dtypes:
            for fusion in (False, True):
                with self.subTest(dtype=dtype, fusion=fusion):
                    idx = self._indexer(dtype, fusion)
                    batch, md, x, q_lora, pos = self._batch(True, padded=6)
                    rows = md.get_seqlens_expanded().numel()
                    graph_pool, eager_pool = _Pool(dtype), _Pool(dtype)
                    output = torch.full(
                        (x.shape[0], 2048), -1, device="cuda", dtype=torch.int32
                    )

                    def run():
                        idx._forward_cuda_k_only(
                            x,
                            pos,
                            batch,
                            0,
                            self.act_quant,
                            md,
                            num_tokens=rows,
                            topk_result=output,
                        )

                    with patch.object(
                        self.dsa, "get_token_to_kv_pool", return_value=graph_pool
                    ):
                        stream = torch.cuda.Stream()
                        stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(stream):
                            for _ in range(3):
                                run()
                        torch.cuda.current_stream().wait_stream(stream)
                        torch.cuda.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            run()
                    for replay in range(3):
                        x.normal_()
                        pos.add_(1)
                        graph_pool.buffer.zero_()
                        output.fill_(-1)
                        graph.replay()
                        eager_pool.buffer.zero_()
                        with patch.object(
                            self.dsa, "get_token_to_kv_pool", return_value=eager_pool
                        ):
                            idx._forward_cuda_k_only(
                                x,
                                pos,
                                batch,
                                0,
                                self.act_quant,
                                md,
                                num_tokens=rows,
                                return_indices=False,
                            )
                        torch.cuda.synchronize()
                        torch.testing.assert_close(
                            graph_pool.buffer, eager_pool.buffer, rtol=0, atol=0
                        )
                        torch.testing.assert_close(
                            output[:rows], self._expected_indices(md), rtol=0, atol=0
                        )
                        self.assertTrue(output[rows:].eq(-1).all().item())
                        packed = graph_pool.packed_bytes_per_token
                        keys = graph_pool.buffer[:, : 64 * packed].reshape(
                            -1, 64, packed
                        )
                        padding_loc = batch.out_cache_loc[rows:]
                        self.assertTrue(
                            keys[padding_loc // 64, padding_loc % 64].eq(0).all().item()
                        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
