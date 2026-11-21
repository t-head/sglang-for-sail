import sys
import unittest
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.dsa import dsa_indexer
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _Stream:
    def __init__(self):
        self.pending_writes = []

    def wait_stream(self, stream):
        for write in stream.pending_writes:
            write()
        stream.pending_writes.clear()


class TestDsaIndexerStreams(CustomTestCase):
    def test_ppu_fp4_paged_graph_reads_the_new_key(self):
        for mode in (
            ForwardMode.DECODE,
            ForwardMode.TARGET_VERIFY,
            ForwardMode.DRAFT_EXTEND_V2,
        ):
            with self.subTest(mode=mode.name):
                self._check_paged_graph_reads_the_new_key(mode)

    def _check_paged_graph_reads_the_new_key(self, mode):
        """A key produced on the auxiliary stream must win TopK immediately.

        The kernel boundary checks the producer's stream ownership. It defers
        auxiliary cache writes until the consumer joins that stream, so an
        omitted join deterministically exposes the previous cache contents.
        """
        main_stream, alt_stream = _Stream(), _Stream()
        active_stream = main_stream
        cache = torch.tensor([[1], [-1]], dtype=torch.int32)
        query = torch.ones((1, 1), dtype=torch.bfloat16)
        key = torch.tensor([7], dtype=torch.int32)

        @contextmanager
        def use_stream(stream):
            nonlocal active_stream
            previous_stream = active_stream
            if stream is not None:
                active_stream = stream
            try:
                yield
            finally:
                active_stream = previous_stream

        def quantize_query(query, axis):
            self.assertIs(active_stream, main_stream)
            return query.to(torch.uint8), None

        def store_key(*, key, **kwargs):
            self.assertIs(
                active_stream,
                alt_stream,
                "FP4 K cache writes must consume the key on its producer stream",
            )
            active_stream.pending_writes.append(lambda: cache[1].copy_(key))

        def select_from_cache(batch, layer_id, quantized_query, weights, metadata):
            logits = quantized_query[0].to(torch.int32) * cache.T
            return logits.argmax(dim=-1, keepdim=True)

        indexer = SimpleNamespace(
            alt_stream=alt_stream,
            use_fp4=True,
            use_fp8=False,
            use_int8=False,
            bf16_indexer=False,
            use_gfx950_fused_indexer=False,
            use_dsa_indexer_fusion=False,
            weights_proj=SimpleNamespace(),
            _wk_gate_weight=None,
            _should_skip_logits_computation=lambda batch: False,
            _aiter_fused_fp8_active=lambda batch: False,
            _get_q_k_bf16=lambda *args, **kwargs: (query, key, None),
            _store_index_k_cache=store_key,
            _get_logits_head_gate_no_q_scale=lambda x: None,
            _get_topk_paged=select_from_cache,
        )
        batch = SimpleNamespace(
            forward_mode=mode,
            seq_lens=torch.tensor([2], dtype=torch.int32),
        )
        act_quant_module = ModuleType(
            "sglang.kernels.ops.quantization.act_quant_triton"
        )
        act_quant_module.act_quant = None
        fp4_module = ModuleType("sglang.kernels.ops.attention.dsa.triton_kernel")
        fp4_module.downcast_to_mxfp4_indexer = quantize_query
        backend = SimpleNamespace(get_indexer_metadata=lambda *args: object())
        with (
            patch.dict(
                sys.modules,
                {
                    act_quant_module.__name__: act_quant_module,
                    fp4_module.__name__: fp4_module,
                },
            ),
            patch.multiple(
                dsa_indexer,
                _is_cuda=True,
                _is_hip=False,
                _is_npu=False,
                DUAL_STREAM_TOKEN_THRESHOLD=8,
                get_is_capture_mode=lambda: True,
                is_sail=lambda: True,
                _is_in_piecewise_or_breakable_cuda_graph=lambda: False,
                is_graph_dsa_split_op_surface=lambda batch: False,
                get_attn_backend=lambda: backend,
                _broadcast_indexer_topk_from_rank0=lambda result: result,
                maybe_capture_indexer_topk=lambda layer, result: result,
            ),
            patch.object(
                torch.cuda, "current_stream", side_effect=lambda: active_stream
            ),
            patch.object(torch.cuda, "stream", side_effect=use_stream),
        ):
            selected = dsa_indexer.Indexer.forward_cuda(
                indexer, query, query, torch.tensor([1]), batch, layer_id=0
            )

        torch.testing.assert_close(selected, torch.tensor([[1]]))


if __name__ == "__main__":
    unittest.main()
