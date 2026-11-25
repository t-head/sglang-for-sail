import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.dsa_backend import (
    DeepseekSparseAttnBackend,
    _flashmla_kv_padded_heads,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDSAFlashMLAKVHeadPadding(CustomTestCase):
    def test_ppu_preserves_native_heads(self):
        head_counts = (1, 8, 32, 64, 65, 96, 128, 129, 256)
        with patch("sglang.srt.layers.attention.dsa_backend._is_sail", True):
            for num_heads in head_counts:
                with self.subTest(num_heads=num_heads):
                    self.assertEqual(
                        _flashmla_kv_padded_heads(num_heads), num_heads
                    )

    def test_other_platforms_keep_flashmla_padding(self):
        cases = ((1, 64), (64, 64), (65, 128), (128, 128), (129, 129))
        with patch("sglang.srt.layers.attention.dsa_backend._is_sail", False):
            for num_heads, expected in cases:
                with self.subTest(num_heads=num_heads):
                    self.assertEqual(_flashmla_kv_padded_heads(num_heads), expected)

    def test_ppu_decode_uses_native_heads_for_q_metadata_and_output(self):
        num_heads = 8
        head_dim = 4
        v_head_dim = 3
        backend = DeepseekSparseAttnBackend.__new__(DeepseekSparseAttnBackend)
        backend.physical_page_size = 64
        backend.kv_cache_dim = head_dim
        backend.dsa_kv_cache_store_fp8 = True
        backend.dsa_index_kpool = 1
        backend.dsa_index_topk = 2
        backend.flashmla_kv_format = "V32"

        kernel = ModuleType("sglang.kernels.ops.attention.flash_mla_interface")
        observed = {}

        def get_mla_metadata(**kwargs):
            observed["metadata_heads"] = kwargs["num_heads_q"]
            return torch.tensor([1]), torch.tensor([0, 1], dtype=torch.int32)

        def flash_mla_with_kvcache(q, **kwargs):
            observed["q_shape"] = q.shape
            output = q.new_zeros(q.shape[0], q.shape[1], q.shape[2], v_head_dim)
            lse = q.new_zeros(q.shape[0], q.shape[1], q.shape[2])
            return output, lse

        kernel.get_mla_metadata = get_mla_metadata
        kernel.flash_mla_with_kvcache = flash_mla_with_kvcache
        cache_seqlens = torch.tensor([2], dtype=torch.int32)

        with (
            patch("sglang.srt.layers.attention.dsa_backend._is_sail", True),
            patch.dict("sys.modules", {kernel.__name__: kernel}),
        ):
            backend.flashmla_kv_num_q_heads = _flashmla_kv_padded_heads(num_heads)
            flashmla_metadata = backend._compute_flashmla_metadata(
                cache_seqlens=cache_seqlens,
                seq_len_q=1,
            )
            metadata = SimpleNamespace(
                dsa_cache_seqlens_int32=cache_seqlens,
                flashmla_metadata=flashmla_metadata,
            )
            output = backend._forward_flashmla_kv(
                q_all=torch.randn(1, num_heads, head_dim),
                kv_cache=torch.zeros(64, head_dim),
                v_head_dim=v_head_dim,
                sm_scale=1.0,
                layer=SimpleNamespace(tp_q_head_num=num_heads, head_dim=head_dim),
                metadata=metadata,
                page_table_1=torch.tensor([[0, 1]], dtype=torch.int32),
            )

        self.assertEqual(observed["metadata_heads"], num_heads)
        self.assertEqual(observed["q_shape"], (1, 1, num_heads, head_dim))
        self.assertEqual(output.shape, (1, 1, num_heads, v_head_dim))


if __name__ == "__main__":
    unittest.main()
