"""SAIL MegaAttention preserves checkpoint FP8 payloads and channel scales."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.deepseek_v4_mega_backend import prepare_sail_wo_a
from sglang.srt.models import deepseek_v4
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMegaWoA(CustomTestCase):
    def setUp(self):
        self.groups, self.rank, self.dim = 2, 4, 4096
        self.weight = (
            torch.arange(8 * self.dim).reshape(8, self.dim).remainder(31) - 15
        ).to(torch.float8_e4m3fn)
        self.scale = torch.arange(1, 9, dtype=torch.float32).view(8, 1) / 128

    def assert_prepared(self, weight, scale):
        actual_w, actual_s = prepare_sail_wo_a(weight, scale, self.groups, self.rank)
        # Independent column-index oracle: [head, tile, lane] -> [tile, head, lane].
        cols = torch.tensor(
            [
                head * 512 + tile * 32 + lane
                for tile in range(16)
                for head in range(8)
                for lane in range(32)
            ]
        )
        expected = self.weight.view(torch.uint8)[:, cols].reshape(2, 4, self.dim)
        self.assertTrue(torch.equal(actual_w.view(torch.uint8), expected))
        self.assertTrue(torch.equal(actual_s, self.scale.reshape(2, 4, 1)))
        self.assertNotEqual(actual_s.data_ptr(), scale.data_ptr())
        self.assertNotEqual(actual_w.data_ptr(), weight.data_ptr())

    def test_checkpoint_and_processed_layouts(self):
        for processed in (False, True):
            with self.subTest(processed=processed):
                self.assert_prepared(
                    self.weight.t() if processed else self.weight,
                    self.scale.t() if processed else self.scale,
                )

    def test_fp8_does_not_requantize(self):
        with patch(
            "sglang.kernels.ops.quantization.fp8_kernel.sglang_per_token_quant_fp8",
            side_effect=AssertionError("FP8 weights must not be requantized"),
        ):
            self.assert_prepared(self.weight, self.scale)

    def test_missing_or_non_channelwise_scales_fail(self):
        for scale in (None, torch.ones(1), torch.ones(8, 32), self.scale.bfloat16()):
            with self.subTest(scale=scale):
                with self.assertRaisesRegex(ValueError, "one FP32 weight_scale"):
                    prepare_sail_wo_a(self.weight, scale, 2, 4)

    def test_invalid_weight_layout_fails(self):
        for weight in (self.weight[:, :2048], self.weight[:4], self.weight.int()):
            with self.subTest(shape=weight.shape, dtype=weight.dtype):
                with self.assertRaises(ValueError):
                    prepare_sail_wo_a(weight, self.scale, 2, 4)

    def test_setup_reload_preserves_graph_buffer_addresses(self):
        layer = SimpleNamespace(
            use_mega_attention=True,
            n_local_heads=8,
            n_local_groups=self.groups,
            o_lora_rank=self.rank,
            wo_a=SimpleNamespace(weight=self.weight.clone(), weight_scale=self.scale),
            mega_wo_a=None,
        )
        with patch.object(deepseek_v4, "_is_sail", True):
            deepseek_v4.MQALayer.setup_mega_attention(layer)
            ptrs = tuple(t.data_ptr() for t in layer.mega_wo_a)
            layer.wo_a.weight = self.weight.t()
            layer.wo_a.weight_scale = self.scale.t() * 2
            deepseek_v4.MQALayer.setup_mega_attention(layer)
        self.assertEqual(ptrs, tuple(t.data_ptr() for t in layer.mega_wo_a))
        self.assertTrue(
            torch.equal(layer.mega_wo_a[1], self.scale.reshape(2, 4, 1) * 2)
        )
        self.assertEqual(layer.mega_wo_a_recipe, (1, 1, 4096))


if __name__ == "__main__":
    unittest.main()
