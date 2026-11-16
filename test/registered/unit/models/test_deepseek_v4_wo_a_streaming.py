"""Unit tests for the DSv4 wo_a streaming-dequant scale-name handling.

``_dequant_fp8_wo_a_streaming`` pairs ``.wo_a`` weights with their block
scales as checkpoints stream past. Beyond the legacy ``.scale`` name, W8A8
INT8 checkpoints carry ``.weight_scale``/``.weight_scale_inv``; a quantized
(fp8/int8) wo_a weight that streams without its scale must fail loudly
instead of silently staying quantized, while a pre-dequantized bf16 weight
still passes through. CPU-only: tensors are tiny and the pairing logic does
not touch a device.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization.mxfp4 import Mxfp4Config
from sglang.srt.models import deepseek_v4
from sglang.srt.models.deepseek_v4 import _dequant_fp8_wo_a_streaming
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def fp8_weight(seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(128, 128, generator=g) * 0.1).to(torch.float8_e4m3fn)


def fp32_scale(seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, 1, generator=g) + 0.5


def expected_bf16(weight, scale):
    return (weight.float() * scale).to(torch.bfloat16)


class TestWoAStreamingScaleNames(CustomTestCase):
    def run_stream(self, entries):
        return list(_dequant_fp8_wo_a_streaming(iter(entries)))

    def assert_dequantized(self, out, name, weight, scale):
        self.assertEqual([n for n, _ in out], [name])
        self.assertEqual(out[0][1].dtype, torch.bfloat16)
        self.assertTrue(torch.equal(out[0][1], expected_bf16(weight, scale)))

    def test_weight_then_scale_legacy_suffix(self):
        w, s = fp8_weight(), fp32_scale()
        out = self.run_stream(
            [("m.wo_a.weight", w), ("m.wo_a.scale", s), ("m.other", torch.ones(2))]
        )
        self.assert_dequantized(out[:1], "m.wo_a.weight", w, s)
        self.assertEqual(out[1][0], "m.other")

    def test_scale_then_weight_weight_scale_suffix(self):
        w, s = fp8_weight(3), fp32_scale(4)
        out = self.run_stream([("m.wo_a.weight_scale", s), ("m.wo_a.weight", w)])
        self.assert_dequantized(out, "m.wo_a.weight", w, s)

    def test_weight_scale_inv_suffix(self):
        w, s = fp8_weight(5), fp32_scale(6)
        out = self.run_stream([("m.wo_a.weight", w), ("m.wo_a.weight_scale_inv", s)])
        self.assert_dequantized(out, "m.wo_a.weight", w, s)

    def test_int8_weight_without_scale_raises(self):
        entries = [("m.wo_a.weight", torch.zeros(128, 128, dtype=torch.int8))]
        with self.assertRaisesRegex(ValueError, "missing its scale"):
            self.run_stream(entries)

    def test_bf16_weight_without_scale_passes_through(self):
        w = torch.randn(8, 8, dtype=torch.bfloat16)
        out = self.run_stream([("m.wo_a.weight", w)])
        self.assertEqual([n for n, _ in out], ["m.wo_a.weight"])
        self.assertTrue(torch.equal(out[0][1], w))

    def test_orphan_scale_raises(self):
        entries = [("m.wo_a.weight_scale", fp32_scale())]
        with self.assertRaisesRegex(ValueError, "missing its weight"):
            self.run_stream(entries)

    def test_only_configured_channelwise_layer_keeps_fp8_and_scale(self):
        w, s = fp8_weight(), fp32_scale()
        keep = "layers.0.attn.wo_a"
        base = "layers.1.attn.wo_a"
        entries = [
            (keep + ".scale", s),
            (base + ".weight", w),
            (keep + ".weight", w),
            (base + ".weight_scale", s),
        ]
        out = dict(
            _dequant_fp8_wo_a_streaming(
                iter(entries), keep_quantized=lambda prefix: prefix == keep
            )
        )
        self.assertEqual(
            set(out), {keep + ".scale", keep + ".weight", base + ".weight"}
        )
        self.assertIs(out[keep + ".weight"], w)
        self.assertIs(out[keep + ".scale"], s)
        self.assertTrue(torch.equal(out[base + ".weight"], expected_bf16(w, s)))

    def test_selected_layer_rejects_inconsistent_weight_dtype(self):
        entries = [("m.wo_a.weight", torch.zeros(2, 2, dtype=torch.bfloat16))]
        with self.assertRaisesRegex(ValueError, "expected per-channel FP8"):
            list(_dequant_fp8_wo_a_streaming(entries, keep_quantized=lambda _: True))

    def test_preserved_channelwise_weight_and_scale_must_be_paired(self):
        prefix = "layers.0.attn.wo_a"
        weight, scale = fp8_weight(), fp32_scale()
        for suffix in ("scale", "weight_scale", "weight_scale_inv"):
            pair = [(prefix + ".weight", weight), (prefix + "." + suffix, scale)]
            for entries in (pair, pair[::-1]):
                with self.subTest(suffix=suffix, first=entries[0][0]):
                    actual = list(
                        _dequant_fp8_wo_a_streaming(
                            entries, keep_quantized=lambda _: True
                        )
                    )
                    self.assertEqual(
                        [name for name, _ in actual], [name for name, _ in entries]
                    )
                    for (_, got), (_, expected) in zip(actual, entries):
                        self.assertIs(got, expected)
            for entry, missing in zip(pair, ("scale", "weight")):
                with (
                    self.subTest(suffix=suffix, missing=missing),
                    self.assertRaisesRegex(ValueError, f"missing its {missing}"),
                ):
                    list(
                        _dequant_fp8_wo_a_streaming(
                            [entry], keep_quantized=lambda _: True
                        )
                    )

    def test_mtp_checkpoint_name_maps_to_the_quantized_decoder_module(self):
        remap = deepseek_v4.DeepseekV4ForCausalLM.remap_weight_name_to_dpsk_hf_format
        raw_name = remap("mtp.0.attn.wo_a.weight", is_nextn=True, num_hidden_layers=40)
        self.assertEqual(
            deepseek_v4._nextn_decoder_weight_name(raw_name, "model.layers.40"),
            "model.decoder.self_attn.wo_a.weight",
        )
        other_layer = "model.layers.400.self_attn.wo_a.weight"
        self.assertEqual(
            deepseek_v4._nextn_decoder_weight_name(other_layer, "model.layers.40"),
            other_layer,
        )

    def test_ppu_channelwise_gate_uses_checkpoint_list_and_existing_switch(self):
        prefix = "model.layers.0.self_attn.wo_a"
        config = Mxfp4Config(fp8_channelwise_layers=["layers.0.self_attn.wo_a"])
        for sail, switch, layer, backend, expected, reads_exec in (
            (True, True, prefix, "flashmla", True, True),
            (True, True, prefix, "flashmla_mega", False, True),
            (True, True, prefix, None, False, False),
            (True, True, "model.layers.1.self_attn.wo_a", "flashmla", False, False),
            (True, False, prefix, "flashmla", False, False),
            (False, True, prefix, "flashmla", False, False),
        ):
            with (
                self.subTest(sail=sail, switch=switch, layer=layer, backend=backend),
                patch.object(deepseek_v4, "_is_sail", sail),
                patch.object(deepseek_v4, "_FP8_WO_A_GEMM", switch),
                patch.object(deepseek_v4, "get_context") as get_context,
                patch.object(deepseek_v4, "get_exec") as get_exec,
            ):
                get_context.return_value.is_config_namespace_published.return_value = (
                    backend is not None
                )
                if backend is None:
                    get_exec.side_effect = ValueError("exec namespace not published")
                else:
                    get_exec.return_value.kernel = SimpleNamespace(
                        dsv4_attn_backend=backend
                    )
                self.assertEqual(
                    deepseek_v4._ppu_channelwise_wo_a_enabled(config, layer),
                    expected,
                )
                self.assertEqual(get_exec.call_count, int(reads_exec))

    def test_ppu_channelwise_online_update_guard_uses_global_layers(self):
        config = Mxfp4Config(fp8_channelwise_layers=["layers.3.self_attn.wo_a"])
        for sail, switch, backend, num_layers, blocked in (
            (True, True, "flashmla", 4, True),
            (True, True, "flashmla", 3, False),
            (True, True, "flashmla_mega", 4, False),
            (True, True, None, 4, False),
            (True, False, "flashmla", 4, False),
            (False, True, "flashmla", 4, False),
        ):
            with (
                self.subTest(sail=sail, switch=switch, backend=backend, n=num_layers),
                patch.object(deepseek_v4, "_is_sail", sail),
                patch.object(deepseek_v4, "_FP8_WO_A_GEMM", switch),
                patch.object(deepseek_v4, "get_context") as get_context,
                patch.object(deepseek_v4, "get_exec") as get_exec,
            ):
                get_context.return_value.is_config_namespace_published.return_value = (
                    backend is not None
                )
                get_exec.return_value.kernel.dsv4_attn_backend = backend
                # The only matching layer need not be present on this rank.
                error = deepseek_v4._ppu_channelwise_wo_a_update_error(
                    config, "model", num_layers
                )
                if blocked:
                    self.assertIn("SGLANG_OPT_FP8_WO_A_GEMM=0", error)
                else:
                    self.assertIsNone(error)


if __name__ == "__main__":
    unittest.main()
