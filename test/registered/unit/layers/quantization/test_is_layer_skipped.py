import unittest

from sglang.srt.layers.quantization.utils import is_layer_skipped
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


# Qwen3-Next FP8 actually publishes the equivalent of this in
# packed_modules_mapping (qwen3_next.py:908-911). in_proj_ba / in_proj_qkvz
# are deliberately omitted because they are real, unified tensors.
QWEN3_NEXT_FUSED_MAPPING = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}


def _qwen3_next_ignored_layers(layer_idx: int, name: str) -> list:
    # Mirrors the normalization in Fp8Config.from_config: each entry is kept in
    # both "model.<...>" and bare "<...>" forms.
    base = f"layers.{layer_idx}.linear_attn.{name}"
    return [base, f"model.{base}"]


class TestIsLayerSkipped(CustomTestCase):
    def test_explicit_fused_module_exclusion(self):
        for name in ("qkv_proj", "gate_up_proj"):
            prefix = f"model.layers.0.{name}"
            for mapping in ({}, QWEN3_NEXT_FUSED_MAPPING):
                with self.subTest(name=name, mapping=mapping):
                    self.assertTrue(is_layer_skipped(prefix, [prefix], mapping))
                    self.assertFalse(
                        is_layer_skipped(prefix, [prefix + "_extra"], mapping)
                    )

    def test_unfused_shards_still_require_uniform_precision(self):
        for fused, shards in QWEN3_NEXT_FUSED_MAPPING.items():
            prefix = f"model.layers.0.{fused}"
            ignored = [f"model.layers.0.{name}" for name in shards]
            with self.subTest(fused=fused):
                self.assertTrue(is_layer_skipped(prefix, ignored, {}))
                with self.assertRaisesRegex(ValueError, "some but not all shards"):
                    is_layer_skipped(prefix, ignored[:1], {})
                self.assertFalse(is_layer_skipped(prefix, [], {}))

    def test_qwen3_next_in_proj_ba_is_skipped(self):
        # Regression for #23467: in_proj_ba is a unified tensor in the FP8
        # checkpoint. modules_to_not_convert lists it explicitly, so it must
        # bypass FP8 quantization (otherwise validate_block_quant_shapes raises
        # on output_partition_size=8 vs block_n=128 at tp=4).
        prefix = "model.layers.0.linear_attn.in_proj_ba"
        ignored = _qwen3_next_ignored_layers(0, "in_proj_ba")
        self.assertTrue(is_layer_skipped(prefix, ignored, QWEN3_NEXT_FUSED_MAPPING))

    def test_qwen3_next_in_proj_qkvz_is_skipped(self):
        prefix = "model.layers.5.linear_attn.in_proj_qkvz"
        ignored = _qwen3_next_ignored_layers(5, "in_proj_qkvz")
        self.assertTrue(is_layer_skipped(prefix, ignored, QWEN3_NEXT_FUSED_MAPPING))

    def test_mlp_gate_does_not_match_gate_up_proj(self):
        # The motivation for #23467: an entry "mlp.gate" in
        # modules_to_not_convert must NOT skip a sibling "mlp.gate_up_proj".
        ignored = ["mlp.gate"]
        self.assertFalse(
            is_layer_skipped("model.layers.0.mlp.gate_up_proj", ignored, {})
        )
        self.assertTrue(is_layer_skipped("model.layers.0.mlp.gate", ignored, {}))

    def test_trailing_dot_prefix_matches_child_modules(self):
        # Mixed-precision checkpoints may use a trailing-dot layer prefix to keep
        # every module under the layer in higher precision.
        ignored = ["model.layers.34."]
        self.assertTrue(
            is_layer_skipped("model.layers.34.mlp.experts.0.down_proj", ignored, {})
        )
        self.assertFalse(
            is_layer_skipped("model.layers.340.mlp.experts.0.down_proj", ignored, {})
        )


class TestMxfp4FusedCheckpointExclusion(CustomTestCase):
    def test_mapped_bf16_vision_qkv_loads_without_packing(self):
        import torch

        from sglang.srt.layers.linear import QKVParallelLinear
        from sglang.srt.layers.quantization.mxfp4 import Mxfp4Config
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
        from sglang.srt.models.glm5_next import Glm5NextForConditionalGeneration
        from sglang.srt.runtime_context import get_context

        model = Glm5NextForConditionalGeneration
        config = Mxfp4Config.from_config(
            {
                "quant_method": "mxfp4",
                "ignore": ["model.visual.blocks.0.attn.qkv"],
            }
        )
        config.update_packed_modules_mapping(model.packed_modules_mapping)
        config.apply_weight_name_mapper(model.hf_to_sglang_mapper)
        prefix = "visual.blocks.0.attn.qkv_proj"

        with (
            get_context().override_server_args(
                model_path="dummy", device="cpu", enable_lora=False, lora_paths=None
            ),
            torch.device("cpu"),
        ):
            layer = QKVParallelLinear(
                1024,
                64,
                16,
                bias=False,
                params_dtype=torch.bfloat16,
                quant_config=config,
                prefix=prefix,
                parallel_group="replicated",
            )
            self.assertIsInstance(layer.quant_method, UnquantizedLinearMethod)
            self.assertEqual(layer.weight.dtype, torch.bfloat16)
            self.assertEqual(layer.weight.shape, (3072, 1024))
            checkpoint_weight = (
                torch.arange(3072 * 1024).remainder(251).to(torch.bfloat16)
            ).reshape(3072, 1024)
            layer.weight.weight_loader(layer.weight, checkpoint_weight)
            torch.testing.assert_close(layer.weight, checkpoint_weight, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
