"""GLM-5.2 NextN quantization configuration and weight-loading regressions.

The feature is off by default. Without a case that turns the flag on, CI never
touches the Quark scheme rewrite and cannot claim the path works. These tests
exercise that ON wiring on CPU without loading a 70B MXFP4 checkpoint.

A full serve+generate job still needs the MXFP4 weights in the runner cache;
register that separately as nightly if the checkpoint is present.
"""

from __future__ import annotations

import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization.quark.utils import should_ignore_layer
from sglang.srt.models.glm4_moe import (
    GlmMoeDsaForCausalLMNextN,
    enable_glm_nextn_moe_ptpc,
    should_apply_glm_nextn_moe_ptpc,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

LAYER = 78
PREFIX = f"model.layers.{LAYER}"
EXPERT_LEAF = f"{PREFIX}.mlp.experts.0.w1"
SHARED_EXPERT_LEAF = f"{PREFIX}.mlp.shared_experts.gate_proj"
ATTN_LEAF = f"{PREFIX}.self_attn.q_proj"
EXPERT_WEIGHT = f"{PREFIX}.mlp.experts.0.gate_proj.weight"
SHARED_EXPERT_WEIGHT = f"{PREFIX}.mlp.shared_experts.gate_proj.weight"
_PTPC_ENV = "sglang.srt.models.glm4_moe.envs.SGLANG_GLM_NEXTN_MOE_PTPC.get"


def _quark_cfg(*, exclude=None, layer_quant=None):
    return SimpleNamespace(
        get_name=lambda: "quark",
        quant_config={"layer_quant_config": dict(layer_quant or {}), "exclude": []},
        exclude_layers=list(
            exclude if exclude is not None else [EXPERT_LEAF, ATTN_LEAF]
        ),
    )


class TestEnableGlmNextnMoePtpc(CustomTestCase):
    def test_off_by_default(self):
        self.assertFalse(enable_glm_nextn_moe_ptpc(_quark_cfg()))

    def test_on_requires_quark(self):
        with patch(
            _PTPC_ENV,
            return_value=True,
        ):
            self.assertTrue(enable_glm_nextn_moe_ptpc(_quark_cfg()))
            self.assertFalse(
                enable_glm_nextn_moe_ptpc(SimpleNamespace(get_name=lambda: "fp8"))
            )
            self.assertFalse(enable_glm_nextn_moe_ptpc(None))

    def test_apply_requires_excluded_mtp_experts(self):
        with patch(
            _PTPC_ENV,
            return_value=True,
        ):
            self.assertTrue(should_apply_glm_nextn_moe_ptpc(_quark_cfg(), LAYER))
            self.assertFalse(
                should_apply_glm_nextn_moe_ptpc(_quark_cfg(exclude=[ATTN_LEAF]), LAYER)
            )

    def test_ptpc_disables_shared_expert_fusion(self):
        hf_config = SimpleNamespace(
            architectures=["GlmMoeDsaForCausalLMNextN"],
            n_routed_experts=256,
            n_shared_experts=1,
            num_hidden_layers=LAYER,
        )
        # Shared expert BF16 (excluded) or still quantized: neither fits the slot.
        for exclude in ([EXPERT_LEAF, SHARED_EXPERT_LEAF], [EXPERT_LEAF]):
            with self.subTest(exclude=exclude), patch(_PTPC_ENV, return_value=True):
                reason = GlmMoeDsaForCausalLMNextN.shared_experts_fusion_disable_reason(
                    hf_config, _quark_cfg(exclude=exclude)
                )
                self.assertIn("PTPC", reason or "")


class TestResolveNextnQuantConfigPtpcOn(CustomTestCase):
    def _resolve(self, cfg, flag: bool):
        model = GlmMoeDsaForCausalLMNextN.__new__(GlmMoeDsaForCausalLMNextN)
        hf = SimpleNamespace(num_hidden_layers=LAYER)
        with patch(
            _PTPC_ENV,
            return_value=flag,
        ):
            return model._resolve_nextn_quant_config(hf, cfg)

    def test_flag_off_excludes_fused_experts(self):
        src = _quark_cfg()
        out = self._resolve(src, flag=False)
        self.assertIn("model.decoder.mlp.experts", out.exclude_layers)
        self.assertNotIn(
            "model.decoder.mlp.experts",
            out.quant_config.get("layer_quant_config", {}),
        )

    def test_flag_on_assigns_ptpc_scheme_instead_of_bf16_exclude(self):
        src = _quark_cfg()
        out = self._resolve(src, flag=True)
        self.assertNotIn("model.decoder.mlp.experts", out.exclude_layers)
        scheme = out.quant_config["layer_quant_config"]["model.decoder.mlp.experts"]
        self.assertEqual(scheme["weight"]["dtype"], "fp8_e4m3")
        self.assertEqual(scheme["weight"]["qscheme"], "per_channel")
        self.assertFalse(scheme["weight"]["is_dynamic"])
        self.assertEqual(scheme["input_tensors"]["dtype"], "fp8_e4m3")
        self.assertEqual(scheme["input_tensors"]["qscheme"], "per_channel")
        self.assertTrue(scheme["input_tensors"]["is_dynamic"])

    def test_flag_on_fused_experts_not_ignored_by_quark(self):
        self.assertTrue(
            should_ignore_layer(
                "model.decoder.mlp.experts",
                ignore=self._resolve(_quark_cfg(), flag=False).exclude_layers,
            )
        )
        self.assertFalse(
            should_ignore_layer(
                "model.decoder.mlp.experts",
                ignore=self._resolve(_quark_cfg(), flag=True).exclude_layers,
            )
        )

    def test_flag_on_does_not_mutate_caller_config(self):
        src = _quark_cfg()
        orig_exclude = list(src.exclude_layers)
        orig_layer = dict(src.quant_config.get("layer_quant_config") or {})
        self._resolve(src, flag=True)
        self.assertEqual(src.exclude_layers, orig_exclude)
        self.assertEqual(src.quant_config.get("layer_quant_config") or {}, orig_layer)

    def test_flag_on_skips_ptpc_when_experts_not_excluded(self):
        src = _quark_cfg(exclude=[ATTN_LEAF])
        out = self._resolve(src, flag=True)
        self.assertNotIn("model.decoder.mlp.experts", out.exclude_layers)
        self.assertNotIn(
            "model.decoder.mlp.experts",
            out.quant_config.get("layer_quant_config", {}),
        )

    def test_flag_on_skips_ptpc_when_mtp_not_in_exclude(self):
        src = _quark_cfg(exclude=[])
        out = self._resolve(src, flag=True)
        self.assertNotIn(
            "model.decoder.mlp.experts",
            out.quant_config.get("layer_quant_config", {}),
        )


class TestMaybeQuantGlmNextnMoeToPtpc(CustomTestCase):
    def _cast(self, cfg, flag: bool):
        loader = GlmMoeDsaForCausalLMNextN.__new__(GlmMoeDsaForCausalLMNextN)
        loader.quant_config = cfg
        loader.config = SimpleNamespace(num_hidden_layers=LAYER)
        weights = [
            (EXPERT_WEIGHT, torch.ones(4, 8, dtype=torch.bfloat16)),
            (
                SHARED_EXPERT_WEIGHT,
                torch.full((4, 8), 1e-4, dtype=torch.bfloat16),
            ),
        ]
        with patch(
            _PTPC_ENV,
            return_value=flag,
        ):
            return list(loader._maybe_quant_glm_nextn_moe_to_ptpc(weights))

    def test_flag_on_casts_excluded_bf16_experts(self):
        out = self._cast(_quark_cfg(), flag=True)
        names = [name for name, _ in out]
        self.assertIn(EXPERT_WEIGHT, names)
        self.assertIn(EXPERT_WEIGHT[: -len("weight")] + "weight_scale", names)
        weight = dict(out)[EXPERT_WEIGHT]
        self.assertEqual(weight.dtype, torch.float8_e4m3fn)
        self.assertEqual(dict(out)[SHARED_EXPERT_WEIGHT].dtype, torch.bfloat16)
        self.assertNotIn(SHARED_EXPERT_WEIGHT[: -len("weight")] + "weight_scale", names)

    def test_flag_on_does_not_cast_when_experts_not_excluded(self):
        src = _quark_cfg(exclude=[ATTN_LEAF])
        out = self._cast(src, flag=True)
        self.assertEqual(len(out), 2)
        self.assertEqual(dict(out)[EXPERT_WEIGHT].dtype, torch.bfloat16)
        self.assertEqual(dict(out)[SHARED_EXPERT_WEIGHT].dtype, torch.bfloat16)


class TestResolveNextnMxfp4QuantConfig(CustomTestCase):
    """Checkpoint FP8/BF16 rules must survive the MTP runtime prefix change.

    Losing that mapping allocated MXFP4 packed shared-expert parameters for
    FP8 checkpoint tensors, so the merged gate/up loader rejected their shapes.
    """

    def _config(self, layers):
        from sglang.srt.layers.quantization.mxfp4 import Mxfp4Config

        return Mxfp4Config(
            is_checkpoint_mxfp4_serialized=True,
            fp8_channelwise_layers=[
                f"model.layers.{layer}.{name}"
                for layer in layers
                for name in (
                    "mlp.shared_experts.gate_proj",
                    "mlp.shared_experts.up_proj",
                    "self_attn.q_a_proj",
                    "self_attn.kv_a_proj_with_mqa",
                )
            ],
            ignored_layers=[
                f"model.layers.{layer}.{name}"
                for layer in layers
                for name in ("eh_proj", "mlp.gate", "self_attn.indexer.weights_proj")
            ],
            packed_modules_mapping=GlmMoeDsaForCausalLMNextN.packed_modules_mapping,
        )

    def _resolve(self, cfg, layer):
        model = GlmMoeDsaForCausalLMNextN.__new__(GlmMoeDsaForCausalLMNextN)
        return model._resolve_nextn_quant_config(
            SimpleNamespace(num_hidden_layers=layer), cfg
        )

    def test_dynamic_mtp_layer_selects_fp8_and_bf16_methods(self):
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
        from sglang.srt.layers.quantization.w8a8_fp8 import W8A8Fp8LinearMethod

        # Hardware predicates are boundaries; selection uses the real configs.
        with (
            patch("sglang.srt.layers.quantization.mxfp4._is_sail", True),
            patch(
                "sglang.srt.layers.quantization.w8a8_fp8.cutlass_fp8_supported",
                return_value=False,
            ),
        ):
            for layer in (61, 78, 80):
                cfg = self._resolve(self._config([layer]), layer)
                for prefix, expected in (
                    (
                        "model.decoder.mlp.shared_experts.gate_up_proj",
                        W8A8Fp8LinearMethod,
                    ),
                    (
                        "model.decoder.self_attn.fused_qkv_a_proj_with_mqa",
                        W8A8Fp8LinearMethod,
                    ),
                    ("model.eh_proj", UnquantizedLinearMethod),
                    ("model.decoder.mlp.gate", UnquantizedLinearMethod),
                    (
                        "model.decoder.self_attn.indexer.weights_proj",
                        UnquantizedLinearMethod,
                    ),
                ):
                    with self.subTest(layer=layer, prefix=prefix):
                        linear = LinearBase(128, 32, quant_config=cfg, prefix=prefix)
                        self.assertIsInstance(linear.quant_method, expected)

    def test_shared_gate_up_loads_fp8_weights_and_channel_scales(self):
        from sglang.srt.layers.linear import MergedColumnParallelLinear

        cfg = self._resolve(self._config([78]), 78)
        with (
            torch.device("cpu"),
            patch("sglang.srt.layers.quantization.mxfp4._is_sail", True),
            patch(
                "sglang.srt.layers.quantization.w8a8_fp8.cutlass_fp8_supported",
                return_value=False,
            ),
        ):
            linear = MergedColumnParallelLinear(
                128,
                [32, 32],
                bias=False,
                params_dtype=torch.bfloat16,
                quant_config=cfg,
                prefix="model.decoder.mlp.shared_experts.gate_up_proj",
                parallel_group="replicated",
            )
            weights, scales = [], []
            for shard in (0, 1):
                weight = torch.full((32, 128), shard + 1.0).to(torch.float8_e4m3fn)
                scale = torch.full((32, 1), (shard + 1) * 0.5)
                linear.weight_loader(linear.weight, weight, shard)
                linear.weight_loader(linear.weight_scale, scale, shard)
                weights.append(weight.float())
                scales.append(scale)
            self.assertEqual(linear.weight.dtype, torch.float8_e4m3fn)
            self.assertEqual(linear.weight.shape, (64, 128))
            self.assertEqual(linear.weight_scale.shape, (64, 1))
            torch.testing.assert_close(linear.weight.float(), torch.cat(weights))
            torch.testing.assert_close(linear.weight_scale, torch.cat(scales))

    def test_remapping_is_private_scoped_and_idempotent(self):
        cfg = self._config([77, 78, 80, 780])
        cfg.fp8_channelwise_layers.append("model.decoder.self_attn.o_proj")
        cfg.ignored_layers.append("model.layers.78")
        original = deepcopy(vars(cfg))
        first = self._resolve(cfg, 78)
        second = self._resolve(cfg, 80)
        self.assertIsNot(first, cfg)
        self.assertEqual(vars(cfg), original)
        self.assertEqual(vars(self._resolve(first, 78)), vars(first))
        self.assertIn("model.decoder", first.ignored_layers)
        self.assertIn("model.layers.78", second.ignored_layers)
        for layer in (77, 80, 780):
            self.assertIn(
                f"model.layers.{layer}.self_attn.q_a_proj", first.fp8_channelwise_layers
            )
        self.assertIn(
            "model.layers.78.self_attn.q_a_proj", second.fp8_channelwise_layers
        )
        self.assertNotIn(
            "model.layers.80.self_attn.q_a_proj", second.fp8_channelwise_layers
        )
        first.fp8_channelwise_layers.append("draft-only")
        first.ignored_layers.clear()
        self.assertEqual(vars(cfg), original)

    def test_absent_or_empty_name_lists_keep_their_meaning(self):
        for names in (None, []):
            with self.subTest(names=names):
                cfg = self._config([])
                cfg.fp8_channelwise_layers = names
                cfg.ignored_layers = names
                resolved = self._resolve(cfg, 78)
                self.assertEqual(resolved.fp8_channelwise_layers, names)
                self.assertEqual(resolved.ignored_layers, names)
                self.assertTrue(resolved.is_checkpoint_mxfp4_serialized)


if __name__ == "__main__":
    unittest.main()
