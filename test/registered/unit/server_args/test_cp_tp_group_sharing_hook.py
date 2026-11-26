"""Kimi, Qwen and GLM CP-TP admission gates without devices or weights."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups import cp_tp_group_sharing_hook
from sglang.srt.arg_groups.cp_tp_group_sharing_hook import (
    resolve_cp_tp_group_sharing,
    validate_cp_tp_group_sharing_after_a2a,
)
from sglang.srt.arg_groups.overrides import declare_resolution, resolved_view
from sglang.srt.environ import envs
from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig
from sglang.srt.runtime_context import override_platform
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_K31 = SimpleNamespace(
    hf_config=SimpleNamespace(architectures=["KimiK31ForConditionalGeneration"])
)
_QWEN_EXP = SimpleNamespace(
    hf_config=SimpleNamespace(architectures=["Qwen4ExpForConditionalGeneration"])
)


def _args(**kwargs):
    base = dict(
        model_path="dummy",
        tp_size=2,
        attn_cp_size=2,
        enable_prefill_cp=True,
        cp_strategy="interleave",
    )
    return ServerArgs(**(base | kwargs))


_GLM = SimpleNamespace(
    hf_config=SimpleNamespace(architectures=["Glm5NextForConditionalGeneration"])
)


def _resolve(
    args,
    model,
    *,
    is_ppu=True,
    device_sm=89,
    protocol="rdma",
    force_tcp=None,
    staging=False,
    num_heads=64,
    after_a2a=False,
):
    # Only the hardware/environment and model-config boundaries are substituted.
    # Keep resolution, declarations and the final runtime view real.
    hook = cp_tp_group_sharing_hook
    with (
        patch.object(
            hook,
            "mambaish_config",
            return_value=SimpleNamespace(linear_attn_config={"num_heads": num_heads}),
        ),
        override_platform(is_ppu=is_ppu, device_sm=device_sm),
        envs.MOONCAKE_PROTOCOL.override(protocol),
        envs.SGLANG_DISAGG_STAGING_BUFFER.override(staging),
        patch.dict(os.environ, {}, clear=False),
    ):
        os.environ.pop("MC_FORCE_TCP", None)
        if force_tcp is not None:
            os.environ["MC_FORCE_TCP"] = force_tcp
        if after_a2a:
            validate_cp_tp_group_sharing_after_a2a(args, model)
        else:
            resolve_cp_tp_group_sharing(args, model)
    return resolved_view(args)


class TestKimiK31CpTpGroupSharing(CustomTestCase):
    def test_pd_prefill_with_dflash(self):
        view = _resolve(
            _args(disaggregation_mode="prefill", speculative_algorithm="DFLASH"), _K31
        )
        self.assertTrue(view.enable_cp_tp_group_sharing)

    def test_rejects_decode_node(self):
        with self.assertRaisesRegex(ValueError, "disaggregation-mode decode"):
            _resolve(_args(disaggregation_mode="decode"), _K31)

    def test_integrated_prefill_with_dflash(self):
        view = _resolve(_args(speculative_algorithm="DFLASH"), _K31)
        self.assertTrue(view.enable_cp_tp_group_sharing)

    def test_rejects_other_speculative_algorithms(self):
        with self.assertRaisesRegex(ValueError, "only supports.*DFLASH"):
            _resolve(
                _args(disaggregation_mode="prefill", speculative_algorithm="EAGLE"),
                _K31,
            )


class TestGlmCpTpGroupSharing(CustomTestCase):
    """PD EP8 is admitted only for the ported PPU SM89/Mooncake RDMA layout."""

    def _pd_args(self, **kwargs):
        return _args(
            **(
                dict(
                    tp_size=8,
                    attn_cp_size=8,
                    ep_size=8,
                    disaggregation_mode="prefill",
                    disaggregation_transfer_backend="mooncake",
                )
                | kwargs
            )
        )

    def test_pd_ep8_prefill_and_native_mtp_resolve_sharing(self):
        for algorithm in (None, "NEXTN", "EAGLE"):
            for explicit_native_draft in (False, True):
                with self.subTest(
                    algorithm=algorithm, explicit_native_draft=explicit_native_draft
                ):
                    args = self._pd_args(
                        speculative_algorithm=algorithm,
                        speculative_draft_model_path=(
                            "dummy" if explicit_native_draft else None
                        ),
                        speculative_eagle_topk=1 if explicit_native_draft else None,
                    )
                    view = _resolve(args, _GLM)
                    self.assertTrue(view.enable_cp_tp_group_sharing)
                    # Runtime resolution must not overwrite the user's input record.
                    self.assertFalse(args.enable_cp_tp_group_sharing)

    def test_pd_uses_resolved_ep_but_does_not_admit_normalized_tcp_alias(self):
        args = self._pd_args(ep_size=1)
        declare_resolution(args, "test model-config boundary", ep_size=8)
        self.assertTrue(_resolve(args, _GLM).enable_cp_tp_group_sharing)
        self.assertEqual(args.ep_size, 1)

        args = self._pd_args(disaggregation_transfer_backend="mooncake_tcp")
        declare_resolution(
            args,
            "test transfer normalization",
            disaggregation_transfer_backend="mooncake",
        )
        with self.assertRaisesRegex(ValueError, "ep-size > 1"):
            _resolve(args, _GLM)
        self.assertFalse(resolved_view(args).enable_cp_tp_group_sharing)

    def test_pd_rechecks_ep_expansion_after_a2a(self):
        for force_tcp in (None, "0"):
            with self.subTest(force_tcp=force_tcp):
                args = self._pd_args(ep_size=1, moe_a2a_backend="deepep")
                self.assertTrue(_resolve(args, _GLM).enable_cp_tp_group_sharing)
                declare_resolution(args, "test A2A normalization", ep_size=8)
                if force_tcp is None:
                    self.assertTrue(
                        _resolve(
                            args,
                            _GLM,
                            protocol="RDMA",
                            after_a2a=True,
                        ).enable_cp_tp_group_sharing
                    )
                else:
                    with self.assertRaisesRegex(ValueError, "ep-size > 1"):
                        _resolve(
                            args,
                            _GLM,
                            force_tcp=force_tcp,
                            after_a2a=True,
                        )

    def test_pd_rejects_unported_transport_and_platform(self):
        cases = (
            ("tcp protocol", {"protocol": "tcp"}),
            ("efa protocol", {"protocol": "efa"}),
            ("forced tcp 1", {"force_tcp": "1"}),
            ("forced tcp 0", {"force_tcp": "0"}),
            ("forced tcp true", {"force_tcp": "true"}),
            ("forced tcp empty", {"force_tcp": ""}),
            ("CUDA SM89 is not PPU", {"is_ppu": False}),
            ("PPU other SM", {"device_sm": 90}),
        )
        for name, boundary in cases:
            with self.subTest(name=name):
                args = self._pd_args()
                with self.assertRaisesRegex(ValueError, "ep-size > 1"):
                    _resolve(args, _GLM, **boundary)
                self.assertFalse(resolved_view(args).enable_cp_tp_group_sharing)

    def test_pd_rejects_unported_topology_and_features(self):
        cases = (
            ({"disaggregation_mode": "decode"}, "disaggregation-mode decode"),
            (
                {"disaggregation_transfer_backend": "nixl"},
                "disaggregation-transfer-backend != mooncake",
            ),
            (
                {"disaggregation_transfer_backend": "mooncake_tcp"},
                "disaggregation-transfer-backend != mooncake",
            ),
            ({"ep_size": 4}, "ep-size > 1"),
            ({"tp_size": 4, "attn_cp_size": 4}, "ep-size > 1"),
            ({"tp_size": 4, "attn_cp_size": 4, "ep_size": 4}, "ep-size > 1"),
            ({"attn_cp_size": 4}, "attn-cp-size == --tp-size"),
            ({"nnodes": 2}, "ep-size > 1"),
            ({"pp_size": 2}, "pp-size > 1"),
            ({"dp_size": 2}, "dp-size > 1"),
            ({"cp_strategy": "zigzag"}, "cp-strategy interleave"),
            ({"enable_pd_role_switch": True}, "ep-size > 1"),
            ({"enable_unified_memory": True}, "ep-size > 1"),
            ({"dcp_size": 2}, "dcp-size > 1"),
            ({"enable_mixed_chunk": True}, "enable-mixed-chunk"),
            ({"moe_dp_size": 2}, "moe-dp-size > 1"),
            ({"enable_dsa_cache_layer_split": True}, "enable-dsa-cache-layer-split"),
        )
        for changes, error in cases:
            with self.subTest(changes=changes):
                args = self._pd_args(**changes)
                with self.assertRaisesRegex(ValueError, error):
                    _resolve(args, _GLM)
                self.assertFalse(resolved_view(args).enable_cp_tp_group_sharing)

    def test_pd_rejects_non_native_speculation(self):
        cases = (
            ({"speculative_algorithm": "DFLASH"}, "ep-size > 1"),
            ({"speculative_algorithm": "STANDALONE"}, "native MTP with top-k 1"),
            (
                {
                    "speculative_algorithm": "NEXTN",
                    "speculative_draft_model_path": "different-draft",
                },
                "native MTP with top-k 1",
            ),
            (
                {"speculative_algorithm": "EAGLE", "speculative_eagle_topk": 2},
                "native MTP with top-k 1",
            ),
        )
        for changes, error in cases:
            with self.subTest(changes=changes):
                args = self._pd_args(**changes)
                with self.assertRaisesRegex(ValueError, error):
                    _resolve(args, _GLM)
                self.assertFalse(resolved_view(args).enable_cp_tp_group_sharing)

    def test_pd_rejects_staging_and_indivisible_linear_heads(self):
        for boundary, error in (
            ({"staging": True}, "SGLANG_DISAGG_STAGING_BUFFER"),
            ({"num_heads": 62}, "num_heads=62 must be divisible"),
        ):
            with self.subTest(boundary=boundary):
                args = self._pd_args()
                with self.assertRaisesRegex(ValueError, error):
                    _resolve(args, _GLM, **boundary)
                self.assertFalse(resolved_view(args).enable_cp_tp_group_sharing)

    def test_legacy_integrated_and_pd_ep1_remain_admitted(self):
        # The new EP8 exception must not narrow the pre-existing TP-only path,
        # nor restrict integrated GLM's DFLASH support to PPU.
        for args in (
            self._pd_args(disaggregation_mode="null", speculative_algorithm="DFLASH"),
            self._pd_args(ep_size=1, tp_size=2, attn_cp_size=2),
        ):
            with self.subTest(mode=args.disaggregation_mode, ep=args.ep_size):
                self.assertTrue(
                    _resolve(args, _GLM, is_ppu=False).enable_cp_tp_group_sharing
                )

    def test_disabled_cp_single_rank_and_other_arch_do_not_enable_sharing(self):
        for args, model in (
            (self._pd_args(enable_prefill_cp=False), _GLM),
            (self._pd_args(attn_cp_size=1), _GLM),
            (
                self._pd_args(),
                SimpleNamespace(
                    hf_config=SimpleNamespace(architectures=["Glm4MoeForCausalLM"])
                ),
            ),
        ):
            with self.subTest(args=args, architecture=model.hf_config.architectures):
                self.assertFalse(_resolve(args, model).enable_cp_tp_group_sharing)


class TestQwenExpCpTpGroupSharing(CustomTestCase):
    def test_pp_cp_pd_prefill_with_native_mtp(self):
        args = _args(
            tp_size=4,
            attn_cp_size=4,
            pp_size=4,
            cp_strategy="zigzag",
            disaggregation_mode="prefill",
            moe_a2a_backend="none",
            speculative_algorithm="NEXTN",
            speculative_eagle_topk=1,
        )
        declare_resolution(
            args, "test graph resolution", cuda_graph_config=CudaGraphConfig()
        )
        view = _resolve(args, _QWEN_EXP)
        self.assertTrue(view.enable_cp_tp_group_sharing)


if __name__ == "__main__":
    unittest.main()
