"""CP-TP group sharing admission gates without devices or weights.

Two assertion surfaces coexist in this file, each with its own helper:

- ``_resolve_declare`` (mainline Qwen/Kimi surface): substitutes the
  ``declare_resolution`` / ``with_phase`` boundaries and asserts what the
  hook declares (``declare.call_args.kwargs``).
- ``_resolve_view`` (GLM EP8 surface, ported layout): resolution,
  declarations and the final runtime view stay real; only the
  hardware/environment and model-config boundaries are substituted, and
  assertions read ``resolved_view(args)``.

The helpers are named apart because their mock boundaries and return
values are incompatible; neither suite's assertions are weakened.
"""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups import cp_tp_group_sharing_hook
from sglang.srt.arg_groups.cp_tp_group_sharing_hook import (
    resolve_cp_tp_group_sharing,
    validate_cp_tp_group_sharing_final_ep,
)
from sglang.srt.arg_groups.overrides import (
    _a2a_ep_size,
    declare_resolution,
    resolved_view,
    run_post_process_pass,
)
from sglang.srt.environ import envs
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

# One registration per file. The mainline mock-only suite estimated 5s; the
# real-resolution GLM suite raised it to 10s. The merged file keeps 10: the
# extra declare-surface tests are pure mocks (sub-second), so the doubled
# estimate already covers the union. Revisit only if CI timing exceeds it.
register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_K31 = SimpleNamespace(
    hf_config=SimpleNamespace(architectures=["KimiK31ForConditionalGeneration"])
)
_QWEN_EXP = SimpleNamespace(
    hf_config=SimpleNamespace(architectures=["Qwen4ExpForConditionalGeneration"])
)
_GLM = SimpleNamespace(
    hf_config=SimpleNamespace(architectures=["Glm5NextForConditionalGeneration"])
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


def _resolve_declare(args, model):
    """Mainline declare-mock surface (Qwen/Kimi): assert what the hook declares."""
    hook = cp_tp_group_sharing_hook
    with (
        patch.object(
            hook,
            "mambaish_config",
            return_value=SimpleNamespace(linear_attn_config={"num_heads": 64}),
        ),
        patch.object(hook, "with_phase"),
        patch.object(hook, "declare_resolution") as declare,
    ):
        resolve_cp_tp_group_sharing(args, model)
    return declare


def _environment(protocol, force_tcp, staging):
    # MC_FORCE_TCP is presence-sensitive upstream: force_tcp=None means the
    # variable is unset; any string (including "0" and "") means it is set.
    env = dict(os.environ)
    env.pop("MC_FORCE_TCP", None)
    if force_tcp is not None:
        env["MC_FORCE_TCP"] = force_tcp
    return (
        envs.MOONCAKE_PROTOCOL.override(protocol),
        envs.SGLANG_DISAGG_STAGING_BUFFER.override(staging),
        patch.dict(os.environ, env, clear=True),
    )


def _resolve_view(
    args,
    model,
    *,
    is_ppu=True,
    device_sm=89,
    protocol="rdma",
    force_tcp=None,
    staging=False,
    num_heads=64,
):
    """GLM real-resolution surface: assert the final resolved view."""
    hook = cp_tp_group_sharing_hook
    protocol_ctx, staging_ctx, environ_ctx = _environment(protocol, force_tcp, staging)
    with (
        # patch.dict(clear=True) must enter first: the env overrides below
        # write into the patched environment.
        environ_ctx,
        patch.object(
            hook,
            "mambaish_config",
            return_value=SimpleNamespace(linear_attn_config={"num_heads": num_heads}),
        ),
        patch.object(
            hook,
            "get_platform",
            return_value=SimpleNamespace(is_ppu=is_ppu, device_sm=device_sm),
        ),
        protocol_ctx,
        staging_ctx,
    ):
        resolve_cp_tp_group_sharing(args, model)
    return resolved_view(args)


def _validate(
    args,
    *,
    arch="Glm5NextForConditionalGeneration",
    is_ppu=True,
    device_sm=89,
    protocol="rdma",
    force_tcp=None,
):
    hook = cp_tp_group_sharing_hook
    protocol_ctx, staging_ctx, environ_ctx = _environment(protocol, force_tcp, False)
    with (
        environ_ctx,
        patch.object(
            hook,
            "model_config_of",
            return_value=SimpleNamespace(
                hf_config=SimpleNamespace(architectures=[arch])
            ),
        ),
        patch.object(
            hook,
            "get_platform",
            return_value=SimpleNamespace(is_ppu=is_ppu, device_sm=device_sm),
        ),
        protocol_ctx,
        staging_ctx,
    ):
        validate_cp_tp_group_sharing_final_ep(args)


class TestKimiK31CpTpGroupSharing(CustomTestCase):
    """Real-resolution surface (e810837 rewrite of the Kimi gates)."""

    def test_pd_prefill_with_dflash(self):
        view = _resolve_view(
            _args(disaggregation_mode="prefill", speculative_algorithm="DFLASH"), _K31
        )
        self.assertTrue(view.enable_cp_tp_group_sharing)

    def test_rejects_decode_node(self):
        with self.assertRaisesRegex(ValueError, "disaggregation-mode decode"):
            _resolve_view(_args(disaggregation_mode="decode"), _K31)

    def test_integrated_prefill_with_dflash(self):
        view = _resolve_view(_args(speculative_algorithm="DFLASH"), _K31)
        self.assertTrue(view.enable_cp_tp_group_sharing)

    def test_rejects_other_speculative_algorithms(self):
        with self.assertRaisesRegex(ValueError, "only supports.*DFLASH"):
            _resolve_view(
                _args(disaggregation_mode="prefill", speculative_algorithm="EAGLE"),
                _K31,
            )


class TestKimiK31DeclareSurface(CustomTestCase):
    """Mainline declare-mock surface kept verbatim (release-side assertions).

    The e810837 rewrite moved the Kimi gates to the resolved-view surface;
    these four tests preserve the original declare.call_args assertions so
    no mainline assertion is dropped by the merge.
    """

    def _resolve(self, args):
        return _resolve_declare(args, _K31)

    def test_pd_prefill_with_dflash(self):
        declare = self._resolve(
            _args(disaggregation_mode="prefill", speculative_algorithm="DFLASH")
        )
        self.assertTrue(declare.call_args.kwargs["enable_cp_tp_group_sharing"])

    def test_rejects_decode_node(self):
        with self.assertRaisesRegex(ValueError, "disaggregation-mode decode"):
            self._resolve(_args(disaggregation_mode="decode"))

    def test_integrated_prefill_with_dflash(self):
        declare = self._resolve(_args(speculative_algorithm="DFLASH"))
        self.assertTrue(declare.call_args.kwargs["enable_cp_tp_group_sharing"])

    def test_rejects_other_speculative_algorithms(self):
        with self.assertRaisesRegex(ValueError, "only supports.*DFLASH"):
            self._resolve(
                _args(disaggregation_mode="prefill", speculative_algorithm="EAGLE")
            )


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
        declare = _resolve_declare(args, _QWEN_EXP)
        self.assertTrue(declare.call_args.kwargs["enable_cp_tp_group_sharing"])


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
                    view = _resolve_view(args, _GLM)
                    self.assertTrue(view.enable_cp_tp_group_sharing)
                    # Runtime resolution must not overwrite the user's input record.
                    self.assertFalse(args.enable_cp_tp_group_sharing)

    def test_pd_uses_resolved_ep_but_does_not_admit_normalized_tcp_alias(self):
        args = self._pd_args(ep_size=1)
        declare_resolution(args, "test model-config boundary", ep_size=8)
        self.assertTrue(_resolve_view(args, _GLM).enable_cp_tp_group_sharing)
        self.assertEqual(args.ep_size, 1)

        args = self._pd_args(disaggregation_transfer_backend="mooncake_tcp")
        declare_resolution(
            args,
            "test transfer normalization",
            disaggregation_transfer_backend="mooncake",
        )
        with self.assertRaisesRegex(ValueError, "ep-size > 1"):
            _resolve_view(args, _GLM)
        self.assertFalse(resolved_view(args).enable_cp_tp_group_sharing)

    def test_pd_rejects_unported_transport_and_platform(self):
        cases = (
            ("tcp protocol", {"protocol": "tcp"}),
            ("efa protocol", {"protocol": "efa"}),
            ("CUDA SM89 is not PPU", {"is_ppu": False}),
            ("PPU other SM", {"device_sm": 90}),
        )
        for name, boundary in cases:
            with self.subTest(name=name):
                args = self._pd_args()
                with self.assertRaisesRegex(ValueError, "ep-size > 1"):
                    _resolve_view(args, _GLM, **boundary)
                self.assertFalse(resolved_view(args).enable_cp_tp_group_sharing)

    def test_mc_force_tcp_presence_rejects_regardless_of_value(self):
        # Upstream forces the TCP transport whenever the variable exists,
        # even with value "0" or the empty string.
        for value in ("1", "0", ""):
            with self.subTest(value=value):
                args = self._pd_args()
                with self.assertRaisesRegex(ValueError, "ep-size > 1"):
                    _resolve_view(args, _GLM, force_tcp=value)
                self.assertFalse(resolved_view(args).enable_cp_tp_group_sharing)

    def test_mooncake_protocol_is_case_insensitive(self):
        for protocol in ("RDMA", "Rdma", "rdma"):
            with self.subTest(protocol=protocol):
                args = self._pd_args()
                self.assertTrue(
                    _resolve_view(args, _GLM, protocol=protocol).enable_cp_tp_group_sharing
                )

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
                    _resolve_view(args, _GLM)
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
                    _resolve_view(args, _GLM)
                self.assertFalse(resolved_view(args).enable_cp_tp_group_sharing)

    def test_pd_rejects_staging_and_indivisible_linear_heads(self):
        for boundary, error in (
            ({"staging": True}, "SGLANG_DISAGG_STAGING_BUFFER"),
            ({"num_heads": 62}, "num_heads=62 must be divisible"),
        ):
            with self.subTest(boundary=boundary):
                args = self._pd_args()
                with self.assertRaisesRegex(ValueError, error):
                    _resolve_view(args, _GLM, **boundary)
                self.assertFalse(resolved_view(args).enable_cp_tp_group_sharing)

    def test_legacy_integrated_and_pd_ep1_remain_admitted(self):
        # The EP8 exception must not narrow the pre-existing TP-only path,
        # nor restrict integrated GLM's DFLASH support to PPU.
        for args in (
            self._pd_args(disaggregation_mode="null", speculative_algorithm="DFLASH"),
            self._pd_args(ep_size=1, tp_size=2, attn_cp_size=2),
        ):
            with self.subTest(mode=args.disaggregation_mode, ep=args.ep_size):
                self.assertTrue(
                    _resolve_view(args, _GLM, is_ppu=False).enable_cp_tp_group_sharing
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
                self.assertFalse(_resolve_view(args, model).enable_cp_tp_group_sharing)


class TestGlmFinalEpTopology(CustomTestCase):
    """The post-A2A re-check: EP auto-expansion must not bypass admission.

    handle_context_parallelism admits EP1 trivially; handle_a2a_moe then runs
    the real _a2a_ep_size pass, which expands ep_size to tp_size for DeepEP
    and the other EP-spanning backends. The final-topology validation must
    re-run the same ported-layout gate on that resolved state.
    """

    def _pd_ep1_deepep_args(self, **kwargs):
        return _args(
            **(
                dict(
                    tp_size=8,
                    attn_cp_size=8,
                    ep_size=1,
                    moe_a2a_backend="deepep",
                    disaggregation_mode="prefill",
                    disaggregation_transfer_backend="mooncake",
                )
                | kwargs
            )
        )

    def test_ep1_deepep_autoexpansion_cannot_bypass_the_gate(self):
        cases = (
            ("forced tcp value 0", {"force_tcp": "0"}),
            ("forced tcp value 1", {"force_tcp": "1"}),
            ("forced tcp empty value", {"force_tcp": ""}),
            ("tcp protocol", {"protocol": "tcp"}),
            ("CUDA is not PPU", {"is_ppu": False}),
            ("PPU other SM", {"device_sm": 90}),
        )
        for name, boundary in cases:
            with self.subTest(name=name):
                args = self._pd_ep1_deepep_args()
                # Admission sees EP1 and admits without needing the exception.
                view = _resolve_view(args, _GLM, **boundary)
                self.assertTrue(view.enable_cp_tp_group_sharing)
                # The real A2A EP normalization expands EP1 -> tp_size.
                run_post_process_pass(args, _a2a_ep_size)
                self.assertEqual(resolved_view(args).ep_size, 8)
                with self.assertRaisesRegex(ValueError, "ep-size > 1"):
                    _validate(args, **boundary)
                # The raw input record is never rewritten.
                self.assertEqual(args.ep_size, 1)

    def test_ep1_deepep_autoexpansion_admitted_when_final_gate_passes(self):
        args = self._pd_ep1_deepep_args()
        self.assertTrue(_resolve_view(args, _GLM).enable_cp_tp_group_sharing)
        run_post_process_pass(args, _a2a_ep_size)
        self.assertEqual(resolved_view(args).ep_size, 8)
        _validate(args)
        self.assertTrue(resolved_view(args).enable_cp_tp_group_sharing)
        self.assertEqual(args.ep_size, 1)

    def test_normalized_tcp_alias_rejected_after_autoexpansion(self):
        args = self._pd_ep1_deepep_args(disaggregation_transfer_backend="mooncake_tcp")
        declare_resolution(
            args,
            "test transfer normalization",
            disaggregation_transfer_backend="mooncake",
        )
        self.assertTrue(_resolve_view(args, _GLM).enable_cp_tp_group_sharing)
        run_post_process_pass(args, _a2a_ep_size)
        with self.assertRaisesRegex(ValueError, "ep-size > 1"):
            _validate(args)

    def test_autoexpanded_non_ported_topology_rejected(self):
        # tp_size=4 expands EP1 -> 4, which is not the ported EP8 layout.
        args = self._pd_ep1_deepep_args(tp_size=4, attn_cp_size=4)
        self.assertTrue(_resolve_view(args, _GLM).enable_cp_tp_group_sharing)
        run_post_process_pass(args, _a2a_ep_size)
        self.assertEqual(resolved_view(args).ep_size, 4)
        with self.assertRaisesRegex(ValueError, "ep-size > 1"):
            _validate(args)

    def test_validation_is_a_noop_outside_glm_pd_ep(self):
        # Sharing disabled: nothing to re-check.
        args = self._pd_ep1_deepep_args()
        _validate(args)

        # EP stays 1: nothing to re-check.
        args = self._pd_ep1_deepep_args()
        _resolve_view(args, _GLM)
        _validate(args, is_ppu=False)

        # Integrated GLM keeps its existing EP freedom.
        args = self._pd_ep1_deepep_args(disaggregation_mode="null", ep_size=8)
        _resolve_view(args, _GLM, is_ppu=False)
        _validate(args, is_ppu=False)

        # Kimi PD prefill EP behavior is unchanged by this gate.
        args = _args(
            tp_size=8,
            attn_cp_size=8,
            ep_size=8,
            disaggregation_mode="prefill",
            speculative_algorithm="DFLASH",
        )
        _resolve_view(args, _K31, is_ppu=False)
        _validate(args, arch="KimiK31ForConditionalGeneration", is_ppu=False)


if __name__ == "__main__":
    unittest.main()