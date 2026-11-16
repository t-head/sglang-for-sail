"""Glm5Next PD + CP-TP group sharing speculative-algorithm validation gates.

Covers both touchpoints of the fix for the prefill-node startup failure where
``--speculative-algorithm`` was not given (so it resolves to ``None``) and the
PD branch rejected it:

* the pre-disaggregation predicate in ``cp_tp_group_sharing_hook``
  (``_pd_spec_conflicts_with_cp_tp_sharing``), which now accepts ``None``,
* the post-resolution guard at the end of
  ``speculative_hook.handle_speculative_decoding``, which re-applies the
  restriction to the resolved value so an alias cannot bypass the first one.

Rejected spellings are asserted by message, because the two touchpoints (and
the untouched architecture whitelist above them) raise distinct texts; this
pins which gate is responsible for each case.
"""

import json
import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups import cp_tp_group_sharing_hook
from sglang.srt.arg_groups.cp_tp_group_sharing_hook import (
    _pd_spec_conflicts_with_cp_tp_sharing,
    resolve_cp_tp_group_sharing,
)
from sglang.srt.arg_groups.overrides import declare_resolution, resolution_result
from sglang.srt.arg_groups.speculative_hook import handle_speculative_decoding
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

GLM5NEXT = "Glm5NextForConditionalGeneration"
QWEN4EXP = "Qwen4ExpForConditionalGeneration"
KIMI_K31 = "KimiK31ForConditionalGeneration"

# The rejection the defect produced: a legal PD configuration was reported by
# the PD branch of the CP-TP group sharing hook.
PD_SPEC_REJECTED = (
    r"--speculative-algorithm != DFLASH is not supported with CP-TP group sharing\."
)
# Spelling/topology gate above the PD branch, unchanged by this fix.
NATIVE_SPEC_REJECTED = (
    r"CP/TP group sharing only supports DFLASH or native MTP with top-k 1\."
)
# Architecture gate above the PD branch, unchanged by this fix.
ARCH_SPEC_REJECTED = (
    r"--speculative-algorithm is not supported with CP-TP group sharing\."
)
# Prefill-CP-only gate, unchanged by this fix; the D node never enters the hook.
DECODE_NODE_REJECTED = (
    r"--disaggregation-mode decode is not supported with CP-TP group sharing\."
)


def _model(arch: str) -> SimpleNamespace:
    return SimpleNamespace(hf_config=SimpleNamespace(architectures=[arch]))


# A draft config a DFLASH handler can read without weights or network.
_MINI_DRAFT_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_attention_heads": 8,
    "num_key_value_heads": 8,
    "num_hidden_layers": 2,
    "vocab_size": 128,
    "max_position_embeddings": 2048,
}


class _DraftConfigCase(CustomTestCase):
    """Gives a case a local directory holding a minimal draft ``config.json``."""

    def setUp(self):
        super().setUp()
        self.draft_path = tempfile.mkdtemp(prefix="glm5next_pd_spec_draft_")
        self.addCleanup(shutil.rmtree, self.draft_path, ignore_errors=True)
        with open(os.path.join(self.draft_path, "config.json"), "w") as handle:
            json.dump(_MINI_DRAFT_CONFIG, handle)


def _args(arch=GLM5NEXT, **kwargs) -> ServerArgs:
    # Qwen4Exp is a zigzag model; the hook requires the strategy it derives.
    strategy = "zigzag" if arch == QWEN4EXP else "interleave"
    base = dict(
        model_path="dummy",
        tp_size=8,
        attn_cp_size=8,
        enable_prefill_cp=True,
        cp_strategy=strategy,
        disaggregation_mode="prefill",
        disaggregation_transfer_backend="mooncake",
    )
    return ServerArgs(**(base | kwargs))


class _CpTpSharingHarness(CustomTestCase):
    """Drives the real hooks; only the model description is faked.

    ``_resolve`` captures the declaration call so a case can read what the hook
    asked the record to hold; ``_first_hook_declaring_sharing`` below lets the
    declaration through for real instead.
    """

    def _resolve(self, args, arch):
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
            resolve_cp_tp_group_sharing(args, _model(arch))
        return declare


def _first_hook_declaring_sharing(args, arch):
    """Run the first hook with real declarations, then hand back the record with
    its model description faked the way ``model_config_of`` accepts it."""
    hook = cp_tp_group_sharing_hook
    with (
        patch.object(
            hook,
            "mambaish_config",
            return_value=SimpleNamespace(linear_attn_config={"num_heads": 64}),
        ),
        patch.object(hook, "with_phase"),
    ):
        resolve_cp_tp_group_sharing(args, _model(arch))
    args._model_config = _model(arch)
    args._model_config_built_from = None
    return args


class TestGlm5NextPdPrefillHook(_CpTpSharingHarness):
    """``resolve_cp_tp_group_sharing`` on a Glm5Next PD prefill node."""

    def _arch(self, **kwargs):
        return self._resolve(_args(**kwargs), GLM5NEXT)

    def test_pd_prefill_without_speculation_is_accepted(self):
        # The defect: the flag is absent, so the field is None, and PD used to
        # compare it against "DFLASH" and reject.
        declare = self._arch(speculative_algorithm=None)
        self.assertTrue(declare.call_args.kwargs["enable_cp_tp_group_sharing"])

    def test_pd_prefill_without_the_flag_at_all_is_accepted(self):
        declare = self._arch()
        self.assertTrue(declare.call_args.kwargs["enable_cp_tp_group_sharing"])

    def test_pd_prefill_with_dflash_is_accepted(self):
        declare = self._arch(speculative_algorithm="DFLASH")
        self.assertTrue(declare.call_args.kwargs["enable_cp_tp_group_sharing"])

    def test_pd_prefill_rejects_eagle(self):
        with self.assertRaisesRegex(ValueError, PD_SPEC_REJECTED):
            self._arch(speculative_algorithm="EAGLE")

    def test_pd_prefill_rejects_eagle3_before_the_pd_branch(self):
        # EAGLE3 is not DFLASH and not the native-MTP shape, so the whitelist
        # gate above answers first; either way it stays rejected.
        with self.assertRaisesRegex(ValueError, NATIVE_SPEC_REJECTED):
            self._arch(speculative_algorithm="EAGLE3")

    def test_pd_prefill_rejects_nextn(self):
        with self.assertRaisesRegex(ValueError, PD_SPEC_REJECTED):
            self._arch(speculative_algorithm="NEXTN")

    def test_pd_prefill_rejects_native_mtp_nextn_with_matching_draft(self):
        # NEXTN with draft == target and top-k 1 passes the whitelist gate, so
        # the rejection must come from the PD branch, which stays rejected.
        with self.assertRaisesRegex(ValueError, PD_SPEC_REJECTED):
            self._arch(
                speculative_algorithm="NEXTN",
                speculative_draft_model_path="dummy",
                speculative_eagle_topk=1,
            )

    def test_pd_prefill_rejects_lowercase_dflash_before_the_pd_branch(self):
        # Uppercasing is deliberately not generalized here: the whitelist gate
        # above keeps rejecting it, which is the pre-fix behavior.
        with self.assertRaisesRegex(ValueError, NATIVE_SPEC_REJECTED):
            self._arch(speculative_algorithm="dflash")

    def test_pd_prefill_rejects_mixed_case_dflash_before_the_pd_branch(self):
        with self.assertRaisesRegex(ValueError, NATIVE_SPEC_REJECTED):
            self._arch(speculative_algorithm="DFlash")

    def test_pd_prefill_rejects_padded_dflash_before_the_pd_branch(self):
        with self.assertRaisesRegex(ValueError, NATIVE_SPEC_REJECTED):
            self._arch(speculative_algorithm="DFLASH ")

    def test_pd_prefill_rejects_dspark_before_the_pd_branch(self):
        with self.assertRaisesRegex(ValueError, NATIVE_SPEC_REJECTED):
            self._arch(speculative_algorithm="DSPARK")

    def test_pd_prefill_rejects_unknown_algorithm(self):
        with self.assertRaisesRegex(ValueError, NATIVE_SPEC_REJECTED):
            self._arch(speculative_algorithm="NOT_A_THING")

    def test_pd_decode_node_still_rejects_a_decode_node(self):
        # The hook only runs for prefill CP; a node that asks for both stays
        # rejected by the pre-existing entry, which this fix does not touch.
        with self.assertRaisesRegex(ValueError, DECODE_NODE_REJECTED):
            self._arch(disaggregation_mode="decode", speculative_algorithm=None)

    def test_glm5next_without_prefill_cp_is_not_in_the_hook_domain(self):
        args = ServerArgs(
            model_path="dummy",
            tp_size=8,
            disaggregation_mode="decode",
            disaggregation_transfer_backend="mooncake",
            speculative_algorithm=None,
        )
        self.assertIsNone(self._resolve(args, GLM5NEXT).call_args)

    def test_glm5next_integrated_prefill_without_speculation_is_accepted(self):
        declare = self._arch(disaggregation_mode="null", speculative_algorithm=None)
        self.assertTrue(declare.call_args.kwargs["enable_cp_tp_group_sharing"])

    def test_glm5next_integrated_native_mtp_stays_accepted(self):
        # The non-PD path is where NEXTN is legal; the fix leaves it alone.
        declare = self._arch(
            disaggregation_mode="null",
            speculative_algorithm="NEXTN",
            speculative_draft_model_path="dummy",
            speculative_eagle_topk=1,
        )
        self.assertTrue(declare.call_args.kwargs["enable_cp_tp_group_sharing"])

    def test_glm5next_integrated_eagle_stays_accepted(self):
        declare = self._arch(disaggregation_mode="null", speculative_algorithm="EAGLE")
        self.assertTrue(declare.call_args.kwargs["enable_cp_tp_group_sharing"])

    def test_glm5next_integrated_rejects_lowercase_dflash(self):
        with self.assertRaisesRegex(ValueError, NATIVE_SPEC_REJECTED):
            self._arch(disaggregation_mode="null", speculative_algorithm="dflash")


class TestOtherArchitecturesUnaffectedByThePdBranch(_CpTpSharingHarness):
    """Qwen4Exp/KimiK31 keep their own acceptance sets."""

    def test_qwen4exp_pd_prefill_without_speculation_is_accepted(self):
        declare = self._resolve(
            _args(arch=QWEN4EXP, speculative_algorithm=None), QWEN4EXP
        )
        self.assertTrue(declare.call_args.kwargs["enable_cp_tp_group_sharing"])

    def test_qwen4exp_pd_prefill_rejects_any_algorithm(self):
        for algorithm in ("EAGLE", "NEXTN", "DFLASH", "dflash"):
            with self.subTest(algorithm=algorithm):
                with self.assertRaisesRegex(ValueError, ARCH_SPEC_REJECTED):
                    self._resolve(
                        _args(arch=QWEN4EXP, speculative_algorithm=algorithm), QWEN4EXP
                    )

    def test_kimi_k31_pd_prefill_without_speculation_is_accepted(self):
        declare = self._resolve(_args(speculative_algorithm=None), KIMI_K31)
        self.assertTrue(declare.call_args.kwargs["enable_cp_tp_group_sharing"])

    def test_kimi_k31_pd_prefill_accepts_both_dflash_spellings(self):
        for algorithm in ("DFLASH", "dflash"):
            with self.subTest(algorithm=algorithm):
                declare = self._resolve(
                    _args(speculative_algorithm=algorithm), KIMI_K31
                )
                self.assertTrue(declare.call_args.kwargs["enable_cp_tp_group_sharing"])

    def test_kimi_k31_pd_prefill_rejects_other_algorithms(self):
        for algorithm in ("EAGLE", "NEXTN", "DSPARK"):
            with self.subTest(algorithm=algorithm):
                with self.assertRaisesRegex(
                    ValueError, r"Kimi-K3.1 CP-TP group sharing"
                ):
                    self._resolve(_args(speculative_algorithm=algorithm), KIMI_K31)

    def test_kimi_k31_rejects_a_decode_node(self):
        with self.assertRaisesRegex(ValueError, DECODE_NODE_REJECTED):
            self._resolve(
                _args(disaggregation_mode="decode", speculative_algorithm=None),
                KIMI_K31,
            )


class TestPdSpecPredicate(CustomTestCase):
    """``_pd_spec_conflicts_with_cp_tp_sharing`` in isolation."""

    @staticmethod
    def _cfg(algorithm):
        return SimpleNamespace(speculative_algorithm=algorithm)

    def test_glm5next_legal_set_is_none_dflash_and_lowercase_dflash(self):
        for algorithm in (None, "DFLASH", "dflash"):
            with self.subTest(algorithm=algorithm):
                self.assertFalse(
                    _pd_spec_conflicts_with_cp_tp_sharing(
                        self._cfg(algorithm), GLM5NEXT
                    )
                )

    def test_glm5next_rejects_everything_else(self):
        for algorithm in (
            "DFlash",
            "dFlash",
            "DFLASH ",
            "dflash ",
            "NEXTN",
            "nextn",
            "EAGLE",
            "EAGLE3",
            "DSPARK",
            "FROZEN_KV_MTP",
            "",
            "none",
        ):
            with self.subTest(algorithm=algorithm):
                self.assertTrue(
                    _pd_spec_conflicts_with_cp_tp_sharing(
                        self._cfg(algorithm), GLM5NEXT
                    )
                )

    def test_other_architectures_never_conflict_here(self):
        for arch in (QWEN4EXP, KIMI_K31, "DeepseekV32ForCausalLM"):
            for algorithm in (None, "DFLASH", "EAGLE", "DSPARK", "dflash"):
                with self.subTest(arch=arch, algorithm=algorithm):
                    self.assertFalse(
                        _pd_spec_conflicts_with_cp_tp_sharing(
                            self._cfg(algorithm), arch
                        )
                    )


class TestGlm5NextPdPostResolutionGuard(_DraftConfigCase):
    """The guard at the end of ``handle_speculative_decoding``.

    The hook above runs before ``upper()`` and the NEXTN -> EAGLE alias, so a
    value that only becomes final inside this function must still be rejected
    here. Only the model description is faked; ``model_config_of`` hands back a
    fixture it was given (no build key).

    The derived sharing flag is seeded the way production seeds it -- through
    ``declare_resolution`` -- leaving the raw field false. A guard that read the
    raw field instead of the resolved state would accept every case below.
    """

    def _args(self, arch, disaggregation_mode, algorithm, sharing=True, **extra):
        args = ServerArgs(
            model_path="dummy",
            tp_size=8,
            attn_cp_size=8,
            enable_prefill_cp=True,
            cp_strategy="interleave",
            disaggregation_mode=disaggregation_mode,
            disaggregation_transfer_backend="mooncake",
            speculative_algorithm=algorithm,
            **extra,
        )
        if sharing:
            declare_resolution(
                args,
                "resolve_cp_tp_group_sharing",
                enable_cp_tp_group_sharing=True,
            )
        args._model_config = _model(arch)
        args._model_config_built_from = None
        return args

    def _run(self, arch, disaggregation_mode, algorithm, **kwargs):
        handle_speculative_decoding(
            self._args(arch, disaggregation_mode, algorithm, **kwargs)
        )

    def test_rejects_eagle_on_a_prefill_node(self):
        with self.assertRaisesRegex(ValueError, PD_SPEC_REJECTED):
            self._run(GLM5NEXT, "prefill", "EAGLE")

    def test_rejects_eagle3_on_a_prefill_node(self):
        with self.assertRaisesRegex(ValueError, PD_SPEC_REJECTED):
            self._run(GLM5NEXT, "prefill", "EAGLE3")

    def test_rejects_nextn_aliased_to_eagle_on_a_prefill_node(self):
        # The alias is declared inside this function, before the guard runs, so
        # the final value the guard sees is EAGLE rather than the spelling given.
        with self.assertRaisesRegex(ValueError, PD_SPEC_REJECTED):
            self._run(GLM5NEXT, "prefill", "NEXTN")

    def test_rejects_lowercase_nextn_aliased_to_eagle(self):
        with self.assertRaisesRegex(ValueError, PD_SPEC_REJECTED):
            self._run(GLM5NEXT, "prefill", "nextn")

    def test_rejects_eagle_on_a_decode_node(self):
        with self.assertRaisesRegex(ValueError, PD_SPEC_REJECTED):
            self._run(GLM5NEXT, "decode", "EAGLE")

    def test_rejects_dspark_reaching_the_guard_after_its_handler(self):
        # Unlike the spellings above, DSPARK survives every per-algorithm handler
        # and is stopped only by this guard, so a guard narrowed to a few names
        # would let it through.
        with self.assertRaisesRegex(ValueError, PD_SPEC_REJECTED):
            self._run(
                GLM5NEXT,
                "prefill",
                "DSPARK",
                device="cuda",
                speculative_draft_model_path=self.draft_path,
                speculative_num_draft_tokens=4,
            )

    def test_accepts_dflash_reaching_the_guard_after_its_handler(self):
        # The other half of that pair: DFLASH must survive its own handler and
        # the guard. Without this case a guard that rejected every non-None
        # value would still pass the suite, breaking a supported PD deployment.
        args = self._args(
            GLM5NEXT,
            "prefill",
            "DFLASH",
            device="cuda",
            speculative_draft_model_path=self.draft_path,
            speculative_dflash_block_size=4,
            speculative_draft_attention_backend="flashinfer",
        )
        handle_speculative_decoding(args)
        self.assertEqual(resolution_result(args, "speculative_algorithm"), "DFLASH")
        self.assertTrue(resolution_result(args, "enable_cp_tp_group_sharing"))

    def test_accepts_no_speculation_on_a_prefill_node(self):
        self._run(GLM5NEXT, "prefill", None)

    def test_accepts_no_speculation_on_a_decode_node(self):
        self._run(GLM5NEXT, "decode", None)

    def test_stays_out_of_glm5next_without_disaggregation(self):
        self._run(GLM5NEXT, "null", "EAGLE")

    def test_stays_out_of_glm5next_aliased_without_disaggregation(self):
        self._run(GLM5NEXT, "null", "nextn")

    def test_stays_out_of_glm5next_when_sharing_was_not_derived(self):
        self._run(GLM5NEXT, "prefill", "EAGLE", sharing=False)

    def test_stays_out_of_qwen4exp(self):
        self._run(QWEN4EXP, "prefill", "EAGLE")

    def test_stays_out_of_qwen4exp_aliased(self):
        self._run(QWEN4EXP, "prefill", "nextn")

    def test_stays_out_of_kimi_k31(self):
        self._run(KIMI_K31, "prefill", "EAGLE")


class TestBothHooksOnOneRecord(_DraftConfigCase):
    """The two touchpoints as one effective state, with real declarations.

    Nothing here mocks ``declare_resolution``: the sharing flag is whatever the
    first hook actually declared, and the assertions read the settled record
    through ``resolution_result`` rather than a call spy.
    """

    def _resolve_then_run(self, **kwargs):
        args = _first_hook_declaring_sharing(_args(**kwargs), GLM5NEXT)
        handle_speculative_decoding(args)
        return args

    def test_no_speculation_declares_sharing_and_keeps_algorithm_none(self):
        # The configuration the failing batch used: no --speculative-algorithm.
        args = self._resolve_then_run(speculative_algorithm=None)
        self.assertTrue(resolution_result(args, "enable_cp_tp_group_sharing"))
        self.assertIsNone(resolution_result(args, "speculative_algorithm"))
        # The raw field is what a stale reader would see; production reads the
        # declaration, so the record carries both until it is materialized.
        self.assertFalse(args.enable_cp_tp_group_sharing)

    def test_dflash_declares_sharing_and_keeps_the_algorithm(self):
        args = self._resolve_then_run(
            speculative_algorithm="DFLASH",
            device="cuda",
            speculative_draft_model_path=self.draft_path,
            speculative_dflash_block_size=4,
            speculative_draft_attention_backend="flashinfer",
        )
        self.assertTrue(resolution_result(args, "enable_cp_tp_group_sharing"))
        self.assertEqual(resolution_result(args, "speculative_algorithm"), "DFLASH")

    def test_eagle_never_reaches_the_second_hook_on_a_pd_prefill_node(self):
        # Gate precedence, recorded rather than assumed: the first hook refuses
        # this combination, so the guard is not what answers here.
        #
        # The record is built outside the raise and asserted below as the very
        # object the rejected hook was handed. Asserting on a second, freshly
        # built record would keep passing even if the hook declared the sharing
        # flag before raising, and that is exactly the failure mode this case
        # exists to detect.
        args = _args(speculative_algorithm="EAGLE")
        with self.assertRaisesRegex(ValueError, PD_SPEC_REJECTED):
            _first_hook_declaring_sharing(args, GLM5NEXT)
        # The rejected run declared nothing, so both the raw field and the
        # settled value stay false on the record it was handed: the sharing flag
        # only ever becomes true as a declaration (compare the accepted cases
        # above, where the settled value is true while the raw field stays
        # false).
        self.assertFalse(args.enable_cp_tp_group_sharing)
        self.assertIs(resolution_result(args, "enable_cp_tp_group_sharing"), False)


if __name__ == "__main__":
    unittest.main()
