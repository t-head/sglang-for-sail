"""CP-TP group sharing gates for integrated and PD prefill."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups import cp_tp_group_sharing_hook
from sglang.srt.arg_groups.cp_tp_group_sharing_hook import resolve_cp_tp_group_sharing
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


def _resolve(args, model):
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


class TestKimiK31CpTpGroupSharing(CustomTestCase):
    def _resolve(self, args):
        return _resolve(args, _K31)

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
        declare = _resolve(args, _QWEN_EXP)
        self.assertTrue(declare.call_args.kwargs["enable_cp_tp_group_sharing"])


if __name__ == "__main__":
    unittest.main()
