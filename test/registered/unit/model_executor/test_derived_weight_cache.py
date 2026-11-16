"""Online updates must reject model-owned derived buffers before writing."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.managers.io_struct import BeginWeightUpdateReqInput
from sglang.srt.managers.scheduler_components import weight_updater as scheduler_updates
from sglang.srt.model_executor.model_runner_components.weight_updater import (
    WeightUpdater,
    _unsupported_derived_weight_cache_error,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=9, suite="base-a-test-cpu")


def _updater(model):
    return WeightUpdater(
        tp_rank=0,
        device="cpu",
        gpu_id=0,
        model_config=None,
        custom_weight_loaders={},
        get_model=lambda: model,
        update_model_fields=lambda *args, **kwargs: None,
        recapture_cuda_graph=lambda: None,
        get_model_runner=lambda: None,
    )


class TestDerivedWeightCache(CustomTestCase):
    def test_nested_model_cache_rejects_updates(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(2, 2, bias=False, device="cpu"),
            torch.nn.Sequential(torch.nn.Module()),
        )
        model[1][0]._derived_weight_cache_error = "derived scales require restart"
        original = model[0].weight.detach().clone()
        updater = _updater(model)
        with patch(
            "sglang.srt.model_executor.model_runner_components.weight_updater.get_model",
            return_value=SimpleNamespace(weight_cache_mode="off"),
        ):
            self.assertEqual(
                updater.update_weights_from_tensor(
                    [("0.weight", torch.zeros_like(original))], load_format="direct"
                ),
                (False, "derived scales require restart"),
            )
        self.assertTrue(torch.equal(model[0].weight, original))

    def test_model_without_derived_cache_keeps_updates_enabled(self):
        model = torch.nn.Module()
        with patch(
            "sglang.kernels.ops.gemm.bf16_fp32.hpc_bf16xfp32_gemm_enabled",
            return_value=False,
        ):
            self.assertIsNone(_unsupported_derived_weight_cache_error(model))

    def test_session_checks_all_selected_runners_before_restore(self):
        for selector, blocked_role, restored in (
            ("target", "target", []),
            ("all", "target", []),
            ("all", "draft", []),
            ("draft", "target", ["draft"]),
            ("all", None, ["target", "draft"]),
        ):
            with self.subTest(selector=selector, blocked_role=blocked_role):
                runners = {}
                message = "cached weights require a restart"
                for role in ("target", "draft"):
                    model = torch.nn.Module()
                    if role == blocked_role:
                        model._derived_weight_cache_error = message
                    runners[role] = SimpleNamespace(weight_updater=_updater(model))
                cpu_group = object()
                with (
                    patch.object(
                        scheduler_updates,
                        "get_parallel",
                        return_value=SimpleNamespace(
                            tp_group=SimpleNamespace(cpu_group=cpu_group)
                        ),
                    ),
                    patch.object(
                        WeightUpdater, "begin_weight_update", autospec=True
                    ) as restore,
                    patch.object(torch.distributed, "barrier") as barrier,
                ):
                    manager = scheduler_updates.SchedulerWeightUpdaterManager(
                        tp_worker=SimpleNamespace(
                            weight_update_runners=lambda: [
                                ("target", runners["target"])
                            ]
                        ),
                        draft_worker=SimpleNamespace(
                            weight_update_runners=lambda: [("draft", runners["draft"])]
                        ),
                        memory_saver_adapter=None,
                        flush_cache=lambda: True,
                        is_fully_idle=lambda: True,
                    )
                    output = manager.begin_weight_update(
                        BeginWeightUpdateReqInput(selector=selector)
                    )
                    self.assertEqual(output.success, bool(restored))
                    if restored:
                        self.assertEqual(manager._session.selector, selector)
                        self.assertEqual(
                            [call.args[0] for call in restore.call_args_list],
                            [runners[role].weight_updater for role in restored],
                        )
                        barrier.assert_called_once_with(group=cpu_group)
                    else:
                        self.assertEqual(output.message, message)
                        self.assertIsNone(manager._session)
                        restore.assert_not_called()
                        barrier.assert_not_called()


if __name__ == "__main__":
    unittest.main()
