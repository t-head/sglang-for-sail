"""SAIL draft FA3 must accept runners exposing only the current KV interface.

The common backend moved to runner_ids, but the SAIL constructor still read
kv_index_translator and failed before creating its first draft step. Exercise
that constructor through the real draft factory, without GPU allocations.
"""

import importlib
import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _StepBackend:
    """Stand-in for a per-step backend's GPU resource allocation boundary."""

    def __init__(self, runner, **kwargs):
        self.runner = runner
        self.step_id = kwargs["speculative_step_id"]
        self.options = kwargs
        self.events = []

    def init_forward_metadata(self, batch):
        self.events.append(("eager", batch))

    def init_cuda_graph_state(self, max_bs, max_num_tokens):
        self.events.append(("graph_state", max_bs, max_num_tokens))

    def init_forward_metadata_in_graph(self, batch):
        self.events.append(("graph_metadata", batch))


class TestSailFlashAttentionMultiStep(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        maybe_stub_sgl_kernel()
        # Only the external FA3 library is stubbed. Both the SAIL constructor
        # and its common parent/factory are imported from production code.
        external = types.ModuleType("flash_attn_interface")
        external.flash_attn_varlen_func = mock.Mock()
        external.flash_attn_with_kvcache = mock.Mock()
        external.get_scheduler_metadata = mock.Mock()
        cls._modules_before = set(sys.modules)
        patcher = mock.patch.dict(sys.modules, {"flash_attn_interface": external})
        patcher.start()
        cls.addClassCleanup(patcher.stop)
        cls.backend_module = importlib.import_module(
            "sglang.srt.hardware_backend.sail.attention.flashattention_backend"
        )
        cls.draft_module = importlib.import_module("sglang.srt.speculative.draft_utils")

        def remove_test_imports():
            for name in (
                "sglang.srt.hardware_backend.sail.attention.flashattention_backend",
                "sglang.srt.hardware_backend.sail.attention.flash_attention",
            ):
                if name not in cls._modules_before:
                    sys.modules.pop(name, None)

        cls.addClassCleanup(remove_test_imports)

    def test_one_step_does_not_require_the_removed_runner_field(self):
        runner = SimpleNamespace(runner_ids=object())
        backend = self.backend_module.SailFlashAttentionMultiStepBackend(
            runner, topk=1, speculative_num_steps=1
        )
        self.assertIs(backend.runner_ids, runner.runner_ids)
        self.assertIs(backend.attn_backend_list, backend.attn_backends)
        self.assertEqual(backend.attn_backends, [])

    def test_factory_builds_sail_steps_and_preserves_metadata_dispatch(self):
        for topk, steps in ((1, 2), (2, 4)):
            with self.subTest(topk=topk, steps=steps):
                runner = SimpleNamespace(
                    runner_ids=object(), draft_attention_backend="fa3"
                )
                factory = self.draft_module.DraftBackendFactory(runner, topk, steps)
                with (
                    mock.patch.object(self.draft_module, "is_musa", return_value=False),
                    mock.patch.object(self.draft_module, "is_sail", return_value=True),
                    mock.patch.object(
                        self.backend_module, "SailFlashAttentionBackend", _StepBackend
                    ),
                ):
                    name, backend = factory._create_fa_decode_backend(fa_impl_ver=3)
                self.assertEqual(name, "fa3")
                self.assertIsInstance(
                    backend, self.backend_module.SailFlashAttentionMultiStepBackend
                )
                self.assertIs(backend.runner_ids, runner.runner_ids)
                self.assertIs(backend.attn_backend_list, backend.attn_backends)
                self.assertEqual(
                    [step.step_id for step in backend.attn_backend_list],
                    list(range(steps - 1)),
                )
                batch = object()
                backend.init_forward_metadata(batch)
                backend.init_cuda_graph_state(8, 32)
                backend.init_forward_metadata_in_graph(batch)
                for step in backend.attn_backend_list:
                    self.assertIs(step.runner, runner)
                    self.assertEqual(step.options["fa_impl_ver"], 3)
                    self.assertEqual(step.options["topk"], topk)
                    self.assertEqual(step.options["speculative_num_steps"], steps)
                    self.assertEqual(
                        step.events,
                        [
                            ("eager", batch),
                            ("graph_state", 8, 32),
                            ("graph_metadata", batch),
                        ],
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
