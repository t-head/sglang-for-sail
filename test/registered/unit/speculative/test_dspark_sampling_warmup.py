"""CPU control-flow tests; no PPU kernel or complete worker startup claim."""

import sys
import unittest
from contextlib import contextmanager, nullcontext
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.speculative.dspark_components.dspark_sampling_warmup import (
    prewarm_dspark_sampling,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

# Real worker implementation, with the project's CPU import stub scoped to
# this import only. Do not leave new modules/stubs in the global registry.
_import_meta_path = sys.meta_path
_import_finders = tuple(sys.meta_path)
with patch.object(sys, "meta_path", list(sys.meta_path)), patch.dict(sys.modules):
    maybe_stub_sgl_kernel()
    import sglang.srt.speculative.dspark_components.dspark_worker_v2 as worker_module
    from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
    from sglang.srt.speculative.dspark_components.dspark_worker_v2 import DSparkWorkerV2
if sys.meta_path is not _import_meta_path or tuple(sys.meta_path) != _import_finders:
    raise AssertionError("CPU worker import leaked an import finder")

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

HELPER = "sglang.srt.speculative.dspark_components.dspark_sampling_warmup.prewarm_dspark_sampling"


class TestDSparkSamplingWorkerWarmup(CustomTestCase):
    def worker(self):
        worker = object.__new__(DSparkWorkerV2)
        worker._hosts_draft = True
        worker._is_pd_prefill = False
        worker._simulate_acc_len = -1.0
        worker.gpu_id = 2
        worker.gamma = 5
        worker.verify_num_draft_tokens = 6
        worker._target_worker = SimpleNamespace(
            device="cuda",
            model_runner=SimpleNamespace(
                model_config=SimpleNamespace(
                    vocab_size=257,
                    hf_text_config=SimpleNamespace(model_type="deepseek_v41"),
                )
            ),
        )
        return worker

    def test_nonlast_pp_preserves_super_without_reading_draft_fields(self):
        worker = object.__new__(DSparkWorkerV2)
        worker._hosts_draft = False
        result = object()
        with patch.object(
            BaseSpecWorker, "prewarm_sampling", return_value=result
        ) as ordinary:
            self.assertIs(worker.prewarm_sampling(), result)
        ordinary.assert_called_once_with()

    def test_scope_skips_before_gamma_is_read(self):
        cases = ("cpu", "hip", "not_sail", "other_model", "prefill", "simulate")
        for reason in cases:
            with self.subTest(reason=reason):
                worker = self.worker()
                del worker.gamma
                if reason == "cpu":
                    worker._target_worker.device = "cpu"
                if reason == "other_model":
                    worker._target_worker.model_runner.model_config.hf_text_config.model_type = "other"
                worker._is_pd_prefill = reason == "prefill"
                worker._simulate_acc_len = 3.0 if reason == "simulate" else -1.0
                result = object()
                with (
                    patch.object(
                        BaseSpecWorker, "prewarm_sampling", return_value=result
                    ),
                    patch.object(
                        worker_module, "is_sail", return_value=reason != "not_sail"
                    ),
                    patch.object(
                        torch.version, "hip", "test" if reason == "hip" else None
                    ),
                    patch(HELPER) as prepare,
                ):
                    self.assertIs(worker.prewarm_sampling(), result)
                    prepare.assert_not_called()

    def test_super_first_actual_dimensions_stream_and_same_result(self):
        worker, result, stream = self.worker(), object(), object()
        order = []
        with (
            patch.object(
                BaseSpecWorker,
                "prewarm_sampling",
                side_effect=lambda: order.append("super") or result,
            ),
            patch.object(worker_module, "is_sail", return_value=True),
            patch.object(torch.version, "hip", None),
            patch.object(
                worker_module,
                "get_spec",
                return_value=SimpleNamespace(speculative_use_block_verification=True),
            ),
            patch.object(
                torch.cuda, "current_device", return_value=2
            ) as resolve_device,
            patch.object(torch.cuda, "current_stream", return_value=stream) as current,
            patch(HELPER, side_effect=lambda **kw: order.append("private")) as prepare,
        ):
            self.assertIs(worker.prewarm_sampling(), result)
        self.assertEqual(order, ["super", "private"])
        resolve_device.assert_called_once_with()
        current.assert_called_once_with(torch.device("cuda:2"))
        prepare.assert_called_once_with(
            gamma=5,
            verify_slots=6,
            vocab_size=257,
            device=torch.device("cuda:2"),
            stream=stream,
            block_verification=True,
        )

    def test_super_error_is_not_swallowed_or_followed_by_private_work(self):
        worker = self.worker()
        with (
            patch.object(
                BaseSpecWorker, "prewarm_sampling", side_effect=RuntimeError("target")
            ),
            patch(HELPER) as prepare,
        ):
            with self.assertRaisesRegex(RuntimeError, "target"):
                worker.prewarm_sampling()
            prepare.assert_not_called()


class TestDSparkPrivateSamplingWarmup(CustomTestCase):
    @contextmanager
    def cpu_execution_surface(
        self, classic, bonus, *, capturing=False, wrong_stream=False
    ):
        """Real CPU tensors with only the CUDA execution surface replaced."""
        reject = ModuleType("sglang.kernels.ops.speculative.reject_sampling")
        accept = ModuleType("sglang.kernels.ops.speculative.dspark.dspark_accept")
        reject.chain_speculative_sampling_triton = classic
        accept.gather_two_level_bonus_triton = bonus
        stream = SimpleNamespace(
            device=torch.device("cuda:2"), cuda_stream=37, synchronize=Mock()
        )
        current = (
            SimpleNamespace(device=stream.device, cuda_stream=38)
            if wrong_stream
            else stream
        )
        factories = {name: getattr(torch, name) for name in ("zeros", "full", "arange")}
        original_fork = torch.random.fork_rng

        def factory(name):
            def create(*args, **kwargs):
                self.assertEqual(torch.device(kwargs.pop("device")), stream.device)
                return factories[name](*args, **kwargs)

            return create

        @contextmanager
        def fork(devices):
            requested = list(devices)
            with original_fork(devices=[]):
                yield
            self.assertEqual(requested, [2])

        with (
            patch.dict(
                "sys.modules", {reject.__name__: reject, accept.__name__: accept}
            ),
            patch.object(torch.cuda, "device", return_value=nullcontext()),
            patch.object(torch.cuda, "current_stream", return_value=current),
            patch.object(
                torch.cuda, "is_current_stream_capturing", return_value=capturing
            ),
            patch.object(torch.version, "hip", None),
            patch.object(torch.random, "fork_rng", side_effect=fork),
            patch.object(torch, "zeros", side_effect=factory("zeros")),
            patch.object(torch, "full", side_effect=factory("full")),
            patch.object(torch, "arange", side_effect=factory("arange")),
        ):
            yield stream

    def test_real_private_tensor_contract_rng_and_outputs(self):
        calls = []

        def classic(**kw):
            self.assertEqual(kw["target_probs"].shape, (1, 6, 257))
            self.assertEqual(kw["draft_probs"].shape, (1, 5, 257))
            for name in ("target_probs", "draft_probs", "uniform_samples"):
                self.assertEqual(kw[name].dtype, torch.float32)
                self.assertTrue(kw[name].is_contiguous())
            for name in (
                "candidates",
                "retrive_index",
                "retrive_next_token",
                "retrive_next_sibling",
            ):
                self.assertEqual(kw[name].dtype, torch.int64)
            for name in ("predicts", "accept_index", "accept_token_num"):
                self.assertEqual(kw[name].dtype, torch.int32)
            self.assertTrue(kw["block_verification"])
            self.assertEqual(kw["uniform_samples"].shape, (1, 5))
            torch.testing.assert_close(kw["target_probs"].sum(-1), torch.ones((1, 6)))
            torch.testing.assert_close(kw["draft_probs"].sum(-1), torch.ones((1, 5)))
            kw["predicts"].zero_()
            kw["accept_index"].copy_(kw["retrive_index"])
            kw["accept_token_num"].fill_(5)
            calls.append(kw)
            torch.rand(3)  # Check real CPU restoration, not a mocked equality.

        def bonus(**kw):
            self.assertIs(kw["correct_len"], calls[-1]["accept_token_num"])
            return torch.tensor([0], dtype=torch.int64)

        state = torch.random.get_rng_state().clone()
        with self.cpu_execution_surface(classic, bonus) as stream:
            prewarm_dspark_sampling(
                gamma=5,
                verify_slots=6,
                vocab_size=257,
                device="cuda:2",
                stream=stream,
                block_verification=True,
            )
        self.assertEqual(len(calls), 1)
        stream.synchronize.assert_called_once_with()
        self.assertTrue(torch.equal(torch.random.get_rng_state(), state))

    def test_primitive_error_propagates_and_restores_cpu_rng(self):
        def fail(**kwargs):
            torch.rand(7)
            raise RuntimeError("primitive failed")

        bonus = Mock()
        state = torch.random.get_rng_state().clone()
        with self.cpu_execution_surface(fail, bonus) as stream:
            with self.assertRaisesRegex(RuntimeError, "primitive failed"):
                prewarm_dspark_sampling(
                    gamma=2,
                    verify_slots=3,
                    vocab_size=17,
                    device="cuda:2",
                    stream=stream,
                )
        bonus.assert_not_called()
        self.assertTrue(torch.equal(torch.random.get_rng_state(), state))

    def test_capture_or_noncurrent_stream_rejected_before_primitive(self):
        for capture, wrong in ((True, False), (False, True)):
            with self.subTest(capture=capture, wrong=wrong):
                classic, bonus = Mock(), Mock()
                with self.cpu_execution_surface(
                    classic, bonus, capturing=capture, wrong_stream=wrong
                ) as stream:
                    with self.assertRaises((RuntimeError, ValueError)):
                        prewarm_dspark_sampling(
                            gamma=2,
                            verify_slots=3,
                            vocab_size=17,
                            device="cuda:2",
                            stream=stream,
                        )
                classic.assert_not_called()
                bonus.assert_not_called()

    def test_invalid_dimensions_rejected(self):
        for gamma, slots, vocab in ((0, 1, 17), (True, 2, 17), (2, 4, 17), (2, 3, 0)):
            with self.subTest(gamma=gamma, slots=slots, vocab=vocab):
                with self.assertRaises(ValueError):
                    prewarm_dspark_sampling(
                        gamma=gamma,
                        verify_slots=slots,
                        vocab_size=vocab,
                        device="cpu",
                        stream=None,
                    )


if __name__ == "__main__":
    unittest.main()
