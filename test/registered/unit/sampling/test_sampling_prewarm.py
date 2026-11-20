"""CPU tests for filtered-sampling startup preparation (no CUDA execution)."""

import unittest
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import sglang.srt.model_executor.model_runner as runner_module
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.sampling.sampling_prewarm import prewarm_flashinfer_sampling

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestSamplingPrewarm(CustomTestCase):
    @contextmanager
    def _cpu_runtime(self, *, capturing=False):
        """Keep tensor math real while replacing only the CUDA execution surface."""
        metadata = ModuleType("sglang.srt.sampling.sampling_batch_info")
        metadata.SamplingBatchInfo = SimpleNamespace
        tensor_factories = {
            name: getattr(torch, name) for name in ("full", "ones", "zeros", "arange")
        }
        actual_fork_rng = torch.random.fork_rng
        observed_devices = []
        stream = Mock()

        @contextmanager
        def child_rng(devices):
            observed_devices.append(devices)
            # This CPU test verifies restoration of the real CPU state; actual
            # CUDA RNG restoration is checked by the device smoke test.
            with actual_fork_rng(devices=[]):
                yield

        def cpu_factory(name):
            def create(*args, **kwargs):
                self.assertEqual(
                    torch.device(kwargs.pop("device")), torch.device("cuda:3")
                )
                return tensor_factories[name](*args, **kwargs)

            return create

        with (
            patch.dict("sys.modules", {metadata.__name__: metadata}),
            patch.object(
                torch.cuda, "is_current_stream_capturing", return_value=capturing
            ),
            patch.object(
                torch.cuda, "current_stream", return_value=stream
            ) as get_stream,
            patch.object(torch.random, "fork_rng", side_effect=child_rng),
            patch.object(torch, "full", side_effect=cpu_factory("full")),
            patch.object(torch, "ones", side_effect=cpu_factory("ones")),
            patch.object(torch, "zeros", side_effect=cpu_factory("zeros")),
            patch.object(torch, "arange", side_effect=cpu_factory("arange")),
        ):
            yield stream, get_stream, observed_devices

    def test_both_local_primitives_restore_rng_and_finish_selected_stream(self):
        sampler = Mock()
        calls = []

        def sample(probs, info, positions, *, simple_sampling_case):
            self.assertFalse(simple_sampling_case)
            self.assertEqual(probs.shape, (1, 128))
            self.assertEqual(probs.dtype, torch.float32)
            torch.testing.assert_close(probs.sum(dim=-1), torch.tensor([1.0]))
            self.assertGreater(probs[0, 0].item(), probs[0, 1].item())
            self.assertGreater(probs[0, 63].item(), probs[0, 64].item())
            self.assertEqual(info.top_ks.dtype, torch.int32)
            self.assertEqual(info.top_ks.tolist(), [32])
            self.assertEqual(positions.tolist(), [0])
            calls.append((info.need_min_p_sampling, info.min_ps.item()))
            torch.rand(7)
            return torch.tensor([0], dtype=torch.long), None

        sampler._sample_from_probs.side_effect = sample
        state = torch.random.get_rng_state().clone()
        with self._cpu_runtime() as (stream, get_stream, devices):
            prewarm_flashinfer_sampling(sampler, vocab_size=128, device="cuda:3")
        self.assertEqual([row[0] for row in calls], [False, True])
        self.assertEqual(calls[0][1], 0.0)
        self.assertAlmostEqual(calls[1][1], 0.05)
        self.assertEqual(devices, [[3]])
        get_stream.assert_called_once_with(torch.device("cuda:3"))
        stream.synchronize.assert_called_once_with()
        torch.testing.assert_close(torch.random.get_rng_state(), state, rtol=0, atol=0)
        sampler.forward.assert_not_called()
        sampler._sync_token_ids_across_tp.assert_not_called()

    def test_error_is_not_hidden_and_cpu_rng_is_restored(self):
        def fail(*args, **kwargs):
            torch.rand(7)
            raise RuntimeError("kernel preparation failed")

        sampler = Mock()
        sampler._sample_from_probs.side_effect = fail
        state = torch.random.get_rng_state().clone()
        with (
            self._cpu_runtime(),
            self.assertRaisesRegex(RuntimeError, "kernel preparation failed"),
        ):
            prewarm_flashinfer_sampling(sampler, vocab_size=128, device="cuda:3")
        torch.testing.assert_close(torch.random.get_rng_state(), state, rtol=0, atol=0)

    def test_capture_fails_before_any_sampling_or_allocation(self):
        sampler = Mock()
        with self._cpu_runtime(capturing=True) as (_, _, devices):
            with self.assertRaisesRegex(RuntimeError, "outside graph capture"):
                prewarm_flashinfer_sampling(sampler, vocab_size=128, device="cuda:3")
        self.assertEqual(devices, [])
        sampler._sample_from_probs.assert_not_called()

    def test_non_cuda_or_hip_skips_before_sampling_metadata_import(self):
        sampler = Mock()
        for device, hip in (("cpu", None), ("cuda:3", "6.0")):
            with (
                self.subTest(device=device, hip=hip),
                patch.object(torch.version, "hip", hip),
                patch.object(torch.cuda, "is_current_stream_capturing") as capture,
            ):
                prewarm_flashinfer_sampling(sampler, vocab_size=128, device=device)
                capture.assert_not_called()
        sampler._sample_from_probs.assert_not_called()

    def test_non_positive_vocabulary_fails_before_sampling(self):
        with patch.object(torch.cuda, "is_current_stream_capturing") as capture:
            with self.assertRaisesRegex(ValueError, "positive vocabulary"):
                prewarm_flashinfer_sampling(Mock(), vocab_size=0, device="cuda:3")
        capture.assert_not_called()


class TestRunnerSamplingPrewarm(CustomTestCase):
    def _runner(self):
        runner = ModelRunner.__new__(ModelRunner)
        runner.is_generation = True
        runner.is_draft_worker = False
        runner.pp_group = SimpleNamespace(is_last_rank=True)
        runner.device = "cuda:3"
        runner.model_config = SimpleNamespace(vocab_size=128)
        return runner

    def test_disabled_paths_return_default_result_without_prewarm(self):
        for field, value in (
            ("is_generation", False),
            ("is_draft_worker", True),
            ("pp_group", SimpleNamespace(is_last_rank=False)),
            ("device", "cpu"),
            ("backend", "pytorch"),
            ("hip", "6.0"),
        ):
            runner = self._runner()
            backend, hip = "flashinfer", None
            if field == "backend":
                backend = value
            elif field == "hip":
                hip = value
            else:
                setattr(runner, field, value)
            with (
                self.subTest(field=field),
                patch.object(
                    runner_module,
                    "get_exec",
                    return_value=SimpleNamespace(
                        kernel=SimpleNamespace(sampling_backend=backend)
                    ),
                ),
                patch.object(torch.version, "hip", hip),
                patch(
                    "sglang.srt.sampling.sampling_prewarm.prewarm_flashinfer_sampling"
                ) as prewarm,
            ):
                result = runner.prewarm_sampling()
                prewarm.assert_not_called()
                self.assertIs(result, runner.sampling_prewarm_result)
                self.assertEqual(result.sampling_input_bytes, 0)
                self.assertEqual(result.sampling_headroom_bytes, 0)

    def test_builtin_sampler_runs_and_custom_sampler_is_not_called(self):
        from sglang.srt.layers.sampler import Sampler

        class CustomSampler(Sampler):
            pass

        for sampler_type in (Sampler, CustomSampler):
            runner = self._runner()
            runner.sampler = sampler_type.__new__(sampler_type)
            with (
                self.subTest(sampler_type=sampler_type),
                patch.object(
                    runner_module,
                    "get_exec",
                    return_value=SimpleNamespace(
                        kernel=SimpleNamespace(sampling_backend="flashinfer")
                    ),
                ),
                patch.object(torch.version, "hip", None),
                patch(
                    "sglang.srt.sampling.sampling_prewarm.prewarm_flashinfer_sampling"
                ) as prewarm,
            ):
                result = runner.prewarm_sampling()
                if sampler_type is Sampler:
                    prewarm.assert_called_once_with(
                        runner.sampler, vocab_size=128, device="cuda:3"
                    )
                else:
                    prewarm.assert_not_called()
                self.assertEqual(result.sampling_headroom_bytes, 0)


if __name__ == "__main__":
    unittest.main()
