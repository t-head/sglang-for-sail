"""CPU-only TF32 prenorm prewarm contracts; no GPU/RT C/accuracy claim.

Mocked CUDA/DeepGEMM objects validate control flow, argument ownership and the
compile-only mode transition, not kernels, numerics, memory or TTFT benefit.
"""

import unittest
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.models import deepseek_v4_tf32_warmup as target
from sglang.srt.models.deepseek_v4_warmup import plan_prefill_warmup
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def args(**changes):
    values = dict(
        enable_prefill_cp=True,
        cp_strategy="interleave",
        warmups=["prefill_shapes"],
        chunked_prefill_size=32768,
        max_prefill_tokens=32768,
    )
    values.update(changes)
    return SimpleNamespace(**values)


def plan(*, deep_gemm_enabled=True, **changes):
    options = dict(model_type="deepseek_v41", is_sail=True, cp_size=4)
    options.update(changes)
    return target.plan_tf32_prenorm_warmup(
        args(), deep_gemm_enabled=deep_gemm_enabled, **options
    )


def runtime(*, mode=0):
    device = SimpleNamespace(type="cuda", index=0)
    weight = SimpleNamespace(
        device=device, dtype="fp32", shape=(24, 20480), is_contiguous=lambda: True
    )
    stream = SimpleNamespace(
        device=device, wait_stream=Mock(), synchronize=Mock()
    )
    previous = object()
    module = SimpleNamespace(
        device=lambda _: nullcontext(),
        stream=Mock(side_effect=lambda _: nullcontext()),
        current_stream=lambda **_: previous,
        is_current_stream_capturing=Mock(return_value=False),
    )
    torch = ModuleType("torch")
    torch.bfloat16, torch.float32 = "bf16", "fp32"
    torch.inference_mode = nullcontext
    torch.get_device_module = Mock(return_value=module)

    class Tensor:
        def __init__(self, shape, **kwargs):
            self.shape = shape
            self.__dict__.update(kwargs)

        def __getitem__(self, index):
            shape = (index.stop,) + tuple(self.shape[1:])
            return Tensor(shape, dtype=self.dtype, device=self.device)

        def is_contiguous(self):
            return True

    torch.empty = Mock(side_effect=Tensor)
    torch.random = SimpleNamespace(fork_rng=lambda **_: nullcontext())

    state = {"mode": mode}
    deep_gemm = ModuleType("deep_gemm")
    deep_gemm.get_compile_mode = Mock(side_effect=lambda: state["mode"])
    deep_gemm.set_compile_mode = Mock(
        side_effect=lambda m: state.__setitem__("mode", m)
    )
    deep_gemm.tf32_hc_prenorm_gemm = Mock()
    modules = {"torch": torch, "deep_gemm": deep_gemm}
    return SimpleNamespace(**locals())


class TestTf32PrenormWarmup(CustomTestCase):
    def test_plan_reuses_shared_bounds_and_respects_deep_gemm_gate(self):
        self.assertEqual(plan(deep_gemm_enabled=False), ())
        self.assertEqual(plan(is_sail=False), ())
        self.assertEqual(plan(model_type="deepseek_v4"), ())
        self.assertEqual(plan(cp_size=1), ())
        self.assertTrue(plan())
        # Same audited bounds as the shared CP planner, independent of MegaMHC.
        self.assertEqual(
            plan(),
            plan_prefill_warmup(
                args(),
                model_type="deepseek_v41",
                is_sail=True,
                cp_size=4,
                mega_mhc_enabled=True,
            ),
        )

    def test_num_splits_validation(self):
        rt = runtime()
        for bad in (0, -1, 1.5, True, "1"):
            with self.subTest(num_splits=bad), self.assertRaises(ValueError):
                target.prepare_tf32_prenorm_startup(
                    rt.weight, max_rows=4, stream=rt.stream, num_splits=bad
                )
        self.assertFalse(
            target.prepare_tf32_prenorm_startup(
                rt.weight, max_rows=0, stream=rt.stream, num_splits=1
            )
        )
        rt.deep_gemm.tf32_hc_prenorm_gemm.assert_not_called()

    def test_compile_only_mode_is_entered_and_restored(self):
        rt = runtime(mode=0)
        with patch.dict("sys.modules", rt.modules):
            self.assertTrue(
                target.prepare_tf32_prenorm_startup(
                    rt.weight, max_rows=3, stream=rt.stream, num_splits=1
                )
            )
        rt.deep_gemm.set_compile_mode.assert_any_call(1)
        rt.deep_gemm.set_compile_mode.assert_called_with(0)
        self.assertEqual(rt.deep_gemm.get_compile_mode(), 0)
        # Descending row visits; the exact serving signature is precompiled.
        rows = [
            c.args[0].shape[0]
            for c in rt.deep_gemm.tf32_hc_prenorm_gemm.call_args_list
        ]
        self.assertEqual(rows, [3, 2, 1])
        self.assertTrue(
            all(
                c.kwargs["num_splits"] == 1
                for c in rt.deep_gemm.tf32_hc_prenorm_gemm.call_args_list
            )
        )
        rt.stream.wait_stream.assert_called_once_with(rt.previous)
        rt.stream.synchronize.assert_called_once()

    def test_non_sail_signature_passes_none(self):
        rt = runtime()
        with patch.dict("sys.modules", rt.modules):
            target.prepare_tf32_prenorm_startup(
                rt.weight, max_rows=2, stream=rt.stream, num_splits=None
            )
        self.assertTrue(
            all(
                c.kwargs["num_splits"] is None
                for c in rt.deep_gemm.tf32_hc_prenorm_gemm.call_args_list
            )
        )

    def test_rejects_bad_weight_and_stream(self):
        rt = runtime()
        with patch.dict("sys.modules", rt.modules):
            with self.assertRaises(ValueError):
                target.prepare_tf32_prenorm_startup(
                    SimpleNamespace(
                        device=SimpleNamespace(type="cpu", index=0),
                        dtype="fp32",
                        shape=(24, 20480),
                        is_contiguous=lambda: True,
                    ),
                    max_rows=4,
                    stream=rt.stream,
                    num_splits=1,
                )
            with self.assertRaises(ValueError):
                target.prepare_tf32_prenorm_startup(
                    rt.weight, max_rows=4, stream=None, num_splits=1
                )
            rt.module.is_current_stream_capturing = Mock(return_value=True)
            with self.assertRaises(RuntimeError):
                target.prepare_tf32_prenorm_startup(
                    rt.weight, max_rows=4, stream=rt.stream, num_splits=1
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
