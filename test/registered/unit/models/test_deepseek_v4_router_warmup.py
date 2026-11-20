"""CPU-only Router startup contracts; no GPU/RTC/accuracy coverage claim."""

import ast
import copy
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import sglang
from sglang.srt.models import deepseek_v4_router_warmup as target
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


def plan(config=None, **changes):
    options = dict(
        model_type="deepseek_v41",
        is_sail=True,
        cp_size=4,
        row_selector=lambda limit, cp: (cp * 33,),
    )
    options.update(changes)
    return target.plan_router_prefill_warmup(config or args(), **options)


def runtime():
    device = SimpleNamespace(type="cuda", index=0)
    weight = SimpleNamespace(
        device=device, dtype="bf16", shape=(384, 5120), is_contiguous=lambda: True
    )

    class Gate:
        def __init__(self):
            self.weight = weight
            self.tiny_router_gemm_max_tokens = 16
            self.forward = Mock(side_effect=AssertionError("gate.forward called"))

    gate = Gate()
    layers = [
        SimpleNamespace(mlp=SimpleNamespace(is_hash=True, gate=Gate())),
        SimpleNamespace(mlp=SimpleNamespace(is_hash=False, gate=gate)),
    ]
    stream = SimpleNamespace(device=device, wait_stream=Mock(), synchronize=Mock())
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
        def __init__(self, shape, storage=None, **kwargs):
            self.shape = shape
            self.storage = storage if storage is not None else object()
            self.__dict__.update(kwargs)

        def __getitem__(self, index):
            return Tensor(
                (index.stop, self.shape[1]),
                storage=self.storage,
                dtype=self.dtype,
                device=self.device,
            )

        def is_contiguous(self):
            return True

        def stride(self):
            return (5120, 1)

    torch.zeros = Mock(side_effect=Tensor)
    torch.cuda = SimpleNamespace(current_stream=Mock(return_value=stream))
    source = ModuleType("sglang.srt.models.deepseek_v2")
    source._is_sail = source._is_cuda = True
    source._use_aiter = False
    source.MoEGate = Gate
    source.use_intel_amx_backend = Mock(return_value=False)
    source.get_exec = Mock(
        return_value=SimpleNamespace(
            deterministic=SimpleNamespace(enable_deterministic_inference=False)
        )
    )
    gemm = ModuleType("sglang.kernels.ops.gemm.bf16_fp32")
    gemm._linear_bf16_fp32_algo = "cublas"
    gemm.linear_bf16_fp32 = Mock(
        side_effect=lambda x, w: SimpleNamespace(
            shape=(x.shape[0], w.shape[0]), dtype="fp32"
        )
    )
    package = ModuleType("sglang.kernels.ops.gemm")
    package.bf16_fp32 = gemm
    modules = {
        "torch": torch,
        source.__name__: source,
        package.__name__: package,
        gemm.__name__: gemm,
    }
    return SimpleNamespace(**locals())


class TestRouterProductionPreparation(CustomTestCase):
    def test_dynamic_limits_and_unselected_policy(self):
        for cp in (2, 4, 8):
            selector = Mock(return_value=(cp * 33,))
            self.assertEqual(plan(cp_size=cp, row_selector=selector), (cp * 33,))
            selector.assert_called_once_with(32768 // cp, cp)
        selector = Mock(return_value=(132,))
        plan(args(max_prefill_tokens=2048), row_selector=selector)
        # max-prefill is a soft budget; the hard aligned chunk remains32768.
        selector.assert_called_once_with(8192, 4)
        self.assertEqual(plan(row_selector=None), tuple(range(4, 8193, 4)))
        for rows in ((0,), (8196,), (133,), (True,)):
            with self.assertRaises(ValueError):
                plan(row_selector=lambda *_: rows)

    def test_disabled_configurations_do_not_select_rows(self):
        selector = Mock(side_effect=AssertionError("disabled planner ran selector"))
        for changes in (
            dict(warmups=[]),
            dict(warmups=None),
            dict(enable_prefill_cp=False),
            dict(cp_strategy="none"),
            dict(chunked_prefill_size=0),
        ):
            self.assertEqual(plan(args(**changes), row_selector=selector), ())
        for changes in (
            dict(cp_size=1),
            dict(cp_size=True),
            dict(is_sail=False),
            dict(model_type="deepseek_v4"),
        ):
            self.assertEqual(plan(row_selector=selector, **changes), ())

    def test_cp_rows_follow_the_resolved_moe_transport(self):
        for cp in (2, 4, 8):
            for backend in ("none", "megamoe", "deepep", "mori"):
                with self.subTest(cp=cp, backend=backend):
                    resolved = SimpleNamespace(
                        chunked_prefill_size=32768,
                        moe_a2a_backend=backend,
                    )
                    factor = cp if backend == "none" else 1
                    self.assertEqual(
                        plan(cp_size=cp, row_selector=None, runtime=resolved),
                        tuple(n * factor for n in range(cp, 32768 // cp + 1, cp)),
                    )
        resolved = SimpleNamespace(chunked_prefill_size=17, moe_a2a_backend="none")
        self.assertEqual(plan(cp_size=4, row_selector=None, runtime=resolved), (16, 32))
        self.assertEqual(
            plan(
                cp_size=4,
                row_selector=lambda limit, cp: (8, 4, 8),
                runtime=resolved,
            ),
            (16, 32),
        )
        with self.assertRaises(ValueError):
            plan(cp_size=4, row_selector=lambda *_: (12,), runtime=resolved)

    def test_cp_gathered_rows_respect_the_executor_cap(self):
        for cp in (2, 4, 8):
            resolved = SimpleNamespace(
                chunked_prefill_size=(1 << 20) + 1,
                moe_a2a_backend="none",
            )
            selector = Mock(side_effect=AssertionError("out-of-range policy ran"))
            with self.assertRaisesRegex(ValueError, "after CP transport"):
                plan(cp_size=cp, row_selector=selector, runtime=resolved)
            selector.assert_not_called()

    def test_actual_nonhash_weight_private_inputs_and_selected_stream(self):
        rt = runtime()
        with patch.dict("sys.modules", rt.modules):
            target.prepare_nonhash_router_startup(
                rt.layers, local_rows=(132, 516), stream=rt.stream
            )
        self.assertTrue(
            all(c.args[1] is rt.weight for c in rt.gemm.linear_bf16_fp32.call_args_list)
        )
        self.assertEqual(
            [c.args[0].shape for c in rt.gemm.linear_bf16_fp32.call_args_list],
            [(132, 5120), (516, 5120)],
        )
        rt.gate.forward.assert_not_called()
        rt.torch.zeros.assert_called_once_with(
            (516, 5120), device=rt.device, dtype="bf16"
        )
        tensors = [call.args[0] for call in rt.gemm.linear_bf16_fp32.call_args_list]
        self.assertIs(tensors[0].storage, tensors[1].storage)
        rt.module.stream.assert_called_once_with(rt.stream)
        rt.stream.wait_stream.assert_called_once_with(rt.previous)
        rt.stream.synchronize.assert_called_once()

    def test_pure_weight_executor_never_needs_runtime_context(self):
        rt = runtime()
        rt.source.get_exec.side_effect = AssertionError(
            "pure executor queried runtime_context"
        )
        with patch.dict("sys.modules", rt.modules):
            target.prepare_router_bf16_weight(
                rt.weight, local_rows=(132, 516), stream=rt.stream
            )
        self.assertEqual(rt.gemm.linear_bf16_fp32.call_count, 2)
        rt.source.get_exec.assert_not_called()
        rt.stream.synchronize.assert_called_once()

    def test_actual_forward_branch_guards(self):
        for mode in (
            "deterministic",
            "unpublished",
            "not-sail",
            "not-cuda",
            "aiter",
            "intel",
            "other-backend",
            "fp32",
            "wrong-shape",
            "strided",
            "tiny",
            "hash",
        ):
            rt = runtime()
            if mode == "deterministic":
                rt.source.get_exec.return_value.deterministic.enable_deterministic_inference = True
            if mode == "unpublished":
                rt.source.get_exec.side_effect = ValueError("not ready")
            if mode == "not-sail":
                rt.source._is_sail = False
            if mode == "not-cuda":
                rt.source._is_cuda = False
            if mode == "aiter":
                rt.source._use_aiter = True
            if mode == "intel":
                rt.source.use_intel_amx_backend.return_value = True
            if mode == "other-backend":
                rt.gemm._linear_bf16_fp32_algo = "deep_gemm"
            if mode == "fp32":
                rt.weight.dtype = "fp32"
            if mode == "wrong-shape":
                rt.weight.shape = (1024, 5120)
            if mode == "strided":
                rt.weight.is_contiguous = lambda: False
            if mode == "tiny":
                rt.gate.tiny_router_gemm_max_tokens = 516
            if mode == "hash":
                rt.layers[-1].mlp.is_hash = True
            with self.subTest(mode=mode), patch.dict("sys.modules", rt.modules):
                target.prepare_nonhash_router_startup(
                    rt.layers, local_rows=(132, 516), stream=rt.stream
                )
            rt.gemm.linear_bf16_fp32.assert_not_called()

    def test_wrong_stream_and_capture_rejected(self):
        for mode in ("missing", "wrong-device", "captured-current", "captured-target"):
            rt = runtime()
            stream = rt.stream
            if mode == "missing":
                stream = None
            if mode == "wrong-device":
                stream.device = SimpleNamespace(type="cuda", index=1)
            if mode == "captured-current":
                rt.module.is_current_stream_capturing.side_effect = (True,)
            if mode == "captured-target":
                rt.module.is_current_stream_capturing.side_effect = (False, True)
            with (
                self.subTest(mode=mode),
                patch.dict("sys.modules", rt.modules),
                self.assertRaises((ValueError, RuntimeError)),
            ):
                target.prepare_nonhash_router_startup(
                    rt.layers, local_rows=(132,), stream=stream
                )
            rt.gemm.linear_bf16_fp32.assert_not_called()

    def test_real_model_hook_router_does_not_depend_on_mhc(self):
        filename = Path(sglang.__file__).parent / "srt/models/deepseek_v4.py"
        owner = next(
            n
            for n in ast.parse(filename.read_bytes()).body
            if isinstance(n, ast.ClassDef) and n.name == "DeepseekV4ForCausalLM"
        )
        method = copy.deepcopy(
            next(
                n
                for n in owner.body
                if isinstance(n, ast.FunctionDef)
                and n.name == "prewarm_prefill_kernels"
            )
        )
        method.decorator_list = []
        program = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
        rt = runtime()
        scope = dict(
            torch=rt.torch,
            _is_sail=True,
            envs=SimpleNamespace(
                SGLANG_OPT_DSV41_MEGA_MHC_PREFILL=SimpleNamespace(get=lambda: False),
                SGLANG_OPT_DEEPGEMM_HC_PRENORM=SimpleNamespace(get=lambda: True),
            ),
        )
        exec(compile(program, str(filename), "exec"), scope)
        model = SimpleNamespace(
            config=SimpleNamespace(model_type="deepseek_v41"),
            cp_size=4,
            model=SimpleNamespace(layers=rt.layers),
        )
        prepare = Mock()
        with (
            patch.dict("sys.modules", rt.modules),
            patch.object(target, "ROUTER_ROW_SELECTOR", lambda *_: (132,)),
            patch.object(target, "prepare_nonhash_router_startup", prepare),
        ):
            scope[method.name](model, args())
        prepare.assert_called_once_with(rt.layers, local_rows=(132,), stream=rt.stream)


if __name__ == "__main__":
    unittest.main()
