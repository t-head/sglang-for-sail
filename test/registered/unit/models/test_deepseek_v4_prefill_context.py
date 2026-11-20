"""CPU-only resolved prefill planning and startup ordering contracts."""

import ast
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

import sglang
from sglang.srt.models.deepseek_v4_prefill_context import (
    PrefillWarmupRuntime,
    plan_dp_prefill_rows,
)
from sglang.srt.models.deepseek_v4_router_warmup import plan_router_prefill_warmup
from sglang.srt.models.deepseek_v4_warmup import plan_prefill_warmup
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

PACKAGE = Path(sglang.__file__).parent
SCHEDULER = PACKAGE / "srt/managers/scheduler.py"
MODEL = PACKAGE / "srt/models/deepseek_v4.py"


def args(**values):
    data = dict(
        enable_prefill_cp=False,
        cp_strategy="interleave",
        warmups=["prefill_shapes"],
        chunked_prefill_size=65536,
        max_prefill_tokens=65536,
    )
    data.update(values)
    return SimpleNamespace(**data)


def runtime(dp=4, **values):
    data = dict(
        chunked_prefill_size=1024,
        max_prefill_tokens=1024,
        page_size=64,
        attn_dp_size=dp,
        attn_tp_size=1,
        attn_cp_size=1,
        tp_size=dp,
        moe_ep_size=dp,
        moe_a2a_backend="megamoe",
        mixed_chunk=False,
        dynamic_chunk=False,
        encoder_replay=False,
        prefill_graph_disabled=True,
    )
    data.update(values)
    return PrefillWarmupRuntime(**data)


def method(path, name):
    cls = next(
        n
        for n in ast.parse(path.read_bytes()).body
        if isinstance(n, ast.ClassDef) and n.name == "Scheduler"
    )
    return next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name
    )


class TestPrefillContext(CustomTestCase):
    def test_non_v41_legacy_hook_needs_no_v41_imports_or_fields(self):
        import sys

        prepare = method(SCHEDULER, "maybe_init_model_prefill_warmup")
        dispatch = method(SCHEDULER, "_prewarm_model_prefill_kernels")
        module = SimpleNamespace(stream=Mock(return_value=nullcontext()))
        namespace = {
            "torch": SimpleNamespace(get_device_module=lambda device: module),
            "get_parallel": lambda: SimpleNamespace(pp_size=1),
            "use_mlx": lambda: False,
            "get_schedule": Mock(
                side_effect=AssertionError("non-V41 read V41 schedule")
            ),
            "get_exec": Mock(
                side_effect=AssertionError("non-V41 read V41 exec fields")
            ),
        }
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(body=[prepare, dispatch], type_ignores=[])
                ),
                str(SCHEDULER),
                "exec",
            ),
            namespace,
        )
        callback = Mock()
        original_args = object()
        # No model.config, no chunk/page/DP fields, no V41 feature objects.
        runner = SimpleNamespace(
            device="cuda",
            forward_stream="forward",
            is_draft_worker=False,
            server_args=original_args,
            model=SimpleNamespace(prewarm_prefill_kernels=callback),
        )
        obj = SimpleNamespace(
            tp_worker=SimpleNamespace(model_runner=runner),
            enable_overlap=False,
            schedule_stream="schedule",
        )
        obj._prewarm_model_prefill_kernels = MethodType(
            namespace["_prewarm_model_prefill_kernels"], obj
        )
        with patch.dict(
            sys.modules,
            {
                "sglang.srt.models.deepseek_v4_prefill_context": None,
                "sglang.srt.layers.moe.utils": None,
            },
        ):
            namespace["maybe_init_model_prefill_warmup"](obj)
        callback.assert_called_once_with(original_args)
        module.stream.assert_called_once_with("schedule")
        namespace["get_exec"].assert_not_called()
        namespace["get_schedule"].assert_not_called()

    def test_dp_fp8_boundary_skip_does_not_skip_router(self):
        model_path = MODEL
        tree = ast.parse(model_path.read_bytes())
        node = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "prewarm_prefill_kernels"
        )
        node.decorator_list = []
        namespace = {
            "_is_sail": True,
            "torch": SimpleNamespace(
                cuda=SimpleNamespace(current_stream=lambda: "serving")
            ),
            "envs": SimpleNamespace(
                SGLANG_OPT_DSV41_MEGA_MHC_PREFILL=SimpleNamespace(get=lambda: True)
            ),
        }
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                str(model_path),
                "exec",
            ),
            namespace,
        )
        import sglang.srt.models.deepseek_v4_router_warmup as router_module

        obj = SimpleNamespace(
            config=SimpleNamespace(model_type="deepseek_v41"),
            cp_size=None,
            model=SimpleNamespace(
                layers=[
                    SimpleNamespace(
                        self_attn=SimpleNamespace(
                            accepts_deepgemm_mxfp8_input=lambda: True
                        )
                    )
                ]
            ),
        )
        with patch.object(router_module, "prepare_nonhash_router_startup") as prepare:
            namespace["prewarm_prefill_kernels"](obj, args(), runtime=runtime())
            prepare.assert_called_once()
            self.assertEqual(prepare.call_args.kwargs["stream"], "serving")

    def test_real_dp_method_none_model_cp_size_uses_resolved_runtime(self):
        model_path = MODEL
        tree = ast.parse(model_path.read_bytes())
        node = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "prewarm_prefill_kernels"
        )
        node.decorator_list = []
        namespace = {
            "_is_sail": True,
            "torch": SimpleNamespace(
                cuda=SimpleNamespace(current_stream=lambda: "serving")
            ),
            "envs": SimpleNamespace(
                SGLANG_OPT_DSV41_MEGA_MHC_PREFILL=SimpleNamespace(get=lambda: False)
            ),
        }
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                str(model_path),
                "exec",
            ),
            namespace,
        )
        import sglang.srt.models.deepseek_v4_router_warmup as router_module

        obj = SimpleNamespace(
            config=SimpleNamespace(model_type="deepseek_v41"),
            cp_size=None,
            model=SimpleNamespace(layers=[]),
        )
        with patch.object(router_module, "prepare_nonhash_router_startup") as prepare:
            namespace["prewarm_prefill_kernels"](obj, args(), runtime=runtime())
            prepare.assert_called_once()
            self.assertEqual(
                prepare.call_args.kwargs["local_rows"], tuple(range(1, 1025))
            )
            self.assertIsNone(obj.cp_size)

    def test_dp2_dp4_dp8_full_rank_local_domain(self):
        for dp in (2, 4, 8):
            with self.subTest(dp=dp):
                result = plan_dp_prefill_rows(
                    args(), model_type="deepseek_v41", is_sail=True, runtime=runtime(dp)
                )
                self.assertEqual(result, tuple(range(1, 1025)))

    def test_resolved_chunk_not_raw_cli_or_divided_dp(self):
        result = plan_dp_prefill_rows(
            args(chunked_prefill_size=65536),
            model_type="deepseek_v41",
            is_sail=True,
            runtime=runtime(chunked_prefill_size=16384),
        )
        self.assertEqual((result[0], result[-1], len(result)), (1, 16384, 16384))

    def test_soft_input_budget_does_not_shrink_chunk_domain(self):
        result = plan_dp_prefill_rows(
            args(),
            model_type="deepseek_v41",
            is_sail=True,
            runtime=runtime(max_prefill_tokens=128),
        )
        self.assertEqual(result[-1], 1024)

    def test_dp_unsupported_modes_preserve_existing_path(self):
        for change in (
            dict(mixed_chunk=True),
            dict(dynamic_chunk=True),
            dict(encoder_replay=True),
            dict(prefill_graph_disabled=False),
            dict(moe_a2a_backend="deepep"),
            dict(attn_tp_size=2),
            dict(attn_cp_size=2),
            dict(moe_ep_size=2),
            dict(chunked_prefill_size=None),
            dict(chunked_prefill_size=1025),
            dict(attn_dp_size=True),
        ):
            with self.subTest(change=change):
                self.assertEqual(
                    plan_dp_prefill_rows(
                        args(),
                        model_type="deepseek_v41",
                        is_sail=True,
                        runtime=runtime(**change),
                    ),
                    (),
                )
        self.assertEqual(
            plan_dp_prefill_rows(
                args(warmups=[]),
                model_type="deepseek_v41",
                is_sail=True,
                runtime=runtime(),
            ),
            (),
        )
        self.assertEqual(
            plan_dp_prefill_rows(
                args(), model_type="deepseek_v41", is_sail=False, runtime=runtime()
            ),
            (),
        )

    def test_router_and_mhc_enable_guards_are_independent(self):
        common = dict(
            model_type="deepseek_v41", is_sail=True, cp_size=1, runtime=runtime()
        )
        router = plan_router_prefill_warmup(args(), **common)
        self.assertEqual(len(router), 1024)
        self.assertEqual(
            plan_prefill_warmup(args(), mega_mhc_enabled=False, **common), ()
        )
        self.assertEqual(
            plan_prefill_warmup(args(), mega_mhc_enabled=True, **common), router
        )

    def test_cp_router_and_mhc_complete_domain_ignore_soft_cap(self):
        config = args(
            enable_prefill_cp=True, chunked_prefill_size=32768, max_prefill_tokens=8192
        )
        common = dict(model_type="deepseek_v41", is_sail=True, cp_size=4)
        router = plan_router_prefill_warmup(config, **common)
        self.assertEqual(router, tuple(range(4, 8193, 4)))
        self.assertEqual(
            plan_prefill_warmup(config, mega_mhc_enabled=True, **common),
            tuple(range(4, 8193, 4)),
        )

    def test_startup_order_finalizes_weights_and_resolves_chunk_first(self):
        init = method(SCHEDULER, "__init__")
        calls = {
            n.func.attr: n.lineno
            for n in ast.walk(init)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and isinstance(n.func.value, ast.Name)
            and n.func.value.id == "self"
        }
        self.assertLess(calls["init_model_worker"], calls["init_chunked_prefill"])
        self.assertLess(
            calls["init_chunked_prefill"], calls["maybe_init_dynamic_chunk_sizer"]
        )
        self.assertLess(
            calls["maybe_init_dynamic_chunk_sizer"],
            calls["maybe_init_model_prefill_warmup"],
        )
        worker = method(SCHEDULER, "init_model_worker")
        self.assertTrue(
            any(
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "finalize_startup_weight_load"
                for n in ast.walk(worker)
            )
        )

    def test_selected_serving_stream_and_explicit_runtime(self):
        node = method(SCHEDULER, "maybe_init_model_prefill_warmup")
        for overlap, pp, expected in (
            (False, 1, "schedule"),
            (True, 1, "forward"),
            (False, 2, "forward"),
        ):
            with self.subTest(overlap=overlap, pp=pp):
                module = SimpleNamespace(stream=Mock(return_value=nullcontext()))
                torch = SimpleNamespace(get_device_module=lambda device: module)
                parallel = SimpleNamespace(
                    attn_dp_size=4,
                    attn_tp_size=1,
                    attn_cp_size=1,
                    tp_size=4,
                    moe_ep_size=4,
                    pp_size=pp,
                )
                namespace = {
                    "torch": torch,
                    "get_parallel": lambda: parallel,
                    "get_schedule": lambda: SimpleNamespace(
                        enable_dynamic_chunking=False
                    ),
                    "use_mlx": lambda: False,
                    "get_exec": lambda: SimpleNamespace(
                        features=SimpleNamespace(
                            enable_encoder_swa_bounded_replay=False
                        ),
                        graph=SimpleNamespace(
                            cuda_graph_config=SimpleNamespace(
                                prefill=SimpleNamespace(backend="disabled")
                            )
                        ),
                    ),
                }
                program = ast.fix_missing_locations(
                    ast.Module(body=[node], type_ignores=[])
                )
                exec(compile(program, str(SCHEDULER), "exec"), namespace)
                runner = SimpleNamespace(
                    device="cuda",
                    is_draft_worker=False,
                    forward_stream="forward",
                    model=SimpleNamespace(
                        config=SimpleNamespace(model_type="deepseek_v41"),
                        prewarm_prefill_kernels=Mock(),
                    ),
                )
                obj = SimpleNamespace(
                    tp_worker=SimpleNamespace(model_runner=runner),
                    chunked_prefill_size=16384,
                    max_prefill_tokens=8192,
                    page_size=64,
                    is_mixed_chunk=False,
                    enable_overlap=overlap,
                    schedule_stream="schedule",
                    _prewarm_model_prefill_kernels=Mock(),
                )
                import sys

                fake_utils = SimpleNamespace(
                    get_moe_a2a_backend=lambda: SimpleNamespace(value="megamoe")
                )
                with patch.dict(
                    sys.modules, {"sglang.srt.layers.moe.utils": fake_utils}
                ):
                    namespace["maybe_init_model_prefill_warmup"](obj)
                passed = obj._prewarm_model_prefill_kernels.call_args.kwargs["runtime"]
                self.assertEqual(passed.chunked_prefill_size, 16384)
                self.assertEqual(passed.max_prefill_tokens, 8192)
                module.stream.assert_called_once_with(expected)


if __name__ == "__main__":
    unittest.main()
