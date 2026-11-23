"""CPU checks for MHC-only startup planning and production hook boundaries.

Mocked CUDA objects validate control flow and argument ownership, not kernels,
numerics, memory capacity, stream completion, or full-model warmup benefit.
"""

import ast
import copy
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import MethodType, ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import sglang
from sglang.srt.models.deepseek_v4_prefill_context import (
    PrefillWarmupRuntime,
    plan_dp_prefill_rows,
)
from sglang.srt.models.deepseek_v4_warmup import (
    max_cp_physical_rows,
    normalize_local_rows,
    plan_prefill_warmup,
    prefill_warmup_rows,
    prepare_mega_mhc_bf16_startup,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
SCHEDULER_SOURCE = Path(sglang.__file__).parent / "srt/managers/scheduler.py"


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


def plan(*, config=None, **changes):
    options = dict(
        model_type="deepseek_v41", is_sail=True, cp_size=4, mega_mhc_enabled=True
    )
    options.update(changes)
    return plan_prefill_warmup(config or args(), **options)


def model_hook(globals_):
    # Exercise the actual method without importing a GPU-heavy model module.
    # Only its inference-mode decorator is omitted; no body is rewritten.
    source = Path(sglang.__file__).parent / "srt/models/deepseek_v4.py"
    tree = ast.parse(source.read_bytes())
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DeepseekV4ForCausalLM"
    )
    method = copy.deepcopy(
        next(
            node
            for node in owner.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "prewarm_prefill_kernels"
        )
    )
    method.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    exec(compile(module, str(source), "exec"), globals_)
    actual_method = globals_[method.name]

    def mhc_only(*args, **kwargs):
        # These two MHC-specific tests do not exercise independent Router
        # startup preparation. Keep the actual method AST intact and isolate
        # only that unrelated planner; Router has its own seven-method suite.
        with patch(
            "sglang.srt.models.deepseek_v4_router_warmup.plan_router_prefill_warmup",
            return_value=(),
        ):
            return actual_method(*args, **kwargs)

    return mhc_only


def stub_module(name, **members):
    module = ModuleType(name)
    module.__dict__.update(members)
    return module


class TestDeepseekV4Warmup(CustomTestCase):
    def test_dynamic_cp_chunk_and_max_prefill(self):
        cases = (
            (4, 32768, 32768, tuple(range(1, 8193))),
            (2, 32768, 32768, tuple(range(1, 16385))),
            (8, 32768, 32768, tuple(range(1, 4097))),
            (4, 16384, 16384, tuple(range(1, 4097))),
            (4, 32768, 8192, tuple(range(1, 8193))),
            (4, 32769, 8192, tuple(range(1, 8197))),
            (2, 2049, 2049, tuple(range(1, 1027))),
            (8, 1001, 1001, tuple(range(1, 129))),
        )
        for cp, chunk, maximum, expected in cases:
            with self.subTest(cp=cp, chunk=chunk, maximum=maximum):
                self.assertEqual(
                    plan(
                        cp_size=cp,
                        config=args(
                            chunked_prefill_size=chunk, max_prefill_tokens=maximum
                        ),
                    ),
                    expected,
                )
        self.assertEqual(plan(config=args(max_prefill_tokens=None)), plan())

    def test_cp_logical_tails_keep_physical_upper_bound(self):
        for cp in (2, 4, 8):
            for chunk in (1, 3, 5, 17, 257, 2049):
                with self.subTest(cp=cp, chunk=chunk):
                    rows = plan(cp_size=cp, config=args(chunked_prefill_size=chunk))
                    limit = max_cp_physical_rows(chunk, cp, cp)
                    self.assertEqual(rows, tuple(range(1, limit + 1)))
                    self.assertEqual(max(rows), limit)
                    self.assertEqual(rows[0], 1)
                    for tail in (1, 3, 5):
                        if tail <= limit:
                            self.assertIn(tail, rows)
                    self.assertEqual(
                        tuple(n for n in rows if n % cp == 0),
                        tuple(range(cp, limit + 1, cp)),
                    )

    def test_dp_planner_and_disabled_behavior_unchanged(self):
        for dp in (2, 4, 8):
            runtime = PrefillWarmupRuntime(
                chunked_prefill_size=256,
                max_prefill_tokens=128,
                page_size=128,
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
            options = args(enable_prefill_cp=False)
            with self.subTest(dp=dp):
                expected = plan_dp_prefill_rows(
                    options,
                    model_type="deepseek_v41",
                    is_sail=True,
                    runtime=runtime,
                )
                self.assertEqual(expected, tuple(range(1, 257)))
                self.assertEqual(
                    plan(config=options, cp_size=1, runtime=runtime), expected
                )
                self.assertEqual(
                    plan(
                        config=options,
                        cp_size=1,
                        runtime=runtime,
                        mega_mhc_enabled=False,
                    ),
                    (),
                )
                self.assertEqual(
                    plan(
                        config=args(enable_prefill_cp=False, warmups=[]),
                        cp_size=1,
                        runtime=runtime,
                    ),
                    (),
                )

    def test_disabled_and_invalid_plans(self):
        for change in (
            {"enable_prefill_cp": False},
            {"cp_strategy": "other"},
            {"warmups": []},
            {"warmups": None},
            {"chunked_prefill_size": 0},
            {"chunked_prefill_size": True},
        ):
            with self.subTest(change=change):
                self.assertEqual(plan(config=args(**change)), ())
        for change in (
            {"is_sail": False},
            {"model_type": "deepseek_v4"},
            {"mega_mhc_enabled": False},
            {"cp_size": 0},
            {"cp_size": True},
        ):
            with self.subTest(change=change):
                self.assertEqual(plan(**change), ())

    def test_physical_padding_rows_and_validation(self):
        for cp in (1, 2, 3, 4, 8, 16):
            for alignment in (cp, cp * 2):
                for tokens in (0, 1, 255, 1025, 35909, 40961):
                    with self.subTest(cp=cp, alignment=alignment, tokens=tokens):
                        ranks = [(tokens + cp - 1 - rank) // cp for rank in range(cp)]
                        expected = (max(ranks) + alignment - 1) // alignment * alignment
                        self.assertEqual(
                            max_cp_physical_rows(tokens, cp, alignment), expected
                        )
        self.assertEqual(prefill_warmup_rows(0, 4), ())
        self.assertEqual(prefill_warmup_rows(8192, 32), ())
        self.assertIn(2052, prefill_warmup_rows(2052, 4))
        self.assertEqual(normalize_local_rows([0, 4, 4, 8]), (4, 8))
        for value in (-1, True, 1.5, 2**20 + 1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_local_rows([value])
        for value in (0, -1, True):
            with self.subTest(alignment=value), self.assertRaises(ValueError):
                prefill_warmup_rows(8192, value)

    def test_model_disabled_before_cuda(self):
        for change in (
            {"is_sail": False},
            {"model_type": "deepseek_v4"},
            {"mega_enabled": False},
            {"warmups": []},
            {"enable_prefill_cp": False},
            {"cp_strategy": "other"},
        ):
            with self.subTest(change=change):
                cuda = Mock(side_effect=AssertionError("disabled hook touched CUDA"))
                globals_ = dict(
                    _is_sail=change.get("is_sail", True),
                    torch=SimpleNamespace(cuda=SimpleNamespace(current_stream=cuda)),
                    envs=SimpleNamespace(
                        SGLANG_OPT_DSV41_MEGA_MHC_PREFILL=SimpleNamespace(
                            get=lambda: change.get("mega_enabled", True)
                        ),
                        SGLANG_OPT_DEEPGEMM_HC_PRENORM=SimpleNamespace(
                            get=lambda: True
                        ),
                    ),
                )
                owner = SimpleNamespace(
                    config=SimpleNamespace(
                        model_type=change.get("model_type", "deepseek_v41")
                    ),
                    cp_size=4,
                    # TF32 prenorm prewarm is gated independently of MegaMHC;
                    # without a real hc_attn_fn weight it must stay CUDA-free.
                    model=SimpleNamespace(layers=[]),
                )
                model_hook(globals_)(
                    owner,
                    args(
                        **{
                            key: value
                            for key, value in change.items()
                            if key in {"warmups", "enable_prefill_cp", "cp_strategy"}
                        }
                    ),
                )
                cuda.assert_not_called()

    def test_scheduler_calls_hook_on_selected_serving_stream(self):
        source = SCHEDULER_SOURCE
        tree = ast.parse(source.read_bytes())
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "Scheduler"
        )
        setup = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef)
            and n.name == "maybe_init_model_prefill_warmup"
        )
        helper = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef)
            and n.name == "_prewarm_model_prefill_kernels"
        )
        selection = next(
            n
            for n in ast.walk(setup)
            if isinstance(n, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "forward_stream" for t in n.targets
            )
        )
        context = next(
            n
            for n in ast.walk(setup)
            if isinstance(n, ast.With)
            and any(
                isinstance(item.context_expr, ast.Call)
                and isinstance(item.context_expr.func, ast.Attribute)
                and item.context_expr.func.attr == "stream"
                and any(
                    isinstance(arg, ast.Name) and arg.id == "forward_stream"
                    for arg in item.context_expr.args
                )
                for item in n.items
            )
        )
        for overlap, pp in ((False, 1), (True, 1), (False, 2)):
            with self.subTest(overlap=overlap, pp=pp):
                active, observed = [], []

                class Context:
                    def __init__(self, stream):
                        self.stream = stream

                    def __enter__(self):
                        active.append(self.stream)

                    def __exit__(self, *exc):
                        active.pop()

                owner = SimpleNamespace(
                    enable_overlap=overlap, schedule_stream=object(), draft_worker=None
                )
                runner = SimpleNamespace(
                    forward_stream=object(),
                    is_draft_worker=False,
                    server_args=object(),
                    prewarm_sampling=lambda: None,
                )
                runner.model = SimpleNamespace(
                    prewarm_prefill_kernels=lambda config: observed.append(
                        (active[-1], config)
                    )
                )
                scope = dict(
                    self=owner,
                    model_runner=runner,
                    device_module=SimpleNamespace(stream=Context),
                    get_parallel=lambda: SimpleNamespace(pp_size=pp),
                    parallel=SimpleNamespace(pp_size=pp),
                    runtime=None,
                    use_mlx=lambda: False,
                    prewarm_graph_pool_borrow=lambda: None,
                )
                program = ast.Module(body=[copy.deepcopy(helper)], type_ignores=[])
                exec(
                    compile(ast.fix_missing_locations(program), str(source), "exec"),
                    scope,
                )
                owner._prewarm_model_prefill_kernels = MethodType(
                    scope[helper.name], owner
                )
                program = ast.Module(
                    body=[copy.deepcopy(selection), copy.deepcopy(context)],
                    type_ignores=[],
                )
                exec(
                    compile(ast.fix_missing_locations(program), str(source), "exec"),
                    scope,
                )
                expected = (
                    runner.forward_stream
                    if overlap or pp > 1
                    else owner.schedule_stream
                )
                self.assertEqual(observed, [(expected, runner.server_args)])
                self.assertEqual(active, [])
                runner.is_draft_worker = True
                owner._prewarm_model_prefill_kernels(runner)
                self.assertEqual(len(observed), 1)

    def test_model_actual_dispatch_and_serving_stream(self):
        bf16, stream, extend = object(), object(), object()
        hc = SimpleNamespace(
            cfg=SimpleNamespace(mult=4, hidden=5120, pre_from_prev=True),
            norm=SimpleNamespace(weight=SimpleNamespace(dtype=bf16)),
        )
        owner = SimpleNamespace(
            config=SimpleNamespace(model_type="deepseek_v41"),
            cp_size=4,
            model=SimpleNamespace(layers=[SimpleNamespace(ffn_hc=hc)]),
        )
        for supported, eligible in ((False, True), (True, False), (True, True)):
            with self.subTest(supported=supported, eligible=eligible):
                dispatch, current, prepare = (
                    Mock(return_value=eligible),
                    Mock(return_value=stream),
                    Mock(),
                )
                stubs = {
                    "sglang.kernels.ops.layernorm.mhc_mega": stub_module(
                        "sglang.kernels.ops.layernorm.mhc_mega",
                        _mega_mhc_supported=lambda: supported,
                    ),
                    "sglang.srt.model_executor.forward_batch_info": stub_module(
                        "sglang.srt.model_executor.forward_batch_info",
                        ForwardMode=SimpleNamespace(EXTEND=extend),
                    ),
                    "sglang.srt.models.deepseek_v4_mhc": stub_module(
                        "sglang.srt.models.deepseek_v4_mhc", _can_use_mega_mhc=dispatch
                    ),
                }
                globals_ = dict(
                    _is_sail=True,
                    SimpleNamespace=SimpleNamespace,
                    torch=SimpleNamespace(
                        bfloat16=bf16,
                        cuda=SimpleNamespace(
                            current_stream=current, empty_cache=Mock()
                        ),
                    ),
                    envs=SimpleNamespace(
                        SGLANG_OPT_DSV41_MEGA_MHC_PREFILL=SimpleNamespace(
                            get=lambda: True
                        ),
                        SGLANG_OPT_DEEPGEMM_HC_PRENORM=SimpleNamespace(
                            get=lambda: True
                        ),
                    ),
                    time=SimpleNamespace(perf_counter=lambda: 1.0),
                    logger=Mock(),
                )
                with (
                    patch.dict("sys.modules", stubs),
                    patch(
                        "sglang.srt.models.deepseek_v4_warmup.prepare_mega_mhc_bf16_startup",
                        prepare,
                    ),
                ):
                    model_hook(globals_)(owner, args())
                if supported:
                    dispatch.assert_called_once()
                    self.assertIs(dispatch.call_args.args[0], hc.cfg)
                    self.assertIs(dispatch.call_args.args[1].forward_mode, extend)
                else:
                    dispatch.assert_not_called()
                if supported and eligible:
                    current.assert_called_once()
                    prepare.assert_called_once_with(
                        hc, local_rows=plan(), stream=stream
                    )
                else:
                    current.assert_not_called()
                    prepare.assert_not_called()

    def helper_inputs(self):
        device, bf16, fp32 = object(), object(), object()
        device = SimpleNamespace(type="cuda", index=0)

        def tensor(shape, dtype):
            return SimpleNamespace(
                shape=shape, dtype=dtype, device=device, is_contiguous=lambda: True
            )

        hc = SimpleNamespace(
            cfg=SimpleNamespace(
                mult=4,
                hidden=5120,
                pre_from_prev=True,
                sinkhorn_iters=20,
                rms_eps=1e-6,
                eps=1e-6,
            ),
            norm=SimpleNamespace(weight=tensor((5120,), bf16), variance_epsilon=1e-6),
            fn=tensor((24, 20480), fp32),
            scale=tensor((3,), fp32),
            base=tensor((24,), fp32),
        )
        stream = SimpleNamespace(device=device, wait_stream=Mock(), synchronize=Mock())
        previous = object()
        module = SimpleNamespace(
            device=lambda _: nullcontext(),
            current_stream=lambda **_: previous,
            stream=lambda _: nullcontext(),
            is_current_stream_capturing=Mock(return_value=False),
        )
        torch = stub_module(
            "torch",
            bfloat16=bf16,
            float32=fp32,
            get_device_module=lambda _: module,
            inference_mode=lambda: nullcontext(),
            zeros=lambda shape, *, device, dtype: tensor(shape, dtype),
            ones=lambda shape, *, device, dtype: tensor(shape, dtype),
            zeros_like=lambda other: tensor(other.shape, other.dtype),
        )
        boundary = Mock(return_value=object())
        mega = stub_module(
            "sglang.kernels.ops.layernorm.mhc_mega", try_mhc_mega_boundary=boundary
        )
        return (
            hc,
            stream,
            previous,
            module,
            boundary,
            {"torch": torch, mega.__name__: mega},
        )

    def test_helper_private_inputs_actual_weights_and_zero(self):
        hc, stream, previous, _, boundary, stubs = self.helper_inputs()
        with patch.dict("sys.modules", stubs):
            self.assertIsNone(
                prepare_mega_mhc_bf16_startup(hc, local_rows=(0,), stream=stream)
            )
            boundary.assert_not_called()
            self.assertIsNone(
                prepare_mega_mhc_bf16_startup(
                    hc, local_rows=(1, 3, 252, 508), stream=stream
                )
            )
        self.assertEqual(boundary.call_count, 4)
        for n, call in zip((1, 3, 252, 508), boundary.call_args_list):
            private = call.args[:5]
            self.assertEqual(
                [x.shape for x in private],
                [(n, 5120), (n, 4, 5120), (n, 4), (n, 4), (n, 4, 4)],
            )
            self.assertEqual(len({id(x) for x in private}), 5)
            for got, weight in zip(
                call.args[5:9], (hc.fn, hc.scale, hc.base, hc.norm.weight)
            ):
                self.assertIs(got, weight)
                self.assertTrue(all(got is not item for item in private))
            self.assertEqual(
                call.args[9:],
                (
                    hc.cfg.rms_eps,
                    hc.cfg.eps,
                    hc.norm.variance_epsilon,
                    hc.cfg.sinkhorn_iters,
                ),
            )
            self.assertFalse(call.kwargs)
        stream.wait_stream.assert_called_once_with(previous)
        stream.synchronize.assert_called_once()

    def test_helper_rejects_capture_stream_and_bad_dispatch(self):
        for capture_states in ([True], [False, True]):
            with self.subTest(capture_states=capture_states):
                hc, stream, _, module, boundary, stubs = self.helper_inputs()
                module.is_current_stream_capturing.side_effect = capture_states
                with patch.dict("sys.modules", stubs), self.assertRaises(RuntimeError):
                    prepare_mega_mhc_bf16_startup(hc, local_rows=(252,), stream=stream)
                boundary.assert_not_called()
                stream.wait_stream.assert_not_called()
                stream.synchronize.assert_not_called()
        hc, stream, _, _, boundary, stubs = self.helper_inputs()
        boundary.return_value = None
        with (
            patch.dict("sys.modules", stubs),
            self.assertRaisesRegex(RuntimeError, "declined"),
        ):
            prepare_mega_mhc_bf16_startup(hc, local_rows=(252,), stream=stream)
        stream.synchronize.assert_not_called()
        for wrong in (
            None,
            SimpleNamespace(device=SimpleNamespace(type="cuda", index=1)),
        ):
            with (
                self.subTest(wrong=wrong),
                patch.dict("sys.modules", stubs),
                self.assertRaises(ValueError),
            ):
                prepare_mega_mhc_bf16_startup(hc, local_rows=(252,), stream=wrong)
        hc.norm.weight.dtype = object()
        with (
            patch.dict("sys.modules", stubs),
            self.assertRaisesRegex(ValueError, "BF16 wrapper"),
        ):
            prepare_mega_mhc_bf16_startup(hc, local_rows=(252,), stream=stream)


if __name__ == "__main__":
    unittest.main(verbosity=2)
