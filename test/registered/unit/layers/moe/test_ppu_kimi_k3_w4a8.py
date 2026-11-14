"""CPU contract checks for the PPU Kimi-K3 W4A8 MoE path."""

import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch, sentinel

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.srt.layers.deep_gemm_wrapper import compile_utils, entrypoint
from sglang.srt.layers.moe import MoeA2ABackend, MoeRunnerBackend
from sglang.srt.layers.moe.fused_moe_triton import layer as fused_layer
from sglang.srt.layers.moe.moe_runner import deep_gemm as runner_module
from sglang.srt.layers.moe.moe_runner import ppu_deepgemm_moe as ppu_module
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.deep_gemm import (
    DeepGemmMoeQuantInfo,
    DeepGemmRunnerCore,
    DeepGemmRunnerInput,
)
from sglang.srt.layers.moe.token_dispatcher import deepep as deepep_module
from sglang.srt.layers.quantization import mxfp4
from sglang.srt.runtime_context import get_context, get_exec


@pytest.fixture(autouse=True)
def _runtime_context():
    with get_context().override_server_args(model_path="dummy"):
        yield


def _quant_info():
    return DeepGemmMoeQuantInfo(
        w13_weight=torch.empty((2, 256, 128), dtype=torch.uint8),
        w2_weight=torch.empty((2, 256, 64), dtype=torch.uint8),
        w13_scale=torch.empty((2, 4, 512), dtype=torch.uint8),
        w2_scale=torch.empty((2, 4, 256), dtype=torch.uint8),
        use_mxfp4_w4a8=True,
        w13_inv_scale=2.0,
        w2_inv_scale=4.0,
    )


def _runner(activation="situ"):
    runner = object.__new__(DeepGemmRunnerCore)
    runner.config = MoeRunnerConfig(
        activation=activation,
        gemm1_alpha=4.0 if activation == "situ" else None,
        gemm1_clamp_limit=25.0 if activation == "situ" else None,
    )
    runner.swiglu_limit = None
    return runner


@pytest.mark.parametrize("ue8m0", [False, True])
def test_w4a8_normal_dispatch_quant_keeps_channel_scale(ue8m0):
    dispatcher = object.__new__(deepep_module._DeepEPDispatcherImplNormal)
    dispatcher.use_fp8 = True
    dispatcher.async_finish = False
    dispatcher.quant_config = {"is_channel_quant": True}
    x = torch.empty((1, 256), dtype=torch.bfloat16)
    topk = SimpleNamespace(
        topk_weights=torch.ones((1, 1)),
        topk_ids=torch.zeros((1, 1), dtype=torch.int64),
    )
    with (
        patch.object(deepep_module.deep_gemm_wrapper, "ENABLE_JIT_DEEPGEMM", True),
        patch.object(deepep_module.deep_gemm_wrapper, "DEEPGEMM_SCALE_UE8M0", ue8m0),
        patch.object(
            deepep_module, "sglang_per_token_group_quant_fp8", create=True
        ) as quantize,
    ):
        dispatcher.dispatch_a(x, topk)
    assert "scale_multiplier" not in quantize.call_args.kwargs
    assert quantize.call_args.args[1] == 256

    for key in ("column_major_scales", "scale_tma_aligned", "scale_ue8m0"):
        assert quantize.call_args.kwargs[key] == ue8m0


def test_w4a8_standard_preprocess_keeps_channel_scale():
    from sglang.kernels.ops.moe import ep_moe_kernels

    x = torch.empty((1, 256), dtype=torch.bfloat16)
    topk = (torch.ones((1, 1)), torch.zeros((1, 1), dtype=torch.int32), None)
    output = (
        torch.zeros(2, dtype=torch.int32),
        None,
        torch.zeros((1, 1), dtype=torch.int32),
        torch.empty((2, 1, 256), dtype=torch.float8_e4m3fn),
        torch.ones((2, 1, 1), dtype=torch.float32),
    )
    config = SimpleNamespace(num_experts=4, num_local_experts=2, top_k=1, inplace=False)
    with (
        patch.object(runner_module, "_is_sail", True),
        patch.object(
            runner_module.deep_gemm_sm120,
            "allows_masked_standard_layout",
            return_value=False,
        ),
        patch.object(
            ep_moe_kernels, "moe_ep_deepgemm_preprocess", return_value=output
        ) as preprocess,
    ):
        result = runner_module.pre_permute_standard_to_deep_gemm(
            SimpleNamespace(hidden_states=x, topk_output=topk),
            _quant_info(),
            config,
            {},
        )
    assert "scale_multiplier" not in preprocess.call_args.kwargs
    assert result.hidden_states_scale is output[-1]


@pytest.mark.parametrize("a2a", [MoeA2ABackend.NONE, MoeA2ABackend.DEEPEP])
@pytest.mark.parametrize("sm", [89, 90])
@pytest.mark.parametrize(
    "output_dtype,bf16_env,reject",
    [
        ("auto", False, False),
        ("fp8", False, False),
        ("bf16", False, True),
        ("int8", False, True),
        ("uint8", False, True),
        ("auto", True, True),
        ("fp8", True, False),
    ],
)
def test_w4a8_selector_and_dispatch(a2a, sm, output_dtype, bf16_env, reject):
    with (
        patch.object(mxfp4, "_is_sail", True),
        patch.object(mxfp4, "get_device_sm", return_value=sm),
        patch.object(
            mxfp4, "get_moe_runner_backend", return_value=MoeRunnerBackend.DEEP_GEMM
        ),
        patch.object(mxfp4, "get_moe_a2a_backend", return_value=a2a),
        patch.object(mxfp4.deep_gemm_wrapper, "ENABLE_JIT_DEEPGEMM", True),
        patch.object(
            mxfp4.envs.SGLANG_SAIL_DEEPGEMM_MXFP4_W4A8, "get", return_value=True
        ),
        patch.object(
            mxfp4.envs.SGLANG_SAIL_DEEPGEMM_MXFP4_W4A16, "get", return_value=False
        ),
        patch.object(
            mxfp4.envs.SGLANG_SAIL_DEEPGEMM_MXFP4_W4A16_MMA, "get", return_value=False
        ),
        patch.object(fused_layer, "_is_sail", True),
        patch.object(
            mxfp4.envs.SGLANG_DEEPEP_BF16_DISPATCH, "get", return_value=bf16_env
        ),
        get_exec().moe.override(deepep_dispatcher_output_dtype=output_dtype),
    ):
        if a2a.is_deepep() and reject:
            with pytest.raises(ValueError, match="requires FP8 DeepEP dispatch"):
                mxfp4.Mxfp4MoEMethod("model.layers.0.experts")
            return
        method = mxfp4.Mxfp4MoEMethod("model.layers.0.experts")
        layer = SimpleNamespace(quant_method=method, quant_config=None)
        config = fused_layer.FusedMoE._build_dispatcher_quant_config(layer)

    assert method.use_deepgemm_mxfp4_w4a8
    assert not method.use_deepgemm_mxfp4_w4a16_mma
    assert config == {"dispatcher_output_dtype": "fp8", "is_channel_quant": True}


@pytest.mark.parametrize("sail,w4a8", [(False, True), (True, False)])
def test_w4a8_opt_in_does_not_change_other_paths(sail, w4a8):
    with (
        patch.object(mxfp4, "_is_sail", sail),
        patch.object(
            mxfp4, "get_moe_runner_backend", return_value=MoeRunnerBackend.DEEP_GEMM
        ),
        patch.object(
            mxfp4.envs.SGLANG_SAIL_DEEPGEMM_MXFP4_W4A8, "get", return_value=w4a8
        ),
        patch.object(
            mxfp4.envs.SGLANG_SAIL_DEEPGEMM_MXFP4_W4A16, "get", return_value=False
        ),
        patch.object(
            mxfp4.envs.SGLANG_SAIL_DEEPGEMM_MXFP4_W4A16_MMA, "get", return_value=False
        ),
        patch.object(mxfp4, "get_moe_a2a_backend", return_value=MoeA2ABackend.DEEPEP),
        patch.object(mxfp4.envs.SGLANG_DEEPEP_BF16_DISPATCH, "get", return_value=True),
        get_exec().moe.override(deepep_dispatcher_output_dtype="bf16"),
    ):
        method = mxfp4.Mxfp4MoEMethod("model.layers.0.experts")

    assert not method.use_deepgemm_mxfp4_w4a8


@pytest.mark.parametrize("activation", ["silu", "situ"])
def test_w4a8_weight_creation_accepts_activation(activation):
    method = mxfp4.Mxfp4MoEMethod.__new__(mxfp4.Mxfp4MoEMethod)
    method.use_marlin = False
    method.use_deep_gemm = True
    method.use_mega_moe = False
    method.use_deepgemm_mxfp4_w4a8 = True
    method.use_deepgemm_mxfp4_w4a16_valu = False
    method.use_deepgemm_mxfp4_w4a16_mma = False
    layer = torch.nn.Module()
    layer.num_local_experts = 2
    layer.hidden_size = 256
    layer.intermediate_size_per_partition = 128
    layer.moe_runner_config = MoeRunnerConfig(activation=activation)
    with patch.object(mxfp4, "_is_sail", True):
        method.create_weights(layer, 2, 256, 128, torch.bfloat16)
    assert layer.w13_weight.shape == (2, 256, 128)


@pytest.mark.parametrize("a2a", [MoeA2ABackend.NONE, MoeA2ABackend.DEEPEP])
def test_w4a8_preprocess_keeps_independent_inverse_scales(a2a):
    layer = SimpleNamespace(
        w13_weight=torch.nn.Parameter(
            torch.zeros((2, 256, 128), dtype=torch.uint8), requires_grad=False
        ),
        w13_weight_scale=torch.nn.Parameter(
            torch.full((2, 256, 8), 127, dtype=torch.uint8), requires_grad=False
        ),
        w2_weight=torch.nn.Parameter(
            torch.zeros((2, 256, 64), dtype=torch.uint8), requires_grad=False
        ),
        w2_weight_scale=torch.nn.Parameter(
            torch.full((2, 256, 4), 127, dtype=torch.uint8), requires_grad=False
        ),
        dispatcher=SimpleNamespace(
            quant_config={"dispatcher_output_dtype": "fp8"},
            set_quant_config=lambda config: setattr(
                layer.dispatcher, "quant_config", config
            ),
        ),
    )
    method = SimpleNamespace(use_deepgemm_mxfp4_w4a8=True)
    calls = []

    def preprocess(mode, fuse_activation, weight, scale):
        calls.append((mode, fuse_activation, weight.shape, scale.shape))
        return (
            weight,
            torch.full(
                (scale.shape[0], scale.shape[1] // 64, weight.shape[-1] * 4),
                len(calls),
                dtype=torch.uint8,
            ),
            float(len(calls) + 1),
        )

    fake_dg = SimpleNamespace(
        PreprocessMode=SimpleNamespace(W4A8_MXFP4_MMA=sentinel.w4a8),
        preprocess_weights_and_scales=preprocess,
    )
    with (
        patch.dict(sys.modules, {"deep_gemm": fake_dg}),
        patch.object(torch.cuda, "empty_cache"),
        patch.object(mxfp4, "get_moe_a2a_backend", return_value=a2a),
    ):
        mxfp4.Mxfp4MoEMethod.process_weights_after_loading(method, layer)

    assert [call[:2] for call in calls] == [(sentinel.w4a8, False)] * 2
    assert (layer.w13_weight_inv_scale, layer.w2_weight_inv_scale) == (2.0, 3.0)
    assert layer.w13_weight_scale.shape == (2, 4, 512)
    assert layer.w2_weight_scale.shape == (2, 4, 256)
    assert layer.dispatcher.quant_config == {"dispatcher_output_dtype": "fp8"}


def test_w4a8_preprocess_pads_tp_down_k_after_loading():
    layer = SimpleNamespace(
        w13_weight=torch.nn.Parameter(
            torch.zeros((2, 384, 128), dtype=torch.uint8), requires_grad=False
        ),
        w13_weight_scale=torch.nn.Parameter(
            torch.full((2, 384, 8), 127, dtype=torch.uint8), requires_grad=False
        ),
        w2_weight=torch.nn.Parameter(
            torch.ones((2, 256, 96), dtype=torch.uint8), requires_grad=False
        ),
        w2_weight_scale=torch.nn.Parameter(
            torch.full((2, 256, 6), 125, dtype=torch.uint8), requires_grad=False
        ),
    )
    method = SimpleNamespace(use_deepgemm_mxfp4_w4a8=True)
    inputs = []

    def preprocess(mode, fuse_activation, weight, scale):
        inputs.append((weight.clone(), scale.clone()))
        return weight, scale, 1.0

    fake_dg = SimpleNamespace(
        PreprocessMode=SimpleNamespace(W4A8_MXFP4_MMA=sentinel.w4a8),
        preprocess_weights_and_scales=preprocess,
    )
    with (
        patch.dict(sys.modules, {"deep_gemm": fake_dg}),
        patch.object(torch.cuda, "empty_cache"),
        patch.object(mxfp4, "get_moe_a2a_backend", return_value=MoeA2ABackend.NONE),
    ):
        mxfp4.Mxfp4MoEMethod.process_weights_after_loading(method, layer)

    assert inputs[0][0].shape == (2, 384, 128)
    assert inputs[1][0].shape == (2, 256, 128)
    assert inputs[1][1].shape == (2, 256, 8)
    assert torch.count_nonzero(inputs[1][0][..., 96:]) == 0
    assert torch.all(inputs[1][1][..., 6:] == 127)


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("activation", ["silu", "situ"])
def test_w4a8_runner_scales_both_gemms(masked, activation):
    quant_info = _quant_info()
    shape = (2, 3, 256) if masked else (3, 256)
    quantized_input = torch.empty(shape, dtype=torch.float8_e4m3fn)
    scale = torch.full((*shape[:-1], 1), 3.0)
    masked_m = torch.tensor([1, 2], dtype=torch.int32) if masked else None
    runner_input = DeepGemmRunnerInput(
        hidden_states=quantized_input,
        hidden_states_scale=scale,
        use_masked_gemm=masked,
        masked_m=masked_m,
        expected_m=2 if masked else None,
        m_indices=torch.zeros((3,), dtype=torch.int32) if not masked else None,
    )
    down_input = torch.empty((*shape[:-1], 128), dtype=torch.float8_e4m3fn)
    down_scale = torch.full((*shape[:-1], 1), 7.0)
    fake_dg = SimpleNamespace(
        ActivationMode=SimpleNamespace(SILU=sentinel.silu, SITU=sentinel.situ),
        QuantMode=SimpleNamespace(FP8_CHANNEL=sentinel.fp8_channel),
        activation_and_mul=lambda *args, **kwargs: (down_input, down_scale),
        activation_and_mul_masked=lambda *args, **kwargs: (down_input, down_scale),
    )
    fn = (
        "grouped_gemm_nt_f8f4bf16_masked"
        if masked
        else "grouped_gemm_nt_f8f4bf16_nopad"
    )
    with (
        patch.dict(sys.modules, {"deep_gemm": fake_dg}),
        patch.object(runner_module.deep_gemm_wrapper, fn) as gemm,
        patch.object(
            fake_dg,
            "activation_and_mul_masked" if masked else "activation_and_mul",
            wraps=getattr(
                fake_dg, "activation_and_mul_masked" if masked else "activation_and_mul"
            ),
        ) as act,
    ):
        output = (
            _runner(activation)._run_w4a8_masked_gemm(runner_input, quant_info, {})
            if masked
            else _runner(activation)._run_w4a8_contiguous_gemm(
                runner_input, quant_info, {"all_tokens": 3}
            )
        )

    assert output.shape == shape
    assert gemm.call_count == 2
    torch.testing.assert_close(gemm.call_args_list[0].args[0][1], scale)
    assert gemm.call_args_list[0].kwargs["inv_scale"] == 2.0
    assert gemm.call_args_list[1].args[0][0] is down_input
    assert gemm.call_args_list[1].args[0][1] is down_scale
    assert gemm.call_args_list[1].kwargs["inv_scale"] == 4.0
    assert act.call_args.args[:3] == (
        sentinel.situ if activation == "situ" else sentinel.silu,
        sentinel.fp8_channel,
        gemm.call_args_list[0].args[2],
    )
    if masked:
        assert act.call_args.args[3] is masked_m
        assert gemm.call_args_list[1].args[-2:] == (masked_m, 2)
    assert "weight_inv_scale" not in act.call_args.kwargs


@pytest.mark.parametrize("inv_scale", [1.0, 2.0])
@pytest.mark.parametrize("layout", ["nopad", "masked", "fused"])
def test_w4a8_wrapper_uses_exported_grouped_gemm_name(layout, inv_scale):
    lhs_shape = (2, 8, 256) if layout == "masked" else (8, 256)
    lhs = (torch.empty(lhs_shape), torch.empty(lhs_shape[:-1] + (1,)))
    rhs = (torch.empty((2, 256, 128)), torch.empty((2, 256, 8)))
    out = torch.empty(lhs_shape[:-1] + (256,))
    extra_args = {
        "nopad": (sentinel.indices, sentinel.rows),
        "masked": (sentinel.masked_m, 8),
        "fused": (
            sentinel.rows,
            sentinel.experts,
            sentinel.tokens,
            sentinel.aligned,
            sentinel.config,
        ),
    }[layout]
    api_name = f"m_grouped_gemm_fp8_fp4_bf16_nt_{layout}"
    fake_dg = SimpleNamespace(**{api_name: Mock()})
    with (
        patch.object(entrypoint, "_is_sail", True),
        patch.object(entrypoint, "get_device_sm", return_value=89),
        patch.object(entrypoint, "deep_gemm", fake_dg, create=True),
        patch.object(
            compile_utils, "deep_gemm_execution_hook", return_value=nullcontext()
        ) as warmup,
    ):
        getattr(entrypoint, f"grouped_gemm_nt_f8f4bf16_{layout}")(
            lhs, rhs, out, *extra_args, inv_scale=inv_scale
        )
    getattr(fake_dg, api_name).assert_called_once_with(
        lhs, rhs, out, *extra_args, inv_scale=inv_scale
    )
    assert (
        warmup.call_args.kwargs
        == {
            "nopad": {"use_m_rows": True},
            "masked": {"masked_capacity": 8},
            "fused": {"top_k": 1},
        }[layout]
    )


@pytest.mark.parametrize(
    "kernel_type,executor_name",
    [
        (
            compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_F8F4BF16_NOPAD,
            "_GroupedNopadWarmupExecutor_w4a8",
        ),
        (
            compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_F8F4BF16_MASKED,
            "_GroupedMaskedWarmupExecutor_w4a8",
        ),
        (
            compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_F8F4BF16_FUSED,
            "_GroupedFusedWarmupExecutor_w4a8",
        ),
    ],
)
def test_w4a8_precompile_registers_each_layout(kernel_type, executor_name):
    with patch.object(
        compile_utils, executor_name, return_value=sentinel.executor
    ) as executor:
        result = compile_utils._BaseWarmupExecutor.create(
            kernel_type, max_m=8, n=256, k=256, num_groups=4
        )
    assert result is sentinel.executor
    executor.assert_called_once_with(max_m=8, n=256, k=256, num_groups=4)
    assert (
        compile_utils._BaseWarmupExecutor.get_memory_requirement(
            kernel_type, max_m=8, n=256, k=256, num_groups=4
        )
        > 0
    )


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("activation", ["silu", "situ"])
def test_w4a8_none_uses_fused_or_nopad_without_changing_combine(fused, activation):
    hidden_states = torch.empty((2, 256), dtype=torch.bfloat16)
    quantized = torch.empty((2, 256), dtype=torch.float8_e4m3fn)
    row_scale = torch.full((2, 1), 3.0)
    topk_ids = torch.tensor([[0], [1]], dtype=torch.int32)
    topk_weights = torch.ones((2, 1), dtype=torch.float32)
    rows = torch.tensor([1, 1], dtype=torch.int32)
    indices = torch.tensor([0, 1], dtype=torch.int32)
    down_input = torch.empty((2, 128), dtype=torch.float8_e4m3fn)
    down_scale = torch.ones((2, 1), dtype=torch.float32)
    fake_dg = SimpleNamespace(
        ActivationMode=SimpleNamespace(SILU=sentinel.silu, SITU=sentinel.situ),
        QuantMode=SimpleNamespace(FP8_CHANNEL=sentinel.fp8_channel),
        activation_and_mul=lambda *args, **kwargs: (down_input, down_scale),
    )
    with (
        patch.dict(sys.modules, {"deep_gemm": fake_dg}),
        patch.object(
            ppu_module,
            "grouped_gemm_nt_f8f4bf16_nopad",
            ppu_module.grouped_gemm_nt_f8f4bf16_nopad.__wrapped__,
        ),
        patch.object(
            ppu_module,
            "sglang_per_token_quant_fp8",
            return_value=(quantized, row_scale),
            create=True,
        ) as quantize,
        patch.object(
            ppu_module.envs.SGLANG_SAIL_DEEPGEMM_MOE_TP_FUSED, "get", return_value=fused
        ),
        patch.object(
            ppu_module,
            "grouped_gemm_nt_f8f4bf16_fused",
            return_value=(rows, indices, indices),
        ) as fused_gemm,
        patch.object(
            ppu_module.deep_gemm_wrapper, "grouped_gemm_nt_f8f4bf16_nopad"
        ) as nopad_gemm,
        patch.object(
            ppu_module,
            "deepgemm_moe_permute",
            side_effect=lambda **kwargs: (
                quantized,
                kwargs["aq_scale"],
                indices,
                indices,
                rows,
            ),
        ) as permute,
        patch.object(
            fake_dg,
            "activation_and_mul",
            wraps=fake_dg.activation_and_mul,
        ) as act,
        patch.object(ppu_module, "ep_gather") as gather,
        patch.object(ppu_module, "SGLANG_PROFILE_NVTX", False),
    ):
        ppu_module.deep_moe_impl_fused(
            hidden_states,
            torch.empty((2, 256, 128), dtype=torch.uint8),
            torch.empty((2, 256, 64), dtype=torch.uint8),
            torch.empty((2, 4, 512), dtype=torch.uint8),
            torch.empty((2, 4, 256), dtype=torch.uint8),
            topk_weights,
            topk_ids,
            use_mxfp4_w4a8=True,
            w1_inv_scale=2.0,
            w2_inv_scale=4.0,
            activation=activation,
            gemm1_alpha=4.0 if activation == "situ" else None,
            gemm1_limit=25.0 if activation == "situ" else None,
            out_hidden_states=torch.empty_like(hidden_states),
        )

    assert fused_gemm.call_count == int(fused)
    assert permute.call_count == int(not fused)
    assert nopad_gemm.call_count == (1 if fused else 2)
    first = fused_gemm.call_args if fused else nopad_gemm.call_args_list[0]
    first_scale = first.args[1] if fused else first.args[0][1]
    torch.testing.assert_close(first_scale, row_scale)
    assert first.kwargs["inv_scale"] == 2.0
    assert nopad_gemm.call_args.args[0][0] is down_input
    assert nopad_gemm.call_args.args[0][1] is down_scale
    assert nopad_gemm.call_args.kwargs["inv_scale"] == 4.0
    assert act.call_args.args[:2] == (
        sentinel.situ if activation == "situ" else sentinel.silu,
        sentinel.fp8_channel,
    )
    quantize.assert_called_once_with(hidden_states)
    act.assert_called_once()
    gather.assert_called_once()
    assert "weight_inv_scale" not in act.call_args.kwargs


@pytest.mark.parametrize("layout", ["nopad", "masked", "fused"])
def test_w4a8_warmup_uses_dispatch_layout_and_includes_max_m(layout):
    from contextlib import ExitStack

    calls = []

    def preprocess(mode, fuse_activation, weight, scale):
        assert mode is sentinel.w4a8 and not fuse_activation
        return weight, scale, 1.0

    def execute(lhs, rhs, out, *metadata):
        calls.append((lhs[0].shape, lhs[1].shape, out.shape, metadata))
        out.fill_(7)

    fake_dg = SimpleNamespace(
        PreprocessMode=SimpleNamespace(W4A8_MXFP4_MMA=sentinel.w4a8),
        preprocess_weights_and_scales=preprocess,
        moe_align_block_size=lambda lhs, rhs, ids: (
            sentinel.config,
            sentinel.rows,
            sentinel.experts,
            sentinel.tokens,
            sentinel.aligned,
            None,
            None,
        ),
        **{f"m_grouped_gemm_fp8_fp4_bf16_nt_{layout}": execute},
    )
    with ExitStack() as stack:
        # Allocation/DeepGEMM are the accelerator boundaries; executor logic is real.
        for name in ("empty", "zeros", "full", "randint"):
            original = getattr(torch, name)
            stack.enter_context(
                patch.object(
                    torch,
                    name,
                    side_effect=lambda *args, _fn=original, **kwargs: _fn(
                        *args, **(kwargs | {"device": "cpu"})
                    ),
                )
            )
        stack.enter_context(
            patch.object(compile_utils, "deep_gemm", fake_dg, create=True)
        )
        kind = getattr(
            compile_utils.DeepGemmKernelType,
            f"GROUPED_GEMM_NT_F8F4BF16_{layout.upper()}",
        )
        options = {
            "nopad": {"use_m_rows": True},
            "masked": {"masked_capacity": 256},
            "fused": {"top_k": 2},
        }[layout]
        executor = compile_utils._BaseWarmupExecutor.create(
            kind, max_m=8, n=256, k=256, num_groups=2, **options
        )
        executor.execute(8)
    expected_lhs = (2, 256, 256) if layout == "masked" else (8, 256)
    expected_out = (
        (2, 256, 256) if layout == "masked" else (16 if layout == "fused" else 8, 256)
    )
    assert calls[0][:3] == (expected_lhs, expected_lhs[:-1] + (1,), expected_out)
    assert torch.all(executor.out == 7)
    if layout == "nopad":
        assert calls[0][3][-1].tolist() == [8, 0]
    elif layout == "masked":
        assert calls[0][3][-1] == 8


@pytest.mark.parametrize("sail,sm", [(False, 89), (True, 80)])
@pytest.mark.parametrize("layout", ["nopad", "masked", "fused"])
def test_w4a8_entrypoints_reject_unsupported_platform(sail, sm, layout):
    with (
        patch.object(entrypoint, "_is_sail", sail),
        patch.object(entrypoint, "get_device_sm", return_value=sm),
        pytest.raises(AssertionError, match="SAIL SM89"),
    ):
        extras = {"nopad": (None,), "masked": (None, 1), "fused": (None,) * 5}[layout]
        getattr(entrypoint, f"grouped_gemm_nt_f8f4bf16_{layout}")(
            None, None, None, *extras
        )


def test_w4a8_ll_input_remains_uncompensated_until_runner():
    shape = (2, 256, 256)
    output = (
        torch.empty(shape, dtype=torch.float8_e4m3fn),
        torch.ones((2, 256, 1)),
        torch.zeros((1, 1), dtype=torch.int64),
        torch.ones((1, 1)),
        torch.tensor([1, 0], dtype=torch.int32),
        1,
    )
    result = runner_module.pre_permute_deepep_ll_to_deep_gemm(
        output, _quant_info(), MoeRunnerConfig(), {}
    )
    assert result.activation_scale_block_size == 256
    assert result.hidden_states_scale is output[1]


@pytest.mark.parametrize("fused", [False, True])
def test_w4a8_custom_op_fake_accepts_inverse_scale(fused):
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        args = (
            torch.empty((3, 256), dtype=torch.float8_e4m3fn),
            torch.empty((3, 1), dtype=torch.float32),
            torch.empty((2, 256, 128), dtype=torch.uint8),
            torch.empty((2, 256, 8), dtype=torch.uint8),
            torch.empty((3, 256), dtype=torch.bfloat16),
        )
        if fused:
            result = ppu_module.grouped_gemm_nt_f8f4bf16_fused(
                *args, torch.empty((3, 2), dtype=torch.int32), inv_scale=2.0
            )
            assert [tensor.shape for tensor in result] == [(2,), (6,), (6,)]
        else:
            result = ppu_module.grouped_gemm_nt_f8f4bf16_nopad(
                *args,
                torch.empty((3,), dtype=torch.int32),
                torch.empty((2,), dtype=torch.int32),
                inv_scale=2.0,
            )
            assert result is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
