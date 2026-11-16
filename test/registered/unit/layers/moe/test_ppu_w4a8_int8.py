"""CPU contract checks for the PPU W4A8-Int8 (INT8 x packed INT4) MoE path."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner import deep_gemm as runner_module
from sglang.srt.layers.moe.moe_runner.deep_gemm import (
    DeepGemmMoeQuantInfo,
    DeepGemmRunnerCore,
    DeepGemmRunnerInput,
)
from sglang.srt.layers.quantization import w4a8_int8 as w4a8_module
from sglang.srt.layers.quantization.w4a8_int8 import (
    W4A8Int8Config,
    W4A8Int8MoEMethod,
)
from sglang.srt.runtime_context import get_context
from sglang.test.test_utils import CustomTestCase

_E = 2
_K = 256
_INTER2 = 256  # 2 * intermediate_size_per_partition, w13 output columns
_INTER = 128  # intermediate_size_per_partition, w2 output rows


def _runtime_override():
    return get_context().override_server_args(model_path="dummy")


def _int8_quant_info():
    return DeepGemmMoeQuantInfo(
        w13_weight=torch.empty((_E, _INTER2, _K // 2), dtype=torch.uint8),
        w2_weight=torch.empty((_E, _K, _INTER // 2), dtype=torch.uint8),
        use_int4_w4a8=True,
        per_channel_quant=True,
        w13_scale=torch.empty((_E, _INTER2, 1), dtype=torch.float32),
        w2_scale=torch.empty((_E, _K, 1), dtype=torch.float32),
        block_shape=None,
    )


def _reference_silu_and_mul(gateup, out):
    """CPU reference of silu_and_mul used to stub _legacy_silu_and_mul."""
    half = gateup.shape[-1] // 2
    gate, up = gateup[..., :half], gateup[..., half:]
    torch.mul(gate * torch.sigmoid(gate), up, out=out)


def _runner():
    runner = object.__new__(DeepGemmRunnerCore)
    runner.config = MoeRunnerConfig(activation="silu")
    runner.swiglu_limit = None
    return runner


class TestW4A8Int8Path(CustomTestCase):
    def setUp(self):
        self._ctx = _runtime_override()
        self._ctx.__enter__()

    def tearDown(self):
        self._ctx.__exit__(None, None, None)

    # ---- Config ----

    def test_config_name_and_pack_factor(self):
        config = W4A8Int8Config({"ignore": ["mlp.gate"]})
        self.assertEqual(config.get_name(), "w4a8_int8")
        self.assertEqual(config.weight_bits, 4)
        self.assertEqual(config.pack_factor, 2)
        self.assertIs(config.can_fuse_shared_expert(), False)

    def test_get_quant_method_dispatch(self):
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
        from sglang.srt.layers.quantization.unquant import (
            UnquantizedFusedMoEMethod,
            UnquantizedLinearMethod,
        )
        from sglang.srt.layers.quantization.w8a8_int8 import W8A8Int8LinearMethod

        config = W4A8Int8Config({"ignore": ["mlp.gate", "mlp.experts"]})
        with patch.object(w4a8_module, "_is_sail", True):
            linear = config.get_quant_method(MagicMock(spec=LinearBase), "model.dense")
            moe = config.get_quant_method(MagicMock(spec=FusedMoE), "mlp.experts")
            ignored_linear = config.get_quant_method(
                MagicMock(spec=LinearBase), "mlp.gate"
            )
            ignored_moe = config.get_quant_method(MagicMock(spec=FusedMoE), "mlp.gate")
            other = config.get_quant_method(torch.nn.Linear(1, 1), "other")

        self.assertIsInstance(linear, W8A8Int8LinearMethod)
        self.assertIsInstance(moe, W4A8Int8MoEMethod)
        self.assertIsInstance(ignored_linear, UnquantizedLinearMethod)
        self.assertIsInstance(ignored_moe, UnquantizedFusedMoEMethod)
        self.assertIsNone(other)

    # ---- create_weights ----

    def test_create_weights_shapes_and_scales(self):
        method = W4A8Int8MoEMethod.__new__(W4A8Int8MoEMethod)
        method.quant_config = W4A8Int8Config({})
        layer = torch.nn.Module()
        loader = Mock()
        method.create_weights(
            layer,
            num_experts=_E,
            hidden_size=_K,
            intermediate_size_per_partition=_INTER,
            params_dtype=torch.bfloat16,
            weight_loader=loader,
        )
        self.assertEqual(layer.w13_weight.shape, (_E, _INTER2, _K // 2))
        self.assertEqual(layer.w13_weight.dtype, torch.uint8)
        self.assertEqual(layer.w2_weight.shape, (_E, _K, _INTER // 2))
        self.assertEqual(layer.w2_weight.dtype, torch.uint8)
        self.assertEqual(layer.w13_weight_scale.shape, (_E, _INTER2, 1))
        self.assertEqual(layer.w2_weight_scale.shape, (_E, _K, 1))
        self.assertIsNone(layer.w13_input_scale)
        self.assertIsNone(layer.w2_input_scale)
        # Scale params must carry the CHANNEL loader binding.
        self.assertIs(layer.w13_weight_scale.weight_loader, loader)

    def test_create_weights_rejects_bias(self):
        method = W4A8Int8MoEMethod.__new__(W4A8Int8MoEMethod)
        method.quant_config = W4A8Int8Config({})
        layer = torch.nn.Module()
        with self.assertRaises(ValueError) as cm:
            method.create_weights(
                layer,
                num_experts=_E,
                hidden_size=_K,
                intermediate_size_per_partition=_INTER,
                params_dtype=torch.bfloat16,
                with_bias=True,
            )
        self.assertIn("bias", str(cm.exception))

    # ---- process_weights_after_loading ----

    def test_process_weights_after_loading_repacks_both(self):
        calls = []

        def fake_preprocess(weight):
            calls.append(tuple(weight.shape))
            return weight

        fake_dg = SimpleNamespace(preprocess_a8w4_weight=fake_preprocess)
        layer = torch.nn.Module()
        layer.register_parameter(
            "w13_weight",
            torch.nn.Parameter(
                torch.zeros((_E, _INTER2, _K // 2), dtype=torch.uint8),
                requires_grad=False,
            ),
        )
        layer.register_parameter(
            "w2_weight",
            torch.nn.Parameter(
                torch.zeros((_E, _K, _INTER // 2), dtype=torch.uint8),
                requires_grad=False,
            ),
        )
        layer.register_parameter(
            "w13_weight_scale",
            torch.nn.Parameter(
                torch.ones((_E, _INTER2, 1), dtype=torch.float32),
                requires_grad=False,
            ),
        )
        layer.register_parameter(
            "w2_weight_scale",
            torch.nn.Parameter(
                torch.ones((_E, _K, 1), dtype=torch.float32), requires_grad=False
            ),
        )
        method = W4A8Int8MoEMethod.__new__(W4A8Int8MoEMethod)
        with patch.dict("sys.modules", {"deep_gemm": fake_dg}):
            W4A8Int8MoEMethod.process_weights_after_loading(method, layer)

        self.assertEqual(calls, [(_E, _INTER2, _K // 2), (_E, _K, _INTER // 2)])
        self.assertEqual(layer.w13_weight.dtype, torch.uint8)
        self.assertEqual(layer.w13_weight_scale.dtype, torch.float32)

    # ---- create_moe_runner guard ----

    def test_create_moe_runner_requires_deepgemm(self):
        method = W4A8Int8MoEMethod.__new__(W4A8Int8MoEMethod)
        with patch.object(deep_gemm_wrapper, "ENABLE_JIT_DEEPGEMM", False):
            with self.assertRaises(AssertionError) as cm:
                method.create_moe_runner(
                    torch.nn.Module(), MoeRunnerConfig(activation="silu")
                )
        self.assertIn("JIT DeepGEMM", str(cm.exception))

    # ---- apply() ----

    def test_apply_builds_int4_quant_info(self):
        method = W4A8Int8MoEMethod.__new__(W4A8Int8MoEMethod)
        method.runner = Mock()
        dispatch_output = SimpleNamespace(hidden_states=torch.zeros((1, _K)))
        layer = SimpleNamespace(
            w13_weight=torch.empty((_E, _INTER2, _K // 2), dtype=torch.uint8),
            w2_weight=torch.empty((_E, _K, _INTER // 2), dtype=torch.uint8),
            w13_weight_scale=torch.empty((_E, _INTER2, 1)),
            w2_weight_scale=torch.empty((_E, _K, 1)),
        )
        W4A8Int8MoEMethod.apply(method, layer, dispatch_output)

        _, (runner_output, quant_info) = method.runner.run.call_args
        self.assertIs(runner_output, dispatch_output)
        self.assertIs(quant_info.use_int4_w4a8, True)
        self.assertIs(quant_info.per_channel_quant, True)
        self.assertIsNone(quant_info.block_shape)
        self.assertIs(quant_info.w13_scale, layer.w13_weight_scale)
        self.assertIs(quant_info.w2_scale, layer.w2_weight_scale)

    # ---- nopad runner path ----

    def test_nopad_gemm_requires_int8_activations(self):
        runner_input = DeepGemmRunnerInput(
            hidden_states=torch.zeros((3, _K), dtype=torch.bfloat16),
            hidden_states_scale=torch.ones((3, 1)),
            use_masked_gemm=False,
            m_indices=torch.zeros((3,), dtype=torch.int32),
        )
        with self.assertRaises(AssertionError) as cm:
            _runner()._run_int8_int4_nopad_gemm(
                runner_input, _int8_quant_info(), self._running_state(3)
            )
        self.assertIn("INT8 activations", str(cm.exception))

    def test_nopad_gemm_calls_i8i4_kernels_twice(self):
        from sglang.kernels.ops.quantization import int8_kernel

        hidden_states = torch.zeros((3, _K), dtype=torch.int8)
        hidden_states_scale = torch.full((3, 1), 3.0)
        quant_info = _int8_quant_info()
        runner_input = DeepGemmRunnerInput(
            hidden_states=hidden_states,
            hidden_states_scale=hidden_states_scale,
            use_masked_gemm=False,
            m_indices=torch.zeros((3,), dtype=torch.int32),
        )
        down_q = torch.full((3, _INTER), 5, dtype=torch.int8)
        down_s = torch.full((3, 1), 7.0)

        def fake_gemm(lhs, rhs, out, *args, **kwargs):
            # Deterministic gateup activations for the reference silu below.
            out.copy_(torch.linspace(-2, 2, out.numel()).reshape(out.shape))

        gemm = Mock(side_effect=fake_gemm)
        with (
            patch.object(deep_gemm_wrapper, "grouped_gemm_nt_i8i4bf16_nopad", gemm),
            # _legacy_silu_and_mul is CUDA-only; use the CPU reference impl.
            patch.object(
                runner_module, "_legacy_silu_and_mul", _reference_silu_and_mul
            ),
            patch.object(
                int8_kernel, "per_token_quant_int8", return_value=(down_q, down_s)
            ) as quantize,
        ):
            output = _runner()._run_int8_int4_nopad_gemm(
                runner_input, quant_info, self._running_state(3)
            )

        # The activation buffer reaching the down-projection quant must equal
        # silu(gate) * up on the GEMM-1 output.
        gateup = torch.linspace(-2, 2, 3 * _INTER2).reshape(3, _INTER2)
        expected_down = torch.empty((3, _INTER))
        _reference_silu_and_mul(gateup, expected_down)
        quantized_input = quantize.call_args.args[0]
        torch.testing.assert_close(
            quantized_input.float(), expected_down, rtol=1e-4, atol=1e-5
        )
        self.assertEqual(quantize.call_count, 1)  # only for the down projection input
        self.assertEqual(output.shape, (3, _K))
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual(gemm.call_count, 2)

        lhs0, rhs0, out0, m_indices0, m_rows0 = gemm.call_args_list[0].args
        self.assertIs(lhs0[0], hidden_states)
        self.assertIs(lhs0[1], hidden_states_scale)
        self.assertIs(rhs0[0], quant_info.w13_weight)
        self.assertIs(rhs0[1], quant_info.w13_scale)
        self.assertEqual(out0.shape, (3, _INTER2))
        self.assertIs(m_indices0, runner_input.m_indices)
        self.assertIsNone(m_rows0)

        lhs1, rhs1, out1 = gemm.call_args_list[1].args[:3]
        self.assertIs(lhs1[0], down_q)
        self.assertIs(lhs1[1], down_s)
        self.assertIs(rhs1[0], quant_info.w2_weight)
        self.assertIs(rhs1[1], quant_info.w2_scale)
        self.assertIs(out1, output)

    # ---- masked runner path ----

    def test_masked_gemm_requires_int8_activations(self):
        runner_input = DeepGemmRunnerInput(
            hidden_states=torch.zeros((2, 4, _K), dtype=torch.bfloat16),
            hidden_states_scale=torch.ones((2, 4, 1)),
            use_masked_gemm=True,
            masked_m=torch.tensor([1, 2], dtype=torch.int32),
            expected_m=2,
        )
        with self.assertRaises(AssertionError) as cm:
            _runner()._run_masked_int8_int4_gemm(runner_input, _int8_quant_info(), {})
        self.assertIn("INT8 activations", str(cm.exception))

    def test_masked_gemm_calls_i8i4_kernels_twice(self):
        from sglang.kernels.ops.moe import ep_moe_kernels

        num_groups, m = 2, 4
        hidden_states = torch.zeros((num_groups, m, _K), dtype=torch.int8)
        hidden_states_scale = torch.full((num_groups, m, 1), 3.0)
        masked_m = torch.tensor([1, 2], dtype=torch.int32)
        quant_info = _int8_quant_info()
        runner_input = DeepGemmRunnerInput(
            hidden_states=hidden_states,
            hidden_states_scale=hidden_states_scale,
            use_masked_gemm=True,
            masked_m=masked_m,
            expected_m=2,
        )

        def fake_post_quant(gateup, out_q, out_s, scale_block_size, *args, **kwargs):
            out_q.fill_(5)
            out_s.fill_(7.0)
            fake_post_quant.scale_block_size = scale_block_size
            self.assertFalse(kwargs["use_fp8"])
            self.assertTrue(kwargs["use_int8"])

        fake_post_quant.scale_block_size = None
        gemm = Mock()
        with (
            patch.object(deep_gemm_wrapper, "grouped_gemm_nt_i8i4bf16_masked", gemm),
            patch.object(deep_gemm_wrapper, "DEEPGEMM_SCALE_UE8M0", False),
            patch.object(
                ep_moe_kernels,
                "silu_and_mul_masked_post_quant_fwd",
                side_effect=fake_post_quant,
            ) as post_quant,
        ):
            output = _runner()._run_masked_int8_int4_gemm(runner_input, quant_info, {})

        self.assertEqual(output.shape, (num_groups, m, _K))
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual(gemm.call_count, 2)

        lhs0, rhs0, out0, mm0, em0 = gemm.call_args_list[0].args
        self.assertIs(lhs0[0], hidden_states)
        self.assertIs(lhs0[1], hidden_states_scale)
        self.assertIs(rhs0[0], quant_info.w13_weight)
        self.assertIs(rhs0[1], quant_info.w13_scale)
        self.assertEqual(out0.shape, (num_groups, m, _INTER2))
        self.assertIs(mm0, masked_m)
        self.assertEqual(em0, 2)

        # Activation quant: int8 output with per-token scale block = N // 2.
        self.assertEqual(fake_post_quant.scale_block_size, _INTER)
        down_q = post_quant.call_args.args[1]
        self.assertEqual(down_q.shape, (num_groups, m, _INTER))
        self.assertEqual(down_q.dtype, torch.int8)

        lhs1, rhs1, out1, mm1, em1 = gemm.call_args_list[1].args
        self.assertEqual(lhs1[0].dtype, torch.int8)
        self.assertEqual(lhs1[0].shape, (num_groups, m, _INTER))
        self.assertIs(rhs1[0], quant_info.w2_weight)
        self.assertIs(rhs1[1], quant_info.w2_scale)
        self.assertIs(out1, output)
        self.assertIs(mm1, masked_m)
        self.assertEqual(em1, 2)

    @staticmethod
    def _running_state(all_tokens):
        return {
            "all_tokens": all_tokens,
            "hidden_states_device": torch.device("cpu"),
            "hidden_states_shape": (all_tokens, _K),
            "m_rows": None,
        }


if __name__ == "__main__":
    unittest.main(verbosity=2)
