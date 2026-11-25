"""Packed GDN raw-gating, aliasing, state and semantic-fallback contracts."""

import sys
import unittest
from contextlib import ExitStack
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.attention.linear.kernels import gdn_pla
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


def inputs(tokens=5, hq=4, hv=16):
    width = (2 * hq + hv) * 128
    # Both storage offset and token pitch differ from an ordinary dense tensor.
    packed = torch.randn(tokens, width + 16, dtype=torch.bfloat16)[:, 8 : 8 + width]
    b, a = torch.randn(tokens, 2 * hv, dtype=packed.dtype).split(hv, dim=-1)
    return packed, a, b, torch.randn(hv), torch.randn(hv)


class TestPLAPrefillContract(CustomTestCase):
    def test_raw_inputs_are_aliases_and_strided_pool_writes_back(self):
        packed, a, b, a_log, bias = inputs()
        expected = torch.randn(1, 5, 16, 128, dtype=packed.dtype)
        for dtype in (torch.float32, torch.bfloat16):
            for strided in (False, True):
                for values in ([3, 1], [0, -1], [3, -1]):
                    with self.subTest(dtype=dtype, strided=strided, slots=values):
                        pool = torch.randn(5, 16, 128, 128, dtype=dtype)
                        pool[0].zero_()
                        if strided:
                            backing = torch.empty(*pool.shape, 2, dtype=dtype)
                            backing[..., 0] = pool
                            pool = backing[..., 0]
                        before = pool.clone()
                        original = [x.clone() for x in (packed, a, b, a_log, bias)]
                        slots = torch.tensor(values)

                        def forward(**kw):
                            for key, value in (
                                ("packed_qkv", packed),
                                ("raw_gate_a", a),
                                ("raw_beta_b", b),
                                ("A_log", a_log),
                                ("dt_bias", bias),
                            ):
                                self.assertIs(kw[key], value)
                            self.assertNotIn("g", kw)
                            self.assertNotIn("beta", kw)
                            self.assertNotIn("normalize_qk", kw)
                            self.assertEqual(kw["num_qk_heads"], 4)
                            self.assertEqual(kw["num_v_heads"], 16)
                            self.assertEqual(kw["raw_cu_seqlens"], (0, 3, 5))
                            self.assertFalse(kw["output_final_state"])
                            self.assertEqual(kw["final_state_dtype"], dtype)
                            self.assertTrue(kw["initial_state"].is_contiguous())
                            torch.testing.assert_close(
                                kw["initial_state"][slots < 0],
                                torch.zeros_like(kw["initial_state"][slots < 0]),
                            )
                            for key in ("initial_state_indices", "cu_seqlens"):
                                self.assertEqual(kw[key].dtype, torch.int32)
                                self.assertTrue(kw[key].is_contiguous())
                            kw["initial_state"].add_(2)
                            return None, None, expected, None, None

                        module = ModuleType("pla.prefill.flashqla")
                        module.gdn_prefill_packed_qkv_fwd = forward
                        with patch.dict(sys.modules, {"pla.prefill.flashqla": module}):
                            result = gdn_pla.pla_gdn_prefill(
                                packed,
                                a,
                                b,
                                A_log=a_log,
                                dt_bias=bias,
                                num_qk_heads=4,
                                num_v_heads=16,
                                ssm_states=pool,
                                cache_indices=slots,
                                query_start_loc=torch.tensor([0, 3, 5]),
                                raw_cu_seqlens=(0, 3, 5),
                            )
                        self.assertIs(result[0], expected)
                        self.assertEqual(result[1:], (None, None))
                        live = slots >= 0
                        before[slots[live]] += 2
                        torch.testing.assert_close(pool, before, atol=0, rtol=0)
                        for x, copy in zip((packed, a, b, a_log, bias), original):
                            torch.testing.assert_close(x, copy, atol=0, rtol=0)

    def test_transposed_state_pool_is_packed_before_native_entry(self):
        packed, a, b, a_log, bias = inputs()
        pool = torch.randn(3, 16, 128, 128).transpose(-1, -2)
        pool[0].zero_()
        before = pool.clone()
        self.assertFalse(pool.is_contiguous())
        module = ModuleType("pla.prefill.flashqla")

        def native(**kw):
            scratch = kw["initial_state"]
            self.assertTrue(scratch.is_contiguous())
            torch.testing.assert_close(scratch[0], before[1], atol=0, rtol=0)
            scratch.add_(1)
            return None, None, packed, None, None

        module.gdn_prefill_packed_qkv_fwd = native
        with patch.dict(sys.modules, {"pla.prefill.flashqla": module}):
            gdn_pla.pla_gdn_prefill(
                packed,
                a,
                b,
                A_log=a_log,
                dt_bias=bias,
                num_qk_heads=4,
                num_v_heads=16,
                ssm_states=pool,
                cache_indices=torch.tensor([1]),
                query_start_loc=torch.tensor([0, 5]),
                raw_cu_seqlens=(0, 5),
            )
        before[1] += 1
        torch.testing.assert_close(pool, before, atol=0, rtol=0)
        self.assertFalse(pool.is_contiguous())

    def test_parameter_promotion_matches_original_fp32_gate_arithmetic(self):
        packed, a, b, a_log, bias = inputs()
        a_log = a_log.bfloat16()
        bias = bias.half()
        pool = torch.zeros(2, 16, 128, 128, dtype=torch.bfloat16)
        module = ModuleType("pla.prefill.flashqla")
        seen = {}

        def native(**kw):
            seen.update(kw)
            return None, None, packed, None, None

        module.gdn_prefill_packed_qkv_fwd = native
        with patch.dict(sys.modules, {"pla.prefill.flashqla": module}):
            gdn_pla.pla_gdn_prefill(
                packed,
                a,
                b,
                A_log=a_log,
                dt_bias=bias,
                num_qk_heads=4,
                num_v_heads=16,
                ssm_states=pool,
                cache_indices=torch.tensor([1]),
                query_start_loc=torch.tensor([0, 5]),
                raw_cu_seqlens=(0, 5),
            )
        self.assertEqual(seen["A_log"].dtype, torch.float32)
        self.assertEqual(seen["dt_bias"].dtype, torch.float32)
        torch.testing.assert_close(seen["A_log"], a_log.float(), atol=0, rtol=0)
        torch.testing.assert_close(seen["dt_bias"], bias.float(), atol=0, rtol=0)
        self.assertEqual(a_log.dtype, torch.bfloat16)
        self.assertEqual(bias.dtype, torch.float16)
        self.assertEqual(pool.dtype, torch.bfloat16)
        self.assertIs(seen["raw_gate_a"], a)
        self.assertIs(seen["raw_beta_b"], b)

    def test_admission_enforces_packed_layout_and_raw_gate_dtypes(self):
        packed, a, b, a_log, bias = inputs()
        pool = torch.zeros(3, 16, 128, 128)
        ops = ModuleType("pla.prefill.flashqla.ops")
        ops.SUPPORTED_HEAD_CONFIGS = {(16, 4)}
        base = dict(
            packed_qkv=packed,
            a=a,
            b=b,
            A_log=a_log,
            dt_bias=bias,
            ssm_states=pool,
            num_qk_heads=4,
            num_v_heads=16,
        )
        with patch.dict(sys.modules, {"pla.prefill.flashqla.ops": ops}):
            self.assertTrue(gdn_pla.pla_gdn_prefill_supported(**base))
            for update in (
                {"packed_qkv": packed.float()},
                {"packed_qkv": packed[:, :-1]},
                {
                    "packed_qkv": torch.empty(
                        5, packed.shape[1] * 2, dtype=packed.dtype
                    )[:, ::2]
                },
                {"packed_qkv": packed[:1].expand(5, -1)},
                {"a": a.float()},
                {"b": b.float()},
                {"a": a[:1]},
                {"a": torch.zeros(5, 32, dtype=a.dtype)[:, ::2]},
                {"dt_bias": bias.double()},
                {"A_log": a_log.view(1, -1)},
                {"ssm_states": pool.half()},
                {"num_qk_heads": 0},
            ):
                with self.subTest(update=list(update)):
                    self.assertFalse(
                        gdn_pla.pla_gdn_prefill_supported(**(base | update))
                    )
            ops.SUPPORTED_HEAD_CONFIGS = {(8, 4)}
            self.assertFalse(gdn_pla.pla_gdn_prefill_supported(**base))

    def test_state_staging_does_not_read_device_values_on_host(self):
        packed, a, b, a_log, bias = inputs()
        pool = torch.zeros(3, 16, 128, 128)
        module = ModuleType("pla.prefill.flashqla")
        module.gdn_prefill_packed_qkv_fwd = lambda **kw: (
            None,
            None,
            packed,
            None,
            None,
        )
        with (
            patch.dict(sys.modules, {"pla.prefill.flashqla": module}),
            patch.object(
                torch.Tensor, "__bool__", side_effect=AssertionError("host bool")
            ),
            patch.object(torch.Tensor, "item", side_effect=AssertionError("host item")),
            patch.object(
                torch.Tensor, "tolist", side_effect=AssertionError("host list")
            ),
        ):
            gdn_pla.pla_gdn_prefill(
                packed,
                a,
                b,
                A_log=a_log,
                dt_bias=bias,
                num_qk_heads=4,
                num_v_heads=16,
                ssm_states=pool,
                cache_indices=torch.tensor([2, -1]),
                query_start_loc=torch.tensor([0, 3, 5]),
                raw_cu_seqlens=(0, 3, 5),
            )

    def test_host_offsets_reject_empty_stale_and_padded_batches(self):
        cu = torch.empty(3, dtype=torch.int32)
        self.assertEqual(gdn_pla.pla_host_offsets([3, 2], cu, 5), (0, 3, 5))
        self.assertEqual(
            gdn_pla.pla_host_offsets(torch.tensor([3, 2]), cu, 5), (0, 3, 5)
        )
        for lengths, offsets, tokens in (
            ([], torch.empty(1), 0),
            ([5], torch.empty(1), 5),
            ([3, 2], cu, 8),
            ([5], cu, 5),
            ([3, -1], cu, 2),
            ([3, 2.0], cu, 5),
            ([True, 4], cu, 5),
            (None, cu, 5),
        ):
            self.assertIsNone(gdn_pla.pla_host_offsets(lengths, offsets, tokens))

    def test_dispatch_rejects_semantic_fallbacks_and_preserves_aliases(self):
        from sglang.srt.environ import envs
        from sglang.srt.layers.attention.linear.kernels import gdn_triton

        packed, a, b, a_log, bias = inputs()
        kernel = gdn_triton.TritonGDNKernel()
        args = dict(
            A_log=a_log,
            dt_bias=bias,
            num_qk_heads=4,
            num_v_heads=16,
            ssm_states=torch.zeros(3, 16, 128, 128),
            cache_indices=torch.tensor([1]),
            query_start_loc=torch.tensor([0, 5]),
            seq_lens_cpu=[5],
        )
        ops = ModuleType("pla.prefill.flashqla.ops")
        ops.SUPPORTED_HEAD_CONFIGS = {(16, 4)}
        with ExitStack() as stack:
            stack.enter_context(envs.SGLANG_SAIL_PLA_CUDA.override(True))
            stack.enter_context(patch.object(gdn_triton, "is_sail", return_value=True))
            stack.enter_context(
                patch.object(torch.Tensor, "is_cuda", property(lambda _: True))
            )
            capture = stack.enter_context(
                patch.object(
                    torch.cuda, "is_current_stream_capturing", return_value=False
                )
            )
            stack.enter_context(
                patch.dict(sys.modules, {"pla.prefill.flashqla.ops": ops})
            )
            warm = stack.enter_context(patch.object(kernel, "_prepare_triton_capture"))
            native = stack.enter_context(
                patch.object(
                    gdn_pla, "pla_gdn_prefill", return_value=(packed, None, None)
                )
            )
            for extra in (
                {"use_qk_l2norm_in_kernel": False},
                {"inplace_update": False},
                {"requires_intermediate_states": True},
                {"needs_track_state": True},
                {"track_state": object()},
                {"num_state_checkpoints": 1},
                {"seq_lens_cpu": [4]},
                {"out": object()},
                {"output_indices": object()},
                {"chunk_indices": object()},
                {"query_start_loc": torch.tensor([0])},
                {"a": a.float()},
            ):
                with self.subTest(extra=list(extra)):
                    call_args = args | extra
                    raw_a = call_args.pop("a", a)
                    self.assertIsNone(
                        kernel.packed_prefill(packed, raw_a, b, **call_args)
                    )
            capture.return_value = True
            self.assertIsNone(kernel.packed_prefill(packed, a, b, **args))
            capture.return_value = False
            with envs.SGLANG_SAIL_PLA_CUDA.override(False):
                self.assertIsNone(kernel.packed_prefill(packed, a, b, **args))
            native.assert_not_called()
            warm.assert_not_called()
            self.assertIs(kernel.packed_prefill(packed, a, b, **args)[0], packed)
            self.assertIs(native.call_args.args[0], packed)
            self.assertIs(native.call_args.args[1], a)
            self.assertIs(native.call_args.args[2], b)
            q, k, v = warm.call_args.args[:3]
            self.assertEqual(q.data_ptr(), packed.data_ptr())
            self.assertEqual(
                k.data_ptr(), packed.data_ptr() + 4 * 128 * packed.element_size()
            )
            self.assertEqual(
                v.data_ptr(), packed.data_ptr() + 8 * 128 * packed.element_size()
            )
            self.assertEqual(q.stride()[1:], (packed.stride(0), 128, 1))

    def test_srt_routes_post_conv_buffer_before_split_and_gating(self):
        from sglang.srt.layers.attention.linear.gdn_backend import (
            GDNAttnBackend,
            GDNKernelDispatcher,
        )
        from sglang.srt.layers.attention.mamba.mamba2_metadata import ForwardMetadata
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        hq, hv, tokens = 4, 16, 5
        packed, a, b, a_log, bias = inputs(tokens, hq, hv)
        convolved = packed + 1
        layer = SimpleNamespace(
            layer_id=0,
            q_dim=hq * 128,
            k_dim=hq * 128,
            v_dim=hv * 128,
            num_q_heads=hq,
            num_k_heads=hq,
            num_v_heads=hv,
            head_q_dim=128,
            head_k_dim=128,
            head_v_dim=128,
            A_log=a_log,
            dt_bias=bias,
        )
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            extend_seq_lens_cpu=[tokens],
            extend_prefix_lens=torch.tensor([0]),
        )
        pool = torch.zeros(3, hv, 128, 128)
        backend = object.__new__(GDNAttnBackend)
        backend.forward_metadata = ForwardMetadata(
            query_start_loc=torch.tensor([0, tokens]),
            mamba_cache_indices=torch.tensor([1]),
        )
        backend.mis_metadata = None
        backend.requires_contiguous_prefill_state = False
        backend.req_to_token_pool = SimpleNamespace(
            mamba2_layer_cache=lambda _: SimpleNamespace(
                conv=[torch.zeros(3, packed.shape[1], 2)], temporal=pool
            )
        )
        backend._convolve_prefill = Mock(return_value=convolved)
        backend._prefill_gates = Mock(side_effect=AssertionError("duplicate gating"))
        output = torch.zeros(1, tokens, hv, 128, dtype=packed.dtype)
        packed_call = Mock(return_value=(output, None, None))
        backend.kernel_dispatcher = object.__new__(GDNKernelDispatcher)
        backend.kernel_dispatcher.extend_kernel = SimpleNamespace(
            packed_prefill=packed_call
        )
        with patch.object(
            torch, "split", side_effect=AssertionError("unnecessary split")
        ):
            self.assertIs(backend.forward_extend(layer, batch, packed, a, b), output)
        self.assertIs(packed_call.call_args.args[0], convolved)
        self.assertIs(packed_call.call_args.args[1], a)
        self.assertIs(packed_call.call_args.args[2], b)
        self.assertIs(packed_call.call_args.kwargs["A_log"], a_log)
        self.assertIs(packed_call.call_args.kwargs["dt_bias"], bias)
        self.assertIs(packed_call.call_args.kwargs["ssm_states"], pool)
        backend._prefill_gates.assert_not_called()


if __name__ == "__main__":
    unittest.main()
