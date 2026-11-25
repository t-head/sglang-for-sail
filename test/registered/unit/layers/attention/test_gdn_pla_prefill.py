"""Protect PLA's indexed V,K layout and final-state versus checkpoint contract."""

import sys
import unittest
from types import ModuleType
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.linear.kernels import gdn_pla
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestPLAPrefillContract(CustomTestCase):
    def test_padding_rows_start_zero_and_never_write_real_slots(self):
        for dtype in (torch.float32, torch.bfloat16):
            for strided in (False, True):
                for slot_values in ([1, -1, -1], [0, -1, -1], [-1, -1, -1]):
                    with self.subTest(dtype=dtype, strided=strided, slots=slot_values):
                        state = (
                            torch.arange(4 * 2 * 4 * 3).reshape(4, 2, 4, 3).to(dtype)
                        )
                        if strided:
                            storage = torch.empty(4, 2, 4, 3, 2, dtype=dtype)
                            storage[..., 0] = state
                            state = storage[..., 0]
                        state[0].zero_()
                        before = state.clone()
                        slots = torch.tensor(slot_values)
                        cu = torch.tensor([0, 2, 3, 5])
                        q = torch.randn(1, 5, 2, 3)
                        v = torch.randn(1, 5, 2, 4)
                        gates = torch.randn(1, 5, 2)

                        def forward(**kw):
                            rows = kw["initial_state"][kw["initial_state_indices"]]
                            valid = slots > 0
                            torch.testing.assert_close(
                                rows[~valid], torch.zeros_like(rows[~valid])
                            )
                            torch.testing.assert_close(
                                rows[valid], before[slots[valid]]
                            )
                            kw["initial_state"][kw["initial_state_indices"]] += 2
                            return None, None, v, None, None

                        module = ModuleType("pla.prefill.flashqla")
                        module.chunk_gated_delta_rule_fwd = forward
                        with patch.dict(
                            sys.modules,
                            {
                                "pla": ModuleType("pla"),
                                "pla.prefill": ModuleType("pla.prefill"),
                                "pla.prefill.flashqla": module,
                            },
                        ):
                            gdn_pla.pla_gdn_prefill(
                                q,
                                q,
                                v,
                                gates,
                                gates,
                                ssm_states=state,
                                cache_indices=slots,
                                query_start_loc=cu,
                                raw_cu_seqlens=(0, 2, 3, 5),
                            )
                        expected = before.clone()
                        valid = slots > 0
                        expected[slots[valid]] += 2
                        torch.testing.assert_close(state, expected, atol=0, rtol=0)

    def test_indexed_state_layout_precision_and_untouched_slots(self):
        # Rectangular states expose accidental use of PLA's legacy K,V layout.
        for dtype in (torch.float32, torch.bfloat16):
            for strided in (False, True):
                with self.subTest(dtype=dtype, strided=strided):
                    state = torch.arange(5 * 2 * 4 * 3).reshape(5, 2, 4, 3).to(dtype)
                    if strided:
                        storage = torch.empty(5, 2, 4, 3, 2, dtype=dtype)
                        storage[..., 0] = state
                        state = storage[..., 0]
                    before = state.clone()
                    slots = torch.tensor([3, 1], dtype=torch.int64)
                    cu = torch.tensor([0, 3, 5], dtype=torch.int64)
                    q = torch.randn(1, 5, 2, 6)[..., ::2]
                    g = torch.randn(1, 5, 2)
                    expected_output = torch.randn(1, 5, 2, 4)

                    def forward(**kw):
                        self.assertEqual(kw["raw_cu_seqlens"], (0, 3, 5))
                        self.assertTrue(kw["normalize_qk"])
                        self.assertFalse(kw["output_final_state"])
                        self.assertEqual(kw["final_state_dtype"], dtype)
                        self.assertEqual(kw["initial_state"].shape[-2:], (4, 3))
                        for name in ("q", "k", "v", "g", "beta", "initial_state"):
                            self.assertTrue(kw[name].is_contiguous(), name)
                        for name in ("cu_seqlens", "initial_state_indices"):
                            self.assertEqual(kw[name].dtype, torch.int32)
                        torch.testing.assert_close(kw["q"], q)
                        indices = kw["initial_state_indices"]
                        torch.testing.assert_close(
                            kw["initial_state"][indices], before[slots]
                        )
                        kw["initial_state"][indices] += 2
                        return None, None, expected_output, None, object()

                    module = ModuleType("pla.prefill.flashqla")
                    module.chunk_gated_delta_rule_fwd = forward
                    with patch.dict(
                        sys.modules,
                        {
                            "pla": ModuleType("pla"),
                            "pla.prefill": ModuleType("pla.prefill"),
                            "pla.prefill.flashqla": module,
                        },
                    ):
                        out, final, checkpoints = gdn_pla.pla_gdn_prefill(
                            q,
                            q,
                            q,
                            g,
                            g,
                            ssm_states=state,
                            cache_indices=slots,
                            query_start_loc=cu,
                            raw_cu_seqlens=(0, 3, 5),
                        )
                    self.assertIs(out, expected_output)
                    self.assertIsNone(final)
                    self.assertIsNone(checkpoints)
                    expected = before.clone()
                    expected[slots] += 2
                    torch.testing.assert_close(state, expected, atol=0, rtol=0)

    def test_advertised_shape_and_dtype_admission(self):
        q = torch.randn(1, 5, 4, 128, dtype=torch.bfloat16)
        v = torch.randn(1, 5, 16, 128, dtype=torch.bfloat16)
        g = torch.zeros(1, 5, 16)
        state = torch.zeros(4, 16, 128, 128)
        ops = ModuleType("pla.prefill.flashqla.ops")
        ops.SUPPORTED_HEAD_CONFIGS = {(16, 4)}
        with patch.dict(sys.modules, {"pla.prefill.flashqla.ops": ops}):
            self.assertTrue(gdn_pla.pla_gdn_prefill_supported(q, q, v, g, g, state))
            self.assertFalse(
                gdn_pla.pla_gdn_prefill_supported(q, q, v, g.half(), g, state)
            )
            self.assertFalse(
                gdn_pla.pla_gdn_prefill_supported(q, q, v, g, g, state.half())
            )
            self.assertFalse(
                gdn_pla.pla_gdn_prefill_supported(
                    q[..., :64], q[..., :64], v, g, g, state
                )
            )
            ops.SUPPORTED_HEAD_CONFIGS = {(8, 4)}
            self.assertFalse(gdn_pla.pla_gdn_prefill_supported(q, q, v, g, g, state))

    def test_padding_io_does_not_read_tensor_values_on_host(self):
        state = torch.arange(4 * 8, dtype=torch.float32).view(4, 1, 2, 4)
        state[0].zero_()
        before = state.clone()
        slots = torch.tensor([2, -1, 0])
        q = torch.zeros(1, 3, 1, 4)
        v = torch.zeros(1, 3, 1, 2)
        gates = torch.zeros(1, 3, 1)
        module = ModuleType("pla.prefill.flashqla")
        seen = {}

        def forward(**kw):
            seen.update(kw)
            kw["initial_state"].add_(1)
            return None, None, v, None, None

        module.chunk_gated_delta_rule_fwd = forward
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
            output = gdn_pla.pla_gdn_prefill(
                q,
                q,
                v,
                gates,
                gates,
                ssm_states=state,
                cache_indices=slots,
                query_start_loc=torch.tensor([0, 1, 2, 3]),
                raw_cu_seqlens=(0, 1, 2, 3),
                normalize_qk=False,
            )[0]
        self.assertIs(output, v)
        self.assertFalse(seen["normalize_qk"])
        expected = before.clone()
        expected[2] += 1
        torch.testing.assert_close(state, expected, atol=0, rtol=0)

    def test_v1_dispatch_preserves_fallbacks_and_normalization(self):
        from contextlib import ExitStack

        from sglang.srt.environ import envs
        from sglang.srt.layers.attention.linear.kernels import gdn_triton

        q = torch.zeros(1, 5, 1, 128, dtype=torch.bfloat16)
        v = torch.zeros(1, 5, 2, 128, dtype=torch.bfloat16)
        g = torch.zeros(1, 5, 2)
        pool = torch.zeros(3, 2, 128, 128)
        cu = torch.tensor([0, 5], dtype=torch.int32)
        slots = torch.tensor([1], dtype=torch.int32)
        ops = ModuleType("pla.prefill.flashqla.ops")
        ops.SUPPORTED_HEAD_CONFIGS = {(16, 4)}
        kernel = gdn_triton.TritonGDNKernel()
        args = dict(
            ssm_states=pool, cache_indices=slots, query_start_loc=cu, seq_lens_cpu=[5]
        )
        with ExitStack() as stack:
            stack.enter_context(envs.SGLANG_SAIL_PLA_CUDA.override(True))
            stack.enter_context(patch.object(gdn_triton, "is_sail", return_value=True))
            stack.enter_context(patch.object(gdn_triton, "is_cpu", return_value=False))
            stack.enter_context(patch.object(gdn_triton, "is_npu", return_value=False))
            stack.enter_context(
                patch.object(torch.Tensor, "is_cuda", property(lambda _: True))
            )
            stack.enter_context(
                patch.object(
                    torch.cuda, "is_current_stream_capturing", return_value=False
                )
            )
            stack.enter_context(
                patch.dict(sys.modules, {"pla.prefill.flashqla.ops": ops})
            )
            prepare = stack.enter_context(
                patch.object(kernel, "_prepare_triton_capture")
            )
            pla = stack.enter_context(
                patch.object(gdn_pla, "pla_gdn_prefill", return_value=(v, None, None))
            )
            triton = stack.enter_context(
                patch.object(
                    gdn_triton,
                    "chunk_gated_delta_rule",
                    return_value=(v, None, "checkpoint"),
                )
            )
            self.assertEqual(kernel.extend(q, q, v, g, g, **args)[2], "checkpoint")
            pla.assert_not_called()
            prepare.assert_not_called()
            ops.SUPPORTED_HEAD_CONFIGS = {(2, 1)}
            for extra in (
                {"out": v, "output_indices": torch.arange(5)},
                {"chunk_indices": torch.zeros(1, 2), "chunk_offsets": cu},
                {"needs_track_state": True},
                {"requires_intermediate_states": True},
                {"inplace_update": False},
                {"seq_lens_cpu": [4]},
            ):
                with self.subTest(extra=list(extra)):
                    self.assertEqual(
                        kernel.extend(q, q, v, g, g, **(args | extra))[2], "checkpoint"
                    )
            pla.assert_not_called()
            prepare.assert_not_called()
            kernel.extend(q, q, v, g, g, **args, use_qk_l2norm_in_kernel=False)
            self.assertFalse(pla.call_args.kwargs["normalize_qk"])
            self.assertEqual(prepare.call_args.args[-1], False)
            self.assertIs(triton.call_args.kwargs["cu_seqlens"], cu)

    def test_host_offsets_reject_stale_or_padded_batch(self):
        cu = torch.empty(3, dtype=torch.int32)
        self.assertEqual(gdn_pla.pla_host_offsets([3, 2], cu, 5), (0, 3, 5))
        self.assertEqual(
            gdn_pla.pla_host_offsets(torch.tensor([3, 2]), cu, 5), (0, 3, 5)
        )
        for lengths, tokens in (
            ([3, 2], 8),
            ([5], 5),
            ([3, -1], 2),
            ([3, 2.0], 5),
            ([True, 4], 5),
            (None, 5),
        ):
            with self.subTest(lengths=lengths, tokens=tokens):
                self.assertIsNone(gdn_pla.pla_host_offsets(lengths, cu, tokens))


if __name__ == "__main__":
    unittest.main()
