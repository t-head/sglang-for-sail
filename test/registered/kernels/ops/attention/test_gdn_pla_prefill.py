"""PLA/Triton GDN parity with indexed states, chunk continuation and fallbacks."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.srt.utils import is_sail
from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.test_utils import CustomTestCase

register_ppu_ci(est_time=120, suite="stage-b-test-1-gpu-ppu")


@unittest.skipUnless(is_sail(), "PLA requires a PPU SAIL device")
class TestPLAPrefillNumerics(CustomTestCase):
    def test_negative_slots_preserve_pool_and_match_triton(self):
        torch.manual_seed(46)
        lengths = [65, 37, 23]
        total = sum(lengths)
        q, k = [
            torch.randn(1, total, 4, 128, device="cuda", dtype=torch.bfloat16)
            for _ in range(2)
        ]
        v = torch.randn(1, total, 12, 128, device="cuda", dtype=torch.bfloat16)
        g = -torch.rand(1, total, 12, device="cuda")
        beta = torch.rand_like(g)
        cu = torch.tensor([0, 65, 102, total], device="cuda", dtype=torch.int32)
        for dtype in (torch.float32, torch.bfloat16):
            for strided in (False, True):
                for values in ([3, -1, -1], [-1, -1, -1]):
                    with self.subTest(dtype=dtype, strided=strided, slots=values):
                        initial = (
                            torch.randn(5, 12, 128, 128, device="cuda", dtype=dtype)
                            * 0.02
                        )
                        reference = initial.clone()
                        if strided:
                            backing = torch.empty(
                                *initial.shape, 2, device="cuda", dtype=dtype
                            )
                            backing[..., 0] = initial
                            actual = backing[..., 0]
                        else:
                            actual = initial.clone()
                        slots = torch.tensor(values, device="cuda", dtype=torch.int32)
                        results = []
                        kernel = TritonGDNKernel()
                        for enabled, pool in zip((False, True), (reference, actual)):
                            with envs.SGLANG_SAIL_PLA_CUDA.override(enabled):
                                results.append(
                                    kernel.extend(
                                        q,
                                        k,
                                        v,
                                        g,
                                        beta,
                                        ssm_states=pool,
                                        cache_indices=slots,
                                        query_start_loc=cu,
                                        seq_lens_cpu=lengths,
                                    )[0]
                                )
                        torch.testing.assert_close(
                            results[0], results[1], atol=2e-2, rtol=1e-2
                        )
                        torch.testing.assert_close(
                            reference, actual, atol=2e-2, rtol=1e-2
                        )
                        untouched = torch.tensor(
                            [0, 1, 2, 4] if values[0] >= 0 else list(range(5)),
                            device="cuda",
                        )
                        torch.testing.assert_close(
                            actual[untouched], initial[untouched], atol=0, rtol=0
                        )

    def test_ragged_continuation_and_state_semantics(self):
        from pla.prefill import flashqla

        torch.manual_seed(42)
        kernel = TritonGDNKernel()
        lengths = [65, 37]
        total = sum(lengths)
        q, k = [
            torch.randn(1, total, 4, 128, device="cuda", dtype=torch.bfloat16)
            for _ in range(2)
        ]
        v = torch.randn(1, total, 16, 128, device="cuda", dtype=torch.bfloat16)
        g = -torch.rand(1, total, 16, device="cuda")
        beta = torch.rand_like(g)
        slots = torch.tensor([3, 1], device="cuda", dtype=torch.int64)
        cu = torch.tensor([0, 65, total], device="cuda", dtype=torch.int64)
        args = dict(cache_indices=slots, query_start_loc=cu, seq_lens_cpu=lengths)
        for dtype in (torch.float32, torch.bfloat16):
            for strided in (False, True):
                with self.subTest(dtype=dtype, strided=strided):
                    initial = (
                        torch.randn(5, 16, 128, 128, device="cuda", dtype=dtype) * 0.02
                    )
                    pools = []
                    for implementation in range(2):
                        # Triton's input_guard copies noncontiguous inputs and
                        # does not scatter states back. Compare PLA's strided
                        # pool with the contiguous Triton reference used by SRT.
                        if strided and implementation == 1:
                            backing = torch.empty(
                                *initial.shape, 2, device="cuda", dtype=dtype
                            )
                            backing[..., 0] = initial
                            pools.append(backing[..., 0])
                        else:
                            pools.append(initial.clone())
                    # Two successive chunks must consume the preceding final state.
                    for _ in range(2):
                        outputs = []
                        for enabled, pool in zip((False, True), pools):
                            with (
                                envs.SGLANG_SAIL_PLA_CUDA.override(enabled),
                                patch.object(
                                    flashqla,
                                    "chunk_gated_delta_rule_fwd",
                                    wraps=flashqla.chunk_gated_delta_rule_fwd,
                                ) as implementation,
                            ):
                                outputs.append(
                                    kernel.extend(
                                        q, k, v, g, beta, ssm_states=pool, **args
                                    )[0]
                                )
                                self.assertEqual(
                                    implementation.call_count, int(enabled)
                                )
                        torch.testing.assert_close(
                            outputs[0], outputs[1], atol=2e-2, rtol=1e-2
                        )
                        torch.testing.assert_close(
                            pools[0], pools[1], atol=2e-2, rtol=1e-2
                        )
                        torch.testing.assert_close(
                            pools[1][[0, 2, 4]], initial[[0, 2, 4]], atol=0, rtol=0
                        )

    def test_master_switch_controls_packed_decode(self):
        from pla import decode as pla_decode

        torch.manual_seed(46)
        kernel = TritonGDNKernel()
        hq, hv, dim, batch = 4, 12, 128, 2
        mixed_qkv = torch.randn(
            batch, (2 * hq + hv) * dim, device="cuda", dtype=torch.bfloat16
        )
        a, b = [
            torch.randn(batch, hv, device="cuda", dtype=torch.bfloat16)
            for _ in range(2)
        ]
        a_log = torch.randn(hv, device="cuda", dtype=torch.float32)
        dt_bias = torch.randn(hv, device="cuda", dtype=torch.bfloat16)
        slots = torch.tensor([3, 1], device="cuda", dtype=torch.int32)
        native = pla_decode.fused_sigmoid_gating_delta_rule_forward_k_last_packed
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                initial = (
                    torch.randn(5, hv, dim, dim, device="cuda", dtype=dtype) * 0.02
                )
                pools = [initial.clone(), initial.clone()]
                for _ in range(2):
                    outputs = []
                    for enabled, pool in zip((False, True), pools):
                        with (
                            envs.SGLANG_SAIL_PLA_CUDA.override(enabled),
                            patch.object(
                                pla_decode,
                                "fused_sigmoid_gating_delta_rule_forward_k_last_packed",
                                wraps=native,
                            ) as implementation,
                        ):
                            outputs.append(
                                kernel.packed_decode(
                                    mixed_qkv,
                                    a,
                                    b,
                                    A_log=a_log,
                                    dt_bias=dt_bias,
                                    scale=dim**-0.5,
                                    ssm_states=pool,
                                    cache_indices=slots,
                                    num_v_heads=hv,
                                    head_v_dim=dim,
                                )
                            )
                            self.assertEqual(implementation.call_count, int(enabled))
                    torch.testing.assert_close(
                        outputs[0], outputs[1], atol=2e-2, rtol=1e-2
                    )
                    torch.testing.assert_close(pools[0], pools[1], atol=2e-2, rtol=1e-2)
                    torch.testing.assert_close(
                        pools[1][[0, 2, 4]], initial[[0, 2, 4]], atol=0, rtol=0
                    )

    def test_fallbacks_preserve_readonly_and_checkpoint_states(self):
        torch.manual_seed(43)
        kernel = TritonGDNKernel()
        q, k = [
            torch.randn(1, 65, 4, 128, device="cuda", dtype=torch.bfloat16)
            for _ in range(2)
        ]
        v = torch.randn(1, 65, 16, 128, device="cuda", dtype=torch.bfloat16)
        g = -torch.rand(1, 65, 16, device="cuda")
        beta = torch.rand_like(g)
        state = torch.randn(3, 16, 128, 128, device="cuda") * 0.02
        slots = torch.tensor([1], device="cuda", dtype=torch.int32)
        cu = torch.tensor([0, 65], device="cuda", dtype=torch.int32)
        args = dict(cache_indices=slots, query_start_loc=cu, seq_lens_cpu=[65])
        from sglang.srt.layers.attention.linear.kernels import gdn_pla

        for extra in (
            {"inplace_update": False},
            {"requires_intermediate_states": True},
            {"seq_lens_cpu": [64]},
        ):
            with self.subTest(extra=extra):
                pools = [state.clone(), state.clone()]
                results = []
                for enabled, pool in zip((False, True), pools):
                    with envs.SGLANG_SAIL_PLA_CUDA.override(enabled):
                        # A PLA call on these paths is a semantic error, not a parity tolerance.
                        with patch.object(
                            gdn_pla,
                            "pla_gdn_prefill",
                            side_effect=AssertionError("unexpected PLA dispatch"),
                        ):
                            results.append(
                                kernel.extend(
                                    q, k, v, g, beta, ssm_states=pool, **(args | extra)
                                )
                            )
                torch.testing.assert_close(results[0][0], results[1][0], atol=0, rtol=0)
                torch.testing.assert_close(pools[0], pools[1], atol=0, rtol=0)
                if extra.get("inplace_update") is False:
                    torch.testing.assert_close(pools[1], state, atol=0, rtol=0)
                if extra.get("requires_intermediate_states"):
                    self.assertIsNotNone(results[1][2])
                    torch.testing.assert_close(
                        results[0][2], results[1][2], atol=0, rtol=0
                    )

    def test_graph_replay_uses_live_offsets_and_slots(self):
        torch.manual_seed(44)
        kernel = TritonGDNKernel()
        q, k = [
            torch.randn(1, 102, 4, 128, device="cuda", dtype=torch.bfloat16)
            for _ in range(2)
        ]
        v = torch.randn(1, 102, 16, 128, device="cuda", dtype=torch.bfloat16)
        g = -torch.rand(1, 102, 16, device="cuda")
        beta = torch.rand_like(g)
        initial = torch.randn(5, 16, 128, 128, device="cuda") * 0.02
        state = initial.clone()
        cu = torch.tensor([0, 65, 102], device="cuda", dtype=torch.int32)
        slots = torch.tensor([3, 1], device="cuda", dtype=torch.int32)

        def run(pool):
            return kernel.extend(
                q,
                k,
                v,
                g,
                beta,
                ssm_states=pool,
                cache_indices=slots,
                query_start_loc=cu,
                seq_lens_cpu=[65, 37],
            )[0]

        # A fresh cu tensor has no FLA metadata cache entries. Warm only with
        # PLA enabled, then immediately exercise the direct-capture fallback.
        graph = torch.cuda.CUDAGraph()
        with envs.SGLANG_SAIL_PLA_CUDA.override(True):
            run(state)
            with torch.cuda.graph(graph):
                captured = run(state)
        # Direct FLA capture has a fixed per-sequence chunk plan (2,1).
        # Change live lengths and state rows while retaining that plan. SRT's
        # breakable graph instead rebuilds its eager GDN plan per batch.
        for offsets, indices in (([0, 66, 102], [2, 4]), ([0, 67, 102], [1, 3])):
            cu.copy_(torch.tensor(offsets, device="cuda", dtype=torch.int32))
            slots.copy_(torch.tensor(indices, device="cuda", dtype=torch.int32))
            state.copy_(initial)
            reference = initial.clone()
            with envs.SGLANG_SAIL_PLA_CUDA.override(False):
                expected = run(reference)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(captured, expected, atol=0, rtol=0)
            torch.testing.assert_close(state, reference, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
