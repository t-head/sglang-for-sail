"""Packed PLA GDN parity, physical views, state writes and cold Graph fallback."""

import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.linear.fused_gdn_gating import fused_gdn_gating
from sglang.srt.environ import envs
from sglang.srt.layers.attention.linear.kernels.gdn_pla import (
    gather_pool_states,
    scatter_pool_states,
)
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.srt.utils import is_sail
from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.test_utils import CustomTestCase

register_ppu_ci(est_time=180, suite="stage-b-test-1-gpu-ppu")


def data(lengths=(65, 37), hq=4, hv=16):
    tokens = sum(lengths)
    width = (2 * hq + hv) * 128
    # Slices of a padded convolution output, including a nonzero storage offset.
    packed = torch.randn(tokens, width + 16, device="cuda", dtype=torch.bfloat16)[
        :, 8 : width + 8
    ]
    b, a = torch.randn(tokens, 2 * hv, device="cuda", dtype=packed.dtype).split(
        hv, dim=-1
    )
    cu = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )
    slots = torch.arange(len(lengths), device="cuda", dtype=torch.int32)
    return dict(
        packed=packed,
        a=a,
        b=b,
        A_log=torch.randn(hv, device="cuda") * 0.5 - 2,
        dt_bias=torch.randn(hv, device="cuda", dtype=torch.bfloat16),
        hq=hq,
        hv=hv,
        cu=cu,
        slots=slots,
        lengths=list(lengths),
    )


def views(d):
    q, k, v = d["packed"].split((d["hq"] * 128, d["hq"] * 128, d["hv"] * 128), dim=-1)
    tokens = d["packed"].shape[0]
    return (
        q.view(1, tokens, d["hq"], 128),
        k.view(1, tokens, d["hq"], 128),
        v.view(1, tokens, d["hv"], 128),
    )


def run(kernel, d, pool, **extra):
    args = (
        dict(
            ssm_states=pool,
            cache_indices=d["slots"],
            query_start_loc=d["cu"],
            seq_lens_cpu=d["lengths"],
        )
        | extra
    )
    result = kernel.packed_prefill(
        d["packed"],
        d["a"],
        d["b"],
        A_log=d["A_log"],
        dt_bias=d["dt_bias"],
        num_qk_heads=d["hq"],
        num_v_heads=d["hv"],
        **args,
    )
    if result is not None:
        return result
    q, k, v = views(d)
    g, beta = fused_gdn_gating(d["A_log"], d["a"], d["b"], d["dt_bias"])
    return kernel.extend(q, k, v, g, beta, **args)


@unittest.skipUnless(
    is_sail() and torch.cuda.get_device_capability() == (8, 9),
    "native PLA GDN regressions are validated on SAIL SM89",
)
class TestPLAPrefillNumerics(CustomTestCase):
    def test_packed_raw_gating_matches_triton_and_preserves_inputs(self):
        from pla.prefill import flashqla

        torch.manual_seed(42)
        for hq, hv, lengths in (
            (4, 16, (65, 37)),
            (16, 48, (65, 37)),
            (16, 48, (8192,)),
        ):
            d = data(lengths=lengths, hq=hq, hv=hv)
            originals = [d[k].clone() for k in ("packed", "a", "b", "A_log", "dt_bias")]
            for dtype in (torch.float32, torch.bfloat16):
                with self.subTest(hq=hq, hv=hv, state=dtype):
                    initial = (
                        torch.randn(5, hv, 128, 128, device="cuda", dtype=dtype) * 0.02
                    )
                    initial[0].zero_()
                    pools = [initial.clone(), initial.clone()]
                    outputs = []
                    for enabled, pool in zip((False, True), pools):
                        with (
                            envs.SGLANG_SAIL_PLA_CUDA.override(enabled),
                            patch.object(
                                flashqla,
                                "gdn_prefill_packed_qkv_fwd",
                                wraps=flashqla.gdn_prefill_packed_qkv_fwd,
                            ) as native,
                        ):
                            outputs.append(run(TritonGDNKernel(), d, pool)[0])
                            self.assertEqual(native.call_count, int(enabled))
                            if enabled:
                                self.assertIs(
                                    native.call_args.kwargs["packed_qkv"], d["packed"]
                                )
                                self.assertIs(
                                    native.call_args.kwargs["raw_gate_a"], d["a"]
                                )
                                self.assertIs(
                                    native.call_args.kwargs["raw_beta_b"], d["b"]
                                )
                    torch.testing.assert_close(
                        outputs[0], outputs[1], atol=2e-2, rtol=1e-2
                    )
                    torch.testing.assert_close(pools[0], pools[1], atol=2e-2, rtol=1e-2)
            for key, original in zip(
                ("packed", "a", "b", "A_log", "dt_bias"), originals
            ):
                torch.testing.assert_close(d[key], original, atol=0, rtol=0)

    def test_previous_activated_api_and_packed_api_state_semantics(self):
        from pla.prefill.flashqla import chunk_gated_delta_rule_fwd

        torch.manual_seed(43)
        d = data(lengths=(65, 37, 23))
        d["slots"] = torch.tensor([3, 0, -1], device="cuda", dtype=torch.int32)
        q, k, v = views(d)
        g, beta = fused_gdn_gating(d["A_log"], d["a"], d["b"], d["dt_bias"])
        for dtype in (torch.float32, torch.bfloat16):
            for transposed in (False, True):
                initial = (
                    torch.randn(5, 16, 128, 128, device="cuda", dtype=dtype) * 0.02
                )
                initial[0].zero_()
                if transposed:
                    actual = initial.transpose(-1, -2).contiguous().transpose(-1, -2)
                else:
                    backing = torch.empty(*initial.shape, 2, device="cuda", dtype=dtype)
                    backing[..., 0] = initial
                    actual = backing[..., 0]
                reference = initial.clone()
                rows, live, scratch = gather_pool_states(reference, d["slots"])
                _, _, expected, _, _ = chunk_gated_delta_rule_fwd(
                    q.contiguous(),
                    k.contiguous(),
                    v.contiguous(),
                    g,
                    beta,
                    initial_state=scratch,
                    initial_state_indices=torch.arange(
                        3, device="cuda", dtype=torch.int32
                    ),
                    cu_seqlens=d["cu"],
                    raw_cu_seqlens=(0, 65, 102, 125),
                    normalize_qk=True,
                    final_state_dtype=dtype,
                    output_final_state=False,
                )
                scatter_pool_states(reference, rows, live, scratch)
                with envs.SGLANG_SAIL_PLA_CUDA.override(True):
                    output = run(TritonGDNKernel(), d, actual)[0]
                torch.testing.assert_close(output, expected, atol=2e-2, rtol=1e-2)
                torch.testing.assert_close(actual, reference, atol=2e-2, rtol=1e-2)
                torch.testing.assert_close(
                    actual[[1, 2, 4]], initial[[1, 2, 4]], atol=0, rtol=0
                )

    def test_ragged_continuation_consumes_updated_state(self):
        torch.manual_seed(44)
        d = data()
        for dtype in (torch.float32, torch.bfloat16):
            initial = torch.randn(5, 16, 128, 128, device="cuda", dtype=dtype) * 0.02
            initial[0].zero_()
            pools = [initial.clone(), initial.clone()]
            kernels = [TritonGDNKernel(), TritonGDNKernel()]
            for _ in range(2):
                outputs = []
                for enabled, pool, kernel in zip((False, True), pools, kernels):
                    with envs.SGLANG_SAIL_PLA_CUDA.override(enabled):
                        outputs.append(run(kernel, d, pool)[0])
                torch.testing.assert_close(outputs[0], outputs[1], atol=2e-2, rtol=1e-2)
                torch.testing.assert_close(pools[0], pools[1], atol=2e-2, rtol=1e-2)

    def test_unsupported_heads_and_disabled_normalization_keep_triton(self):
        from pla.prefill import flashqla

        for d, extra in (
            (data(hq=1, hv=2), {}),
            (data(), {"use_qk_l2norm_in_kernel": False}),
        ):
            initial = torch.randn(5, d["hv"], 128, 128, device="cuda") * 0.02
            pools = [initial.clone(), initial.clone()]
            outputs = []
            with patch.object(
                flashqla,
                "gdn_prefill_packed_qkv_fwd",
                side_effect=AssertionError("invalid packed dispatch"),
            ):
                for enabled, pool in zip((False, True), pools):
                    with envs.SGLANG_SAIL_PLA_CUDA.override(enabled):
                        outputs.append(run(TritonGDNKernel(), d, pool, **extra)[0])
            torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
            torch.testing.assert_close(pools[0], pools[1], atol=0, rtol=0)

    def test_fallbacks_preserve_readonly_and_checkpoint_states(self):
        from pla.prefill import flashqla

        d = data(lengths=(65,))
        for extra in (
            {"inplace_update": False},
            {"requires_intermediate_states": True},
            {"seq_lens_cpu": [64]},
        ):
            initial = torch.randn(3, 16, 128, 128, device="cuda") * 0.02
            pools = [initial.clone(), initial.clone()]
            results = []
            with patch.object(
                flashqla,
                "gdn_prefill_packed_qkv_fwd",
                side_effect=AssertionError("unexpected packed dispatch"),
            ):
                for enabled, pool in zip((False, True), pools):
                    with envs.SGLANG_SAIL_PLA_CUDA.override(enabled):
                        results.append(run(TritonGDNKernel(), d, pool, **extra))
            torch.testing.assert_close(results[0][0], results[1][0], atol=0, rtol=0)
            torch.testing.assert_close(pools[0], pools[1], atol=0, rtol=0)
            if extra.get("inplace_update") is False:
                torch.testing.assert_close(pools[1], initial, atol=0, rtol=0)
            if extra.get("requires_intermediate_states"):
                self.assertIsNotNone(results[1][2])
                torch.testing.assert_close(results[0][2], results[1][2], atol=0, rtol=0)

    def test_graph_replay_uses_live_offsets_and_slots(self):
        torch.manual_seed(45)
        d = data()
        kernel = TritonGDNKernel()
        initial = torch.randn(5, 16, 128, 128, device="cuda") * 0.02
        initial[0].zero_()
        state = initial.clone()
        graph = torch.cuda.CUDAGraph()
        with envs.SGLANG_SAIL_PLA_CUDA.override(True):
            run(kernel, d, state)
            with torch.cuda.graph(graph):
                captured = run(kernel, d, state)[0]
        for offsets, indices in (([0, 66, 102], [0, 4]), ([0, 67, 102], [1, 3])):
            d["cu"].copy_(torch.tensor(offsets, device="cuda", dtype=torch.int32))
            d["slots"].copy_(torch.tensor(indices, device="cuda", dtype=torch.int32))
            state.copy_(initial)
            reference = initial.clone()
            with envs.SGLANG_SAIL_PLA_CUDA.override(False):
                expected = run(kernel, d, reference)[0]
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(captured, expected, atol=0, rtol=0)
            torch.testing.assert_close(state, reference, atol=0, rtol=0)

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


if __name__ == "__main__":
    unittest.main()
