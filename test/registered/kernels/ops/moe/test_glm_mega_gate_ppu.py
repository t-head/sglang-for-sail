"""Masked IDs can conceal poisoned padding weights; validate the native PPU mask."""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.configs.shared_experts import SharedExpertsConfig
from sglang.srt.environ import envs
from sglang.srt.layers.moe.topk import TopK, TopKOutputFormat
from sglang.srt.layers.moe.utils import get_moe_runner_backend, initialize_moe_config
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.runtime_context import get_context, get_flags, get_model, get_parallel
from sglang.srt.utils import is_sail
from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.test_utils import CustomTestCase

register_ppu_ci(est_time=120, suite="stage-a-test-1-gpu-ppu")

SHAPES = (1, 2, 8, 17, 65, 80, 128, 255, 256, 512)
TOPOLOGIES = ((4, 1, 4), (8, 1, 8), (1, 8, 8))
SCALE = 2.5
# These fixtures deliberately separate the eighth/ninth candidates. A failure
# reports ambiguity instead of accepting a different expert set or loosening it.
MIN_BOUNDARY_GAP = 0.02
WEIGHT_RTOL = 1e-3
WEIGHT_ATOL = 1e-4


@contextmanager
def _runtime(cp, dp, ep, backend="deepep"):
    flags = get_flags().moe
    saved = {name: getattr(flags, name) for name in type(flags).__struct_fields__}
    with (
        get_context().override_server_args(
            tp_size=ep,
            ep_size=ep,
            attn_cp_size=cp,
            attn_dp_size=dp,
            moe_a2a_backend=backend,
            moe_runner_backend="deep_gemm",
            ep_dispatch_algorithm=None,
            enable_eplb=False,
            init_expert_location="trivial",
            ep_num_redundant_experts=0,
            enable_deterministic_inference=False,
            enable_lora=False,
            enable_deepep_waterfill=False,
        ),
        flags.override(**saved),
        get_parallel().override(moe_ep_rank=0),
        get_model().override(shared_experts=SharedExpertsConfig()),
        envs.SGLANG_SIMULATE_UNIFORM_EXPERTS.override(False),
        envs.SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS.override(False),
    ):
        initialize_moe_config()
        yield


def _router(bias):
    router = TopK(
        top_k=8,
        layer_id=0,
        use_grouped_topk=True,
        num_expert_group=1,
        topk_group=1,
        scoring_func="sigmoid",
        correction_bias=bias,
        renormalize=True,
        routed_scaling_factor=SCALE,
        apply_routed_scaling_factor_on_output=True,
        output_format=TopKOutputFormat.STANDARD,
    )
    router.bind_runner_backend(get_moe_runner_backend())
    return router


def _inputs(tokens, layout="row-varying"):
    generator = torch.Generator(device="cuda").manual_seed(341)
    weight = torch.randn(288, 4096, device="cuda", generator=generator)
    weight *= 0.002 if layout == "row-varying" else 0.015
    x = torch.randn(tokens, 4096, device="cuda", generator=generator)
    bias = torch.randn(288, device="cuda", generator=generator) * 0.01
    if layout == "row-varying":
        # Dense GEMM noise plus a row-dependent winning group: all 288 experts
        # are exercised, with a wide sigmoid selection boundary, not random ties.
        experts = torch.arange(288, device="cuda")
        x[:, :36] = 0
        x[
            torch.arange(tokens, device="cuda"),
            torch.arange(tokens, device="cuda") % 36,
        ] = 1
        weight[:, :36] = 0
        weight[experts, experts // 8] = 1.5 + (experts % 8).float() * 0.25
        bias -= 8.0
    else:
        # Bias must affect selection but NOT gathered/normalized weights.
        chosen = torch.tensor([3, 41, 79, 117, 155, 193, 231, 269], device="cuda")
        bias[chosen] += 2.0
    return x.bfloat16(), weight.bfloat16(), bias


class TestGLMMegaGatePPU(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if (
            not is_sail()
            or not torch.cuda.is_available()
            or torch.cuda.get_device_capability() != (8, 9)
        ):
            raise unittest.SkipTest("Requires a real SM89 PPU")
        import deep_gemm

        if not hasattr(deep_gemm, "bf16_mega_gate"):
            raise RuntimeError("PPU DeepGEMM must provide bf16_mega_gate")

    def _reference(self, x, weight, bias):
        # Independent BF16-input GEMM with FP32 accumulation/output; activation
        # stays FP32. Bias ranks experts; weights gather UNBIASED sigmoid scores.
        scores = torch.mm(x, weight.t(), out_dtype=torch.float32).sigmoid()
        candidates = (scores + bias).topk(9, dim=-1)
        gap = candidates.values[:, 7] - candidates.values[:, 8]
        if gap.numel():
            worst = int(gap.argmin())
            self.assertGreater(
                float(gap[worst]),
                MIN_BOUNDARY_GAP,
                f"Ambiguous top-8 boundary at row {worst}: gap={float(gap[worst]):.8g}; "
                f"required > {MIN_BOUNDARY_GAP}. Expert-set checks are not relaxed.",
            )
        ids = candidates.indices[:, :8]
        weights = scores.gather(1, ids)
        weights = weights / weights.sum(-1, keepdim=True) * SCALE
        return weights, ids

    def _assert_routes(self, actual, expected, live):
        self.assertIsNotNone(actual, "Production try_mega_gate unexpectedly fell back")
        self.assertEqual(actual.topk_ids.shape, actual.topk_weights.shape)
        self.assertEqual(actual.topk_ids.shape[1], 8)
        self.assertEqual(actual.topk_ids.dtype, torch.int64)
        self.assertEqual(actual.topk_weights.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(actual.topk_weights).all()))
        weights, ids = expected
        actual_ids, order = actual.topk_ids[:live].sort(dim=-1)
        actual_weights = actual.topk_weights[:live].gather(1, order)
        ids, order = ids.sort(dim=-1)
        weights = weights.gather(1, order)
        torch.testing.assert_close(actual_ids, ids, rtol=0, atol=0)
        torch.testing.assert_close(
            actual_weights, weights, rtol=WEIGHT_RTOL, atol=WEIGHT_ATOL
        )
        torch.testing.assert_close(
            actual.topk_weights[:live].sum(-1),
            torch.full((live,), SCALE, device="cuda"),
            rtol=WEIGHT_RTOL,
            atol=WEIGHT_ATOL,
        )
        self.assertTrue(bool((actual.topk_ids[live:] == -1).all()))
        self.assertTrue(bool((actual.topk_weights[live:] == 0).all()))

    @torch.inference_mode()
    def test_precision_and_shape_boundaries(self):
        for cp, dp, ep in TOPOLOGIES:
            with _runtime(cp, dp, ep):
                self.assertEqual(
                    (
                        get_parallel().attn_cp_size,
                        get_parallel().attn_dp_size,
                        get_parallel().moe_ep_size,
                    ),
                    (cp, dp, ep),
                )
                for layout in ("row-varying", "bias-selected"):
                    for tokens in SHAPES:
                        with self.subTest(cp=cp, dp=dp, tokens=tokens, layout=layout):
                            x, weight, bias = _inputs(tokens, layout)
                            router = _router(bias)
                            count = torch.tensor(
                                tokens, device="cuda", dtype=torch.int32
                            )
                            for bound in (None, count):
                                with forward_context(
                                    ForwardContext(
                                        attn_backend=None,
                                        moe_num_token_non_padded=lambda: bound,
                                    )
                                ):
                                    actual = router.try_mega_gate(x, weight)
                                if dp == 8 and not 2 <= tokens <= 256:
                                    self.assertIsNone(actual)
                                else:
                                    self._assert_routes(
                                        actual, self._reference(x, weight, bias), tokens
                                    )
                for tokens in ((0, 1, 257, 513) if dp == 8 else (0, 513)):
                    with self.subTest(cp=cp, dp=dp, rejected_tokens=tokens):
                        x, weight, bias = _inputs(tokens)
                        self.assertIsNone(_router(bias).try_mega_gate(x, weight))

    @torch.inference_mode()
    def test_shared_mask_and_count_refresh(self):
        from sglang.srt.layers.moe.mega_gate import mega_gate_forward_scope

        tokens = 8
        x, weight, bias = _inputs(tokens, "bias-selected")
        count = torch.tensor(tokens, device="cuda", dtype=torch.int32)
        batch = SimpleNamespace(moe_num_token_non_padded=lambda: count)
        seen = []

        def capture_mask(*args, **kwargs):
            seen.append(kwargs["mask"])
            return original_kernel(*args, **kwargs)

        with _runtime(4, 1, 4):
            import deep_gemm

            routers = [_router(bias), _router(bias)]
            original_kernel = deep_gemm.bf16_mega_gate
            with (
                envs.SGLANG_OPT_DEEPGEMM_MEGA_GATE.override(True),
                forward_context(ForwardContext(attn_backend=None), forward_batch=batch),
                mega_gate_forward_scope(batch),
                patch.object(deep_gemm, "bf16_mega_gate", side_effect=capture_mask),
            ):
                for router in routers:
                    actual = router.try_mega_gate(x, weight)
                    self._assert_routes(actual, self._reference(x, weight, bias), tokens)
            self.assertEqual(len(seen), 2)
            self.assertIs(seen[0], seen[1])

            other_batch = SimpleNamespace(moe_num_token_non_padded=lambda: count)
            with (
                forward_context(ForwardContext(attn_backend=None), forward_batch=other_batch),
                mega_gate_forward_scope(other_batch),
                patch.object(deep_gemm, "bf16_mega_gate", side_effect=capture_mask),
            ):
                actual = routers[0].try_mega_gate(x, weight)
                self._assert_routes(actual, self._reference(x, weight, bias), tokens)
            self.assertEqual(len(seen), 3)
            self.assertIsNot(seen[2], seen[0])

            count.fill_(4)
            x[4:] = float("nan")
            with (
                envs.SGLANG_OPT_DEEPGEMM_MEGA_GATE.override(True),
                forward_context(ForwardContext(attn_backend=None), forward_batch=batch),
                mega_gate_forward_scope(batch),
            ):
                actual = routers[0].try_mega_gate(x, weight)
                self._assert_routes(actual, self._reference(x[:4], weight, bias), 4)

    def test_shared_mask_graph_count_refresh(self):
        from sglang.srt.layers.moe.mega_gate import mega_gate_forward_scope

        tokens = 8
        clean, weight, bias = _inputs(tokens, "bias-selected")
        count = torch.tensor(tokens, device="cuda", dtype=torch.int32)
        batch = SimpleNamespace(moe_num_token_non_padded=lambda: count)
        x = clean.clone()
        with _runtime(4, 1, 4):
            router = _router(bias)
            with (
                envs.SGLANG_OPT_DEEPGEMM_MEGA_GATE.override(True),
                forward_context(ForwardContext(attn_backend=None), forward_batch=batch),
                mega_gate_forward_scope(batch),
            ):
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        router.try_mega_gate(x, weight)
                    with torch.cuda.graph(graph, stream=stream):
                        captured = router.try_mega_gate(x, weight)
                torch.cuda.current_stream().wait_stream(stream)
                for live in (8, 4, 0, 8):
                    count.fill_(live)
                    x.copy_(clean)
                    x[live:] = float("nan")
                    graph.replay()
                    self._assert_routes(captured, self._reference(clean[:live], weight, bias), live)

    @torch.inference_mode()
    def test_padding_and_graph_count_changes(self):
        for cp, dp, ep in TOPOLOGIES:
            with _runtime(cp, dp, ep, backend="megamoe"):
                for tokens in ((65, 256) if dp == 8 else (65, 256, 512)):
                    with self.subTest(cp=cp, dp=dp, tokens=tokens):
                        clean, weight, bias = _inputs(tokens, "bias-selected")
                        x = clean.clone()
                        router = _router(bias)
                        count = torch.tensor(tokens, device="cuda", dtype=torch.int32)
                        with forward_context(
                            ForwardContext(
                                attn_backend=None,
                                moe_num_token_non_padded=lambda: count,
                            )
                        ):
                            self.assertIs(router.routing_inputs()[0], count)
                            baseline = router.try_mega_gate(x, weight)
                            self._assert_routes(
                                baseline, self._reference(clean, weight, bias), tokens
                            )
                            baseline_ids = baseline.topk_ids.clone()
                            baseline_weights = baseline.topk_weights.clone()
                            stream = torch.cuda.Stream()
                            stream.wait_stream(torch.cuda.current_stream())
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.stream(stream):
                                for _ in range(3):
                                    router.try_mega_gate(x, weight)
                                with torch.cuda.graph(graph, stream=stream):
                                    captured = router.try_mega_gate(x, weight)
                            torch.cuda.current_stream().wait_stream(stream)
                            # Reuse the same device count buffer; replay must read
                            # its new value, including zero and restoration to full.
                            for live, poison in (
                                (tokens, float("nan")),
                                (tokens // 2, float("nan")),
                                (1, float("inf")),
                                (0, float("nan")),
                                (0, float("inf")),
                                (tokens, float("inf")),
                            ):
                                with self.subTest(live=live, poison=poison):
                                    count.fill_(live)
                                    x.copy_(clean)
                                    x[live:] = poison
                                    expected = self._reference(
                                        clean[:live], weight, bias
                                    )
                                    eager = router.try_mega_gate(x, weight)
                                    graph.replay()
                                    for actual in (eager, captured):
                                        self._assert_routes(actual, expected, live)
                                        torch.testing.assert_close(
                                            actual.topk_ids[:live],
                                            baseline_ids[:live],
                                            rtol=0,
                                            atol=0,
                                        )
                                        torch.testing.assert_close(
                                            actual.topk_weights[:live],
                                            baseline_weights[:live],
                                            rtol=0,
                                            atol=0,
                                        )


if __name__ == "__main__":
    unittest.main()
