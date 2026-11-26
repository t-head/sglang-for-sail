"""Event-timed production GLM routing: MoEGate + TopK versus MegaGate dispatch.

Times graph replay to avoid Python launch gaps. Both paths use identical BF16
inputs/weights and FP32 bias, including the production shape-based fallback.
This is a single-device routing microbenchmark, not a CP/DP collective benchmark.
"""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace

import torch

from sglang.kernels.jit.benchmark.marker import do_bench
from sglang.srt.configs.shared_experts import SharedExpertsConfig
from sglang.srt.environ import envs
from sglang.srt.layers.moe.topk import TopK, TopKOutputFormat
from sglang.srt.layers.moe.utils import get_moe_runner_backend, initialize_moe_config
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.models.deepseek_v2 import MoEGate
from sglang.srt.runtime_context import get_context, get_flags, get_model, get_parallel
from sglang.srt.utils import is_sail
from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.test_utils import CustomTestCase

register_ppu_ci(est_time=60, suite="stage-a-test-1-gpu-ppu")

SHAPES = (1, 2, 8, 17, 65, 80, 128, 255, 256, 512)
TOPOLOGIES = ((4, 1, 4), (8, 1, 8), (1, 8, 8))


@contextmanager
def _runtime(cp, dp, ep):
    flags = get_flags().moe
    saved = {name: getattr(flags, name) for name in type(flags).__struct_fields__}
    with (
        get_context().override_server_args(
            tp_size=ep,
            ep_size=ep,
            attn_cp_size=cp,
            attn_dp_size=dp,
            moe_a2a_backend="deepep",
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
        get_parallel().override(moe_ep_rank=0, tp_group=None),
        get_model().override(shared_experts=SharedExpertsConfig()),
        envs.SGLANG_SIMULATE_UNIFORM_EXPERTS.override(False),
        envs.SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS.override(False),
    ):
        initialize_moe_config()
        yield


def _make_gate_and_router(weight, bias):
    # This is the real model gate, not a stand-in GEMM or a torch-only router.
    gate = MoEGate(
        SimpleNamespace(n_routed_experts=288, hidden_size=4096, topk_method="noaux_tc"),
        quant_config=None,
    ).to(device="cuda")
    gate.weight = torch.nn.Parameter(weight, requires_grad=False)
    gate.e_score_correction_bias = torch.nn.Parameter(bias, requires_grad=False)
    router = TopK(
        top_k=8,
        layer_id=0,
        use_grouped_topk=True,
        num_expert_group=1,
        topk_group=1,
        scoring_func="sigmoid",
        correction_bias=gate.e_score_correction_bias,
        renormalize=True,
        routed_scaling_factor=2.5,
        apply_routed_scaling_factor_on_output=True,
        output_format=TopKOutputFormat.STANDARD,
    )
    router.bind_runner_backend(get_moe_runner_backend())
    return gate, router


def _event_latency_us(run, replays=50, calls=1):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        for _ in range(3):
            for _ in range(calls):
                run()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(calls):
                run()
    torch.cuda.current_stream().wait_stream(stream)
    for _ in range(3):
        graph.replay()
    result = do_bench(
        graph.replay,
        use_cuda_graph=False,
        warmup_iters=3,
        replay_iters=replays,
        estimated_time_ms=None,
        metrics=(0.5,),
        disable_log_bandwidth=True,
    )
    return result.times[0] * 1e6 / calls


class BenchGLMMegaGatePPU(CustomTestCase):
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

    @torch.inference_mode()
    def test_production_routing_latency(self):
        generator = torch.Generator(device="cuda").manual_seed(341)
        weight = (
            torch.randn(288, 4096, device="cuda", generator=generator) * 0.015
        ).bfloat16()
        bias = torch.randn(288, device="cuda", generator=generator) * 0.01
        all_x = torch.randn(513, 4096, device="cuda", generator=generator).bfloat16()
        print(
            "Cold L2: CP DP EP rows selected fallback_us enabled_us speedup", flush=True
        )
        for cp, dp, ep in TOPOLOGIES:
            with _runtime(cp, dp, ep):
                gate, router = _make_gate_and_router(weight, bias)
                for tokens in (*SHAPES, 257, 513):
                    with self.subTest(cp=cp, dp=dp, tokens=tokens):
                        x = all_x[:tokens]
                        count = torch.tensor(tokens, device="cuda", dtype=torch.int32)
                        with forward_context(
                            ForwardContext(
                                attn_backend=None,
                                moe_num_token_non_padded=lambda: count,
                            )
                        ):

                            def fallback():
                                return router(x, gate(x))

                            def enabled():
                                # Same gate-selection branch as forward_deepep.
                                output = router.try_mega_gate(x, gate.weight)
                                return fallback() if output is None else output

                            selected = router.try_mega_gate(x, gate.weight) is not None
                            expected_selected = (
                                2 <= tokens <= 256 if dp == 8 else 1 <= tokens <= 512
                            )
                            self.assertEqual(selected, expected_selected)
                            # Warm both real paths before collecting GPU events.
                            fallback()
                            enabled()
                            fallback_us = _event_latency_us(fallback)
                            enabled_us = _event_latency_us(enabled)
                        print(
                            f"{cp} {dp} {ep} {tokens} {selected} "
                            f"{fallback_us:.3f} {enabled_us:.3f} "
                            f"{fallback_us / enabled_us:.3f}",
                            flush=True,
                        )

    @torch.inference_mode()
    def test_shared_mask_per_forward(self):
        from types import SimpleNamespace

        from sglang.srt.layers.moe.mega_gate import mega_gate_forward_scope

        generator = torch.Generator(device="cuda").manual_seed(341)
        weight = (
            torch.randn(288, 4096, device="cuda", generator=generator) * 0.015
        ).bfloat16()
        bias = torch.randn(288, device="cuda", generator=generator) * 0.01
        for cp, dp, ep, tokens in ((4, 1, 4, 128), (8, 1, 8, 128), (1, 8, 8, 16), (1, 8, 8, 256)):
            with _runtime(cp, dp, ep), envs.SGLANG_OPT_DEEPGEMM_MEGA_GATE.override(True):
                gate, router = _make_gate_and_router(weight, bias)
                count = torch.tensor(tokens, device="cuda", dtype=torch.int32)
                batch = SimpleNamespace(moe_num_token_non_padded=lambda: count)
                x = torch.randn(tokens, 4096, device="cuda", generator=generator).bfloat16()
                with forward_context(ForwardContext(attn_backend=None), forward_batch=batch):
                    def fallback():
                        return router(x, gate(x))

                    def enabled():
                        output = router.try_mega_gate(x, weight, forward_batch=batch)
                        assert output is not None
                        return output

                    with mega_gate_forward_scope(batch):
                        baseline_us = _event_latency_us(fallback, calls=8)
                        enabled_us = _event_latency_us(enabled, calls=8)
                print(
                    f"shared_forward {cp} {dp} {ep} {tokens} layers=8 "
                    f"fallback_per_call_us={baseline_us:.3f} "
                    f"enabled_per_call_us={enabled_us:.3f} "
                    f"speedup={baseline_us / enabled_us:.3f}",
                    flush=True,
                )


if __name__ == "__main__":
    unittest.main()
