"""Fixed-input single-PPU operator benchmark, NOT a DeepEP communication A/B.

Measures BF16 input -> BF16 output with GPU events after JIT warmup. Baseline:
real channelwise FP8 grouped GEMMs with Torch quantization/SwiGLU/combine and
an offline route permutation. Mega: pre-dispatch plus complete FP8 MegaMoE.
The two paths share canonical weights, source, IDs and routing weights. Router,
weight transformation, route-plan construction and reference checks are untimed.
JSON lines are measurements, not serving throughput or full-model evidence.
"""

import json
import statistics
import unittest

import torch
import torch.nn.functional as F

from sglang.kernels.jit.benchmark.marker import do_bench
from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.kernels.fp8_mega_moe import (
    SingleRankGroup,
    check_output,
    load_deep_gemm,
    make_weights,
    native,
    quantize_rows,
    reference,
)
from sglang.test.test_utils import CustomTestCase

register_ppu_ci(est_time=120, suite="stage-a-test-1-gpu-ppu")


class GroupedFP8Baseline:
    """Two actual DeepGEMM FP8 channelwise grouped matmuls; no BF16 stand-in."""

    def __init__(self, dg, source, ids, weights, raw1, raw2):
        if not callable(getattr(dg, "m_grouped_fp8_gemm_nt_nopad", None)):
            raise RuntimeError("Operator baseline requires m_grouped_fp8_gemm_nt_nopad")
        self.dg, self.source, self.raw1, self.raw2 = dg, source, raw1, raw2
        tokens, topk = ids.shape
        order = ids.reshape(-1).argsort(stable=True)
        self.group_ids = ids.reshape(-1)[order].int().contiguous()
        self.token_ids = torch.arange(tokens, device="cuda").repeat_interleave(topk)[
            order
        ]
        self.weights = weights.reshape(-1)[order, None]
        self.counts = torch.bincount(
            self.group_ids.long(), minlength=raw1[0].shape[0]
        ).int()
        self.gate_up = torch.empty(
            (tokens * topk, raw1[0].shape[1]), device="cuda", dtype=torch.bfloat16
        )
        self.down = torch.empty(
            (tokens * topk, source.shape[1]), device="cuda", dtype=torch.bfloat16
        )
        self.combine = torch.empty(source.shape, device="cuda", dtype=torch.float32)
        self.output = torch.empty_like(source)

    def __call__(self):
        grouped = self.source[self.token_ids].contiguous()
        self.dg.m_grouped_fp8_gemm_nt_nopad(
            quantize_rows(grouped),
            self.raw1,
            self.gate_up,
            self.group_ids,
            self.counts,
        )
        gate, up = self.gate_up.float().chunk(2, -1)
        activated = (F.silu(gate.clamp(max=10.0)) * up.clamp(-10.0, 10.0)).bfloat16()
        self.dg.m_grouped_fp8_gemm_nt_nopad(
            quantize_rows(activated),
            self.raw2,
            self.down,
            self.group_ids,
            self.counts,
        )
        self.combine.zero_()
        self.combine.index_add_(0, self.token_ids, self.down.float() * self.weights)
        self.output.copy_(self.combine)
        return self.output


def gpu_event_times(forward, *, warmup=5, iterations=20, repeats=3):
    # Fixed tensors throughout; allocation/JIT setup and correctness are untimed.
    for _ in range(warmup):
        forward()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        result = do_bench(
            forward,
            use_cuda_graph=False,
            warmup_iters=warmup,
            replay_iters=iterations,
            estimated_time_ms=None,
            metrics=(0.5,),
            stream=torch.cuda.current_stream(),
            disable_log_bandwidth=True,
        )
        samples.append(result.times[0] * 1e6)
    if any(not (0 < value < float("inf")) for value in samples):
        raise AssertionError(f"Invalid GPU-event measurements: {samples}")
    return dict(
        mode="cold_l2_eager_gpu_events",
        warmup=warmup,
        iterations=iterations,
        repeats=repeats,
        median_us=statistics.median(samples),
        samples_us=samples,
    )


class BenchFP8MegaMoEPPU(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.dg = load_deep_gemm()
        cls.previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        cls.distributed = SingleRankGroup()
        cls.group = cls.distributed.open()

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "distributed"):
            cls.distributed.close()
        if hasattr(cls, "previous_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = cls.previous_tf32

    @torch.inference_mode()
    def test_operator_benchmark(self):
        # The last shape covers GLM53 widths, not its full expert count or EP.
        for experts, hidden, intermediate in (
            (8, 512, 256),
            (16, 1024, 640),
            (8, 4096, 2048),
        ):
            rng = torch.Generator(device="cuda").manual_seed(336)
            capacity, topk = 512, 2
            source = (
                torch.randn(
                    (capacity, hidden),
                    device="cuda",
                    dtype=torch.bfloat16,
                    generator=rng,
                )
                * 0.1
            )
            raw1 = make_weights(experts, 2 * intermediate, hidden, rng)
            raw2 = make_weights(experts, hidden, intermediate, rng)
            l1, l2 = self.dg.transform_weights_for_fp8_mega_moe(
                raw1, raw2, quantization="channelwise"
            )
            buffer = self.dg.get_symm_buffer_for_mega_moe(
                self.group,
                experts,
                capacity,
                topk,
                hidden,
                intermediate,
                mma_type="fp8xfp8",
                quantization="channelwise",
            )
            try:
                self.assertEqual(buffer.num_ranks, 1)
                for tokens in (1, 8, 33, 129, 512):
                    with self.subTest(
                        E=experts, H=hidden, I=intermediate, tokens=tokens
                    ):
                        x = source[:tokens]
                        rows = torch.arange(tokens, device="cuda")[:, None]
                        slots = torch.arange(topk, device="cuda")[None, :]
                        ids = ((rows + slots) % experts).long().contiguous()
                        weights = torch.rand(
                            (tokens, topk), device="cuda", generator=rng
                        )
                        weights /= weights.sum(-1, keepdim=True)
                        baseline = GroupedFP8Baseline(
                            self.dg, x, ids, weights, raw1, raw2
                        )
                        mega_out = torch.full_like(x, float("nan"))

                        def mega():
                            return native(
                                self.dg, buffer, x, ids, weights, l1, l2, out=mega_out
                            )

                        expected = reference(
                            x[None], ids[None], weights[None], raw1, raw2
                        )[0]
                        self.assertGreater(expected.float().square().sum().item(), 0)
                        baseline_check = check_output(baseline(), expected)
                        mega_check = check_output(mega(), expected)
                        baseline_timing = gpu_event_times(baseline)
                        mega_timing = gpu_event_times(mega)
                        # Repeated calls must not leave stale/invalid output.
                        check_output(baseline.output, expected)
                        check_output(mega_out, expected)
                        print(
                            json.dumps(
                                dict(
                                    scope="single_rank_operator_not_deepep_communication",
                                    boundary="bf16_input_to_bf16_output",
                                    ep=1,
                                    E=experts,
                                    H=hidden,
                                    I=intermediate,
                                    topk=topk,
                                    tokens=tokens,
                                    seed=336,
                                    fixed_inputs=True,
                                    baseline_label="fp8_grouped_gemms_plus_torch_quant_swiglu_combine_offline_route_plan",
                                    mega_label="fp8_mega_moe_including_pre_dispatch",
                                    baseline=baseline_timing,
                                    mega=mega_timing,
                                    baseline_correctness=baseline_check,
                                    mega_correctness=mega_check,
                                    operator_speedup=baseline_timing["median_us"]
                                    / mega_timing["median_us"],
                                    torch_version=torch.__version__,
                                    device=torch.cuda.get_device_name(),
                                )
                            ),
                            flush=True,
                        )
            finally:
                buffer.destroy()


if __name__ == "__main__":
    unittest.main()
