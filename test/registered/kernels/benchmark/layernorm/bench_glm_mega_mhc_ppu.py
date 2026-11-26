"""Rank-local GLM H4096 hc_post/hc_pre versus model MegaMHC boundary.

GPU-event microbenchmark, not a full-model/communication benchmark or a speed
acceptance gate. Report signed gains, including regressions, for every shape.
"""

import json
import statistics
import unittest

import torch

from sglang.kernels.jit.benchmark.marker import do_bench
from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.kernels.mhc import (
    GLM_MHC_ROWS,
    GLM_MHC_SEEDS,
    GlmMHCFixture,
    assert_glm_mhc_close,
    glm_mhc_ppu_context,
)
from sglang.test.test_utils import CustomTestCase

register_ppu_ci(est_time=120, suite="stage-a-test-1-gpu-ppu")


def event_us(fn, stream, repeats=20):
    result = do_bench(
        fn,
        use_cuda_graph=False,
        stream=stream,
        warmup_iters=5,
        replay_iters=repeats,
        estimated_time_ms=None,
        metrics=(0.5,),
        disable_log_bandwidth=True,
    )
    return result.times[0] * 1e6


class BenchGlmMegaMHCPPU(CustomTestCase):
    def benchmark_shape(self, rows, seed):
        fixture = GlmMHCFixture(rows, seed)
        reference = fixture.reference()
        assert_glm_mhc_close(self, fixture.baseline(), reference)
        assert_glm_mhc_close(self, fixture.mega(), reference)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            # Compile and initialize both backends on the actual timing/capture stream.
            for _ in range(20):
                fixture.baseline()
                fixture.mega()
        stream.synchronize()
        for mode in ("eager", "graph_replay"):
            functions = [fixture.baseline, fixture.mega]
            graphs = []
            if mode == "graph_replay":
                for index, fn in enumerate(functions):
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        output = fn()
                    graph.replay()
                    torch.cuda.synchronize()
                    assert_glm_mhc_close(self, output, reference)
                    graphs.append(graph)
                    functions[index] = graph.replay
            with torch.cuda.stream(stream):
                for _ in range(20):
                    for fn in functions:
                        fn()
            stream.synchronize()
            pairs = []
            for pair in range(7):
                order = (0, 1) if pair % 2 == 0 else (1, 0)
                values = {}
                for index in order:
                    values[index] = event_us(functions[index], stream)
                pairs.append((values[0], values[1]))
            baseline_us = statistics.median(a for a, _ in pairs)
            mega_us = statistics.median(b for _, b in pairs)
            self.assertGreater(baseline_us, 0)
            self.assertGreater(mega_us, 0)
            print(
                json.dumps(
                    dict(
                        rows=rows,
                        hidden=4096,
                        seed=seed,
                        mode=mode,
                        device=torch.cuda.get_device_name(),
                        cp=4,
                        dp=1,
                        baseline="original hc_post/hc_pre + RMSNorm if unfused",
                        mega="Glm5NextDecoderLayer.hc_ffn_post_pre",
                        timing="rank-local cold-L2 GPU events; no communication or model inference",
                        warmup=20,
                        repeats=20,
                        pairs=7,
                        baseline_us=baseline_us,
                        mega_us=mega_us,
                        latency_reduction_pct=100 * (1 - mega_us / baseline_us),
                        paired_us=pairs,
                    )
                ),
                flush=True,
            )

    def test_benchmark(self):
        with glm_mhc_ppu_context():
            for rows in GLM_MHC_ROWS:
                for seed in GLM_MHC_SEEDS:
                    with self.subTest(rows=rows, seed=seed):
                        self.benchmark_shape(rows, seed)


if __name__ == "__main__":
    unittest.main()
