"""Complete sparse selection plus slot mapping on PPU, with cold L2."""

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.attention.dsv4.topk import (
    topk_transform_sparse,
    topk_transform_sparse_bf16_v2,
)
from sglang.srt.utils import is_sail
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=10, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


@marker.parametrize("rows", [8, 256])
@marker.parametrize("width", [2048, 8192, 16384])
@marker.benchmark("impl", ["legacy", "bf16_v2"], unit="us")
def benchmark(rows, width, impl):
    if not is_sail():
        marker.skip("PPU-only BF16 v2 opt-in")
    scores = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
    lengths = torch.full((rows,), width, device="cuda", dtype=torch.int32)
    blocks = torch.arange(width // 8, device="cuda", dtype=torch.int32)
    blocks = blocks.expand(rows, -1).contiguous()
    output = torch.empty(rows, 512, device="cuda", dtype=torch.int32)
    fn = topk_transform_sparse if impl == "legacy" else topk_transform_sparse_bf16_v2
    # Eager timing flushes L2 before every call, including tiny score matrices
    # for which the graph helper's capped input rotation can leave L2 hot.
    return marker.do_bench(
        fn,
        input_args=(scores, lengths, blocks, output),
        use_cuda_graph=False,
        estimated_time_ms=None,
        warmup_iters=5,
        replay_iters=60,
        memory_output=(output,),
    )


if __name__ == "__main__":
    benchmark.run()
