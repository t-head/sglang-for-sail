"""Global-column reconstruction must ignore poisoned compressed padding."""

import pytest
import torch

from sglang.kernels.ops.attention.dsa.ppu_logits import restore_ragged_logits
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=3, stage="base-b-kernel-unit", runner_config="1-gpu-small")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("strided", [False, True])
def test_restore_and_graph(dtype, strided):
    raw = torch.full((3, 20), float("nan"), device="cuda", dtype=torch.bfloat16)
    source = raw[:, ::2] if strided else raw[:, :10]
    source[0, :4] = torch.tensor([1, 2, 3, 4], device="cuda")
    source[1, :2] = torch.tensor([5, 6], device="cuda")
    starts = torch.tensor([3, 0, 8], device="cuda", dtype=torch.int32)
    ends = torch.tensor([7, 2, 8], device="cuda", dtype=torch.int32)
    restore_ragged_logits(source, starts, ends, 13, dtype)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = restore_ragged_logits(source, starts, ends, 13, dtype)
    for shift in [0, 2]:
        starts.copy_(
            torch.tensor([3 + shift, shift, 8], device="cuda", dtype=torch.int32)
        )
        ends.copy_(
            torch.tensor([7 + shift, 2 + shift, 8], device="cuda", dtype=torch.int32)
        )
        graph.replay()
        expected = torch.full((3, 13), -float("inf"), device="cuda", dtype=dtype)
        expected[0, 3 + shift : 7 + shift] = source[0, :4]
        expected[1, shift : 2 + shift] = source[1, :2]
        torch.testing.assert_close(result, expected, rtol=0, atol=0)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
