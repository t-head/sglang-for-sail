"""CPU checks for PPU prefill logits entering the FP32 ragged TopK kernel."""

import sys
from types import ModuleType
from unittest.mock import patch

import pytest
import torch

from sglang.srt.layers.attention.dsa import dsa_topk_backend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


@pytest.mark.parametrize("width", [355, 356, 357, 512])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("ppu", [False, True])
def test_ragged_topk_preserves_scores_and_platform_boundary(width, dtype, ppu):
    rows, topk = 3, 8
    torch.manual_seed(width)
    logits = torch.randn(rows, 1024, dtype=dtype)[:, :width]
    before = logits.clone()
    starts = torch.tensor([1, 2, 3], dtype=torch.int32)
    lengths = torch.tensor([width - 1, width - 3, width - 5], dtype=torch.int32)
    offsets = torch.tensor([100, 200, 300], dtype=torch.int32)
    expected = torch.stack(
        [
            before[row, start : start + length].float().topk(topk).indices + offset
            for row, (start, length, offset) in enumerate(zip(starts, lengths, offsets))
        ]
    ).to(torch.int32)

    def kernel(scores, seq_lens, *, out_offsets, out_indices, row_starts):
        if ppu and dtype == torch.bfloat16:
            assert scores.dtype == torch.float32
            assert scores.shape == logits.shape
            assert scores.stride(1) == 1
            assert scores.stride(0) % 4 == 0
            assert torch.equal(scores, before.float())
        else:
            assert scores is logits
        assert seq_lens is lengths
        assert out_offsets is offsets
        assert row_starts is starts
        for row, (start, length, offset) in enumerate(zip(starts, lengths, offsets)):
            out_indices[row] = (
                scores[row, start : start + length].float().topk(topk).indices + offset
            )

    module_name = "sglang.kernels.ops.attention.dsv4.topk"
    kernel_module = ModuleType(module_name)
    kernel_module.topk_transform_ragged_v2 = kernel
    with (
        patch.dict(sys.modules, {module_name: kernel_module}),
        patch.object(dsa_topk_backend, "is_sail", return_value=ppu),
    ):
        out = dsa_topk_backend._topk_transform_v2_ragged(
            logits, lengths, topk, offsets, starts
        )

    assert out.dtype == torch.int32
    assert torch.equal(out, expected)
    assert torch.equal(logits, before)


def test_ppu_ragged_topk_reported_shape():
    rows, width, topk = 356, 356, 2048
    logits = torch.randn(rows, 512, dtype=torch.bfloat16)[:, :width]
    lengths = torch.arange(1, rows + 1, dtype=torch.int32)
    starts = torch.zeros_like(lengths)
    offsets = torch.arange(rows, dtype=torch.int32) * width
    columns = torch.arange(topk, dtype=torch.int32).expand(rows, -1)
    expected = torch.where(columns < lengths[:, None], columns + offsets[:, None], -1)

    def kernel(scores, seq_lens, *, out_offsets, out_indices, row_starts):
        assert scores.shape == (356, 356)
        assert scores.dtype == torch.float32
        assert scores.stride(0) % 4 == 0
        assert torch.equal(scores, logits.float())
        assert seq_lens is lengths
        assert out_offsets is offsets
        assert row_starts is starts
        out_indices.copy_(expected)

    module_name = "sglang.kernels.ops.attention.dsv4.topk"
    kernel_module = ModuleType(module_name)
    kernel_module.topk_transform_ragged_v2 = kernel
    with (
        patch.dict(sys.modules, {module_name: kernel_module}),
        patch.object(dsa_topk_backend, "is_sail", return_value=True),
        dsa_topk_backend.envs.SGLANG_DSA_FUSE_TOPK.override(True),
        dsa_topk_backend.envs.SGLANG_OPT_USE_TOPK_V2.override(True),
    ):
        out = dsa_topk_backend.DSATopKBackend.SGL_KERNEL.topk_transform(
            logits,
            lengths,
            topk,
            dsa_topk_backend.TopkTransformMethod.RAGGED,
            attn_metadata=None,
            topk_indices_offset=offsets,
            row_starts=starts,
        )

    assert torch.equal(out, expected)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
