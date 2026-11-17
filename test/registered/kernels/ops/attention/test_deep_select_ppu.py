"""PPU wheel integration: BF16 row windows, offsets, alignment and padding."""

import subprocess
import sys
import textwrap
from unittest.mock import Mock

import pytest
import torch

from sglang.kernels.ops.attention.dsa.deep_select import topk_transform_ragged
from sglang.kernels.ops.attention.dsv4 import index_logits
from sglang.kernels.ops.attention.dsv4.topk import topk_transform_sparse
from sglang.srt.hardware_backend.ppu.attention import ppu_deep_select_hooks
from sglang.srt.utils import is_sail
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not is_sail(), reason="Requires the PPU deep_select wheel"
)


@pytest.fixture
def bf16_deep_select(monkeypatch):
    # Patch the cached resolver itself, not its environment input. Restoring the
    # patch leaves the original resolver and its existing cache untouched.
    monkeypatch.setattr(index_logits, "index_logits_dtype", lambda: torch.bfloat16)
    assert ppu_deep_select_hooks._ppu_is_deep_select_supported()


@pytest.mark.parametrize("topk", [1, 8, 512, 2048])
@pytest.mark.parametrize(
    "width,layout",
    [
        (356, "zero"),
        (1025, "none"),
        (8193, "unaligned"),
        (16385, "packed"),
        (65537, "packed"),
        (65537, "packed_compact"),
        (1025, "unaligned_storage"),
        (1025, "strided_columns"),
    ],
)
@pytest.mark.parametrize("distribution", ["random", "ties", "negative_inf"])
@torch.inference_mode()
def test_deep_select_ragged_windows(topk, width, layout, distribution):
    torch.manual_seed(width)
    rows, device = 4, "cuda"
    padded_width = (width + 511) // 512 * 512
    if layout == "unaligned_storage":
        scores = torch.empty(rows, padded_width, dtype=torch.bfloat16, device=device)[
            :, 1 : width + 1
        ]
    elif layout == "strided_columns":
        scores = torch.empty(
            rows, padded_width * 2, dtype=torch.bfloat16, device=device
        )[:, : width * 2 : 2]
    else:
        scores = torch.empty(rows, padded_width, dtype=torch.bfloat16, device=device)[
            :, :width
        ]
    scores.fill_(1000)
    if layout in ("none", "zero"):
        starts_cpu = [0, 0, 0, 0]
    elif layout in ("packed", "packed_compact"):
        starts_cpu = [0, 3, width // 4 + 1, width // 2 + 3]
    else:
        starts_cpu = [0, 1, 2, 3]
    lengths_cpu = [
        0,
        min(topk - 1, width - starts_cpu[1]),
        min(topk + 1, width - starts_cpu[2]),
        width - starts_cpu[3],
    ]
    for row, (start, length) in enumerate(zip(starts_cpu, lengths_cpu)):
        values = torch.randn(length, device=device)
        if distribution == "ties":
            values.fill_(0.5)
        elif distribution == "negative_inf":
            values.fill_(-torch.inf)
            values[: min(length, 3)] = torch.arange(min(length, 3), device=device)
        scores[row, start : start + length] = values
    before = scores.clone()
    starts = torch.tensor(starts_cpu, dtype=torch.int32, device=device)
    lengths = torch.tensor(lengths_cpu, dtype=torch.int32, device=device)
    offsets = torch.tensor([100, 20000, 40000, 60000], dtype=torch.int32, device=device)

    out = topk_transform_ragged(
        scores,
        lengths,
        topk,
        offsets,
        None if layout == "none" else starts,
        **({"max_seq_len": max(lengths_cpu)} if layout == "packed_compact" else {}),
    )
    torch.cuda.synchronize()
    assert out.shape == (rows, topk)
    assert out.dtype == torch.int32
    assert out.is_contiguous()
    assert torch.equal(scores, before)
    for row, (start, length) in enumerate(zip(starts_cpu, lengths_cpu)):
        valid = out[row] != -1
        assert int(valid.sum()) == min(topk, length)
        indices = out[row, valid].long() - offsets[row]
        assert bool(((indices >= 0) & (indices < length)).all())
        assert indices.unique().numel() == indices.numel()
        window = before[row, start : start + length].float()
        expected = window.topk(min(topk, length)).values.sort().values
        assert torch.equal(window[indices].sort().values, expected)


@torch.inference_mode()
def test_deep_select_empty_rows():
    scores = torch.empty((0, 1024), dtype=torch.bfloat16, device="cuda")
    lengths = torch.empty(0, dtype=torch.int32, device="cuda")
    out = topk_transform_ragged(scores, lengths, 2048, lengths)
    assert out.shape == (0, 2048)
    assert out.dtype == torch.int32


@pytest.mark.parametrize("topk", [8, 512, 2048])
@pytest.mark.parametrize("width", [1025, 4093, 16384])
@pytest.mark.parametrize("graph", [False, True])
@torch.inference_mode()
def test_deep_select_sparse_block_positions(topk, width, graph, bf16_deep_select):
    torch.manual_seed(width + topk)
    rows = 4
    stride = (width + 511) // 512 * 512
    scores = torch.randn(rows, stride, device="cuda", dtype=torch.bfloat16)[:, :width]
    # Include ties, an empty row, a short row and a strided valid-length view.
    scores[2].fill_(0.5)
    lengths_cpu = [0, min(topk - 1, width), width - 5, width]
    lengths = torch.zeros(rows * 2, device="cuda", dtype=torch.int32)[::2]
    lengths.copy_(torch.tensor(lengths_cpu, device="cuda", dtype=torch.int32))
    block_count = (width + 7) // 8
    blocks = torch.randperm(block_count, device="cuda", dtype=torch.int32)
    blocks = blocks[None, :].expand(rows, -1).contiguous() + 17
    out = torch.empty(rows, topk, device="cuda", dtype=torch.int32)

    def run():
        return ppu_deep_select_hooks._ppu_topk_transform_sparse(
            topk_transform_sparse, scores, lengths, blocks, out
        )

    def check_output():
        torch.cuda.synchronize()
        for row, length in enumerate(lengths_cpu):
            valid = out[row] != -1
            positions = out[row, valid].long()
            assert positions.numel() == min(topk, length)
            assert positions.unique().numel() == positions.numel()
            if positions.numel() == 0:
                continue
            logical_blocks = positions // 8
            compact_blocks = (logical_blocks[:, None] == blocks[row]).int().argmax(1)
            assert torch.equal(blocks[row, compact_blocks].long(), logical_blocks)
            compact = compact_blocks * 8 + positions % 8
            assert bool((compact < length).all())
            expected = scores[row, :length].float().topk(min(topk, length)).values
            actual = scores[row, compact].float()
            torch.testing.assert_close(actual.sort().values, expected.sort().values)

    assert run() is None
    check_output()
    if graph:
        # Warm both allocation and dispatch before entering the capture pool.
        run()
        torch.cuda.synchronize()
        capture = torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture):
            run()
        for replay in range(2):
            scores.copy_(torch.randn_like(scores))
            scores[2].fill_(0.5 + replay)
            out.fill_(-777)
            capture.replay()
            check_output()


def test_deep_select_sparse_consumer_hook():
    # Applying hooks mutates imported bindings process-wide. Exercise the actual
    # registration and the sparse-table consumer in a fresh process.
    subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import torch
                from unittest.mock import patch
                from sglang.kernels.ops.attention.dsv4 import index_logits, topk
                from sglang.srt.layers.attention.dsv4.v41_indexer import sparse_table
                from sglang.srt.hardware_backend.ppu.attention import ppu_deep_select_hooks as hooks
                from sglang.srt.plugins.hook_registry import HookRegistry, HookType

                target = "sglang.kernels.ops.attention.dsv4.topk.topk_transform_sparse"
                registered = HookRegistry._hooks.get(target, [])
                assert any(kind == HookType.AROUND and fn is hooks._ppu_topk_transform_sparse
                           for kind, fn, _ in registered), "sparse hook not registered"
                HookRegistry._hooks = {target: registered}
                HookRegistry._patched = set()
                before = [sparse_table.topk_transform_sparse]
                HookRegistry.apply_hooks()
                assert sparse_table.topk_transform_sparse is not before[0]
                assert sparse_table.topk_transform_sparse is topk.topk_transform_sparse

                scores = torch.tensor([[1, 4, 2, 3, 0, 0, 0, 0]], device="cuda", dtype=torch.bfloat16)
                lengths = torch.tensor([4], device="cuda", dtype=torch.int32)
                blocks = torch.tensor([[17]], device="cuda", dtype=torch.int32)
                out = torch.empty(1, 2, device="cuda", dtype=torch.int32)
                with patch.object(index_logits, "index_logits_dtype", return_value=torch.bfloat16), \
                     patch.object(topk, "topk_transform_bf16_small", side_effect=AssertionError("generic fallback used")):
                    assert hooks._ppu_is_deep_select_supported()
                    sparse_table.topk_transform_sparse(scores, lengths, blocks, out)
                torch.testing.assert_close(out.sort().values,
                    torch.tensor([[137, 139]], device="cuda", dtype=torch.int32))
                """
            ),
        ],
        check=True,
        timeout=180,
    )


@torch.inference_mode()
def test_deep_select_sparse_without_backend_uses_original(monkeypatch):
    monkeypatch.setattr(
        ppu_deep_select_hooks, "_ppu_is_deep_select_supported", lambda: False
    )
    scores = torch.tensor(
        [[1, 4, 2, 3, 0, 0, 0, 0]], device="cuda", dtype=torch.bfloat16
    )
    lengths = torch.tensor([4], device="cuda", dtype=torch.int32)
    blocks = torch.tensor([[17]], device="cuda", dtype=torch.int32)
    out = torch.empty(1, 2, device="cuda", dtype=torch.int32)
    original = Mock(wraps=topk_transform_sparse)
    ppu_deep_select_hooks._ppu_topk_transform_sparse(
        original, scores, lengths, blocks, out
    )
    original.assert_called_once_with(scores, lengths, blocks, out)
    torch.testing.assert_close(
        out.sort().values,
        torch.tensor([[137, 139]], device="cuda", dtype=torch.int32),
    )


@pytest.mark.parametrize("rows,topk", [(0, 512), (4, 0)])
@torch.inference_mode()
def test_deep_select_sparse_empty_output(rows, topk, bf16_deep_select):
    scores = torch.empty(rows, 1024, device="cuda", dtype=torch.bfloat16)
    lengths = torch.zeros(rows, device="cuda", dtype=torch.int32)
    blocks = torch.empty(rows, 128, device="cuda", dtype=torch.int32)
    out = torch.empty(rows, topk, device="cuda", dtype=torch.int32)
    original = Mock(side_effect=AssertionError("empty output used generic fallback"))
    assert (
        ppu_deep_select_hooks._ppu_topk_transform_sparse(
            original, scores, lengths, blocks, out
        )
        is None
    )
    original.assert_not_called()
    assert out.numel() == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
