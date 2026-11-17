"""Check that fused index-Q preparation agrees with the split operator chain."""

import math
import unittest

import torch

from sglang.kernels.ops.attention.dsv4.fp4_indexer_rope import (
    index_q_rope_pack_weights,
)
from sglang.kernels.ops.quantization.fp4_indexer_quant import (
    quantize_fp4_indexer_tensor,
)
from sglang.srt.layers.attention.dsv4.dsv41_sparse import _rope_fq4
from sglang.srt.utils import is_sail
from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.test_utils import CustomTestCase

register_ppu_ci(est_time=60, suite="stage-a-test-1-gpu-ppu")


def make_index_q_inputs(rows, heads, max_pos, pos_dtype):
    q = torch.randn(rows, heads, 128, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(rows, heads, device="cuda", dtype=torch.bfloat16)
    angles = torch.randn(max_pos, 32, device="cuda")
    freqs = torch.polar(torch.ones_like(angles), angles)
    # Non-monotonic positions distinguish a table lookup from a row lookup.
    positions = torch.randint(max_pos, (rows,), device="cuda", dtype=pos_dtype)
    return q, freqs, positions, weights, 128**-0.5 * heads**-0.5


def run_split_index_q(q, freqs, positions, raw_weights, weight_scale):
    rotated = _rope_fq4(q, freqs[positions], 64)
    payload, scales = quantize_fp4_indexer_tensor(rotated.flatten(0, 1), rne=True)
    return payload, scales, (raw_weights * weight_scale).float()


def run_fused_index_q(q, freqs, positions, raw_weights, weight_scale):
    return index_q_rope_pack_weights(
        q,
        torch.view_as_real(freqs).flatten(-2),
        positions,
        raw_weights,
        weight_scale,
    )


def _assert_bytes_equal(actual, expected, case):
    for name, got, ref in zip(("payload", "scales", "weights"), actual, expected):
        if got.shape != ref.shape or got.dtype != ref.dtype:
            raise AssertionError(f"{case}: {name} shape/dtype mismatch")
        # Byte views also catch signed-zero differences in the FP32 weights.
        a, b = got.contiguous().view(torch.uint8), ref.contiguous().view(torch.uint8)
        mismatches = int((a != b).sum().item())
        if mismatches:
            raise AssertionError(f"{case}: {name}: {mismatches} bytes differ")


def _check_boundaries(heads):
    # Fix each block's amax at 6, making its fake-quant scale exactly 1.
    # The other entries then exercise every E2M1 tie, both signs and signed zero.
    values = torch.tensor(
        [0.0, -0.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0],
        device="cuda",
    )
    # Immediate BF16 neighbors distinguish the strict and inclusive midpoint
    # rounding decisions after the RoPE BF16 round trip.
    ties = values[2:-1].to(torch.bfloat16)
    values = torch.cat(
        (
            values,
            torch.nextafter(ties, torch.full_like(ties, float("-inf"))).float(),
            torch.nextafter(ties, torch.full_like(ties, float("inf"))).float(),
        )
    )
    values = torch.cat((values, -values))
    q, freqs, positions, weights, scale = make_index_q_inputs(7, heads, 8, torch.int64)
    freqs.fill_(1 + 0j)
    q.copy_(
        values.repeat(math.ceil(q.numel() / values.numel()))[: q.numel()].view_as(q)
    )
    q.unflatten(-1, (4, 32))[..., -1] = 6
    for magnitude in (0.0, 1.0, 2**-16, 2**-120, 2**-130, 2**8):
        case_q = q * magnitude
        # Flash's 32-head scale is exactly 1/64. A non-power-of-two scale
        # additionally guards the BF16 rounding before widening weights to FP32.
        for weight_scale in (scale, scale * math.sqrt(2)):
            _assert_bytes_equal(
                run_fused_index_q(case_q, freqs, positions, weights, weight_scale),
                run_split_index_q(case_q, freqs, positions, weights, weight_scale),
                f"E2M1 boundaries, magnitude={magnitude}, weight_scale={weight_scale}",
            )


@unittest.skipUnless(is_sail() and torch.cuda.is_available(), "requires a PPU")
class TestIndexQRoPEPackWeights(CustomTestCase):
    def test_shapes_and_position_dtypes(self):
        torch.manual_seed(1234)
        # Exercise empty dispatch, CTA tails and power-of-two/non-power-of-two
        # head counts. 513 tokens also crosses the usual 512-token test point.
        for tokens, heads in ((0, 3), (1, 1), (3, 3), (7, 5), (17, 32), (513, 64)):
            for dtype in (torch.int32, torch.int64):
                with self.subTest(tokens=tokens, heads=heads, positions=dtype):
                    q, freqs, pos, weights, scale = make_index_q_inputs(
                        tokens, heads, 8192, dtype
                    )
                    if tokens >= 3:
                        pos[:3] = pos.new_tensor([8191, 0, 8191])
                    args = (q, freqs, pos, weights, scale)
                    _assert_bytes_equal(
                        run_fused_index_q(*args), run_split_index_q(*args), "index-Q"
                    )

    def test_e2m1_midpoints_and_scale_floors(self):
        torch.manual_seed(1234)
        for heads in (3, 32):
            with self.subTest(heads=heads):
                _check_boundaries(heads)


if __name__ == "__main__":
    unittest.main()
