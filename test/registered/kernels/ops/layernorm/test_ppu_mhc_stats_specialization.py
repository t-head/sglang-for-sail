"""PPU regression for runtime row counts in the mHC Stats kernels."""

import math
import unittest
from collections import defaultdict
from contextlib import ExitStack
from unittest.mock import patch

import torch

from sglang.kernels.ops.layernorm import mhc
from sglang.srt.utils import is_sail
from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.test_utils import CustomTestCase

register_ppu_ci(est_time=60, suite="stage-a-test-1-gpu-ppu")


@unittest.skipUnless(torch.cuda.is_available() and is_sail(), "requires a PPU")
class TestPpuMhcStatsSpecialization(CustomTestCase):
    def test_reuse_for_different_row_counts_in_each_tile(self):
        # Within each group, M changes but BLOCK_M does not. Use both aligned
        # and unaligned lengths so the test rejects the old M specialization.
        row_groups = ((1, 7, 8), (9, 16, 17, 32, 2048), (2049, 2064))
        k, mix, eps = 20480, 24, 1e-6
        generator = torch.Generator(device="cuda").manual_seed(72)
        x = torch.randn(
            (2064, k), device="cuda", dtype=torch.bfloat16, generator=generator
        )
        w = torch.randn(
            (mix, k), device="cuda", dtype=torch.float32, generator=generator
        ) / math.sqrt(k)
        ref_w = w.cpu().double()
        kernels = (mhc._hc_mix_stats_partial_kernel, mhc._hc_mix_stats_reduce_kernel)
        device = torch.cuda.current_device()
        first_row = None

        with ExitStack() as stack:
            for kernel in kernels:
                stack.enter_context(
                    patch.object(
                        kernel, "device_caches", defaultdict(kernel.create_binder)
                    )
                )
            caches = [kernel.device_caches[device][0] for kernel in kernels]
            for expected_variants, rows_group in enumerate(row_groups, start=1):
                for rows in rows_group:
                    with self.subTest(rows=rows):
                        out = mhc.hc_mix_stats(x[:rows], w, eps)
                        torch.cuda.synchronize()
                        self.assertEqual(
                            [len(cache) for cache in caches], [expected_variants] * 2
                        )
                        self.assertEqual(out.shape, (rows, mix))
                        if first_row is None:
                            first_row = out[0].clone()
                        else:
                            torch.testing.assert_close(out[0], first_row, atol=0, rtol=0)
                        # Check the last row as well: that reaches the mask in
                        # every non-aligned batch, and weight/scale are unchanged.
                        ref_x = x[rows - 1].cpu().double()
                        ref = (ref_w @ ref_x) * torch.rsqrt(ref_x.square().mean() + eps)
                        torch.testing.assert_close(
                            out[rows - 1].cpu().double(), ref, atol=2e-4, rtol=2e-4
                        )


if __name__ == "__main__":
    unittest.main()
