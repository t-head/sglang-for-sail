"""Check the grouped DeepSeek V4 Q-only RoPE against the Q+K launch."""

import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.dsv4 import norm_rope
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_ppu_ci

register_ppu_ci(est_time=30, suite="stage-b-test-1-gpu-ppu")


@unittest.skipUnless(torch.cuda.is_available(), "requires a device")
class TestQOnlyRoPE(unittest.TestCase):
    def test_strided_q_and_dispatch_boundaries(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                with envs.SGLANG_OPT_DSV4_Q_ONLY_ROPE_FREQ_REUSE.override(enabled):
                    module = norm_rope._jit_fused_rope_module.__wrapped__()
                with patch.object(norm_rope, "_jit_fused_rope_module", return_value=module):
                    self._check_strided_q_and_dispatch_boundaries()

    def _check_strided_q_and_dispatch_boundaries(self):
        torch.manual_seed(1733)
        phase = torch.randn((521, 32), device="cuda")
        freqs = torch.polar(torch.ones_like(phase), phase)
        scenarios = [(255, 64), (256, 64), (257, 128), (256, 8), (256, 16), (256, 32)]
        for rows, heads in scenarios:
            for dtype in (torch.int32, torch.int64):
                for inverse in (False, True):
                    with self.subTest(rows=rows, heads=heads, dtype=dtype, inverse=inverse):
                        q = torch.randn(
                            (rows, heads, 512), dtype=torch.bfloat16, device="cuda"
                        )
                        expected = q.clone()
                        k = torch.zeros_like(q[:, :1, -64:]).contiguous()
                        pos = torch.arange(521 - rows, 521, dtype=dtype, device="cuda")
                        norm_rope.fused_rope_inplace(expected[..., -64:], k, freqs, pos, inverse)
                        norm_rope.fused_rope_inplace(q[..., -64:], None, freqs, pos, inverse)
                        self.assertTrue(
                            torch.equal(q.view(torch.int16), expected.view(torch.int16))
                        )


if __name__ == "__main__":
    unittest.main()
