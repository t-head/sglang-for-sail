"""Runtime Router row alignment must not change math or compile another kernel."""

import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.moe.moe_fused_gate import (
    _router_triton_kernel,
    moe_fused_gate,
)
from sglang.test.ci.ci_register import register_cuda_ci, register_ppu_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_ppu_ci(est_time=15, suite="stage-a-test-1-gpu-ppu")


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA-compatible device")
class TestMoeRouterRowAlignment(CustomTestCase):
    @torch.inference_mode()
    def test_aligned_unaligned_aligned_reuses_kernel(self):
        rows, experts, topk = (128, 129, 144), 384, 6
        generator = torch.Generator().manual_seed(54119)
        logits = torch.randn(max(rows), experts, generator=generator) * 0.2
        # Well-separated winners avoid making the test sensitive to near ties
        # between CPU and device transcendental implementations.
        winners = torch.tensor([3, 17, 80, 127, 255, 383])
        logits[:, winners] = torch.arange(topk).float() * 0.4 + 3.0
        bias_cpu = torch.linspace(-0.02, 0.02, experts)
        activated = torch.nn.functional.softplus(logits).sqrt()
        order = torch.argsort(activated + bias_cpu, dim=1, descending=True)
        expected_ids = order[:, :topk].int()
        expected_weights = activated.gather(1, order[:, :topk])
        expected_weights /= expected_weights.sum(dim=1, keepdim=True)

        scores, bias = logits.cuda(), bias_cpu.cuda()
        live_rows = torch.empty(1, device="cuda", dtype=torch.int32)
        sentinel = 99123
        packed_storage = torch.full(
            (max(rows), 16), sentinel, device="cuda", dtype=torch.int32
        )
        launched = []
        run = _router_triton_kernel.run

        def record_kernel(*args, **kwargs):
            kernel = run(*args, **kwargs)
            launched.append(kernel)
            return kernel

        # Observe the compiled handles returned by real Triton launches, not
        # filenames in a backend-specific disk cache. Keep all pointer layouts
        # and constexpr arguments fixed; only runtime M changes alignment.
        with patch.object(_router_triton_kernel, "run", side_effect=record_kernel):
            for n in rows:
                with self.subTest(rows=n):
                    live_rows.fill_(n - 1)
                    packed = packed_storage[:n, :topk]
                    weights, indices = moe_fused_gate(
                        scores[:n],
                        bias,
                        topk=topk,
                        scoring_func="sqrtsoftplus",
                        sqrtsoftplus_log1p=True,
                        num_token_non_padded=live_rows,
                        packed_out=packed,
                    )
                    torch.cuda.synchronize()
                    actual_w, actual_i = weights.cpu(), indices.cpu()
                    ref_w = expected_weights[:n].clone()
                    ref_i = expected_ids[:n].clone()
                    ref_w[-1] = 0
                    ref_i[-1] = -1
                    torch.testing.assert_close(actual_w, ref_w, rtol=5e-5, atol=5e-6)
                    torch.testing.assert_close(actual_i, ref_i, rtol=0, atol=0)
                    repacked = (actual_i << 16) | (
                        actual_w.to(torch.bfloat16).view(torch.int16).int() & 0xFFFF
                    )
                    torch.testing.assert_close(packed.cpu(), repacked, rtol=0, atol=0)
                    self.assertTrue(
                        torch.all(packed_storage[:n, topk:] == sentinel).item()
                    )

        self.assertEqual(len(launched), len(rows))
        self.assertIsNotNone(launched[0])
        for kernel in launched[1:]:
            self.assertIs(kernel, launched[0], "runtime M alignment recompiled Router")


if __name__ == "__main__":
    unittest.main()
