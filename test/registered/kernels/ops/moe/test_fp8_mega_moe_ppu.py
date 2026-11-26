"""PR336: real channelwise W8A8 SwiGLU, with a one-PPU numerical oracle.

No router implementation is substituted: routes are explicit operator inputs.
The shared helpers also support the standalone EP check and operator benchmark.
"""

import json
import unittest

import torch

from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.kernels.fp8_mega_moe import (
    SingleRankGroup,
    check_output,
    load_deep_gemm,
    make_weights,
    native,
    reference,
)
from sglang.test.test_utils import CustomTestCase

register_ppu_ci(est_time=210, suite="stage-a-test-1-gpu-ppu")


class TestFP8MegaMoEPPU(CustomTestCase):
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
    def test_channelwise_swiglu_numerics(self):
        # Supported upstream smoke geometry, plus cached/tail quantization shapes.
        for experts, hidden, intermediate in (
            (8, 512, 256),
            (8, 512, 640),
            (16, 1024, 512),
        ):
            with self.subTest(E=experts, H=hidden, I=intermediate):
                rng = torch.Generator(device="cuda").manual_seed(336 + intermediate)
                capacity, topk = 129, 2
                raw1 = make_weights(experts, 2 * intermediate, hidden, rng)
                raw2 = make_weights(experts, hidden, intermediate, rng)
                l1, l2 = self.dg.transform_weights_for_fp8_mega_moe(
                    raw1, raw2, quantization="channelwise"
                )
                try:
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
                except (RuntimeError, ValueError, AssertionError) as exc:
                    # A build that only supports EP4/8 is a blocker, NOT a skip or
                    # a four-rank test mislabeled as one-GPU CI.
                    raise RuntimeError(
                        "PR336 one-PPU CI blocker: real single-rank FP8 workspace construction failed; "
                        "this test cannot be satisfied by launching EP4/8 on the one-GPU lane"
                    ) from exc
                try:
                    self.assertEqual(buffer.num_ranks, 1)
                    self.assertEqual(buffer.mma_type, "fp8xfp8")
                    self.assertEqual(buffer.quantization, "channelwise")
                    source = (
                        torch.randn(
                            (capacity, hidden),
                            device="cuda",
                            dtype=torch.bfloat16,
                            generator=rng,
                        )
                        * 0.1
                    )
                    for iteration, tokens in enumerate((1, 33, 129, 0, 8)):
                        with self.subTest(tokens=tokens):
                            x = source[:tokens].clone()
                            if tokens > 1:
                                x[0].zero_()
                            if iteration == 2:
                                x.mul_(
                                    128
                                )  # exercise gate/up clipping, not just small activations
                            rows = torch.arange(tokens, device="cuda")[:, None]
                            slots = torch.arange(topk, device="cuda")[None, :]
                            ids = (
                                ((rows + slots + iteration) % experts)
                                .long()
                                .contiguous()
                            )
                            if iteration == 2:
                                ids[:] = slots  # hot experts after random/sparse calls
                            weights = torch.rand(
                                (tokens, topk), device="cuda", generator=rng
                            )
                            if tokens > 1:
                                ids[1] = (
                                    -1
                                )  # entire invalid route row must produce zero
                                weights[-1, 0] = 0  # valid, zero-weight assignment
                            out = torch.full(
                                (tokens, hidden),
                                float("nan"),
                                device="cuda",
                                dtype=torch.bfloat16,
                            )
                            actual = native(
                                self.dg, buffer, x, ids, weights, l1, l2, out=out
                            )
                            expected = reference(
                                x[None], ids[None], weights[None], raw1, raw2
                            )[0]
                            torch.cuda.synchronize()
                            report = check_output(actual, expected)
                            if tokens:
                                self.assertGreater(
                                    expected.float().square().sum().item(), 0
                                )
                            if tokens > 1:
                                self.assertEqual(actual[0].count_nonzero().item(), 0)
                                self.assertEqual(actual[1].count_nonzero().item(), 0)
                            print(
                                json.dumps(
                                    dict(
                                        E=experts,
                                        H=hidden,
                                        I=intermediate,
                                        tokens=tokens,
                                        **report,
                                    )
                                ),
                                flush=True,
                            )
                finally:
                    buffer.destroy()


if __name__ == "__main__":
    unittest.main()
