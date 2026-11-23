"""PPU regression tests for private DSPARK sampling warmup."""

import unittest

import torch

from sglang.kernels.ops.speculative.dspark.dspark_accept import (
    gather_two_level_bonus_triton,
)
from sglang.srt.speculative.dspark_components.dspark_sampling_warmup import (
    prewarm_dspark_sampling,
)
from sglang.srt.utils import is_sail
from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.test_utils import CustomTestCase

register_ppu_ci(est_time=15, suite="stage-a-test-1-gpu-ppu")


@unittest.skipUnless(is_sail() and torch.cuda.is_available(), "PPU only")
class TestDsparkSamplingWarmupPPU(CustomTestCase):
    def test_private_onehot_classic_outputs_rng_and_selected_stream(self):
        device = torch.device("cuda", torch.cuda.current_device())
        stream = torch.cuda.Stream(device=device)
        for block in (False, True):
            with self.subTest(block_verification=block), torch.cuda.stream(stream):
                cpu_rng = torch.random.get_rng_state().clone()
                gpu_rng = torch.cuda.get_rng_state(device).clone()
                original_stream = torch.cuda.current_stream(device)
                predicts, indices, num_correct_drafts, bonus_tokens = (
                    prewarm_dspark_sampling(
                        gamma=2,
                        verify_slots=3,
                        vocab_size=257,
                        device=device,
                        stream=stream,
                        block_verification=block,
                    )
                )
                self.assertEqual(torch.cuda.current_stream(device), original_stream)
                self.assertTrue(torch.equal(torch.random.get_rng_state(), cpu_rng))
                self.assertTrue(torch.equal(torch.cuda.get_rng_state(device), gpu_rng))
                torch.testing.assert_close(
                    predicts, torch.zeros_like(predicts), rtol=0, atol=0
                )
                torch.testing.assert_close(
                    indices, torch.tensor([[0, 1, 2]], device=device, dtype=torch.int32)
                )
                torch.testing.assert_close(
                    num_correct_drafts,
                    torch.tensor([2], device=device, dtype=torch.int32),
                )
                torch.testing.assert_close(
                    bonus_tokens, torch.tensor([0], device=device, dtype=torch.int64)
                )

    def test_bonus_full_finite_batch_domain_values(self):
        # Finite numerical coverage only. Compiler classes and cold-cache keys
        # are verified by the independent native driver, not Python object IDs.
        capacity, slots = 257, 8
        device = torch.device("cuda", torch.cuda.current_device())
        indices = torch.arange(capacity * slots, device=device, dtype=torch.int32).view(
            capacity, slots
        )
        predicts = torch.arange(capacity * slots, device=device, dtype=torch.int32) + 13
        correct = torch.arange(capacity, device=device, dtype=torch.int32) % slots
        for batch in range(1, capacity + 1):
            actual = gather_two_level_bonus_triton(
                accept_index=indices[:batch],
                predicts=predicts,
                correct_len=correct[:batch],
            )
            expected = predicts[
                indices[:batch]
                .gather(1, correct[:batch].long().unsqueeze(1))
                .flatten()
                .long()
            ].long()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
