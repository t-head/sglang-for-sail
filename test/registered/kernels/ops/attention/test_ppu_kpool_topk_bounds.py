"""Unclean MQA logits must never expose another request's score columns."""

import unittest

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.kpool_fp8_index import (
    topk_from_pooled_history_logits,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-small")


class TestKPoolTopkBounds(CustomTestCase):
    def test_poisoned_prefix_and_suffix_are_not_selected(self):
        if not torch.cuda.is_available():
            self.skipTest("requires device")
        starts = torch.tensor([0, 128, 256, 512], dtype=torch.int32, device="cuda")
        lengths = torch.tensor([64, 64, 128, 1], dtype=torch.int32, device="cuda")
        scores = torch.full((4, 640), float("nan"), device="cuda")
        for row, (start, length) in enumerate(zip(starts.tolist(), lengths.tolist())):
            scores[row, :start] = 1e30
            scores[row, start : start + length] = torch.arange(
                length, device="cuda"
            ).float()
        with envs.SGLANG_OPT_USE_TOPK_V2.override(False):
            out = topk_from_pooled_history_logits(
                scores, lengths, kpool=4, topk=512, row_starts=starts
            )
        for row, length in enumerate(lengths.tolist()):
            valid = out[row][out[row] >= 0]
            self.assertEqual(sorted(valid.cpu().tolist()), list(range(length * 4)))
            self.assertEqual(int((out[row] == -1).sum()), 512 - length * 4)


if __name__ == "__main__":
    unittest.main()
