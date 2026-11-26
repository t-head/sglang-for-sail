"""Replica loading must cover the entire physical placement."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.eplb.expert_location import ExpertLocationMetadata
from sglang.test.test_utils import CustomTestCase


class TestReplicaLoading(CustomTestCase):
    def test_loading_uses_all_resident_copies_not_dispatch_subset(self):
        physical = torch.tensor([[0, 1, 0, 2, 3, 2]])
        # The nearest-replica dispatch map intentionally omits two resident slots.
        logical = torch.full((1, 4, 6), -1, dtype=torch.int64)
        logical[0, :, 0] = torch.tensor([0, 1, 3, 4])
        metadata = ExpertLocationMetadata(
            physical_to_logical_map=physical,
            physical_to_logical_map_cpu=physical,
            logical_to_all_physical_map=logical,
            logical_to_all_physical_map_cpu=logical,
            logical_to_all_physical_map_num_valid=(logical != -1).sum(-1),
            ep_size=2,
            logical_to_rank_dispatch_physical_map=None,
        )
        loaded = torch.full((6,), -1)
        for expert in range(4):
            loaded[metadata.logical_to_all_physical(0, expert)] = expert
        torch.testing.assert_close(loaded, physical[0])
        self.assertEqual(metadata.logical_to_all_physical(0, 0), [0, 2])
        self.assertEqual(metadata.logical_to_all_physical(1, 2), [2])
        self.assertEqual(metadata.logical_to_all_physical(0, 0, True), [0, 4])
        with self.assertRaises(IndexError):
            metadata.logical_to_all_physical(0, 4)


if __name__ == "__main__":
    unittest.main()
