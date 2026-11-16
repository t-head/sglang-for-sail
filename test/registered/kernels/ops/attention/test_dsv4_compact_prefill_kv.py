import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.dsv4 import sparse_prefill_utils as utils
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestCompactPrefillKV(CustomTestCase):
    def test_selection_keeps_order_and_duplicates(self):
        compressed = torch.tensor([4, 2], dtype=torch.int32)
        swa = torch.tensor([5, 1, 3], dtype=torch.int32)
        indices = torch.tensor([[4, 0, 4, -1, 1, 99], [99] * 6], dtype=torch.int32)

        def gather(cache, slots, *, out, **_):
            out.copy_(cache[slots.long()])

        table = torch.arange(8, dtype=torch.bfloat16).view(8, 1, 1)
        table = table.expand(-1, 1, utils.WORKSPACE_DIM)
        with patch.object(utils, "dequantize_k_cache_paged_dsv4", side_effect=gather) as fn:
            workspace, remapped = utils.compact_sparse_prefill_kv(
                indices, torch.tensor([5, 0]), compressed, swa, table, table + 10,
                8, 8, None, None,
            )
            self.assertEqual(fn.call_count, 2)
        self.assertEqual(remapped.tolist(), [[2, 0, 2, -1, 1, -1], [-1] * 6])
        self.assertEqual(workspace[:, 0, 0].tolist(), [4, 2, 13])

    def test_empty_and_swa_only(self):
        slots = torch.tensor([2, 1], dtype=torch.int32)
        table = torch.ones(3, 1, utils.WORKSPACE_DIM, dtype=torch.bfloat16)
        indices = torch.tensor([[0, -1]], dtype=torch.int32)

        def gather(cache, slots, *, out, **_):
            out.copy_(cache[slots.long()])

        with patch.object(utils, "dequantize_k_cache_paged_dsv4", side_effect=gather) as fn:
            for length in (0, 1):
                with self.subTest(length=length):
                    workspace, remapped = utils.compact_sparse_prefill_kv(
                        indices, torch.tensor([length]), None, slots, None, table,
                        None, 8, None, None,
                    )
                    self.assertEqual(remapped.tolist(), [[0 if length else -1, -1]])
                    self.assertEqual(workspace[0, 0, 0].item(), length)
            self.assertEqual(fn.call_count, 1)

    def test_large_workspaces_only(self):
        threshold = (4 << 30) // (2 * utils.WORKSPACE_DIM)
        self.assertFalse(utils.should_compact_sparse_prefill_kv(threshold - 1, 1))
        self.assertTrue(utils.should_compact_sparse_prefill_kv(threshold, threshold // 2))
        self.assertFalse(
            utils.should_compact_sparse_prefill_kv(threshold, threshold // 2 + 1)
        )


if __name__ == "__main__":
    unittest.main()
