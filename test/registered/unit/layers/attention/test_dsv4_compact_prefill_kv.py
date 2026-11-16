import ast
import gc
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestCompactPrefillKV(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        path = (
            Path(__file__).resolve().parents[5]
            / "python/sglang/srt/layers/attention/dsv4/sparse_prefill_utils.py"
        )
        # Load the production helpers without importing the GPU-only dequantizer.
        names = {
            "SparsePrefillKVBuffer",
            "should_compact_sparse_prefill_kv",
            "compact_sparse_prefill_kv",
        }
        tree = ast.parse(path.read_text(encoding="utf-8"))
        nodes = [node for node in tree.body if getattr(node, "name", None) in names]
        future = ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
        module = ast.fix_missing_locations(
            ast.Module(body=[future, *nodes], type_ignores=[])
        )
        cls.utils = SimpleNamespace(
            torch=torch, WORKSPACE_DIM=512, dequantize_k_cache_paged_dsv4=None
        )
        exec(compile(module, str(path), "exec"), vars(cls.utils))

    def test_buffer_reuse_requires_bf16_and_sufficient_capacity(self):
        buffer = self.utils.SparsePrefillKVBuffer(torch.device("cpu"))
        self.assertFalse(buffer.can_reuse_bf16(4))
        workspace = buffer.get(4)
        self.assertTrue(buffer.can_reuse_bf16(4))
        self.assertFalse(buffer.can_reuse_bf16(5))
        self.assertEqual(buffer.get(3).data_ptr(), workspace.data_ptr())
        buffer.get(4, dtype=torch.float32)
        self.assertFalse(buffer.can_reuse_bf16(4))

    def test_selection_keeps_order_and_duplicates(self):
        utils = self.utils
        compressed = torch.tensor([4, 2], dtype=torch.int32)
        swa = torch.tensor([5, 1, 3], dtype=torch.int32)
        indices = torch.tensor([[4, 0, 4, -1, 1, 99], [99] * 6], dtype=torch.int32)

        def gather(cache, slots, *, out, **_):
            out.copy_(cache[slots.long()])

        table = torch.arange(8, dtype=torch.bfloat16).view(8, 1, 1)
        table = table.expand(-1, 1, utils.WORKSPACE_DIM)
        with patch.object(
            utils, "dequantize_k_cache_paged_dsv4", side_effect=gather
        ) as fn:
            workspace, remapped = utils.compact_sparse_prefill_kv(
                indices,
                torch.tensor([5, 0]),
                compressed,
                swa,
                table,
                table + 10,
                8,
                8,
                None,
                None,
            )
            self.assertEqual(fn.call_count, 2)
        self.assertEqual(remapped.tolist(), [[2, 0, 2, -1, 1, -1], [-1] * 6])
        self.assertEqual(workspace[:, 0, 0].tolist(), [4, 2, 13])

    def test_release_only_discards_eager_bf16_and_forgets_head(self):
        buffer = self.utils.SparsePrefillKVBuffer(torch.device("cpu"))
        buffer.get_with_head(4, ("source", 0))
        buffer.release_eager_bf16()
        self.assertIsNone(buffer._buffer)
        self.assertIsNone(buffer._head_source)
        _, filled = buffer.get_with_head(4, ("source", 0))
        self.assertFalse(filled)

        buffer.get(4, dtype=torch.float32)
        non_bf16 = buffer._buffer
        buffer.release_eager_bf16()
        self.assertIs(buffer._buffer, non_bf16)

    def test_graph_storage_survives_growth_and_eager_release(self):
        buffer = self.utils.SparsePrefillKVBuffer(torch.device("cpu"))
        captured = buffer.get(4, retain_for_graph=True)
        captured_address = captured.data_ptr()
        captured_storage = weakref.ref(buffer._buffer)
        del captured
        buffer.get(3, retain_for_graph=True)
        self.assertEqual(len(buffer._graph_buffers), 1)
        buffer.release_eager_bf16()
        self.assertEqual(buffer._buffer.data_ptr(), captured_address)

        # No caller keeps the original tensor/view alive. The buffer owner must
        # retain its address when another eager shape replaces current scratch.
        buffer.get(8)
        gc.collect()
        self.assertIsNotNone(captured_storage())
        self.assertEqual(captured_storage().data_ptr(), captured_address)
        buffer.release_eager_bf16()
        self.assertIsNone(buffer._buffer)
        self.assertIsNotNone(captured_storage())
        self.assertEqual(len(buffer._graph_buffers), 1)

    def test_graph_head_storage_survives_dtype_and_capacity_changes(self):
        buffer = self.utils.SparsePrefillKVBuffer(torch.device("cpu"))
        head = ("source", 1)
        _, filled = buffer.get_with_head(4, head, retain_for_graph=True)
        self.assertFalse(filled)
        first = weakref.ref(buffer._buffer)
        _, filled = buffer.get_with_head(3, head, retain_for_graph=True)
        self.assertTrue(filled)
        self.assertEqual(len(buffer._graph_buffers), 1)

        # Changing dtype replaces scratch, but not the allocation kept for the
        # first graph. A later graph needs its own retained BF16 allocation.
        buffer.get(4, dtype=torch.float32)
        _, filled = buffer.get_with_head(8, head, retain_for_graph=True)
        self.assertFalse(filled)
        second = weakref.ref(buffer._buffer)
        self.assertEqual(len(buffer._graph_buffers), 2)
        self.assertIsNot(first(), second())
        del _
        buffer.get(16)
        buffer.release_eager_bf16()
        gc.collect()
        self.assertIsNone(buffer._buffer)
        self.assertIsNone(buffer._head_source)
        self.assertIsNotNone(first())
        self.assertIsNotNone(second())

    def test_empty_and_swa_only(self):
        utils = self.utils
        slots = torch.tensor([2, 1], dtype=torch.int32)
        table = torch.ones(3, 1, utils.WORKSPACE_DIM, dtype=torch.bfloat16)
        indices = torch.tensor([[0, -1]], dtype=torch.int32)

        def gather(cache, slots, *, out, **_):
            out.copy_(cache[slots.long()])

        with patch.object(
            utils, "dequantize_k_cache_paged_dsv4", side_effect=gather
        ) as fn:
            for length in (0, 1):
                with self.subTest(length=length):
                    workspace, remapped = utils.compact_sparse_prefill_kv(
                        indices,
                        torch.tensor([length]),
                        None,
                        slots,
                        None,
                        table,
                        None,
                        8,
                        None,
                        None,
                    )
                    self.assertEqual(remapped.tolist(), [[0 if length else -1, -1]])
                    self.assertEqual(workspace[0, 0, 0].item(), length)
            self.assertEqual(fn.call_count, 1)

    def test_large_workspaces_only(self):
        utils = self.utils
        threshold = (4 << 30) // (2 * utils.WORKSPACE_DIM)
        self.assertFalse(utils.should_compact_sparse_prefill_kv(threshold - 1, 1))
        self.assertTrue(
            utils.should_compact_sparse_prefill_kv(threshold, threshold // 2)
        )
        self.assertFalse(
            utils.should_compact_sparse_prefill_kv(threshold, threshold // 2 + 1)
        )


if __name__ == "__main__":
    unittest.main()
