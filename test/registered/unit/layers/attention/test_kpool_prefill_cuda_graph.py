"""Live KPool bridge outputs across active and DP-padded idle replays."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.dsa.kpool_prefill_cuda_graph import (
    _kpool_indexer_prefill_with_output,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    set_tc_piecewise_forward_context,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestKPoolPrefillCudaGraph(CustomTestCase):
    def _batch(self, logical_rows, original_mode=None):
        return ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            _original_forward_mode=original_mode,
            batch_size=1,
            input_ids=torch.zeros(48, dtype=torch.int64),
            req_pool_indices=torch.zeros(1, dtype=torch.int64),
            seq_lens=torch.tensor([logical_rows], dtype=torch.int64),
            out_cache_loc=torch.arange(48),
            seq_lens_sum=logical_rows,
            extend_num_tokens=48,
            extend_seq_lens_cpu=[logical_rows],
        )

    def _run(self, indexer, batch, output):
        with set_tc_piecewise_forward_context(batch, [], None, [], []):
            _kpool_indexer_prefill_with_output(
                indexer,
                torch.zeros(48, 2),
                torch.zeros(48, 2),
                torch.arange(48),
                output,
                0,
            )

    def test_active_idle_active_replay_clears_stale_indices(self):
        # The idle rank is relabeled EXTEND48 for DP collectives, while the
        # backend retains six-row capture metadata from the preceding graph.
        output = torch.empty(48, 2051, dtype=torch.int32)
        writes = []
        values = torch.arange(6 * 2051, dtype=torch.int32).view(6, 2051)

        def forward(**kwargs):
            writes.append(kwargs["layer_id"])
            return values

        indexer = SimpleNamespace(_forward_cuda_impl=forward)
        self._run(indexer, self._batch(6), output)
        torch.testing.assert_close(output[:6], values)
        self.assertTrue(torch.all(output[6:] == -1))
        address = output.data_ptr()

        self._run(indexer, self._batch(48, ForwardMode.IDLE), output)
        self.assertTrue(torch.all(output == -1))
        self.assertEqual(writes, [0])
        self.assertEqual(output.data_ptr(), address)

        values = values[:3] + 100
        self._run(indexer, self._batch(3), output)
        torch.testing.assert_close(output[:3], values)
        self.assertTrue(torch.all(output[3:] == -1))
        self.assertEqual(writes, [0, 0])
        self.assertEqual(output.data_ptr(), address)

    def test_idle_replay_without_returned_indices_has_no_cache_writes(self):
        cache = torch.full((6, 2), 17.0)

        def forward(**kwargs):
            cache.zero_()

        output = torch.empty(0, 2051, dtype=torch.int32)
        self._run(
            SimpleNamespace(_forward_cuda_impl=forward),
            self._batch(48, ForwardMode.IDLE),
            output,
        )
        self.assertTrue(torch.all(cache == 17))
        self.assertEqual(output.numel(), 0)

    def test_active_reuse_layer_still_updates_cache(self):
        cache = torch.full((6, 2), 17.0)

        def forward(**kwargs):
            self.assertFalse(kwargs["return_indices"])
            cache.copy_(kwargs["x"][:6])

        output = torch.empty(0, 2051, dtype=torch.int32)
        self._run(SimpleNamespace(_forward_cuda_impl=forward), self._batch(6), output)
        self.assertTrue(torch.all(cache == 0))

    def test_active_replay_rejects_malformed_topk(self):
        for shape in ((5, 2051), (49, 2051), (6, 2048)):
            with self.subTest(shape=shape):
                indexer = SimpleNamespace(
                    _forward_cuda_impl=lambda **kwargs: torch.zeros(
                        shape, dtype=torch.int32
                    )
                )
                with self.assertRaisesRegex(ValueError, "unexpected top-k shape"):
                    self._run(
                        indexer,
                        self._batch(6),
                        torch.full((48, 2051), -1, dtype=torch.int32),
                    )


if __name__ == "__main__":
    unittest.main()
