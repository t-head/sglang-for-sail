"""Metadata ownership must participate in decode TP status consensus."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.disaggregation import utils as disagg_utils
from sglang.srt.disaggregation.base import KVPoll
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestMetadataPollGate(CustomTestCase):
    def _poll(self, actual_room, *, staged=False, fake=False, poll=KVPoll.Success):
        receiver = Mock(require_staging=staged)
        receiver.poll.return_value = poll
        decode_req = SimpleNamespace(
            req=SimpleNamespace(
                rid="request-7",
                bootstrap_room=7,
                bootstrap_host=(
                    disagg_utils.FAKE_BOOTSTRAP_HOST if fake else "127.0.0.1"
                ),
            ),
            metadata_buffer_index=1,
            kv_receiver=receiver,
        )
        metadata = SimpleNamespace(
            bootstrap_room=torch.tensor([[0], [actual_room]], dtype=torch.int64)
        )
        group = object()
        with (
            patch.object(
                disagg_utils, "_all_reduce_polls", side_effect=lambda p, g: p
            ) as consensus,
            patch.object(
                disagg_utils.envs.SGLANG_TEST_DISAGG_FAILURE_PROB, "get", return_value=0
            ),
        ):
            if staged:
                handler = Mock()
                handler.is_done.return_value = True
                handler.is_failed.return_value = False
                result = disagg_utils.poll_and_all_reduce_with_staging(
                    [decode_req], handler, group, metadata
                )
            else:
                result = disagg_utils.poll_and_all_reduce(
                    [receiver], group, [decode_req], metadata
                )
        consensus.assert_called_once_with(result, group)
        receiver.abort.assert_not_called()
        receiver.clear.assert_not_called()
        return result

    def test_mismatched_metadata_is_failed_before_consensus(self):
        for staged in (False, True):
            with (
                self.subTest(staged=staged),
                self.assertLogs(disagg_utils.logger, level="ERROR") as logs,
            ):
                self.assertEqual(self._poll(99, staged=staged), [int(KVPoll.Failed)])
            self.assertIn("request-7", logs.output[0])
            self.assertIn("expected=7, actual=99", logs.output[0])
            self.assertIn("metadata_buffer_index=1", logs.output[0])

    def test_missing_metadata_waits_and_matching_metadata_succeeds(self):
        for staged in (False, True):
            with self.subTest(staged=staged):
                self.assertEqual(
                    self._poll(0, staged=staged), [int(KVPoll.Transferring)]
                )
                self.assertEqual(self._poll(7, staged=staged), [int(KVPoll.Success)])

    def test_fake_transfer_skips_metadata_ownership(self):
        for staged in (False, True):
            with self.subTest(staged=staged):
                self.assertEqual(
                    self._poll(99, staged=staged, fake=True), [int(KVPoll.Success)]
                )

    def test_failure_is_not_overwritten_by_pending_metadata(self):
        for staged in (False, True):
            with self.subTest(staged=staged):
                self.assertEqual(
                    self._poll(0, staged=staged, poll=KVPoll.Failed),
                    [int(KVPoll.Failed)],
                )


if __name__ == "__main__":
    unittest.main()
