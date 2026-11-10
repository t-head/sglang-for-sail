"""PP release consensus must include source drain, not only terminal polls."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.disaggregation import prefill as prefill_mod
from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.transfer_lifetime import TransferLifetimeTracker
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Sender:
    def __init__(self, poll, *, pending=False):
        self.status = poll
        self.lifetime = TransferLifetimeTracker()
        self.lifetime.open(1)
        if pending:
            self.lifetime.try_acquire(1)
        self.abort_calls = 0
        self.abort_error = None

    def poll(self):
        return self.status

    def abort(self):
        self.abort_calls += 1
        if self.abort_error:
            raise self.abort_error
        self.lifetime.close(1)
        self.status = KVPoll.Failed

    def is_source_release_safe(self):
        return self.lifetime.is_drained(1)


def _req(rid, sender, *, gpu_pending=False):
    return SimpleNamespace(
        rid=rid,
        disagg_kv_sender=sender,
        finished_reason=None,
        to_finish=None,
        gpu_pending=gpu_pending,
    )


class _Scheduler(SchedulerPPMixin, SchedulerDisaggregationPrefillMixin):
    def __init__(self, reqs, *, first=True):
        self.disagg_prefill_inflight_queue = reqs
        self.pp_group = SimpleNamespace(is_first_rank=first)
        self.attn_tp_cpu_group = object()
        self.attn_cp_cpu_group = object()
        self._pp_recv_pyobj_from_prev_stage = Mock(return_value=[])

    def has_pending_prefill_result(self, req):
        return req.gpu_pending


class TestPPPrefillReleaseConsensus(CustomTestCase):
    def setUp(self):
        super().setUp()
        # The production status/source gating runs unchanged; only distributed
        # communication is local so this regression test needs no processes.
        self.poll_patch = patch.object(
            prefill_mod,
            "poll_and_all_reduce_attn_cp_tp_group",
            side_effect=lambda senders, *_: [sender.poll() for sender in senders],
        )
        self.poll_mock = self.poll_patch.start()
        self.addCleanup(self.poll_patch.stop)
        self.reduce_patch = patch.object(prefill_mod.dist, "all_reduce")
        self.reduce_mock = self.reduce_patch.start()
        self.addCleanup(self.reduce_patch.stop)

    def test_no_stage_publishes_release_until_blocked_source_drains(self):
        first_sender = _Sender(KVPoll.Failed, pending=True)
        last_sender = _Sender(KVPoll.Failed)
        first = _Scheduler([_req("request-a", first_sender)])
        last = _Scheduler([_req("request-a", last_sender)], first=False)

        initial_ids = first._pp_pd_get_prefill_transferred_ids()
        last._pp_recv_pyobj_from_prev_stage.return_value = initial_ids
        self.assertEqual(initial_ids, [])
        self.assertEqual(last._pp_pd_get_prefill_transferred_ids(), [])
        self.assertTrue(first_sender.lifetime.is_closed(1))
        self.assertEqual(len(first.disagg_prefill_inflight_queue), 1)
        self.assertEqual(len(last.disagg_prefill_inflight_queue), 1)

        first_sender.lifetime.release(1)
        drained_ids = first._pp_pd_get_prefill_transferred_ids()
        last._pp_recv_pyobj_from_prev_stage.return_value = drained_ids
        self.assertEqual(drained_ids, ["request-a"])
        self.assertEqual(last._pp_pd_get_prefill_transferred_ids(), ["request-a"])

    def test_later_stage_with_pending_write_rejects_upstream_release_id(self):
        sender = _Sender(KVPoll.Failed, pending=True)
        stage = _Scheduler([_req("request-a", sender)], first=False)
        stage._pp_recv_pyobj_from_prev_stage.return_value = ["request-a"]
        self.assertEqual(stage._pp_pd_get_prefill_transferred_ids(), [])
        sender.lifetime.release(1)
        self.assertEqual(stage._pp_pd_get_prefill_transferred_ids(), ["request-a"])

    def test_tp_peer_failure_closes_local_producer_before_drain_vote(self):
        sender = _Sender(KVPoll.Transferring, pending=True)
        stage = _Scheduler([_req("request-a", sender)])
        self.poll_mock.side_effect = lambda *_: [KVPoll.Failed]
        self.assertEqual(stage._pp_pd_get_prefill_transferred_ids(), [])
        self.assertEqual(sender.abort_calls, 1)
        self.assertTrue(sender.lifetime.is_closed(1))
        self.assertFalse(sender.lifetime.try_acquire(1))
        sender.lifetime.release(1)
        self.assertEqual(stage._pp_pd_get_prefill_transferred_ids(), ["request-a"])

    def test_pending_gpu_result_is_not_published_even_after_transport_drains(self):
        req = _req("request-a", _Sender(KVPoll.Success), gpu_pending=True)
        stage = _Scheduler([req])
        self.assertEqual(stage._pp_pd_get_prefill_transferred_ids(), [])
        req.gpu_pending = False
        self.assertEqual(stage._pp_pd_get_prefill_transferred_ids(), ["request-a"])

    def test_failed_local_cancel_cannot_publish_optimistic_drained_state(self):
        sender = _Sender(KVPoll.Failed)
        sender.abort_error = RuntimeError("cancel failed")
        stage = _Scheduler([_req("request-a", sender)])
        with self.assertLogs(prefill_mod.logger, level="ERROR"):
            self.assertEqual(stage._pp_pd_get_prefill_transferred_ids(), [])
        sender.abort_error = None
        self.assertEqual(stage._pp_pd_get_prefill_transferred_ids(), ["request-a"])

    def test_empty_stage_receives_pp_message_without_tp_cp_collectives(self):
        stage = _Scheduler([], first=False)
        stage._pp_recv_pyobj_from_prev_stage.return_value = ["request-a"]
        self.assertEqual(stage._pp_pd_get_prefill_transferred_ids(), [])
        stage._pp_recv_pyobj_from_prev_stage.assert_called_once_with()
        self.poll_mock.assert_not_called()
        self.reduce_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
