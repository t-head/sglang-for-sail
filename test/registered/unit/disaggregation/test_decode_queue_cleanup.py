import threading
import time
import unittest
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from disagg_test_utils import make_decode_kv_manager

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.common.conn import (
    CommonKVManager,
    KVTransferError,
    ParallelInfoState,
)
from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    DecodeTransferQueue,
    HiCacheRestoreResult,
)
from sglang.srt.disaggregation.decode_hicache_mixin import (
    HiCacheRestoreGatedKVReceiver,
)
from sglang.srt.disaggregation.fake.conn import FakeKVManager, FakeKVReceiver
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    poll_and_all_reduce_with_staging,
)
from sglang.srt.managers.schedule_batch import FINISH_ABORT, ReqKvInfo
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.runtime_context import get_context, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.separate_buffer_allocator_double import (
    bind_separate_buffer_capacity,
)
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class FakeReceiver:
    def __init__(self):
        self.clear_called = False
        self.conclude_state = None

    def clear(self):
        self.clear_called = True

    def failure_exception(self):
        return None


class TestDeferredReleaseRecovery(CustomTestCase):
    def _queue(self, *, sent=True, already_notified=False):
        manager = CommonKVManager.__new__(CommonKVManager)
        manager.requires_transfer_drain = True
        manager._deferred_abort_ack_tracker = {}
        manager._deferred_abort_generation = 0
        # The quarantined transfer's armed ABORT generation (#37077).
        self.generation = manager.register_deferred_abort_room(7)
        receiver = SimpleNamespace(
            kv_mgr=manager,
            _abort_generation=self.generation,
            retry_abort=MagicMock(return_value=sent),
            abort_notified=already_notified,
            bootstrap_infos=[{"rank": 0}, {"rank": 1}],
        )
        req = SimpleNamespace(
            req=SimpleNamespace(rid="quarantined", bootstrap_room=7),
            kv_receiver=receiver,
            metadata_buffer_index=3,
            hicache_restore_lock=None,
        )
        queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
        queue.enable_host_receive = False
        queue.scheduler = SimpleNamespace(
            metrics_reporter=SimpleNamespace(enable_metrics=False)
        )
        queue._release_tp_size = 1
        queue._deferred_releases = []
        queue._failed_deferred_releases = []
        queue._deferred_release_error = None
        queue.deferred_kv_release_timeout = 30
        queue.enable_staging = False
        queue._do_release = MagicMock()
        with patch("sglang.srt.disaggregation.decode.time.monotonic", return_value=0):
            queue._defer_release(req)
        return queue, req, manager

    def test_unsent_abort_retries_before_warning_deadline(self):
        queue, req, manager = self._queue()
        with (
            patch("sglang.srt.disaggregation.decode.time.monotonic") as clock,
            patch("sglang.srt.disaggregation.decode.logger") as logger,
        ):
            clock.return_value = 0.049
            queue.resolve_deferred_releases()
            req.kv_receiver.retry_abort.assert_not_called()
            clock.return_value = 0.05
            queue.resolve_deferred_releases()
            req.kv_receiver.retry_abort.assert_called_once_with()
            logger.warning.assert_not_called()
        self.assertAlmostEqual(req._abort_retry_deadline, 30.05)
        self.assertFalse(manager.is_abort_release_safe(7, 2))
        queue._do_release.assert_not_called()

    def test_backpressure_backs_off_without_releasing_buffers(self):
        queue, req, manager = self._queue(sent=False)
        intervals = []
        with (
            patch("sglang.srt.disaggregation.decode.time.monotonic") as clock,
            patch("sglang.srt.disaggregation.decode.logger"),
        ):
            for attempt in range(20):
                now = req._abort_retry_deadline
                clock.return_value = now
                queue.resolve_deferred_releases()
                intervals.append(req._abort_retry_deadline - now)
                self.assertEqual(req.kv_receiver.retry_abort.call_count, attempt + 1)
                # A second scheduler step at the same time cannot busy-loop.
                queue.resolve_deferred_releases()
                self.assertEqual(req.kv_receiver.retry_abort.call_count, attempt + 1)
        self.assertAlmostEqual(intervals[0], 0.1)
        for previous, current in zip(intervals, intervals[1:]):
            self.assertGreaterEqual(current + 1e-9, previous)
        self.assertAlmostEqual(max(intervals), 60.0)
        self.assertTrue(queue.has_pending_deferred_releases())
        self.assertFalse(manager.is_abort_release_safe(7, 2))
        queue._do_release.assert_not_called()

    def test_matching_late_acks_release_during_retry_backoff(self):
        queue, req, manager = self._queue(sent=False)
        receiver = req.kv_receiver
        with patch("sglang.srt.disaggregation.decode.time.monotonic") as clock:
            clock.return_value = 0.05
            queue.resolve_deferred_releases()
            manager.note_abort_ack(7, 0, self.generation + 1)
            manager.note_abort_ack(7, 1, self.generation)
            clock.return_value = 0.06
            queue.resolve_deferred_releases()
            queue._do_release.assert_not_called()
            manager.note_abort_ack(7, 0, self.generation)
            clock.return_value = 0.07
            queue.resolve_deferred_releases()
        receiver.retry_abort.assert_called_once_with()
        queue._do_release.assert_called_once_with(req, 3)
        self.assertFalse(queue.has_pending_deferred_releases())

    def test_accepted_abort_waits_for_ack_and_retries_with_backoff(self):
        queue, req, manager = self._queue(already_notified=True)
        with (
            patch("sglang.srt.disaggregation.decode.time.monotonic") as clock,
            patch("sglang.srt.disaggregation.decode.logger"),
        ):
            clock.return_value = 29.99
            queue.resolve_deferred_releases()
            req.kv_receiver.retry_abort.assert_not_called()
            clock.return_value = 30
            queue.resolve_deferred_releases()
            self.assertEqual(req._abort_retry_deadline, 90)
            clock.return_value = 89
            queue.resolve_deferred_releases()
            req.kv_receiver.retry_abort.assert_called_once_with()
            clock.return_value = 90
            queue.resolve_deferred_releases()
            self.assertEqual(req.kv_receiver.retry_abort.call_count, 2)
            self.assertEqual(req._abort_retry_deadline, 150)
        self.assertFalse(manager.is_abort_release_safe(7, 2))
        queue._do_release.assert_not_called()

    def test_warning_does_not_override_retry_backoff(self):
        queue, req, manager = self._queue(sent=False)
        req._abort_retry_deadline = 120
        with (
            patch("sglang.srt.disaggregation.decode.time.monotonic", return_value=30),
            self.assertLogs(
                "sglang.srt.disaggregation.decode", level="WARNING"
            ) as logs,
        ):
            queue.resolve_deferred_releases()
        self.assertIn("retrying ABORT with backoff", logs.output[0])
        req.kv_receiver.retry_abort.assert_not_called()
        # (start_time, deadline, metadata_idx, required_acks)
        self.assertEqual(queue._deferred_releases[0][1:], (0, 60, 3, 2))
        queue._do_release.assert_not_called()


class TestDecodeQueueCleanup(CustomTestCase):
    def setUp(self):
        # The code under test reads its config from the bags.
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")

    def test_paged_swa_retraction_resume_uses_physical_page_budget(self):
        # resume_retracted_reqs reads the retraction backend off the disagg
        # bag, so the case publishes a config instead of injecting one.
        override = get_context().override_server_args(
            disaggregation_decode_retraction_backup="cpu_tensor"
        )
        override.install()
        self.addCleanup(override.restore)

        page_size = 128
        fill_len = 574
        physical_tokens_per_req = 5 * page_size
        physical_available = 18 * page_size

        reqs = [
            SimpleNamespace(
                rid=f"req-{i}",
                origin_input_ids=[0] * fill_len,
                output_ids=[],
                is_retracted=True,
                retraction_backup=None,
                load_kv_cache=MagicMock(),
            )
            for i in range(4)
        ]

        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.retracted_queue = reqs.copy()
        queue.num_reserved_decode_tokens = 0
        queue.req_to_token_pool = SimpleNamespace(available_size=lambda: len(reqs))
        queue.token_to_kv_pool_allocator = MagicMock(page_size=page_size)
        bind_separate_buffer_capacity(queue.token_to_kv_pool_allocator)
        queue.tree_cache = MagicMock()
        queue.scheduler = SimpleNamespace(
            sliding_window_size=2047,
            server_args=SimpleNamespace(disable_radix_cache=True),
        )
        queue._uses_swa_tail_prealloc = MagicMock(return_value=True)
        queue._swa_aware_allocatable_token_budgets = MagicMock(
            return_value=(physical_available, physical_available)
        )
        queue._allocatable_token_budgets = MagicMock(
            side_effect=lambda **_: physical_available
        )
        queue._swa_tail_allocatable_token_budget = MagicMock(
            side_effect=lambda **_: physical_available
        )

        def pre_alloc(_req):
            nonlocal physical_available
            self.assertGreaterEqual(physical_available, physical_tokens_per_req)
            physical_available -= physical_tokens_per_req

        queue._pre_alloc = MagicMock(side_effect=pre_alloc)

        resumed = queue.resume_retracted_reqs()

        self.assertEqual(resumed, reqs[:3])
        self.assertEqual(queue.retracted_queue, reqs[3:])
        self.assertEqual(physical_available, 3 * page_size)
        self.assertEqual(queue._pre_alloc.call_count, 3)

    def test_prealloc_abort_clears_receiver_before_removing_request(self):
        receiver = FakeReceiver()
        req = SimpleNamespace(
            rid="abort-prealloc",
            bootstrap_room=42,
            kv=ReqKvInfo(),
            finished_reason=None,
            return_logprob=False,
        )
        decode_req = SimpleNamespace(
            req=req, kv_receiver=receiver, waiting_for_input=True
        )

        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.pp_size = 1
        queue.tp_rank = 0
        queue.gloo_group = object()
        queue.queue = [decode_req]
        queue.pending_reqs = []
        queue.retracted_queue = []
        queue._resolve_pending_reqs = MagicMock()
        queue._uses_swa_tail_prealloc = MagicMock(return_value=False)
        # `_uses_swa_reservation` consults the allocator once tail prealloc is
        # off, so this abort path needs one even though it never allocates.
        queue.token_to_kv_pool_allocator = MagicMock()
        bind_separate_buffer_capacity(queue.token_to_kv_pool_allocator)
        queue._allocatable_token_budgets = MagicMock(return_value=0)
        queue._hicache_pending_restore_tokens = MagicMock(return_value=0)

        scheduler = MagicMock()
        scheduler.running_batch.reqs = []
        scheduler.enable_priority_scheduling = False
        scheduler.enable_hisparse = False
        scheduler.enable_lora = False
        scheduler.metrics_reporter.enable_metrics = False
        scheduler.output_streamer = MagicMock()
        queue.scheduler = scheduler

        with patch(
            "sglang.srt.disaggregation.decode.poll_and_all_reduce",
            return_value=[KVPoll.Failed],
        ) as poll:
            preallocated, failed = queue.pop_preallocated()

        poll.assert_called_once_with([receiver], queue.gloo_group)
        self.assertEqual(preallocated, [])
        self.assertEqual(failed, [decode_req])
        self.assertEqual(queue.queue, [])
        self.assertTrue(receiver.clear_called)
        self.assertIsNone(decode_req.kv_receiver)
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        scheduler.output_streamer.stream_output.assert_called_once_with(
            [req], req.return_logprob
        )

    def test_prealloc_abort_also_drops_from_pending_reqs(self):
        # Same DecodeRequest lives in both queue and pending_reqs (add() slow
        # path). Aborting must drop it from both, and compare by identity since
        # DecodeRequest's dataclass __eq__ would compare the tensor receiver.
        class BadEqReceiver(FakeReceiver):
            def __eq__(self, other):
                raise TypeError("use identity comparison, not value equality")

            __hash__ = object.__hash__

        receiver = BadEqReceiver()
        req = SimpleNamespace(
            rid="abort-shared",
            finished_reason=FINISH_ABORT("aborted"),
            kv=ReqKvInfo(),
            return_logprob=False,
        )
        decode_req = SimpleNamespace(req=req, kv_receiver=receiver)

        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.pp_size = 1
        queue.queue = [decode_req]
        queue.pending_reqs = [decode_req]  # same object, dual ownership
        queue.retracted_queue = []
        queue._resolve_pending_reqs = MagicMock()
        queue._update_handshake_waiters = MagicMock()
        queue._uses_swa_tail_prealloc = MagicMock(return_value=False)
        # `_uses_swa_reservation` consults the allocator once tail prealloc is
        # off, so this abort path needs one even though it never allocates.
        queue.token_to_kv_pool_allocator = MagicMock()
        bind_separate_buffer_capacity(queue.token_to_kv_pool_allocator)
        queue._allocatable_token_budgets = MagicMock(return_value=0)
        queue._hicache_pending_restore_tokens = MagicMock(return_value=0)

        scheduler = MagicMock()
        scheduler.running_batch.reqs = []
        scheduler.enable_priority_scheduling = False
        scheduler.enable_hisparse = False
        scheduler.enable_lora = False
        scheduler.output_streamer = MagicMock()
        queue.scheduler = scheduler

        # Must not raise on the receiver __eq__ above.
        preallocated, failed = queue.pop_preallocated()

        self.assertEqual(preallocated, [])
        self.assertEqual(failed, [decode_req])
        self.assertEqual(queue.queue, [])
        self.assertTrue(all(r is not decode_req for r in queue.pending_reqs))
        self.assertIsNone(decode_req.kv_receiver)

    def test_swa_reclaim_failure_rejects_only_request(self):
        receiver = FakeReceiver()
        req = SimpleNamespace(
            rid="swa-reclaim-failed",
            origin_input_ids=[1, 2, 3],
            kv=ReqKvInfo(),
            output_ids=[],
            finished_reason=None,
            return_logprob=False,
            sampling_params=SimpleNamespace(max_new_tokens=1),
        )
        decode_req = SimpleNamespace(
            req=req,
            kv_receiver=receiver,
            waiting_for_input=True,
            is_rebootstrap=False,
        )

        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.pp_size = 1
        queue.queue = [decode_req]
        queue.pending_reqs = [decode_req]
        queue.retracted_queue = []
        queue.num_reserved_decode_tokens = 0
        queue._resolve_pending_reqs = MagicMock()
        queue._update_handshake_waiters = MagicMock()
        queue._uses_swa_tail_prealloc = MagicMock(return_value=True)
        queue._swa_aware_allocatable_token_budgets = MagicMock(
            return_value=(1024, 1024)
        )
        queue._prealloc_required_tokens = MagicMock(return_value=(3, 3))
        queue._prealloc_kv_lens = MagicMock(return_value=(3, 3))
        queue._reclaim_swa_tail_capacity = MagicMock(
            return_value=(
                "SWA eviction insufficient: needed=64, available=0, "
                "req=swa-reclaim-failed"
            )
        )
        queue._hicache_pending_restore_tokens = MagicMock(return_value=0)
        queue._pre_alloc = MagicMock()
        # SWA tail pre-allocation runs on a paged pool.
        queue.token_to_kv_pool_allocator = MagicMock(page_size=64)
        bind_separate_buffer_capacity(queue.token_to_kv_pool_allocator)
        queue.tree_cache = MagicMock()
        queue.req_to_token_pool = MagicMock()
        queue.req_to_token_pool.available_size.return_value = 1
        # Non-hybrid pools have no mamba allocator; MagicMock would otherwise
        # auto-create one and break the `available_size() <= 0` comparison in
        # pop_preallocated.
        queue.req_to_token_pool.mamba_allocator = None
        queue.req_to_metadata_buffer_idx_allocator = MagicMock()
        queue.req_to_metadata_buffer_idx_allocator.available_size.return_value = 1

        scheduler = MagicMock()
        scheduler.running_batch.reqs = []
        scheduler.enable_priority_scheduling = False
        scheduler.enable_hisparse = False
        scheduler.enable_lora = False
        scheduler.server_args.disaggregation_decode_enable_radix_cache = False
        scheduler.output_streamer = MagicMock()
        queue.scheduler = scheduler

        preallocated, failed = queue.pop_preallocated()

        self.assertEqual(preallocated, [])
        self.assertEqual(failed, [decode_req])
        self.assertEqual(queue.queue, [])
        self.assertEqual(queue.pending_reqs, [])
        self.assertTrue(receiver.clear_called)
        self.assertIsNone(decode_req.kv_receiver)
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        queue._pre_alloc.assert_not_called()
        scheduler.output_streamer.stream_output.assert_called_once_with(
            [req], req.return_logprob
        )

    def test_ensure_prefill_info_tolerates_cleared_receiver(self):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue._max_ensure_retries = 1
        queue._ensure_retry_interval = 0
        queue._ensure_retry_count = {"127.0.0.1:11500": 0}
        queue._ensure_last_attempt_time = {}
        queue.kv_manager = MagicMock()
        queue.kv_manager.try_ensure_parallel_info.return_value = (
            ParallelInfoState.FAILED
        )
        queue.kv_manager.has_parallel_info.return_value = False

        cleared_req = SimpleNamespace(
            req=SimpleNamespace(rid="cleared"), kv_receiver=None
        )
        addr_to_reqs = {"127.0.0.1:11500": [cleared_req]}

        ready, remaining = queue._ensure_prefill_info(addr_to_reqs)

        self.assertEqual(ready, {})
        self.assertEqual(remaining, [])

    @staticmethod
    def _pending_decode_req(room):
        return SimpleNamespace(
            req=SimpleNamespace(
                bootstrap_host="127.0.0.1",
                bootstrap_port=11500,
                bootstrap_room=room,
                finished_reason=None,
            ),
            kv_receiver=MagicMock(),
        )

    @staticmethod
    def _dp_rank_queue(pending_reqs, lookups):
        """Topology is ready; DP-rank lookups return ``lookups`` in order."""
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.pending_reqs = list(pending_reqs)
        queue._prefill_dp_rank_queries = {}
        queue._resolve_prefill_dp_rank = MagicMock(return_value=None)
        queue._ensure_prefill_info = lambda groups: (groups, [])
        queue.kv_manager = MagicMock()
        queue.kv_manager.has_parallel_info.return_value = True
        queue.kv_manager.parallel_info_epoch.return_value = 0
        queue.kv_manager.submit_dp_rank_query.side_effect = lookups
        return queue

    @staticmethod
    def _answered(room_to_rank):
        future = Future()
        future.set_result(room_to_rank)
        return future

    def test_prefetches_prefill_dp_rank_query(self):
        addr = "127.0.0.1:11500"
        first = self._pending_decode_req(7)
        tail = self._pending_decode_req(8)
        queue = self._dp_rank_queue(
            [first], [self._answered({"7": 1}), self._answered({"8": 2})]
        )

        queue.prefetch_prefill_dp_rank_queries()
        queue.pending_reqs.append(tail)
        queue._resolve_pending_reqs()

        submitted = [
            call.kwargs for call in queue.kv_manager.submit_dp_rank_query.call_args_list
        ]
        self.assertEqual(
            submitted,
            [
                {"bootstrap_addr": addr, "bootstrap_rooms": [7]},
                {"bootstrap_addr": addr, "bootstrap_rooms": [8]},
            ],
        )
        first.kv_receiver.init.assert_called_once_with(1)
        tail.kv_receiver.init.assert_not_called()
        self.assertEqual(queue.pending_reqs, [tail])

        queue._resolve_pending_reqs()
        tail.kv_receiver.init.assert_called_once_with(2)
        self.assertEqual(queue.pending_reqs, [])

    def test_unresolved_dp_rank_query_defers_instead_of_blocking(self):
        req = self._pending_decode_req(7)
        pending_future = Future()  # deliberately never resolved
        queue = self._dp_rank_queue([req], [pending_future])

        queue._resolve_pending_reqs()
        queue._resolve_pending_reqs()

        queue.kv_manager.submit_dp_rank_query.assert_called_once()
        req.kv_receiver.init.assert_not_called()
        self.assertEqual(queue.pending_reqs, [req])
        self.assertFalse(pending_future.cancelled())

    def test_dp_rank_answer_does_not_cover_a_later_request_reusing_the_room(self):
        """An answer is bound to the request that asked, not to its room: a new
        request that reuses the room is asked about again."""
        old = self._pending_decode_req(7)
        lookup = Future()
        queue = self._dp_rank_queue([old], [lookup, Future()])
        queue._resolve_pending_reqs()

        new = self._pending_decode_req(7)
        queue.pending_reqs = [new]
        lookup.set_result({"7": 1})
        queue._resolve_pending_reqs()

        new.kv_receiver.init.assert_not_called()
        self.assertEqual(queue.pending_reqs, [new])
        self.assertEqual(queue.kv_manager.submit_dp_rank_query.call_count, 2)

    def test_dp_rank_answer_from_before_eviction_is_discarded(self):
        req = self._pending_decode_req(7)
        queue = self._dp_rank_queue([req], [self._answered({"7": 1}), Future()])
        queue.kv_manager.parallel_info_epoch.side_effect = [0, 1, 1]

        queue._resolve_pending_reqs()
        queue._resolve_pending_reqs()

        req.kv_receiver.init.assert_not_called()
        self.assertEqual(queue.pending_reqs, [req])

    def test_steady_arrivals_are_never_aborted_by_the_dp_rank_lookup(self):
        """With requests arriving every cycle, some request is always waiting for
        its room to be registered on the prefill; that must not count as one
        lookup stuck for the whole handshake timeout and abort fresh arrivals."""
        queue = self._dp_rank_queue([], [])
        registered = set()
        in_flight = []

        def submit_dp_rank_query(*, bootstrap_addr, bootstrap_rooms):
            future = Future()
            in_flight.append((future, list(bootstrap_rooms)))
            return future

        queue.kv_manager.submit_dp_rank_query.side_effect = submit_dp_rank_query
        arrived = []
        now = 1000.0
        for room in range(1, 801):
            now += 1.0
            with patch(
                "sglang.srt.disaggregation.decode.time.monotonic", return_value=now
            ):
                registered.update(range(1, room))  # prefill lags by one cycle
                decode_req = self._pending_decode_req(room)
                arrived.append(decode_req)
                queue.pending_reqs.append(decode_req)
                queue.prefetch_prefill_dp_rank_queries()
                queue._resolve_pending_reqs()
            for future, rooms in in_flight:
                future.set_result({str(r): 0 for r in rooms if r in registered})
            in_flight.clear()

        for decode_req in arrived:
            decode_req.kv_receiver.abort.assert_not_called()
        self.assertTrue(all(r.kv_receiver.init.called for r in arrived[:-2]))

    def _ensure_info_queue(self, addr, state, cached=False):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue._max_ensure_retries = 15
        queue._ensure_retry_interval = 1.0
        queue._ensure_retry_interval_max = 4.0
        queue._ensure_retry_count = {}
        queue._ensure_last_attempt_time = {}
        queue.kv_manager = MagicMock()
        queue.kv_manager.try_ensure_parallel_info.return_value = state
        queue.kv_manager.has_parallel_info.return_value = cached
        return queue

    def test_pending_topology_fetch_does_not_consume_retry_budget(self):
        addr = "127.0.0.1:11500"
        queue = self._ensure_info_queue(addr, ParallelInfoState.PENDING)
        req = SimpleNamespace(
            req=SimpleNamespace(rid="pending"), kv_receiver=MagicMock()
        )

        with patch(
            "sglang.srt.disaggregation.decode.time.monotonic", return_value=100.0
        ):
            for _ in range(50):
                ready, remaining = queue._ensure_prefill_info({addr: [req]})

        self.assertEqual(ready, {})
        self.assertEqual(remaining, [req])
        self.assertEqual(queue._ensure_retry_count, {})
        self.assertEqual(queue._ensure_last_attempt_time, {})
        self.assertEqual(
            queue.kv_manager.try_ensure_parallel_info.call_count,
            50,
        )
        req.kv_receiver.abort.assert_not_called()

    def test_completed_pending_fetch_is_consumed_without_retry_delay(self):
        addr = "127.0.0.1:11500"
        queue = self._ensure_info_queue(addr, ParallelInfoState.PENDING)
        queue.kv_manager.try_ensure_parallel_info.side_effect = [
            ParallelInfoState.PENDING,
            ParallelInfoState.READY,
        ]
        req = SimpleNamespace(
            req=SimpleNamespace(rid="pending-ready"), kv_receiver=MagicMock()
        )

        with patch(
            "sglang.srt.disaggregation.decode.time.monotonic", return_value=100.0
        ):
            first_ready, first_remaining = queue._ensure_prefill_info({addr: [req]})
            second_ready, second_remaining = queue._ensure_prefill_info({addr: [req]})

        self.assertEqual((first_ready, first_remaining), ({}, [req]))
        self.assertEqual((second_ready, second_remaining), ({addr: [req]}, []))
        self.assertEqual(
            queue.kv_manager.try_ensure_parallel_info.call_count,
            2,
        )

    def test_failed_topology_fetch_still_consumes_retry_budget(self):
        addr = "127.0.0.1:11500"
        queue = self._ensure_info_queue(addr, ParallelInfoState.FAILED)
        queue._max_ensure_retries = 2
        req = SimpleNamespace(
            req=SimpleNamespace(rid="failed"), kv_receiver=MagicMock()
        )

        ready, remaining = queue._ensure_prefill_info({addr: [req]})
        self.assertEqual((ready, remaining), ({}, [req]))
        self.assertEqual(queue._ensure_retry_count[addr], 1)

        queue._ensure_last_attempt_time = {}
        ready, remaining = queue._ensure_prefill_info({addr: [req]})
        self.assertEqual((ready, remaining), ({}, []))
        req.kv_receiver.abort.assert_called_once()
        self.assertNotIn(addr, queue._ensure_retry_count)

    def test_topology_retry_delay_backs_off_and_caps(self):
        addr = "127.0.0.1:11500"
        queue = self._ensure_info_queue(addr, ParallelInfoState.FAILED)

        delays = []
        for count in range(7):
            queue._ensure_retry_count = {addr: count}
            delays.append(queue._retry_delay_for(addr))

        self.assertEqual(delays, [1.0, 1.0, 2.0, 4.0, 4.0, 4.0, 4.0])

    def test_cached_topology_is_consumed_before_the_pacing_gate(self):
        addr = "127.0.0.1:11500"
        queue = self._ensure_info_queue(addr, ParallelInfoState.FAILED, cached=True)
        queue._ensure_retry_count = {addr: 3}
        queue._ensure_last_attempt_time = {addr: time.monotonic()}
        req = SimpleNamespace(
            req=SimpleNamespace(rid="cached"), kv_receiver=MagicMock()
        )

        ready, remaining = queue._ensure_prefill_info({addr: [req]})

        self.assertEqual(ready, {addr: [req]})
        self.assertEqual(remaining, [])
        queue.kv_manager.try_ensure_parallel_info.assert_not_called()
        self.assertNotIn(addr, queue._ensure_retry_count)
        self.assertNotIn(addr, queue._ensure_last_attempt_time)

    def test_missing_dp_rank_future_is_submitted_async(self):
        """A newly ingested request must not use the synchronous query fallback."""
        req = self._pending_decode_req(7)
        queue = self._dp_rank_queue([req], [Future()])

        with patch(
            "sglang.srt.disaggregation.decode.CommonKVReceiver.query_prefill_dp_ranks"
        ) as query:
            queue._resolve_pending_reqs()

        query.assert_not_called()
        queue.kv_manager.submit_dp_rank_query.assert_called_once_with(
            bootstrap_addr="127.0.0.1:11500", bootstrap_rooms=[7]
        )
        req.kv_receiver.init.assert_not_called()
        self.assertEqual(queue.pending_reqs, [req])

    @patch("sglang.srt.disaggregation.decode.release_kv_cache")
    @patch("sglang.srt.disaggregation.decode.prepare_abort")
    @patch("sglang.srt.disaggregation.decode.poll_and_all_reduce")
    def test_transfer_failure_cleanup_respects_deferred_release_gates(
        self, mock_poll, mock_prepare_abort, mock_release_kv_cache
    ):
        receiver = FakeReceiver()
        req = SimpleNamespace(
            rid="failed-transfer",
            bootstrap_room=7,
            return_logprob=False,
        )
        decode_req = SimpleNamespace(
            req=req,
            kv_receiver=receiver,
            metadata_buffer_index=3,
            hicache_restore_status=HiCacheRestoreResult.READY,
        )

        queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
        queue.queue = [decode_req]
        queue.enable_staging = False
        queue.enable_deferred_kv_release = False
        queue.gloo_group = MagicMock()
        queue.req_to_metadata_buffer_idx_allocator = MagicMock()
        queue.tp_rank = 0
        queue.tree_cache = MagicMock()
        queue.metadata_buffers = SimpleNamespace(bootstrap_room=[None] * 4)
        queue.spec_algorithm = MagicMock()
        queue.spec_algorithm.is_none.return_value = True
        queue._clean_hicache_prefetch_resources = MagicMock()

        scheduler = MagicMock()
        scheduler.enable_decode_hicache = False
        scheduler.enable_hisparse = False
        scheduler.output_streamer = MagicMock()
        scheduler.metrics_reporter.enable_metrics = False
        queue.scheduler = scheduler

        mock_poll.return_value = [KVPoll.Failed]

        transferred = queue.pop_transferred()

        self.assertEqual(transferred, [])
        self.assertEqual(queue.queue, [])
        self.assertTrue(receiver.clear_called)
        self.assertIsNone(decode_req.kv_receiver)
        queue.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(3)
        scheduler.output_streamer.stream_output.assert_called_once_with(
            [req], req.return_logprob
        )
        mock_prepare_abort.assert_called_once()
        mock_release_kv_cache.assert_called_once_with(
            req, queue.tree_cache, checkpoint=False
        )

        receiver = FakeReceiver()
        receiver.kv_mgr = FakeKVManager.__new__(FakeKVManager)
        decode_req.kv_receiver = receiver
        queue.queue = [decode_req]
        queue.enable_deferred_kv_release = True
        queue.req_to_metadata_buffer_idx_allocator.reset_mock()
        mock_release_kv_cache.reset_mock()

        transferred = queue.pop_transferred()

        self.assertEqual(transferred, [])
        self.assertEqual(queue.queue, [])
        self.assertTrue(receiver.clear_called)
        self.assertIsNone(decode_req.kv_receiver)
        queue.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(3)
        mock_release_kv_cache.assert_called_once_with(
            req, queue.tree_cache, checkpoint=False
        )

        receiver = MagicMock()
        decode_req.kv_receiver = receiver
        decode_req.host_staged = True
        queue.enable_host_receive = True
        queue.enable_deferred_kv_release = False
        queue._defer_release = MagicMock()
        queue.queue = [decode_req]
        queue.req_to_metadata_buffer_idx_allocator.reset_mock()
        mock_release_kv_cache.reset_mock()
        self.assertEqual(queue.pop_transferred(), [])
        receiver.abort.assert_called_once_with()
        queue._defer_release.assert_called_once_with(decode_req)
        receiver.clear.assert_not_called()
        queue.req_to_metadata_buffer_idx_allocator.free.assert_not_called()
        mock_release_kv_cache.assert_not_called()

        queue.queue = [decode_req]
        decode_req.req.finished_reason = FINISH_ABORT("cancelled")
        with (
            patch.object(
                queue, "_poll_with_metadata_gate", return_value=[KVPoll.Success]
            ),
            patch(
                "sglang.srt.disaggregation.decode.discard_kv_cache_backup"
            ) as discard,
        ):
            self.assertEqual(queue.pop_transferred(), [])
            discard.assert_called_once_with(req, queue.tree_cache, "host_pool")
        receiver.clear.assert_called_once_with()
        queue.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(3)
        mock_release_kv_cache.assert_not_called()

    def test_fake_receiver_initializes_deferred_release_state(self):
        manager = MagicMock()
        receiver = FakeKVReceiver(manager, "")

        self.assertIs(receiver.kv_mgr, manager)
        self.assertFalse(receiver.abort_notified)

    @patch("sglang.srt.disaggregation.decode.release_kv_cache")
    @patch("sglang.srt.disaggregation.decode.prepare_abort")
    @patch("sglang.srt.disaggregation.decode.poll_and_all_reduce")
    def test_failed_fake_transfer_releases_at_once_under_deferred_release(
        self, mock_poll, mock_prepare_abort, mock_release_kv_cache
    ):
        # A health-check request gets a FakeKVReceiver over the real transfer
        # manager, so a failed one reaches the deferred-release path.
        receiver = FakeKVReceiver(MagicMock(enable_deferred_decode_kv_release=True), "")
        req = SimpleNamespace(
            rid="health-check", bootstrap_room=0, return_logprob=False
        )
        decode_req = SimpleNamespace(
            req=req,
            kv_receiver=receiver,
            metadata_buffer_index=3,
            hicache_restore_status=HiCacheRestoreResult.READY,
        )

        queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
        queue.queue = [decode_req]
        queue.enable_staging = False
        queue.enable_host_receive = False
        queue.enable_deferred_kv_release = True
        queue._defer_release = MagicMock()
        queue.gloo_group = MagicMock()
        queue.req_to_metadata_buffer_idx_allocator = MagicMock()
        queue.tp_rank = 0
        queue.tree_cache = MagicMock()
        queue.metadata_buffers = SimpleNamespace(bootstrap_room=[None] * 4)
        queue.spec_algorithm = MagicMock()
        queue.spec_algorithm.is_none.return_value = True
        queue._clean_hicache_prefetch_resources = MagicMock()
        queue.scheduler = MagicMock(enable_decode_hicache=False, enable_hisparse=False)
        queue.scheduler.metrics_reporter.enable_metrics = False
        mock_poll.return_value = [KVPoll.Failed]

        self.assertEqual(queue.pop_transferred(), [])
        self.assertFalse(receiver.abort_notified)
        queue._defer_release.assert_not_called()
        mock_release_kv_cache.assert_called_once_with(
            req, queue.tree_cache, checkpoint=False
        )

    @patch("sglang.srt.disaggregation.decode.release_kv_cache")
    @patch("sglang.srt.disaggregation.decode.prepare_abort")
    def test_mooncake_failures_hold_destinations_even_without_optional_flag(
        self, mock_prepare_abort, mock_release_kv_cache
    ):
        # A restore failure folds into the poll only once transport is terminal
        # (#43562), so Failed is the case that reaches this branch.
        for poll, restore_status in (
            (KVPoll.Failed, HiCacheRestoreResult.READY),
            (KVPoll.Failed, HiCacheRestoreResult.FAILED),
        ):
            with self.subTest(poll=poll, restore_status=restore_status):
                receiver = FakeReceiver()
                receiver.kv_mgr = SimpleNamespace(
                    requires_transfer_drain=True,
                    enable_deferred_decode_kv_release=True,
                )
                receiver.bootstrap_infos = [{"rank": 0}, {"rank": 1}]
                receiver.abort_notified = False
                receiver.abort = MagicMock()

                def ensure_abort_notified(*, force_arm=False, receiver=receiver):
                    receiver.abort()
                    receiver.abort_notified = True

                receiver.ensure_abort_notified = ensure_abort_notified
                req = SimpleNamespace(
                    rid="failed-transfer", bootstrap_room=7, return_logprob=False
                )
                dreq = SimpleNamespace(
                    req=req,
                    kv_receiver=receiver,
                    metadata_buffer_index=3,
                    hicache_restore_status=restore_status,
                )
                q = DecodeTransferQueue.__new__(DecodeTransferQueue)
                q.queue = [dreq]
                q.enable_staging = False
                q.enable_deferred_kv_release = False
                q.deferred_kv_release_timeout = 30
                q._deferred_releases = []
                q.tp_rank = 0
                q._poll_with_metadata_gate = MagicMock(return_value=[poll])
                q._clean_hicache_prefetch_resources = MagicMock()
                q.req_to_metadata_buffer_idx_allocator = MagicMock()
                q.scheduler = MagicMock()
                q.scheduler.enable_decode_hicache = False
                q.scheduler.enable_hisparse = True
                q.scheduler.metrics_reporter.enable_metrics = False

                self.assertEqual(q.pop_transferred(), [])
                self.assertEqual(q.queue, [])
                self.assertEqual(len(q._deferred_releases), 1)
                self.assertEqual(q._deferred_releases[0][3:], (3, 2))
                self.assertIs(dreq.kv_receiver, receiver)
                self.assertFalse(receiver.clear_called)
                q._clean_hicache_prefetch_resources.assert_not_called()
                q.scheduler.hisparse_coordinator.request_finished.assert_not_called()
                q.req_to_metadata_buffer_idx_allocator.free.assert_not_called()
                mock_release_kv_cache.assert_not_called()
                receiver.abort.assert_called_once()

    def test_retracted_decode_requests_keep_scheduler_non_idle(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.running_batch = MagicMock()
        scheduler.running_batch.is_empty.return_value = True
        scheduler.chunked_req = None
        scheduler._engine_paused = False
        scheduler.dllm_manager = MagicMock()
        scheduler.dllm_manager.any_staging_reqs.return_value = False
        scheduler.last_batch = None
        scheduler.cur_batch_for_debug = None
        scheduler.enable_overlap = False
        scheduler.running_mbs = []
        scheduler.waiting_queue = []
        scheduler.grammar_manager = SimpleNamespace(grammar_queue=[])
        scheduler.disaggregation_mode = DisaggregationMode.DECODE
        scheduler.disagg_decode_prealloc_queue = SimpleNamespace(
            queue=[], retracted_queue=[object()]
        )
        scheduler.disagg_decode_transfer_queue = SimpleNamespace(
            queue=[], has_pending_deferred_releases=lambda: False
        )
        scheduler.decode_offload_manager = None
        scheduler.enable_hisparse = False
        scheduler.enable_hierarchical_cache = False
        scheduler.enable_lmcache = False

        self.assertFalse(scheduler.is_fully_idle())

        scheduler.disagg_decode_prealloc_queue.retracted_queue.clear()
        self.assertTrue(scheduler.is_fully_idle())
        scheduler.disagg_decode_transfer_queue.has_pending_deferred_releases = lambda: (
            True
        )
        self.assertFalse(scheduler.is_fully_idle())
        # Quarantined buffers have no GPU forward to carry a health response;
        # permit a fresh health request while still blocking cache flush/offload.
        self.assertTrue(scheduler.is_fully_idle(for_health_check=True))


class _GateReceiver:
    def __init__(self, poll):
        self._poll = poll
        self.require_staging = False

    def poll(self):
        return self._poll

    def clear(self):
        pass

    def failure_exception(self):
        raise KVTransferError(
            7, "Failed due to an unknown reason from another rank", True
        )


def _gated_decode_req(poll, restore_status):
    return SimpleNamespace(
        req=SimpleNamespace(rid="r", bootstrap_room=7, return_logprob=False),
        kv_receiver=_GateReceiver(poll),
        metadata_buffer_index=3,
        hicache_restore_status=restore_status,
    )


class TestDecodeHiCacheRestoreGate(CustomTestCase):
    def test_gate_folds_restore_outcome_into_terminal_poll_only(self):
        cases = [
            (KVPoll.Success, HiCacheRestoreResult.READY, KVPoll.Success),
            (KVPoll.Success, HiCacheRestoreResult.PENDING, KVPoll.Transferring),
            (KVPoll.Success, HiCacheRestoreResult.FAILED, KVPoll.Failed),
            (KVPoll.Transferring, HiCacheRestoreResult.FAILED, KVPoll.Transferring),
            (
                KVPoll.WaitingForInput,
                HiCacheRestoreResult.FAILED,
                KVPoll.WaitingForInput,
            ),
            (KVPoll.Failed, HiCacheRestoreResult.PENDING, KVPoll.Failed),
        ]
        for poll, status, expected in cases:
            with self.subTest(poll=poll, status=status):
                gated = HiCacheRestoreGatedKVReceiver(_gated_decode_req(poll, status))
                self.assertEqual(gated.poll(), expected)

    def test_failed_restore_keeps_reserving_admission_budget(self):
        def dr(status, lock):
            return SimpleNamespace(
                prefix_match=SimpleNamespace(restore_token_count=3000),
                hicache_restore_status=status,
                hicache_restore_lock=lock,
            )

        prealloc = DecodePreallocQueue.__new__(DecodePreallocQueue)
        prealloc.scheduler = SimpleNamespace(enable_decode_hicache=True)
        prealloc.transfer_queue = SimpleNamespace(
            queue=[
                dr(HiCacheRestoreResult.PENDING, None),
                dr(HiCacheRestoreResult.FAILED, None),
                dr(HiCacheRestoreResult.PENDING, object()),
                dr(HiCacheRestoreResult.READY, object()),
            ]
        )
        self.assertEqual(prealloc._hicache_pending_restore_tokens(), 6000)

    def test_staging_poll_uses_gated_pollers(self):
        decode_req = _gated_decode_req(KVPoll.Success, HiCacheRestoreResult.PENDING)
        with patch(
            "sglang.srt.disaggregation.utils._all_reduce_polls",
            side_effect=lambda polls, group: polls,
        ):
            polls = poll_and_all_reduce_with_staging(
                [decode_req],
                staging_handler=MagicMock(),
                gloo_group=MagicMock(),
                pollers=[HiCacheRestoreGatedKVReceiver(decode_req)],
            )
        self.assertEqual(polls, [KVPoll.Transferring])

    @patch("sglang.srt.disaggregation.decode.release_kv_cache")
    @patch("sglang.srt.disaggregation.decode.prepare_abort")
    def test_pop_transferred_branches_on_reduced_poll_only(
        self, mock_prepare_abort, mock_release_kv_cache
    ):
        def make_queue(decode_req, peer_poll):
            queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
            queue.queue = [decode_req]
            queue.enable_staging = False
            queue.enable_host_receive = False
            queue.enable_deferred_kv_release = False
            queue.gloo_group = MagicMock()
            queue.req_to_metadata_buffer_idx_allocator = MagicMock()
            queue.tp_rank = 0
            queue.tree_cache = MagicMock()
            queue.metadata_buffers = SimpleNamespace(bootstrap_room=[None] * 4)
            queue._clean_hicache_prefetch_resources = MagicMock()
            queue._commit_transfer_to_req = MagicMock()
            queue._record_transfer_failure = MagicMock()
            scheduler = MagicMock()
            scheduler.enable_decode_hicache = True
            scheduler.enable_hisparse = False
            scheduler.metrics_reporter.enable_metrics = False
            queue.scheduler = scheduler

            def reduce_with_peer(
                pollers, group, decode_reqs=None, metadata_buffers=None
            ):
                return [min(int(p.poll()), int(peer_poll)) for p in pollers]

            patcher = patch(
                "sglang.srt.disaggregation.decode.poll_and_all_reduce",
                side_effect=reduce_with_peer,
            )
            patcher.start()
            self.addCleanup(patcher.stop)
            return queue

        decode_req = _gated_decode_req(KVPoll.Success, HiCacheRestoreResult.FAILED)
        queue = make_queue(decode_req, peer_poll=KVPoll.Failed)
        self.assertEqual(queue.pop_transferred(), [])
        self.assertEqual(queue.queue, [])
        queue._commit_transfer_to_req.assert_not_called()
        queue._record_transfer_failure.assert_called_once_with("HiCache restore failed")
        mock_prepare_abort.assert_called_once()

        mock_prepare_abort.reset_mock()
        decode_req = _gated_decode_req(KVPoll.Success, HiCacheRestoreResult.READY)
        queue = make_queue(decode_req, peer_poll=KVPoll.Failed)
        self.assertEqual(queue.pop_transferred(), [])
        queue._commit_transfer_to_req.assert_not_called()
        queue._record_transfer_failure.assert_called_once_with(
            "Failed due to an unknown reason from another rank"
        )
        mock_prepare_abort.assert_called_once()

        mock_prepare_abort.reset_mock()
        mock_release_kv_cache.reset_mock()
        decode_req = _gated_decode_req(KVPoll.Transferring, HiCacheRestoreResult.FAILED)
        queue = make_queue(decode_req, peer_poll=KVPoll.Success)
        self.assertEqual(queue.pop_transferred(), [])
        self.assertEqual(queue.queue, [decode_req])
        mock_prepare_abort.assert_not_called()
        mock_release_kv_cache.assert_not_called()
class TestParallelInfoFetchEpoch(CustomTestCase):
    """The topology fetch runs on an executor, so it can outlive the heartbeat's
    decision that the prefill it was querying is dead."""

    ADDR = "prefill:8998"

    def _manager(self, payload=None, fetch=None):
        mgr = make_decode_kv_manager(self)
        mgr._resolve_rank_mapping = MagicMock()
        if fetch is None:
            row = payload if payload is not None else self._payload()
            fetch = MagicMock(return_value=row)
        mgr._fetch_parallel_info_payload = fetch
        return mgr, fetch

    @staticmethod
    def _payload(**overrides):
        row = {
            "attn_tp_size": 8,
            "attn_cp_size": 1,
            "dp_size": 1,
            "pp_size": 1,
            "page_size": 64,
            "kv_cache_dtype": "bfloat16",
            "follow_bootstrap_room": True,
        }
        row.update(overrides)
        return row

    def _drain(self, mgr):
        """Advance the state machine until it leaves PENDING."""
        for _ in range(200):
            state = mgr.try_ensure_parallel_info(self.ADDR)
            if state != ParallelInfoState.PENDING:
                return state
            time.sleep(0.005)
        self.fail("fetch never left PENDING")

    def test_fetch_is_not_awaited_on_the_calling_thread(self):
        entered = threading.Event()
        release = threading.Event()
        self.addCleanup(release.set)

        def blocking_fetch(addr):
            entered.set()
            release.wait(5)
            return self._payload()

        mgr, _ = self._manager(fetch=blocking_fetch)

        state = mgr.try_ensure_parallel_info(self.ADDR)
        self.assertEqual(state, ParallelInfoState.PENDING)
        self.assertTrue(entered.wait(5), "fetch never started")
        self.assertEqual(
            mgr.try_ensure_parallel_info(self.ADDR), ParallelInfoState.PENDING
        )
        self.assertNotIn(self.ADDR, mgr.prefill_info_table)

        release.set()
        self.assertEqual(self._drain(mgr), ParallelInfoState.READY)
        self.assertIn(self.ADDR, mgr.prefill_info_table)

    def test_in_flight_fetch_is_not_duplicated(self):
        entered = threading.Event()
        release = threading.Event()
        self.addCleanup(release.set)
        calls = []

        def blocking_fetch(addr):
            calls.append(addr)
            entered.set()
            release.wait(5)
            return self._payload()

        mgr, _ = self._manager(fetch=blocking_fetch)

        mgr.try_ensure_parallel_info(self.ADDR)
        self.assertTrue(entered.wait(5))
        for _ in range(20):
            mgr.try_ensure_parallel_info(self.ADDR)

        release.set()
        self._drain(mgr)
        self.assertEqual(calls, [self.ADDR], "one GET per addr while in flight")

    def test_eviction_pops_the_in_flight_fetch(self):
        entered = threading.Event()
        release = threading.Event()
        self.addCleanup(release.set)

        def blocking_fetch(addr):
            entered.set()
            release.wait(5)
            return self._payload()

        mgr, _ = self._manager(fetch=blocking_fetch)

        self.assertEqual(
            mgr.try_ensure_parallel_info(self.ADDR), ParallelInfoState.PENDING
        )
        self.assertTrue(entered.wait(5), "fetch never started")

        with patch(
            "sglang.srt.disaggregation.common.conn.CommonKVReceiver.disconnect_endpoint"
        ):
            mgr._handle_node_failure(self.ADDR)

        self.assertNotIn(self.ADDR, mgr._parallel_info_futures)
        self.assertEqual(mgr._parallel_info_epochs[self.ADDR], 1)
        release.set()

    def test_eviction_between_fetch_and_publish_does_not_resurrect_address(self):
        mgr, _ = self._manager()
        evicted = []

        def evict_mid_publish(info):
            if evicted:
                return
            evicted.append(True)
            with patch(
                "sglang.srt.disaggregation.common.conn.CommonKVReceiver.disconnect_endpoint"
            ):
                mgr._handle_node_failure(self.ADDR)

        mgr._resolve_rank_mapping = evict_mid_publish

        state = self._drain(mgr)

        self.assertTrue(evicted, "injection point never ran")
        self.assertEqual(state, ParallelInfoState.FAILED)
        self.assertNotIn(
            self.ADDR,
            mgr.prefill_info_table,
            "a fetch that resolved before eviction republished a dead prefill",
        )

    def test_fetch_after_eviction_publishes_again(self):
        mgr, _ = self._manager()

        with patch(
            "sglang.srt.disaggregation.common.conn.CommonKVReceiver.disconnect_endpoint"
        ):
            mgr._handle_node_failure(self.ADDR)
        self.assertEqual(mgr._parallel_info_epochs[self.ADDR], 1)

        self.assertEqual(self._drain(mgr), ParallelInfoState.READY)
        self.assertIn(self.ADDR, mgr.prefill_info_table)

    def test_cancelled_fetch_is_reported_as_a_failed_attempt(self):
        mgr, _ = self._manager()
        cancelled = Future()
        self.assertTrue(cancelled.cancel())
        mgr._parallel_info_futures[self.ADDR] = (0, cancelled)

        self.assertEqual(
            mgr.try_ensure_parallel_info(self.ADDR), ParallelInfoState.FAILED
        )
        self.assertNotIn(self.ADDR, mgr._parallel_info_futures)

    def test_cache_hit_needs_no_executor(self):
        mgr, fetch = self._manager()
        mgr.prefill_info_table[self.ADDR] = object()

        self.assertTrue(mgr.has_parallel_info(self.ADDR))
        self.assertEqual(
            mgr.try_ensure_parallel_info(self.ADDR), ParallelInfoState.READY
        )
        fetch.assert_not_called()

    def test_failed_fetch_is_retried_by_the_next_call(self):
        mgr, fetch = self._manager(fetch=MagicMock(return_value=None))

        self.assertEqual(self._drain(mgr), ParallelInfoState.FAILED)
        self.assertNotIn(self.ADDR, mgr._parallel_info_futures)

        fetch.return_value = self._payload()
        self.assertEqual(self._drain(mgr), ParallelInfoState.READY)
        self.assertIn(self.ADDR, mgr.prefill_info_table)

    def test_page_size_mismatch_raises_on_the_calling_thread(self):
        mgr, _ = self._manager(payload=self._payload(page_size=32))

        with self.assertRaises(RuntimeError) as ctx:
            self._drain(mgr)
        self.assertIn("Page size mismatch", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
