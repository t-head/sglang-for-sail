"""Unit tests for srt/disaggregation/common/conn -- receiver connection_pool
invalidation and the decode-side bootstrap handshake lifecycle."""

from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import zmq
from disagg_test_utils import (
    complete_receiver_setup,
    make_decode_kv_manager,
    prefill_info_stub,
)

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import (
    AbortNotification,
    CommonKVReceiver,
    KVTransferError,
)
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVReceiver
from sglang.test.test_utils import CustomTestCase

ADDR = "prefill:8998"


class _ConcreteReceiver(CommonKVReceiver):
    def poll(self) -> KVPoll:
        raise NotImplementedError

    def failure_exception(self):
        raise NotImplementedError


def _receiver(connection_pool, entries):
    receiver = object.__new__(_ConcreteReceiver)
    receiver.kv_mgr = SimpleNamespace(
        connection_pool=connection_pool,
        connection_lock=threading.Lock(),
        enable_deferred_decode_kv_release=False,
    )
    receiver._connection_pool_entries = entries
    return receiver


class _FetchingReceiver(_ConcreteReceiver):
    fetch_count = 0
    fetch_gate = None

    def _get_bootstrap_info_from_server(
        self, prefill_dp_rank, prefill_cp_rank, target_tp_rank, target_pp_rank
    ):
        if self.fetch_gate is not None:
            self.fetch_gate.wait(5)
        type(self).fetch_count += 1
        return {"rank_ip": "10.0.0.1", "rank_port": 2001, "pp_rank": target_pp_rank}

    def _register_kv_args(self, bootstrap_infos):
        return True


def _armed(mgr, receiver, room):
    """Whether ``receiver``'s generation is armed for ``room`` (#37077)."""
    return mgr.is_deferred_abort_generation_armed(room, receiver._abort_generation)


def _abort_frames(receiver, room):
    return AbortNotification(
        room=room,
        decode_ip="127.0.0.1",
        decode_port=17000,
        generation=receiver._abort_generation,
    ).to_zmq()


class TestReceiverConnectionPool(CustomTestCase):
    def setUp(self):
        _FetchingReceiver.fetch_count = 0
        _FetchingReceiver.fetch_gate = None
        # init() reads the decode allocation policy from the disagg bag.
        override = get_context().override_server_args()
        override.install()
        self.addCleanup(override.restore)

    def _decode_manager(self, **overrides):
        mgr = make_decode_kv_manager(self, **overrides)
        mgr.prefill_info_table[ADDR] = prefill_info_stub()
        return mgr

    def _abort_receiver(self, room=7):
        mgr = self._decode_manager(
            enable_deferred_decode_kv_release=True,
            requires_transfer_drain=True,
        )
        mgr.connection_pool[f"{ADDR}_0_0_0"] = [
            {"rank_ip": "127.0.0.1", "rank_port": 2001},
        ]
        receiver = _FetchingReceiver(mgr, ADDR, room)
        receiver.init(0)
        return mgr, receiver

    def _cached_abort_socket(self, sock, *, endpoint_lock=None, global_lock=None):
        endpoint = "tcp://127.0.0.1:2001"
        patches = [
            patch.object(CommonKVReceiver, "_socket_cache", {endpoint: sock}),
            patch.object(
                CommonKVReceiver,
                "_socket_locks",
                {endpoint: endpoint_lock or threading.Lock()},
            ),
            patch.object(
                CommonKVReceiver, "_global_lock", global_lock or threading.Lock()
            ),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_abort_does_not_wait_for_busy_metadata_socket_locks(self):
        """Either lock may belong to bootstrap work; the scheduler must return
        while its owner is still holding the lock, retaining the ACK state."""
        for busy_lock_name in ("endpoint_lock", "global_lock"):
            with self.subTest(lock=busy_lock_name):
                mgr, receiver = self._abort_receiver()
                sock = Mock()
                busy_lock = threading.Lock()
                self._cached_abort_socket(sock, **{busy_lock_name: busy_lock})
                returned = threading.Event()
                errors = []

                def abort():
                    try:
                        receiver.abort()
                    except Exception as error:
                        errors.append(error)
                    finally:
                        returned.set()

                busy_lock.acquire()
                scheduler = threading.Thread(target=abort, daemon=True)
                scheduler.start()
                try:
                    self.assertTrue(returned.wait(5), "abort waited for a socket lock")
                    self.assertEqual(errors, [])
                    sock.send_multipart.assert_not_called()
                    self.assertFalse(receiver.abort_notified)
                    self.assertEqual(mgr.request_status[7], KVPoll.Failed)
                    self.assertTrue(_armed(mgr, receiver, 7))
                    self.assertTrue(mgr.connection_lock.acquire(blocking=False))
                    mgr.connection_lock.release()
                finally:
                    busy_lock.release()
                    scheduler.join(5)

                self.assertTrue(receiver.retry_abort())
                self.assertTrue(receiver.abort_notified)
                self.assertEqual(
                    sock.send_multipart.call_args.kwargs["flags"], zmq.DONTWAIT
                )

    def test_abort_rebuilds_missing_metadata_socket_without_waiting(self):
        mgr, receiver = self._abort_receiver()
        sock = Mock()
        context = Mock()
        context.socket.return_value = sock
        seen_ack_state = []
        sent_under_manager_lock = []

        def send(frames, flags=0):
            sent_under_manager_lock.append(mgr.connection_lock.locked())
            seen_ack_state.append(_armed(mgr, receiver, 7))

        sock.send_multipart.side_effect = send
        sockets = {}
        locks = {}
        with (
            patch.object(CommonKVReceiver, "_ctx", context),
            patch.object(CommonKVReceiver, "_socket_cache", sockets),
            patch.object(CommonKVReceiver, "_socket_locks", locks),
            patch.object(CommonKVReceiver, "_global_lock", threading.Lock()),
            patch.object(receiver, "_connect_to_bootstrap_server") as blocking_connect,
        ):
            self.assertTrue(receiver.retry_abort())
            self.assertTrue(receiver.retry_abort())
        blocking_connect.assert_not_called()
        context.socket.assert_called_once_with(zmq.PUSH)
        sock.connect.assert_called_once_with("tcp://127.0.0.1:2001")
        self.assertIs(sockets["tcp://127.0.0.1:2001"], sock)
        self.assertIn("tcp://127.0.0.1:2001", locks)
        self.assertEqual(sock.send_multipart.call_args.kwargs["flags"], zmq.DONTWAIT)
        self.assertEqual(seen_ack_state, [True] * 2)
        self.assertEqual(sent_under_manager_lock, [True, True])
        self.assertTrue(receiver.abort_notified)

    def test_socket_failures_leave_consistent_cache_and_allow_retry(self):
        for failure_step in ("socket", "configure", "connect", "send"):
            with self.subTest(failure_step=failure_step):
                mgr, receiver = self._abort_receiver()
                context = Mock()
                sock = context.socket.return_value
                failing = {
                    "socket": context.socket,
                    "configure": sock.setsockopt,
                    "connect": sock.connect,
                    "send": sock.send_multipart,
                }[failure_step]
                failing.side_effect = zmq.ZMQError(zmq.EINVAL)
                sockets = {}
                locks = {}
                with (
                    patch.object(CommonKVReceiver, "_ctx", context),
                    patch.object(CommonKVReceiver, "_socket_cache", sockets),
                    patch.object(CommonKVReceiver, "_socket_locks", locks),
                    patch.object(CommonKVReceiver, "_global_lock", threading.Lock()),
                ):
                    self.assertFalse(receiver.retry_abort())
                    if failure_step == "send":
                        endpoint = "tcp://127.0.0.1:2001"
                        self.assertEqual(sockets, {endpoint: sock})
                        self.assertEqual(set(locks), {endpoint})
                        self.assertTrue(locks[endpoint].acquire(blocking=False))
                        locks[endpoint].release()
                    else:
                        self.assertEqual(sockets, {})
                        self.assertEqual(locks, {})
                    self.assertFalse(receiver.abort_notified)
                    self.assertTrue(_armed(mgr, receiver, 7))
                    if failure_step in ("configure", "connect"):
                        sock.close.assert_called_once()
                    failing.side_effect = None
                    self.assertTrue(receiver.retry_abort())
                    self.assertTrue(receiver.abort_notified)

    def test_retry_abort_does_not_wait_for_manager_lock(self):
        mgr, receiver = self._abort_receiver()
        sock = Mock()
        self._cached_abort_socket(sock)
        returned = threading.Event()
        results = []

        def retry():
            try:
                results.append(receiver.retry_abort())
            finally:
                returned.set()

        mgr.connection_lock.acquire()
        scheduler = threading.Thread(target=retry, daemon=True)
        scheduler.start()
        try:
            self.assertTrue(returned.wait(5), "retry waited for the manager lock")
            self.assertEqual(results, [False])
            self.assertNotIn(7, mgr._deferred_abort_ack_tracker)
            sock.send_multipart.assert_not_called()
        finally:
            mgr.connection_lock.release()
            scheduler.join(5)
        self.assertTrue(receiver.retry_abort())
        self.assertTrue(_armed(mgr, receiver, 7))

    def test_abort_again_keeps_early_and_late_ack_state(self):
        mgr, receiver = self._abort_receiver()
        sock = Mock()
        observed_ack_state = []

        def backpressure(frames, flags=0):
            observed_ack_state.append(_armed(mgr, receiver, 7))
            raise zmq.Again()

        sock.send_multipart.side_effect = backpressure
        self._cached_abort_socket(sock)
        self.assertFalse(receiver.retry_abort())
        self.assertFalse(receiver.abort_notified)
        self.assertEqual(observed_ack_state, [True])
        # An ACK from an earlier attempt can arrive while this retry is unwritable.
        mgr.note_abort_ack(7, 0, receiver._abort_generation)
        receiver.retry_abort()
        self.assertEqual(mgr._deferred_abort_ack_tracker[7].prefill_ranks, {0})
        self.assertTrue(mgr.is_abort_release_safe(7, 1))
        self.assertEqual(sock.send_multipart.call_args.kwargs["flags"], zmq.DONTWAIT)
        sock.send_multipart.side_effect = None
        receiver.retry_abort()
        self.assertTrue(receiver.abort_notified)
        self.assertEqual(mgr._deferred_abort_ack_tracker[7].prefill_ranks, {0})

    def test_abort_on_real_zmq_backpressure_uses_dontwait(self):
        """Exercise real libzmq with both an absent peer and a saturated pipe.
        SNDTIMEO only prevents a regressed test from hanging forever."""
        for connected in (False, True):
            with self.subTest(connected=connected):
                mgr, receiver = self._abort_receiver()
                context = zmq.Context()
                self.addCleanup(context.term)
                sock = context.socket(zmq.PUSH)
                self.addCleanup(sock.close, 0)
                sock.setsockopt(zmq.IMMEDIATE, 1)
                sock.setsockopt(zmq.SNDHWM, 1)
                sock.setsockopt(zmq.SNDTIMEO, 1000)
                if connected:
                    peer = context.socket(zmq.PULL)
                    self.addCleanup(peer.close, 0)
                    peer.setsockopt(zmq.RCVHWM, 1)
                    sock.bind("inproc://abort-backpressure")
                    peer.connect("inproc://abort-backpressure")
                    for _ in range(100):
                        try:
                            sock.send_multipart([b"METADATA"], flags=zmq.DONTWAIT)
                        except zmq.Again:
                            break
                    else:
                        self.fail("test did not fill the bounded ZMQ pipe")
                flags_seen = []

                class RecordingSocket:
                    def send_multipart(self, frames, flags=0):
                        flags_seen.append(flags)
                        sock.send_multipart(frames, flags=flags)

                self._cached_abort_socket(RecordingSocket())
                self.assertFalse(receiver.retry_abort())
                self.assertFalse(receiver.abort_notified)
                self.assertEqual(flags_seen, [zmq.DONTWAIT])
                self.assertTrue(_armed(mgr, receiver, 7))

    def test_abort_and_metadata_keep_same_socket_fifo(self):
        mgr, receiver = self._abort_receiver()
        context = zmq.Context()
        self.addCleanup(context.term)
        sender = context.socket(zmq.PUSH)
        peer = context.socket(zmq.PULL)
        self.addCleanup(sender.close, 0)
        self.addCleanup(peer.close, 0)
        sender.bind("inproc://abort-metadata-fifo")
        peer.connect("inproc://abort-metadata-fifo")
        self._cached_abort_socket(sender)
        metadata = [b"METADATA", b"7"]
        sender.send_multipart(metadata)

        receiver.retry_abort()

        self.assertTrue(receiver.abort_notified)
        replacement_metadata = [b"METADATA", b"reused-room"]
        sender.send_multipart(replacement_metadata)
        self.assertTrue(peer.poll(timeout=5000, flags=zmq.POLLIN))
        self.assertEqual(peer.recv_multipart(), metadata)
        self.assertTrue(peer.poll(timeout=5000, flags=zmq.POLLIN))
        self.assertEqual(peer.recv_multipart(), _abort_frames(receiver, 7))
        self.assertTrue(peer.poll(timeout=5000, flags=zmq.POLLIN))
        self.assertEqual(peer.recv_multipart(), replacement_metadata)

    def test_abort_rechecks_room_and_epoch_before_each_target(self):
        mgr, old = self._abort_receiver()
        replacement = _ConcreteReceiver(mgr, ADDR, 7)
        replacement_status = mgr.request_status[7]
        sock = Mock()
        self._cached_abort_socket(sock)

        self.assertFalse(old.retry_abort())
        self.assertFalse(old._send_abort_notification(force_arm=True))

        sock.send_multipart.assert_not_called()
        self.assertNotIn(7, mgr._deferred_abort_ack_tracker)
        self.assertEqual(mgr.request_status[7], replacement_status)
        self.assertIs(mgr.room_generations[7], replacement._room_generation)

        # A different epoch invalidates only this attempt. The next attempt must
        # snapshot the current epoch and contact both original writers again.
        for direct_send in (False, True):
            with self.subTest(direct_send=direct_send):
                mgr, receiver = self._abort_receiver(room=8)
                receiver.bootstrap_infos.append(
                    {"rank_ip": "127.0.0.1", "rank_port": 2002}
                )
                first, second = Mock(), Mock()
                self._cached_abort_socket(first)
                CommonKVReceiver._socket_cache["tcp://127.0.0.1:2002"] = second
                CommonKVReceiver._socket_locks["tcp://127.0.0.1:2002"] = (
                    threading.Lock()
                )
                sent_under_manager_lock = []
                advance_after_send = [False]
                manager_lock = mgr.connection_lock
                wrapped_lock = Mock(wraps=manager_lock)

                def release():
                    manager_lock.release()
                    if advance_after_send[0]:
                        advance_after_send[0] = False
                        # Model the heartbeat taking the same lock after the
                        # first target's send and before the next target.
                        with manager_lock:
                            mgr._parallel_info_epochs[ADDR] += 1

                wrapped_lock.release.side_effect = release
                mgr.connection_lock = wrapped_lock

                def observe_send(frames, flags=0):
                    sent_under_manager_lock.append(mgr.connection_lock.locked())

                def advance_epoch(frames, flags=0):
                    observe_send(frames, flags)
                    advance_after_send[0] = True

                first.send_multipart.side_effect = advance_epoch
                second.send_multipart.side_effect = observe_send
                if direct_send:
                    receiver._abort_generation = mgr.register_deferred_abort_room(8)
                send_attempt = (
                    (lambda: receiver._send_abort_notification(force_arm=True))
                    if direct_send
                    else receiver.retry_abort
                )
                self.assertFalse(send_attempt())
                self.assertEqual(first.send_multipart.call_count, 1)
                second.send_multipart.assert_not_called()
                first.send_multipart.side_effect = observe_send

                self.assertTrue(receiver.retry_abort())

                self.assertEqual(first.send_multipart.call_count, 2)
                self.assertEqual(second.send_multipart.call_count, 1)
                self.assertEqual(sent_under_manager_lock, [True, True, True])
                self.assertTrue(_armed(mgr, receiver, 8))
                self.assertTrue(receiver.abort_notified)

    def test_false_heartbeat_eviction_still_delivers_first_abort(self):
        """No ABORT has been sent when a false heartbeat failure evicts the
        healthy peer. Reconnect on the metadata channel and wait for its ACK."""
        mgr, receiver = self._abort_receiver()
        context = zmq.Context()
        self.addCleanup(context.term)
        peer = context.socket(zmq.PULL)
        self.addCleanup(peer.close, 0)
        port = peer.bind_to_random_port("tcp://127.0.0.1")
        endpoint = f"tcp://127.0.0.1:{port}"
        receiver.bootstrap_infos[0]["rank_port"] = port
        original = context.socket(zmq.PUSH)
        self.addCleanup(original.close, 0)
        original.connect(endpoint)
        sockets = {endpoint: original}
        locks = {endpoint: threading.Lock()}

        def close_cached_sockets():
            for sock in sockets.values():
                sock.close(0)

        self.addCleanup(close_cached_sockets)
        with (
            patch.object(CommonKVReceiver, "_ctx", context),
            patch.object(CommonKVReceiver, "_socket_cache", sockets),
            patch.object(CommonKVReceiver, "_socket_locks", locks),
            patch.object(CommonKVReceiver, "_global_lock", threading.Lock()),
        ):
            mgr._handle_node_failure(ADDR)
            self.assertEqual(sockets, {})
            self.assertTrue(original.closed)
            self.assertNotIn(7, mgr._deferred_abort_ack_tracker)
            self.assertNotEqual(mgr.parallel_info_epoch(ADDR), receiver._setup_epoch)

            self.assertTrue(receiver.retry_abort())

            self.assertTrue(receiver.abort_notified)
            self.assertIsNot(sockets[endpoint], original)
            self.assertTrue(peer.poll(timeout=5000, flags=zmq.POLLIN))
            self.assertEqual(peer.recv_multipart(), _abort_frames(receiver, 7))
            self.assertFalse(mgr.is_abort_release_safe(7, 1))
            mgr.note_abort_ack(7, 0, receiver._abort_generation + 1000)
            self.assertFalse(mgr.is_abort_release_safe(7, 1))
            mgr.note_abort_ack(7, 0, receiver._abort_generation)
            mgr.note_abort_ack(7, 0, receiver._abort_generation)
            self.assertTrue(mgr.is_abort_release_safe(7, 1))
            self.assertTrue(receiver.retry_abort())
            self.assertEqual(mgr._deferred_abort_ack_tracker[7].prefill_ranks, {0})
            self.assertTrue(mgr.is_abort_release_safe(7, 1))

    def test_legacy_abort_keeps_metadata_socket_path(self):
        mgr = self._decode_manager()
        receiver = _ConcreteReceiver(mgr, ADDR, 7)
        receiver.bootstrap_infos = [{"rank_ip": "127.0.0.1", "rank_port": 2001}]
        sock = Mock()
        with patch.object(
            receiver,
            "_connect_to_bootstrap_server",
            return_value=(sock, threading.Lock()),
        ):
            receiver.retry_abort()
        self.assertEqual(sock.send_multipart.call_args.args[0][0], b"ABORT")

    def test_invalidate_removes_matching_generation(self):
        stale = [
            {"rank_ip": "10.0.0.1", "rank_port": 1001},
            {"rank_ip": "10.0.0.1", "rank_port": 1002},
        ]
        retained = [{"rank_ip": "10.0.0.2", "rank_port": 2001}]
        receiver = _receiver(
            {"stale": stale, "retained": retained},
            {"stale": stale},
        )

        receiver.invalidate_cached_bootstrap_infos()

        self.assertEqual(receiver.kv_mgr.connection_pool, {"retained": retained})
        self.assertEqual(receiver._connection_pool_entries, {})

    def test_invalidate_preserves_concurrent_replacement_generation(self):
        stale = [{"rank_ip": "10.0.0.1", "rank_port": 1001}]
        replacement = [{"rank_ip": "10.0.0.1", "rank_port": 2001}]
        receiver = _receiver(
            {"key": replacement},
            {"key": stale},
        )

        receiver.invalidate_cached_bootstrap_infos()

        self.assertEqual(receiver.kv_mgr.connection_pool, {"key": replacement})

    def test_invalidate_removes_all_matching_cp_entries(self):
        stale_cp0 = [{"rank_ip": "10.0.0.1", "rank_port": 1001}]
        stale_cp1 = [{"rank_ip": "10.0.0.1", "rank_port": 1002}]
        receiver = _receiver(
            {"cp0": stale_cp0, "cp1": stale_cp1},
            {"cp0": stale_cp0, "cp1": stale_cp1},
        )

        receiver.invalidate_cached_bootstrap_infos()

        self.assertEqual(receiver.kv_mgr.connection_pool, {})

    def test_next_receiver_refetches_after_invalidation(self):
        stale = [{"rank_ip": "10.0.0.1", "rank_port": 1001}]
        mgr = self._decode_manager()
        mgr.connection_pool[f"{ADDR}_0_0_0"] = stale
        stale_receiver = _receiver(mgr.connection_pool, {f"{ADDR}_0_0_0": stale})
        stale_receiver.kv_mgr = mgr
        stale_receiver.invalidate_cached_bootstrap_infos()

        receiver = _FetchingReceiver(mgr, ADDR, 1)
        receiver.init(0)
        complete_receiver_setup(receiver)

        self.assertEqual(_FetchingReceiver.fetch_count, 1)
        self.assertEqual(receiver.bootstrap_infos[0]["rank_port"], 2001)
        self.assertIs(
            mgr.connection_pool[f"{ADDR}_0_0_0"],
            receiver._connection_pool_entries[f"{ADDR}_0_0_0"],
        )
        self.assertEqual(mgr.request_status[1], KVPoll.WaitingForInput)

    def test_cached_connection_completes_handshake_inside_init(self):
        """A cache hit must reach WaitingForInput in init() itself: deferring it
        to an executor costs every request an extra scheduler cycle of TTFT."""
        mgr = self._decode_manager()
        cached = [{"rank_ip": "10.0.0.1", "rank_port": 2001}]
        mgr.connection_pool[f"{ADDR}_0_0_0"] = cached

        receiver = _FetchingReceiver(mgr, ADDR, 1)
        receiver.init(0)

        self.assertEqual(mgr.request_status[1], KVPoll.WaitingForInput)
        self.assertEqual(receiver.bootstrap_infos, cached)
        self.assertEqual(_FetchingReceiver.fetch_count, 0)

    def test_concurrent_cache_misses_share_one_fetch(self):
        """Receivers that miss the same connection key while its fetch is in
        flight wait for that one fetch instead of each querying the prefill."""
        mgr = self._decode_manager()
        _FetchingReceiver.fetch_gate = threading.Event()
        self.addCleanup(_FetchingReceiver.fetch_gate.set)

        receivers = [_FetchingReceiver(mgr, ADDR, room) for room in (1, 2, 3)]
        for receiver in receivers:
            receiver.init(0)
        self.assertEqual(mgr.request_status[1], KVPoll.Bootstrapping)
        self.assertIsNone(receivers[0].bootstrap_infos)

        _FetchingReceiver.fetch_gate.set()
        for receiver in receivers:
            complete_receiver_setup(receiver)

        self.assertEqual(_FetchingReceiver.fetch_count, 1)
        for room, receiver in zip((1, 2, 3), receivers):
            self.assertEqual(mgr.request_status[room], KVPoll.WaitingForInput)
            self.assertIs(receiver.bootstrap_infos[0], receivers[0].bootstrap_infos[0])

    @patch("sglang.srt.disaggregation.common.conn.time.time", return_value=3.0)
    def test_waiting_timeout_invalidates_cached_generation(self, _mock_time):
        stale = [{"rank_ip": "10.0.0.1", "rank_port": 1001}]
        mgr = self._decode_manager(waiting_timeout=1.0)
        mgr.connection_pool["key"] = stale
        receiver = _ConcreteReceiver(mgr, ADDR, 1)
        receiver._connection_pool_entries = {"key": stale}
        receiver.bootstrap_infos = stale
        receiver.init_time = 1.0
        receiver._send_abort_notification = Mock()

        self.assertEqual(receiver._check_waiting_timeout(), KVPoll.Failed)
        self.assertEqual(mgr.connection_pool, {})
        self.assertEqual(mgr.request_status[1], KVPoll.Failed)

    def test_bootstrap_deadline_counts_from_request_arrival(self):
        """The decode-side handshake deadline runs from receiver creation, so time
        spent waiting for topology or the DP rank counts against it."""
        clock = "sglang.srt.disaggregation.common.conn.time.monotonic"
        with patch(clock, return_value=100.0):
            mgr = self._decode_manager()
            mgr.decode_bootstrap_timeout = 10
            receiver = _ConcreteReceiver(mgr, ADDR, 1)

        with patch(clock, return_value=109.9):
            self.assertEqual(receiver._poll_bootstrapping(), KVPoll.Bootstrapping)
        with patch(clock, return_value=110.0):
            self.assertEqual(receiver._poll_bootstrapping(), KVPoll.Failed)
        self.assertEqual(mgr.request_status[1], KVPoll.Failed)
        self.assertIn("Bootstrapping", mgr.failure_records[1])

    def test_decode_bootstrap_timeout_defaults_to_bootstrap_timeout(self):
        """Leaving the decode-side knob unset keeps the deadline deployments
        already tuned through SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT."""
        from sglang.srt.environ import envs

        with (
            envs.SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT.override(3600),
            envs.SGLANG_DISAGGREGATION_DECODE_BOOTSTRAP_TIMEOUT.override(None),
        ):
            self.assertEqual(self._decode_manager().decode_bootstrap_timeout, 3600)
        with (
            envs.SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT.override(3600),
            envs.SGLANG_DISAGGREGATION_DECODE_BOOTSTRAP_TIMEOUT.override(30),
        ):
            self.assertEqual(self._decode_manager().decode_bootstrap_timeout, 30)

    def test_setup_finishing_after_abort_does_not_publish(self):
        """A fetch that lands after the request was aborted must not move the
        room back to WaitingForInput or hand the receiver bootstrap infos."""
        mgr = self._decode_manager()
        _FetchingReceiver.fetch_gate = threading.Event()
        self.addCleanup(_FetchingReceiver.fetch_gate.set)
        receiver = _FetchingReceiver(mgr, ADDR, 1)
        receiver.init(0)
        pending = [part for _, part in receiver._setup_parts]

        receiver.abort()
        _FetchingReceiver.fetch_gate.set()
        for future in pending:
            future.result(timeout=5)
        receiver._advance_setup()

        self.assertEqual(mgr.request_status[1], KVPoll.Failed)
        self.assertIsNone(receiver.bootstrap_infos)

    def test_stale_receiver_clear_keeps_reused_room(self):
        """clear() of a receiver whose room was reused by a newer request must
        leave the newer request's status and room tracking alone."""
        mgr = self._decode_manager()
        old = _ConcreteReceiver(mgr, ADDR, 7)
        new = _ConcreteReceiver(mgr, ADDR, 7)

        old.clear()

        self.assertEqual(mgr.request_status[7], KVPoll.Bootstrapping)
        self.assertIn(7, mgr.addr_to_rooms_tracker[ADDR])
        new.clear()
        self.assertNotIn(7, mgr.request_status)

    def test_node_failure_does_not_match_address_prefix(self):
        """Evicting host:80 must not drop the cached connections of host:8000."""
        mgr = self._decode_manager()
        mgr.connection_pool["host:8000_0_0_0"] = [{"rank_ip": "10.0.0.1"}]
        mgr.connection_pool["host:80_0_0_0"] = [{"rank_ip": "10.0.0.2"}]

        with patch.object(CommonKVReceiver, "disconnect_endpoint"):
            mgr._handle_node_failure("host:80")

        self.assertEqual(list(mgr.connection_pool), ["host:8000_0_0_0"])

    def test_prealloc_abort_then_clear_drops_its_ack_tracker_entry(self):
        """A request aborted before KV pre-allocation never reaches the deferred
        release, so its own clear() must drop the ACK state it armed."""
        mgr, receiver = self._abort_receiver(room=1234)
        self._cached_abort_socket(Mock())
        receiver.abort()
        self.assertTrue(_armed(mgr, receiver, 1234))

        receiver.clear()

        self.assertNotIn(1234, mgr._deferred_abort_ack_tracker)

    def test_stale_receiver_clear_keeps_newer_owners_ack_state(self):
        """clear() drops only the ACK state of the generation it armed."""
        mgr, old = self._abort_receiver(room=7)
        self._cached_abort_socket(Mock())
        old.abort()
        old_generation = old._abort_generation
        # The room's tracker now belongs to a later generation (armed through
        # the manager; a second receiver could not take a held room).
        mgr.clear_deferred_abort_state(7)
        newer_generation = mgr.register_deferred_abort_room(7)

        old.clear()

        self.assertNotEqual(old_generation, newer_generation)
        self.assertTrue(mgr.is_deferred_abort_generation_armed(7, newer_generation))

    def _held_room(self, room=7):
        """An aborted receiver that still holds its room awaiting the drain ACK."""
        mgr, owner = self._abort_receiver(room=room)
        self._cached_abort_socket(Mock())
        owner.abort()
        self.assertTrue(_armed(mgr, owner, room))
        return mgr, owner

    def test_held_room_rejects_a_new_receiver_without_touching_the_owner(self):
        mgr, owner = self._held_room()
        owner_reason = mgr.failure_records[7]
        owner_generation = owner._abort_generation

        newer = _FetchingReceiver(mgr, ADDR, 7)
        newer.init(0)
        newer.abort()
        newer.clear()

        self.assertEqual(newer._poll(), KVPoll.Failed)
        self.assertIsNotNone(newer._local_failure_reason)
        self.assertIsNone(newer.bootstrap_infos)
        self.assertIsNone(newer._abort_generation)
        self.assertEqual(_FetchingReceiver.fetch_count, 0)
        # The owner keeps the room, its status, its root cause and its ACK state.
        self.assertIs(mgr.room_generations[7], owner._room_generation)
        self.assertEqual(mgr.request_status[7], KVPoll.Failed)
        self.assertEqual(mgr.failure_records[7], owner_reason)
        self.assertTrue(mgr.is_deferred_abort_generation_armed(7, owner_generation))

    def test_owner_still_resends_abort_and_releases_after_a_rejected_reuse(self):
        mgr, owner = self._held_room()
        owner_generation = owner._abort_generation
        _FetchingReceiver(mgr, ADDR, 7).init(0)

        # The owner's first ABORT may not have gone out; it can still resend.
        self.assertTrue(owner.retry_abort())
        self.assertEqual(owner._abort_generation, owner_generation)

        mgr.note_abort_ack(7, 0, owner_generation)
        self.assertTrue(mgr.is_abort_release_safe(7, 1, generation=owner_generation))
        owner.clear()
        self.assertNotIn(7, mgr._deferred_abort_ack_tracker)
        self.assertNotIn(7, mgr.room_generations)

        # Once released, the room is usable again.
        fresh = _FetchingReceiver(mgr, ADDR, 7)
        fresh.init(0)
        self.assertIsNone(fresh._local_failure_reason)
        self.assertEqual(mgr.request_status[7], KVPoll.WaitingForInput)

    def test_room_is_reusable_after_a_successful_receiver_clears(self):
        """PD retraction rebootstrap reuses the room of a concluded transfer."""
        mgr = self._decode_manager(
            enable_deferred_decode_kv_release=True, requires_transfer_drain=True
        )
        mgr.connection_pool[f"{ADDR}_0_0_0"] = [
            {"rank_ip": "127.0.0.1", "rank_port": 2001},
        ]
        first = _FetchingReceiver(mgr, ADDR, 7)
        first.init(0)
        mgr.update_status(7, KVPoll.Success)
        first.clear()

        again = _FetchingReceiver(mgr, ADDR, 7)
        again.init(0)

        self.assertIsNone(again._local_failure_reason)
        self.assertEqual(mgr.request_status[7], KVPoll.WaitingForInput)

    def test_non_drain_backend_still_lets_a_reused_room_be_taken_over(self):
        mgr = self._decode_manager(enable_deferred_decode_kv_release=True)
        old = _ConcreteReceiver(mgr, ADDR, 7)
        newer = _ConcreteReceiver(mgr, ADDR, 7)

        self.assertIsNone(newer._local_failure_reason)
        self.assertIs(mgr.room_generations[7], newer._room_generation)
        self.assertIsNot(old._room_generation, newer._room_generation)

    def test_arm_keeps_own_generation_and_refuses_another_undrained_one(self):
        mgr = self._decode_manager(requires_transfer_drain=True)
        generation = mgr.arm_deferred_abort_room(7, None)

        self.assertEqual(mgr.arm_deferred_abort_room(7, generation), generation)
        self.assertIsNone(mgr.arm_deferred_abort_room(7, None))
        self.assertIsNone(mgr.arm_deferred_abort_room(7, generation + 1))
        mgr.note_abort_ack(7, 0, generation)
        self.assertEqual(mgr._deferred_abort_ack_tracker[7].prefill_ranks, {0})

        non_drain = self._decode_manager()
        first = non_drain.arm_deferred_abort_room(7, None)
        self.assertNotEqual(non_drain.arm_deferred_abort_room(7, None), first)

    def test_release_check_counts_only_the_callers_generation(self):
        mgr = self._decode_manager(requires_transfer_drain=True)
        generation = mgr.register_deferred_abort_room(7)
        mgr.note_abort_ack(7, 0, generation)

        self.assertTrue(mgr.is_abort_release_safe(7, 1, generation=generation))
        self.assertFalse(mgr.is_abort_release_safe(7, 1, generation=generation + 1))
        self.assertFalse(mgr.is_abort_release_safe(7, 1, generation=None))
        # Without a generation the check keeps upstream's room-wide behavior.
        self.assertTrue(mgr.is_abort_release_safe(7, 1))

    def test_scoped_clear_leaves_a_newer_generation_alone(self):
        mgr, receiver = self._abort_receiver(room=7)
        self._cached_abort_socket(Mock())
        receiver.abort()
        armed = receiver._abort_generation

        mgr.clear_deferred_abort_state(7, generation=armed + 1)
        mgr.clear_deferred_abort_state(7, generation=None)
        self.assertTrue(mgr.is_deferred_abort_generation_armed(7, armed))

        mgr.clear_deferred_abort_state(7, generation=armed)
        self.assertNotIn(7, mgr._deferred_abort_ack_tracker)

    def test_mooncake_rejected_receiver_reports_its_own_reason(self):
        mgr = self._decode_manager(failure_records={7: "the owner's root cause"})
        receiver = object.__new__(MooncakeKVReceiver)
        receiver.kv_mgr = mgr
        receiver.bootstrap_room = 7
        receiver.conclude_state = None
        receiver._local_failure_reason = "held by an undrained receiver"

        with self.assertRaisesRegex(KVTransferError, "undrained receiver"):
            receiver.failure_exception()
        self.assertEqual(mgr.failure_records[7], "the owner's root cause")
        self.assertEqual(receiver.conclude_state, KVPoll.Failed)

    def test_stale_receiver_cannot_arm_or_notify_reused_room(self):
        mgr = self._decode_manager(enable_deferred_decode_kv_release=True)
        receiver = _ConcreteReceiver(mgr, ADDR, 7)
        receiver.bootstrap_infos = [{}]
        _ConcreteReceiver(mgr, ADDR, 7)
        mgr.register_deferred_abort_room = Mock()
        receiver._connect_to_bootstrap_server = Mock()

        receiver.ensure_abort_notified(force_arm=True)
        receiver.retry_abort()
        receiver._send_abort_notification(force_arm=True)

        self.assertFalse(receiver.abort_notified)
        receiver.kv_mgr.register_deferred_abort_room.assert_not_called()
        receiver._connect_to_bootstrap_server.assert_not_called()


if __name__ == "__main__":
    unittest.main()
