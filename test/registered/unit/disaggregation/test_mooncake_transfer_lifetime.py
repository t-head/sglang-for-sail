"""Mooncake cancellation keeps source and destination buffers live until drain."""

import concurrent.futures
import queue
import threading
import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
import zmq

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.common import conn as common_mod
from sglang.srt.disaggregation.common.conn import KVTransferError
from sglang.srt.disaggregation.common.staging_buffer import StagingAllocator
from sglang.srt.disaggregation.common.staging_handler import PrefillStagingStrategy
from sglang.srt.disaggregation.common.transfer_lifetime import TransferLifetimeTracker
from sglang.srt.disaggregation.mooncake.conn import (
    MooncakeKVManager,
    MooncakeKVReceiver,
    MooncakeKVSender,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _manager():
    mgr = MooncakeKVManager.__new__(MooncakeKVManager)
    mgr.disaggregation_mode = DisaggregationMode.PREFILL
    mgr._transfer_lifetime = TransferLifetimeTracker()
    mgr._abort_drain_lock = threading.RLock()
    mgr._deferred_ack_targets = {}
    mgr._deferred_ack_poisoned_rooms = set()
    mgr._deferred_abort_ack_tracker = {}
    mgr._deferred_abort_generation = 0
    mgr._staging_outstanding = defaultdict(int)
    mgr.deferred_bootstrap = None
    mgr.defer_decode_allocation = False
    mgr.connection_lock = threading.Lock()
    mgr.request_status = {}
    mgr.state_layout_rejections = {}
    mgr.bootstrap_port = 7000
    mgr.failure_lock = threading.Lock()
    mgr.failure_records = {}
    mgr.transfer_infos = {}
    mgr.req_to_decode_prefix_len = {}
    mgr.transfer_queues = [queue.Queue()]
    mgr.enable_deferred_decode_kv_release = True
    mgr.enable_trace = False
    mgr.enable_staging = False
    mgr._staging_ctx = SimpleNamespace(prefetch_requested=set(), prefetched_rooms=set())
    mgr.is_dummy_cp_rank = False
    mgr.attn_tp_rank = 0
    mgr.pp_size = 1
    mgr.pp_rank = 0
    mgr.attn_cp_size = 1
    mgr.attn_cp_rank = 0
    # Only network I/O is mocked; admission, state transitions and ACK framing
    # below are the actual production implementations.
    mgr._send_multipart_locked = Mock()
    return mgr


def _sender(mgr, room=7):
    with patch.object(
        common_mod, "get_parallel", return_value=SimpleNamespace(dp_size=1, num_dp_ranks=1)
    ):
        sender = MooncakeKVSender(mgr, "127.0.0.1:7000", room)
    mgr.transfer_infos[room] = {"127.0.0.1:7101": object()}
    mgr.update_status(room, KVPoll.WaitingForInput)
    return sender


def _enqueue(mgr, room=7):
    mgr.add_transfer_request(room, np.array([1, 2], dtype=np.int32), slice(0, 2), False)


def _abort(mgr, room=7, port=7101, generation=1):
    mgr._handle_abort_notification(
        [
            b"ABORT",
            str(room).encode(),
            b"127.0.0.1",
            str(port).encode(),
            str(generation).encode(),
        ]
    )


def _complete(mgr, chunk):
    """What transfer_worker does once every I/O of a chunk has ended."""
    mgr._release_transfer_lease(chunk)
    mgr._maybe_ack_drained_abort(chunk.room)


def _staging_worker_case(*, local_fit=True, dst_size=1024, oversized=False):
    mgr = _manager()
    sender = _sender(mgr)
    mgr.enable_staging = True
    mgr.attn_tp_size = 4
    mgr.is_mla_backend = False
    mgr.is_hybrid_mla_backend = False
    mgr.bootstrap_port = 7000
    mgr.kv_args = SimpleNamespace(
        kv_data_ptrs=[100, 200], total_kv_head_num=8, engine_rank=0, gpu_id=0
    )
    mgr.kv_buffer_tensors = {
        "k_buffers": [torch.zeros((2, 2, 4), dtype=torch.bfloat16)],
        "v_buffers": [torch.zeros((2, 2, 4), dtype=torch.bfloat16)],
        "page_size": 1,
    }
    mgr.session_lock = threading.Lock()
    mgr.session_failures = defaultdict(int)
    mgr.failed_sessions = set()
    mgr.engine = SimpleNamespace(batch_transfer_sync=Mock(return_value=0))
    mgr._get_dsa_cache_transfer_skip_flags = Mock(return_value=(False, False))
    mgr._is_watermark_ready = Mock(return_value=True)
    staging_buffer = SimpleNamespace(
        fits=Mock(return_value=local_fit), get_ptr=Mock(return_value=400)
    )
    strategy = PrefillStagingStrategy.__new__(PrefillStagingStrategy)
    strategy.kv_manager = mgr
    strategy.staging_buffer = staging_buffer
    strategy.full_chunk_pages = 2
    mgr._try_create_staging_strategy = Mock(return_value=strategy)
    session = "127.0.0.1:7101"
    mgr.transfer_infos[7] = {
        session: SimpleNamespace(
            room=7,
            is_dummy=False,
            endpoint="127.0.0.1",
            dst_port=7101,
            mooncake_session_id=session,
            dst_kv_indices=np.array([3, 4], dtype=np.int32),
            dst_device_kv_indices=None,
            required_dst_info_num=1,
            staging=SimpleNamespace(
                offsets=[StagingAllocator.ALLOC_OVERSIZED if oversized else 0],
                rounds=[0],
                ends=[dst_size],
            ),
        )
    }
    mgr.decode_kv_args_table = {
        session: SimpleNamespace(
            requires_dcp_relayout=False,
            dst_tp_rank=0,
            dst_attn_tp_size=8,
            dst_kv_ptrs=[500, 600],
            dst_kv_layer_ids=[0, 0],
            dst_kv_item_len=16,
            staging_base_ptr=800,
            staging_total_size=dst_size,
            staging=None,
        )
    }
    _enqueue(mgr)
    chunk = mgr.transfer_queues[0].get_nowait()
    work_queue = SimpleNamespace(get=Mock(side_effect=[chunk, SystemExit]), put=Mock())
    return mgr, sender, work_queue, staging_buffer


class TestMooncakeTransferLifetime(CustomTestCase):
    def test_lease_exists_before_queue_publication_and_abort_cannot_ack(self):
        mgr = _manager()
        sender = _sender(mgr)
        chunks = []

        def publish(chunk):
            self.assertFalse(sender.is_source_release_safe())
            _abort(mgr)
            mgr._send_multipart_locked.assert_not_called()
            chunks.append(chunk)

        mgr.transfer_queues = [SimpleNamespace(put=publish)]
        _enqueue(mgr)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(sender.poll(), KVPoll.Failed)
        self.assertFalse(sender.is_source_release_safe())
        _complete(mgr, chunks[0])
        self.assertEqual(sender.poll(), KVPoll.Failed)
        self.assertTrue(sender.is_source_release_safe())
        self.assertEqual(
            mgr._send_multipart_locked.call_args.args[1],
            [b"ABORT_ACK", b"7", b"0", b"1"],
        )

    def test_queued_chunk_aborted_before_worker_start_drains_without_io(self):
        mgr = _manager()
        sender = _sender(mgr)
        _enqueue(mgr)
        chunk = mgr.transfer_queues[0].get_nowait()
        _abort(mgr)
        self.assertFalse(sender.is_source_release_safe())
        mgr._send_multipart_locked.assert_not_called()
        work_queue = SimpleNamespace(get=Mock(side_effect=[chunk, SystemExit]))
        # SystemExit stops the otherwise permanent worker after one iteration.
        with self.assertRaises(SystemExit):
            mgr.transfer_worker(work_queue, executor=None)
        self.assertTrue(sender.is_source_release_safe())
        self.assertEqual(sender.poll(), KVPoll.Failed)
        self.assertEqual(mgr._send_multipart_locked.call_count, 1)

    def test_aux_failure_cannot_requeue_for_a_later_staging_target(self):
        mgr = _manager()
        sender = _sender(mgr)
        mgr.enable_staging = True
        mgr.attn_tp_size = 4
        mgr.is_mla_backend = False
        mgr.is_hybrid_mla_backend = False
        mgr.kv_args = SimpleNamespace(kv_data_ptrs=[100])
        mgr.session_lock = threading.Lock()
        mgr.session_failures = defaultdict(int)
        mgr.failed_sessions = set()
        mgr._try_create_staging_strategy = Mock(return_value=object())
        mgr._get_dsa_cache_transfer_skip_flags = Mock(return_value=(False, False))
        mgr.send_kvcache = Mock(return_value=0)
        mgr.send_aux = Mock(return_value=-1)
        mgr._do_staging_transfer = Mock(return_value=(-1, True))
        mgr.transfer_infos[7] = {}
        mgr.decode_kv_args_table = {}
        for index, tp_size in enumerate((4, 8)):
            session = f"127.0.0.1:{7101 + index}"
            mgr.transfer_infos[7][session] = SimpleNamespace(
                room=7,
                is_dummy=False,
                endpoint="127.0.0.1",
                dst_port=7101 + index,
                mooncake_session_id=session,
                dst_kv_indices=np.array([3, 4], dtype=np.int32),
                dst_device_kv_indices=None,
                required_dst_info_num=2,
            )
            mgr.decode_kv_args_table[session] = SimpleNamespace(
                requires_dcp_relayout=False,
                dst_attn_tp_size=tp_size,
                dst_kv_ptrs=[200],
                dst_kv_layer_ids=[0],
                dst_kv_item_len=16,
                dst_aux_ptrs=[300],
                staging_base_ptr=400,
                staging_total_size=4096,
            )
        mgr.add_transfer_request(
            7, np.array([1, 2], dtype=np.int32), slice(0, 2), True, aux_index=1
        )
        chunk = mgr.transfer_queues[0].get_nowait()
        work_queue = SimpleNamespace(
            get=Mock(side_effect=[chunk, SystemExit]), put=Mock()
        )
        with self.assertRaises(SystemExit):
            mgr.transfer_worker(work_queue, executor=None, staging_buffer=object())
        self.assertEqual(mgr.send_kvcache.call_count, 1)
        self.assertEqual(mgr.send_aux.call_count, 1)
        self.assertEqual(mgr.send_aux.call_args.args[0].dst_port, 7101)
        mgr._do_staging_transfer.assert_not_called()
        work_queue.put.assert_not_called()
        self.assertEqual(sender.poll(), KVPoll.Failed)
        self.assertFalse(sender.is_source_release_safe())
        with self.assertRaisesRegex(RuntimeError, "PD buffers are in use"):
            sender.clear()
        _abort(mgr)
        self.assertFalse(sender.is_source_release_safe())
        self.assertFalse(
            any(
                call.args[1][0] == b"ABORT_ACK"
                for call in mgr._send_multipart_locked.call_args_list
            )
        )

    def test_staging_capacity_rejection_releases_lease_without_native_io(self):
        cases = (
            {"oversized": True},
            {"local_fit": False},
            {"dst_size": 1},
        )
        for case in cases:
            with self.subTest(**case):
                mgr, sender, work_queue, staging_buffer = _staging_worker_case(**case)
                with patch(
                    "sglang.srt.disaggregation.common.staging_buffer.gather_all_layers_to_staging"
                ) as gather:
                    with self.assertRaises(SystemExit):
                        mgr.transfer_worker(
                            work_queue, executor=None, staging_buffer=staging_buffer
                        )
                gather.assert_not_called()
                mgr.engine.batch_transfer_sync.assert_not_called()
                work_queue.put.assert_not_called()
                self.assertEqual(sender.poll(), KVPoll.Failed)
                self.assertTrue(sender.is_source_release_safe())
                self.assertEqual(mgr.failed_sessions, set())
                _abort(mgr)
                self.assertEqual(
                    mgr._send_multipart_locked.call_args.args[1],
                    [b"ABORT_ACK", b"7", b"0", b"1"],
                )

    def test_staging_native_minus_one_keeps_source_lease_and_withholds_ack(self):
        mgr, sender, work_queue, staging_buffer = _staging_worker_case()
        mgr.engine.batch_transfer_sync.return_value = -1
        with patch(
            "sglang.srt.disaggregation.common.staging_buffer.gather_all_layers_to_staging"
        ) as gather:
            with self.assertRaisesRegex(RuntimeError, "Bulk RDMA transfer failed"):
                mgr.transfer_worker(
                    work_queue, executor=None, staging_buffer=staging_buffer
                )
        gather.assert_called_once()
        mgr.engine.batch_transfer_sync.assert_called_once()
        self.assertFalse(sender.is_source_release_safe())
        _abort(mgr)
        self.assertFalse(sender.is_source_release_safe())
        self.assertFalse(
            any(
                call.args[1][0] == b"ABORT_ACK"
                for call in mgr._send_multipart_locked.call_args_list
            )
        )

    def test_two_chunks_keep_source_pointers_until_both_drain(self):
        mgr = _manager()
        sender = _sender(mgr)
        _enqueue(mgr)
        _enqueue(mgr)
        chunks = [mgr.transfer_queues[0].get_nowait() for _ in range(2)]
        _abort(mgr)
        _complete(mgr, chunks[0])
        self.assertEqual(sender.poll(), KVPoll.Failed)
        self.assertFalse(sender.is_source_release_safe())
        with self.assertRaisesRegex(RuntimeError, "PD buffers are in use"):
            sender.clear()
        self.assertIn(7, mgr.transfer_infos)
        self.assertIn(7, mgr.request_status)
        mgr._send_multipart_locked.assert_not_called()
        _complete(mgr, chunks[1])
        self.assertEqual(sender.poll(), KVPoll.Failed)
        sender.clear()
        self.assertNotIn(7, mgr.transfer_infos)
        self.assertNotIn(7, mgr.request_status)
        self.assertEqual(mgr._send_multipart_locked.call_count, 1)

    def test_abort_before_sender_open_prevents_late_transfer_admission(self):
        mgr = _manager()
        _abort(mgr)
        self.assertEqual(mgr._send_multipart_locked.call_count, 1)
        sender = _sender(mgr)
        self.assertEqual(sender.poll(), KVPoll.Failed)
        _enqueue(mgr)
        self.assertTrue(mgr.transfer_queues[0].empty())
        self.assertFalse(mgr._transfer_lifetime.try_acquire(7))
        sender.clear()

    def test_every_abort_target_gets_its_own_generation_after_drain(self):
        mgr = _manager()
        _sender(mgr)
        _enqueue(mgr)
        chunk = mgr.transfer_queues[0].get_nowait()
        _abort(mgr, port=7101, generation=1)
        _abort(mgr, port=7102, generation=2)
        mgr._send_multipart_locked.assert_not_called()
        _complete(mgr, chunk)
        messages = {
            (call.args[0], tuple(call.args[1]))
            for call in mgr._send_multipart_locked.call_args_list
        }
        self.assertEqual(
            messages,
            {
                ("tcp://127.0.0.1:7101", (b"ABORT_ACK", b"7", b"0", b"1")),
                ("tcp://127.0.0.1:7102", (b"ABORT_ACK", b"7", b"0", b"2")),
            },
        )
        # A repeated ABORT after drain re-sends the lost ACK.
        _abort(mgr, port=7101, generation=1)
        self.assertEqual(mgr._send_multipart_locked.call_count, 3)
        self.assertTrue(mgr._transfer_lifetime.is_closed(7))

    def test_queue_publication_exception_releases_its_lease(self):
        mgr = _manager()
        sender = _sender(mgr)
        mgr.transfer_queues = [
            SimpleNamespace(put=Mock(side_effect=RuntimeError("queue")))
        ]
        with self.assertRaisesRegex(RuntimeError, "queue"):
            _enqueue(mgr)
        self.assertTrue(sender.is_source_release_safe())
        mgr._send_multipart_locked.assert_not_called()

    def test_failure_exception_preserves_sender_transport_bookkeeping(self):
        mgr = _manager()
        sender = _sender(mgr)
        _enqueue(mgr)
        mgr.record_failure(7, "remote writer failed")
        with self.assertRaisesRegex(KVTransferError, "remote writer failed"):
            sender.failure_exception()
        self.assertIn(7, mgr.transfer_infos)
        self.assertIn(7, mgr.request_status)
        self.assertFalse(sender.is_source_release_safe())
        self.assertEqual(sender.poll(), KVPoll.Failed)

    def test_success_status_does_not_release_still_running_source_io(self):
        mgr = _manager()
        sender = _sender(mgr)
        _enqueue(mgr)
        chunk = mgr.transfer_queues[0].get_nowait()
        mgr.update_status(7, KVPoll.Success)
        self.assertEqual(sender.poll(), KVPoll.Success)
        self.assertFalse(sender.is_source_release_safe())
        with self.assertRaisesRegex(RuntimeError, "PD buffers are in use"):
            sender.clear()
        _complete(mgr, chunk)
        self.assertEqual(sender.poll(), KVPoll.Success)
        sender.clear()
        self.assertNotIn(7, mgr.transfer_infos)

    def test_receiver_arms_before_send_and_duplicate_abort_preserves_ack(self):
        mgr = _manager()
        mgr.disaggregation_mode = DisaggregationMode.DECODE
        mgr.engine = SimpleNamespace(get_session_id=lambda: "decode-session")
        mgr.addr_to_rooms_tracker = defaultdict(set)
        mgr.required_prefill_response_num_table = {}
        mgr.prefill_response_tracker = {}
        mgr.local_ip = "127.0.0.1"
        mgr.rank_port = 7101
        mgr._init_decode_handshake_state()
        self.addCleanup(mgr._bootstrap_executor.shutdown, wait=True)
        self.addCleanup(mgr._parallel_info_executor.shutdown, wait=True)
        receiver = MooncakeKVReceiver(mgr, "127.0.0.1:7000", 7)
        receiver.bootstrap_infos = [
            {"rank_ip": "127.0.0.1", "rank_port": 8000 + rank} for rank in (0, 1)
        ]
        sends = []

        def connect(info):
            rank = info["rank_port"] - 8000

            def send(parts, *, flags):
                self.assertEqual(flags, zmq.DONTWAIT)
                sends.append(parts)
                state = mgr._deferred_abort_ack_tracker[7]
                self.assertEqual(state.generation, receiver._abort_generation)
                # An immediate ACK must not be lost before the caller returns.
                mgr.note_abort_ack(7, rank, int(parts[4].decode()))

            return SimpleNamespace(send_multipart=send), threading.Lock()

        receiver._connect_to_bootstrap_server = connect
        # Pre-existing metadata sockets must carry ABORT as well, preserving
        # FIFO ordering without reconnecting or waiting for writable capacity.
        socket_cache = {}
        socket_locks = {}
        for info in receiver.bootstrap_infos:
            endpoint = f"tcp://{info['rank_ip']}:{info['rank_port']}"
            socket_cache[endpoint], socket_locks[endpoint] = connect(info)
        with (
            patch.object(common_mod.CommonKVReceiver, "_socket_cache", socket_cache),
            patch.object(common_mod.CommonKVReceiver, "_socket_locks", socket_locks),
            patch.object(common_mod.CommonKVReceiver, "_global_lock", threading.Lock()),
        ):
            receiver.abort()
            self.assertTrue(mgr.is_abort_release_safe(7, 2))
            generation = receiver._abort_generation
            # A retried cancellation re-sends the armed generation and keeps ACKs.
            receiver.abort()
            self.assertTrue(mgr.is_abort_release_safe(7, 2))
            self.assertEqual(receiver._abort_generation, generation)
        self.assertEqual(len(sends), 4)
        self.assertTrue(all(parts[4] == str(generation).encode() for parts in sends))

        mgr.record_failure(7, "decode timeout")
        with self.assertRaisesRegex(KVTransferError, "decode timeout"):
            receiver.failure_exception()
        self.assertIn(7, mgr.request_status)
        self.assertTrue(mgr.is_abort_release_safe(7, 2))


class TestMooncakeTransferFutureDrain(CustomTestCase):
    def test_failed_future_still_joins_other_running_io(self):
        for raises in (False, True):
            with self.subTest(raises=raises):
                observed_failure = threading.Event()

                class FailureFuture(concurrent.futures.Future):
                    def result(self, timeout=None):
                        observed_failure.set()
                        return super().result(timeout)

                failed = FailureFuture()
                if raises:
                    failed.set_exception(RuntimeError("transport exception"))
                else:
                    failed.set_result(-1)
                running = concurrent.futures.Future()
                self.assertTrue(running.set_running_or_notify_cancel())
                mgr = _manager()
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    joined = executor.submit(
                        mgr._await_transfer_futures, [failed, running]
                    )
                    try:
                        self.assertTrue(observed_failure.wait(2))
                        with self.assertRaises(concurrent.futures.TimeoutError):
                            joined.result(timeout=0.05)
                        self.assertFalse(running.cancelled())
                    finally:
                        running.set_result(0)
                    if raises:
                        with self.assertRaisesRegex(
                            RuntimeError, "transport exception"
                        ):
                            joined.result(timeout=2)
                    else:
                        self.assertEqual(joined.result(timeout=2), -1)


if __name__ == "__main__":
    unittest.main()
