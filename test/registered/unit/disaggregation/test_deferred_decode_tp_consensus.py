"""Real Gloo ranks must retire identical decode destinations in the same order."""

import datetime
import multiprocessing
import tempfile
import time
import traceback
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.decode import DecodeTransferQueue
from sglang.srt.disaggregation.decode_hicache_mixin import HiCacheRestoreResult
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


def _queue():
    q = DecodeTransferQueue.__new__(DecodeTransferQueue)
    q.queue = []
    q.gloo_group = dist.group.WORLD
    q._release_tp_size = dist.get_world_size()
    q._deferred_release_error = None
    q._deferred_releases = []
    q._failed_deferred_releases = []
    q.deferred_kv_release_timeout = 30.0
    q.enable_host_receive = False
    q.enable_staging = False
    q.staging_handler = None
    q.scheduler = SimpleNamespace(
        enable_hisparse=False,
        enable_decode_hicache=False,
        metrics_reporter=SimpleNamespace(enable_metrics=False),
    )
    q.tree_cache = SimpleNamespace(is_load_back_event_done=lambda index: True)
    q.released = []

    def release(req, metadata_idx):
        # Stand-in for allocator/GPU teardown only. The release decision,
        # collectives and error handling remain production DecodeTransferQueue.
        q.released.append((req.req.bootstrap_room, metadata_idx))
        req.kv_receiver = None

    q._do_release = release
    return q


def _hold(q, room, metadata_idx, *, acked=True):
    mgr = CommonKVManager.__new__(CommonKVManager)
    mgr._deferred_abort_ack_tracker = {}
    mgr._deferred_abort_generation = 0
    generation = mgr.register_deferred_abort_room(room)
    if acked:
        mgr.note_abort_ack(room, 0, generation)
    mgr.test_generation = generation
    receiver = SimpleNamespace(
        kv_mgr=mgr,
        _abort_generation=generation,
        bootstrap_infos=[{"rank": 0}],
        retry_abort=lambda: None,
    )
    req = SimpleNamespace(
        req=SimpleNamespace(
            rid=f"req-{room}",
            bootstrap_room=room,
            kv=SimpleNamespace(req_pool_idx=metadata_idx),
        ),
        kv_receiver=receiver,
        metadata_buffer_index=metadata_idx,
        hicache_restore_lock=None,
        hicache_load_consumer_index=-1,
    )
    q._defer_release(req)
    return req, mgr


def _must_raise_release_error(q):
    try:
        q.resolve_deferred_releases()
    except RuntimeError:
        return
    raise AssertionError("every TP rank must reject an inconsistent release")


def _consensus_worker(rank, rendezvous, out):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=20),
    )
    try:
        result = {}
        q = _queue()
        q.resolve_deferred_releases()
        dist.barrier()
        result["empty"] = list(q.released)

        q = _queue()
        _req, mgr = _hold(q, 101, 7, acked=rank == 0)
        q.resolve_deferred_releases()
        result["one_rank_acked"] = list(q.released)
        mgr.note_abort_ack(101, 0, mgr.test_generation)
        q.resolve_deferred_releases()
        result["all_ranks_acked"] = list(q.released)

        q = _queue()
        req, _mgr = _hold(q, 102, 8)
        req.hicache_restore_lock = object()
        req.hicache_load_consumer_index = 3
        q.tree_cache.is_load_back_event_done = lambda index: rank == 0
        q.resolve_deferred_releases()
        result["one_restore_pending"] = list(q.released)
        q.tree_cache.is_load_back_event_done = lambda index: True
        q.resolve_deferred_releases()
        result["all_restores_done"] = list(q.released)

        q = _queue()
        q.enable_staging = True
        q.staging_handler = SimpleNamespace(is_staging_room=lambda room: True)
        req, _mgr = _hold(q, 109, 14)
        event = SimpleNamespace(query=lambda: rank == 0)
        req._chunk_events = [(event, 0)]
        q.resolve_deferred_releases()
        result["one_scatter_pending"] = list(q.released)
        event.query = lambda: True
        q.resolve_deferred_releases()
        result["all_scatters_done"] = list(q.released)

        q = _queue()
        entries = [(103, 9), (104, 10)]
        for room, idx in entries if rank == 0 else reversed(entries):
            _hold(q, room, idx)
        q.resolve_deferred_releases()
        result["canonical_release_order"] = list(q.released)

        for mode in ("missing", "wrong_room", "wrong_slot"):
            q = _queue()
            if rank == 0 or mode != "missing":
                room = 106 if rank == 1 and mode == "wrong_room" else 105
                idx = 12 if rank == 1 and mode == "wrong_slot" else 11
                _hold(q, room, idx)
            _must_raise_release_error(q)
            result[mode] = list(q.released)
            # The fatal state must remain sticky even on the empty rank.
            _must_raise_release_error(q)
            dist.barrier()

        q = _queue()
        _hold(q, 110, 15)
        if rank == 1:

            def fail_drain_check(req, required_acks):
                raise RuntimeError("injected local DMA completion check failure")

            q._local_deferred_release_ready = fail_drain_check
        _must_raise_release_error(q)
        result["drain_check_failure"] = list(q.released)
        _must_raise_release_error(q)
        dist.barrier()

        q = _queue()
        _hold(q, 107, 13)
        if rank == 1:

            def fail_cleanup(req, metadata_idx):
                raise RuntimeError("injected allocator cleanup failure")

            q._do_release = fail_cleanup
        _must_raise_release_error(q)
        result["cleanup_failed_globally"] = True
        _must_raise_release_error(q)
        result["cleanup_failure_sticky"] = True
        dist.barrier()

        q = _queue()
        q.scheduler.enable_decode_hicache = True
        q.metadata_buffers = SimpleNamespace(bootstrap_room=torch.tensor([[108]]))
        q.queue = [
            SimpleNamespace(
                req=SimpleNamespace(
                    rid="req-108", bootstrap_host="127.0.0.1", bootstrap_room=108
                ),
                metadata_buffer_index=0,
                hicache_restore_status=(
                    HiCacheRestoreResult.FAILED
                    if rank == 1
                    else HiCacheRestoreResult.READY
                ),
                kv_receiver=SimpleNamespace(
                    poll=lambda: KVPoll.Success, require_staging=True
                ),
            )
        ]
        result["hicache_failure_poll"] = q._poll_with_metadata_gate()
        q.staging_handler = SimpleNamespace(
            is_done=lambda req: True, is_failed=lambda req: False
        )
        result["staging_hicache_failure_poll"] = q._poll_with_staging()
        q.queue[0].hicache_restore_status = (
            HiCacheRestoreResult.PENDING if rank == 1 else HiCacheRestoreResult.READY
        )
        result["staging_hicache_pending_poll"] = q._poll_with_staging()
        q.queue[0].hicache_restore_status = HiCacheRestoreResult.READY
        q.metadata_buffers.bootstrap_room[0] = 999 if rank == 0 else 108
        result["metadata_corruption_poll"] = q._poll_with_metadata_gate()
        result["staging_metadata_corruption_poll"] = q._poll_with_staging()
        out.put((rank, result, None))
    except BaseException:
        out.put((rank, None, traceback.format_exc()))
        raise
    finally:
        dist.destroy_process_group()


class TestDeferredDecodeTPConsensus(CustomTestCase):
    def test_real_two_rank_gloo_release_consensus(self):
        context = multiprocessing.get_context("spawn")
        out = context.Queue()
        with tempfile.TemporaryDirectory() as directory:
            rendezvous = str(Path(directory) / "gloo-rendezvous")
            processes = [
                context.Process(target=_consensus_worker, args=(rank, rendezvous, out))
                for rank in range(2)
            ]
            try:
                for process in processes:
                    process.start()
                deadline = time.monotonic() + 90
                reports = [
                    out.get(timeout=max(1, deadline - time.monotonic()))
                    for _ in processes
                ]
                for process in processes:
                    process.join(timeout=max(1, deadline - time.monotonic()))
                self.assertEqual(sorted(report[0] for report in reports), [0, 1])
                for rank, result, error in reports:
                    self.assertIsNone(error, f"rank {rank}: {error}")
                    self.assertEqual(result["empty"], [])
                    self.assertEqual(result["one_rank_acked"], [])
                    self.assertEqual(result["all_ranks_acked"], [(101, 7)])
                    self.assertEqual(result["one_restore_pending"], [])
                    self.assertEqual(result["all_restores_done"], [(102, 8)])
                    self.assertEqual(result["one_scatter_pending"], [])
                    self.assertEqual(result["all_scatters_done"], [(109, 14)])
                    self.assertEqual(
                        result["canonical_release_order"], [(103, 9), (104, 10)]
                    )
                    for mismatch in ("missing", "wrong_room", "wrong_slot"):
                        self.assertEqual(result[mismatch], [])
                    self.assertEqual(result["drain_check_failure"], [])
                    self.assertTrue(result["cleanup_failed_globally"])
                    self.assertTrue(result["cleanup_failure_sticky"])
                    self.assertEqual(result["hicache_failure_poll"], [KVPoll.Failed])
                    self.assertEqual(
                        result["staging_hicache_failure_poll"], [KVPoll.Failed]
                    )
                    self.assertEqual(
                        result["staging_hicache_pending_poll"], [KVPoll.Transferring]
                    )
                    self.assertEqual(
                        result["metadata_corruption_poll"], [KVPoll.Failed]
                    )
                    self.assertEqual(
                        result["staging_metadata_corruption_poll"], [KVPoll.Failed]
                    )
                for process in processes:
                    self.assertEqual(process.exitcode, 0)
            finally:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=5)
                out.close()
                out.join_thread()


if __name__ == "__main__":
    unittest.main()
