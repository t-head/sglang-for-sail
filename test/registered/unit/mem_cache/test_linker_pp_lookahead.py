"""PP Linker lookups start from the admission window, not from the first check."""

import json
import tempfile
import threading
import time
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.mem_cache.unified_cache.linker_pp import PPLinkerCoordinator
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    UnifiedCacheLinkerWrapper,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

MAX_AGE = 0.2


def _wait(predicate, what):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError(f"timed out waiting for {what}")


def _expired_lookup_across_ranks(rank, directory):
    """Rank 0 is the source; rank 1 a TP peer that cannot judge the age."""
    dist.init_process_group(
        "gloo",
        init_method=Path(directory, "store").as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=15),
    )
    group = dist.new_group(backend="gloo")
    coordinator = PPLinkerCoordinator(group, lambda rid, hashes: [1], lambda: True)
    coordinator.max_age = MAX_AGE
    wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)
    wrapper.cache = SimpleNamespace(
        pp_rank=0,
        _all_reduce=lambda tensor, op: dist.all_reduce(tensor, op=op),
    )
    wrapper.pp_coordinator = coordinator
    wrapper.pp_hits = {}

    def admit():
        # Every rank calls check_pp_ready the same number of times.
        for _ in range(500):
            wrapper.start_pp_lookups(["r"])
            if wrapper.check_pp_ready("r"):
                return wrapper.pp_hits["r"].sequence
            time.sleep(0.01)
        raise AssertionError("never admitted")

    try:
        coordinator.submit("r", ["a"])
        coordinator.start_lookup("r")
        _wait(lambda: coordinator.ready_sequence("r") == 1, "the first lookup")
        dist.barrier()
        time.sleep(MAX_AGE * 2)
        if rank == 0:
            assert coordinator.ready_sequence("r") == 0  # Expires generation 1.
        dist.barrier()
        if rank == 1:
            # Give the release ticket time to land, then stall this rank's
            # publication of generation 2 behind its lock.
            deadline = time.monotonic() + 1
            while coordinator.expected.get("r") == 1 and time.monotonic() < deadline:
                time.sleep(0.005)
            coordinator.condition.acquire()
        dist.barrier()
        if rank == 0:
            coordinator.start_lookup("r")
            _wait(lambda: coordinator.ready_sequence("r") == 2, "the source result")
        dist.barrier()
        # The source has published generation 2 and the peer has not. A peer
        # still exposing generation 1 would make MIN agree on a released one.
        ready = torch.tensor(coordinator.ready_sequence("r"), dtype=torch.int64)
        dist.all_reduce(ready, op=dist.ReduceOp.MIN)
        agreed_while_stalled = int(ready.item())
        if rank == 1:
            coordinator.condition.release()
        sequence = admit()
        Path(directory, f"rank-{rank}.json").write_text(
            json.dumps([agreed_while_stalled, sequence, coordinator.expected.get("r")])
        )
    finally:
        coordinator.close()
        dist.destroy_process_group(group)
        dist.destroy_process_group()


class TestPPLinkerLookahead(CustomTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        dist.init_process_group(
            "gloo",
            init_method=Path(self.tmp.name, "store").as_uri(),
            rank=0,
            world_size=1,
            timeout=timedelta(seconds=10),
        )
        self.addCleanup(dist.destroy_process_group)
        self.queried = []
        self.gate = threading.Event()
        self.gate.set()
        self.addCleanup(self.gate.set)

    def lookup(self, rid, hashes):
        self.queried.append(rid)
        self.gate.wait(timeout=5)
        return [len(hashes)]

    def coordinator(self):
        coordinator = PPLinkerCoordinator(dist.group.WORLD, self.lookup, lambda: True)
        self.addCleanup(coordinator.close)
        return coordinator

    def wait_ready(self, coordinator, rid):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            sequence = coordinator.ready_sequence(rid)
            if sequence:
                return sequence
            time.sleep(0.005)
        self.fail(f"lookup for {rid} never completed")

    def test_window_lookups_complete_before_admission_checks(self):
        coordinator = self.coordinator()
        for rid in "abc":
            coordinator.submit(rid, ["h1", "h2"])
        self.assertEqual(coordinator.ready_sequence("a"), 0)
        self.assertEqual(self.queried, [])  # A check alone queries nothing.
        coordinator.start_lookup("a")
        coordinator.start_lookup("b")
        self.assertEqual(self.wait_ready(coordinator, "a"), 1)
        self.assertEqual(self.wait_ready(coordinator, "b"), 2)
        self.assertEqual(coordinator.result("b", 2).pages, [2])
        self.assertEqual(self.queried, ["a", "b"])  # "c" is outside the window.
        coordinator.start_lookup("a")
        time.sleep(0.05)
        self.assertEqual(self.queried, ["a", "b"])  # Started once per generation.

    def test_check_does_not_wait_for_an_inflight_lookup(self):
        self.gate.clear()
        coordinator = self.coordinator()
        coordinator.submit("slow", ["h1"])
        coordinator.start_lookup("slow")
        started = time.monotonic()
        self.assertEqual(coordinator.ready_sequence("slow"), 0)
        self.assertLess(time.monotonic() - started, 0.5)
        self.gate.set()
        self.assertEqual(self.wait_ready(coordinator, "slow"), 1)

    def test_expired_lookup_is_queried_again_from_the_window(self):
        coordinator = self.coordinator()
        coordinator.max_age = MAX_AGE
        coordinator.submit("r", ["h1"])
        coordinator.start_lookup("r")
        first = self.wait_ready(coordinator, "r")
        self.assertEqual(coordinator.ready_sequence("r"), first)  # Still fresh.
        time.sleep(MAX_AGE * 1.5)
        with self.assertLogs(
            "sglang.srt.mem_cache.unified_cache.linker_pp", level="INFO"
        ) as logs:
            self.assertEqual(coordinator.ready_sequence("r"), 0)
        self.assertIn("expired", logs.output[0])
        self.assertEqual(self.queried, ["r"])  # Waits for its next window slot.
        coordinator.start_lookup("r")
        second = self.wait_ready(coordinator, "r")
        self.assertGreater(second, first)
        self.assertEqual(self.queried, ["r", "r"])
        # The release ticket retired the expired generation.
        _wait(lambda: first not in coordinator.results.get("r", {}), "release")

    def test_non_positive_max_age_never_expires(self):
        coordinator = self.coordinator()
        coordinator.max_age = 0
        coordinator.submit("r", ["h1"])
        coordinator.start_lookup("r")
        first = self.wait_ready(coordinator, "r")
        time.sleep(0.05)
        self.assertEqual(coordinator.ready_sequence("r"), first)
        self.assertEqual(self.queried, ["r"])

    def test_peers_agree_on_the_fresh_generation_after_expiry(self):
        mp.spawn(
            _expired_lookup_across_ranks, args=(self.tmp.name,), nprocs=2, join=True
        )
        for rank in range(2):
            observed = json.loads(Path(self.tmp.name, f"rank-{rank}.json").read_text())
            self.assertEqual(observed, [0, 2, 2])

    def test_deferred_request_is_queried_again_from_the_window(self):
        coordinator = self.coordinator()
        coordinator.submit("r", ["h1"])
        coordinator.start_lookup("r")
        first = self.wait_ready(coordinator, "r")
        coordinator.defer("r", first)
        self.assertEqual(coordinator.ready_sequence("r"), 0)
        self.assertEqual(self.queried, ["r"])  # Waits for its next window slot.
        coordinator.start_lookup("r")
        second = self.wait_ready(coordinator, "r")
        self.assertGreater(second, first)
        self.assertEqual(self.queried, ["r", "r"])


if __name__ == "__main__":
    unittest.main()
