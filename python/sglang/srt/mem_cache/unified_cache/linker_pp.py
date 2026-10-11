"""Background PP agreement for direct L3 cache requests.

The scheduler cannot all-reduce across PP: later stages receive a request only
when earlier stages yield the pipeline. This independent group sees ordered
lookup/offload tickets before request admission and never touches the tree.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from queue import Empty, Queue

import torch
import torch.distributed as dist

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PPLookupResult:
    sequence: int
    hashes: list[str]
    pages: list[int]
    # When this rank published the result; the store leased the hit pages
    # during the lookup, shortly before.
    completed_at: float = 0.0


class PPLinkerCoordinator:
    def __init__(
        self,
        group,
        lookup: Callable[[str, list[str]], list[int]],
        wait_offload: Callable[[], bool],
    ):
        self.group = group
        self.source = dist.get_process_group_ranks(group)[0]
        self.is_source = dist.get_rank() == self.source
        self.lookup = lookup
        self.wait_offload = wait_offload
        self.max_age = envs.SGLANG_LINKER_PP_LOOKUP_MAX_AGE_S.get()
        self.commands = Queue()
        self.condition = threading.Condition()
        self.expected: dict[str, int] = {}
        self.pending_queries: dict[str, list[str]] = {}
        self.results: dict[str, dict[int, PPLookupResult]] = {}
        self.offloads: deque[bool] = deque()
        self.sequence = 0
        self.requested_resets = 0
        self.completed_resets = 0
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, name="linker-pp", daemon=True)
        self.thread.start()

    def _check(self) -> None:
        if self.error is not None:
            raise RuntimeError("PP linker coordinator failed") from self.error

    def submit(self, rid: str, hashes: list[str]) -> None:
        if not self.is_source:
            return
        with self.condition:
            self._check()
            if rid in self.expected:
                return
            self.sequence += 1
            self.expected[rid] = self.sequence
            self.results.pop(rid, None)
            # A queued request may wait minutes while L3 evicts its prefix.
            # Query only once the scheduler schedules it for admission.
            self.pending_queries[rid] = list(hashes)

    def start_lookup(self, rid: str) -> None:
        """Query ahead of admission so the result is known when it is checked."""
        with self.condition:
            self._check()
            hashes = self.pending_queries.pop(rid, None)
            if hashes is not None:
                self.commands.put(
                    ("lookup", rid, self.expected[rid], hashes, time.monotonic())
                )

    def ready_sequence(self, rid: str) -> int:
        with self.condition:
            self._check()
            result = self.results.get(rid, {}).get(self.expected.get(rid))
            if result is None:
                return 0
            age = time.monotonic() - result.completed_at
            if self.is_source and 0 < self.max_age < age:
                # The store may have evicted the hit pages once the lookup's
                # lease ran out. Only the source decides; the scheduler's
                # MIN reduction makes every rank see this request not ready.
                logger.info(
                    "PP linker lookup rid=%s expired after %.1fs; querying again",
                    rid,
                    age,
                )
                self._defer(rid, result.sequence)
                # Peers still hold this generation; retire it in ticket order.
                self.commands.put(("release", rid, result.sequence))
                return 0
            return result.sequence

    def defer(self, rid: str, sequence: int) -> None:
        """A rejected admission must query again on its next attempt."""
        with self.condition:
            self._check()
            self._defer(rid, sequence)

    def _defer(self, rid: str, sequence: int) -> None:
        result = self.results.get(rid, {}).pop(sequence, None)
        if self.expected.get(rid) == sequence:
            self.expected.pop(rid)
            if self.is_source and result is not None:
                self.sequence += 1
                self.expected[rid] = self.sequence
                self.pending_queries[rid] = result.hashes

    def result(self, rid: str, sequence: int) -> PPLookupResult:
        with self.condition:
            while True:
                self._check()
                result = self.results.get(rid, {}).get(sequence)
                if result is not None:
                    return result
                self.condition.wait(timeout=1)

    def release(self, rid: str) -> None:
        if self.is_source:
            with self.condition:
                sequence = self.expected.pop(rid, None)
                self.pending_queries.pop(rid, None)
                self.results.pop(rid, None)
                if sequence is not None:
                    self.commands.put(("release", rid, sequence))

    def offload_submitted(self) -> None:
        if self.is_source:
            self._check()
            self.commands.put(("offload",))

    def num_completed_offloads(self) -> int:
        with self.condition:
            self._check()
            return len(self.offloads)

    def take_completed_offloads(self, count: int) -> list[bool]:
        with self.condition:
            # PP0 has already observed a completed global reduction. Its peer
            # worker may not yet have published that result to its local queue.
            while len(self.offloads) < count:
                self._check()
                self.condition.wait(timeout=1)
            return [self.offloads.popleft() for _ in range(count)]

    def reset(self) -> None:
        with self.condition:
            self.requested_resets += 1
            target = self.requested_resets
            if self.is_source:
                self.commands.put(("reset",))
            while self.completed_resets < target:
                self._check()
                self.condition.wait(timeout=1)

    def close(self) -> None:
        if self.is_source and self.thread.is_alive():
            self.commands.put(("close",))
        self.thread.join(timeout=30)
        if self.thread.is_alive():
            raise RuntimeError("PP linker coordinator did not stop")
        self._check()

    def _run(self) -> None:
        try:
            while True:
                command = None
                if self.is_source:
                    try:
                        command = self.commands.get(timeout=1)
                    except Empty:
                        pass
                objects = [command]
                dist.broadcast_object_list(objects, src=self.source, group=self.group)
                command = objects[0]
                if command is None:
                    continue
                kind = command[0]
                if kind == "close":
                    return
                if kind == "lookup":
                    _, rid, sequence, hashes, queued_at = command
                    started = time.monotonic()
                    mask = torch.zeros(len(hashes) + 1, dtype=torch.int32)
                    try:
                        for pages in self.lookup(rid, hashes):
                            if 0 < pages <= len(hashes):
                                mask[pages] = 1
                    except Exception:
                        mask.zero_()
                        logger.exception("PP linker lookup failed for req=%s", rid)
                    dist.all_reduce(mask, op=dist.ReduceOp.MIN, group=self.group)
                    result = PPLookupResult(
                        sequence,
                        hashes,
                        mask.nonzero().flatten().tolist(),
                        time.monotonic(),
                    )
                    with self.condition:
                        # The source may already have queued release/reuse. Do
                        # not overwrite its newer expected generation.
                        if not self.is_source:
                            self.expected[rid] = sequence
                        # A later lookup can finish before a downstream PP
                        # scheduler consumes the preceding admission attempt.
                        self.results.setdefault(rid, {})[sequence] = result
                        self.condition.notify_all()
                    if self.is_source:
                        logger.info(
                            "PP linker lookup rid=%s pages=%d/%d queued=%.0fms "
                            "took=%.0fms backlog=%d",
                            rid,
                            result.pages[-1] if result.pages else 0,
                            len(hashes),
                            (started - queued_at) * 1e3,
                            (result.completed_at - started) * 1e3,
                            self.commands.qsize(),
                        )
                elif kind == "offload":
                    # This callback waits for actual completion, not a timeout
                    # converted to a miss: source device pages remain pinned.
                    success = torch.tensor(int(self.wait_offload()), dtype=torch.int32)
                    dist.all_reduce(success, op=dist.ReduceOp.MIN, group=self.group)
                    with self.condition:
                        self.offloads.append(bool(success.item()))
                        self.condition.notify_all()
                elif kind == "release":
                    _, rid, sequence = command
                    with self.condition:
                        versions = self.results.get(rid, {})
                        for old in [s for s in versions if s <= sequence]:
                            versions.pop(old)
                        if not versions:
                            self.results.pop(rid, None)
                        if self.expected.get(rid, sequence + 1) <= sequence:
                            self.expected.pop(rid)
                elif kind == "reset":
                    # Ordered after all prior I/O tickets on every rank.
                    with self.condition:
                        self.expected.clear()
                        self.pending_queries.clear()
                        self.results.clear()
                        self.offloads.clear()
                        self.completed_resets += 1
                        self.condition.notify_all()
                else:
                    raise ValueError(f"Unknown PP linker command: {kind}")
        except BaseException as error:
            logger.exception("PP linker background agreement failed")
            with self.condition:
                self.error = error
                self.condition.notify_all()
