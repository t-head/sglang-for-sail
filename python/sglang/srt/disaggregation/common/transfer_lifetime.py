"""Track source-buffer leases across queued and running PD transfers.

Admission and cancellation share a lock: after ``close`` returns, no new
transfer can acquire the room. Every admitted chunk retains one lease through
queueing, execution, and any staging requeue, until all its I/O has completed.
"""

import threading
from dataclasses import dataclass


@dataclass
class _RoomLifetime:
    opened: bool = False
    closed: bool = False
    pending: int = 0


class TransferLifetimeTracker:
    """Thread-safe admission and drain accounting for one prefill rank.

    Room IDs must identify a single request until that request's sender is
    cleared. A cancellation arriving before sender construction leaves a
    tombstone, so the late sender cannot start a write after the drain ACK.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._rooms: dict[int, _RoomLifetime] = {}

    def open(self, room: int) -> bool:
        """Register the local sender; return False if already cancelled.

        Claiming an early-cancellation tombstone does not reopen admission.
        It only lets ``clear`` remove it once this sender has finished.
        """
        with self._lock:
            state = self._rooms.setdefault(room, _RoomLifetime())
            state.opened = True
            return not state.closed

    def try_acquire(self, room: int) -> bool:
        """Acquire one lease BEFORE enqueueing; unknown/closed rooms reject it."""
        with self._lock:
            state = self._rooms.get(room)
            if state is None or not state.opened or state.closed:
                return False
            state.pending += 1
            return True

    def release(self, room: int) -> None:
        """Release one completed chunk, never a temporarily requeued chunk.

        The caller must first drain all reads and writes launched by the chunk,
        including futures that are still running after another future failed.
        """
        with self._lock:
            state = self._rooms.get(room)
            if state is None or state.pending <= 0:
                raise RuntimeError(f"PD transfer lease underflow for room {room}")
            state.pending -= 1

    def close(self, room: int) -> None:
        """Atomically stop future admission, preserving all existing leases."""
        with self._lock:
            state = self._rooms.setdefault(room, _RoomLifetime())
            state.closed = True

    def is_closed(self, room: int) -> bool:
        with self._lock:
            state = self._rooms.get(room)
            return state is None or state.closed

    def is_drained(self, room: int) -> bool:
        """Whether this rank has no queued or running chunks for the room.

        A drain ACK additionally requires closing admission first. A merely
        idle, open room can accept another chunk after this check returns.
        """
        with self._lock:
            state = self._rooms.get(room)
            return state is None or state.pending == 0

    def clear(self, room: int) -> None:
        """Retire a drained sender without discarding early-abort tombstones."""
        with self._lock:
            state = self._rooms.get(room)
            if state is None:
                return
            state.closed = True
            if state.pending:
                raise RuntimeError(
                    f"Cannot clear room {room} with {state.pending} PD transfers pending"
                )
            if state.opened:
                del self._rooms[room]
