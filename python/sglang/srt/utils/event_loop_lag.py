"""Report asyncio event-loop stalls together with the call that caused them."""

import asyncio
import logging
import sys
import threading
import time
import traceback
from collections import Counter
from typing import Optional

logger = logging.getLogger(__name__)

_MIN_THRESHOLD_S = 0.02


class EventLoopLagMonitor:
    """Log when an event loop stops running callbacks for longer than a threshold.

    A heartbeat callback on the loop stamps the time; a daemon thread watches the
    stamp. While the stamp is stale past the threshold, the loop thread's stack is
    sampled every tick. The first sample is logged when the stall is detected (so
    a hang is still reported); when the loop recovers, the stall length and the
    most frequently sampled stack, i.e. the dominant blocking call, are logged.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, threshold_s: float, name: str):
        self._loop = loop
        # The heartbeat period must stay well below the threshold.
        self._threshold_s = max(threshold_s, _MIN_THRESHOLD_S)
        self._interval_s = self._threshold_s / 4
        self._name = name
        self._last_beat = time.monotonic()
        self._loop_thread_id: Optional[int] = None

    def start(self) -> None:
        self._loop.call_soon_threadsafe(self._beat)
        threading.Thread(
            target=self._watch, name=f"{self._name}-loop-lag", daemon=True
        ).start()

    def _beat(self) -> None:
        self._loop_thread_id = threading.get_ident()
        self._last_beat = time.monotonic()
        self._loop.call_later(self._interval_s, self._beat)

    def _watch(self) -> None:
        stall_start = None
        samples: Counter = Counter()
        was_idle = False
        idle_end = 0.0
        while True:
            time.sleep(self._interval_s)
            if self._loop.is_closed():
                return
            if not self._loop.is_running():
                # A stopped loop runs no callbacks; that is idleness, not a stall.
                stall_start, was_idle = None, True
                samples.clear()
                continue
            if was_idle:
                # Measure lag from the restart, not from the last pre-stop beat.
                was_idle, idle_end = False, time.monotonic()

            last_beat = max(self._last_beat, idle_end)
            lag_s = time.monotonic() - last_beat
            if lag_s < self._threshold_s:
                if stall_start is not None:
                    self._report_end(last_beat - stall_start, samples)
                    stall_start = None
                    samples.clear()
                continue

            stack = self._loop_thread_stack()
            samples[stack] += 1
            if stall_start is None:
                stall_start = last_beat
                logger.warning(
                    "%s event loop blocked for %.0f ms (threshold %.0f ms); "
                    "loop thread stack:\n%s",
                    self._name,
                    lag_s * 1000,
                    self._threshold_s * 1000,
                    stack,
                )

    def _report_end(self, stall_s: float, samples: Counter) -> None:
        stack, hits = samples.most_common(1)[0]
        logger.warning(
            "%s event loop stall ended after %.0f ms; most sampled stack "
            "(%d/%d samples):\n%s",
            self._name,
            stall_s * 1000,
            hits,
            sum(samples.values()),
            stack,
        )

    def _loop_thread_stack(self) -> str:
        frame = sys._current_frames().get(self._loop_thread_id)
        if frame is None:
            return "<unavailable>"
        return "".join(traceback.format_stack(frame, limit=15))
