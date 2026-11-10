"""EventLoopLagMonitor names the call that blocked the event loop."""

import asyncio
import time

from sglang.srt.utils.event_loop_lag import EventLoopLagMonitor
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

LOGGER = "sglang.srt.utils.event_loop_lag"


def _short_blocking_call(seconds):
    time.sleep(seconds)


def _dominant_blocking_call(seconds):
    time.sleep(seconds)


def _run(loop, coro, caplog):
    with caplog.at_level("WARNING", logger=LOGGER):
        loop.run_until_complete(coro)


def _messages(caplog, needle):
    return [r.getMessage() for r in caplog.records if needle in r.getMessage()]


def test_reports_stall_duration_and_dominant_stack(caplog):
    loop = asyncio.new_event_loop()
    try:
        EventLoopLagMonitor(loop, threshold_s=0.05, name="TestLoop").start()

        async def main():
            await asyncio.sleep(0.1)
            _short_blocking_call(0.08)
            _dominant_blocking_call(0.4)
            await asyncio.sleep(0.2)

        _run(loop, main(), caplog)
    finally:
        loop.close()

    blocked = _messages(caplog, "event loop blocked")
    ended = _messages(caplog, "stall ended")
    assert len(blocked) == 1 and len(ended) == 1
    assert "_short_blocking_call" in blocked[0]  # first sample, at detection
    assert "_dominant_blocking_call" in ended[0]  # most sampled over the stall
    stall_ms = float(ended[0].split("after ")[1].split(" ms")[0])
    assert 430 <= stall_ms <= 650


def test_quiet_loop_logs_nothing(caplog):
    loop = asyncio.new_event_loop()
    try:
        # Below the minimum threshold, which is clamped up.
        EventLoopLagMonitor(loop, threshold_s=0.005, name="TestLoop").start()
        _run(loop, asyncio.sleep(0.4), caplog)
    finally:
        loop.close()

    assert not caplog.records


def test_stopped_loop_is_not_a_stall(caplog):
    loop = asyncio.new_event_loop()
    try:
        EventLoopLagMonitor(loop, threshold_s=0.05, name="TestLoop").start()
        _run(loop, asyncio.sleep(0.1), caplog)
        time.sleep(0.4)  # loop exists but is not running
        _run(loop, asyncio.sleep(0.2), caplog)
    finally:
        loop.close()

    assert not caplog.records
