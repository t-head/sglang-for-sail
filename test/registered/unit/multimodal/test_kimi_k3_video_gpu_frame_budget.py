"""Kimi-K3 video GPU-preprocess frame-weighted admission gate.

``KimiK3ImageProcessor._video_gpu_preprocess_slot`` used to be a plain
``asyncio.Semaphore`` counting concurrent *requests* -- two requests each
individually within ``SGLANG_K3_VIDEO_MAX_SAMPLED_FRAMES`` could still add up
past available GPU memory when they land at the same time, regardless of how
many frames either one carries. It is now a frame-count-weighted gate backed
by an ``asyncio.Condition``: requests block until enough of a shared frame
budget (``SGLANG_K3_VIDEO_MAX_INFLIGHT_FRAMES``) is free, weighted by their
own sampled-frame count rather than counted as one fixed-size slot.

On top of that, ``SGLANG_K3_VIDEO_MAX_QUEUE_DEPTH`` bounds how many requests
may be waiting at this gate at once -- a request that would push the queue
past this depth is rejected with HTTP 400 before it joins the queue, instead
of queueing indefinitely behind an unbounded pile of already-decoded frame
tensors.

These tests exercise the gate directly against a minimal stand-in object
carrying just the state the method reads/writes (``_video_gpu_frame_budget``,
``_video_gpu_inflight_frames``, ``_video_gpu_condition``,
``_video_gpu_queue_depth_limit``, ``_video_gpu_queue_depth``) plus the method
itself bound onto it -- this is the real production code path, not a
reimplementation, just invoked without constructing a full
``KimiK3ImageProcessor`` (which needs a tokenizer/HF processor and is
unrelated to this gate's own logic).
"""

import asyncio
import unittest

from fastapi import HTTPException

from sglang.srt.multimodal.processors.kimi_k3 import KimiK3ImageProcessor
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_gate(max_inflight_frames, max_queue_depth=0):
    """A bare object carrying only the state _video_gpu_preprocess_slot needs."""
    gate = type("_Gate", (), {})()
    gate._video_gpu_frame_budget = (
        max_inflight_frames if max_inflight_frames > 0 else None
    )
    gate._video_gpu_inflight_frames = 0
    gate._video_gpu_condition = asyncio.Condition()
    gate._video_gpu_queue_depth_limit = (
        max_queue_depth if max_queue_depth > 0 else None
    )
    gate._video_gpu_queue_depth = 0
    gate._video_gpu_preprocess_slot = (
        KimiK3ImageProcessor._video_gpu_preprocess_slot.__get__(gate)
    )
    return gate


class TestKimiK3VideoGpuFrameBudget(unittest.IsolatedAsyncioTestCase):
    async def test_single_request_acquires_and_releases_its_frame_count(self):
        gate = _make_gate(256)

        async with gate._video_gpu_preprocess_slot(100):
            self.assertEqual(gate._video_gpu_inflight_frames, 100)

        self.assertEqual(gate._video_gpu_inflight_frames, 0)

    async def test_two_requests_that_individually_fit_but_together_exceed_serialize(
        self,
    ):
        # The exact incident shape this gate exists for: 112 + 153 sampled
        # frames, each under a 256 budget alone, 265 combined over it.
        gate = _make_gate(256)
        events = []

        async def video_request(name, sampled_frames, hold_seconds):
            async with gate._video_gpu_preprocess_slot(sampled_frames):
                events.append(f"{name}:enter")
                await asyncio.sleep(hold_seconds)
                events.append(f"{name}:exit")

        await asyncio.gather(
            video_request("a", 112, 0.05),
            video_request("b", 153, 0.05),
        )

        # "a" must fully exit before "b" enters -- 112 + 153 > 256, so the
        # gate must not let both be in flight together, even though each
        # individually passes SGLANG_K3_VIDEO_MAX_SAMPLED_FRAMES's 256 cap.
        self.assertEqual(events, ["a:enter", "a:exit", "b:enter", "b:exit"])

    async def test_requests_that_fit_together_run_concurrently(self):
        gate = _make_gate(256)
        events = []

        async def video_request(name, sampled_frames, hold_seconds):
            async with gate._video_gpu_preprocess_slot(sampled_frames):
                events.append(f"{name}:enter")
                await asyncio.sleep(hold_seconds)
                events.append(f"{name}:exit")

        await asyncio.gather(
            video_request("a", 100, 0.05),
            video_request("b", 100, 0.05),
        )

        # 100 + 100 = 200 <= 256: both should be in flight together, not
        # serialized -- "b" enters before "a" exits.
        self.assertEqual(events[:2], ["a:enter", "b:enter"])
        self.assertEqual(set(events[2:]), {"a:exit", "b:exit"})

    async def test_request_bigger_than_the_whole_budget_still_runs_alone(self):
        # A single request whose own sampled-frame count already exceeds the
        # global budget must not deadlock forever waiting for a threshold it
        # can never reach on its own -- it runs once nothing else is in flight.
        gate = _make_gate(256)

        async with gate._video_gpu_preprocess_slot(500):
            self.assertEqual(gate._video_gpu_inflight_frames, 500)

        self.assertEqual(gate._video_gpu_inflight_frames, 0)

    async def test_oversized_request_still_waits_for_other_inflight_requests(self):
        gate = _make_gate(256)
        events = []

        async def small_request():
            async with gate._video_gpu_preprocess_slot(50):
                events.append("small:enter")
                await asyncio.sleep(0.05)
                events.append("small:exit")

        async def oversized_request():
            # Give the small request a chance to acquire first.
            await asyncio.sleep(0.01)
            async with gate._video_gpu_preprocess_slot(500):
                events.append("oversized:enter")

        await asyncio.gather(small_request(), oversized_request())

        self.assertEqual(
            events, ["small:enter", "small:exit", "oversized:enter"]
        )

    async def test_third_request_queues_behind_the_first_two_until_a_slot_frees(self):
        gate = _make_gate(256)
        events = []

        async def video_request(name, sampled_frames, hold_seconds):
            async with gate._video_gpu_preprocess_slot(sampled_frames):
                events.append(f"{name}:enter")
                await asyncio.sleep(hold_seconds)
                events.append(f"{name}:exit")

        # a+b = 200 <= 256, both fit; c (100) needs one of them to free up
        # first (200+100=300 > 256, but 100+100=200 <= 256).
        await asyncio.gather(
            video_request("a", 100, 0.08),
            video_request("b", 100, 0.03),
            video_request("c", 100, 0.02),
        )

        self.assertEqual(events[:2], ["a:enter", "b:enter"])
        self.assertIn("c:enter", events)
        self.assertLess(events.index("b:exit"), events.index("c:enter"))

    async def test_zero_sampled_frames_is_a_no_op_for_image_only_requests(self):
        gate = _make_gate(1)

        async with gate._video_gpu_preprocess_slot(0):
            # Even though the budget is tiny, an image-only request (0
            # sampled frames) must not consume or wait on it.
            self.assertEqual(gate._video_gpu_inflight_frames, 0)

    async def test_disabled_budget_is_a_no_op(self):
        gate = _make_gate(0)
        self.assertIsNone(gate._video_gpu_frame_budget)

        async with gate._video_gpu_preprocess_slot(10_000):
            self.assertEqual(gate._video_gpu_inflight_frames, 0)

    async def test_queue_depth_limit_rejects_with_400_once_exceeded(self):
        # Budget of 100, queue depth limit of 1: the first 100-frame request
        # occupies the whole budget, a second must queue (queue depth 1, at
        # the limit), and a third arriving while the second is still queued
        # must be rejected with 400 rather than queueing as well.
        gate = _make_gate(100, max_queue_depth=1)
        events = []

        async def holder():
            async with gate._video_gpu_preprocess_slot(100):
                events.append("holder:enter")
                await asyncio.sleep(0.05)
                events.append("holder:exit")

        async def queued_request():
            await asyncio.sleep(0.01)
            async with gate._video_gpu_preprocess_slot(50):
                events.append("queued:enter")

        async def rejected_request():
            await asyncio.sleep(0.02)
            with self.assertRaises(HTTPException) as ctx:
                async with gate._video_gpu_preprocess_slot(50):
                    pass
            self.assertEqual(ctx.exception.status_code, 400)
            self.assertEqual(ctx.exception.detail, "Internal video format error")
            events.append("rejected")

        await asyncio.gather(holder(), queued_request(), rejected_request())

        self.assertEqual(
            events,
            ["holder:enter", "rejected", "holder:exit", "queued:enter"],
        )
        # The queue-depth counter must be back to 0 once the queued request
        # was admitted -- it must not leak past the request that used it.
        self.assertEqual(gate._video_gpu_queue_depth, 0)

    async def test_queue_depth_counter_frees_up_after_admission(self):
        # After the queue-depth limit rejects a request, a later request
        # must still be able to queue (and succeed) once a slot frees up --
        # the counter must not stay stuck at the limit forever.
        gate = _make_gate(100, max_queue_depth=1)
        events = []

        async def holder():
            async with gate._video_gpu_preprocess_slot(100):
                events.append("holder:enter")
                await asyncio.sleep(0.05)
                events.append("holder:exit")

        async def queued_request():
            await asyncio.sleep(0.01)
            async with gate._video_gpu_preprocess_slot(50):
                events.append("queued:enter")

        async def rejected_request():
            await asyncio.sleep(0.02)
            with self.assertRaises(HTTPException):
                async with gate._video_gpu_preprocess_slot(50):
                    pass
            events.append("rejected")

        async def follow_up_request():
            # By the time this runs, holder has released (t=0.05) and
            # queued_request's brief 50-frame hold has also released, so
            # the budget should be fully free again -- this must succeed
            # without being rejected or queueing.
            await asyncio.sleep(0.08)
            async with gate._video_gpu_preprocess_slot(100):
                events.append("follow_up:enter")

        await asyncio.gather(
            holder(), queued_request(), rejected_request(), follow_up_request()
        )

        self.assertEqual(gate._video_gpu_queue_depth, 0)
        self.assertIn("follow_up:enter", events)

    async def test_request_that_fits_immediately_is_never_rejected_by_queue_depth(
        self,
    ):
        # Even a queue-depth limit as tight as 0 (the smallest a caller
        # could set _video_gpu_queue_depth_limit to directly, bypassing the
        # "0 or below disables the limit" env-var convention) must not
        # reject a request that never actually has to wait -- the limit
        # only applies to requests that queue.
        gate = _make_gate(1000)
        gate._video_gpu_queue_depth_limit = 0

        async def video_request(sampled_frames):
            async with gate._video_gpu_preprocess_slot(sampled_frames):
                pass

        # Neither call needs to wait (cumulative well under the 1000
        # budget, and never overlapping in a way that forces a wait), so
        # the zero-depth limit is never actually tested against a real
        # queue.
        await video_request(10)
        await video_request(10)
        self.assertEqual(gate._video_gpu_queue_depth, 0)

    async def test_disabled_queue_depth_limit_allows_unbounded_queueing(self):
        gate = _make_gate(100, max_queue_depth=0)
        self.assertIsNone(gate._video_gpu_queue_depth_limit)
        events = []

        async def video_request(name, hold_seconds):
            async with gate._video_gpu_preprocess_slot(100):
                events.append(f"{name}:enter")
                await asyncio.sleep(hold_seconds)
                events.append(f"{name}:exit")

        # All three fully saturate the budget on their own (100/100), so
        # each of the 2nd and 3rd must queue behind the previous one -- with
        # no queue-depth limit, none of them are rejected.
        await asyncio.gather(
            video_request("a", 0.03),
            video_request("b", 0.02),
            video_request("c", 0.01),
        )

        self.assertEqual(len(events), 6)
        self.assertEqual(gate._video_gpu_queue_depth, 0)


if __name__ == "__main__":
    unittest.main()
