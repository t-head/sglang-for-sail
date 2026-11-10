"""PD transfer admission/drain races without a GPU or transport engine."""

import queue
import threading
import unittest

from sglang.srt.disaggregation.common.transfer_lifetime import TransferLifetimeTracker
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestTransferLifetime(CustomTestCase):
    def test_queued_and_requeued_chunks_hold_independent_leases(self):
        lifetime = TransferLifetimeTracker()
        room = 10
        self.assertTrue(lifetime.open(room))
        # Both chunks are admitted while still queued. Requeueing the first
        # staging chunk keeps its lease; completing the second cannot drain it.
        self.assertTrue(lifetime.try_acquire(room))
        self.assertTrue(lifetime.try_acquire(room))
        lifetime.close(room)
        self.assertFalse(lifetime.try_acquire(room))
        lifetime.release(room)
        self.assertFalse(lifetime.is_drained(room))
        lifetime.release(room)
        self.assertTrue(lifetime.is_drained(room))

    def test_inflight_write_prevents_reuse_after_cancel(self):
        lifetime = TransferLifetimeTracker()
        room = 20
        lifetime.open(room)
        self.assertTrue(lifetime.try_acquire(room))
        writer_started = threading.Event()
        finish_write = threading.Event()
        errors = queue.Queue()
        buffer = bytearray(b"initial")

        def writer():
            try:
                writer_started.set()
                if not finish_write.wait(timeout=5):
                    raise TimeoutError("Test did not release transfer writer")
                buffer[:] = b"A-state"
                lifetime.release(room)
            except BaseException as exc:
                errors.put(exc)

        thread = threading.Thread(target=writer)
        thread.start()
        try:
            self.assertTrue(writer_started.wait(timeout=5))
            lifetime.close(room)
            self.assertFalse(lifetime.is_drained(room))
            with self.assertRaisesRegex(RuntimeError, "transfers pending"):
                lifetime.clear(room)
            self.assertEqual(buffer, b"initial")
            self.assertFalse(lifetime.try_acquire(room))
        finally:
            finish_write.set()
            thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        if not errors.empty():
            raise errors.get_nowait()
        self.assertTrue(lifetime.is_drained(room))
        self.assertEqual(buffer, b"A-state")
        lifetime.clear(room)
        # Only now would the owning allocator be allowed to hand the buffer to B.
        buffer[:] = b"B-state"
        self.assertEqual(buffer, b"B-state")
        self.assertFalse(lifetime.try_acquire(room))

    def test_abort_before_sender_open_does_not_resurrect_room(self):
        lifetime = TransferLifetimeTracker()
        room = 30
        lifetime.close(room)
        lifetime.close(room)  # Repeated ABORT is harmless.
        self.assertTrue(lifetime.is_drained(room))
        self.assertFalse(lifetime.try_acquire(room))
        lifetime.clear(room)  # No sender has claimed the tombstone yet.
        self.assertFalse(lifetime.open(room))
        self.assertTrue(lifetime.is_closed(room))
        self.assertFalse(lifetime.try_acquire(room))
        lifetime.clear(room)
        self.assertFalse(lifetime.try_acquire(room))

    def test_close_and_enqueue_race_never_acknowledges_an_admitted_write(self):
        for room in range(100, 132):
            with self.subTest(room=room):
                lifetime = TransferLifetimeTracker()
                lifetime.open(room)
                start = threading.Barrier(2)
                result = queue.Queue()

                def enqueue():
                    start.wait(timeout=5)
                    result.put(lifetime.try_acquire(room))

                thread = threading.Thread(target=enqueue)
                thread.start()
                start.wait(timeout=5)
                lifetime.close(room)
                thread.join(timeout=5)
                self.assertFalse(thread.is_alive())
                admitted = result.get(timeout=5)
                self.assertEqual(lifetime.is_drained(room), not admitted)
                self.assertFalse(lifetime.try_acquire(room))
                if admitted:
                    lifetime.release(room)
                self.assertTrue(lifetime.is_drained(room))
                lifetime.clear(room)

    def test_unregistered_and_duplicate_completion_fail_closed(self):
        lifetime = TransferLifetimeTracker()
        self.assertFalse(lifetime.try_acquire(40))
        with self.assertRaisesRegex(RuntimeError, "underflow"):
            lifetime.release(40)
        lifetime.open(40)
        self.assertTrue(lifetime.try_acquire(40))
        lifetime.release(40)
        with self.assertRaisesRegex(RuntimeError, "underflow"):
            lifetime.release(40)


if __name__ == "__main__":
    unittest.main()
