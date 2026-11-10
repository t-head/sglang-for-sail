"""Staging allocations cannot outlive teardown of their decode request."""

import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.disaggregation.common import staging_handler as staging_mod
from sglang.srt.disaggregation.common.staging_handler import (
    DecodeStagingHandler,
    StagingManagerMixin,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestStagingLifetime(CustomTestCase):
    def _make_room(self):
        receiver = SimpleNamespace(
            prefill_info=SimpleNamespace(attn_tp_size=8),
            chunk_staging_infos=[],
        )
        req = SimpleNamespace(kv_receiver=receiver, _chunk_events=[])
        mgr = StagingManagerMixin()
        mgr._staging_ctx = SimpleNamespace(
            allocator=object(), room_receivers={7: receiver}, room_bootstrap={7: []}
        )
        mgr.kv_args = object()
        mgr.attn_tp_size = 4
        handler = DecodeStagingHandler.__new__(DecodeStagingHandler)
        handler._lifecycle_lock = threading.RLock()
        handler._room_to_decode_req = {7: req}
        handler._room_to_receiver = {7: receiver}
        handler._writer_counts = {}
        handler.kv_manager = mgr
        handler.staging_allocator = SimpleNamespace(_scatter_stream=None)
        handler.register_wm_subscriber = Mock()
        handler._free_and_send_watermark = Mock()
        mgr._staging_handler = handler
        return mgr, handler, req, receiver

    def test_allocation_started_before_teardown_is_reclaimed(self):
        mgr, handler, req, receiver = self._make_room()
        allocation_entered = threading.Event()
        finish_allocation = threading.Event()
        teardown_finished = threading.Event()

        def delayed_allocation(*args):
            allocation_entered.set()
            if not finish_allocation.wait(2):
                raise TimeoutError("test did not release staging allocation")
            receiver.chunk_staging_infos.append((31, 128, 0, 256, 1))

        def teardown():
            handler.unregister_decode_req(7)
            teardown_finished.set()

        with (
            patch.object(staging_mod, "handle_staging_req", delayed_allocation),
            ThreadPoolExecutor(max_workers=2) as executor,
        ):
            allocation = executor.submit(
                mgr._handle_staging_req, [b"STAGING_REQ", b"7", b"0", b"1", b"peer"]
            )
            retirement = None
            try:
                self.assertTrue(allocation_entered.wait(2))
                retirement = executor.submit(teardown)
                # The allocation already holds the request's lifecycle. A
                # teardown must not complete before its publication is visible.
                self.assertFalse(teardown_finished.wait(0.1))
            finally:
                finish_allocation.set()
            allocation.result(timeout=2)
            retirement.result(timeout=2)

        self.assertNotIn(7, handler._room_to_decode_req)
        self.assertNotIn(7, mgr._staging_ctx.room_receivers)
        self.assertEqual(receiver.chunk_staging_infos, [(-1, -1, 0, -1, 0)])
        handler._free_and_send_watermark.assert_called_once_with(31, req)

    def test_allocation_arriving_after_teardown_is_ignored(self):
        mgr, handler, _req, receiver = self._make_room()
        handler.unregister_decode_req(7)
        with patch.object(staging_mod, "handle_staging_req") as allocate:
            mgr._handle_staging_req([b"STAGING_REQ", b"7", b"0", b"1", b"peer"])
        allocate.assert_not_called()
        self.assertEqual(receiver.chunk_staging_infos, [])
        self.assertNotIn(7, mgr._staging_ctx.room_receivers)


if __name__ == "__main__":
    unittest.main()
