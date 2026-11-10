"""CPU transport contracts for direct Mooncake DCP page sends."""

import concurrent.futures
import unittest
from itertools import product
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from disagg_test_utils import CopyTransport

from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMooncakeDcpPage(CustomTestCase):
    page_size = 4
    widths = (3, 5)

    def _case(self, *, custom_pool=False):
        sources = [
            ((np.arange(64 * width) + layer * 41) % 251)
            .astype(np.uint8)
            .reshape(64, width)
            for layer, width in enumerate(self.widths)
        ]
        destinations = [np.full_like(source, 0xEE) for source in sources]
        manager = object.__new__(MooncakeKVManager)
        manager.kv_args = SimpleNamespace(
            page_size=self.page_size,
            kv_data_ptrs=[buffer.ctypes.data for buffer in sources],
            kv_item_lens=[buffer[0].nbytes * self.page_size for buffer in sources],
            kv_layer_ids=[11, 17],
        )
        manager.enable_custom_mem_pool = custom_pool
        manager.custom_mem_pool_type = "NVLINK" if custom_pool else None
        manager.max_transfer_batch_indices = 0
        manager.enable_deferred_decode_kv_release = False
        manager.is_mla_backend = True
        manager.is_hybrid_mla_backend = False
        manager.pp_size = 1
        transport = CopyTransport(sources, destinations)
        manager._transfer_data = transport
        return manager, sources, destinations, transport

    def _send(self, manager, destinations, *, executor=None, **kwargs):
        return manager.send_kvcache_dcp(
            "session",
            kwargs.pop("source_pages", np.array([7], dtype=np.int32)),
            # Reversed destination registration exercises the existing layer map.
            [buffer.ctypes.data for buffer in reversed(destinations)],
            kwargs.pop("destination_pages", np.array([10], dtype=np.int32)),
            dcp_token_item_lens=list(self.widths),
            dst_dcp_kv_layout="page",
            dst_dcp_size=kwargs.pop("dcp_size", 3),
            dst_dcp_rank=kwargs.pop("rank", 0),
            src_page_offset=kwargs.pop("offset", 0),
            decode_prefix_len=kwargs.pop("prefix", 12),
            num_kv_tokens=kwargs.pop("length", 4),
            executor=executor,
            dst_layer_ids=[17, 11],
        )

    def test_fragmented_chunks_copy_owned_pages_including_unused_tail(self):
        source_pages = np.array([7, 1, 9, 3, 12, 4, 8], dtype=np.int32)
        destination_pages = np.array([10, 2, 14], dtype=np.int32)
        for rank, custom_pool in product(range(3), (False, True)):
            with self.subTest(rank=rank, custom_pool=custom_pool):
                manager, sources, destinations, transport = self._case(
                    custom_pool=custom_pool
                )
                with (
                    concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor,
                    patch(
                        "sglang.srt.disaggregation.common.dcp_pack.try_pack_dcp_src",
                        side_effect=AssertionError("page send must not pack"),
                    ),
                ):
                    for offset, count, length in ((0, 2, 8), (2, 3, 12), (5, 2, 5)):
                        self.assertEqual(
                            self._send(
                                manager,
                                destinations,
                                executor=executor,
                                source_pages=source_pages[offset : offset + count],
                                destination_pages=destination_pages,
                                rank=rank,
                                offset=offset,
                                length=length,
                            ),
                            0,
                        )
                        if rank == 2 and offset == 0:
                            self.assertEqual(transport.bytes_sent, 0)

                owned = [i for i in range(28) if ((12 + i) // 4) % 3 == rank]
                for source, destination in zip(sources, destinations):
                    expected = np.full_like(destination, 0xEE)
                    for local, logical in enumerate(owned):
                        source_row = source_pages[logical // 4] * 4 + logical % 4
                        destination_row = destination_pages[local // 4] * 4 + local % 4
                        expected[destination_row] = source[source_row]
                    np.testing.assert_array_equal(destination, expected)
                self.assertEqual(transport.bytes_sent, len(owned) * sum(self.widths))

    def test_page_send_propagates_transport_failure(self):
        """A failed page transfer must not be reported as a successful handoff."""
        for custom_pool in (False, True):
            with self.subTest(custom_pool=custom_pool):
                manager, _, destinations, _ = self._case(custom_pool=custom_pool)
                with (
                    concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor,
                    patch.object(manager, "_transfer_data", return_value=-9),
                ):
                    self.assertEqual(
                        self._send(manager, destinations, executor=executor), -9
                    )

    def test_generic_send_merges_pages_without_crossing_physical_gaps(self):
        manager, sources, destinations, transport = self._case()
        with patch.object(manager, "_transfer_data", wraps=transport) as transfer:
            self.assertEqual(
                self._send(
                    manager,
                    destinations,
                    source_pages=np.array([10, 13, 11, 14, 3, 15, 4, 12, 5], np.int32),
                    destination_pages=np.array([1, 2, 3, 8, 9], np.int32),
                    dcp_size=2,
                    prefix=0,
                    length=34,
                ),
                0,
            )
        expected = [
            (
                source.ctypes.data + src_page * 4 * width,
                destination.ctypes.data + dst_page * 4 * width,
                page_count * 4 * width,
            )
            for source, destination, width in zip(sources, destinations, self.widths)
            for src_page, dst_page, page_count in ((10, 1, 2), (3, 3, 1), (4, 8, 2))
        ]
        transfer.assert_called_once_with("session", expected)
        self.assertEqual(transport.bytes_sent, 20 * sum(self.widths))


if __name__ == "__main__":
    unittest.main()
