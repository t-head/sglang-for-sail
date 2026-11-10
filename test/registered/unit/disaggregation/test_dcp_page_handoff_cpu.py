"""CPU handoff and retraction contracts for page-layout DCP MLA KV."""

import concurrent.futures
from collections import defaultdict
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch
from disagg_test_utils import CopyTransport

from sglang.srt.disaggregation.base.conn import KVPoll, StateType
from sglang.srt.disaggregation.common.transfer_lifetime import TransferLifetimeTracker
from sglang.srt.disaggregation.common.utils import TransferKVChunk
from sglang.srt.disaggregation.mooncake.conn import (
    KVArgsRegisterInfo,
    MooncakeKVManager,
    MooncakeKVSender,
    TransferInfo,
)
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool, ReqToTokenPool
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


def _pool(rows: int = 64) -> MLATokenToKVPool:
    pool = object.__new__(MLATokenToKVPool)
    pool.page_size = 2
    pool.layer_num = 2
    pool.cpu_offloading_chunk_size = 2
    pool.kv_buffer = [
        torch.full((rows, 1, 576), -1.0, dtype=torch.float32)
        for _ in range(pool.layer_num)
    ]
    return pool


class TestDcpPageHandoffCpu(CustomTestCase):
    def _allocate_page_retraction(self, pool):
        """Allocate the global page map before P writes the local rows."""
        allocator = PagedTokenToKVPoolAllocator(
            size=192,
            page_size=6,
            dtype=torch.float32,
            device="cpu",
            kvcache=pool,
            need_sort=False,
        )
        priority_pages = [20, 4, 6, 9, 8, 2, 10, 13]
        allocator.free_pages = torch.tensor(
            priority_pages
            + [page for page in range(1, 33) if page not in priority_pages],
            dtype=torch.int64,
        )
        source_slots_all = allocator.alloc(24)
        source_slots = source_slots_all[:19].to(torch.int32)
        torch.testing.assert_close(
            source_slots,
            torch.tensor(
                list(range(120, 126))
                + list(range(24, 30))
                + list(range(36, 42))
                + [54],
                dtype=torch.int32,
            ),
        )
        req = Req(
            rid="page-retraction",
            origin_input_text="",
            origin_input_ids=list(range(19)),
            sampling_params=SamplingParams(),
        )
        req.output_ids.append(5)
        req_to_token = ReqToTokenPool(
            size=3, max_context_len=24, device="cpu", enable_memory_saver=False
        )
        self.assertIsNotNone(req_to_token.alloc([req]))
        req_to_token.write((req.kv.req_pool_idx, slice(0, 19)), source_slots)
        req.kv.kv_committed_len = req.kv.kv_allocated_len = 19
        # A disabled cache: the registry's replacement for ChunkCache (#43201).
        tree_cache = UnifiedRadixCache(
            CacheInitParams(
                disable=True,
                req_to_token_pool=req_to_token,
                token_to_kv_pool_allocator=allocator,
                page_size=6,
                tree_components=(ComponentType.FULL,),
            )
        )
        override = get_context().override_server_args()
        override.install()
        self.addCleanup(override.restore)
        return req, tree_cache, source_slots_all

    def _send_page_chunks(self, manager, destination, source):
        """Send the complete global prefix [0, 19), including its partial page."""
        chunks = [
            ([9, 3, 8, 1, 7], [20, 4], 0, 10),
            ([4, 5], [20, 4, 6], 5, 4),
            ([2, 11, 12], [20, 4, 6, 9], 7, 5),
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            for src_pages, dst_pages, offset, num_tokens in chunks:
                self.assertEqual(
                    manager.send_kvcache_dcp(
                        "page-session",
                        np.asarray(src_pages, dtype=np.int32),
                        [destination[1].ctypes.data, destination[0].ctypes.data],
                        np.asarray(dst_pages, dtype=np.int32),
                        dcp_token_item_lens=[source[0][0].nbytes, source[1][0].nbytes],
                        dst_dcp_kv_layout="page",
                        dst_dcp_size=3,
                        dst_dcp_rank=0,
                        src_page_offset=offset,
                        decode_prefix_len=0,
                        num_kv_tokens=num_tokens,
                        executor=executor,
                        dst_layer_ids=[17, 11],
                    ),
                    0,
                )

    def _assert_handoff_bytes(self, pool, source):
        received_rows = torch.tensor([40, 41, 8, 9, 12, 13, 18])
        source_rows = torch.tensor([18, 19, 2, 3, 10, 11, 24])
        received = []
        for layer, buffer in enumerate(pool.kv_buffer):
            expected = torch.from_numpy(source[layer][source_rows.numpy()])
            torch.testing.assert_close(buffer[received_rows], expected)
            received.append(expected.clone())
        return received_rows, received

    def _restore_and_release(
        self,
        pool,
        req,
        tree_cache,
        source_slots_all,
        received_rows,
        received,
    ):
        allocator = tree_cache.token_to_kv_pool_allocator
        req_to_token = tree_cache.req_to_token_pool
        with (
            get_context().override_server_args(dcp_size=3, dcp_kv_layout="page"),
            get_parallel().override(dcp_rank=0),
        ):
            req.offload_kv_cache(req_to_token, allocator)
            self.assertIsNotNone(req.kv.retraction_backup)
            release_kv_cache(req, tree_cache, checkpoint=False)
            self.assertEqual(allocator.available_size(), allocator.size)
            self.assertEqual(req_to_token.available_size(), req_to_token.size)
            blocker = Req(
                rid="page-retraction-blocker",
                origin_input_text="",
                origin_input_ids=list(range(24)),
                sampling_params=SamplingParams(),
            )
            self.assertIsNotNone(req_to_token.alloc([blocker]))
            blocker_slots = allocator.alloc(24)
            self.assertIsNotNone(blocker_slots)
            torch.testing.assert_close(blocker_slots, source_slots_all)
            req_to_token.write((blocker.kv.req_pool_idx, slice(0, 24)), blocker_slots)
            blocker.kv.kv_committed_len = blocker.kv.kv_allocated_len = 24
            self.assertIsNotNone(req_to_token.alloc([req]))
            new_slots_all = allocator.alloc(24)
            self.assertIsNotNone(new_slots_all)
            new_slots = new_slots_all[:19].to(torch.int32)
            torch.testing.assert_close(
                new_slots,
                torch.tensor(
                    list(range(48, 54))
                    + list(range(12, 18))
                    + list(range(60, 66))
                    + [78],
                    dtype=torch.int32,
                ),
            )
            req_to_token.write((req.kv.req_pool_idx, slice(0, 19)), new_slots)
            req.kv.kv_committed_len = req.kv.kv_allocated_len = 19
            for buffer in pool.kv_buffer:
                buffer.zero_()
            req.load_kv_cache(req_to_token, allocator)
        self.assertIsNone(req.kv.retraction_backup)
        new_rows = torch.tensor([16, 17, 4, 5, 20, 21, 26])
        for layer, buffer in enumerate(pool.kv_buffer):
            torch.testing.assert_close(buffer[new_rows], received[layer])
            self.assertTrue(
                torch.equal(
                    buffer[received_rows], torch.zeros_like(buffer[received_rows])
                )
            )
        release_kv_cache(req, tree_cache, checkpoint=False)
        release_kv_cache(blocker, tree_cache, checkpoint=False)
        self.assertEqual(allocator.available_size(), allocator.size)
        self.assertEqual(req_to_token.available_size(), req_to_token.size)

    def test_page_handoff_and_oom_restore_use_new_rows(self):
        """Restore the received prefix into newly allocated rows, then release it."""
        pool = _pool()
        source = [
            np.arange(64 * 576, dtype=np.float32).reshape(64, 1, 576) + layer * 100_000
            for layer in range(2)
        ]
        destination = [buffer.numpy() for buffer in pool.kv_buffer]
        manager = object.__new__(MooncakeKVManager)
        manager.kv_args = SimpleNamespace(
            page_size=2,
            kv_data_ptrs=[buffer.ctypes.data for buffer in source],
            kv_item_lens=[buffer[0].nbytes * 2 for buffer in source],
            kv_layer_ids=[11, 17],
        )
        manager.enable_custom_mem_pool = False
        manager.max_transfer_batch_indices = 0
        manager.enable_deferred_decode_kv_release = False
        manager._transfer_data = CopyTransport(source, destination)
        manager.is_mla_backend = True
        manager.is_hybrid_mla_backend = False
        manager.pp_size = 1
        req, tree_cache, source_slots_all = self._allocate_page_retraction(pool)
        self._send_page_chunks(manager=manager, destination=destination, source=source)
        received_rows, received = self._assert_handoff_bytes(pool=pool, source=source)
        self._restore_and_release(
            pool=pool,
            req=req,
            tree_cache=tree_cache,
            source_slots_all=source_slots_all,
            received_rows=received_rows,
            received=received,
        )

    def test_worker_sends_rows_for_registered_decode_layout(self):
        room = 81
        for layout, rank, num_tokens, source_rows in (
            ("page", 0, 6, [0, 1]),
            ("token", 0, 6, [0, 3]),
            ("page", 2, 2, []),  # No local MLA page; Mamba and aux still transfer.
        ):
            with self.subTest(layout=layout, rank=rank):
                source = np.arange(16, dtype=np.uint8).reshape(8, 2)
                destination = np.full_like(source, 0xEE)
                mamba_source = np.arange(32, dtype=np.uint8).reshape(4, 8)
                aux_source = (np.arange(24, dtype=np.uint8) + 100).reshape(3, 8)
                mamba_destination = np.full_like(mamba_source, 0xEE)
                aux_destination = np.full_like(aux_source, 0xEE)
                manager = object.__new__(MooncakeKVManager)
                manager.dcp_kv_layout = "token"
                manager.enable_trace = False
                manager.enable_staging = False
                manager.enable_custom_mem_pool = False
                manager.max_transfer_batch_indices = 0
                manager.enable_deferred_decode_kv_release = True
                manager.is_mla_backend = True
                manager.is_hybrid_mla_backend = False
                manager.pp_size = 1
                manager.pp_rank = 0
                manager.attn_tp_size = 1
                manager.attn_tp_rank = 0
                manager.attn_cp_size = 1
                manager.attn_cp_rank = 0
                manager._transfer_lifetime = TransferLifetimeTracker()
                manager._abort_drain_lock = threading.RLock()
                manager._deferred_ack_targets = {}
                manager._deferred_ack_poisoned_rooms = set()
                manager._staging_outstanding = defaultdict(int)
                manager.state_layout_rejections = {}
                manager.is_dummy_cp_rank = False
                manager.request_status = {room: KVPoll.WaitingForInput}
                manager.failed_sessions = set()
                manager.session_lock = threading.Lock()
                manager.failure_lock = threading.Lock()
                manager.failure_records = {}
                manager.req_to_decode_prefix_len = {room: 0}
                manager.bootstrap_port = 30000
                manager.kv_args = SimpleNamespace(
                    page_size=2,
                    num_draft_entries=0,
                    kv_item_lens=[source[0].nbytes * 2],
                    kv_data_ptrs=[source.ctypes.data],
                    kv_layer_ids=[0],
                    state_types=[StateType.MAMBA],
                    state_data_ptrs=[[mamba_source.ctypes.data]],
                    state_item_lens=[[mamba_source[0].nbytes]],
                    state_dim_per_tensor=[[]],
                    state_conv_shard_groups=[[]],
                    state_slice_outer_counts=[[]],
                    state_layer_ids=[[]],
                    aux_data_ptrs=[aux_source.ctypes.data],
                    aux_item_lens=[aux_source[0].nbytes],
                )
                manager._dcp_pack_buffers = []
                manager._send_multipart_locked = MagicMock()
                manager.decode_kv_args_table = {
                    "session": KVArgsRegisterInfo(
                        room="None",
                        endpoint="127.0.0.1",
                        dst_port=9999,
                        mooncake_session_id="session",
                        dst_kv_ptrs=[destination.ctypes.data],
                        dst_aux_ptrs=[aux_destination.ctypes.data],
                        dst_state_data_ptrs=[[mamba_destination.ctypes.data]],
                        dst_tp_rank=0,
                        dst_attn_tp_size=1,
                        dst_kv_item_len=4,
                        dst_state_item_lens=[[mamba_destination[0].nbytes]],
                        dst_state_dim_per_tensor=[[]],
                        dst_kv_layer_ids=[0],
                        dst_state_layer_ids=[[]],
                        dst_dcp_size=3,
                        dst_dcp_rank=rank,
                        requires_dcp_relayout=True,
                        dcp_token_item_lens=[2],
                        dcp_kv_layout=layout,
                    )
                }
                manager.transfer_infos = {
                    room: {
                        "session": TransferInfo(
                            room=room,
                            endpoint="127.0.0.1",
                            dst_port=9999,
                            mooncake_session_id="session",
                            dst_kv_indices=np.asarray(
                                [0] if source_rows else [], dtype=np.int32
                            ),
                            dst_aux_index=1,
                            dst_state_indices=[[1]],
                            required_dst_info_num=1,
                            is_dummy=False,
                            decode_prefix_len=0,
                        )
                    }
                }
                manager._transfer_data = CopyTransport(
                    [source, mamba_source, aux_source],
                    [destination, mamba_destination, aux_destination],
                )
                chunk = TransferKVChunk(
                    room=room,
                    prefill_kv_indices=np.arange(num_tokens // 2, dtype=np.int32),
                    index_slice=slice(0, num_tokens // 2),
                    is_last_chunk=True,
                    prefill_aux_index=0,
                    state_indices=[[0]],
                    num_kv_tokens=num_tokens,
                )
                with get_context().override_server_args(dp_size=1):
                    sender = MooncakeKVSender(
                        mgr=manager,
                        bootstrap_addr="127.0.0.1:30000",
                        bootstrap_room=room,
                    )
                manager.update_status(room, KVPoll.WaitingForInput)
                self.assertTrue(manager._transfer_lifetime.try_acquire(room))
                chunk.lease_held = True
                queue = MagicMock()
                queue.get.side_effect = [chunk, SystemExit]
                with self.assertRaises(SystemExit):
                    manager.transfer_worker(queue, executor=None)

                expected = np.full_like(destination, 0xEE)
                expected[: len(source_rows)] = source[source_rows]
                np.testing.assert_array_equal(destination, expected)
                for src, dst in (
                    (mamba_source, mamba_destination),
                    (aux_source, aux_destination),
                ):
                    expected = np.full_like(dst, 0xEE)
                    expected[1] = src[0]
                    np.testing.assert_array_equal(dst, expected)
                self.assertEqual(
                    manager._transfer_data.bytes_sent, len(source_rows) * 2 + 16
                )
                self.assertEqual(manager.request_status[room], KVPoll.Success)
                manager._send_multipart_locked.assert_called_once()
                self.assertEqual(
                    manager._send_multipart_locked.call_args.args[1],
                    [
                        str(room).encode("ascii"),
                        str(KVPoll.Success).encode("ascii"),
                        b"0",
                    ],
                )
                # Upstream's worker retires a concluded room's destinations.
                self.assertNotIn(room, manager.transfer_infos)
                self.assertTrue(sender.is_source_release_safe())
                sender.clear()
                self.assertNotIn(room, manager.transfer_infos)
                self.assertNotIn(room, manager.req_to_decode_prefix_len)


if __name__ == "__main__":
    unittest.main()
