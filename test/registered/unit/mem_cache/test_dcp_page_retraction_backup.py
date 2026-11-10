import unittest

import torch

from sglang.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,
    MambaPool,
    MLATokenToKVPool,
)
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _pool(rows: int = 32) -> MLATokenToKVPool:
    pool = object.__new__(MLATokenToKVPool)
    pool.page_size = 2
    pool.layer_num = 2
    pool.cpu_offloading_chunk_size = 2
    pool.kv_buffer = [
        torch.arange(rows * 3, dtype=torch.float32).reshape(rows, 1, 3) + layer * 1000
        for layer in range(pool.layer_num)
    ]
    return pool


def _hybrid_pool() -> HybridLinearKVPool:
    pool = object.__new__(HybridLinearKVPool)
    pool.full_kv_pool = _pool()
    mamba = object.__new__(MambaPool)
    mamba.mamba_cache = MambaPool.State(
        conv=[torch.arange(10, dtype=torch.float32).reshape(1, 5, 2)],
        temporal=torch.arange(10, dtype=torch.float32).reshape(1, 5, 2) + 100,
    )
    pool.mamba_pool = mamba
    pool._mamba_translate = lambda ids: ids
    return pool


class TestDCPPageRetractionBackup(CustomTestCase):
    def test_restores_owned_rows_to_new_fragmented_slots(self):
        source = torch.tensor([30, 31, 32, 33, 34, 35, 6])
        destination = torch.tensor([48, 49, 50, 51, 52, 53, 12])
        next_destination = torch.tensor([60, 61, 62, 63, 64, 65, 18])

        for rank in range(3):
            with (
                self.subTest(rank=rank),
                get_context().override_server_args(dcp_size=3, dcp_kv_layout="page"),
                get_parallel().override(dcp_rank=rank),
            ):
                pool = _pool()
                source_rows = [10, 11, 2] if rank == 0 else [10, 11]
                destination_rows = [16, 17, 4] if rank == 0 else [16, 17]
                next_rows = [20, 21, 6] if rank == 0 else [20, 21]
                expected = [buffer[source_rows].clone() for buffer in pool.kv_buffer]

                backup = pool.get_cpu_copy(source)
                self.assertIsInstance(backup, list)
                for chunks, values in zip(backup, expected, strict=True):
                    torch.testing.assert_close(torch.cat(chunks), values)
                for buffer in pool.kv_buffer:
                    buffer.zero_()
                pool.load_cpu_copy(backup, destination)
                second_backup = pool.get_cpu_copy(destination)
                pool.load_cpu_copy(second_backup, next_destination)

                for buffer, values in zip(pool.kv_buffer, expected, strict=True):
                    torch.testing.assert_close(buffer[destination_rows], values)
                    torch.testing.assert_close(buffer[next_rows], values)
                    torch.testing.assert_close(
                        buffer[[0, 1, 2, 3]], torch.zeros(4, 1, 3)
                    )

    def test_rank_with_no_owned_rows_has_empty_copy(self):
        pool = _pool()
        first_page = torch.tensor([30, 31])

        with (
            get_context().override_server_args(dcp_size=3, dcp_kv_layout="page"),
            get_parallel().override(dcp_rank=2),
        ):
            backup = pool.get_cpu_copy(first_page)
            pool.load_cpu_copy(backup, torch.tensor([48, 49]))

        self.assertEqual(backup, [[], []])

    def test_hybrid_restores_mla_and_mamba(self):
        pool = _hybrid_pool()
        source = torch.tensor([30, 31, 32, 33, 34, 35, 6])
        destination = torch.tensor([48, 49, 50, 51, 52, 53, 12])
        source_state = torch.tensor([1])
        destination_state = torch.tensor([3])

        with (
            get_context().override_server_args(dcp_size=3, dcp_kv_layout="page"),
            get_parallel().override(dcp_rank=0),
        ):
            backup = pool.get_cpu_copy(source, mamba_indices=source_state)
            expected_mla = [[chunk.clone() for chunk in layer] for layer in backup[0]]
            expected_conv = [chunk.clone() for chunk in backup[1][0]]
            expected_mamba = backup[1][1].clone()
            for buffer in pool.full_kv_pool.kv_buffer:
                buffer.zero_()
            pool.mamba_pool.mamba_cache.temporal[:, destination_state] = 0
            pool.load_cpu_copy(backup, destination, mamba_indices=destination_state)

        for layer, expected in enumerate(expected_mla):
            self.assertTrue(
                torch.equal(pool.full_kv_pool.kv_buffer[layer][[16, 17]], expected[0])
            )
            self.assertTrue(
                torch.equal(pool.full_kv_pool.kv_buffer[layer][[4]], expected[1])
            )
        for conv, expected in zip(
            pool.mamba_pool.mamba_cache.conv, expected_conv, strict=True
        ):
            self.assertTrue(torch.equal(conv[:, destination_state], expected))
        self.assertTrue(
            torch.equal(
                pool.mamba_pool.mamba_cache.temporal[:, destination_state],
                expected_mamba,
            )
        )

    def test_token_layout_maps_owner_rows_without_changing_cpu_copy_format(self):
        for dcp_size in (1, 3):
            for rank in range(dcp_size):
                with (
                    self.subTest(dcp_size=dcp_size, rank=rank),
                    get_context().override_server_args(
                        dcp_size=dcp_size, dcp_kv_layout="token"
                    ),
                    get_parallel().override(
                        attn_dcp_size=dcp_size, attn_dcp_rank=rank
                    ),
                ):
                    pool = _pool()
                    source = torch.arange(6, 12)
                    destination = torch.arange(12, 18)
                    source_rows = source[rank::dcp_size] // dcp_size
                    destination_rows = destination[rank::dcp_size] // dcp_size
                    expected = [
                        buffer[source_rows].clone() for buffer in pool.kv_buffer
                    ]
                    backup = pool.get_cpu_copy(source)
                    self.assertIsInstance(backup, list)
                    for buffer in pool.kv_buffer:
                        buffer.zero_()
                    pool.load_cpu_copy(backup, destination)
                    for buffer, values in zip(pool.kv_buffer, expected, strict=True):
                        torch.testing.assert_close(buffer[destination_rows], values)


if __name__ == "__main__":
    unittest.main()
