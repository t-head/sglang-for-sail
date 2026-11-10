from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import sys

sys.modules["libtpu"] = None
import ctypes
import ctypes.util
import mmap
import os
import tempfile
import unittest
import unittest.mock

import torch

from sglang.srt.mem_cache.pool_host.common import ShmHostTensorAllocator
from sglang.srt.mem_cache.storage.mmap import alloc_mmap, alloc_shm
from sglang.srt.mem_cache.storage.mmap.mmap_allocator import (
    _mmap_prefaulted,
    _prefault_shm_shard,
    create_shm,
    map_shm,
)
from sglang.test.test_utils import CustomTestCase


class TestMmapAllocator(unittest.TestCase):
    def test_alloc_mmap(self):
        dims = (10, 1024)
        dtype = torch.float32
        tensor = alloc_mmap(dims, dtype)
        self.assertEqual(tensor.shape, dims)
        self.assertEqual(tensor.dtype, dtype)
        # Verify it has mapped memory address
        self.assertGreater(tensor.data_ptr(), 0)

    def test_alloc_shm(self):
        dims = (10, 1024)
        dtype = torch.float32
        tensor, fd, mm = alloc_shm(dims, dtype)

        self.assertEqual(tensor.shape, dims)
        self.assertEqual(tensor.dtype, dtype)
        self.assertGreater(tensor.data_ptr(), 0)
        self.assertGreaterEqual(fd, 0)
        self.assertIsInstance(mm, mmap.mmap)

        # Check that we can write to the tensor
        tensor[0, 0] = 42.0
        self.assertEqual(tensor[0, 0].item(), 42.0)

        # Check that the FD is open and valid
        try:
            os.lseek(fd, 0, os.SEEK_SET)
        except OSError:
            self.fail("FD is not valid or closed")

        # Cleanup
        mm.close()
        os.close(fd)

    def _assert_resident(self, mm, alloc_bytes):
        # mincore() reports per-page residency, so the invariant is checked
        # rather than inferred from the flags that were passed.
        addr = ctypes.addressof(ctypes.c_char.from_buffer(mm))
        npages = alloc_bytes // mmap.PAGESIZE
        vec = (ctypes.c_ubyte * npages)()
        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        if libc.mincore(ctypes.c_void_p(addr), ctypes.c_size_t(alloc_bytes), vec) != 0:
            self.skipTest("mincore unavailable")
        self.assertTrue(all(v & 1 for v in vec), "mapping was not fully pre-faulted")

    def test_mmap_prefaulted_leaves_no_lazy_page(self):
        """Both populate paths must return a mapping with every page resident.

        cudaHostRegister pins these buffers, so a page still unfaulted at
        registration lets the device read memory that is not backed yet.
        """
        alloc_bytes = 64 * mmap.PAGESIZE
        flags = mmap.MAP_SHARED | mmap.MAP_ANONYMOUS

        with self.subTest(path="madvise"):
            mm = _mmap_prefaulted(-1, alloc_bytes, flags)
            try:
                self._assert_resident(mm, alloc_bytes)
            finally:
                mm.close()

        # MAP_POPULATE is unreachable on a 5.14+ kernel, so CI never runs it;
        # force the branch or it ships untested.
        with (
            self.subTest(path="map_populate"),
            unittest.mock.patch(
                "sglang.srt.mem_cache.storage.mmap.mmap_allocator._has_madv_populate_write",
                return_value=False,
            ),
        ):
            mm = _mmap_prefaulted(-1, alloc_bytes, flags)
            try:
                self._assert_resident(mm, alloc_bytes)
            finally:
                mm.close()

    def test_shm_host_tensor_allocator(self):
        allocator = ShmHostTensorAllocator()
        dims = (2, 512)
        dtype = torch.int32

        tensor = allocator.allocate(dims, dtype, "cpu")
        self.assertEqual(tensor.shape, dims)
        self.assertEqual(tensor.dtype, dtype)
        self.assertIsNotNone(allocator.fd)
        self.assertGreaterEqual(allocator.fd, 0)

        # Write data and check
        tensor[1, 1] = 99
        self.assertEqual(tensor[1, 1].item(), 99)

        # Test destructor cleans up fd
        fd = allocator.fd
        # Trigger GC / deletion
        del allocator

        # Verify fd is closed
        with self.assertRaises(OSError):
            os.fstat(fd)

    def test_alloc_shm_unlinked(self):
        dims = (4, 256)
        dtype = torch.float32
        tensor, fd, mm = alloc_shm(dims, dtype)

        # On Linux, the path of an unlinked fd shows up in /proc/self/fd/
        # with a ' (deleted)' suffix.
        fd_path = f"/proc/self/fd/{fd}"
        try:
            resolved_path = os.readlink(fd_path)
            self.assertIn("sglang_host_pool_", resolved_path)
            self.assertTrue(resolved_path.endswith(" (deleted)"))
        except OSError:
            # If procfs is not available or readlink fails, fallback to direct path existence check
            self.assertFalse(os.path.exists(f"/dev/shm/sglang_host_pool_"))

        # Cleanup
        mm.close()
        os.close(fd)

    def test_alloc_shm_hugepage_warning(self):
        from sglang.srt.environ import envs

        envs.SGLANG_HUGEPAGE_SIZE.override("2MB")
        try:
            # Should succeed by falling back to plain page size mapping
            dims = (2, 2)
            tensor, fd, mm = alloc_shm(dims, torch.float32)
            self.assertEqual(tensor.shape, dims)
            mm.close()
            os.close(fd)
        finally:
            envs.SGLANG_HUGEPAGE_SIZE.override(None)

    def test_shm_host_tensor_allocator_invalid_device(self):
        allocator = ShmHostTensorAllocator()
        with self.assertRaises(AssertionError) as ctx:
            allocator.allocate((2, 2), torch.float32, device="cuda")
        self.assertIn("only supports CPU allocations", str(ctx.exception))


class TestShardedShmPopulation(CustomTestCase):
    def test_failed_shared_mapping_is_closed(self):
        """Failure during prefault or tensor creation must release the open mapping."""
        module = "sglang.srt.mem_cache.storage.mmap.mmap_allocator"
        for stage in ("prefault", "tensor"):
            with self.subTest(stage=stage), tempfile.TemporaryFile() as file:
                file.truncate(mmap.PAGESIZE)
                mappings = []

                def populate(mm, *, index, count):
                    mappings.append(mm)
                    if stage == "prefault":
                        raise OSError("prefault allocation failed")

                try:
                    with (
                        unittest.mock.patch(
                            f"{module}._prefault_shm_shard", side_effect=populate
                        ),
                        unittest.mock.patch(
                            "torch.frombuffer",
                            side_effect=MemoryError("tensor allocation failed"),
                        ),
                    ):
                        error = OSError if stage == "prefault" else MemoryError
                        with self.assertRaisesRegex(
                            error, f"{stage} allocation failed"
                        ):
                            map_shm(
                                file.fileno(),
                                (mmap.PAGESIZE,),
                                torch.uint8,
                                populate_shard=(0, 1),
                            )
                    self.assertTrue(mappings[0].closed)
                finally:
                    for mm in mappings:
                        mm.close()

    @unittest.skipUnless(sys.platform == "linux", "memfd residency needs Linux")
    def test_create_does_not_claim_pages_before_workers_touch_them(self):
        """The creator must leave placement to the workers' first touches."""
        fd = create_shm((16 * mmap.PAGESIZE,), torch.uint8)
        try:
            self.assertEqual(os.fstat(fd).st_size, 16 * mmap.PAGESIZE)
            self.assertEqual(os.fstat(fd).st_blocks, 0)
        finally:
            os.close(fd)

    @unittest.skipUnless(sys.platform == "linux", "Shared prefault uses Linux madvise")
    def test_shards_cover_uneven_and_smaller_than_world_mappings(self):
        """Shards must not overlap or omit pages, including empty shares."""
        for pages, workers in ((13, 4), (2, 4)):
            with self.subTest(pages=pages, workers=workers):
                fd = create_shm((pages * mmap.PAGESIZE,), torch.uint8)
                mm = mmap.mmap(fd, pages * mmap.PAGESIZE, flags=mmap.MAP_SHARED)
                os.close(fd)
                try:
                    # Keep mincore residency page-granular;
                    # a THP fault would mark a whole huge page.
                    mm.madvise(mmap.MADV_NOHUGEPAGE)
                    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
                    addr = ctypes.addressof(ctypes.c_char.from_buffer(mm))
                    previous = [0] * pages
                    sizes = []
                    for rank in range(workers):
                        _prefault_shm_shard(mm, index=rank, count=workers)
                        resident = (ctypes.c_ubyte * pages)()
                        self.assertEqual(
                            libc.mincore(
                                ctypes.c_void_p(addr),
                                ctypes.c_size_t(len(mm)),
                                resident,
                            ),
                            0,
                        )
                        current = [value & 1 for value in resident]
                        new_pages = [
                            i for i in range(pages) if current[i] and not previous[i]
                        ]
                        if new_pages:
                            self.assertEqual(
                                new_pages,
                                list(range(new_pages[0], new_pages[-1] + 1)),
                            )
                        sizes.append(len(new_pages))
                        previous = current
                    self.assertEqual(previous, [1] * pages)
                    self.assertEqual(sum(sizes), pages)
                    self.assertLessEqual(max(sizes) - min(sizes), 1)
                finally:
                    mm.close()

    def test_shard_population_failure_propagates(self):
        """A failed shard madvise must raise instead of leaving pages lazily faulted."""
        mm = unittest.mock.create_autospec(mmap.mmap, instance=True)
        mm.__len__.return_value = 4 * mmap.PAGESIZE
        mm.madvise.side_effect = OSError("unsupported population")
        with self.assertRaisesRegex(OSError, "unsupported population"):
            _prefault_shm_shard(mm, index=0, count=2)


if __name__ == "__main__":
    unittest.main()
