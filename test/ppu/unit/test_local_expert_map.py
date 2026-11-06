# SPDX-License-Identifier: Apache-2.0
"""Equivalence tests for the fused logical->local expert-id lookup on PPU.

``StandardDispatcher`` translates the router's global expert ids through an
int32 per-rank table. Written as ``table[topk_ids]`` that costs two launches --
ATen promotes the int32 index to int64 before indexing -- which is the pair of
grid=(1,1,1) kernels the decode trace shows after every router. The fused port
has to reproduce the eager result exactly, including the two properties the
dispatcher relies on:

* the ``-1`` drop sentinel wraps to the table's spare trailing slot, because
  ATen wraps negative indices by the table length;
* ids owned by another EP rank map to that slot's ``-1``, not to a valid index.

Run standalone on one idle PPU device::

    python test/ppu/unit/test_local_expert_map.py
    python test/ppu/unit/test_local_expert_map.py --bench
"""

import sys
import unittest

import torch

from sglang.kernels.ops.moe.local_expert_map import local_expert_lookup

DEV = "cuda"
NUM_EXPERTS = 352
LOCAL_EXPERTS = 22  # 352 / EP16


def _table(rank=0, fill=-1):
    """Public mapping shape: [E+1] with the spare last slot as the drop target."""
    table = torch.full((NUM_EXPERTS + 1,), fill, dtype=torch.int32, device=DEV)
    start = rank * LOCAL_EXPERTS
    table[start : start + LOCAL_EXPERTS] = torch.arange(
        LOCAL_EXPERTS, dtype=torch.int32, device=DEV
    )
    table[NUM_EXPERTS] = -1
    return table


def _ids(rows, top_k, *, rank=0, seed=0):
    generator = torch.Generator(device=DEV).manual_seed(seed)
    ids = torch.randint(
        0,
        NUM_EXPERTS,
        (rows, top_k),
        dtype=torch.int32,
        device=DEV,
        generator=generator,
    )
    # Mix owned, foreign and dropped ids in every row.
    start = rank * LOCAL_EXPERTS
    ids[:, 0] = torch.randint(
        start, start + LOCAL_EXPERTS, (rows,), device=DEV, generator=generator
    )
    ids[:, 1] = -1
    return ids


class TestLocalExpertLookup(unittest.TestCase):
    def test_matches_eager_index(self):
        for rows in (1, 4, 16, 256):
            for top_k in (8, 9):
                with self.subTest(rows=rows, top_k=top_k):
                    table = _table(rank=rows % 16)
                    ids = _ids(rows, top_k, rank=rows % 16, seed=rows * 31 + top_k)
                    got = local_expert_lookup(ids, table)
                    want = table[ids]
                    self.assertEqual(got.dtype, want.dtype)
                    self.assertTrue(bool((got == want).all()), "fused lookup diverged")

    def test_every_rank_matches(self):
        """The table differs per rank; the gather must not assume rank 0."""
        ids = _ids(32, 8, seed=7)
        for rank in (0, 1, 7, 15):
            with self.subTest(rank=rank):
                table = _table(rank=rank)
                self.assertTrue(
                    bool((local_expert_lookup(ids, table) == table[ids]).all())
                )

    def test_drop_sentinel_uses_trailing_slot(self):
        table = _table()
        ids = torch.full((3, 8), -1, dtype=torch.int32, device=DEV)
        out = local_expert_lookup(ids, table)
        self.assertTrue(bool((out == -1).all()), "-1 must map to the drop slot")
        self.assertEqual(int(table[-1]), -1, "the trailing slot is the drop target")

    def test_foreign_ids_drop(self):
        """Ids owned by another rank must resolve to -1 rather than stay global."""
        table = _table(rank=0)
        ids = torch.tensor(
            [0, LOCAL_EXPERTS - 1, LOCAL_EXPERTS, NUM_EXPERTS - 1, -1],
            dtype=torch.int32,
            device=DEV,
        )
        got = local_expert_lookup(ids, table)
        self.assertEqual(
            got.tolist(), [0, LOCAL_EXPERTS - 1, -1, -1, -1], "wrong ep remap"
        )

    def test_empty_and_error_paths(self):
        table = _table()
        empty = torch.empty((0, 8), dtype=torch.int32, device=DEV)
        out = local_expert_lookup(empty, table)
        self.assertEqual(tuple(out.shape), (0, 8))
        self.assertEqual(out.dtype, torch.int32)

        ids = _ids(4, 8)
        with self.assertRaises(ValueError):
            local_expert_lookup(ids[:, ::2], table)  # non-contiguous ids
        with self.assertRaises(TypeError):
            local_expert_lookup(ids.float(), table)
        with self.assertRaises(ValueError):
            local_expert_lookup(ids, torch.empty(0, dtype=torch.int32, device=DEV))


def _bench(fn, iters=20, reps=5):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(iters):
            fn()
    best = float("inf")
    for _ in range(reps):
        begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
        begin.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        best = min(best, begin.elapsed_time(end) * 1e3 / iters)
    return best


def report_cost():
    """Compare the eager ``table[ids]`` pair against the fused single launch."""
    table = _table()
    print("\n== table[ids] vs fused lookup (us per call)")
    for rows in (1, 4, 16, 128, 2048, 8192):
        ids = _ids(rows, 8)
        eager = _bench(lambda: table[ids])
        fused = _bench(lambda: local_expert_lookup(ids, table))
        print(f"  rows={rows:5d}  eager {eager:6.2f}  fused {fused:6.2f}")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "needs a CUDA/PPU device"
    bench = "--bench" in sys.argv
    sys.argv = [a for a in sys.argv if a != "--bench"]
    unittest.main(verbosity=2, exit=False)
    if bench:
        report_cost()
