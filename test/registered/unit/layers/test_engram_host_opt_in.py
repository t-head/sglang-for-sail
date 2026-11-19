"""The default-off Host optimization must not initialize its implementation."""

import builtins
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.engram import EngramEmbedding
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _HostMemory:
    """Replace only the mmap/pinning/collective boundary with CPU storage."""

    def __init__(self, layout, nbytes, name, group):
        self.layout = layout
        self.bytes = torch.zeros(nbytes, dtype=torch.uint8)
        self.device_ptr = self.bytes.data_ptr()
        self.finished = []

    def finish_load(self, label):
        self.finished.append(label)


class TestEngramHostOptIn(CustomTestCase):
    def context(self, enabled, host, staging):
        stack = ExitStack()
        self.addCleanup(stack.close)
        for field, value in (
            (envs.SGLANG_OPT_ENGRAM_HOST, enabled),
            (envs.SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE, host),
            (envs.SGLANG_OPT_DSV41_ENGRAM_HOST_STAGING, staging),
            (envs.SGLANG_OPT_DSV41_ENGRAM_PEER_SHARDS, False),
            (envs.SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT, "shared"),
        ):
            stack.enter_context(field.override(value))
        stack.enter_context(
            patch(
                "sglang.srt.layers.engram.get_parallel",
                return_value=SimpleNamespace(tp_size=1, tp_rank=0, tp_group=None),
            )
        )
        stack.enter_context(
            patch("sglang.srt.layers.engram._HostTable", _HostMemory)
        )
        stack.enter_context(
            patch("sglang.srt.layers.engram._sorted_host_reads", return_value=False)
        )
        stack.enter_context(patch("torch.cuda.current_device", return_value=0))
        original = builtins.__import__

        def no_h2(name, *args, **kwargs):
            if name in (
                "sglang.srt.layers.engram_host",
                "sglang.kernels.ops.embeddings.engram_host",
            ):
                raise AssertionError("Unexpected H2 import")
            return original(name, *args, **kwargs)

        stack.enter_context(patch("builtins.__import__", side_effect=no_h2))
        return stack

    def test_disabled_shared_host_load_preserves_original_state(self):
        for staging in (False, True):
            with self.subTest(staging=staging), self.context(False, True, staging):
                embed = EngramEmbedding(4, 256, 0, (0, 4))
                self.assertFalse(hasattr(embed, "_host_opt"))
                embed.weight.data.copy_(
                    torch.full((4, 256), 2.0).to(torch.float8_e4m3fn)
                )
                embed.scale.data.view(torch.uint8).fill_(127)
                payload = embed.weight.data.view(torch.uint8).clone()
                embed.finish_load("cpu")
                self.assertEqual(embed.host_table.finished, ["cpu"])
                torch.testing.assert_close(
                    embed.weight.data.view(torch.uint8), payload, rtol=0, atol=0
                )

    def test_enabled_without_shared_host_keeps_original_load(self):
        with self.context(True, False, False):
            embed = EngramEmbedding(4, 256, 0, (0, 4))
            embed.finish_load("cpu")
            self.assertFalse(hasattr(embed, "_host_opt"))
            self.assertIsNone(embed.host_table)

    def test_staging_conflict_fails_before_h2_import(self):
        with self.context(True, True, True):
            with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                EngramEmbedding(4, 256, 0, (0, 4))


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA-compatible device")
class TestEngramHostDeviceSafety(CustomTestCase):
    def test_gate_final_cast_matches_original(self):
        from sglang.kernels.ops.embeddings.engram_gate import fused_engram_gate
        from sglang.srt.layers.engram_host import SharedHostEngram

        torch.manual_seed(733427905)
        for dtype in (torch.bfloat16, torch.float32):
            for tokens in (1, 17, 128):
                with self.subTest(dtype=dtype, tokens=tokens):
                    x = torch.randn(tokens, 4, 4096, device="cuda", dtype=dtype)
                    kv = torch.randn(tokens, 5 * 4096, device="cuda", dtype=dtype)
                    q = torch.randn(4, 4096, device="cuda", dtype=dtype)
                    k = torch.randn_like(q)
                    actual = SharedHostEngram.gate(x, kv, q, k, 1e-6, 1e-6)
                    expected = fused_engram_gate(x, kv, q, k, 1e-6, 1e-6)
                    self.assertEqual(actual.dtype, dtype)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def prefetch_fixture(self):
        from sglang.srt.layers.engram_host import SharedHostEngram

        owner = torch.cuda.current_stream()
        side = torch.cuda.Stream()
        opt = SharedHostEngram()

        def allocate(ids):
            self.assertEqual(torch.cuda.current_stream(), owner)
            return torch.empty((ids.numel(), 256), device=ids.device, dtype=torch.bfloat16)

        def gather(embed, ids, out):
            # Real asynchronous device work exercises the allocator independently
            # of Host table setup and the gather kernel's numerical behavior.
            out.copy_(ids[:, None].expand_as(out))

        opt._gather = gather
        return opt, SimpleNamespace(new_rows=allocate), owner, side

    def test_prefetch_consume_survives_allocation_reuse(self):
        opt, embed, owner, side = self.prefetch_fixture()
        observed = []
        for value in range(32):
            ids = torch.full((512,), value, device="cuda", dtype=torch.int64)
            side.wait_stream(owner)
            opt.prefetch(embed, ids, side)
            output = opt.lookup(embed, ids)
            self.assertIsNone(opt._prefetched)
            observed.append(output.clone())
            del output, ids
            # Exercise reuse on the output's allocation/consumer stream while
            # the clone and prefetch can still be outstanding on the device.
            scratch = torch.empty((512, 256), device="cuda", dtype=torch.bfloat16)
            scratch.fill_(-1)
            del scratch
        torch.cuda.synchronize()
        for value, actual in enumerate(observed):
            torch.testing.assert_close(actual, torch.full_like(actual, value), rtol=0, atol=0)

    def test_discarded_prefetch_preserves_live_allocations(self):
        opt, embed, owner, side = self.prefetch_fixture()
        live = []
        for value in range(32):
            ids = torch.full((512,), value, device="cuda", dtype=torch.int64)
            side.wait_stream(owner)
            opt.prefetch(embed, ids, side)
            # Drop the result without waiting/consuming, as a discarded batch
            # would do. record_stream must protect both ids and output storage.
            opt._prefetched = None
            del ids
            replacement_ids = torch.full((512,), -7, device="cuda", dtype=torch.int64)
            replacement_out = torch.full((512, 256), -3, device="cuda", dtype=torch.bfloat16)
            live.append((replacement_ids, replacement_out))
        torch.cuda.synchronize()
        for ids, output in live:
            torch.testing.assert_close(ids, torch.full_like(ids, -7), rtol=0, atol=0)
            torch.testing.assert_close(output, torch.full_like(output, -3), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
