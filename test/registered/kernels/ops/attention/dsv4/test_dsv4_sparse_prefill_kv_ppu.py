"""PPU BF16 sparse Prefill equivalence and retained workspace regression."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.test_utils import CustomTestCase

register_ppu_ci(est_time=30, suite="stage-a-test-1-gpu-ppu")


class TestSparsePrefillKVPPU(CustomTestCase):
    def test_compaction_and_bf16_workspace_reuse(self):
        from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
        from sglang.srt.layers.attention import deepseek_v4_backend as backend
        from sglang.srt.layers.attention.dsv4 import sparse_prefill_utils as helper
        from sglang.srt.plugins import load_plugins

        load_plugins()
        self.assertTrue(backend.get_platform().is_ppu)

        device = torch.device("cuda:0")
        page_size = 256
        layout = KVLayout.V41_NEW
        queries = 16
        comp_rows, swa_rows = 2048, 2048
        col = torch.arange(512, device=device, dtype=torch.int32)[None, :]
        row = torch.arange(page_size, device=device, dtype=torch.int32)[:, None]

        def fp8_page(seed):
            page = torch.empty(
                (1, layout.page_bytes(page_size)), dtype=torch.uint8, device=device
            )
            entries = page.view(page_size, layout.bytes_per_token)
            entries[:, :512] = ((row * 13 + col * 7 + seed) % 17 + 38).to(torch.uint8)
            entries[:, 512:].fill_(127)
            return page

        comp, swa = fp8_page(0), fp8_page(11)
        pool = SimpleNamespace(
            get_extra_key_page_size=lambda _: page_size,
            get_extra_key_buffer=lambda _: comp,
            get_extra_key_layout=lambda _: layout,
            get_swa_key_buffer_radix=lambda _: swa,
            get_swa_key_layout=lambda: layout,
        )
        rng = torch.Generator(device=device).manual_seed(31415)
        q = (
            torch.randn(
                (queries, 1, 64, 512),
                dtype=torch.bfloat16,
                device=device,
                generator=rng,
            )
            * 0.1
        )
        sink = torch.zeros(64, device=device, dtype=torch.float32)

        # A duplicate in each region, an invalid index inside topk_length, and
        # a valid-looking index beyond topk_length that must be ignored.
        qrow = torch.arange(queries, device=device, dtype=torch.int32)
        lhs, rhs = (qrow * 97 + 155) % comp_rows, comp_rows + (qrow * 193) % swa_rows
        indices = torch.full((queries, 128), -1, device=device, dtype=torch.int32)
        indices[:, 0] = indices[:, 2] = lhs
        indices[:, 1] = indices[:, 4] = rhs
        indices[:, 5] = 4
        lengths = torch.full((queries,), 5, device=device, dtype=torch.int32)
        slots = torch.arange(comp_rows, device=device, dtype=torch.int32).remainder_(
            page_size
        )
        swa_slots = torch.arange(swa_rows, device=device, dtype=torch.int32).remainder_(
            page_size
        )
        cache = SimpleNamespace(
            swa_token_ids=swa_slots,
            swa_page_size=page_size,
            layer_inputs=lambda *_: (slots, indices, lengths),
        )

        def run(buffer):
            self = SimpleNamespace(
                forward_metadata=SimpleNamespace(sparse_prefill_cache=cache),
                sparse_prefill_kv_buffer=buffer,
                dsv4_prefill_backend="flashmla_sparse",
                head_dim_v=512,
                softmax_scale=512**-0.5,
            )
            value = backend.DeepseekV4AttnBackend._forward_prefill_sparse(
                self, q, 0, 4, None, pool, None, sink
            )
            torch.cuda.synchronize()
            return value

        def check(actual, ref):
            assert torch.isfinite(actual).all().item()
            torch.testing.assert_close(actual, ref, atol=0, rtol=0)

        spy_count = []
        real_compact = helper.compact_sparse_prefill_kv

        def record_compact(*values):
            workspace, remapped = real_compact(*values)
            assert workspace.shape[0] == 2 * queries
            assert torch.equal(remapped[:, 0], remapped[:, 2])
            assert torch.equal(remapped[:, 1], remapped[:, 4])
            assert (remapped[:, (3, 5)] == -1).all().item()
            spy_count.append(workspace.shape[0])
            return workspace, remapped

        retained = helper.SparsePrefillKVBuffer(device)
        empty = helper.SparsePrefillKVBuffer(device)
        undersized = helper.SparsePrefillKVBuffer(device)
        undersized.get(comp_rows)
        pinned = helper.SparsePrefillKVBuffer(device)
        pinned_addr = pinned.get(comp_rows, retain_for_graph=True).data_ptr()
        with patch.object(
            backend, "compact_sparse_prefill_kv", side_effect=record_compact
        ):
            with patch.object(
                backend, "should_compact_sparse_prefill_kv", return_value=False
            ) as gate:
                ref = run(retained)
                gate.assert_called_once_with(comp_rows + swa_rows, queries * 128)
            assert torch.isfinite(ref).all().item() and ref.abs().max().item() > 0
            assert retained.can_reuse_bf16(comp_rows + swa_rows)
            address = retained._buffer.data_ptr()
            with patch.object(
                backend, "should_compact_sparse_prefill_kv", return_value=True
            ) as gate:
                check(run(empty), ref)
                assert empty._buffer is None and spy_count == [2 * queries]
                check(run(retained), ref)
                assert retained._buffer.data_ptr() == address and spy_count == [
                    2 * queries
                ]
                check(run(undersized), ref)
                assert undersized._buffer is None and spy_count == [2 * queries] * 2
                check(run(pinned), ref)
                assert pinned._buffer.data_ptr() == pinned_addr
                assert spy_count == [2 * queries] * 3
                assert gate.call_count == 4


if __name__ == "__main__":
    unittest.main(verbosity=2)
