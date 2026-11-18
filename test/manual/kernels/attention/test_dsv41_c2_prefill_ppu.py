"""Validate fused C2 prefill on PPU against the production torch path."""

import unittest

import torch

from sglang.kernels.ops.attention.dsv4 import fused_store_cache
from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
from sglang.kernels.ops.attention.dsv4.low_ratio_compress import (
    c2_prefill_norm_rope_store,
)
from sglang.kernels.ops.layernorm.rmsnorm_fp32 import rmsnorm_fp32
from sglang.srt.layers.attention.dsv4.dsv41_sparse import _rope_fq4
from sglang.test.test_utils import CustomTestCase


@torch.no_grad()
def check(layout, prefixes, chunks, ring, pos_dtype, seed):
    torch.manual_seed(seed)
    d, rd, ps = 512, 64, 128
    nr = len(prefixes)
    last = max(p + sum(c[r] for c in chunks) for r, p in enumerate(prefixes))
    angles = torch.randn(last + 2, rd // 2, device="cuda")
    freqs = torch.polar(torch.ones_like(angles), angles)
    real_freqs = torch.view_as_real(freqs).flatten(-2)
    weight = (1 + torch.randn(d, device="cuda") * 0.2).bfloat16()
    original = torch.randn((nr + 1) * ring + 1, 2 * d, device="cuda")
    original[-1, :d] = 0
    original[-1, d:] = -torch.inf
    actual = original.clone()
    max_slot = (nr * 300000 + last + 64) // 2 + 1
    npages = (max_slot + ps - 1) // ps
    cache_ref = torch.zeros(
        npages, layout.page_bytes(ps), device="cuda", dtype=torch.uint8
    )
    cache_got = torch.zeros_like(cache_ref)
    offsets = list(prefixes)
    results = []
    for lengths in chunks:
        reqs, positions, locs = [], [], []
        starts, at = [], 0
        for r, length in enumerate(lengths):
            starts.append(at)
            reqs.extend([r + 1] * length)
            positions.extend(range(offsets[r], offsets[r] + length))
            locs.extend(
                (r + 1) * 300000 + 64 + p
                for p in range(offsets[r], offsets[r] + length)
            )
            offsets[r] += length
            at += length
        pad = 3
        req = torch.tensor(reqs + [0] * pad, device="cuda", dtype=torch.int64)
        pos = torch.tensor(positions + [0] * pad, device="cuda", dtype=pos_dtype)
        loc = torch.tensor(locs + [0] * pad, device="cuda", dtype=torch.int64)
        x = torch.randn(at + pad, 2 * d, device="cuda", dtype=torch.float32)
        output = torch.full(
            (at + pad, d), float("nan"), device="cuda", dtype=torch.bfloat16
        )
        c2_prefill_norm_rope_store(
            x,
            actual,
            weight,
            pos,
            req,
            loc,
            1e-6,
            real_freqs,
            cache_got,
            torch.tensor(starts, device="cuda", dtype=torch.int32),
            torch.tensor(lengths, device="cuda", dtype=torch.int32),
            page_size=ps,
            ring_size=ring,
            layout=layout,
            out=output,
        )
        # Snapshot reads precede every chronological ring write in the reference.
        read_pos = (pos.long() - 1).masked_fill(loc == 0, -1)
        slots = (req * ring + read_pos % ring).masked_fill(
            read_pos < 0, original.shape[0] - 1
        )
        partner = original[slots]
        adjacent = torch.zeros(at + pad, device="cuda", dtype=torch.bool)
        adjacent[1:] = (
            (req[1:] == req[:-1])
            & (pos[1:] == pos[:-1] + 1)
            & (loc[1:] != 0)
            & (loc[:-1] != 0)
        )
        partner = torch.where(adjacent[:, None], torch.roll(x, 1, 0), partner)
        kv = torch.stack((partner[:, :d], x[:, :d]), dim=1)
        scores = torch.stack((partner[:, d:], x[:, d:]), dim=1)
        pooled = (kv * scores.softmax(dim=1)).sum(dim=1)
        latent = rmsnorm_fp32(pooled.bfloat16(), weight, 1e-6)
        valid = (loc > 0) & ((pos % 2) == 1)
        values = latent[valid].contiguous()
        selected_freqs = freqs[pos[valid].long() - 1].contiguous()
        if layout.is_fp4:
            fused_store_cache(
                input=values,
                cache=cache_ref,
                indices=loc[valid] // 2,
                page_size=ps,
                type="flashmla",
                layout=layout,
                freqs_cis=selected_freqs,
            )
        else:
            values = _rope_fq4(values, selected_freqs, rd, compressed_kv=True)
            fused_store_cache(
                input=values,
                cache=cache_ref,
                indices=loc[valid] // 2,
                page_size=ps,
                type="flashmla",
                layout=layout,
            )
        for r, (start, length) in enumerate(zip(starts, lengths)):
            for row in range(start + max(0, length - ring), start + length):
                original[(r + 1) * ring + int(pos[row]) % ring] = x[row]
        torch.cuda.synchronize()
        assert torch.equal(original, actual), "ring state differs"
        assert torch.isnan(output[~valid]).all(), (
            "uncompleted/padded output overwritten"
        )
        diff = (output[valid] != latent[valid]).sum().item()
        cache_diff = (cache_ref != cache_got).sum().item()
        assert cache_diff == 0, "main-KV cache bytes differ"
        # The fused analytic pair pooling can differ from torch softmax by
        # up to two BF16 ULPs, as in the existing fused-compressor UT.
        torch.testing.assert_close(
            output[valid].float(), latent[valid].float(), rtol=2**-6, atol=2**-20
        )
        maximum = (
            (output[valid].float() - latent[valid].float()).abs().max().item()
            if valid.any()
            else 0
        )
        results.append(
            {
                "rows": at,
                "latent_mismatches": diff,
                "cache_byte_mismatches": cache_diff,
                "latent_max_abs": maximum,
                "ring_exact": True,
            }
        )
    return {
        "layout": layout.value,
        "ring": ring,
        "positions": str(pos_dtype),
        "results": results,
    }


class TestDSV41C2PrefillPPU(CustomTestCase):
    def test_chunk_boundaries_and_padding(self):
        for layout in (KVLayout.V41_NEW, KVLayout.V41_FP4_NEW):
            for dtype in (torch.int32, torch.int64):
                for ring in (2, 8):
                    with self.subTest(layout=layout.name, dtype=dtype, ring=ring):
                        check(
                            layout,
                            [0, 7, 1],
                            [[5, 4, 1], [3, 6, 2], [1, 1, 1]],
                            ring,
                            dtype,
                            7,
                        )

    def test_model_shapes(self):
        for layout in (KVLayout.V41_NEW, KVLayout.V41_FP4_NEW):
            for n in (512, 4096, 8192, 16384):
                with self.subTest(layout=layout.name, tokens=n):
                    check(layout, [126976] * 4, [[n // 4] * 4], 8, torch.int64, n)


if __name__ == "__main__":
    unittest.main()
