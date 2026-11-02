# SPDX-License-Identifier: Apache-2.0
"""Precision tests for the Step5 Q/K-norm + RoPE Triton kernel on PPU.

``qknorm_rope.py`` dispatches the fused port to CuTe on Hopper/Blackwell and to
``qknorm_rope_triton.py`` on everything else, so PPU always runs the Triton half.
These cases pin the contract that port has to keep, each against a plain torch
reference derived from the documented semantics (not from the kernel body):

* norm precision -- GemmaRMSNorm (FP32 ``weight + 1``) materializes BF16 before
  RoPE; a full-FP32 chain reorders the rounding and must stay within one BF16
  ulp of the documented boundary.
* half-split pairing -- rotated channel ``i`` pairs with ``i + rd`` inside
  ``[0, 2 * rd)`` and takes ``cos[i]`` from the merged table, so a GPT-J-style
  interleave or a second-half table read is visible.
* tail passthrough -- channels in ``[2 * rd, 192)`` keep the *normed* value, for
  both the 64-wide (SWA, partial rotary) and 192-wide (DSA, full rotary) table.
* fused SWA store -- rows written through ``kv_write`` are bit-identical to the
  returned K/V, ``loc < 0`` leaves the slot untouched, and the row view the
  backend hands over lands on the slot the paged reader indexes.
* head counts -- 4/8/16 are the TP16/TP8/TP4 geometries; 4 is Triton-only, so
  the packed-width guard must accept it instead of raising.

Run standalone on one idle PPU device::

    python test/ppu/unit/test_step5_qknorm_triton.py
"""

import unittest

import torch

from sglang.kernels.ops.attention.step5.qknorm_rope import (
    step5_qknorm_rope,
    step5_qknorm_rope_gated,
    supports_fused_qknorm_rope,
)

DEV = "cuda"
HEAD_DIM = 192
EPS = 1e-5
# Two BF16 ulps at unit magnitude: both chains round the same FP32 math, but
# torch's ``mean`` and the kernel's tree reduction differ in FP32 order.
TOL = 2.0**-6


def _rotary_cache(positions_cap: int, width: int) -> torch.Tensor:
    """Merged ``[cos | sin]`` table of width ``width`` (64 or 192)."""
    half = width // 2
    exponents = torch.arange(half, dtype=torch.float32) * 2 / width
    inv_freq = 1.0 / 10000.0**exponents
    angles = torch.arange(positions_cap, dtype=torch.float32)[:, None] * inv_freq
    merged = torch.cat((angles.cos(), angles.sin()), dim=-1)
    return merged.to(device=DEV, dtype=torch.bfloat16).contiguous()


def _reference(qkv, q_weight, k_weight, positions, cache, heads):
    """Documented math: FP32 Gemma norm -> BF16 -> FP32 half-split RoPE -> BF16."""
    tokens = qkv.shape[0]
    packed = qkv.view(tokens, heads + 2, HEAD_DIM)

    def norm(x, weight):
        xf = x.float()
        variance = xf.pow(2).mean(dim=-1, keepdim=True)
        return (xf * torch.rsqrt(variance + EPS) * (1.0 + weight.float())).to(
            torch.bfloat16
        )

    half = cache.shape[1] // 2
    cos, sin = cache.index_select(0, positions).split(half, dim=-1)

    def rope(x):
        rot = x[..., : 2 * half].float()
        x1, x2 = rot.chunk(2, dim=-1)
        c = cos.unsqueeze(1)
        s = sin.unsqueeze(1)
        roped = torch.cat((x1 * c - x2 * s, x2 * c + x1 * s), dim=-1)
        return torch.cat((roped, x[..., 2 * half :].float()), dim=-1).to(
            torch.bfloat16
        )

    q = rope(norm(packed[:, :heads, :], q_weight))
    k = rope(norm(packed[:, heads : heads + 1, :], k_weight))
    v = packed[:, heads + 1 :, :].contiguous()
    return q, k, v


def _make_case(heads, rotary_width, tokens=37, seed=0):
    generator = torch.Generator(device=DEV).manual_seed(seed)
    positions = torch.randint(
        0, 512, (tokens,), dtype=torch.int64, device=DEV, generator=generator
    )
    cache = _rotary_cache(512, rotary_width)
    qkv = torch.randn(
        tokens,
        (heads + 2) * HEAD_DIM,
        dtype=torch.bfloat16,
        device=DEV,
        generator=generator,
    )
    # Zero-centered Gemma weights sit near 0, so scale them like a trained model.
    q_weight = (0.1 * torch.randn(HEAD_DIM, device=DEV, generator=generator)).float()
    k_weight = (0.1 * torch.randn(HEAD_DIM, device=DEV, generator=generator)).float()
    return qkv, q_weight, k_weight, positions, cache


class TestStep5QKNormTriton(unittest.TestCase):
    def test_fused_port_serves_this_device(self):
        """PPU must take the fused port; the Triton-only heads=4 case proves it."""
        self.assertTrue(supports_fused_qknorm_rope(torch.device(DEV)))
        qkv, qw, kw, positions, cache = _make_case(heads=4, rotary_width=192)
        q = step5_qknorm_rope(qkv, qw, kw, positions, cache, eps=EPS)[0]
        self.assertEqual(q.shape, (qkv.shape[0], 4, HEAD_DIM))

    def test_matches_reference(self):
        for heads in (4, 8, 16):
            for rotary_width in (64, 192):
                with self.subTest(heads=heads, rotary_width=rotary_width):
                    qkv, qw, kw, positions, cache = _make_case(
                        heads, rotary_width, seed=heads * 100 + rotary_width
                    )
                    q, k, v = step5_qknorm_rope(qkv, qw, kw, positions, cache, eps=EPS)
                    ref_q, ref_k, ref_v = _reference(
                        qkv, qw, kw, positions, cache, heads
                    )
                    torch.testing.assert_close(q, ref_q, atol=TOL, rtol=TOL)
                    torch.testing.assert_close(k, ref_k, atol=TOL, rtol=TOL)
                    self.assertTrue(bool((v == ref_v).all()), "V is not a plain copy")

    def test_rotated_tail_keeps_normed_value(self):
        """The partial-rotary layer must norm channels past the window, not copy."""
        qkv, qw, kw, positions, cache = _make_case(heads=8, rotary_width=64)
        q = step5_qknorm_rope(qkv, qw, kw, positions, cache, eps=EPS)[0]
        packed = qkv.view(qkv.shape[0], 10, HEAD_DIM).float()
        rows = packed[:, :8, :]
        # The norm spans the whole 192-wide head; only the first 64 rotate.
        variance = rows.pow(2).mean(dim=-1, keepdim=True)
        expected = (rows * torch.rsqrt(variance + EPS) * (1.0 + qw)).to(torch.bfloat16)
        torch.testing.assert_close(q[..., 64:], expected[..., 64:], atol=TOL, rtol=TOL)
        self.assertTrue(
            bool((q[..., 64:] != packed[:, :8, 64:].to(torch.bfloat16)).any()),
            "tail channels were passed through instead of normed",
        )
        self.assertTrue(
            bool((q[..., :64] != expected[..., :64]).any()),
            "head window was not rotated",
        )

    def test_fused_store_is_bit_exact(self):
        qkv, qw, kw, positions, cache = _make_case(heads=8, rotary_width=64)
        tokens = qkv.shape[0]
        slots = 64
        k_cache = torch.randn(slots, HEAD_DIM, dtype=torch.bfloat16, device=DEV)
        v_cache = torch.randn(slots, HEAD_DIM, dtype=torch.bfloat16, device=DEV)
        # Half the rows are skipped so both the store and the skip are covered.
        loc = torch.arange(tokens, dtype=torch.int64, device=DEV) % slots
        loc[1::2] = -1
        before_k, before_v = k_cache.clone(), v_cache.clone()
        written = loc >= 0
        _, k, v = step5_qknorm_rope(
            qkv, qw, kw, positions, cache, eps=EPS, kv_write=(k_cache, v_cache, loc)
        )
        torch.cuda.synchronize()
        self.assertTrue(
            bool((k_cache[loc[written]] == k[written, 0]).all()),
            "stored K differs from the returned K",
        )
        self.assertTrue(
            bool((v_cache[loc[written]] == v[written, 0]).all()),
            "stored V differs from the returned V",
        )
        untouched = torch.ones(slots, dtype=torch.bool, device=DEV)
        untouched[loc[written]] = False
        self.assertTrue(
            bool((k_cache[untouched] == before_k[untouched]).all())
            and bool((v_cache[untouched] == before_v[untouched]).all()),
            "skipped rows were written",
        )

    def test_store_row_matches_paged_slot(self):
        """Row ``slot`` of ``k_buffer.view(-1, 192)`` is what the paged reader
        indexes as ``[page, :, offset]``, so the fused store and ``set_kv_buffer``
        write the same bytes for both the NHD and the HND pool layout."""
        page_size, pages = 4, 16
        k_buffer = torch.randn(
            pages, 1, page_size, HEAD_DIM, dtype=torch.bfloat16, device=DEV
        )
        v_buffer = torch.randn_like(k_buffer)
        qkv, qw, kw, positions, cache = _make_case(heads=4, rotary_width=192)
        loc = torch.randperm(pages * page_size, device=DEV)[: qkv.shape[0]]
        _, k, v = step5_qknorm_rope(
            qkv,
            qw,
            kw,
            positions,
            cache,
            eps=EPS,
            kv_write=(
                k_buffer.view(-1, HEAD_DIM),
                v_buffer.view(-1, HEAD_DIM),
                loc,
            ),
        )
        torch.cuda.synchronize()
        page_of, offset_of = loc // page_size, loc % page_size
        self.assertTrue(
            bool((k_buffer[page_of, 0, offset_of] == k[:, 0]).all()),
            "stored K missed the slot the paged reader reads",
        )
        self.assertTrue(
            bool((v_buffer[page_of, 0, offset_of] == v[:, 0]).all()),
            "stored V missed the slot the paged reader reads",
        )

    def test_gated_variant_returns_gate_copy(self):
        """heads=4 is Triton-only, so the gated width 1156 must be accepted too."""
        heads = 4
        qkv, qw, kw, positions, cache = _make_case(heads=heads, rotary_width=192)
        gate = torch.randn(qkv.shape[0], heads, dtype=torch.bfloat16, device=DEV)
        qkv_gate = torch.cat((qkv, gate), dim=-1).contiguous()
        q, k, v, out_gate = step5_qknorm_rope_gated(
            qkv_gate, qw, kw, positions, cache, eps=EPS
        )
        ref_q, ref_k, ref_v = _reference(qkv, qw, kw, positions, cache, heads)
        torch.testing.assert_close(q, ref_q, atol=TOL, rtol=TOL)
        torch.testing.assert_close(k, ref_k, atol=TOL, rtol=TOL)
        self.assertTrue(bool((v == ref_v).all()), "V is not a plain copy")
        self.assertTrue(bool((out_gate == gate).all()), "gate is not a plain copy")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "needs a CUDA/PPU device"
    unittest.main(verbosity=2)
