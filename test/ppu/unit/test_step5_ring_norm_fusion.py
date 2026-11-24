# SPDX-License-Identifier: Apache-2.0
"""Byte-equality contract behind the two Step5 single-pass fusions.

Both fusions replace "write then re-read" with "write once, in the same
kernel", so their only acceptable failure mode is a numerical difference.
These cases pin that difference to zero, each against the split form it
replaced:

* ring append -- ``_compress`` can publish the ring in the same launch
  (``FUSE_WRITE``). The fused and split launches must leave the paged keys,
  both rings, the history tags and the success flags bit-identical, including
  a region that mixes history with pending tokens.
* indexer norm -- ``_indexer_norm_rope`` can emit Q as E4M3 bytes directly
  (``WRITE_Q_E4M3``) instead of BF16 plus a Torch ``to(torch.float8_e4m3fn)``
  cast. Ports without a native ``fp8e4nv`` dtype (sm80 PPU) encode with the
  integer encoder, so the bytes must still equal the Torch cast exactly.
  The packed decode layout gets the same guarantee at its mapped slots.

Run standalone on one idle PPU device::

    python test/ppu/unit/test_step5_ring_norm_fusion.py
"""

import types
import unittest
from unittest import mock

import torch

from sglang.kernels.ops.attention.step5 import csa_ring
from sglang.kernels.ops.attention.step5.csa_ring import (
    AppendRingPlan,
    append_ring,
    supports_fused_ring_append,
)
from sglang.kernels.ops.attention.step5.indexer_norm import (
    step5_indexer_norm_rope,
    supports_fused_fp8_cast,
)

DEV = "cuda"
ROWS = 8
CAPACITY = 16
EPS = 1e-6


def _ring_buffers(base, length):
    """Plan inputs plus the buffers the append reads and writes."""
    tokens = ROWS * length
    torch.manual_seed(1234)  # both launch forms must see identical bytes
    slots = torch.arange(ROWS, dtype=torch.int64, device=DEV)
    tags = torch.full((ROWS, CAPACITY), -1, dtype=torch.int32, device=DEV)
    for slot in range(ROWS):
        for cell in range(CAPACITY):
            # Cells below the base carry their own position; the rest are invalid.
            tags[slot, cell] = cell if cell < base else -1
    return dict(
        slots=slots,
        bases=torch.full((ROWS,), base, dtype=torch.int64, device=DEV),
        sources=torch.arange(0, tokens, length, dtype=torch.int64, device=DEV),
        lengths=torch.full((ROWS,), length, dtype=torch.int64, device=DEV),
        active=torch.ones(ROWS, dtype=torch.bool, device=DEV),
        admitted=torch.ones(ROWS, dtype=torch.bool, device=DEV),
        count_valid=torch.ones(1, dtype=torch.bool, device=DEV),
        physical_regions=(torch.arange(ROWS, device=DEV) % 8).to(torch.int64)[None],
        enabled=torch.zeros(ROWS, dtype=torch.bool, device=DEV),
        success=torch.zeros(ROWS, dtype=torch.bool, device=DEV),
        index_key=torch.randn(tokens, 1, 256, dtype=torch.bfloat16, device=DEV),
        compression_logits=torch.randn(
            tokens, 1, 256, dtype=torch.bfloat16, device=DEV
        ),
        index_key_ring=torch.randn(
            ROWS, CAPACITY, 256, dtype=torch.bfloat16, device=DEV
        ),
        logits_ring=torch.randn(ROWS, CAPACITY, 256, dtype=torch.bfloat16, device=DEV),
        paged=torch.zeros(8 * 2048, dtype=torch.uint8, device=DEV),
        tags=tags,
    )


def _run_append(buffers, fused):
    plan = AppendRingPlan(
        slots=buffers["slots"],
        bases=buffers["bases"],
        sources=buffers["sources"],
        lengths=buffers["lengths"],
        active=buffers["active"],
        admitted=buffers["admitted"],
        count_valid=buffers["count_valid"],
        physical_regions=buffers["physical_regions"],
        max_regions=buffers["physical_regions"].shape[0],
        enabled=buffers["enabled"],
        success=buffers["success"],
    )
    with mock.patch.object(csa_ring, "supports_fused_ring_append", return_value=fused):
        append_ring(
            plan=plan,
            index_key=buffers["index_key"],
            compression_logits=buffers["compression_logits"],
            index_key_ring=buffers["index_key_ring"],
            compression_logits_ring=buffers["logits_ring"],
            positions=buffers["tags"],
            compressed_index_keys=buffers["paged"],
        )
    return buffers


def _as_bytes(tensor):
    return tensor.view(torch.uint8) if tensor.dtype != torch.uint8 else tensor


def _norm_inputs(tokens, heads):
    torch.manual_seed(0)
    positions = 64
    exponents = torch.arange(16, dtype=torch.float32, device=DEV) / 16
    angles = torch.arange(positions, dtype=torch.float32, device=DEV)[
        :, None
    ] * (1.0 / 10000.0**exponents)[None]
    cache = torch.cat((angles.cos(), angles.sin()), dim=-1).to(torch.bfloat16)
    return dict(
        q=torch.randn(tokens, heads, 256, dtype=torch.bfloat16, device=DEV),
        k=torch.randn(tokens, 1, 256, dtype=torch.bfloat16, device=DEV),
        qw=torch.randn(256, dtype=torch.bfloat16, device=DEV),
        kw=torch.randn(256, dtype=torch.bfloat16, device=DEV),
        kb=torch.randn(256, dtype=torch.bfloat16, device=DEV),
        cache=cache,
        positions=torch.randint(0, positions, (tokens,), device=DEV),
    )


def _norm_call(inputs, q_out, k_out, **kwargs):
    step5_indexer_norm_rope(
        inputs["q"],
        inputs["k"],
        q_weight=inputs["qw"],
        k_weight=inputs["kw"],
        k_bias=inputs["kb"],
        cache=inputs["cache"],
        positions=inputs["positions"],
        q_out=q_out,
        k_out=k_out,
        eps=EPS,
        zero_centered=True,
        fp8_saturate=True,
        **kwargs,
    )


class TestStep5RingNormFusion(unittest.TestCase):
    def test_fused_ports_serve_this_device(self):
        device = torch.device(DEV)
        self.assertTrue(supports_fused_ring_append(device))
        self.assertTrue(supports_fused_fp8_cast(device))

    def test_ring_append_fused_matches_split(self):
        for base, length in ((8, 8), (12, 8)):
            with self.subTest(base=base, length=length):
                fused = _run_append(_ring_buffers(base, length), True)
                split = _run_append(_ring_buffers(base, length), False)
                torch.cuda.synchronize()
                self.assertEqual(int(fused["enabled"].sum()), ROWS)
                for key in (
                    "paged",
                    "index_key_ring",
                    "logits_ring",
                    "tags",
                    "enabled",
                    "success",
                ):
                    torch.testing.assert_close(
                        _as_bytes(fused[key]), _as_bytes(split[key])
                    )

    def test_norm_fp8_bytes_match_torch_cast(self):
        for tokens, heads in ((16, 4), (5, 2), (1, 4)):
            with self.subTest(tokens=tokens, heads=heads):
                inputs = _norm_inputs(tokens, heads)
                bf16_q = torch.zeros(
                    tokens, heads, 256, dtype=torch.bfloat16, device=DEV
                )
                bf16_k = torch.zeros(tokens, 1, 256, dtype=torch.bfloat16, device=DEV)
                _norm_call(inputs, bf16_q, bf16_k)
                fused_q = torch.zeros(
                    tokens, heads, 256, dtype=torch.float8_e4m3fn, device=DEV
                )
                fused_k = torch.zeros(tokens, 1, 256, dtype=torch.bfloat16, device=DEV)
                _norm_call(inputs, fused_q, fused_k)
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    fused_q.view(torch.uint8),
                    bf16_q.to(torch.float8_e4m3fn).view(torch.uint8),
                )
                torch.testing.assert_close(fused_k, bf16_k)

    def test_packed_layout_matches_unpacked_bytes(self):
        tokens, heads = 8, 4
        inputs = _norm_inputs(tokens, heads)
        reference = torch.zeros(
            tokens, heads, 256, dtype=torch.float8_e4m3fn, device=DEV
        )
        empty_k = torch.zeros(tokens, 1, 256, dtype=torch.bfloat16, device=DEV)
        _norm_call(inputs, reference, empty_k)
        q_runtime = torch.full(
            (tokens * 16, 256), 0x7F, dtype=torch.float8_e4m3fn, device=DEV
        )
        workspace = types.SimpleNamespace(
            rows_per_group=1,
            q_runtime=q_runtime,
            kernel_weights=torch.zeros(tokens, 32, 4, dtype=torch.float32, device=DEV),
        )
        _norm_call(
            inputs,
            None,
            empty_k,
            packed_workspace=workspace,
            scoring_weights=torch.ones(tokens, heads, dtype=torch.bfloat16, device=DEV),
            row_to_req=torch.ones(tokens, dtype=torch.int32, device=DEV),
            weight_scale=1.0,
            packed_q_e4m3=True,
        )
        torch.cuda.synchronize()
        for token in range(tokens):
            for head in range(heads):
                # GROUP=1, slot=0: packed_h = token*16 + (head//2)*8 + head%2.
                packed_h = token * 16 + (head // 2) * 8 + head % 2
                torch.testing.assert_close(
                    q_runtime[packed_h].view(torch.uint8),
                    reference[token, head].view(torch.uint8),
                )


if __name__ == "__main__":
    unittest.main()
