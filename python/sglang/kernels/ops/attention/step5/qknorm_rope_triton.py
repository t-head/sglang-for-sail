# SPDX-License-Identifier: Apache-2.0
"""Step5 Q/K norm + half-split RoPE in portable Triton (CuTe replacement).

Mirrors ``cute/qknorm_rope.py`` ``FusedQKNormRope`` for devices that cannot run
the CuTe DSL kernel (e.g. sm80 PPU). Each program normalizes one 192-wide head
row with the zero-centered GemmaRMSNorm (``weight + 1``), materializes BF16 --
matching the CuTe smem round -- then applies the half-split NeoX RoPE straight
from the model's merged ``[cos | sin]`` table.

Q and K are two separate launches so the norm weight tensor and the output
offsets stay compile-time constants. V and the head gate are plain copies, and
the optional SWA paged KV-cache scatter is fused into the K launch (``loc < 0``
rows are skipped). RoPE and tail columns are assembled in registers and written
with a single masked store per row, so the rotated ``[0, 2*rd)`` segment and the
passthrough ``[2*rd, 192)`` tail never race on overlapping addresses.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _norm_rope_row(
    X,
    W,
    COS,
    SIN,
    row_off,
    pos,
    RD: tl.constexpr,
    EPS: tl.constexpr,
):
    """RMSNorm(192, zero-centered) -> BF16 -> half-split RoPE. Returns BF16[256].

    ``row_off`` is the element offset of this head's 192-wide row inside ``X``;
    ``pos`` is the token's rotary table row. ``COS``/``SIN`` are the split views
    of the merged table, so each has a row stride of ``2 * RD``.
    """
    channel = tl.arange(0, 256)
    mask = channel < 192
    x = tl.load(X + row_off + channel, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + channel, mask=mask, other=0.0)
    inverse_rms = tl.rsqrt(tl.sum(x * x, 0) / 192 + EPS)
    # Norm has its own BF16 boundary before RoPE, matching the CuTe smem round.
    norm = (x * inverse_rms * (w + 1.0)).to(tl.bfloat16).to(tl.float32)
    # Half-split partner: [0,rd) <-> [rd,2rd); the tail partners with itself and
    # is discarded by the final select.
    rot = channel < 2 * RD
    partner = tl.where(rot, tl.where(channel < RD, channel + RD, channel - RD), channel)
    cos_idx = tl.where(rot, tl.where(channel < RD, channel, channel - RD), 0)
    pair = tl.gather(norm, partner, axis=0)
    cos = tl.load(COS + pos * (2 * RD) + cos_idx, mask=mask, other=0.0).to(tl.float32)
    sin = tl.load(SIN + pos * (2 * RD) + cos_idx, mask=mask, other=0.0).to(tl.float32)
    sign = tl.where(channel < RD, -1.0, 1.0)
    roped = norm * cos + sign * pair * sin
    return tl.where(rot, roped, norm).to(tl.bfloat16)


@triton.jit
def _qknorm_rope_q(
    QKV,
    QW,
    COS,
    SIN,
    POS,
    Q,
    ROW_WIDTH: tl.constexpr,
    HEADS: tl.constexpr,
    RD: tl.constexpr,
    EPS: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    pos = tl.load(POS + token).to(tl.int64)
    out = _norm_rope_row(
        QKV, QW, COS, SIN, token * ROW_WIDTH + head * 192, pos, RD, EPS
    )
    channel = tl.arange(0, 256)
    q_off = token * (HEADS * 192) + head * 192
    tl.store(Q + q_off + channel, out, mask=channel < 192)


@triton.jit
def _qknorm_rope_kv(
    QKV,
    KW,
    COS,
    SIN,
    POS,
    K,
    V,
    GATE,
    KCACHE,
    VCACHE,
    LOC,
    ROW_WIDTH: tl.constexpr,
    HEADS: tl.constexpr,
    RD: tl.constexpr,
    EPS: tl.constexpr,
    GATED: tl.constexpr,
    STORE_CACHE: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    channel = tl.arange(0, 256)
    mask = channel < 192
    pos = tl.load(POS + token).to(tl.int64)
    # K head: norm + RoPE, written to the [T,192] K output and (optionally) cache.
    k_out = _norm_rope_row(
        QKV, KW, COS, SIN, token * ROW_WIDTH + HEADS * 192, pos, RD, EPS
    )
    tl.store(K + token * 192 + channel, k_out, mask=mask)
    # V head: plain copy, no norm and no RoPE.
    v_raw = tl.load(
        QKV + token * ROW_WIDTH + (HEADS + 1) * 192 + channel, mask=mask, other=0.0
    )
    tl.store(V + token * 192 + channel, v_raw, mask=mask)
    if GATED:
        g = tl.arange(0, 16)
        gmask = g < HEADS
        gate_val = tl.load(
            QKV + token * ROW_WIDTH + (HEADS + 2) * 192 + g, mask=gmask, other=0.0
        )
        tl.store(GATE + token * HEADS + g, gate_val, mask=gmask)
    if STORE_CACHE:
        loc = tl.load(LOC + token).to(tl.int64)
        if loc >= 0:
            tl.store(KCACHE + loc * 192 + channel, k_out, mask=mask)
            tl.store(VCACHE + loc * 192 + channel, v_raw, mask=mask)


def launch(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate,
    k_cache,
    v_cache,
    loc,
    *,
    heads: int,
    gated: bool,
    store_cache: bool,
    eps: float,
) -> None:
    """Run the Q launch then the fused K/V/gate/cache launch, in place.

    Argument order mirrors the CuTe compiled callable so ``_qknorm_rope`` can
    pick a backend without reshaping its call site. ``cos``/``sin`` are the
    ``rotary_cache.split(width // 2, -1)`` views (row stride ``width``). Unused
    optional pointers are placeholder-bound to ``qkv``/``positions``; their loads
    sit behind ``GATED``/``STORE_CACHE`` constexpr guards, so a distinct value
    recompiles a distinct specialization and the placeholder is never touched.
    """
    tokens = qkv.shape[0]
    row_width = qkv.shape[1]
    rd = cos.shape[1]
    _qknorm_rope_q[(tokens, heads)](
        qkv,
        q_weight,
        cos,
        sin,
        positions,
        q,
        row_width,
        heads,
        rd,
        eps,
        num_warps=4,
        enable_fp_fusion=False,
    )
    _qknorm_rope_kv[(tokens,)](
        qkv,
        k_weight,
        cos,
        sin,
        positions,
        k,
        v,
        gate if gated else qkv,
        k_cache if store_cache else qkv,
        v_cache if store_cache else qkv,
        loc if store_cache else positions,
        row_width,
        heads,
        rd,
        eps,
        gated,
        store_cache,
        num_warps=4,
        enable_fp_fusion=False,
    )
