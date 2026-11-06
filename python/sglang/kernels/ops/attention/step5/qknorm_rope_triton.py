# SPDX-License-Identifier: Apache-2.0
"""Step5 Q/K norm + half-split RoPE in portable Triton (CuTe replacement).

Mirrors ``cute/qknorm_rope.py`` ``FusedQKNormRope`` for devices that cannot run
the CuTe DSL kernel (e.g. sm80 PPU). Each 192-wide head row is normalized with
the zero-centered GemmaRMSNorm (``weight + 1``), materialized as BF16 --
matching the CuTe smem round -- then rotated with the half-split NeoX RoPE
straight from the model's merged ``[cos | sin]`` table.

Each program owns a tile of rows instead of a single row: a one-row program
spent most of its time in the grid prologue and in the cross-thread
``tl.gather`` that built the RoPE partner. A row is instead loaded as its two
rotary halves ``[0, rd)`` / ``[rd, 2*rd)`` plus the passthrough tail
``[2*rd, 192)``, so the partner of one half is simply the other half and every
segment is stored once to a disjoint address range.

Q rows and K/V tokens share one launch over a flat grid: the leading programs
cover ``tokens * heads`` Q rows, the trailing ones cover token tiles that carry
K, the V copy, the head gate and the optional paged KV-cache scatter (``loc <
0`` rows are skipped).
"""

import torch
import triton
import triton.language as tl

# Rows per Q program and tokens per K/V program; 16 rows of 192 BF16 keep a
# program at ~6 KiB of traffic, enough to amortize its prologue.
_BLOCK_Q = 16
_BLOCK_KV = 16


@triton.jit
def _norm_rope_rows(
    X,
    W,
    COS,
    SIN,
    x_off,
    pos,
    row_mask,
    OUT,
    out_off,
    OUT2,
    out2_off,
    out2_mask,
    RD: tl.constexpr,
    RD_P: tl.constexpr,
    TAIL: tl.constexpr,
    TAIL_P: tl.constexpr,
    EPS: tl.constexpr,
    HAS_OUT2: tl.constexpr,
):
    """RMSNorm(192, zero-centered) -> BF16 -> half-split RoPE for a row tile.

    ``x_off`` / ``out_off`` / ``out2_off`` are per-row element offsets of the
    192-wide row; ``pos`` is each row's rotary table row. ``COS``/``SIN`` are
    the split views of the merged table, so each has a row stride of ``2 * RD``.
    ``OUT2`` is an optional second destination (the paged cache) masked by
    ``out2_mask``.
    """
    c = tl.arange(0, RD_P)
    cmask = c < RD
    m = row_mask[:, None] & cmask[None, :]
    x_lo = X + x_off[:, None] + c[None, :]
    x1 = tl.load(x_lo, mask=m, other=0.0).to(tl.float32)
    x2 = tl.load(x_lo + RD, mask=m, other=0.0).to(tl.float32)
    ss = tl.sum(x1 * x1, 1) + tl.sum(x2 * x2, 1)
    if TAIL > 0:
        t = tl.arange(0, TAIL_P)
        tmask = t < TAIL
        mt = row_mask[:, None] & tmask[None, :]
        xt = tl.load(X + x_off[:, None] + 2 * RD + t[None, :], mask=mt, other=0.0)
        xt = xt.to(tl.float32)
        ss += tl.sum(xt * xt, 1)
    inverse_rms = tl.rsqrt(ss / 192 + EPS)[:, None]

    w1 = tl.load(W + c, mask=cmask, other=0.0)[None, :]
    w2 = tl.load(W + RD + c, mask=cmask, other=0.0)[None, :]
    # Norm has its own BF16 boundary before RoPE, matching the CuTe smem round.
    n1 = (x1 * inverse_rms * (w1 + 1.0)).to(tl.bfloat16).to(tl.float32)
    n2 = (x2 * inverse_rms * (w2 + 1.0)).to(tl.bfloat16).to(tl.float32)
    table = pos[:, None] * (2 * RD) + c[None, :]
    cos = tl.load(COS + table, mask=m, other=0.0).to(tl.float32)
    sin = tl.load(SIN + table, mask=m, other=0.0).to(tl.float32)
    o1 = (n1 * cos - n2 * sin).to(tl.bfloat16)
    o2 = (n2 * cos + n1 * sin).to(tl.bfloat16)

    o_lo = OUT + out_off[:, None] + c[None, :]
    tl.store(o_lo, o1, mask=m)
    tl.store(o_lo + RD, o2, mask=m)
    if HAS_OUT2:
        m2 = out2_mask[:, None] & cmask[None, :]
        o2_lo = OUT2 + out2_off[:, None] + c[None, :]
        tl.store(o2_lo, o1, mask=m2)
        tl.store(o2_lo + RD, o2, mask=m2)
    if TAIL > 0:
        wt = tl.load(W + 2 * RD + t, mask=tmask, other=0.0)[None, :]
        nt = (xt * inverse_rms * (wt + 1.0)).to(tl.bfloat16)
        tl.store(OUT + out_off[:, None] + 2 * RD + t[None, :], nt, mask=mt)
        if HAS_OUT2:
            mt2 = out2_mask[:, None] & tmask[None, :]
            tl.store(OUT2 + out2_off[:, None] + 2 * RD + t[None, :], nt, mask=mt2)


@triton.jit
def _qknorm_rope(
    QKV,
    QW,
    KW,
    COS,
    SIN,
    POS,
    Q,
    K,
    V,
    GATE,
    KCACHE,
    VCACHE,
    LOC,
    tokens,
    q_programs,
    ROW_WIDTH: tl.constexpr,
    HEADS: tl.constexpr,
    RD: tl.constexpr,
    RD_P: tl.constexpr,
    TAIL: tl.constexpr,
    TAIL_P: tl.constexpr,
    EPS: tl.constexpr,
    GATED: tl.constexpr,
    STORE_CACHE: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < q_programs:
        # Q rows: flat row ``r`` is head ``r % HEADS`` of token ``r // HEADS``,
        # and Q is contiguous [T, HEADS * 192], so its output row is ``r``.
        r = pid.to(tl.int64) * BLOCK_Q + tl.arange(0, BLOCK_Q)
        rmask = r < tokens * HEADS
        token = r // HEADS
        head = r % HEADS
        pos = tl.load(POS + token, mask=rmask, other=0).to(tl.int64)
        _norm_rope_rows(
            QKV,
            QW,
            COS,
            SIN,
            token * ROW_WIDTH + head * 192,
            pos,
            rmask,
            Q,
            r * 192,
            Q,
            r * 192,
            rmask,
            RD,
            RD_P,
            TAIL,
            TAIL_P,
            EPS,
            False,
        )
    else:
        # K/V tokens. The gate and cache pointers are placeholders unless their
        # constexpr guards hold, so only the specialization touches them.
        token = (pid - q_programs).to(tl.int64) * BLOCK_KV + tl.arange(0, BLOCK_KV)
        tmask = token < tokens
        pos = tl.load(POS + token, mask=tmask, other=0).to(tl.int64)
        row = token * ROW_WIDTH
        if STORE_CACHE:
            loc = tl.load(LOC + token, mask=tmask, other=-1).to(tl.int64)
            cache_mask = tmask & (loc >= 0)
            cache_off = loc * 192
        else:
            cache_mask = tmask
            cache_off = token * 192
        # K head: norm + RoPE, written to the [T,192] K output and (optionally)
        # cache.
        _norm_rope_rows(
            QKV,
            KW,
            COS,
            SIN,
            row + HEADS * 192,
            pos,
            tmask,
            K,
            token * 192,
            KCACHE,
            cache_off,
            cache_mask,
            RD,
            RD_P,
            TAIL,
            TAIL_P,
            EPS,
            STORE_CACHE,
        )
        # V head: plain copy, no norm and no RoPE.
        channel = tl.arange(0, 256)
        cmask = channel < 192
        vm = tmask[:, None] & cmask[None, :]
        v_raw = tl.load(
            QKV + (row + (HEADS + 1) * 192)[:, None] + channel[None, :],
            mask=vm,
            other=0.0,
        )
        tl.store(V + (token * 192)[:, None] + channel[None, :], v_raw, mask=vm)
        if STORE_CACHE:
            tl.store(
                VCACHE + cache_off[:, None] + channel[None, :],
                v_raw,
                mask=cache_mask[:, None] & cmask[None, :],
            )
        if GATED:
            g = tl.arange(0, 16)
            gm = tmask[:, None] & (g < HEADS)[None, :]
            gate_val = tl.load(
                QKV + (row + (HEADS + 2) * 192)[:, None] + g[None, :],
                mask=gm,
                other=0.0,
            )
            tl.store(GATE + (token * HEADS)[:, None] + g[None, :], gate_val, mask=gm)


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
    """Run the Q rows and the K/V/gate/cache tokens in one launch, in place.

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
    tail = 192 - 2 * rd
    q_programs = triton.cdiv(tokens * heads, _BLOCK_Q)
    kv_programs = triton.cdiv(tokens, _BLOCK_KV)
    _qknorm_rope[(q_programs + kv_programs,)](
        qkv,
        q_weight,
        k_weight,
        cos,
        sin,
        positions,
        q,
        k,
        v,
        gate if gated else qkv,
        k_cache if store_cache else qkv,
        v_cache if store_cache else qkv,
        loc if store_cache else positions,
        tokens,
        q_programs,
        row_width,
        heads,
        rd,
        triton.next_power_of_2(rd),
        tail,
        triton.next_power_of_2(tail) if tail > 0 else 1,
        eps,
        gated,
        store_cache,
        _BLOCK_Q,
        _BLOCK_KV,
        num_warps=4,
        enable_fp_fusion=False,
    )
