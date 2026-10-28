# SPDX-License-Identifier: Apache-2.0
"""Step5 grouped-union sparse paged-KV prefill in portable Triton (SM90 replacement).

Mirrors ``cute/sm90/prefill_union_paged_sm90_gqa.py`` for devices that cannot run
the CuTe DSL kernel (e.g. sm80 PPU) and keeps its work decomposition: one program
covers a tile of a group's query rows for *all* logical heads, flattened into the
MMA M dimension as ``q_slot * HEADS + head``. GQA has a single KV head, so a
per-head program (the previous layout) re-read this layer's union from HBM once
per head; merging heads into one M tile reduces that traffic by ``HEADS`` and
lets QK^T / PV run as real MMAs.

A union entry ``window_idx`` expands to eight KV tokens::

    phys_region   = union_physical[group, window_idx]   # physical region id
    logical_region = union_logical[group, window_idx]    # logical region id
    phys_tok      = phys_region * 8 + lane               # -> K/V slot
    logical_tok   = logical_region * 8 + lane            # -> causal position

A key is usable only when all three hold, exactly as the SM90 exact-union mask:

1. physical slot valid: ``0 <= phys_region < slots // 8``
2. membership selects this query: ``(membership[group, window_idx] >> q_slot) & 1``
3. causal: ``logical_region >= 0`` and ``logical_tok <= query_positions[row]``

Output is BF16; LSE is the natural logarithm of the softmax denominator. Empty
rows write zero output and ``-inf`` LSE; rows outside the group's range are left
untouched.
"""

import torch
import triton
import triton.language as tl

REGION_TOKENS = tl.constexpr(8)
HEAD_DIM = tl.constexpr(192)
PAD_DIM = tl.constexpr(256)


@triton.jit(
    do_not_specialize=("ROWS", "SLOTS_DIV8"),
    do_not_specialize_on_alignment=("ROWS", "SLOTS_DIV8"),
)
def _sparse_prefill(
    Q,
    K,
    V,
    UNION_PHYS,
    UNION_LOGICAL,
    UNION_COUNTS,
    MEMBERSHIP,
    GSTART,
    GEND,
    QPOS,
    OUT,
    LSE,
    ROWS,
    SLOTS_DIV8,
    SOFTMAX_SCALE,
    HEADS: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK_N: tl.constexpr,
    Q_GROUP: tl.constexpr,
    Q_TILE: tl.constexpr,
):
    group = tl.program_id(0).to(tl.int64)
    slot_base = tl.program_id(1) * Q_TILE

    q_base = tl.load(GSTART + group)
    q_end = tl.load(GEND + group)
    q_tile_count = tl.minimum(tl.maximum(q_end - q_base, 0), Q_GROUP)
    if slot_base < q_tile_count:
        union_count = tl.minimum(tl.load(UNION_COUNTS + group), CAPACITY)
        count_tokens = union_count * REGION_TOKENS

        # M rows of the MMA tile: one (query slot, head) pair each, the same
        # layout the SM90 warpgroup uses for its query rows.
        m = tl.arange(0, Q_TILE * HEADS)
        slot = slot_base + m // HEADS
        head = m % HEADS
        row = (q_base + slot).to(tl.int64)
        mmask = (slot < q_tile_count) & (row < ROWS)

        channel = tl.arange(0, PAD_DIM)
        cmask = channel < HEAD_DIM
        q_vec = tl.load(
            Q + row[:, None] * (HEADS * HEAD_DIM) + head[:, None] * HEAD_DIM + channel[None, :],
            mask=mmask[:, None] & cmask[None, :],
            other=0.0,
        )
        q_pos = tl.load(QPOS + row, mask=mmask, other=0)

        m_i = tl.full((Q_TILE * HEADS,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((Q_TILE * HEADS,), dtype=tl.float32)
        acc = tl.zeros((Q_TILE * HEADS, PAD_DIM), dtype=tl.float32)

        for start in range(0, count_tokens, BLOCK_N):
            tok = start + tl.arange(0, BLOCK_N)
            tvalid = tok < count_tokens
            win = tok // REGION_TOKENS
            lane = tok - win * REGION_TOKENS

            phys_region = tl.load(UNION_PHYS + group * CAPACITY + win, mask=tvalid, other=-1)
            logical_region = tl.load(UNION_LOGICAL + group * CAPACITY + win, mask=tvalid, other=-1)
            memb = tl.load(MEMBERSHIP + group * CAPACITY + win, mask=tvalid, other=0)

            valid_phys = (phys_region >= 0) & (phys_region < SLOTS_DIV8)
            load_ok = tvalid & valid_phys
            phys_tok = (phys_region * REGION_TOKENS + lane).to(tl.int64)
            logical_tok = logical_region * REGION_TOKENS + lane
            # Membership is per query slot, causality per query row: both ride
            # along the M axis of the score tile.
            selected = ((memb[None, :] >> slot[:, None]) & 1) != 0
            keep = (
                tvalid[None, :]
                & valid_phys[None, :]
                & selected
                & (logical_region >= 0)[None, :]
                & (logical_tok[None, :] <= q_pos[:, None])
            )

            # K is read transposed so QK^T is a single MMA over the 256-wide
            # (192 real) head dimension; the same 192-wide rows then feed PV.
            k = tl.load(
                K + phys_tok[None, :] * HEAD_DIM + channel[:, None],
                mask=load_ok[None, :] & cmask[:, None],
                other=0.0,
            )
            score = tl.dot(q_vec, k) * SOFTMAX_SCALE
            score = tl.where(keep, score, float("-inf"))
            v = tl.load(
                V + phys_tok[:, None] * HEAD_DIM + channel[None, :],
                mask=load_ok[:, None] & cmask[None, :],
                other=0.0,
            )

            m_new = tl.maximum(m_i, tl.max(score, axis=1))
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            p = tl.exp(score - m_safe[:, None])
            alpha = tl.exp(m_i - m_safe)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
            m_i = m_new

        has = l_i > 0.0
        l_safe = tl.where(has, l_i, 1.0)
        out_val = tl.where(has[:, None], acc / l_safe[:, None], 0.0)
        lse_val = tl.where(has, m_i + tl.log(l_safe), float("-inf"))
        tl.store(
            OUT + row[:, None] * (HEADS * HEAD_DIM) + head[:, None] * HEAD_DIM + channel[None, :],
            out_val.to(tl.bfloat16),
            mask=mmask[:, None] & cmask[None, :],
        )
        tl.store(LSE + row * HEADS + head, lse_val, mask=mmask)


def launch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    union_physical: torch.Tensor,
    union_logical: torch.Tensor,
    union_counts: torch.Tensor,
    membership: torch.Tensor,
    group_query_start: torch.Tensor,
    group_query_end: torch.Tensor,
    query_positions: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    *,
    softmax_scale: float,
) -> None:
    """Run one program per (group, query-row tile); K/V are native [slots,1,192] BF16.

    Argument order mirrors the CuTe ``compile_prefill`` call so
    ``step5_sparse_prefill`` can pick a backend without reshaping its call site.
    ``heads`` and ``capacity`` are compile-time specialization keys, matching the
    CuTe dispatch cache key.

    ``Q_TILE * heads`` is the MMA M dimension. 64 rows keeps the 8-warp tile
    resident and was the fastest split measured on PPU for heads 4/8/16; a whole
    group is one program when the head count allows (heads=4).
    """
    rows, heads, _ = q.shape
    groups, capacity = union_physical.shape
    slots_div8 = k.shape[0] // 8
    # Max packed query rows per group: the caller's ranges hold at most 16 rows,
    # and the head count caps the CuTe q_group contract below that.
    q_group = min(16, 128 // heads)
    q_tile = min(q_group, 64 // heads)
    _sparse_prefill[(groups, triton.cdiv(q_group, q_tile))](
        q,
        k,
        v,
        union_physical,
        union_logical,
        union_counts,
        membership,
        group_query_start,
        group_query_end,
        query_positions,
        out,
        lse,
        rows,
        slots_div8,
        softmax_scale,
        heads,
        capacity,
        64,
        q_group,
        q_tile,
        num_warps=8,
        num_stages=2,
    )
