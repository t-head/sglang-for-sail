# SPDX-License-Identifier: Apache-2.0
"""Step5 grouped-union sparse paged-KV prefill in portable Triton (SM90 replacement).

Mirrors ``cute/sm90/prefill_union_paged_sm90_gqa.py`` for devices that cannot run
the CuTe DSL kernel, which is every PPU part: sm80 and sm89+ both take this path.
GQA has a single KV head, so the gather of K/V over a group's union is identical
for every query head: one program covers a block of query heads (``HB``, at most
8) and one group's query rows, loading each union token once and reusing it for
all of those heads' QK and PV products.

QK^T and PV run as ``tl.dot`` MMAs with the flattened (query row, head) pairs on
the M axis. The 192-wide head dim is split into 128 + 64 dot columns so neither
the operands nor the MMA touch the 256-wide padding the previous layout needed;
softmax runs in base-2 space and the online-softmax rescale is fused into the PV
dot accumulator.

A union entry ``window_idx`` expands to eight KV tokens::

    phys_region    = union_physical[group, window_idx]   # physical region id
    logical_region = union_logical[group, window_idx]    # logical region id
    phys_tok       = phys_region * 8 + lane              # -> K/V slot
    logical_tok    = logical_region * 8 + lane           # -> causal position

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
# 192 = BLOCK_D1 + BLOCK_D2 keeps the dots on real data instead of padding every
# K/V tile and MMA to 256.
BLOCK_D1 = 128
BLOCK_D2 = 64
BLOCK_N = 64
# Query rows the union builder packs per group, and the head block a single CTA
# covers; the launcher adds a second grid block (NB) for the 16-head layout.
GROUP_ROWS = 16
MAX_HEADS_PER_CTA = 8


@triton.jit
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
    num_rows,
    num_slots,
    softmax_scale,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D1: tl.constexpr,
    BLOCK_D2: tl.constexpr,
    HB: tl.constexpr,
    NB: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Head blocks of one group stay adjacent so their identical K/V streams share
    # cache, and groups never hit grid.y's 65535 limit.
    pid = tl.program_id(0)
    group = pid // NB
    head_block = pid % NB

    q_base = tl.load(GSTART + group)
    q_end = tl.load(GEND + group)
    q_len = tl.minimum(q_end - q_base, BLOCK_M)
    union_count = tl.minimum(tl.load(UNION_COUNTS + group), CAPACITY)
    count_tokens = union_count * REGION_TOKENS

    # M row m is query (m % BLOCK_M) of head (head_block * HB + m // BLOCK_M), so
    # the group's rows and the CTA's heads fill one MMA tile.
    m = tl.arange(0, HB * BLOCK_M)
    h_local = m // BLOCK_M
    q_local = m % BLOCK_M
    head = head_block * HB + h_local
    q_row = q_base + q_local
    mmask = (q_local < q_len) & (q_row < num_rows) & (head < HEADS)
    q_pos = tl.load(QPOS + q_row, mask=mmask, other=0)

    dims1 = tl.arange(0, BLOCK_D1)
    dims2 = tl.arange(0, BLOCK_D2)
    q_off = q_row[:, None] * (HEADS * D) + head[:, None] * D
    q1 = tl.load(Q + q_off + dims1[None, :], mask=mmask[:, None], other=0.0)
    q2 = tl.load(Q + q_off + BLOCK_D1 + dims2[None, :], mask=mmask[:, None], other=0.0)

    acc1 = tl.zeros((HB * BLOCK_M, BLOCK_D1), dtype=tl.float32)
    acc2 = tl.zeros((HB * BLOCK_M, BLOCK_D2), dtype=tl.float32)
    m_i = tl.full((HB * BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((HB * BLOCK_M,), dtype=tl.float32)
    slots_div8 = num_slots // REGION_TOKENS
    # Base-2 softmax: fold log2(e) into the scale and convert the LSE back once.
    scale2 = softmax_scale * 1.4426950408889634

    for start in range(0, count_tokens, BLOCK_N):
        tok = start + tl.arange(0, BLOCK_N)
        in_range = tok < count_tokens
        win = tok // REGION_TOKENS
        lane = tok - win * REGION_TOKENS
        union_off = group * CAPACITY + win

        phys_region = tl.load(UNION_PHYS + union_off, mask=in_range, other=-1)
        logical_region = tl.load(UNION_LOGICAL + union_off, mask=in_range, other=-1)
        memb = tl.load(MEMBERSHIP + union_off, mask=in_range, other=0)

        # Every (row, head) of the CTA reuses this gather; only the masks below
        # differ along the M axis.
        key_valid = (
            in_range
            & (phys_region >= 0)
            & (phys_region < slots_div8)
            & (logical_region >= 0)
        )
        phys_tok = tl.where(key_valid, phys_region * REGION_TOKENS + lane, 0)
        kv_row = phys_tok[:, None] * D
        kv_mask = key_valid[:, None]

        k1 = tl.load(K + kv_row + dims1[None, :], mask=kv_mask, other=0.0)
        k2 = tl.load(K + kv_row + BLOCK_D1 + dims2[None, :], mask=kv_mask, other=0.0)
        score = (
            tl.dot(q1, tl.trans(k1), out_dtype=tl.float32)
            + tl.dot(q2, tl.trans(k2), out_dtype=tl.float32)
        ) * scale2

        # Membership is per query slot, causality per query row: both ride
        # along the M axis of the score tile.
        selected = ((memb[None, :] >> q_local[:, None]) & 1) != 0
        logical_tok = logical_region * REGION_TOKENS + lane
        keep = (
            key_valid[None, :]
            & selected
            & (logical_tok[None, :] <= q_pos[:, None])
            & mmask[:, None]
        )
        score = tl.where(keep, score, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(score, axis=1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(score - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)

        v1 = tl.load(V + kv_row + dims1[None, :], mask=kv_mask, other=0.0)
        v2 = tl.load(V + kv_row + BLOCK_D1 + dims2[None, :], mask=kv_mask, other=0.0)
        p_bf16 = p.to(tl.bfloat16)
        # The online-softmax rescale rides on the dot accumulator instead of a
        # separate multiply of the whole (M, D) tile.
        acc1 = tl.dot(p_bf16, v1, acc=acc1 * alpha[:, None], out_dtype=tl.float32)
        acc2 = tl.dot(p_bf16, v2, acc=acc2 * alpha[:, None], out_dtype=tl.float32)
        m_i = m_new

    has = l_i > 0.0
    l_safe = tl.where(has, l_i, 1.0)
    inv_sum = 1.0 / l_safe
    out_off = q_row[:, None] * (HEADS * D) + head[:, None] * D
    out_mask = mmask[:, None]
    tl.store(
        OUT + out_off + dims1[None, :],
        (acc1 * inv_sum[:, None]).to(OUT.dtype.element_ty),
        mask=out_mask,
    )
    tl.store(
        OUT + out_off + BLOCK_D1 + dims2[None, :],
        (acc2 * inv_sum[:, None]).to(OUT.dtype.element_ty),
        mask=out_mask,
    )
    # m_i/p live in base-2 space; the natural LSE needs one log2 and one rescale.
    lse_val = tl.where(has, (m_i + tl.log2(l_safe)) * 0.6931471805599453, float("-inf"))
    tl.store(LSE + q_row * HEADS + head, lse_val, mask=mmask)


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
    """Run grouped-union prefill; K/V are native [slots,1,192] BF16.

    Argument order mirrors the CuTe ``compile_prefill`` call so
    ``step5_sparse_prefill`` can pick a backend without reshaping its call site.
    ``heads``, ``capacity`` and the head-dim split are compile-time specialization
    keys, matching the CuTe dispatch cache key.

    One program owns one group and one block of ``HB`` heads, so its union K/V
    gather is shared by ``HB * BLOCK_M`` MMA rows. ``BLOCK_M`` is the group row cap
    the caller already enforces (``16`` rows, or ``128 // heads`` below that), so
    the M tile stays fully occupied for 4, 8 and 16 heads per rank.
    """
    rows, heads, head_dim = q.shape
    groups, capacity = union_physical.shape
    if not groups or not rows:
        return
    if head_dim != BLOCK_D1 + BLOCK_D2:
        raise NotImplementedError("Step5 Triton sparse prefill requires head_dim=192")
    # Max packed query rows per group, matching the CuTe q_group contract and the
    # union builder's 16-slot packing.
    q_group = min(GROUP_ROWS, 128 // heads)
    head_block = min(heads, MAX_HEADS_PER_CTA)
    num_blocks = triton.cdiv(heads, head_block)
    _sparse_prefill[(num_blocks * groups,)](
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
        k.shape[0],
        softmax_scale,
        heads,
        head_dim,
        BLOCK_D1,
        BLOCK_D2,
        head_block,
        num_blocks,
        capacity,
        q_group,
        BLOCK_N,
        num_warps=8,
        num_stages=2,
    )
