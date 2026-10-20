# SPDX-License-Identifier: Apache-2.0
"""Step5 sparse decode over native FULL KV in portable Triton (SM90 replacement).

Mirrors ``cute/sm90/decode_sm90_gqa.py`` for devices that cannot run the CuTe DSL
kernel, which is every PPU part: sm80 and sm89+ both take this path. One program
owns one query row plus a block of ``BLOCK_M`` logical heads, and runs an online
softmax over that row's own packed region list, writing the final BF16 output and
FP32 natural LSE directly.

QK^T and PV run as ``tl.dot`` MMAs: the query heads fill the M dimension and the
row's selected tokens the N dimension, with the 192-wide head dim split into
128 + 64 dot columns so no operand is padded to 256. Softmax runs in base-2
space and the PV accumulation absorbs the online-softmax rescale through the dot
accumulator. The SM90 split-KV schedule (1/2/4-way partitions plus a separate
natural-LSE merge) stays bypassed: every row is computed in one pass.

Each row's ``packed_regions[row, win]`` is an int64 whose high word is the
physical region id and low word the logical start token. The kernel reads it as
the reinterpreted int32 pair view, so on a little-endian device::

    start_tok   = packed32[2 * win]       # low word: logical position of lane 0
    phys_region = packed32[2 * win + 1]   # high word: -> K/V slot region
    phys_tok    = phys_region * 8 + lane
    logical_tok = start_tok + lane

A key is usable only when all hold, matching the SM90 ``_mask_scores``:

1. ``start_tok >= 0`` and ``phys_region >= 0`` (both words of a packed ``-1``)
2. physical slot valid: ``0 <= phys_region < slots // 8``
3. within the row's selection: ``tok < counts[row] * 8``
4. causal: ``logical_tok < causal_lengths[row]``

GQA has a single KV head, so all logical heads share K/V. ``BLOCK_M`` is the
16-row MMA minimum, so the 4 and 8 head layouts (TP16/TP8) leave part of the M
tile idle; that padding still costs less than the per-(row, head) scalar dot it
replaces, and TP16 is measured separately. Rows at or beyond ``valid_rows[0]``
(or with zero counts) write zero output and ``-inf`` LSE. Only BF16 K/V is
supported; the FP8 paged path is not implemented.
"""

import torch
import triton
import triton.language as tl

REGION_TOKENS = tl.constexpr(8)
# 192 = BLOCK_D1 + BLOCK_D2 keeps both dots on real data: padding the head dim to
# 256 made every K/V tile load and MMA 33% wider than the row it carries.
BLOCK_D1 = 128
BLOCK_D2 = 64
# MMA minimum along the query-head axis, and 16 windows of 8 tokens along the
# selected-token axis.
BLOCK_M = 16
BLOCK_N = 128


@triton.jit
def _sparse_decode(
    Q,
    K,
    V,
    PACKED32,
    COUNTS,
    CAUSAL_LEN,
    VALID_ROWS,
    OUT,
    LSE,
    num_slots,
    softmax_scale,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D1: tl.constexpr,
    BLOCK_D2: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    head_block = tl.program_id(1)
    heads = head_block * BLOCK_M + tl.arange(0, BLOCK_M)
    head_valid = heads < HEADS

    valid_row = row < tl.load(VALID_ROWS)
    count = tl.minimum(tl.load(COUNTS + row), WIDTH)
    count = tl.where(valid_row, count, 0)
    count_tokens = count * REGION_TOKENS
    causal_len = tl.load(CAUSAL_LEN + row)

    dims1 = tl.arange(0, BLOCK_D1)
    dims2 = tl.arange(0, BLOCK_D2)
    q_off = row * (HEADS * D) + heads[:, None] * D
    q1 = tl.load(Q + q_off + dims1[None, :], mask=head_valid[:, None], other=0.0)
    q2 = tl.load(
        Q + q_off + BLOCK_D1 + dims2[None, :],
        mask=head_valid[:, None],
        other=0.0,
    )

    acc1 = tl.zeros((BLOCK_M, BLOCK_D1), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_M, BLOCK_D2), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    num_slots_div8 = num_slots // REGION_TOKENS
    # Base-2 softmax: fold log2(e) into the scale and convert the LSE back once.
    scale2 = softmax_scale * 1.4426950408889634

    for start in range(0, count_tokens, BLOCK_N):
        tok = start + tl.arange(0, BLOCK_N)
        in_range = tok < count_tokens
        win = tok // REGION_TOKENS
        lane = tok - win * REGION_TOKENS

        # The int64 packing is read as its low/high int32 words, so no 64-bit
        # shift is needed per token and a packed -1 negates both words.
        pack_off = row * (2 * WIDTH) + 2 * win
        start_tok = tl.load(PACKED32 + pack_off, mask=in_range, other=-1)
        physical = tl.load(PACKED32 + pack_off + 1, mask=in_range, other=-1)
        key_valid = (
            in_range
            & (start_tok >= 0)
            & (physical >= 0)
            & (physical < num_slots_div8)
            & ((start_tok + lane) < causal_len)
        )
        phys_tok = tl.where(key_valid, physical * REGION_TOKENS + lane, 0)
        kv_row = phys_tok[:, None] * D
        kv_mask = key_valid[:, None]

        k1 = tl.load(K + kv_row + dims1[None, :], mask=kv_mask, other=0.0)
        k2 = tl.load(K + kv_row + BLOCK_D1 + dims2[None, :], mask=kv_mask, other=0.0)
        score = (
            tl.dot(q1, tl.trans(k1), out_dtype=tl.float32)
            + tl.dot(q2, tl.trans(k2), out_dtype=tl.float32)
        ) * scale2
        score = tl.where(key_valid[None, :], score, float("-inf"))

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
    # m_i/p live in base-2 space; the natural LSE needs one log2 and one rescale.
    lse_val = tl.where(has, (m_i + tl.log2(l_safe)) * 0.6931471805599453, float("-inf"))
    out_off = row * (HEADS * D) + heads[:, None] * D
    out_mask = head_valid[:, None]
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
    tl.store(LSE + row * HEADS + heads, lse_val, mask=head_valid)


def launch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    packed_regions: torch.Tensor,
    counts: torch.Tensor,
    causal_lengths: torch.Tensor,
    valid_rows: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    *,
    softmax_scale: float,
) -> None:
    """Run one program per (row, head block); K/V are native [slots,1,192] BF16.

    Single-pass decode with no split-KV partitioning or merge, so ``out`` is the
    final BF16 attention of shape ``[rows, heads, 192]`` and ``lse`` the FP32
    natural log-sum-exp ``[rows, heads]``. ``heads``, the packed width and the
    head-dim split are compile-time specialization keys.
    """
    if k.dtype != torch.bfloat16:
        raise ValueError("Step5 Triton sparse decode supports only BF16 K/V")
    rows, heads, head_dim = q.shape
    if not rows:
        return
    if head_dim != BLOCK_D1 + BLOCK_D2:
        raise NotImplementedError("Step5 Triton sparse decode requires head_dim=192")
    _sparse_decode[(rows, triton.cdiv(heads, BLOCK_M))](
        q,
        k,
        v,
        # Two int32 words per packed int64: the low word is the logical start and
        # the high word the physical region, matching the CuTe fixed-layout view.
        packed_regions.view(torch.int32),
        counts,
        causal_lengths,
        valid_rows,
        out,
        lse,
        k.shape[0],
        softmax_scale,
        heads,
        head_dim,
        BLOCK_D1,
        BLOCK_D2,
        packed_regions.shape[1],
        BLOCK_M,
        BLOCK_N,
        num_warps=4,
        num_stages=2,
    )
