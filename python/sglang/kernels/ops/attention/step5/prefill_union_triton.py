# SPDX-License-Identifier: Apache-2.0
"""Step5 grouped prefill-union construction in portable Triton (CuTe replacement).

Mirrors ``cute/prefill_union.py`` for devices that cannot run the CuTe DSL kernel
(e.g. sm80 PPU). Two launches, one program per group of up to 16 packed query
rows from a single request:

1. ``_union_bitset_fill`` clears the group's bitset then scans the 16x513
   selections and ``atomic_or``s one bit per selected logical region
   (``logical_start // 8``) into the bitset.
2. ``_union_bitset_compact`` popcounts the bitset words, runs an inclusive
   ``cumsum`` to get each word's exclusive bit prefix, re-scans the selections,
   and scatters each region to its sorted ``rank`` (bits set before it) in
   ``union_physical``/``union_logical`` while ``atomic_or``ing the selecting
   query slot into ``membership``.

Popcount is hand-written SWAR (no libdevice), which is what the CuTe source
approximates with ``popc.b32``. Logical/physical selection tensors support
``stride(1)`` of 1 or 2 -- the packed-int64-as-int32 high/low word view.
"""

import torch
import triton
import triton.language as tl

Q_GROUP = tl.constexpr(16)
TOPK_WINDOWS = tl.constexpr(513)
WIN_BLOCK = tl.constexpr(1024)  # next power of two >= TOPK_WINDOWS
CLEAR_BLOCK = tl.constexpr(1024)


@triton.jit
def _popcount(x):
    """Hand-written 32-bit SWAR popcount. Returns int32 in [0, 32].

    Bitcast to uint32 so the shifts stay logical; the int32 bitset words may have
    the sign bit set (region 31 of a word), and an arithmetic shift would corrupt
    the SWAR stages.
    """
    u = x.to(tl.uint32, bitcast=True)
    u = u - ((u >> 1) & 0x55555555)
    u = (u & 0x33333333) + ((u >> 2) & 0x33333333)
    u = (u + (u >> 4)) & 0x0F0F0F0F
    u = u + (u >> 8)
    u = u + (u >> 16)
    return (u & 0x3F).to(tl.int32)


@triton.jit
def _union_bitset_fill(
    LOGICAL,
    PHYS,
    COUNTS,
    GSTART,
    GEND,
    BITSET,
    ROWS,
    SEL_STRIDE: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    BIT_WORDS: tl.constexpr,
    NUM_REGION_BINS: tl.constexpr,
):
    group = tl.program_id(0)
    q_base = tl.load(GSTART + group)
    q_end = tl.load(GEND + group)

    # Clear this group's bitset words, then barrier before the atomic accumulates.
    for off in tl.static_range(0, BIT_WORDS, CLEAR_BLOCK):
        idx = off + tl.arange(0, CLEAR_BLOCK)
        tl.store(BITSET + group * BIT_WORDS + idx, 0, mask=idx < BIT_WORDS)
    tl.debug_barrier()

    for q_slot in tl.static_range(Q_GROUP):
        q_pos = q_base + q_slot
        row_valid = (q_pos < ROWS) & (q_pos < q_end)
        count = tl.load(COUNTS + q_pos, mask=row_valid, other=0)
        win = tl.arange(0, WIN_BLOCK)
        valid = row_valid & (win < count) & (win < TOPK_WINDOWS)
        loff = q_pos * ROW_STRIDE + win * SEL_STRIDE
        logical_start = tl.load(LOGICAL + loff, mask=valid, other=0)
        phys = tl.load(PHYS + loff, mask=valid, other=-1)
        logical_region = logical_start // 8
        keep = (
            valid
            & (logical_region >= 0)
            & (logical_region < NUM_REGION_BINS)
            & (phys >= 0)
        )
        word_idx = tl.where(keep, logical_region // 32, 0)
        bit_mask = 1 << (logical_region - word_idx * 32)
        tl.atomic_or(BITSET + group * BIT_WORDS + word_idx, bit_mask, mask=keep)


@triton.jit
def _union_bitset_compact(
    LOGICAL,
    PHYS,
    COUNTS,
    GSTART,
    GEND,
    BITSET,
    UNION_PHYS,
    UNION_LOGICAL,
    UNION_COUNTS,
    MEMBERSHIP,
    ROWS,
    SEL_STRIDE: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    BIT_WORDS: tl.constexpr,
    WORDS_BLOCK: tl.constexpr,
    NUM_REGION_BINS: tl.constexpr,
    CAPACITY: tl.constexpr,
):
    group = tl.program_id(0)
    q_base = tl.load(GSTART + group)
    q_end = tl.load(GEND + group)

    # Load the group's bitset words, popcount, and build the exclusive bit prefix.
    widx = tl.arange(0, WORDS_BLOCK)
    words = tl.load(BITSET + group * BIT_WORDS + widx, mask=widx < BIT_WORDS, other=0)
    popc = _popcount(words)
    incl = tl.cumsum(popc, axis=0)
    excl = incl - popc  # excl[i] = number of set bits in words before i
    union_count = tl.minimum(tl.sum(popc, axis=0), CAPACITY)
    tl.store(UNION_COUNTS + group, union_count)

    # Clear membership before the atomic_or accumulation below.
    for off in tl.static_range(0, CAPACITY, CLEAR_BLOCK):
        idx = off + tl.arange(0, CLEAR_BLOCK)
        tl.store(MEMBERSHIP + group * CAPACITY + idx, 0, mask=idx < CAPACITY)
    tl.debug_barrier()

    for q_slot in tl.static_range(Q_GROUP):
        q_pos = q_base + q_slot
        row_valid = (q_pos < ROWS) & (q_pos < q_end)
        count = tl.load(COUNTS + q_pos, mask=row_valid, other=0)
        win = tl.arange(0, WIN_BLOCK)
        valid = row_valid & (win < count) & (win < TOPK_WINDOWS)
        loff = q_pos * ROW_STRIDE + win * SEL_STRIDE
        logical_start = tl.load(LOGICAL + loff, mask=valid, other=0)
        phys = tl.load(PHYS + loff, mask=valid, other=-1)
        logical_region = logical_start // 8
        keep = (
            valid
            & (logical_region >= 0)
            & (logical_region < NUM_REGION_BINS)
            & (phys >= 0)
        )
        word_idx = tl.where(keep, logical_region // 32, 0)
        bit_idx = logical_region - word_idx * 32
        bit_mask = 1 << bit_idx
        gathered_word = tl.gather(words, word_idx, axis=0)
        gathered_excl = tl.gather(excl, word_idx, axis=0)
        rank = gathered_excl + _popcount(gathered_word & (bit_mask - 1))
        write = keep & ((gathered_word & bit_mask) != 0) & (rank < CAPACITY)
        tl.store(UNION_PHYS + group * CAPACITY + rank, phys, mask=write)
        tl.store(UNION_LOGICAL + group * CAPACITY + rank, logical_region, mask=write)
        tl.atomic_or(MEMBERSHIP + group * CAPACITY + rank, 1 << q_slot, mask=write)


def launch(
    logical_starts: torch.Tensor,
    physical_regions: torch.Tensor,
    counts: torch.Tensor,
    group_query_start: torch.Tensor,
    group_query_end: torch.Tensor,
    bitset: torch.Tensor,
    union_physical: torch.Tensor,
    union_logical: torch.Tensor,
    union_counts: torch.Tensor,
    membership: torch.Tensor,
    groups: int,
    rows: int,
    *,
    num_region_bins: int,
    capacity: int,
    selection_stride: int,
) -> None:
    """Run the bitset fill then the compaction, one program per group.

    Argument order mirrors the CuTe ``run`` closure so ``step5_build_prefill_union``
    can pick a backend without reshaping its call site. ``num_region_bins``,
    ``capacity`` and ``selection_stride`` are compile-time specialization keys,
    matching ``_compile_union``'s cache key.
    """
    bit_words = (num_region_bins + 31) // 32
    words_block = triton.next_power_of_2(bit_words)
    row_stride = TOPK_WINDOWS.value * selection_stride
    _union_bitset_fill[(groups,)](
        logical_starts,
        physical_regions,
        counts,
        group_query_start,
        group_query_end,
        bitset,
        rows,
        selection_stride,
        row_stride,
        bit_words,
        num_region_bins,
        num_warps=4,
    )
    _union_bitset_compact[(groups,)](
        logical_starts,
        physical_regions,
        counts,
        group_query_start,
        group_query_end,
        bitset,
        union_physical,
        union_logical,
        union_counts,
        membership,
        rows,
        selection_stride,
        row_stride,
        bit_words,
        words_block,
        num_region_bins,
        capacity,
        num_warps=4,
    )
