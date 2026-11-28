"""Private SAIL MSA page/score metadata kernels (no runtime configuration)."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False
_PAGE_SIZE = 128
_INIT_SCORE = 1.0e30
_LOCAL_SCORE = 1.0e29

if _HAS_TRITON:

    @triton.jit
    def _pack_len_meta_kernel(
        out_ptr,  # int32 [>= sum(pages)] packed physical page ids
        req_to_token_ptr,  # int64 [max_reqs, max_kv_len]
        req_pool_ptr,  # int64 [num_reqs]
        seq_lens_ptr,  # int32/int64 [num_reqs] request lengths (read-only)
        kv_lens_ptr,  # int32 [num_reqs] out: lengths cast to int32
        tok_pages_ptr,  # int32 [num_reqs] out: per-request page counts
        start_ptr,  # int32 [num_reqs] out: exclusive cumsum of page counts
        r2t_stride0,
        n_phys_pages,
        PAGE_SIZE: tl.constexpr,
        BLOCK_J: tl.constexpr,
        BLOCK_S: tl.constexpr,
    ):
        pid_req = tl.program_id(0)
        pid_blk = tl.program_id(1)
        kv_len = tl.load(seq_lens_ptr + pid_req)
        n_pages = (kv_len + (PAGE_SIZE - 1)) // PAGE_SIZE
        # Exclusive prefix sum over the request head: every program recomputes
        # it because a kernel cannot share results across the grid, and only
        # the owner of a segment knows where to pack its pages. BLOCK_J is
        # sized from the caller's page capacity so the second grid dimension
        # stays at 1-2 and the redundant reads stay within a small multiple of
        # the standalone metadata kernel's (seq_lens stays L2-resident).
        i = tl.arange(0, BLOCK_S)
        head = tl.load(seq_lens_ptr + i, mask=i < pid_req, other=0)
        start = tl.sum((head + (PAGE_SIZE - 1)) // PAGE_SIZE, axis=0)
        if pid_blk == 0:
            tl.store(kv_lens_ptr + pid_req, kv_len.to(tl.int32))
            tl.store(tok_pages_ptr + pid_req, n_pages.to(tl.int32))
            tl.store(start_ptr + pid_req, start.to(tl.int32))
        j = pid_blk * BLOCK_J + tl.arange(0, BLOCK_J)
        mask = j < n_pages
        row = tl.load(req_pool_ptr + pid_req)
        # Physical page id = req_to_token[req, p] // PAGE_SIZE (one slot per page).
        slot = tl.load(
            req_to_token_ptr + row * r2t_stride0 + j * PAGE_SIZE,
            mask=mask,
            other=0,
        )
        page = tl.minimum(slot // PAGE_SIZE, n_phys_pages - 1)
        tl.store(out_ptr + start + j, page.to(tl.int32), mask=mask)

    @triton.jit
    def _force_init_local_scores_kernel(
        score_ptr,  # fp32/bf16 [H, K, T] contiguous: (head, k_tile, token)
        pages_ptr,  # int32 [T] valid page-tile count per token
        stride_h,
        stride_k,
        stride_t,
        init_blocks,
        local_blocks,
        INIT_SCORE: tl.constexpr,
        LOCAL_SCORE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        pid_h = tl.program_id(1)
        pages = tl.load(pages_ptr + pid_t)
        base = score_ptr + pid_h * stride_h + pid_t * stride_t
        k = tl.arange(0, BLOCK)
        # Init blocks [0, min(init_blocks, pages)) at 1e30.
        n_init = tl.minimum(init_blocks, pages)
        tl.store(base + k * stride_k, INIT_SCORE, mask=k < n_init)
        # Local window [max(0, pages - local_blocks), pages) at 1e29.
        lo = tl.maximum(pages - local_blocks, 0)
        lk = lo + k
        tl.store(base + lk * stride_k, LOCAL_SCORE, mask=lk < pages)


def _pack_len_meta_into(
    out: torch.Tensor,  # int32 [>= bs * max_pages] capacity-sized page table
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,  # int32/int64 [bs] (CUDA)
    kv_lens: torch.Tensor,  # int32 [>=bs] out: lengths cast to int32
    tok_pages: torch.Tensor,  # int32 [>=bs] out: per-request page counts
    page_starts: torch.Tensor,  # int32 [>=bs] out: exclusive cumsum
    bs: int,
    max_pages: int,  # page-capacity grid bound (>= any request's page count)
    n_phys_pages: int,
) -> None:
    """Fill length metadata and pack the page table in one kernel.

    Fuses the former metadata kernel + page-table pack (two launches into
    one): each program recomputes its request's exclusive page-count prefix
    sum (a kernel cannot share it across the grid), and the ``pid_blk == 0``
    column persists the metadata triple for the plans and host readers.
    ``BLOCK_J`` is sized from ``max_pages`` so the second grid dimension
    stays at 1-2 and the redundant prefix-sum reads stay within a small
    multiple of the standalone metadata kernel's traffic.
    """
    if bs == 0:
        return
    block_s = triton.next_power_of_2(max(bs, 16))
    block_j = min(triton.next_power_of_2(max(max_pages, 16)), 1024)
    grid = (bs, triton.cdiv(max_pages, block_j))
    _pack_len_meta_kernel[grid](
        out,
        req_to_token,
        req_pool_indices,
        seq_lens,
        kv_lens,
        tok_pages,
        page_starts,
        req_to_token.stride(0),
        n_phys_pages,
        PAGE_SIZE=_PAGE_SIZE,
        BLOCK_J=block_j,
        BLOCK_S=block_s,
    )


def _force_init_local_scores_into(
    max_score: torch.Tensor,  # fp32/bf16 [H, K, T]
    tok_pages: torch.Tensor,  # int32 [T]
    init_blocks: int,
    local_blocks: int,
) -> None:
    """Write init/local block scores into ``max_score``."""
    if init_blocks <= 0 and local_blocks <= 0:
        return
    assert max_score.is_contiguous(), "max_score must be contiguous"
    stride_h, stride_k, stride_t = max_score.stride()
    block = triton.next_power_of_2(max(init_blocks, local_blocks, 1))
    grid = (max_score.shape[2], max_score.shape[0])
    _force_init_local_scores_kernel[grid](
        max_score,
        tok_pages,
        stride_h,
        stride_k,
        stride_t,
        init_blocks,
        local_blocks,
        INIT_SCORE=_INIT_SCORE,
        LOCAL_SCORE=_LOCAL_SCORE,
        BLOCK=block,
    )
