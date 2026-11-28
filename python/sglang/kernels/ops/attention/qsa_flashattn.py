import torch
import triton
import triton.language as tl


@triton.jit
def _prepare_qsa(
    Q, Indices, Table, RowBatch, ReqIds, SeqLens, Q8, Slots, Counts,
    QWIDTH: tl.constexpr, WIDTH: tl.constexpr, CONTEXT: tl.constexpr,
    CACHE_ROWS: tl.constexpr, MAPPED: tl.constexpr, FP8: tl.constexpr,
    QBLOCK: tl.constexpr, IBLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    if FP8:
        d = tl.arange(0, QBLOCK)
        q = tl.load(Q + row * QWIDTH + d, d < QWIDTH, other=0)
        tl.store(Q8 + row * QWIDTH + d,
                 q.to(tl.float8e4nv, fp_downcast_rounding="rtne"), d < QWIDTH)
    col = tl.arange(0, IBLOCK)
    idx = tl.load(Indices + row * WIDTH + col, col < WIDTH, other=-1)
    valid = (col < WIDTH) & (idx >= 0)
    if MAPPED:
        batch = tl.load(RowBatch + row)
        req = tl.load(ReqIds + batch).to(tl.int64)
        length = tl.load(SeqLens + batch)
        valid &= (idx < length) & (idx < CONTEXT)
        idx = tl.load(Table + req * CONTEXT + tl.maximum(idx, 0), valid, other=-1)
    valid &= (idx >= 0) & (idx < CACHE_ROWS)
    dest = tl.cumsum(valid.to(tl.int32)) - 1
    tl.store(Slots + row * WIDTH + dest, idx, valid)
    tl.store(Counts + row, tl.sum(valid.to(tl.int32)))


def sparse_qsa_flashattn(
    q, k_cache, v_cache, indices, softmax_scale=None, kv_head_base=-1, *,
    req_to_token=None, row_to_batch=None, req_indices=None, seq_lens=None,
    flash_attn_with_kvcache=None,
):
    """BF16 or unit-scale Q8KV8 attention over physical or logical token slots.

    The indexer supplies causal selections; holes are compacted before PR39.
    Each query is a one-token sequence, with its own sparse token-slot table.
    Mapping tensors stay live during CUDA graph replay.
    """
    if flash_attn_with_kvcache is None:
        from flash_attn_interface import flash_attn_with_kvcache

    if q.ndim != 3 or k_cache.ndim != 3 or v_cache.shape != k_cache.shape:
        raise ValueError("QSA expects Q [rows, heads, dim] and matching K/V caches")
    rows, heads, dim = q.shape
    kv_heads = k_cache.shape[1]
    if not (
        q.is_cuda and k_cache.device == v_cache.device == q.device
        and q.dtype == torch.bfloat16
        and k_cache.dtype == v_cache.dtype
        and k_cache.dtype in (torch.bfloat16, torch.float8_e4m3fn)
        and dim == k_cache.shape[-1] == 256
        and kv_heads > 0 and heads % kv_heads == 0
        and k_cache.shape[0] % 64 == 0
        and k_cache.is_contiguous() and v_cache.is_contiguous()
        and (kv_head_base == -1 or (kv_heads == 1 and kv_head_base == 0))
    ):
        raise ValueError("Unsupported FlashAttention QSA dtype, head mapping or page layout")
    if indices.ndim != 2 or indices.shape[0] != rows or indices.device != q.device:
        raise ValueError("QSA selections must have one row per query on the same device")
    mapping = (req_to_token, row_to_batch, req_indices, seq_lens)
    mapped = req_to_token is not None
    if any(x is not None for x in mapping) != all(x is not None for x in mapping):
        raise ValueError("Logical QSA addressing requires all four mapping tensors")
    if mapped and (
        req_to_token.ndim != 2 or not req_to_token.is_contiguous()
        or any(x.ndim != 1 or not x.is_contiguous() for x in mapping[1:])
        or any(x.device != q.device for x in mapping)
        or row_to_batch.numel() < rows
    ):
        raise ValueError("Invalid live QSA request mapping")
    if rows == 0 or indices.shape[1] == 0:
        return torch.zeros_like(q)
    q, indices = q.contiguous(), indices.contiguous()
    width = indices.shape[1]
    fp8 = k_cache.dtype == torch.float8_e4m3fn
    query = torch.empty_like(q, dtype=torch.float8_e4m3fn) if fp8 else q
    slots = torch.empty_like(indices, dtype=torch.int32)
    counts = torch.empty(rows, device=q.device, dtype=torch.int32)
    _prepare_qsa[(rows,)](
        q, indices, *mapping, query, slots, counts,
        heads * dim, width, req_to_token.shape[1] if mapped else 0,
        k_cache.shape[0], mapped, fp8,
        triton.next_power_of_2(heads * dim), triton.next_power_of_2(width),
        num_warps=4,
    )
    cu = torch.arange(rows + 1, device=q.device, dtype=torch.int32)
    scales = (
        torch.ones((rows, kv_heads), device=q.device, dtype=torch.float32)
        if fp8 else None
    )
    return flash_attn_with_kvcache(
        query, k_cache.view(-1, 64, kv_heads, dim),
        v_cache.view(-1, 64, kv_heads, dim),
        cache_seqlens=counts, page_table=slots[:, None, :],
        cu_seqlens_q=cu, max_seqlen_q=1, max_seqlen_k=width,
        softmax_scale=softmax_scale if softmax_scale is not None else dim ** -0.5,
        causal=True, num_splits=0, pack_gqa=True, qsa_allow_aiu=False,
        q_descale=scales, k_descale=scales, v_descale=scales,
    )
