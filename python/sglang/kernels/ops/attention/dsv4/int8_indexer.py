"""V4.1 INT8 indexer: per-head Q scales, per-token K scales, no Hadamard.

Cache pages match the release C4 INT8 layout: [page_size * 128 INT8 bytes |
page_size * 4 FP32 scale bytes]. Q scales are folded into the head weights
before the ReLU-weighted head reduction, as in the release INT8 indexer.
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.quantization.int8_kernel import per_token_quant_int8


def quantize_query(q, weights):
    assert q.shape[-1] == 128
    q = q.contiguous()
    quant, scale = per_token_quant_int8(q.flatten(0, 1))
    quant = quant.view_as(q)
    folded = weights.float() * scale.view(q.shape[:2])
    heads = q.shape[1]
    if heads < 32:
        # Weighted INT8 scoring supports 32/64 heads. Zero heads preserve the
        # weighted ReLU sum; the planner uses the same padded head count.
        quant = torch.nn.functional.pad(quant, (0, 0, 0, 32 - heads))
        folded = torch.nn.functional.pad(folded, (0, 32 - heads))
    elif heads not in (32, 64):
        raise ValueError("PPU INT8 indexer requires at most 32, or 64 heads")
    return quant, folded


@triton.jit
def _store_int8_k(k, scales, cache, loc, P: tl.constexpr, STRIDE: tl.constexpr):
    row = tl.program_id(0)
    slot = tl.load(loc + row).to(tl.int64)
    d = tl.arange(0, 128)
    page, off = slot // P, slot % P
    value = tl.load(k + row * 128 + d).to(tl.uint8, bitcast=True)
    tl.store(cache + page * STRIDE + off * 128 + d, value, mask=slot > 0)
    scale = tl.load(scales + row).to(tl.int32, bitcast=True)
    byte = tl.arange(0, 4)
    tl.store(
        cache + page * STRIDE + P * 128 + off * 4 + byte,
        (scale >> (8 * byte)) & 255,
        mask=slot > 0,
    )


def store_int8_index_k(input, cache, loc, page_size):
    assert input.shape[-1] == 128 and cache.shape[1] == page_size * 132
    assert input.shape[0] == loc.numel()
    if not input.shape[0]:
        return
    k, scales = per_token_quant_int8(input.contiguous())
    loc = loc.contiguous()
    _store_int8_k[(input.shape[0],)](
        k, scales, cache, loc, P=page_size, STRIDE=cache.stride(0)
    )


def gather_int8_index_k(cache, slots, page_size):
    """Gather one request's packed K and FP32 scales once for all query chunks."""
    assert slots.ndim == 1 and cache.shape[1] == page_size * 132
    pages, offsets = slots // page_size, slots % page_size
    k_pages = cache[:, : page_size * 128].view(cache.shape[0], page_size, 128)
    scale_pages = cache[:, page_size * 128 :].view(torch.float32)
    k = k_pages[pages, offsets].contiguous().view(torch.int8)
    scales = scale_pages[pages, offsets].contiguous()
    return k, scales


def int8_logits_rows_per_chunk(width, max_logits_bytes):
    # FP32 rows are 1024-byte aligned, with a guard KV tile. Budget
    # complete 32-row tiles conservatively across the native heuristics.
    stride = (width + 256 + 255) // 256 * 256
    return max(1, (max_logits_bytes // (stride * 4) // 32) * 32)


def int8_index_logits(q, weights, kv, lens):
    """DeepGEMM dense INT8 logits over a single request's gathered K rows.

    Q scales are folded into weights; K retains its per-token FP32 scales.
    The caller chunks query rows and masks columns outside each causal length.
    """
    from deep_gemm import int8_mqa_logits

    q_int8, w = quantize_query(q, weights)
    ends = lens.to(torch.int32).contiguous()
    starts = torch.zeros_like(ends)
    return int8_mqa_logits(
        q_int8,
        kv,
        w.contiguous(),
        starts,
        ends,
        clean_logits=False,
    )


def int8_paged_index_logits(
    q, weights, cache, lens, page_table, plan, width, page_size
):
    from deep_gemm import int8_paged_mqa_logits

    assert q.shape[-1] == 128 and cache.shape[1] == page_size * 132
    q_int8, w = quantize_query(q, weights)
    return int8_paged_mqa_logits(
        q_int8.unsqueeze(1),
        cache.view(cache.shape[0], page_size, 1, 132),
        w.contiguous(),
        lens.to(torch.int32).reshape(-1, 1),
        page_table,
        plan,
        width,
        clean_logits=False,
    ).float()
