"""V4.1 INT8 indexer: per-head Q scales, per-token K scales, no Hadamard.

Cache pages match the release C4 INT8 layout: [page_size * 128 INT8 bytes |
page_size * 4 FP32 scale bytes]. Q scales are folded into the head weights
before the ReLU-weighted head reduction, as in the release INT8 indexer.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from sglang.kernels.ops.quantization.int8_kernel import per_token_quant_int8
from sglang.srt.layers.attention.mqa_logits_utils import (
    mqa_logits_row_bytes,
    sail_paged_mqa_logits,
)
from sglang.srt.utils import is_hip

_IS_HIP = is_hip()
INDEX_K_INT8_PAYLOAD_BYTES = 128
INDEX_K_INT8_SCALE_BYTES = 4
INDEX_K_INT8_SLOT_BYTES = INDEX_K_INT8_PAYLOAD_BYTES + INDEX_K_INT8_SCALE_BYTES


def quantize_query(q, weights):
    assert q.shape[-1] == 128
    q = q.contiguous()
    quant, scale = per_token_quant_int8(q.flatten(0, 1))
    return quant.view_as(q), weights.float() * scale.view(q.shape[:2])


@triton.jit
def _quantize_row_int8(value, IS_HIP: tl.constexpr):
    absmax = tl.maximum(tl.max(tl.abs(value), axis=0), 1e-10)
    scale = absmax / 127.0
    q_val = value * (127.0 / absmax)
    if IS_HIP:
        rounded = libdevice.round(q_val)
    else:
        rounded = tl.extra.cuda.libdevice.round(q_val)
    q_int8 = tl.minimum(tl.maximum(rounded, -127.0), 127.0).to(tl.int8)
    return q_int8, scale


@triton.jit
def _rope_tail_pairs(offsets, RD: tl.constexpr):
    tail = offsets >= 128 - RD
    # Keep the frequency-table offset in bounds even for masked lanes.
    pair = tl.where(tail, (offsets - (128 - RD)) // 2, 0)
    return tail, pair


@triton.jit
def _norm_rope_int8_row(
    X,
    W,
    F,
    row,
    EPS,
    F_STRIDE: tl.constexpr,
    RD: tl.constexpr,
    IS_HIP: tl.constexpr,
):
    offsets = tl.arange(0, 128)
    value = tl.load(X + row * 128 + offsets).to(tl.float32)
    weight = tl.load(W + offsets).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(value * value, axis=0) / 128 + EPS)
    value = ((value * inv_rms) * weight).to(tl.bfloat16).to(tl.float32)

    if RD > 0:
        tail, pair = _rope_tail_pairs(offsets, RD)
        imaginary = (offsets % 2) == 1
        real = tl.load(X + row * 128 + 128 - RD + 2 * pair, tail, 0).to(tl.float32)
        imag = tl.load(X + row * 128 + 128 - RD + 2 * pair + 1, tail, 0).to(tl.float32)
        w_real = tl.load(W + 128 - RD + 2 * pair, tail, 0.0).to(tl.float32)
        w_imag = tl.load(W + 128 - RD + 2 * pair + 1, tail, 0.0).to(tl.float32)
        real = ((real * inv_rms) * w_real).to(tl.bfloat16).to(tl.float32)
        imag = ((imag * inv_rms) * w_imag).to(tl.bfloat16).to(tl.float32)
        fr = tl.load(F + row * F_STRIDE + 2 * pair, tail, 1.0)
        fi = tl.load(F + row * F_STRIDE + 2 * pair + 1, tail, 0.0)
        rotated = tl.where(imaginary, real * fi + imag * fr, real * fr - imag * fi)
        value = tl.where(tail, rotated.to(tl.bfloat16).to(tl.float32), value)

    return _quantize_row_int8(value, IS_HIP=IS_HIP)


@triton.jit
def _store_index_k_norm_rope_int8_kernel(
    X,
    W,
    F,
    OutK,
    OutScale,
    Cache,
    Loc,
    EPS,
    F_STRIDE: tl.constexpr,
    RD: tl.constexpr,
    STORE_CACHE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    CACHE_STRIDE: tl.constexpr,
    IS_HIP: tl.constexpr,
):
    """Fused index-K RMSNorm, RoPE, per-token INT8 quant, and paged cache store."""
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, 128)
    if STORE_CACHE:
        slot = tl.load(Loc + row).to(tl.int64)
        if slot > 0:
            q_int8, scale = _norm_rope_int8_row(
                X, W, F, row, EPS, F_STRIDE=F_STRIDE, RD=RD, IS_HIP=IS_HIP
            )
            page = slot // PAGE_SIZE
            off = slot % PAGE_SIZE
            tl.store(
                Cache + page * CACHE_STRIDE + off * 128 + offsets,
                q_int8.to(tl.uint8, bitcast=True),
            )
            scale_bits = scale.to(tl.int32, bitcast=True)
            byte = tl.arange(0, 4)
            tl.store(
                Cache + page * CACHE_STRIDE + PAGE_SIZE * 128 + off * 4 + byte,
                (scale_bits >> (8 * byte)) & 255,
            )
    else:
        q_int8, scale = _norm_rope_int8_row(
            X, W, F, row, EPS, F_STRIDE=F_STRIDE, RD=RD, IS_HIP=IS_HIP
        )
        tl.store(OutK + row * 128 + offsets, q_int8)
        tl.store(OutScale + row, scale)


def store_index_k_norm_rope_int8(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    freqs: torch.Tensor,
    rope_dim: int,
    *,
    loc: torch.Tensor | None = None,
    cache: torch.Tensor | None = None,
    num_warps: int = 1,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Index-K write for one 128-wide row per token: RMSNorm, RoPE, per-token
    INT8 quantization and paged cache write in one launch.

    ``x`` is ``wk(latent)`` before ``k_norm``; ``freqs`` is already gathered per
    token. With ``cache``, slots with ``loc <= 0`` are skipped (preserving slot 0
    and incomplete c2 pairs). Without ``cache``, returns ``(k_int8, k_scales)``.
    """
    assert x.dtype == torch.bfloat16 and x.shape[-1] == 128
    assert 0 <= rope_dim <= 128 and rope_dim % 2 == 0
    assert norm_weight.shape == (128,) and norm_weight.is_contiguous()
    x = x.contiguous()
    rows = x.numel() // 128
    f = torch.view_as_real(freqs.contiguous()).flatten(-2) if rope_dim > 0 else None
    f_stride = f.stride(0) if f is not None else 0
    if cache is None:
        out_k = torch.empty((rows, 128), dtype=torch.int8, device=x.device)
        out_scale = torch.empty((rows,), dtype=torch.float32, device=x.device)
        page_size = cache_stride = 0
    else:
        assert loc is not None and loc.numel() == rows
        assert cache.ndim == 2 and cache.shape[1] % INDEX_K_INT8_SLOT_BYTES == 0
        out_k = out_scale = None
        page_size = cache.shape[1] // INDEX_K_INT8_SLOT_BYTES
        cache_stride = cache.stride(0)
    if rows:
        _store_index_k_norm_rope_int8_kernel[(rows,)](
            x,
            norm_weight,
            f,
            out_k,
            out_scale,
            cache,
            loc,
            eps,
            F_STRIDE=f_stride,
            RD=rope_dim,
            STORE_CACHE=cache is not None,
            PAGE_SIZE=page_size,
            CACHE_STRIDE=cache_stride,
            IS_HIP=_IS_HIP,
            num_warps=num_warps,
        )
    return (out_k, out_scale) if cache is None else None


@triton.jit
def _index_q_rope_int8_quant_weights_kernel(
    Q,
    F,
    Pos,
    W,
    OutQ,
    OutW,
    W_SCALE,
    F_STRIDE: tl.constexpr,
    HEADS: tl.constexpr,
    RD: tl.constexpr,
    IS_HIP: tl.constexpr,
):
    """Fused Q RoPE (gathering freqs[positions]), per-head INT8 quant, and
    head-weight scaling + Q-scale folding."""
    row = tl.program_id(0).to(tl.int64)
    token = row // HEADS
    offsets = tl.arange(0, 128)
    value = tl.load(Q + row * 128 + offsets).to(tl.float32)

    if RD > 0:
        pos = tl.load(Pos + token).to(tl.int64)
        tail, pair = _rope_tail_pairs(offsets, RD)
        imaginary = (offsets % 2) == 1
        real = tl.load(Q + row * 128 + 128 - RD + 2 * pair, tail, 0).to(tl.float32)
        imag = tl.load(Q + row * 128 + 128 - RD + 2 * pair + 1, tail, 0).to(tl.float32)
        fr = tl.load(F + pos * F_STRIDE + 2 * pair, tail, 1.0)
        fi = tl.load(F + pos * F_STRIDE + 2 * pair + 1, tail, 0.0)
        rotated = tl.where(imaginary, real * fi + imag * fr, real * fr - imag * fi)
        value = tl.where(tail, rotated.to(tl.bfloat16).to(tl.float32), value)

    q_int8, q_scale = _quantize_row_int8(value, IS_HIP=IS_HIP)
    tl.store(OutQ + row * 128 + offsets, q_int8)

    # Preserve head_weights(x) rounding in W's dtype before folding q_scale in fp32.
    w_raw = tl.load(W + row)
    w_scaled = (w_raw.to(tl.float32) * W_SCALE).to(W.dtype.element_ty).to(tl.float32)
    tl.store(OutW + row, (w_scaled * q_scale).to(OutW.dtype.element_ty))


def index_q_rope_int8_quant_weights(
    q_raw: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    rope_dim: int,
    head_weights_raw: torch.Tensor,
    weight_scale: float,
    *,
    weights_dtype: torch.dtype = torch.float32,
    num_warps: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused Q RoPE + per-head INT8 quantization + head-weight scale folding.

    ``q_raw`` is ``[T, H, 128]`` bf16 from ``wq_b``; ``freqs_cis`` is the full
    complex frequency table ``[max_pos, rope_dim // 2]`` indexed by ``positions``
    ``[T]``; ``head_weights_raw`` is ``[T, H]`` before ``weight_scale``.
    Returns contiguous ``(q_int8 [T, H, 128], weights [T, H] in weights_dtype)``.
    """
    assert q_raw.ndim == 3 and q_raw.shape[-1] == 128
    assert q_raw.dtype == torch.bfloat16
    assert 0 <= rope_dim <= 128 and rope_dim % 2 == 0
    num_tokens, num_heads, _ = q_raw.shape
    assert positions.shape == (num_tokens,)
    assert head_weights_raw.shape == (num_tokens, num_heads)

    q_raw = q_raw.contiguous()
    positions = positions.contiguous()
    head_weights_raw = head_weights_raw.contiguous()
    out_q = torch.empty(
        (num_tokens, num_heads, 128), dtype=torch.int8, device=q_raw.device
    )
    out_w = torch.empty(
        (num_tokens, num_heads), dtype=weights_dtype, device=q_raw.device
    )
    if num_tokens == 0:
        return out_q, out_w

    f = torch.view_as_real(freqs_cis.contiguous()).flatten(-2) if rope_dim > 0 else None
    f_stride = f.stride(0) if f is not None else 0
    _index_q_rope_int8_quant_weights_kernel[(num_tokens * num_heads,)](
        q_raw,
        f,
        positions,
        head_weights_raw,
        out_q,
        out_w,
        weight_scale,
        F_STRIDE=f_stride,
        HEADS=num_heads,
        RD=rope_dim,
        IS_HIP=_IS_HIP,
        num_warps=num_warps,
    )
    return out_q, out_w


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
    _store_int8_k[(input.shape[0],)](
        k, scales, cache, loc, P=page_size, STRIDE=cache.stride(0)
    )


@triton.jit
def _gather_int8_index_k_kernel(
    cache,
    cache_stride,
    slots,
    payload,
    scales,
    n,
    page_size: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    # int32 view of a page: page_size * 32 payload words, then page_size scale words.
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    valid = rows < n
    slot = tl.load(slots + rows, valid, 0).to(tl.int64)
    page = slot // page_size
    offset = slot - page * page_size
    page_base = cache + page * cache_stride
    words = tl.arange(0, 32)
    payload_words = tl.load(
        page_base[:, None] + offset[:, None] * 32 + words[None, :],
        valid[:, None],
    )
    tl.store(
        payload + rows[:, None].to(tl.int64) * 32 + words[None, :],
        payload_words,
        valid[:, None],
    )
    tl.store(
        scales + rows,
        tl.load(page_base + page_size * 32 + offset, valid),
        valid,
    )


def gather_int8_index_k(cache, slots, page_size):
    """Gather packed INT8 K and FP32 scales at ``slots``."""
    assert slots.ndim == 1 and cache.shape[1] == page_size * 132
    if cache.is_cuda and cache.stride(1) == 1:
        slots = slots.contiguous()
        n = slots.shape[0]
        words = cache.view(torch.int32)
        payload = torch.empty((n, 32), device=cache.device, dtype=torch.int32)
        scales = torch.empty((n,), device=cache.device, dtype=torch.int32)
        if n > 0:
            block_rows = 64
            _gather_int8_index_k_kernel[(triton.cdiv(n, block_rows),)](
                words,
                words.stride(0),
                slots,
                payload,
                scales,
                n,
                page_size,
                BLOCK_ROWS=block_rows,
                num_warps=4,
            )
        return payload.view(torch.int8), scales.view(torch.float32)
    pages, offsets = slots // page_size, slots % page_size
    k_pages = cache[:, : page_size * 128].view(cache.shape[0], page_size, 128)
    scale_pages = cache[:, page_size * 128 :].view(torch.float32)
    k = k_pages[pages, offsets].contiguous().view(torch.int8)
    scales = scale_pages[pages, offsets].contiguous()
    return k, scales


def int8_logits_rows_per_chunk(width, max_logits_bytes):
    return max(1, (max_logits_bytes // mqa_logits_row_bytes(width) // 4) * 4)


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
    q, weights, cache, lens, page_table, plan, width, page_size, rows_per_request=1
):
    """Paged INT8 logits for flattened query rows, including multi-row MTP.

    Weighted PPU kernels require next_n=1: each query keeps its own context
    length and block-table row. rows_per_request remains a caller hint, not
    a kernel grouping; the adapter rebuilds the schedule for the actual rows.
    """
    assert q.shape[-1] == 128 and cache.shape[1] == page_size * 132
    q_int8, w = quantize_query(q, weights)
    return sail_paged_mqa_logits(
        q_int8.unsqueeze(1),
        cache.view(cache.shape[0], page_size, 1, 132),
        w.contiguous(),
        lens.to(torch.int32).reshape(-1, 1),
        page_table,
        plan,
        width,
    ).float()
