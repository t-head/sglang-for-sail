"""Triton kernels for the opt-in Shared Host Engram path."""

from typing import Optional

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from sglang.kernels.ops.embeddings.engram_gather import _E8M0_ZERO, _e4m3fn_to_float
from sglang.srt.utils import get_device_sm, is_sail


@triton.jit
def _engram_shared_host_gather_kernel(
    w_ptr,
    s_ptr,
    ids_ptr,
    out_ptr,
    lut_ptr,
    row_hi,
    DIM: tl.constexpr,
    BLK: tl.constexpr,
    E8M0_ZERO: tl.constexpr,
    DECODE_E4M3_BYTES: tl.constexpr,
    PER_ROW_SCALE: tl.constexpr,
    INT8_PAYLOAD: tl.constexpr,
    SCALE_LUT: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    idx = tl.load(ids_ptr + row).to(tl.int64)
    owned = (idx >= 0) & (idx < row_hi)
    local = tl.where(owned, idx, 0)
    offs = tl.arange(0, DIM)
    if INT8_PAYLOAD:
        w = w_ptr.to(tl.int64).to(tl.pointer_type(tl.int8))
        vals = tl.load(w + local * DIM + offs, mask=owned, other=0).to(tl.float32)
    elif DECODE_E4M3_BYTES:
        w = w_ptr.to(tl.int64).to(tl.pointer_type(tl.uint8))
        payload = tl.load(w + local * DIM + offs, mask=owned, other=0)
        vals = _e4m3fn_to_float(payload)
    else:
        w = w_ptr.to(tl.int64).to(tl.pointer_type(tl.float8e4nv))
        vals = tl.load(w + local * DIM + offs, mask=owned, other=0.0).to(tl.float32)
    if PER_ROW_SCALE and SCALE_LUT:
        codes = s_ptr.to(tl.int64).to(tl.pointer_type(tl.uint8))
        code = tl.load(codes + local, mask=owned, other=0).to(tl.int32)
        out = tl.where(owned, vals * tl.load(lut_ptr + code), 0.0)
    elif PER_ROW_SCALE:
        sf = s_ptr.to(tl.int64).to(tl.pointer_type(tl.float32))
        out = tl.where(owned, vals * tl.load(sf + local, mask=owned, other=0.0), 0.0)
    else:
        s = s_ptr.to(tl.int64).to(tl.pointer_type(tl.uint8))
        exps = tl.load(s + local * (DIM // BLK) + offs // BLK, mask=owned, other=0).to(
            tl.int32
        )
        # 2**(e - 127) from the exponent bits: exact, no exp2/denormal flush.
        scale = (exps << 23).to(tl.float32, bitcast=True)
        scale = tl.where(exps == 0, E8M0_ZERO, scale)
        scale = tl.where(exps == 255, float("nan"), scale)
        out = tl.where(owned, vals * scale, 0.0)
    tl.store(out_ptr + row * DIM + offs, out.to(tl.bfloat16))


def engram_shared_host_gather(
    weight_ptr: int,
    scale_ptr: int,
    ids: torch.Tensor,
    out: torch.Tensor,
    dim: int,
    block_size: int,
    *,
    rows: int,
    per_row_scale: bool = False,
    int8_payload: bool = False,
    scale_lut: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Gather unsorted shared Host rows with two warps per row."""
    assert dim > 0 and dim & (dim - 1) == 0 and block_size > 0, (dim, block_size)
    assert dim % block_size == 0, (dim, block_size)
    assert (
        ids.is_cuda and ids.is_contiguous() and ids.dtype in (torch.int32, torch.int64)
    )
    assert out.is_contiguous() and out.dtype == torch.bfloat16
    assert out.device == ids.device and out.shape == (ids.numel(), dim), out.shape
    assert scale_lut is None or (
        per_row_scale
        and scale_lut.dtype == torch.float32
        and scale_lut.device == ids.device
    )
    n = ids.numel()
    if n:
        _engram_shared_host_gather_kernel[(n,)](
            weight_ptr,
            scale_ptr,
            ids,
            out,
            scale_lut if scale_lut is not None else out,
            rows,
            DIM=dim,
            BLK=block_size,
            E8M0_ZERO=_E8M0_ZERO,
            DECODE_E4M3_BYTES=is_sail() and get_device_sm() < 89,
            PER_ROW_SCALE=per_row_scale,
            INT8_PAYLOAD=int8_payload,
            SCALE_LUT=scale_lut is not None,
            num_warps=2,
        )
    return out


@triton.jit
def _engram_shared_host_gate_kernel(
    X,
    KV,
    QW,
    KW,
    O,
    D: tl.constexpr,
    HC: tl.constexpr,
    EPS: tl.constexpr,
    CLAMP: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0)
    token = row // HC
    hc = row % HC
    col = tl.arange(0, B)
    mask = col < D
    q_weight = tl.load(QW + hc * D + col, mask, 0).to(tl.float32)
    k_weight = tl.load(KW + hc * D + col, mask, 0).to(tl.float32)
    weight = q_weight * k_weight
    x = tl.load(X + row * D + col, mask, 0).to(tl.float32)
    xx = tl.sum(x * x, 0)
    xw = x * weight
    key = tl.load(KV + token * (HC + 1) * D + hc * D + col, mask, 0).to(tl.float32)
    rstd = tl.rsqrt(xx / D + EPS) * tl.rsqrt(tl.sum(key * key, 0) / D + EPS)
    dot = tl.sum(xw * key, 0) * rstd * (D**-0.5)
    gate = tl.sigmoid(libdevice.copysign(tl.sqrt(tl.maximum(tl.abs(dot), CLAMP)), dot))
    value = tl.load(KV + token * (HC + 1) * D + HC * D + col, mask, 0).to(tl.float32)
    tl.store(O + row * D + col, (x + gate * value).to(O.dtype.element_ty), mask)


def fused_engram_shared_host_gate(
    x: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    clamp_value: float,
) -> torch.Tensor:
    assert x.ndim == 3 and kv.ndim == 2
    t, hc, d = x.shape
    assert kv.shape == (t, (hc + 1) * d)
    assert q_weight.shape == k_weight.shape == (hc, d)
    assert all(
        a.is_cuda and a.is_contiguous() and a.device == x.device
        for a in (x, kv, q_weight, k_weight)
    )
    assert all(
        a.dtype in (torch.bfloat16, torch.float32) for a in (x, kv, q_weight, k_weight)
    )
    out = torch.empty_like(x)
    if t:
        _engram_shared_host_gate_kernel[(t * hc,)](
            x,
            kv,
            q_weight,
            k_weight,
            out,
            d,
            hc,
            eps,
            clamp_value,
            triton.next_power_of_2(d),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out
