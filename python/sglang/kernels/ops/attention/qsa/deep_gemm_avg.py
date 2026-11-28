"""Optional SAIL compressed-output FP8 AVG-MQA scoring."""

from __future__ import annotations

import functools
import math
from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.utils.log_utils import print_info_once, print_warning_once


def _validate_q(q: torch.Tensor) -> None:
    if q.ndim != 3 or q.shape[1] <= 0 or q.shape[2] <= 0:
        raise ValueError(f"QSA requires q [tokens, heads, head_dim], got {q.shape}")


def _validate_k(k: torch.Tensor) -> None:
    if k.ndim != 3 or k.shape[1] != 1 or k.shape[2] <= 0:
        raise ValueError(f"QSA MQA requires k [tokens, 1, head_dim], got {k.shape}")


_FP8_E4M3_FMAX = 448.0
# PPU avg-MQA supports logical head dimensions 64/128, not 32.
_FP8_AVG_HEAD_DIMS = (64, 128)


@functools.lru_cache(maxsize=1)
def _resolve_fp8_avg_mqa_logits():
    """SAIL DeepGEMM fp8_mqa_avg_logits kernel; None when unavailable."""
    from sglang.srt.utils.device import get_device_sm, is_sail

    # Only enable the PPU0015 FP8 MMA implementation validated on 890P.
    # Other PPU generations and upstream CUDA DeepGEMM are separate targets.
    if not is_sail() or get_device_sm() != 89:
        print_warning_once(
            "SGLANG_SAIL_QSA_INDEXER_FP8_AVG_LOGITS requires PPU SM89 (890P); "
            "falling back to the TileLang/torch QSA indexer scoring."
        )
        return None
    try:
        from deep_gemm import fp8_mqa_avg_logits
    except ImportError:
        print_warning_once(
            "SGLANG_SAIL_QSA_INDEXER_FP8_AVG_LOGITS is set but the installed "
            "deep_gemm has no fp8_mqa_avg_logits; falling back to the "
            "TileLang/torch QSA indexer scoring."
        )
        return None
    signature = fp8_mqa_avg_logits.__doc__ or ""
    if "max_seqlen_k" not in signature or "clean_logits" in signature:
        print_warning_once(
            "QSA requires the compressed-output DeepGEMM AVG API with mandatory "
            "max_seqlen_k and no clean_logits argument; falling back to "
            "TileLang/torch for an unrecognized or legacy API."
        )
        return None
    return fp8_mqa_avg_logits


def can_use_deep_gemm_fp8_avg_mqa(q: torch.Tensor) -> bool:
    """Whether QSA may use the PPU compressed-output scorer for these queries."""
    return (
        q.is_cuda
        and envs.SGLANG_SAIL_QSA_INDEXER_FP8_AVG_LOGITS.get()
        and q.shape[1] == 4
        and q.shape[-1] in _FP8_AVG_HEAD_DIMS
        and _resolve_fp8_avg_mqa_logits() is not None
    )


def _fp8_e4m3_per_row_quant(x: torch.Tensor):
    """[rows, dim] -> (fp8_e4m3 [rows, dim], fp32 dequant scale [rows]).

    On CUDA this reuses the in-tree JIT per-token quant kernel (single
    launch, warp per row); the torch fallback keeps the CPU reference path.
    """

    if x.is_cuda:
        from sglang.kernels.ops.quantization.per_token_quant_fp8 import (
            per_token_quant_fp8,
        )

        x = x.contiguous()
        x_fp8 = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        scale = torch.empty(x.shape[0], 1, dtype=torch.float32, device=x.device)
        per_token_quant_fp8(x, x_fp8, scale)
        return x_fp8, scale.squeeze(1)
    xf = x.float()
    scale = xf.abs().amax(dim=-1).clamp_min(1e-12) / _FP8_E4M3_FMAX
    return (xf / scale.unsqueeze(-1)).to(torch.float8_e4m3fn), scale


def deep_gemm_fp8_avg_mqa_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    score_scale: Optional[float] = None,
    *,
    max_seqlen_k: int,
) -> torch.Tensor:
    """SAIL DeepGEMM avg-MQA logits for the QSA indexer prefill scoring.

    Computes sum_h(relu(q·k)) * q_scale * k_scale / sqrt(head_dim): q is
    quantized per query row and the compressed keys per token, with the
    dequant scales passed through the kernel's q_scale / k_scales arguments,
    so the result matches the BF16 weight-free formula up to FP8 rounding.
    The kernel hardcodes the 1/sqrt(head_dim) score scale.

    Returns FP32 [rows, max_seqlen_k] with possibly padded row stride.
    Each row's valid prefix has length row_end - row_start; column j refers
    to K[row_start + j]. All other entries are unspecified. The positive,
    host-known capacity must cover every valid window (in compressed K units).
    """

    fp8_mqa_avg_logits = _resolve_fp8_avg_mqa_logits()
    if fp8_mqa_avg_logits is None:
        raise RuntimeError("fp8_mqa_avg_logits is unavailable")
    if not isinstance(max_seqlen_k, int) or max_seqlen_k <= 0:
        raise ValueError("compressed AVG requires a positive integer max_seqlen_k")
    _validate_q(q)
    _validate_k(k)
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("QSA query and key head dimensions must match")
    head_dim = q.shape[-1]
    if q.shape[1] != 4 or head_dim not in _FP8_AVG_HEAD_DIMS:
        raise ValueError(
            "the validated fp8_mqa_avg_logits path requires 4 query heads "
            "and head_dim 64/128"
        )
    if score_scale is not None and not math.isclose(score_scale, math.sqrt(head_dim)):
        raise ValueError(
            "fp8_mqa_avg_logits hardcodes score_scale = sqrt(head_dim), "
            f"got {score_scale} for head_dim {head_dim}"
        )
    rows, keys = q.shape[0], k.shape[0]
    if not rows or not keys:
        return torch.empty((rows, max_seqlen_k), dtype=torch.float32, device=q.device)
    q_fp8, q_scale = _fp8_e4m3_per_row_quant(q.reshape(rows, -1))
    q_fp8 = q_fp8.view(rows, q.shape[1], head_dim)
    k_fp8, k_scales = _fp8_e4m3_per_row_quant(k[:, 0])
    print_info_once(
        "Using SAIL DeepGEMM fp8_mqa_avg_logits for QSA prefill scoring "
        "(FP8 Q/K, FP32 compressed logits, bounded strided top-k)."
    )
    return fp8_mqa_avg_logits(
        q_fp8.contiguous(),
        (k_fp8.contiguous(), k_scales.contiguous()),
        row_starts.to(device=q.device, dtype=torch.int32).contiguous(),
        row_ends.to(device=q.device, dtype=torch.int32).contiguous(),
        max_seqlen_k=max_seqlen_k,
        q_scale=q_scale.contiguous(),
        logits_dtype=torch.float32,
    )
