"""Native PPU FlashMLA sparse attention over raw E4M3 tensors."""

from __future__ import annotations

from typing import Optional

import torch

try:
    import flash_mla as _flashmla

    _import_error = None
except Exception as _e:
    _flashmla = None
    _import_error = _e

_IMPORT_ERROR = ImportError(
    "Failed to import flash_mla for PPU. "
    "Ensure the flash_mla package is installed in your PPU environment."
)
_validated = False


def require_sparse_fp8_attention() -> None:
    """Validate the FlashMLA-for-sail dependency and PPU0015 runtime."""
    global _validated
    if _validated:
        return
    if _import_error is not None:
        raise _IMPORT_ERROR from _import_error
    if not getattr(torch.version, "ppu", None):
        raise RuntimeError("PPU flashmla_sparse_q8 requires a SAIL Torch build.")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9):
        raise RuntimeError("PPU flashmla_sparse_q8 requires PPU0015 (SM89).")
    if not all(
        callable(getattr(_flashmla, name, None))
        for name in ("flash_mla_sparse_quantized_fwd", "QuantizedAttentionConfig")
    ):
        raise RuntimeError(
            "PPU flashmla_sparse_q8 requires a FlashMLA wheel with "
            "flash_mla_sparse_quantized_fwd and QuantizedAttentionConfig."
        )
    _validated = True


def flash_mla_sparse_quantized_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    topk_length: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run native PPU sparse attention over raw E4M3 Q and KV."""
    require_sparse_fp8_attention()
    out, _, _ = _flashmla.flash_mla_sparse_quantized_fwd(
        q,
        kv,
        indices,
        sm_scale,
        quantization=_flashmla.QuantizedAttentionConfig(q_scale=1.0, kv_scale=1.0),
        d_v=d_v,
        attn_sink=None,
        topk_length=topk_length,
        out=None,
        compute_aux=False,
    )
    return out
