"""Per-32 MXFP4 activation quantization in the SAIL DeepGEMM layout."""

from functools import lru_cache

import torch
import triton
import triton.language as tl


@triton.jit
def _quantize(
    src,
    packed,
    scales,
    rows,
    width: tl.constexpr,
    src_stride,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    col_start = tl.program_id(1) * BLOCK_K
    col = col_start + tl.arange(0, BLOCK_K)
    values = tl.load(
        src + row[:, None] * src_stride + col[None, :],
        (row[:, None] < rows) & (col[None, :] < width),
        other=0,
    ).to(tl.float32)
    groups = values.reshape(BLOCK_M, BLOCK_K // 32, 32)
    absmax = tl.maximum(tl.max(tl.abs(groups), 2), 1e-10)

    # Round max/6 up to the next E8M0 power of two as in the source PPU path.
    exponent = ((absmax / 6.0).to(tl.uint32, bitcast=True) + 0x007FFFFF) & 0x7F800000
    quantized = groups / exponent.to(tl.float32, bitcast=True)[:, :, None]
    lo, hi = tl.split(quantized.reshape(BLOCK_M, BLOCK_K // 2, 2))
    e2m1 = tl.inline_asm_elementwise(
        """
        {
            .reg .b8 r;
            cvt.rn.satfinite.e2m1x2.f32 r, $1, $2;
            mov.b32 $0, {r, r, r, r};
        }
        """,
        constraints="=r,f,f",
        args=[hi, lo],
        dtype=tl.uint8,
        is_pure=True,
        pack=1,
    )
    packed_col = col_start // 2 + tl.arange(0, BLOCK_K // 2)
    tl.store(
        packed + row[:, None] * (width // 2) + packed_col[None, :],
        e2m1,
        (row[:, None] < rows) & (packed_col[None, :] < width // 2),
    )

    # DeepGEMM expects two exponent bytes per uint16, with rows contiguous.
    lo, hi = tl.split(exponent.reshape(BLOCK_M, BLOCK_K // 64, 2))
    scale_col = col_start // 64 + tl.arange(0, BLOCK_K // 64)
    tl.store(
        scales + scale_col[None, :] * rows + row[:, None],
        ((lo >> 23) | ((hi >> 23) << 8)).to(tl.uint16),
        (row[:, None] < rows) & (scale_col[None, :] < width // 64),
    )


@lru_cache(maxsize=None)
def _supports_e2m1(device: torch.device) -> bool:
    return (
        device.type == "cuda"
        and "ZW" in torch.cuda.get_device_name(device)
        and torch.cuda.get_device_capability(device) >= (8, 9)
    )


def sail_mxfp4_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize BF16 [M, K] to packed FP4 and column-major E8M0 scale pairs."""
    if not _supports_e2m1(x.device):
        raise ValueError("SAIL MXFP4 quantization requires a SAIL SM89+ device")
    if (
        x.ndim != 2
        or x.dtype != torch.bfloat16
        or x.stride(1) != 1
        or x.shape[1] == 0
        or x.shape[1] % 64
    ):
        raise ValueError("Expected BF16 [M, K] with contiguous columns and K % 64 == 0")

    rows, width = x.shape
    packed = x.new_empty((rows, width // 2), dtype=torch.uint8)
    scales = x.new_empty((width // 64, rows), dtype=torch.uint16)
    if rows:
        block_m, block_k = 32, 128
        _quantize[(triton.cdiv(rows, block_m), triton.cdiv(width, block_k))](
            x, packed, scales, rows, width, x.stride(0), block_m, block_k
        )
    return packed, scales.t()
