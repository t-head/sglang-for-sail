"""Bridge compressed PPU MQA output to the global-column DSA contract."""

import torch
import triton
import triton.language as tl


@triton.jit
def _restore_ragged_logits(
    Src,
    Dst,
    Starts,
    Ends,
    SRC_ROW: tl.constexpr,
    SRC_COL: tl.constexpr,
    DST_ROW: tl.constexpr,
    WIDTH: tl.constexpr,
    SRC_WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    start = tl.load(Starts + row)
    end = tl.load(Ends + row)
    relative = col - start
    valid = (col < WIDTH) & (col < end) & (relative >= 0) & (relative < SRC_WIDTH)
    value = tl.load(Src + row * SRC_ROW + relative * SRC_COL, valid, -float("inf"))
    tl.store(Dst + row * DST_ROW + col, value, col < WIDTH)


def restore_ragged_logits(
    compressed: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    num_columns: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Move each valid [0, end-start) row to [start, end), filling holes with -inf."""
    if compressed.ndim != 2 or starts.ndim != 1 or ends.shape != starts.shape:
        raise ValueError("Expected matrix logits and matching one-dimensional bounds.")
    if starts.shape[0] != compressed.shape[0] or num_columns < 0:
        raise ValueError(
            "Logit rows and bounds must agree; column count must be nonnegative."
        )
    out = torch.empty(
        (compressed.shape[0], num_columns), device=compressed.device, dtype=dtype
    )
    if out.numel():
        _restore_ragged_logits[(out.shape[0], triton.cdiv(num_columns, 256))](
            compressed,
            out,
            starts,
            ends,
            compressed.stride(0),
            compressed.stride(1),
            out.stride(0),
            num_columns,
            compressed.shape[1],
            256,
        )
    return out
