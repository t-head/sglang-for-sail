# SPDX-License-Identifier: Apache-2.0
"""Step5 region selection through the SAIL JIT radix selector.

The selector ranks only the live history prefix of each score row and writes the
current region into its last output slot, which is exactly what the portable
pack kernel consumes as its ``indices`` input. Scratch is caller-owned so graph
capture keeps fixed addresses.
"""

import functools
from typing import TYPE_CHECKING

import msgspec
import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# Step5 compresses eight tokens per region and stores them in page-64 KV pages.
STEP5_SELECTOR_REGION_BLOCK = 8
STEP5_SELECTOR_PAGE_SIZE = 64
STEP5_SELECTOR_TOPK = 512
# The selector streams above this width and needs a chunk-candidate workspace.
STEP5_SELECTOR_STREAMING_REGIONS = 16384


class Step5PpuSelectorWorkspace(msgspec.Struct, frozen=True, kw_only=True):
    """Per-tile radix selector scratch, sized once by the selection tile.

    ``history`` doubles as the pack kernel's ``indices`` input: its row stride is
    the selector's output width, so slots [0, topk) hold the ranked history and
    slot topk holds the current region.
    """

    history: torch.Tensor  # [rows, 513] int32, selector output
    starts: torch.Tensor  # [rows, 513] int32, logical starts, unused by the pack step
    candidates: torch.Tensor  # [rows, 1, chunks * topk] int32, or one column
    positions64: torch.Tensor  # [rows] int64, converted positions when needed
    empty_pages: torch.Tensor  # [1] int32, block table placeholder for unpacked selection


def ppu_selector_chunks(width: int) -> int:
    return (width + STEP5_SELECTOR_STREAMING_REGIONS - 1) // (
        STEP5_SELECTOR_STREAMING_REGIONS
    )


def allocate_step5_ppu_selector_workspace(
    *, rows: int, width: int, device
) -> Step5PpuSelectorWorkspace:
    if rows <= 0 or width <= 0:
        raise ValueError("Step5 PPU selector requires positive rows and width")
    chunks = ppu_selector_chunks(width)
    return Step5PpuSelectorWorkspace(
        history=torch.empty((rows, 513), dtype=torch.int32, device=device),
        starts=torch.empty((rows, 513), dtype=torch.int32, device=device),
        candidates=torch.empty(
            (rows, 1, max(chunks, 1) * STEP5_SELECTOR_TOPK),
            dtype=torch.int32,
            device=device,
        ),
        positions64=torch.empty(rows, dtype=torch.int64, device=device),
        empty_pages=torch.zeros(1, dtype=torch.int32, device=device),
    )


@cache_once
def _jit_step5_radix_module() -> "Module":
    return load_jit(
        "step5_dsa_radix_select",
        cuda_files=["step5_dsa/dsa_radix_selector.cuh"],
        cuda_wrappers=[("radix_select", "step5_dsa::radix_select")],
    )


@functools.cache
def _selector_op_available() -> bool:
    """Whether the JIT radix selector module can build on this machine."""
    try:
        _jit_step5_radix_module()
    except Exception:
        return False
    return True


@functools.cache
def _is_ppu_device(device_index: int | None) -> bool:
    """Whether the device reports the PPU vendor string."""
    return "ZW" in torch.cuda.get_device_name(device_index)


def can_use_step5_selection_ppu(scores: torch.Tensor) -> bool:
    """Whether the JIT radix selector can rank this score tile on this device."""
    if scores.device.type != "cuda" or torch.version.hip is not None:
        return False
    if scores.dtype != torch.float32 or scores.ndim != 2 or not scores.is_contiguous():
        return False
    if not 1 <= scores.shape[1] < 2**31:
        return False
    if not _is_ppu_device(scores.device.index):
        return False
    return _selector_op_available()


def step5_select_regions_ppu(
    scores: torch.Tensor,
    *,
    positions: torch.Tensor,
    workspace: Step5PpuSelectorWorkspace,
) -> torch.Tensor:
    """Select the live history regions of each row into the workspace history.

    Returns the int32 [rows, 513] selector output, where slots [0, min(topk,
    position // 8)) hold the selected region ids in an unspecified order and
    slot topk holds the current region. The set is stable across runs; the slot
    order is not, because the staging pass hands out slots with atomics. The
    caller still appends the current region and computes the packed addresses,
    because both need request-slot and causal-length metadata the selector does
    not read.
    """
    rows, width = scores.shape
    if rows > workspace.history.shape[0]:
        raise ValueError("Step5 PPU selector workspace is smaller than the tile")
    if positions.shape != (rows,):
        raise ValueError("Step5 PPU selector requires one position per row")
    out = workspace.history[:rows]
    starts = workspace.starts[:rows]
    candidates = workspace.candidates[:rows]
    if positions.dtype == torch.int64:
        cast_positions = positions
    else:
        cast_positions = workspace.positions64[:rows]
        cast_positions.copy_(positions)
    # The selector reads block_table only when packing; the pack step below owns
    # the physical translation, so pass the placeholder page row.
    module = _jit_step5_radix_module()
    module.radix_select(
        scores.view(rows, 1, width),
        cast_positions,
        workspace.empty_pages,
        candidates,
        out.view(rows, 1, 513),
        starts.view(rows, 1, 513),
        STEP5_SELECTOR_REGION_BLOCK,
        STEP5_SELECTOR_PAGE_SIZE,
        STEP5_SELECTOR_TOPK,
        False,
    )
    return out
