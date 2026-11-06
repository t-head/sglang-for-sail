# SPDX-License-Identifier: Apache-2.0
"""Step5 sparse decode through the SAIL JIT paged attention kernel.

The JIT kernel owns the whole batch in one launch: each row walks its own
selected regions, split-KV partials merge inside the call, and the natural LSE
is reduced from the split states afterwards. SGLang stores the selection as
packed int64 words while the kernel indexes two int32 planes, so one unpack
launch per layer bridges the layouts. Scratch is caller-owned so graph capture
keeps fixed addresses.
"""

import functools
from typing import TYPE_CHECKING

import msgspec
import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# Step5 compresses eight tokens per region and stores them in page-64 KV pages.
STEP5_PPU_DECODE_REGION_TOKENS = 8
STEP5_PPU_DECODE_PAGE_SIZE = 64
# Decode keeps the 512 selected history regions plus the current one.
STEP5_PPU_DECODE_REGIONS = 513


class Step5PpuDecodeWorkspace(msgspec.Struct, frozen=True, kw_only=True):
    """Per-layer decode scratch sized once by the query layout.

    ``physical_regions`` and ``logical_starts`` receive the unpacked selection;
    the split buffers hold the (max, denominator, accumulator) states the kernel
    merges before this module reduces them to the natural LSE.
    """

    physical_regions: torch.Tensor  # [rows, 1, regions] int32
    logical_starts: torch.Tensor  # [rows, 1, regions] int32
    positions64: torch.Tensor  # [rows] int64, converted positions when needed
    split_max: torch.Tensor  # [rows, heads, splits] FP32
    split_denom: torch.Tensor  # [rows, heads, splits] FP32
    split_acc: torch.Tensor  # [rows, heads, splits, 192] FP32


def step5_ppu_decode_splits(rows: int, heads: int, kv_heads: int, regions: int) -> int:
    """Split count the launcher picks for this shape; 0 when it cannot run."""
    if rows <= 0 or regions <= 0 or kv_heads <= 0:
        return 0
    return _jit_step5_decode_module().num_splits(rows, heads, kv_heads, regions)


def allocate_step5_ppu_decode_workspace(
    *, rows: int, heads: int, kv_heads: int, regions: int, device
) -> Step5PpuDecodeWorkspace:
    if rows <= 0 or regions <= 0 or kv_heads <= 0:
        raise ValueError("Step5 PPU decode requires positive rows and regions")
    splits = step5_ppu_decode_splits(rows, heads, kv_heads, regions)
    if splits <= 0:
        raise ValueError("Step5 PPU decode has no split schedule for this shape")
    return Step5PpuDecodeWorkspace(
        physical_regions=torch.empty(
            (rows, 1, regions), dtype=torch.int32, device=device
        ),
        logical_starts=torch.empty(
            (rows, 1, regions), dtype=torch.int32, device=device
        ),
        positions64=torch.empty(rows, dtype=torch.int64, device=device),
        split_max=torch.empty(
            (rows, heads, splits), dtype=torch.float32, device=device
        ),
        split_denom=torch.empty(
            (rows, heads, splits), dtype=torch.float32, device=device
        ),
        split_acc=torch.empty(
            (rows, heads, splits, 192), dtype=torch.float32, device=device
        ),
    )


@cache_once
def _jit_step5_decode_module() -> "Module":
    return load_jit(
        "step5_dsa_decode",
        cuda_files=["step5_dsa/dsa_paged_decode_attn.cuh"],
        cuda_wrappers=[
            ("unpack_regions", "step5_dsa::unpack_packed_regions"),
            ("paged_decode", "step5_dsa::paged_decode"),
            ("num_splits", "step5_dsa::paged_decode_num_splits"),
        ],
    )


@functools.cache
def _decode_op_available() -> bool:
    """Whether the JIT paged decode module can build on this machine."""
    try:
        _jit_step5_decode_module()
    except Exception:
        return False
    return True


@functools.cache
def _is_ppu_device(device_index: int | None) -> bool:
    """Whether this PPU core carries the tc02 attention path.

    The tc02 ldmatrix and the 2d.tile AIU bulk copy exist only on the ppu0015
    core, which reports capability 8.9; the ZW810E (8.0) keeps the portable
    schedule.
    """
    if "ZW" not in torch.cuda.get_device_name(device_index):
        return False
    major, minor = torch.cuda.get_device_capability(device_index)
    return major * 10 + minor >= 89


def can_allocate_step5_ppu_decode(
    *, heads: int, kv_heads: int, device
) -> bool:
    """Whether the JIT paged decode kernel can serve this shape on this device.

    Only the shape and device are consulted, so the caller can size scratch
    before the first query arrives.
    """
    index = torch.device(device).index
    if index is None:
        index = torch.cuda.current_device()
    # The kernel is specialized for 8 or 16 query heads per KV head; the TP16
    # four-head layout keeps the Triton path.
    if kv_heads <= 0 or heads not in (kv_heads * 8, kv_heads * 16):
        return False
    if not _is_ppu_device(index):
        return False
    return _decode_op_available()


def can_use_step5_decode_ppu(q: torch.Tensor, k: torch.Tensor) -> bool:
    """Whether the JIT paged decode kernel can serve this query/KV pair."""
    if q.device.type != "cuda" or torch.version.hip is not None:
        return False
    if q.dtype != torch.bfloat16 or q.ndim != 3 or q.shape[2] != 192:
        return False
    if k.dtype != torch.bfloat16 or k.ndim != 3 or tuple(k.shape[1:]) != (1, 192):
        return False
    if q.shape[0] == 0 or k.shape[0] % STEP5_PPU_DECODE_PAGE_SIZE:
        return False
    return can_allocate_step5_ppu_decode(
        heads=q.shape[1], kv_heads=k.shape[1], device=q.device
    )


def step5_sparse_decode_ppu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    packed: torch.Tensor,
    query_positions: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    scale: float,
    workspace: Step5PpuDecodeWorkspace,
) -> None:
    """Run one fused sparse decode launch for the whole batch.

    ``packed`` keeps SGLang's int64 layout and is split into the kernel's int32
    planes first; ``out`` receives the BF16 attention and ``lse`` the FP32
    natural log-sum-exp. Every buffer is caller-owned and reused across layers.
    """
    rows, heads, _ = q.shape
    packed_rows, regions = packed.shape
    if rows > workspace.physical_regions.shape[0]:
        raise ValueError("Step5 PPU decode workspace is smaller than the batch")
    if regions != workspace.physical_regions.shape[2] or packed_rows < rows:
        raise ValueError("Step5 PPU decode workspace does not match the selection")
    kv_heads = k.shape[1]
    page_size = STEP5_PPU_DECODE_PAGE_SIZE
    physical = workspace.physical_regions[:rows]
    starts = workspace.logical_starts[:rows]
    module = _jit_step5_decode_module()
    module.unpack_regions(packed[:rows], physical, starts)
    if query_positions.dtype == torch.int64:
        cast_positions = query_positions
    else:
        cast_positions = workspace.positions64[:rows]
        cast_positions.copy_(query_positions)
    module.paged_decode(
        q,
        k.view(-1, page_size, kv_heads, 192),
        v.view(-1, page_size, kv_heads, 192),
        physical,
        starts,
        cast_positions,
        out,
        lse,
        workspace.split_max[:rows],
        workspace.split_denom[:rows],
        workspace.split_acc[:rows],
        float(scale),
    )
