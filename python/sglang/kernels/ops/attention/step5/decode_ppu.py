# SPDX-License-Identifier: Apache-2.0
"""Step5 sparse decode through the SAIL JIT paged attention kernel.

The JIT kernel owns the whole batch in one launch: each row walks its own
selected regions, split-KV partials merge inside the call, and the natural LSE
is reduced from the split states afterwards. SGLang stores the selection as
packed int64 words while the kernel indexes two int32 planes, so one unpack
launch per layer bridges the layouts. Scratch is caller-owned so graph capture
keeps fixed addresses.

The native build of the same kernel keeps the packed words and merges its
splits on device, so it needs a four-dimensional partial block instead of the
flat decode workspace and derives the query position from the causal length.
``can_use_step5_decode_ppu`` gates both builds and the launch picks between
them.
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

# The hand-written tc02 ldmatrix and 2d.tile AIU bulk copy in the native kernel
# depend on backend tuning from the standalone experiments: pin the vector
# register file, sink matrix/async/load/store addresses into the instruction
# operands, widen the alloca budget for the staged KV tiles, and keep the warp
# scheduling and register rotation the hand-written PTX expects.
STEP5_PPU_NATIVE_CUDA_FLAGS = [
    "-lineinfo",
    "-mllvm",
    "-ppu-max-vreg-count=256",
    "-mllvm",
    "-ppu-sink-matrix-addr=true",
    "-mllvm",
    "-ppu-max-alloca-byte-size=320",
    "-mllvm",
    "-ppu-sink-async-addr=true",
    "-mllvm",
    "-ppu-sink-load-addr=true",
    "-mllvm",
    "-ppu-sink-store-addr=true",
    "-mllvm",
    "-ppu-alloca-half-ldst-simplify=true",
    "-mllvm",
    "-ppu-force-warpage=true",
    "-mllvm",
    "-ppu-force-vregrr=true",
]


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
def _is_ppu_device(device) -> bool:
    """Whether this PPU core carries the tc02 attention path.

    The tc02 ldmatrix and the 2d.tile AIU bulk copy exist only on the ppu0015
    core, which reports capability 8.9; the ZW810E (8.0) keeps the portable
    schedule.
    """
    if "ZW" not in torch.cuda.get_device_name(device):
        return False
    major, minor = torch.cuda.get_device_capability(device)
    return major * 10 + minor >= 89


@cache_once
def _jit_step5_decode_native_module() -> "Module":
    return load_jit(
        "step5_sparse_decode_native",
        cuda_files=["step5_dsa/sparse_decode_890p.cuh"],
        cuda_wrappers=[
            ("run", "sglang::step5_decode::run"),
            ("split_capacity", "sglang::step5_decode::split_capacity"),
        ],
        extra_cuda_cflags=STEP5_PPU_NATIVE_CUDA_FLAGS,
    )


@functools.cache
def _decode_native_available() -> bool:
    """Whether the native decode builds on this machine.

    The tuning flags are backend-private, so a compiler that does not know them
    rejects the whole translation unit; report that as "not available" and let
    the caller keep the paged decode instead of failing the forward.
    """
    try:
        _jit_step5_decode_native_module()
    except Exception:
        return False
    return True


def allocate_step5_decode_native_partial(rows: int, heads: int, dim: int, device):
    """Allocate the split scratch the native decode merges on device, or Nones.

    The kernel only reads the block when it actually splits, and refuses a row
    count at or above its direct threshold with an empty tensor of the same
    rank, so the leading extent follows the split capacity rather than ``rows``.
    """
    if heads not in (8, 16) or dim != 192 or not _is_ppu_device(device):
        return None, None
    if not _decode_native_available():
        return None, None
    capacity = _jit_step5_decode_native_module().split_capacity(rows)
    scratch_rows = rows if capacity > 1 else 0
    return (
        torch.empty(
            (scratch_rows, heads, capacity, dim), device=device, dtype=torch.float32
        ),
        torch.empty(
            (scratch_rows, heads, capacity), device=device, dtype=torch.float32
        ),
    )


def can_allocate_step5_ppu_decode(*, heads: int, kv_heads: int, device) -> bool:
    """Whether the JIT paged decode kernel can serve this shape on this device.

    Only the shape and device are consulted, so the caller can size scratch
    before the first query arrives.
    """
    # The kernel is specialized for 8 or 16 query heads per KV head; the TP16
    # four-head layout keeps the Triton path.
    if kv_heads <= 0 or heads not in (kv_heads * 8, kv_heads * 16):
        return False
    if not _is_ppu_device(device):
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


def _can_use_decode_native(
    q: torch.Tensor,
    packed: torch.Tensor,
    counts: torch.Tensor | None,
    causal_lengths: torch.Tensor | None,
    valid_rows: torch.Tensor | None,
    partial: torch.Tensor | None,
    partial_lse: torch.Tensor | None,
) -> bool:
    """Whether this launch can take the native build instead of the paged one.

    The native build is recognised by the four-dimensional partial block its
    caller allocated: it merges the splits on device, so it reads the packed
    selection, the per-row counts and causal lengths, and the one-element valid
    count instead of the unpacked planes.
    """
    if partial is None or partial.ndim != 4 or partial_lse is None:
        return False
    if counts is None or causal_lengths is None or valid_rows is None:
        return False
    rows = q.shape[0]
    if packed.shape[0] < rows or not 1 <= packed.shape[1] <= 516:
        return False
    if counts.numel() != rows or causal_lengths.numel() != rows:
        return False
    if valid_rows.numel() != 1:
        return False
    return _is_ppu_device(q.device) and _decode_native_available()


def step5_sparse_decode_ppu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    packed: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    scale: float,
    query_positions: torch.Tensor | None = None,
    workspace: Step5PpuDecodeWorkspace | None = None,
    counts: torch.Tensor | None = None,
    causal_lengths: torch.Tensor | None = None,
    valid_rows: torch.Tensor | None = None,
    partial: torch.Tensor | None = None,
    partial_lse: torch.Tensor | None = None,
) -> None:
    """Run one fused sparse decode launch for the whole batch.

    ``packed`` keeps SGLang's int64 layout and is split into the kernel's int32
    planes first; ``out`` receives the BF16 attention and ``lse`` the FP32
    natural log-sum-exp. Every buffer is caller-owned and reused across layers.

    When the split counts and a four-dimensional partial block are supplied and
    the native build is usable, that build runs instead: it reads the packed
    words directly, merges the splits inside the launch and needs neither the
    unpacked planes nor the query positions.
    """
    rows, heads, _ = q.shape
    packed_rows, regions = packed.shape
    if packed_rows < rows:
        raise ValueError("Step5 PPU decode selection does not match the batch")
    if _can_use_decode_native(
        q, packed, counts, causal_lengths, valid_rows, partial, partial_lse
    ):
        _jit_step5_decode_native_module().run(
            q,
            k,
            v,
            packed,
            counts,
            causal_lengths,
            valid_rows,
            partial,
            partial_lse,
            out,
            lse,
            float(scale),
        )
        return
    if workspace is None or query_positions is None:
        raise ValueError("Step5 PPU decode needs its workspace and query positions")
    if rows > workspace.physical_regions.shape[0]:
        raise ValueError("Step5 PPU decode workspace is smaller than the batch")
    if regions != workspace.physical_regions.shape[2]:
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
