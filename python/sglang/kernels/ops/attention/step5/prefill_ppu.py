# SPDX-License-Identifier: Apache-2.0
"""Step5 sparse prefill through the SAIL JIT attention kernel.

The JIT kernel owns the whole prefill chunk in one launch: every query row walks
its own selected regions, so no region union, membership bitmap or gather step
runs first. SGLang stores the selection as packed int64 words while the kernel
indexes two int32 planes, and the caller already owns those planes for the union
path, so one unpack launch per layer bridges the layouts in place.

The native build of the same kernel drops that bridge: it reads the packed
words itself and unions a sixteen-row window inside the launch, so the two
int32 planes become pure scratch and the union membership plane carries its
per-tile counts. ``can_use_step5_prefill_ppu`` gates both builds and the launch
picks between them.
"""

import functools
from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# Step5 compresses eight tokens per region, and a region owns eight contiguous
# KV slots.
STEP5_PPU_PREFILL_REGION_TOKENS = 8

# The native kernel fixes the selection width at 513 packed words and walks a
# sixteen-row query group in tiles of one, four or eight rows.
STEP5_PPU_PREFILL_WIDTH = 513
STEP5_PPU_PREFILL_GROUP_ROWS = 16

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


@cache_once
def _jit_step5_prefill_module() -> "Module":
    return load_jit(
        "step5_dsa_prefill",
        cuda_files=["step5_dsa/dsa_prefill_attn.cuh"],
        cuda_wrappers=[
            ("unpack_regions", "step5_dsa::unpack_packed_regions"),
            ("prefill_attn", "step5_dsa::prefill_attn"),
        ],
    )


@functools.cache
def _prefill_op_available() -> bool:
    """Whether the JIT sparse prefill module can build on this machine."""
    try:
        _jit_step5_prefill_module()
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
def _jit_step5_prefill_native_module(heads: int) -> "Module":
    """Compile the native build for one query-head layout.

    ``heads`` is a compile-time constant there (the group CTA shape and the MMA
    tiling depend on it), so the eight- and sixteen-head layouts are separate
    builds and the marker keeps their cache entries apart.
    """
    return load_jit(
        "step5_sparse_prefill_native",
        str(heads),
        cuda_files=["step5_dsa/sparse_prefill_890p.cuh"],
        cuda_wrappers=[("run", "sglang::step5_prefill::run")],
        extra_cuda_cflags=[
            *STEP5_PPU_NATIVE_CUDA_FLAGS,
            f"-DSTEP5_PREFILL_HEADS={heads}",
        ],
    )


@functools.cache
def _prefill_native_available(heads: int) -> bool:
    """Whether the native build compiles on this machine for this head layout.

    The tuning flags are backend-private, so a compiler that does not know them
    rejects the whole translation unit; report that as "not available" and let
    the caller keep the tc02 build instead of failing the forward.
    """
    try:
        _jit_step5_prefill_native_module(heads)
    except Exception:
        return False
    return True


def can_use_step5_prefill_ppu(q: torch.Tensor, k: torch.Tensor) -> bool:
    """Whether the JIT sparse prefill kernel can serve this query/KV pair.

    The kernel addresses one KV head, so it covers the eight- and sixteen-head
    GQA layouts of this backend and leaves the TP16 four-head layout on the
    portable schedule.
    """
    if q.device.type != "cuda" or torch.version.hip is not None:
        return False
    if q.dtype != torch.bfloat16 or q.ndim != 3 or q.shape[2] != 192:
        return False
    if k.dtype != torch.bfloat16 or k.ndim != 3 or tuple(k.shape[1:]) != (1, 192):
        return False
    if (
        q.shape[0] == 0
        or q.shape[1] not in (8, 16)
        or k.shape[0] % STEP5_PPU_PREFILL_REGION_TOKENS
    ):
        return False
    return _is_ppu_device(q.device) and _prefill_op_available()


def _can_use_prefill_native(
    q: torch.Tensor,
    packed: torch.Tensor,
    positions: torch.Tensor,
    group_start: torch.Tensor | None,
    group_end: torch.Tensor | None,
    tile_counts: torch.Tensor | None,
) -> bool:
    """Whether this launch can take the native build instead of the tc02 one.

    The native kernel reads the packed selection itself, so it needs the fixed
    513-word width, the group bounds it walks and one int32 column per tile.
    Anything else stays on the tc02 build, which bridges the packed words
    through an unpack launch.
    """
    if group_start is None or group_end is None or tile_counts is None:
        return False
    groups = group_start.numel()
    if groups == 0 or group_end.numel() != groups:
        return False
    if packed.shape[1] != STEP5_PPU_PREFILL_WIDTH or packed.shape[0] != q.shape[0]:
        return False
    if positions.dtype != torch.int32 or positions.numel() != q.shape[0]:
        return False
    if tile_counts.ndim != 2 or tile_counts.shape[0] != groups:
        return False
    if tile_counts.shape[1] < STEP5_PPU_PREFILL_GROUP_ROWS:
        return False
    return _is_ppu_device(q.device) and _prefill_native_available(q.shape[1])


def step5_sparse_prefill_ppu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    packed: torch.Tensor,
    physical_regions: torch.Tensor,
    logical_starts: torch.Tensor,
    live_regions: torch.Tensor,
    positions: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    scale: float,
    group_start: torch.Tensor | None = None,
    group_end: torch.Tensor | None = None,
    tile_counts: torch.Tensor | None = None,
) -> None:
    """Run one sparse prefill launch for the whole chunk.

    ``packed`` keeps SGLang's int64 layout and is split into the kernel's int32
    planes first; ``out`` receives the BF16 attention and ``lse`` the FP32
    natural log-sum-exp. The planes and every other buffer are caller-owned, so
    graph capture keeps fixed addresses.

    When the group bounds and a tile-count plane are supplied and the native
    build is usable, that build runs instead: it fills the same two planes
    itself and writes the per-tile region counts into ``tile_counts``.
    """
    rows = q.shape[0]
    if packed.shape != physical_regions.shape or packed.shape != logical_starts.shape:
        raise ValueError("Step5 PPU prefill planes must match the packed selection")
    if packed.shape[0] != rows or live_regions.numel() != rows:
        raise ValueError("Step5 PPU prefill selection does not match the batch")
    if _can_use_prefill_native(
        q, packed, positions, group_start, group_end, tile_counts
    ):
        _jit_step5_prefill_native_module(q.shape[1]).run(
            q,
            k,
            v,
            packed,
            live_regions,
            positions,
            group_start,
            group_end,
            physical_regions,
            logical_starts,
            tile_counts,
            out,
            lse,
            float(scale),
        )
        return
    module = _jit_step5_prefill_module()
    module.unpack_regions(packed, physical_regions, logical_starts)
    module.prefill_attn(
        q,
        k,
        v,
        physical_regions.unsqueeze(1),
        logical_starts.unsqueeze(1),
        live_regions,
        positions,
        out,
        lse,
        float(scale),
    )