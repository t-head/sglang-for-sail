# SPDX-License-Identifier: Apache-2.0
"""Step5 sparse prefill through the SAIL JIT attention kernel.

The JIT kernel owns the whole prefill chunk in one launch: every query row walks
its own selected regions, so no region union, membership bitmap or gather step
runs first. SGLang stores the selection as packed int64 words while the kernel
indexes two int32 planes, and the caller already owns those planes for the union
path, so one unpack launch per layer bridges the layouts in place.
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
    index = q.device.index
    if index is None:
        index = torch.cuda.current_device()
    return _is_ppu_device(index) and _prefill_op_available()


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
) -> None:
    """Run one sparse prefill launch for the whole chunk.

    ``packed`` keeps SGLang's int64 layout and is split into the kernel's int32
    planes first; ``out`` receives the BF16 attention and ``lse`` the FP32
    natural log-sum-exp. The planes and every other buffer are caller-owned, so
    graph capture keeps fixed addresses.
    """
    rows = q.shape[0]
    if packed.shape != physical_regions.shape or packed.shape != logical_starts.shape:
        raise ValueError("Step5 PPU prefill planes must match the packed selection")
    if packed.shape[0] != rows or live_regions.numel() != rows:
        raise ValueError("Step5 PPU prefill selection does not match the batch")
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