"""Kimi-K3 shared-expert weight streaming over the shared-expert TP subgroup.

Each rank keeps its TP shard of every MoE layer's shared MLP in one CUDA IPC
pool. Before a layer's shared MLP runs, every rank's shard of that layer is
pulled (copy engine: no SMs, no NCCL) into one full-width staging weight, so
the shared MLP runs on the rank-local rows exactly as in TP1 and needs no
activation all-gather / reduce-scatter inside the MoE block. That block is
where MegaMoE leaves ~2 SMs free, so nothing on a side stream progresses
there; the weights are static, so their copy can run in the attention block
instead. Memory: the pool holds the same shards as plain shared-expert TP,
plus one layer of full-width staging.
"""

from __future__ import annotations

import ctypes
import logging
from types import SimpleNamespace
from typing import List, Optional

import msgspec
import torch
import torch.distributed as dist

from sglang.srt.distributed.device_communicators.cuda_wrapper import CudaRTLibrary

logger = logging.getLogger(__name__)

_GATE_UP = "gate_up"
_DOWN = "down"
# cudaMemcpyDefault: the direction comes from the UVA pointers.
_CUDA_MEMCPY_DEFAULT = 4
# cudaStreamCaptureStatusNone.
_CUDA_STREAM_CAPTURE_STATUS_NONE = 0


WEIGHT_STREAM_OFF = "off"
# Stream when the layers and the subgroup allow it, else keep activations.
WEIGHT_STREAM_AUTO = "auto"
# Stream or fail at load.
WEIGHT_STREAM_ON = "on"


def weight_stream_policy(*, disaggregation_mode: str, env_value: Optional[bool]) -> str:
    """Shared-expert TP mode per PD role.

    Prefill pulls the weights: its row count makes the per-layer activation
    AG/RS cost more than the fixed weight copy. Decode pulls the activations:
    a few rows per step move far fewer bytes than a full layer of weights, and
    the staging copy would not hide behind the short decode attention. The
    env var (None when unset) forces the choice on prefill and on non-PD
    servers; decode ignores it."""
    if disaggregation_mode == "decode":
        return WEIGHT_STREAM_OFF
    if env_value is None:
        return (
            WEIGHT_STREAM_AUTO
            if disaggregation_mode == "prefill"
            else WEIGHT_STREAM_OFF
        )
    return WEIGHT_STREAM_ON if env_value else WEIGHT_STREAM_OFF


class CopyOp(msgspec.Struct, frozen=True):
    """One pitched pull copy; all offsets / pitches / widths are elements."""

    dst: str
    dst_offset: int
    dst_pitch: int
    src_rank: int
    src_offset: int
    src_pitch: int
    width: int
    height: int


def slot_copy_plan(
    *, slot: int, rank: int, world_size: int, shard: int, hidden: int
) -> List[CopyOp]:
    """Copies assembling one layer's full-width weights on ``rank`` from every
    rank's pool.

    A pool slot is [gate_up shard (2*shard, hidden) | down shard (hidden,
    shard)], i.e. the row-major TP shards of MergedColumnParallelLinear and
    RowParallelLinear. Staging is the TP1 layout: gate_up (2*full, hidden),
    down (hidden, full). Two copies per rank: each is one CPU launch."""
    block = shard * hidden
    base = slot * 3 * block
    full = shard * world_size
    ops = []
    # Peers first, rotated so the ranks do not all pull from the same pool at
    # once; the local copies last. The copy stream runs these serially, and a
    # same-GPU pitched copy is an SM kernel that waits for free SMs, so placed
    # first it would hold the peer copies (copy engine) behind it.
    for step in range(1, world_size + 1):
        src_rank = (rank + step) % world_size
        # Rank r's [gate_r; up_r] is contiguous in its pool. In the TP1 gate_up,
        # gate_r starts at row r*shard and up_r at row full + r*shard, so one
        # 2-row pitched copy places both.
        ops.append(
            CopyOp(
                dst=_GATE_UP,
                dst_offset=src_rank * block,
                dst_pitch=full * hidden,
                src_rank=src_rank,
                src_offset=base,
                src_pitch=block,
                width=block,
                height=2,
            )
        )
        # Down is column-sharded: rank r's (hidden, shard) block fills
        # columns [r*shard, (r+1)*shard) of the (hidden, full) weight.
        ops.append(
            CopyOp(
                dst=_DOWN,
                dst_offset=src_rank * shard,
                dst_pitch=full,
                src_rank=src_rank,
                src_offset=base + 2 * block,
                src_pitch=shard,
                width=shard,
                height=hidden,
            )
        )
    return ops


class _DeviceBuffer:
    """Exposes a raw cudaMalloc pointer to torch.as_tensor."""

    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "strides": None,
            "version": 3,
        }


class SharedWeightStream:
    """IPC pool, staging weights and copy stream for the shared MLPs of every
    local MoE layer; slot i is the i-th such layer in forward order."""

    def __init__(
        self,
        *,
        mlps: List[torch.nn.Module],
        cpu_group: dist.ProcessGroup,
        rank: int,
        world_size: int,
    ):
        from sglang.srt.distributed.parallel_state import in_the_same_node_as

        if not all(in_the_same_node_as(cpu_group, source_rank=0)):
            raise ValueError(
                "SGLANG_K3_SHARED_EXPERTS_WEIGHT_STREAM needs the shared-expert "
                "TP subgroup on one node (CUDA IPC)."
            )
        hidden, shard = mlps[0].down_proj.weight.shape
        dtype = mlps[0].down_proj.weight.dtype
        device = mlps[0].down_proj.weight.device
        for mlp in mlps:
            gate_up = mlp.gate_up_proj.weight
            down = mlp.down_proj.weight
            if (
                gate_up.shape != (2 * shard, hidden)
                or down.shape != (hidden, shard)
                or gate_up.dtype != dtype
                or down.dtype != dtype
            ):
                raise ValueError(
                    "Shared-expert weight streaming needs identical dense "
                    "shared MLP shards in every MoE layer."
                )

        self._lib = CudaRTLibrary()
        block = shard * hidden
        elem = torch.empty((), dtype=dtype).element_size()
        nbytes = 3 * block * len(mlps) * elem
        pool_ptr = self._lib.cudaMalloc(nbytes)
        pool = torch.as_tensor(_DeviceBuffer(pool_ptr.value, nbytes), device=device)
        pool = pool.view(dtype)
        # The pool replaces the module parameters, so the activation path and
        # weight reloads keep working on the same storage.
        for slot, mlp in enumerate(mlps):
            base = slot * 3 * block
            gate_up = pool[base : base + 2 * block].view(2 * shard, hidden)
            down = pool[base + 2 * block : base + 3 * block].view(hidden, shard)
            gate_up.copy_(mlp.gate_up_proj.weight.data)
            down.copy_(mlp.down_proj.weight.data)
            mlp.gate_up_proj.weight.data = gate_up
            mlp.down_proj.weight.data = down
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        handle = self._lib.cudaIpcGetMemHandle(pool_ptr)
        handles = [None] * world_size
        dist.all_gather_object(handles, handle, group=cpu_group)
        pool_ptrs = [
            pool_ptr.value if r == rank else self._lib.cudaIpcOpenMemHandle(h).value
            for r, h in enumerate(handles)
        ]

        full = shard * world_size
        staging = {
            _GATE_UP: torch.empty((2 * full, hidden), dtype=dtype, device=device),
            _DOWN: torch.empty((hidden, full), dtype=dtype, device=device),
        }
        # Stand-ins for the linear layers: quant_method.apply only reads .weight.
        self._gate_up = SimpleNamespace(weight=staging[_GATE_UP])
        self._down = SimpleNamespace(weight=staging[_DOWN])
        self._copy_stream = torch.cuda.Stream()
        # Prebuilt cudaMemcpy2DAsync arguments per slot. Every pointer and the
        # stream are fixed for the process lifetime, so CUDA graphs can bake them.
        stream_handle = self._copy_stream.cuda_stream
        self._plans = [
            [
                (
                    staging[op.dst].data_ptr() + op.dst_offset * elem,
                    op.dst_pitch * elem,
                    pool_ptrs[op.src_rank] + op.src_offset * elem,
                    op.src_pitch * elem,
                    op.width * elem,
                    op.height,
                    _CUDA_MEMCPY_DEFAULT,
                    stream_handle,
                )
                for op in slot_copy_plan(
                    slot=slot,
                    rank=rank,
                    world_size=world_size,
                    shard=shard,
                    hidden=hidden,
                )
            ]
            for slot in range(len(mlps))
        ]
        # Called raw: the wrapper's per-launch Python cost lands on the MLA
        # layers, where eager prefill is CPU launch-bound.
        self._memcpy_2d = self._lib.funcs["cudaMemcpy2DAsync"]
        self._stream_is_capturing = self._lib.funcs["cudaStreamIsCapturing"]
        self._capture_status = ctypes.c_int()
        self._capture_status_ref = ctypes.byref(self._capture_status)
        # Eager prefetch forks / joins the copy stream with these reused events.
        self._fork_event = torch.cuda.Event()
        self._ready_event = torch.cuda.Event()
        # Slot whose copy is queued on the copy stream and not yet acquired,
        # and whether it was queued by graph replay (joined via _ready_event).
        self._pending_slot: Optional[int] = None
        self._pending_replayed = False
        self._slot_graphs = self._capture_slot_graphs()
        logger.info(
            "K3 shared-expert weight streaming: %d layers, pool %.2f GiB, "
            "staging %.2f GiB, per-layer copy graphs %s",
            len(mlps),
            nbytes / 2**30,
            3 * full * hidden * elem / 2**30,
            "on" if self._slot_graphs is not None else "off",
        )

    def prefetch(self, slot: Optional[int]) -> None:
        """Queue the copy of ``slot`` after all work on the current stream, which
        must already have joined every reader of the staging weights."""
        if slot is None or slot == self._pending_slot:
            return
        current = torch.cuda.current_stream()
        if self._slot_graphs is None or self._is_capturing(current):
            # An enclosing capture (breakable prefill / decode graph) records
            # these launches itself; wait_stream keeps its fork tracking.
            self._copy_stream.wait_stream(current)
            self._launch_raw(self._plans[slot])
            self._pending_replayed = False
        else:
            self._fork_event.record(current)
            self._copy_stream.wait_event(self._fork_event)
            with torch.cuda.stream(self._copy_stream):
                self._slot_graphs[slot].replay()
            self._ready_event.record(self._copy_stream)
            self._pending_replayed = True
        self._pending_slot = slot

    def acquire(self, slot: int) -> None:
        """Make the current stream wait until ``slot`` is staged. The join is
        on the current stream so a breakable-CUDA-graph segment boundary
        between prefetch and acquire is covered by the segment-end join.
        prefetch and acquire of a slot run under the same capture state."""
        self.prefetch(slot)
        current = torch.cuda.current_stream()
        if self._pending_replayed:
            current.wait_event(self._ready_event)
        else:
            current.wait_stream(self._copy_stream)
        self._pending_slot = None
        self._pending_replayed = False

    def _launch_raw(self, plan) -> None:
        memcpy_2d = self._memcpy_2d
        for args in plan:
            err = memcpy_2d(*args)
            if err:
                self._lib.CUDART_CHECK(err)

    def _is_capturing(self, stream: torch.cuda.Stream) -> bool:
        # cudaStreamIsCapturing is capture-safe even under the "global" mode;
        # torch.cuda.is_current_stream_capturing() is unreliable inside a
        # breakable capture (see breakable_cuda_graph._is_stream_capturing).
        err = self._stream_is_capturing(stream.cuda_stream, self._capture_status_ref)
        if err:
            self._lib.CUDART_CHECK(err)
        return self._capture_status.value != _CUDA_STREAM_CAPTURE_STATUS_NONE

    def _capture_slot_graphs(self) -> Optional[List[torch.cuda.CUDAGraph]]:
        """One CUDA graph per slot holding its copies, so an eager prefetch is
        one launch instead of one per copy. None (raw launches) if the driver
        cannot capture the peer IPC memcpys."""
        graphs = []
        torch.cuda.synchronize()
        try:
            with torch.cuda.stream(self._copy_stream):
                for plan in self._plans:
                    graph = torch.cuda.CUDAGraph()
                    # thread_local: other threads' CUDA calls cannot void it.
                    graph.capture_begin(capture_error_mode="thread_local")
                    try:
                        self._launch_raw(plan)
                    finally:
                        graph.capture_end()
                    graphs.append(graph)
        except Exception as exc:
            if self._is_capturing(self._copy_stream):
                raise RuntimeError(
                    "Shared-expert weight-stream graph capture left the copy "
                    "stream capturing"
                ) from exc
            logger.warning(
                "K3 shared-expert weight streaming: cannot capture the per-layer "
                "copies as CUDA graphs (%r); launching them one by one.",
                exc,
            )
            return None
        torch.cuda.synchronize()
        return graphs

    def forward(self, mlp: torch.nn.Module, hidden_states: torch.Tensor):
        """Full-width shared MLP on the staged weights (call after acquire)."""
        gate_up = mlp.gate_up_proj.quant_method.apply(
            layer=self._gate_up, x=hidden_states
        )
        return mlp.down_proj.quant_method.apply(
            layer=self._down, x=mlp.act_fn(gate_up)
        )
