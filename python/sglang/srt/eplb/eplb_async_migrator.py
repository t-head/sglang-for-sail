# Copyright 2023-2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import annotations

import logging
import queue
import threading
import time
from concurrent.futures import Future
from typing import Dict, List, Optional, Tuple

import msgspec
import torch
import torch.distributed

from sglang.srt.eplb.eplb_cpu_worker import (
    METADATA_UPLOAD_FIELDS,
    PlannerConfig,
    allocate_upload_buffers,
    create_planner_executor,
    decode_layer_plan,
)
from sglang.srt.eplb.expert_location import ExpertLocationMetadata
from sglang.srt.runtime_context import get_device

logger = logging.getLogger(__name__)


class LayerPullPlan(msgspec.Struct):
    weight2buffer_copies: List[Tuple[int, int]]
    buffer2weight_copies: List[Tuple[int, int]]
    reads: Dict[int, List[Tuple[int, int]]]


def plan_layer_pull(
    old_physical_to_logical_map: List[int],
    new_physical_to_logical_map: List[int],
    num_local_physical_experts: int,
    num_gpu_per_node: int,
    rank: int,
) -> LayerPullPlan:
    lo = rank * num_local_physical_experts
    hi = lo + num_local_physical_experts
    node_of_rank = rank // num_gpu_per_node
    weight2buffer_copies: List[Tuple[int, int]] = []
    buffer2weight_copies: List[Tuple[int, int]] = []
    reads: Dict[int, List[Tuple[int, int]]] = {}

    for dst in range(lo, hi):
        logical_expert_id = new_physical_to_logical_map[dst]
        if old_physical_to_logical_map[dst] == logical_expert_id:
            continue

        src = next(
            (
                p
                for p in range(lo, hi)
                if old_physical_to_logical_map[p] == logical_expert_id
            ),
            None,
        )
        if src is not None:
            weight2buffer_copies.append((src - lo, dst - lo))
            buffer2weight_copies.append((dst - lo, dst - lo))
            continue

        src = next(
            (
                p
                for p in range(lo, dst)
                if new_physical_to_logical_map[p] == logical_expert_id
            ),
            None,
        )
        if src is not None:
            buffer2weight_copies.append((src - lo, dst - lo))
            continue

        holders = [
            p
            for p, e in enumerate(old_physical_to_logical_map)
            if e == logical_expert_id
        ]
        same_node = [
            p
            for p in holders
            if (p // num_local_physical_experts) // num_gpu_per_node == node_of_rank
        ]
        candidates = same_node or holders
        p = candidates[(dst - lo) % len(candidates)]
        reads.setdefault(p // num_local_physical_experts, []).append(
            (p % num_local_physical_experts, dst - lo)
        )
        buffer2weight_copies.append((dst - lo, dst - lo))

    return LayerPullPlan(
        weight2buffer_copies=weight2buffer_copies,
        buffer2weight_copies=buffer2weight_copies,
        reads=reads,
    )


def _expert_storage_bytes(tensor: torch.Tensor) -> int:
    """Validate densely packed expert slices, including permuted scale layouts."""
    if tensor.ndim < 1 or tensor.numel() == 0:
        raise ValueError("EPLB transfer requires nonempty expert slices")
    extent = 1
    for stride, size in sorted(
        (stride, size)
        for size, stride in zip(tensor.shape[1:], tensor.stride()[1:])
        if size > 1
    ):
        if stride != extent:
            raise ValueError(
                "EPLB transfer requires dense, non-overlapping expert slices"
            )
        extent *= size
    if tensor.stride(0) != extent:
        raise ValueError("EPLB transfer requires evenly packed expert slices")
    return extent * tensor.element_size()


def expert_storage_layout(weights, slots):
    if not weights:
        return None
    layers = tuple(sorted(weights))
    signature = None
    for layer in layers:
        tensors = weights[layer]
        if not tensors:
            return None
        current = []
        for t in tensors:
            if t.device.type != "cuda" or t.ndim < 1 or t.shape[0] != slots:
                return None
            try:
                _expert_storage_bytes(t)
            except ValueError:
                return None
            current.append((tuple(t.shape), t.stride(), t.dtype))
        if signature is None:
            signature = current
        elif signature != current:
            return None
    return layers, signature


class AsyncRebalanceJob(msgspec.Struct):
    layer_ids: List[int]
    old_metadata: ExpertLocationMetadata
    routed_experts_weights_of_layer: Dict[int, List[torch.Tensor]]
    planner_config: PlannerConfig
    logical_count_host: torch.Tensor
    count_event: torch.Event
    time_start: float
    logical_count_device: Optional[torch.Tensor] = None
    placement_future: Optional[Future] = None
    encoded_plans: Optional[torch.Tensor] = None
    placement_error: Optional[BaseException] = None
    plans: Dict[int, LayerPullPlan] = {}
    new_metadata: Optional[ExpertLocationMetadata] = None
    layers_done: Dict[int, bool] = {}
    layers_installed: int = 0


class AsyncExpertMigrator:
    def __init__(self, transport):
        self._device = get_device().device
        self._device_module = torch.get_device_module(self._device)
        self._transport = transport
        self._staging: Optional[List[torch.Tensor]] = None
        self._cpu_group = None
        self._planner_executor = None
        self._metadata_upload_event = self._device_module.Event()
        self._metadata_upload_event.record()
        self._install_event = self._device_module.Event()
        self._install_event.record()
        self._copy_queue: queue.Queue[Optional[Tuple[AsyncRebalanceJob, int]]] = (
            queue.Queue()
        )
        self._copy_thread: Optional[threading.Thread] = None
        self._copy_error: Optional[BaseException] = None

    def prepare(
        self,
        routed_experts_weights_of_layer: Dict[int, List[torch.Tensor]],
        cpu_group: Optional[torch.distributed.ProcessGroup] = None,
        *,
        metadata: ExpertLocationMetadata,
    ):
        if cpu_group is None:
            from sglang.srt.distributed.parallel_state import get_moe_ep_group

            cpu_group = get_moe_ep_group().cpu_group

        layer_ids = sorted(routed_experts_weights_of_layer.keys())
        self._staging = self._transport.prepare(
            routed_experts_weights_of_layer[layer_ids[0]]
        )
        for tensors in routed_experts_weights_of_layer.values():
            if len(tensors) != len(self._staging):
                raise ValueError("EPLB layers must have identical weight layouts")
            for tensor, buffer in zip(tensors, self._staging, strict=True):
                if (tensor.shape, tensor.stride(), tensor.dtype) != (
                    buffer.shape,
                    buffer.stride(),
                    buffer.dtype,
                ):
                    raise ValueError(
                        "EPLB layers and staging must have identical weight layouts"
                    )
        self._metadata_device_buffers = {
            name: torch.empty_like(tensor, device=self._device)
            for name in METADATA_UPLOAD_FIELDS
            if (tensor := getattr(metadata, name)) is not None
        }
        self._planner_executor = create_planner_executor(
            allocate_upload_buffers(metadata)
        )
        self._cpu_group = cpu_group
        self._copy_thread = threading.Thread(
            target=self._copy_loop, name="eplb-async-copies", daemon=True
        )
        self._copy_thread.start()

    @property
    def prepared(self) -> bool:
        return self._staging is not None

    def start(
        self,
        *,
        logical_count: torch.Tensor,
        old_metadata: ExpertLocationMetadata,
        routed_experts_weights_of_layer: Dict[int, List[torch.Tensor]],
        planner_config: PlannerConfig,
    ) -> AsyncRebalanceJob:
        assert self.prepared
        logical_count_host = torch.empty(
            logical_count.shape,
            dtype=logical_count.dtype,
            device="cpu",
            pin_memory=torch.device(self._device).type != "cpu",
        )
        logical_count_host.copy_(logical_count, non_blocking=True)
        count_event = self._device_module.Event()
        count_event.record()
        return AsyncRebalanceJob(
            layer_ids=sorted(routed_experts_weights_of_layer.keys()),
            old_metadata=old_metadata,
            routed_experts_weights_of_layer=routed_experts_weights_of_layer,
            planner_config=planner_config,
            logical_count_host=logical_count_host,
            count_event=count_event,
            time_start=time.time(),
            logical_count_device=logical_count,
        )

    def placement_ready(self, job: AsyncRebalanceJob) -> bool:
        if job.placement_future is None and job.placement_error is None:
            # Never let the IPC thread overwrite staging still consumed by H2D.
            if job.count_event.query() and self._metadata_upload_event.query():
                job.logical_count_device = None
                try:
                    job.placement_future = self._planner_executor.submit(
                        job.planner_config,
                        job.logical_count_host,
                        job.old_metadata.physical_to_logical_map_cpu,
                        job.layer_ids,
                    )
                except Exception as error:
                    job.placement_error = error
        ready = job.placement_future is not None and job.placement_future.done()
        if ready:
            job.placement_error = job.placement_future.exception()
        # All ranks propagate a failed worker at the same collective boundary.
        status = torch.tensor(
            [-1 if job.placement_error is not None else int(ready)], dtype=torch.int32
        )
        torch.distributed.all_reduce(
            status, op=torch.distributed.ReduceOp.MIN, group=self._cpu_group
        )
        state = status.item()
        if state < 0:
            raise RuntimeError(
                "EPLB CPU placement failed on an EP rank"
            ) from job.placement_error
        if state == 0:
            return False
        result = job.placement_future.result()
        job.new_metadata = result.metadata
        job.encoded_plans = result.pull_plans
        return True

    def build_metadata(self, job: AsyncRebalanceJob) -> ExpertLocationMetadata:
        assert job.new_metadata is not None
        # Staging is already populated by the CPU-only IPC thread. Keep CUDA
        # submission here, without pageable uploads that drain the stream queue.
        for name in METADATA_UPLOAD_FIELDS:
            tensor = getattr(job.new_metadata, name)
            if tensor is not None:
                assert torch.device(self._device).type == "cpu" or tensor.is_pinned()
                target = self._metadata_device_buffers[name]
                target.copy_(tensor, non_blocking=True)
                setattr(job.new_metadata, name, target)
        self._metadata_upload_event.record()
        return job.new_metadata

    def issue_layer(self, job: AsyncRebalanceJob, layer_id: int):
        plan = decode_layer_plan(job.encoded_plans[job.layer_ids.index(layer_id)])
        job.plans[layer_id] = plan
        job.layers_done[layer_id] = False
        self._copy_queue.put((job, layer_id))

    def _copy_loop(self):
        while True:
            item = self._copy_queue.get()
            if item is None:
                self._copy_queue.task_done()
                return
            job, layer_id = item
            try:
                device = self._staging[0].device
                if device.type == "cuda":
                    torch.cuda.set_device(device)
                self._install_event.synchronize()
                plan = job.plans[layer_id]
                self._transport.transfer(
                    job.routed_experts_weights_of_layer[layer_id], plan
                )
                job.layers_done[layer_id] = True
            except BaseException as e:
                self._copy_error = e
                job.layers_done[layer_id] = True
            finally:
                self._copy_queue.task_done()

    def layer_ready(self, job: AsyncRebalanceJob, layer_id: int) -> bool:
        status = torch.tensor(
            [
                -1
                if self._copy_error is not None
                else int(job.layers_done.get(layer_id, False))
            ],
            dtype=torch.int32,
        )
        torch.distributed.all_reduce(
            status, op=torch.distributed.ReduceOp.MIN, group=self._cpu_group
        )
        if status.item() < 0:
            raise RuntimeError(
                f"Expert copy failed on an EP rank for layer {layer_id}"
            ) from self._copy_error
        return bool(status.item())

    def install_layer(self, job: AsyncRebalanceJob, layer_id: int):
        plan = job.plans[layer_id]
        weights = job.routed_experts_weights_of_layer[layer_id]
        assert job.new_metadata is not None
        for src_slot, dst_slot in plan.buffer2weight_copies:
            for weight, buffer in zip(weights, self._staging, strict=True):
                weight[dst_slot].copy_(buffer[src_slot])
        for current, prepared in job.old_metadata.iter_metadata_field_pairs(
            job.new_metadata
        ):
            current[layer_id].copy_(prepared[layer_id], non_blocking=True)
        self._install_event = self._device_module.Event()
        self._install_event.record()
        job.layers_installed += 1

    def cancel(self, job: Optional[AsyncRebalanceJob]):
        if job is not None and job.placement_future is not None:
            future = job.placement_future
            if not future.cancel() and not future.done():
                # Reset keeps the migrator usable even if the old worker hung.
                self._planner_executor.shutdown()
                self._planner_executor = create_planner_executor(
                    allocate_upload_buffers(job.old_metadata)
                )
        # A running transfer owns the staging storage even after it left the queue.
        self._copy_queue.join()
        self._copy_error = None
        self._device_module.synchronize()

    def close(self):
        """Release the worker and transport thread after the last active job."""
        if self._planner_executor is not None:
            self._planner_executor.shutdown()
            self._planner_executor = None
        self.cancel(None)
        if self._copy_thread is not None:
            self._copy_queue.put(None)
            self._copy_thread.join()
            self._copy_thread = None
        if hasattr(self._transport, "close"):
            self._transport.close()
        self._transport = None
        self._staging = None
