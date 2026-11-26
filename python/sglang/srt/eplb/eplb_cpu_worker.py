"""CPU-only EPLB planning, isolated from the inference interpreter."""

from __future__ import annotations

import atexit
import multiprocessing
import os
import socket
import subprocess
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Optional

import torch
import torch.nn.functional as F

from sglang.srt.eplb.expert_location import (
    ExpertLocationMetadata,
    compute_logical_to_rank_dispatch_physical_map,
)


@dataclass(frozen=True)
class PlannerConfig:
    num_physical_experts: int
    ep_size: int
    num_groups: Optional[int]
    num_nodes: int
    rank: int
    algorithm: str
    static_dispatch: bool
    prefer_same_node: bool
    active_ranks: Optional[tuple[int, ...]] = None

    @classmethod
    def from_runtime(cls, model_config):
        from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager
        from sglang.srt.eplb.expert_location import _prefer_same_node_experts
        from sglang.srt.runtime_context import get_exec, get_parallel

        common = ExpertLocationMetadata._init_common(model_config)
        if common is None:
            raise ValueError("EPLB requires an expert model")
        algorithm = get_exec().moe.eplb_algorithm
        active_ranks = None
        if algorithm.startswith("elasticity_aware"):
            state = ElasticEPStateManager.instance()
            active_ranks = tuple(
                (
                    state.active_ranks
                    if state is not None
                    else ElasticEPStateManager.healthy_rank_state()
                )
                .cpu()
                .tolist()
            )
        return cls(
            num_physical_experts=common["num_physical_experts"],
            ep_size=common["ep_size"],
            num_groups=common["model_config_for_expert_location"].num_groups,
            num_nodes=get_parallel().nnodes,
            rank=get_parallel().moe_ep_rank,
            algorithm=algorithm,
            static_dispatch=get_exec().moe.ep_dispatch_algorithm == "static",
            prefer_same_node=_prefer_same_node_experts(),
            active_ranks=active_ranks,
        )


@dataclass
class PlannerResult:
    metadata: ExpertLocationMetadata
    # [layer index, operation (local/install/remote rank/remote slot), dst slot].
    # Tensor IPC avoids rebuilding a model-wide Python object graph in the parent.
    pull_plans: torch.Tensor
    worker_pid: int


def _initialize_worker(num_threads: int):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(num_threads)
    torch.set_grad_enabled(False)
    if torch.cuda.is_initialized():
        raise RuntimeError("EPLB CPU worker must not inherit a CUDA context")


METADATA_UPLOAD_FIELDS = (
    "physical_to_logical_map",
    "logical_to_all_physical_map",
    "logical_to_all_physical_map_num_valid",
    "logical_to_rank_dispatch_physical_map",
)


def allocate_upload_buffers(metadata: ExpertLocationMetadata):
    """Allocate once on the inference thread before serving starts."""
    return {
        name: torch.empty(
            tensor.shape,
            dtype=tensor.dtype,
            device="cpu",
            pin_memory=tensor.device.type != "cpu",
        )
        for name in METADATA_UPLOAD_FIELDS
        if (tensor := getattr(metadata, name)) is not None
    }


class CPUPlanner:
    """A persistent CPU interpreter with a small asynchronous IPC reader."""

    def __init__(self, upload_buffers=None):
        self._upload_buffers = upload_buffers
        parent, child = socket.socketpair()
        self._connection = Connection(parent.detach())
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        command = (
            "from sglang.srt.eplb.eplb_cpu_worker import _serve; "
            f"_serve({child.fileno()}, {torch.get_num_threads()})"
        )
        try:
            self._process = subprocess.Popen(
                [sys.executable, "-c", command],
                env=env,
                pass_fds=(child.fileno(),),
            )
        except BaseException:
            self._connection.close()
            raise
        finally:
            child.close()
        self._io = ThreadPoolExecutor(max_workers=1, thread_name_prefix="eplb-ipc")
        self._closed = False
        try:
            # Tensor file-descriptor sharing authenticates against the parent's key.
            self._connection.send_bytes(
                bytes(multiprocessing.current_process().authkey)
            )
            if not self._connection.poll(120):
                raise RuntimeError("EPLB CPU worker did not initialize")
            self.pid = self._connection.recv()
        except BaseException:
            self.shutdown()
            raise
        atexit.register(self.shutdown)

    def submit(self, config, logical_count, old_map, layer_ids):
        if self._closed:
            raise RuntimeError("EPLB CPU worker is closed")
        return self._io.submit(
            self._exchange, (config, logical_count, old_map, layer_ids)
        )

    def _exchange(self, arguments):
        try:
            config, counts, old_map, layers = arguments
            # The forward path will not update the live map until this result is ready.
            snapshot = (config, counts.clone(), old_map.clone(), layers)
            self._connection.send(snapshot)
            success, result = self._connection.recv()
        except (EOFError, OSError) as error:
            raise RuntimeError("EPLB CPU worker exited during planning") from error
        if not success:
            raise RuntimeError(f"EPLB CPU worker failed:\n{result}")
        if self._upload_buffers is not None:
            # CPU copies only: no CUDA context operations on the IPC thread.
            # CPU metadata snapshots remain immutable and separate from staging.
            for name in METADATA_UPLOAD_FIELDS:
                tensor = getattr(result.metadata, name)
                if tensor is None:
                    if name in self._upload_buffers:
                        raise ValueError(
                            "EPLB metadata layout changed after initialization"
                        )
                    continue
                buffer = self._upload_buffers.get(name)
                if buffer is None or (buffer.shape, buffer.dtype) != (
                    tensor.shape,
                    tensor.dtype,
                ):
                    raise ValueError(
                        "EPLB metadata layout changed after initialization"
                    )
                buffer.copy_(tensor)
                setattr(result.metadata, name, buffer)
        return result

    def shutdown(self, wait=True, cancel_futures=True):
        if self._closed:
            return
        self._closed = True
        self._io.shutdown(wait=False, cancel_futures=cancel_futures)
        try:
            # Interrupt send/recv even when the child is alive but unresponsive.
            # Keep the Connection fd valid until the exchange thread has exited.
            with socket.fromfd(
                self._connection.fileno(), socket.AF_UNIX, socket.SOCK_STREAM
            ) as sock:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass  # The peer may have already disconnected.
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=5)
        finally:
            self._io.shutdown(wait=True, cancel_futures=cancel_futures)
            self._connection.close()
            atexit.unregister(self.shutdown)


def create_planner_executor(upload_buffers=None) -> CPUPlanner:
    return CPUPlanner(upload_buffers)


def _serve(fd: int, num_threads: int):
    connection = Connection(fd)
    try:
        multiprocessing.current_process().authkey = connection.recv_bytes()
        _initialize_worker(num_threads)
        connection.send(os.getpid())
        while True:
            arguments = connection.recv()
            if arguments is None:
                return
            try:
                result = (True, compute_plan(*arguments))
            except Exception:
                result = (False, traceback.format_exc())
            connection.send(result)
    except EOFError:
        return
    finally:
        connection.close()


def compute_plan(
    config: PlannerConfig,
    logical_count: torch.Tensor,
    old_map: torch.Tensor,
    layer_ids: list[int],
) -> PlannerResult:
    from sglang.srt.eplb import eplb_algorithms as algorithms
    from sglang.srt.eplb.eplb_async_migrator import plan_layer_pull

    if logical_count.device.type != "cpu" or old_map.device.type != "cpu":
        raise ValueError("CPU planner accepts only CPU tensors")
    if logical_count.ndim == 2:
        logical_count = logical_count.unsqueeze(0)
    slots, remainder = divmod(config.num_physical_experts, config.ep_size)
    if remainder or config.ep_size % config.num_nodes:
        raise ValueError("EPLB requires evenly distributed slots and ranks")
    algorithm = algorithms.compute_algorithm(
        config.algorithm, config.num_groups, config.num_nodes
    )
    if config.active_ranks is not None:
        physical, logical, _ = algorithms.elasticity_aware.rebalance_experts(
            weight=logical_count.sum(dim=0),
            num_replicas=config.num_physical_experts,
            num_groups=config.num_groups,
            num_nodes=config.num_nodes,
            num_gpus=config.ep_size,
            enable_hierarchical=(
                algorithm == algorithms.EplbAlgorithm.elasticity_aware_hierarchical
            ),
            active_ranks=torch.tensor(config.active_ranks, dtype=torch.int64),
        )
    else:
        physical, logical, _ = algorithms.rebalance_experts(
            tokens_per_expert=logical_count,
            num_physical_experts=config.num_physical_experts,
            num_local_physical_experts=slots,
            num_groups=config.num_groups,
            num_nodes=config.num_nodes,
            algorithm=algorithm,
        )
    dispatch = None
    if config.static_dispatch:
        dispatch = compute_logical_to_rank_dispatch_physical_map(
            logical,
            config.ep_size,
            config.num_physical_experts,
            config.rank % config.ep_size,
            prefer_same_node=config.prefer_same_node,
            num_nodes=config.num_nodes,
        )
    physical_cpu = physical.clone()
    logical_cpu = F.pad(
        logical, (0, config.num_physical_experts - logical.shape[-1]), value=-1
    )
    metadata = ExpertLocationMetadata(
        physical_to_logical_map=physical_cpu,
        physical_to_logical_map_cpu=physical_cpu,
        logical_to_all_physical_map=logical_cpu,
        logical_to_all_physical_map_cpu=logical_cpu,
        logical_to_all_physical_map_num_valid=torch.count_nonzero(
            logical != -1, dim=-1
        ),
        ep_size=config.ep_size,
        logical_to_rank_dispatch_physical_map=dispatch,
    )
    encoded = torch.full((len(layer_ids), 4, slots), -1, dtype=torch.int64)
    for index, layer in enumerate(layer_ids):
        plan = plan_layer_pull(
            old_map[layer].tolist(),
            physical[layer].tolist(),
            slots,
            config.ep_size // config.num_nodes,
            config.rank,
        )
        for src, dst in plan.weight2buffer_copies:
            encoded[index, 0, dst] = src
        for src, dst in plan.buffer2weight_copies:
            encoded[index, 1, dst] = src
        for peer, pairs in plan.reads.items():
            for src, dst in pairs:
                encoded[index, 2, dst] = peer
                encoded[index, 3, dst] = src
    return PlannerResult(metadata, encoded, os.getpid())


def decode_layer_plan(encoded: torch.Tensor):
    from sglang.srt.eplb.eplb_async_migrator import LayerPullPlan

    local, install, peers, remote = encoded.tolist()
    reads = {}
    for dst, peer in enumerate(peers):
        if peer >= 0:
            reads.setdefault(peer, []).append((remote[dst], dst))
    return LayerPullPlan(
        [(src, dst) for dst, src in enumerate(local) if src >= 0],
        [(src, dst) for dst, src in enumerate(install) if src >= 0],
        reads,
    )
