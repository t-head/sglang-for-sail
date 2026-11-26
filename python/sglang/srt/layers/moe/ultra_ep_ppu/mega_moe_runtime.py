"""UltraEP placement and replica pools for PPU channelwise FP8 MegaMoE."""

from __future__ import annotations

import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_resources


def _managers() -> dict:
    buffers = get_resources().buffers
    managers = buffers.get("ultra_ep_ppu_mega_managers")
    if managers is None:
        managers = {}
        buffers["ultra_ep_ppu_mega_managers"] = managers
    return managers


def is_enabled() -> bool:
    return envs.SGLANG_OPT_ULTRA_EP_PPU.get()


def redundant_per_rank() -> int:
    replicas = envs.SGLANG_OPT_ULTRA_EP_PPU_REDUNDANT_PER_RANK.get()
    if replicas <= 0:
        raise ValueError(
            "SGLANG_OPT_ULTRA_EP_PPU_REDUNDANT_PER_RANK must be positive "
            "when SGLANG_OPT_ULTRA_EP_PPU is enabled"
        )
    return replicas


def _load_ultra_ep_ppu():
    try:
        from ultra_ep import ppu
    except ImportError as exc:
        raise RuntimeError(
            "PPU UltraEP MegaMoE requires ultra_ep==1.0.0+ppu.79cffc8"
        ) from exc
    missing = [
        name
        for name in (
            "Manager",
            "PhysicalWeightPool",
            "ReplicaWeightPool",
            "allocate_local_expert_tensors",
            "allocate_symmetric_expert_tensors",
        )
        if not hasattr(ppu, name)
    ]
    if missing:
        raise RuntimeError("ultra_ep.ppu is missing: " + ", ".join(missing))
    return ppu


def register_physical_weights(layer) -> None:
    from sglang.srt.runtime_context import get_parallel

    api = _load_ultra_ep_ppu()
    group = get_parallel().moe_ep_group.device_group
    num_master = layer.num_local_experts
    replicas = redundant_per_rank()
    if layer.num_experts != num_master * group.size() or num_master + replicas > 128:
        raise ValueError("Invalid local master/replica geometry for FP8 MegaMoE")

    # Target and draft share the EP group but not the layer IDs or weight pools.
    key = (id(group), getattr(layer, "is_draft", False))
    managers = _managers()
    manager = managers.get(key)
    if manager is None:
        manager = api.Manager(
            group=group,
            num_local_master_experts=num_master,
            num_local_redundant_experts=replicas,
        )
        managers[key] = manager
    num_physical = num_master + replicas
    if (
        manager.num_local_master_experts != num_master
        or manager.num_local_redundant_experts != replicas
        or manager.num_local_physical_experts != num_physical
        or manager.num_global_physical_experts != num_physical * group.size()
    ):
        raise RuntimeError("Inconsistent UltraEP MegaMoE expert geometry")

    # Give the symmetric allocator room after releasing replaced checkpoint rows.
    torch.cuda.empty_cache()
    master_alloc = api.allocate_symmetric_expert_tensors(
        group, [*layer.mega_l1_weights, *layer.mega_l2_weights], 0
    )
    w1, s1, w2, s2 = master_alloc.tensors
    master = api.PhysicalWeightPool(
        w1=w1,
        w1_scale=s1,
        w1_peer_base_ptrs=master_alloc.peer_base_ptrs,
        w2=w2,
        w2_scale=s2,
        w2_peer_base_ptrs=master_alloc.peer_base_ptrs,
        keepalive=(master_alloc.backing,),
    )
    manager.register_physical_weight_pool(layer.layer_id, master)

    replica = manager.replica_weight_pool()
    if replica is None:
        torch.cuda.empty_cache()
        replica_l1 = api.allocate_local_expert_tensors([w1, s1], replicas)
        replica_l2 = api.allocate_local_expert_tensors([w2, s2], replicas)
        replica = api.ReplicaWeightPool(
            w1=replica_l1.tensors[0],
            w1_scale=replica_l1.tensors[1],
            w2=replica_l2.tensors[0],
            w2_scale=replica_l2.tensors[1],
            keepalive=(replica_l1.backing, replica_l2.backing),
        )
        manager.register_replica_weight_pool(replica)
    for physical, template in zip(
        (replica.w1, replica.w1_scale, replica.w2, replica.w2_scale),
        (w1, s1, w2, s2),
    ):
        if (
            physical.shape != (replicas, *template.shape[1:])
            or physical.stride()[1:] != template.stride()[1:]
            or physical.dtype != template.dtype
            or physical.device != template.device
        ):
            raise RuntimeError("All UltraEP MegaMoE layers must share one expert layout")

    layer.w13_weight.data, layer.w13_weight_scale.data = w1, s1
    layer.w2_weight.data, layer.w2_weight_scale.data = w2, s2
    layer.mega_l1_weights = (w1, s1)
    layer.mega_l2_weights = (w2, s2)
    layer._ultra_ep_ppu_mega_manager = manager
    layer._ultra_ep_ppu_mega_allocation = master_alloc


def prepare(layer, logical_topk_ids: torch.Tensor):
    """Reroute logical IDs and start the replica pull before pre-dispatch."""
    manager = layer._ultra_ep_ppu_mega_manager
    manager.update_placement(layer.layer_id, logical_topk_ids)
    weight_event = manager.weight_sync(layer.layer_id, async_finish=True)
    physical_topk_ids = manager.reroute(layer.layer_id, logical_topk_ids)
    if (
        not torch.is_tensor(physical_topk_ids)
        or physical_topk_ids.shape != logical_topk_ids.shape
        or physical_topk_ids.dtype != torch.int64
        or physical_topk_ids.device != logical_topk_ids.device
        or not physical_topk_ids.is_contiguous()
    ):
        raise RuntimeError("UltraEP MegaMoE returned incompatible physical top-k IDs")

    replica = manager.replica_weight_pool()
    return physical_topk_ids, weight_event, {
        "l1_weights_replicas": (replica.w1, replica.w1_scale),
        "l2_weights_replicas": (replica.w2, replica.w2_scale),
    }
