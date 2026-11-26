"""Isolated GLM-shaped FP8 MoE evidence; no server, checkpoint or fused router.

Run with torchrun --standalone --nproc-per-node=4 (or 8), --ep 4 (or 8)
--outdir <directory>. The default fixture is GLM-5.3-Flash E288/H4096/I2048/K8.
--fixture smoke uses smaller weights, but still requires real EP4/8 placement.
This is not full-model accuracy or TP/CP serving evidence.
"""

import argparse
import datetime
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from sglang.test.kernels.fp8_mega_moe import (
    check_output,
    load_deep_gemm,
    make_weights,
    native,
    reference,
)


def require_all(condition, message):
    flag = torch.tensor(int(bool(condition)), device="cuda", dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if not flag.item():
        raise AssertionError(message)


def gather(value):
    gathered = [torch.empty_like(value) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, value.contiguous())
    return torch.stack(gathered)


def verify_complete_copies(master_tensors, replica_tensors, replica_logical, masters):
    """Exact comparison of ALL weight/scale elements of EVERY populated replica.

    Broadcast one complete canonical expert at a time, not all GLM weights.
    Byte views preserve FP8 bits and FP32 scale bits, even for strided pools.
    Padding outside tensor elements is not initialized data and is not compared.
    Every rank follows the same collective order, including idle ranks.
    """
    rank, world = dist.get_rank(), dist.get_world_size()
    slots_by_rank = [None] * world
    dist.all_gather_object(slots_by_rank, replica_logical)
    logicals = sorted(
        {logical for slots in slots_by_rank for logical in slots if logical >= 0}
    )
    checked_bytes, checked_slots = 0, 0
    exact = True
    for master, replica in zip(master_tensors, replica_tensors):
        row_shape = master.shape[1:]
        bytes_per_row = master[0].numel() * master.element_size()
        for logical in logicals:
            owner, local = divmod(logical, masters)
            if rank == owner:
                expected = master[local].contiguous().view(torch.uint8).reshape(-1)
            else:
                expected = torch.empty(bytes_per_row, device="cuda", dtype=torch.uint8)
            dist.broadcast(expected, src=owner)
            for slot, copied_logical in enumerate(replica_logical):
                if copied_logical == logical:
                    copied = replica[slot].contiguous().view(torch.uint8).reshape(-1)
                    exact &= tuple(replica[slot].shape) == tuple(
                        row_shape
                    ) and torch.equal(copied, expected)
                    checked_bytes += bytes_per_row
                    checked_slots += 1
    require_all(exact, "UltraEP copied weights/scales differ from full owner tensors")
    return dict(
        full_copy_exact=True,
        local_bytes_compared=checked_bytes,
        local_tensor_rows_compared=checked_slots,
        global_logical_experts_checked=len(logicals),
    )


@torch.inference_mode()
def run(args, records):
    from ultra_ep import ppu

    dg = load_deep_gemm()
    names = (
        "Manager",
        "PhysicalWeightPool",
        "ReplicaWeightPool",
        "allocate_local_expert_tensors",
        "allocate_symmetric_expert_tensors",
    )
    missing = [name for name in names if not callable(getattr(ppu, name, None))]
    if missing:
        raise RuntimeError(f"Missing real ultra_ep.ppu APIs: {missing}")
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != args.ep:
        raise ValueError(f"--ep {args.ep} but torchrun launched {world} ranks")
    experts, hidden, intermediate, topk, replicas = (
        (288, 4096, 2048, 8, 8)
        if args.fixture == "glm53"
        else (8 * world, 512, 256, 2, 2)
    )
    masters, capacity = experts // world, args.capacity
    if capacity < args.tokens or args.tokens < 2 or masters + replicas > 128:
        raise ValueError(
            "Require capacity >= tokens >= 2 and <=128 local physical experts"
        )
    group = dist.group.WORLD
    rng = torch.Generator(device="cuda").manual_seed(args.seed + rank)
    raw1 = make_weights(masters, 2 * intermediate, hidden, rng)
    raw2 = make_weights(masters, hidden, intermediate, rng)
    l1, l2 = dg.transform_weights_for_fp8_mega_moe(
        raw1, raw2, quantization="channelwise"
    )
    allocation = ppu.allocate_symmetric_expert_tensors(group, [*l1, *l2], 0)
    w1, s1, w2, s2 = allocation.tensors
    manager = ppu.Manager(group, masters, replicas, min_tokens_per_replica=1)
    manager.register_physical_weight_pool(
        0,
        ppu.PhysicalWeightPool(
            w1,
            s1,
            allocation.peer_base_ptrs,
            w2,
            s2,
            allocation.peer_base_ptrs,
            keepalive=(allocation.backing,),
        ),
    )
    copy1 = ppu.allocate_local_expert_tensors([w1, s1], replicas)
    copy2 = ppu.allocate_local_expert_tensors([w2, s2], replicas)
    replica = ppu.ReplicaWeightPool(
        *copy1.tensors, *copy2.tensors, keepalive=(copy1.backing, copy2.backing)
    )
    manager.register_replica_weight_pool(replica)
    replica_tensors = (replica.w1, replica.w1_scale, replica.w2, replica.w2_scale)
    require_all(
        all(
            copied.shape == (replicas, *master.shape[1:])
            and copied.stride()[1:] == master.stride()[1:]
            and copied.dtype == master.dtype
            and copied.device == master.device
            for master, copied in zip(allocation.tensors, replica_tensors)
        ),
        "UltraEP master/replica layout mismatch",
    )
    buffers = []
    try:
        for num_experts in (experts, (masters + replicas) * world):
            buffers.append(
                dg.get_symm_buffer_for_mega_moe(
                    group,
                    num_experts,
                    capacity,
                    topk,
                    hidden,
                    intermediate,
                    mma_type="fp8xfp8",
                    quantization="channelwise",
                )
            )
        source = (
            torch.randn(
                (capacity, hidden), device="cuda", dtype=torch.bfloat16, generator=rng
            )
            * 0.1
        )
        fixed_weights = torch.rand((capacity, topk), device="cuda", generator=rng)
        fixed_weights /= fixed_weights.sum(-1, keepdim=True)
        scale_before = None
        initial_placement = None
        scenarios = (
            "populate",
            "idle_rank",
            "scale_change",
            "reroute_new_hot_experts",
            "all_idle",
        )
        for iteration, scenario in enumerate(scenarios):
            tokens = args.tokens
            if scenario == "all_idle" or (
                scenario == "idle_rank" and rank == world - 1
            ):
                tokens = 0
            ids = torch.full((capacity, topk), -1, device="cuda", dtype=torch.int64)
            offset = masters if scenario == "reroute_new_hot_experts" else 0
            ids[:tokens] = torch.arange(topk, device="cuda") + offset
            weights = fixed_weights.clone()
            weights[tokens:] = 0
            if scenario == "scale_change":
                # Keep routes and inputs unchanged: stale replica scales must fail.
                raw1[1].mul_(1.25)
                raw2[1].mul_(0.75)
                s1.mul_(1.25)
                s2.mul_(0.75)
            baseline = native(
                dg,
                buffers[0],
                source[:tokens],
                ids[:tokens],
                weights[:tokens],
                (w1, s1),
                (w2, s2),
            )
            manager.update_placement(0, ids[:tokens])
            event = manager.weight_sync(0, async_finish=True)
            physical_ids = manager.reroute(0, ids[:tokens].clone())
            if not callable(getattr(event, "current_stream_wait", None)):
                raise AssertionError("weight_sync must return a real stream-wait event")
            event.current_stream_wait()
            placement = manager.placement(0)
            mapping = placement.physical_to_logical
            require_all(
                torch.is_tensor(physical_ids)
                and physical_ids.shape == ids[:tokens].shape
                and physical_ids.dtype == torch.int64
                and physical_ids.is_contiguous()
                and physical_ids.device == ids.device,
                "UltraEP reroute output contract mismatch",
            )
            valid = ids[:tokens] >= 0
            in_range = (physical_ids[valid] >= 0) & (
                physical_ids[valid] < (masters + replicas) * world
            )
            require_all(in_range.all().item(), "Out-of-range physical expert ID")
            require_all(
                torch.equal(mapping[physical_ids[valid]].long(), ids[:tokens][valid]),
                "Physical-to-logical routing does not round-trip",
            )
            if initial_placement is None:
                initial_placement = mapping.clone()
            if scenario == "reroute_new_hot_experts":
                require_all(
                    not torch.equal(mapping, initial_placement),
                    "Placement did not change for new hot experts",
                )
            replica_mask = valid & (
                physical_ids.remainder(masters + replicas) >= masters
            )
            remote_mask = replica_mask & (
                physical_ids.div(masters + replicas, rounding_mode="floor")
                != ids[:tokens].div(masters, rounding_mode="floor")
            )
            counts = torch.stack((replica_mask.sum(), remote_mask.sum())).long()
            dist.all_reduce(counts)
            if scenario != "all_idle":
                require_all(
                    counts[0].item() > 0,
                    "No real assignments were rerouted to replicas",
                )
                require_all(
                    counts[1].item() > 0, "No remote-owner replicas were exercised"
                )
            local_mapping = mapping[
                rank * (masters + replicas)
                + masters : (rank + 1) * (masters + replicas)
            ]
            replica_logical = local_mapping.cpu().tolist()
            torch.cuda.synchronize()  # copy reads below must follow the real sync event
            copy_report = verify_complete_copies(
                allocation.tensors, replica_tensors, replica_logical, masters
            )
            actual = native(
                dg,
                buffers[1],
                source[:tokens],
                physical_ids,
                weights[:tokens],
                (w1, s1),
                (w2, s2),
                replicas=(
                    (replica.w1, replica.w1_scale),
                    (replica.w2, replica.w2_scale),
                ),
            )
            expected = reference(
                gather(source), gather(ids), gather(weights), raw1, raw2, rank=rank
            )[rank, :tokens]
            torch.cuda.synchronize()
            numerical = dict(
                logical_vs_torch=check_output(baseline, expected, distributed=True),
                replica_vs_torch=check_output(actual, expected, distributed=True),
                replica_vs_logical=check_output(actual, baseline, distributed=True),
            )
            if scenario == "populate":
                scale_before = expected.clone()
            elif scenario == "scale_change":
                # Non-vacuity: the reference itself changes under changed scales.
                require_all(
                    not torch.equal(scale_before, expected),
                    "Scale mutation did not affect the fixture",
                )
            record = dict(
                status="passed",
                scope="isolated_moe_block_not_full_model",
                fixture=args.fixture,
                ep=world,
                rank=rank,
                E=experts,
                H=hidden,
                I=intermediate,
                topk=topk,
                seed=args.seed,
                capacity=capacity,
                tokens=tokens,
                scenario=scenario,
                replica_routes=counts[0].item(),
                remote_replica_routes=counts[1].item(),
                replica_logical=replica_logical,
                numerical=numerical,
                **copy_report,
            )
            records.append(record)
            print(json.dumps(record), flush=True)
    finally:
        torch.cuda.synchronize()
        for buffer in reversed(buffers):
            buffer.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ep", type=int, choices=(4, 8), required=True)
    parser.add_argument("--fixture", choices=("glm53", "smoke"), default="glm53")
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--capacity", type=int, default=32)
    parser.add_argument("--seed", type=int, default=112400)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    records = []
    status, failure = "failed", None
    try:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        torch.backends.cuda.matmul.allow_tf32 = False
        dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=600))
        run(args, records)
        status = "passed"
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        args.outdir.mkdir(parents=True, exist_ok=True)
        report = dict(
            status=status,
            error=failure,
            ep=args.ep,
            rank=rank,
            device=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            torch_version=torch.__version__,
            records=records,
        )
        (args.outdir / f"ultraep_fp8_ep{args.ep}_rank{rank}.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
