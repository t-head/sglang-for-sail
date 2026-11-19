"""Compare DCP output reduction with V projection before or after exchange.

Run with torchrun --standalone --nproc-per-node=8. The default 120 rows are
the flattened verify rows in the c80 trace, not 120 independent requests.
"""

import argparse
import json
import os
from datetime import timedelta
from statistics import median

import torch
import torch.distributed as dist


def check_result(output, expected, group):
    # Agree before the next collective, including on numerical failure.
    ok = torch.isclose(output.float(), expected, atol=0.025, rtol=0.025).all()
    flag = ok.to(torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group.device_group)
    if not flag.item():
        torch.testing.assert_close(output.float(), expected, atol=0.025, rtol=0.025)
        raise AssertionError("DCP output differed from the oracle on another rank")


@torch.inference_mode()
def run(rows, group, trials, replays):
    from sglang.srt.layers.dcp import comm

    rank = group.rank_in_group
    heads, local_heads, latent_dim, value_dim = 96, 12, 512, 128
    owned = slice(rank * local_heads, (rank + 1) * local_heads)
    rng = torch.Generator(device="cuda").manual_seed(101 + rank)
    output = torch.empty(rows, 128, latent_dim, device="cuda", dtype=torch.bfloat16)[
        :, :heads
    ]
    lse = torch.empty(128, rows, device="cuda", dtype=torch.float32).t()[:, :heads]
    local_weight = torch.randn(
        local_heads,
        latent_dim,
        value_dim,
        device="cuda",
        dtype=torch.bfloat16,
        generator=rng,
    )
    local_weight.mul_(latent_dim**-0.5)
    # As in serving, gather the full V weights once, outside the measured step.
    weight = group.all_gather(local_weight, dim=0)
    parts = [
        torch.empty_like(output, memory_format=torch.contiguous_format)
        for _ in range(8)
    ]
    stats = [
        torch.empty_like(lse, memory_format=torch.contiguous_format) for _ in range(8)
    ]

    def update():
        output.normal_(generator=rng)
        lse.normal_(generator=rng)
        # One empty shard for a row, never an all-empty row.
        lse[0, rank] = -float("inf")
        dist.all_gather(parts, output.contiguous(), group=group.device_group)
        dist.all_gather(stats, lse.contiguous(), group=group.device_group)
        probs = torch.stack(stats)[:, :, owned].softmax(dim=0).unsqueeze(-1)
        latent = (torch.stack(parts)[:, :, owned].float() * probs).sum(dim=0)
        return torch.bmm(latent.transpose(0, 1), local_weight.float()).transpose(0, 1)

    def project(part, w):
        return torch.bmm(part.transpose(0, 1), w).transpose(0, 1)

    def ag_rs():
        reduced = comm.cp_lse_ag_out_rs_mla(output, lse, group, is_lse_base_on_e=True)
        return project(reduced.transpose(0, 1), local_weight)

    def a2a(part):
        return comm.dcp_a2a_lse_reduce(
            part, lse, group, is_lse_base_on_e=True, comm_backend="a2a"
        )

    def postproject():
        return project(a2a(output), local_weight)

    def preproject():
        return a2a(project(output, weight))

    fns = {"ag_rs": ag_rs, "a2a": postproject, "a2a_v_preproject": preproject}
    graphs = {}
    expected = update()
    for name, fn in fns.items():
        for _ in range(3):
            actual = fn()
        check_result(actual, expected, group)
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = fn()
        graphs[name] = (graph, captured)

    for _ in range(3):
        expected = update()
        for graph, captured in graphs.values():
            graph.replay()
            check_result(captured, expected, group)

    samples = {name: [] for name in graphs}
    for trial in range(trials):
        order = list(graphs)
        if trial % 2:
            order.reverse()
        for name in order:
            graph, _ = graphs[name]
            for _ in range(8):
                graph.replay()
            torch.cuda.synchronize()
            dist.barrier()
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            for _ in range(replays):
                graph.replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000 / replays)

    rank_samples = [None] * 8
    dist.all_gather_object(rank_samples, samples)
    if rank == 0:
        result = {
            name: median(
                max(sample[name][i] for sample in rank_samples) for i in range(trials)
            )
            for name in graphs
        }
        print(
            json.dumps(
                {
                    "rows": rows,
                    "median_worst_rank_us": result,
                    "samples_us": rank_samples,
                }
            ),
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[120])
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--replays", type=int, default=50)
    args = parser.parse_args()
    if (
        min(*args.rows, args.trials, args.replays) < 1
        or int(os.environ.get("WORLD_SIZE", "1")) != 8
    ):
        parser.error("positive dimensions and an eight-rank torchrun are required")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    torch.cuda.set_stream(torch.cuda.Stream())
    import sglang.srt.distributed.parallel_state as ps
    from sglang.srt.runtime_context import get_parallel
    from sglang.test.test_utils import published_topology

    dist.init_process_group("gloo", timeout=timedelta(seconds=120))
    try:
        with published_topology(
            tp_size=8,
            dcp_size=8,
            dcp_comm_backend="a2a",
            dcp_replicate_q_proj=False,
            ranks={"world_rank": rank},
        ):
            ps._WORLD = ps.init_world_group(list(range(8)), rank, "nccl")
            get_parallel().override_permanently(world_group=ps._WORLD)
            group = ps.init_model_parallel_group(
                [list(range(8))],
                rank,
                "nccl",
                use_pynccl=True,
                use_custom_allreduce=False,
                use_mscclpp=False,
                use_torch_symm_mem_allreduce=False,
                group_name="dcp_output_probe",
            )
            try:
                group.pynccl_comm.disabled = False
                for rows in args.rows:
                    run(rows, group, args.trials, args.replays)
            finally:
                group.destroy()
                ps._WORLD.destroy()
                ps._WORLD = None
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
