"""Multi-GPU test of the one-kernel DCP reduce (dcp_push_combine): every rank
holds [B, H, 128] partials of all heads with their LSE and gets back the LSE
combine of its own H / N heads, as gathering every rank's partials and running
the Triton combine does. Covers +0.0 payload words (the lamport marker), rows
with no local key (+inf LSE), zero LSE, uneven batches, and repeated calls
through both push-plane phases.

Usage::

    python test/registered/kernels/ops/attention/test_dcp_push_combine.py   # relaunches under torchrun (8 GPUs)
"""

from __future__ import annotations

import atexit
import logging
import os

import pytest
import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.kernels.jit.utils import cache_once
from sglang.kernels.ops.attention import dcp_push_combine as pc
from sglang.kernels.ops.attention.dcp_kernels import dcp_lse_combine_triton
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)
from sglang.srt.runtime_context import SpawnRanks, get_parallel
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main
from sglang.test.parallel_groups import publish

register_cuda_ci(est_time=60, stage="extra-b", runner_config="8-gpu-h200")

H = 96  # Kimi-K3.1 MLA heads, all held by every DCP rank
MAX_ROWS = 1024


@cache_once
def _setup() -> ps.GroupCoordinator:
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")
    publish(
        ServerArgs(model_path="dummy", tp_size=world_size),
        role="test",
        ranks=SpawnRanks(world_rank=local_rank),
    )
    coord = ps.init_world_group(
        ranks=list(range(world_size)), local_rank=local_rank, backend="nccl"
    )
    ps._WORLD = coord
    get_parallel().override_permanently(world_group=coord)
    atexit.register(dist.destroy_process_group)
    logging.disable(logging.INFO)
    comm = CustomAllReduceV2(
        coord.cpu_group,
        torch.device(f"cuda:{local_rank}"),
        max_pull_size=0,
        max_push_size=MAX_ROWS * (H // world_size) * pc.SLOT_BYTES_PER_ROW,
        max_push_blocks=pc.max_active_blocks(world_size, local_rank),
    )
    assert not comm.disabled
    pc.register_comm(comm.obj)
    return coord


def _reference(coord, v, lse):
    n, rank = coord.world_size, coord.rank_in_group
    hpr = v.shape[1] // n
    all_v = [torch.empty_like(v) for _ in range(n)]
    all_lse = [torch.empty_like(lse) for _ in range(n)]
    dist.all_gather(all_v, v, group=coord.device_group)
    dist.all_gather(all_lse, lse, group=coord.device_group)
    recv_v = torch.stack([x[:, rank * hpr : (rank + 1) * hpr] for x in all_v])
    recv_lse = torch.stack([x[:, rank * hpr : (rank + 1) * hpr] for x in all_lse])
    out, _ = dcp_lse_combine_triton(recv_v.contiguous(), recv_lse.contiguous())
    return out.reshape(v.shape[0], -1)


@pytest.mark.parametrize("batch", [1, 7, 64, 384, MAX_ROWS])
@pytest.mark.parametrize("calls", [3])
def test_matches_gather_and_combine(batch, calls):
    coord = _setup()
    n, rank = coord.world_size, coord.rank_in_group
    g = torch.Generator(device="cuda").manual_seed(batch * 131 + rank)
    for call in range(calls):  # odd and even push-plane phases
        v = (
            torch.randn(batch, H, pc.HEAD_DIM, device="cuda", generator=g) * (rank + 1)
        ).to(torch.bfloat16)
        v[:, :, :4] = 0  # +0.0 payload words must not read as "not arrived"
        lse = torch.randn(batch, H, device="cuda", generator=g) * 4
        lse[::5, rank] = float("inf")  # no local key
        if call == 1:
            lse[: min(batch, 3)] = 0.0
        out = pc.dcp_push_combine(n, v, lse)
        ref = _reference(coord, v, lse)
        assert torch.isfinite(out).all()
        torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


def test_strided_inputs():
    """Decode hands over the W_UV projection head-major and the LSE as a slice
    of the padded-head buffer; neither is contiguous."""
    coord = _setup()
    n, rank = coord.world_size, coord.rank_in_group
    g = torch.Generator(device="cuda").manual_seed(11 + rank)
    batch = 64
    v = torch.randn(H, batch, pc.HEAD_DIM, device="cuda", generator=g)
    v = v.to(torch.bfloat16).transpose(0, 1)
    lse = (torch.randn(batch, H + 32, device="cuda", generator=g) * 4)[:, :H]
    out = pc.dcp_push_combine(n, v, lse)
    ref = _reference(coord, v.contiguous(), lse.contiguous())
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


def test_cuda_graph_replay():
    coord = _setup()
    n, rank = coord.world_size, coord.rank_in_group
    g = torch.Generator(device="cuda").manual_seed(7 + rank)
    v = torch.randn(64, H, pc.HEAD_DIM, device="cuda", generator=g).to(torch.bfloat16)
    lse = torch.randn(64, H, device="cuda", generator=g)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        pc.dcp_push_combine(n, v, lse)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = pc.dcp_push_combine(n, v, lse)
    for _ in range(5):  # replays alternate the push-plane phase
        v.copy_(torch.randn(v.shape, device="cuda", generator=g).to(torch.bfloat16))
        graph.replay()
        torch.testing.assert_close(out, _reference(coord, v, lse), rtol=2e-2, atol=2e-2)


def test_changing_batch_skewed_replay():
    """Back-to-back calls with changing batch sizes (e.g. B=26 -> B=120, H=96,
    DCP=8) launch different block counts, leaving tail blocks on the prior
    phase. With rank skew and no intervening collectives, fixed capacity-derived
    payload/LSE regions must prevent any cross-phase byte overlap."""
    coord = _setup()
    n, rank = coord.world_size, coord.rank_in_group
    g = torch.Generator(device="cuda").manual_seed(97 + rank)
    batches = (26, 120)
    vs = {
        b: torch.randn(b, H, pc.HEAD_DIM, device="cuda", generator=g).to(torch.bfloat16)
        for b in batches
    }
    lses = {b: torch.randn(b, H, device="cuda", generator=g) * 4 for b in batches}
    skew_buf = torch.eye(512, device="cuda", dtype=torch.bfloat16)

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for b in batches:
            pc.dcp_push_combine(n, vs[b], lses[b])
    torch.cuda.synchronize()
    graphs = {}
    outs = {}
    for b in batches:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outs[b] = pc.dcp_push_combine(n, vs[b], lses[b])
        graphs[b] = graph

    seq = (26, 120, 26, 120, 120, 26, 120)
    for round_idx in range(4):
        steps = []
        for step, b in enumerate(seq):
            v_step = (
                torch.randn(b, H, pc.HEAD_DIM, device="cuda", generator=g) * (rank + 1)
            ).to(torch.bfloat16)
            lse_step = torch.randn(b, H, device="cuda", generator=g) * 4
            if step % 2 == 0:
                lse_step[::3, rank] = float("inf")
            steps.append((b, v_step, lse_step))
        actual = [
            torch.empty((b, H // n * pc.HEAD_DIM), device="cuda", dtype=torch.bfloat16)
            for b, _, _ in steps
        ]
        dist.barrier()
        # Delay a subset of ranks on the device stream with no intervening
        # collectives between back-to-back replays so fast ranks enter B=120
        # while slow ranks are still consuming B=26.
        if (rank + round_idx) % 2 == 1:
            for _ in range(12):
                skew_buf = skew_buf @ skew_buf
        for idx, (b, v_step, lse_step) in enumerate(steps):
            vs[b].copy_(v_step)
            lses[b].copy_(lse_step)
            graphs[b].replay()
            actual[idx].copy_(outs[b])
        # Verify all outputs after the unbarriered sequence completes.
        for out, (_, v_step, lse_step) in zip(actual, steps):
            ref = _reference(coord, v_step, lse_step)
            assert torch.isfinite(out).all()
            torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=[8])
