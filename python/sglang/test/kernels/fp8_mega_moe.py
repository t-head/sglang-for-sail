"""Shared channelwise FP8 MegaMoE fixtures and numerical oracles."""

import datetime
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

MAX_DIFF = 0.001  # sum((actual-reference)^2) / sum(actual^2+reference^2)
DEEP_GEMM_APIS = (
    "transform_weights_for_fp8_mega_moe",
    "get_symm_buffer_for_mega_moe",
    "mega_moe_pre_dispatch",
    "fp8_fp8_mega_moe",
)


def load_deep_gemm():
    import deep_gemm

    missing = [
        name for name in DEEP_GEMM_APIS if not callable(getattr(deep_gemm, name, None))
    ]
    if missing:
        raise RuntimeError(f"PR336 requires the PPU DeepGEMM FP8 APIs: {missing}")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9):
        raise RuntimeError("PR336 numerical evidence requires a PPU SM89 device")
    return deep_gemm


class SingleRankGroup:
    """Real NCCL group, without torchrun, ports or invented multi-rank peers."""

    def __init__(self):
        self.owned = False
        self.directory = None

    def open(self):
        if dist.is_initialized():
            if dist.get_world_size() != 1:
                raise RuntimeError("The registered PR336 UT requires exactly one rank")
        else:
            self.directory = tempfile.TemporaryDirectory(prefix="pr336-fp8-")
            dist.init_process_group(
                "nccl",
                init_method=(Path(self.directory.name) / "store").as_uri(),
                rank=0,
                world_size=1,
                timeout=datetime.timedelta(seconds=600),
            )
            self.owned = True
        return dist.group.WORLD

    def close(self):
        if self.owned and dist.is_initialized():
            dist.destroy_process_group()
        if self.directory is not None:
            self.directory.cleanup()


def make_weights(experts, n, k, generator):
    # Bound temporary memory to one expert; every channel has its own FP32 scale.
    values = torch.empty((experts, n, k), device="cuda", dtype=torch.float8_e4m3fn)
    for expert in range(experts):
        values[expert] = torch.randint(
            -8, 9, (n, k), device="cuda", generator=generator
        ).to(torch.float8_e4m3fn)
    scales = (
        torch.rand((experts, n, 1), device="cuda", generator=generator) + 0.25
    ) / 64
    return values, scales


def quantize_rows(values):
    """Independent Torch dynamic per-token quantization, not a DeepGEMM helper."""
    values = values.float()
    scales = values.abs().amax(-1, keepdim=True).clamp_min(1e-10) / 448.0
    return (values * scales.reciprocal()).to(torch.float8_e4m3fn), scales


def quantized_values(values):
    quantized, scales = quantize_rows(values)
    return quantized.float() * scales


def reference(source, ids, routing_weights, raw1, raw2, *, rank=0, clamp=10.0):
    """FP32 Torch GEMMs over canonical [gate; up] weights, then BF16 epilogues.

    source/ids/weights are [world, capacity, ...]. Each owner computes only its
    logical experts, then all-reduces contributions. No packed/replica weights
    or fused dispatch, quantization, GEMM or activation APIs enter this oracle.
    """
    world, capacity, hidden = source.shape
    experts = raw1[0].shape[0]
    source = quantized_values(source)
    result = torch.zeros(
        (world * capacity, hidden), device=source.device, dtype=torch.float32
    )
    limit = (
        None
        if clamp is None
        else torch.tensor(clamp, device=source.device).bfloat16().float()
    )
    for local_expert in range(experts):
        logical = rank * experts + local_expert
        owner, row, slot = (ids == logical).nonzero(as_tuple=True)
        if not row.numel():
            continue
        w1 = raw1[0][local_expert].float() * raw1[1][local_expert]
        projection = (source[owner, row] @ w1.T).bfloat16().float()
        gate, up = projection.chunk(2, -1)
        if limit is not None:
            gate = gate.clamp(max=limit)
            up = up.clamp(-limit, limit)
        activated = (F.silu(gate) * up).bfloat16()
        w2 = raw2[0][local_expert].float() * raw2[1][local_expert]
        down = (quantized_values(activated) @ w2.T).bfloat16().float()
        result.index_add_(
            0, owner * capacity + row, down * routing_weights[owner, row, slot, None]
        )
    if world > 1:
        dist.all_reduce(result)
    return result.reshape(world, capacity, hidden).bfloat16()


def native(
    dg,
    buffer,
    source,
    ids,
    routing_weights,
    l1,
    l2,
    *,
    clamp=10.0,
    replicas=None,
    out=None,
):
    if out is None:
        out = torch.empty_like(source)
    dg.mega_moe_pre_dispatch(
        source.contiguous(),
        ids.contiguous(),
        routing_weights.contiguous(),
        buffer.x,
        buffer.x_sf,
        buffer.topk_idx,
        buffer.topk_weights,
        num_tokens=source.shape[0],
        group_size=0,
        mma_type="fp8xfp8",
    )
    options = (
        {}
        if replicas is None
        else {"l1_weights_replicas": replicas[0], "l2_weights_replicas": replicas[1]}
    )
    dg.fp8_fp8_mega_moe(
        out,
        l1,
        l2,
        buffer,
        activation="swiglu",
        activation_clamp=clamp,
        fast_math=True,
        **options,
    )
    return out


def error_metrics(actual, expected):
    if actual.shape != expected.shape or actual.dtype != torch.bfloat16:
        raise AssertionError(
            f"Incorrect output contract: {actual.shape}, {actual.dtype}, {expected.shape}"
        )
    a, b = actual.double(), expected.double()
    denominator = (a.square() + b.square()).sum()
    nse = (a - b).square().sum() / denominator.clamp_min(1e-30)
    max_abs = (a - b).abs().max() if a.numel() else a.new_zeros(())
    finite = (
        torch.isfinite(a).all() & torch.isfinite(b).all() & torch.isfinite(nse)
    ).to(torch.int32)
    return torch.stack((nse, max_abs)), finite


def check_output(actual, expected, *, distributed=False):
    metrics, finite = error_metrics(actual, expected)
    if distributed:
        dist.all_reduce(metrics, op=dist.ReduceOp.MAX)
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if not finite.item() or metrics[0].item() >= MAX_DIFF:
        raise AssertionError(
            f"Nonfinite or inaccurate FP8 MegaMoE: {metrics.tolist()}, finite={finite.item()}"
        )
    return {
        "normalized_squared_error": metrics[0].item(),
        "max_abs_error": metrics[1].item(),
    }
