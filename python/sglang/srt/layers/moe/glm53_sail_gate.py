"""PPU MegaGate admission policy for the GLM-5.3 text checkpoint."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import torch

from sglang.kernels import get_fused_op_backend
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_exec

if TYPE_CHECKING:
    from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
    from sglang.srt.models.deepseek_v2 import DeepseekV2MoE


@functools.lru_cache(maxsize=None)
def _is_sail_ppu(device: torch.device) -> bool:
    return "ZW" in torch.cuda.get_device_name(
        device
    ) and torch.cuda.get_device_capability(device) in ((8, 0), (8, 9))


def can_route(
    moe: DeepseekV2MoE,
    hidden: torch.Tensor,
    dispatch_info: ExpertLocationDispatchInfo | None,
) -> bool:
    from sglang.srt.layers.moe.topk import TopKOutputFormat

    # The DeepGEMM binding has no fake implementation for torch.compile.
    if torch.compiler.is_compiling():
        return False
    if (
        not envs.SGLANG_OPT_DEEPGEMM_MEGA_GATE.get()
        or moe.config.model_type != "glm_moe_dsa"
        or moe.is_hash
        or get_fused_op_backend() is not None
    ):
        return False
    # Waterfill appends the shared slot after TopK has reset its fused count.
    if moe.topk.enable_waterfill or moe.num_fused_shared_experts:
        return False
    # These modes need the original routing transform or its LP dispatch solver.
    if dispatch_info is not None and dispatch_info.ep_dispatch_algorithm in (
        "fake",
        "lp",
    ):
        return False
    cfg = moe.topk.topk_config
    weight = moe.gate.weight
    bias = moe.gate.e_score_correction_bias
    return (
        hidden.is_cuda
        and _is_sail_ppu(hidden.device)
        and hidden.ndim == 2
        and hidden.shape[1] == 6144
        and 0 < hidden.shape[0] <= 1 << 20
        and hidden.is_contiguous()
        and hidden.storage_offset() % 8 == 0
        and hidden.device == weight.device
        and hidden.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and weight.shape == (256, 6144)
        and weight.is_contiguous()
        and weight.storage_offset() % 8 == 0
        and bias is not None
        and bias.device == hidden.device
        and bias.dtype == torch.float32
        and bias.shape == (256,)
        and bias.is_contiguous()
        and moe.gate.e_score_correction_bias_vl is None
        and cfg.output_format == TopKOutputFormat.STANDARD
        and cfg.scoring_func == "sigmoid"
        and cfg.top_k == 8
        and cfg.renormalize
        and cfg.num_expert_group == cfg.topk_group == 1
        and not cfg.num_shared_topk_slots
        and cfg.custom_routing_function is None
        and not cfg.torch_native
        and not get_exec().moe.enable_eplb
        and not get_exec().deterministic.enable_deterministic_inference
        and not envs.SGLANG_SIMULATE_UNIFORM_EXPERTS.get()
        and not envs.SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS.get()
    )
