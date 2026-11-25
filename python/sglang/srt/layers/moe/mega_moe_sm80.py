# SPDX-License-Identifier: Apache-2.0
"""PPU SM80 INT8 MegaMoE forward path and expert-weight preparation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from deep_gemm import SymmBuffer


SM80_INT8_MMA_TYPE = "int8xint8"


def require_sm80_int8_mega_moe_apis() -> None:
    try:
        import deep_gemm
    except ImportError as exc:
        raise ValueError(
            "--moe-a2a-backend megamoe on PPU SM80 requires the PPU DeepGEMM build"
        ) from exc

    required_apis = (
        "get_symm_buffer_for_mega_moe",
        "mega_moe_pre_dispatch",
        "int8_int8_mega_moe",
        "transform_weights_for_mega_moe",
    )
    missing = [name for name in required_apis if not hasattr(deep_gemm, name)]
    if missing:
        raise ValueError(
            "--moe-a2a-backend megamoe on PPU SM80 requires the INT8 DeepGEMM "
            "entry points; missing: " + ", ".join(missing)
        )


def run_sm80_mega_routed(
    experts,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    buf: SymmBuffer,
    num_tokens: int,
    *,
    hidden_size: int,
    activation: str,
    activation_clamp: Optional[float] = None,
) -> torch.Tensor:
    import deep_gemm

    deep_gemm.mega_moe_pre_dispatch(
        hidden_states.contiguous(),
        topk_ids.contiguous(),
        topk_weights.contiguous(),
        buf.x,
        buf.x_sf,
        buf.topk_idx,
        buf.topk_weights,
        num_tokens=num_tokens,
        group_size=0,
        mma_type=SM80_INT8_MMA_TYPE,
    )
    y = torch.empty(
        (num_tokens, hidden_size),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )
    deep_gemm.int8_int8_mega_moe(
        y,
        (experts.w13_weight, experts.w13_weight_scale),
        (experts.w2_weight, experts.w2_weight_scale),
        buf,
        activation=activation,
        activation_clamp=activation_clamp,
        fast_math=True,
    )
    return y


def build_sm80_mega_moe_experts_weights(experts) -> None:
    if getattr(experts, "_mega_moe_weights_built", False):
        return

    from deep_gemm import transform_weights_for_mega_moe

    l1, l2 = transform_weights_for_mega_moe(
        (experts.w13_weight.data, experts.w13_weight_scale.data.squeeze(-1)),
        (experts.w2_weight.data, experts.w2_weight_scale.data.squeeze(-1)),
    )
    # Keep the packed weights and 2D kernel scales in registered parameters.
    # This in-place packing does not support online checkpoint reloads.
    experts.w13_weight.data = l1[0]
    experts.w13_weight_scale.data = l1[1]
    experts.w2_weight.data = l2[0]
    experts.w2_weight_scale.data = l2[1]
    experts._mega_moe_mma_type = SM80_INT8_MMA_TYPE
    experts._mega_moe_weights_built = True
