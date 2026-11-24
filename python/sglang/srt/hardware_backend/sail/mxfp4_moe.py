"""SAIL FP4×FP4 expert execution for Step 5's standard MoE dispatch."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.ops.moe.ep_moe_kernels import (
    ep_scatter_from_psum,
    fused_moe_dispatch_index,
    post_reorder_deepgemm,
)
from sglang.kernels.ops.quantization.sail_mxfp4_quant import sail_mxfp4_quantize
from sglang.srt.utils import round_up

if TYPE_CHECKING:
    from sglang.kernels.ops.activation.silu_clamp_and_mul import SiluClampAndMulOp


def sail_mxfp4_moe(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    gate_up: tuple[torch.Tensor, torch.Tensor],
    down: tuple[torch.Tensor, torch.Tensor],
    activation: SiluClampAndMulOp,
    output: torch.Tensor,
    routed_scaling_factor: float,
) -> None:
    from deep_gemm import m_grouped_gemm_fp4_fp4_bf16_nt_nopad

    num_tokens, hidden_size = hidden_states.shape
    if num_tokens == 0:
        return

    topk_ids = topk_ids.contiguous()
    topk_weights = topk_weights.contiguous()
    num_experts, gate_up_width, _ = gate_up[0].shape
    counts, _ = fused_moe_dispatch_index(topk_ids, num_experts, num_tokens)
    ends = counts.cumsum(0, dtype=torch.int32)
    # Keep a static allocation size and give the unused tail to the last
    # expert. DeepGEMM's nopad ABI requires non-negative, grouped expert IDs.
    rows = round_up(topk_ids.numel(), 128)
    ends[-1] = rows
    packed_input = hidden_states.new_zeros((rows, hidden_size))
    expert_starts = torch.empty_like(ends)
    expert_ids = ends.new_empty(rows)
    src2dst = torch.empty_like(topk_ids, dtype=torch.int32)
    ep_scatter_from_psum(
        hidden_states,
        None,
        topk_ids,
        ends,
        expert_starts,
        packed_input,
        None,
        expert_ids,
        src2dst,
    )

    gate_up_output = hidden_states.new_empty((rows, gate_up_width))
    m_grouped_gemm_fp4_fp4_bf16_nt_nopad(
        sail_mxfp4_quantize(packed_input), gate_up, None, gate_up_output, expert_ids
    )
    del packed_input
    # Preserve Step 5's post-SiLU clamp and the BF16 rounding of its product.
    down_input = sail_mxfp4_quantize(activation(gate_up_output))
    del gate_up_output
    down_output = hidden_states.new_empty((rows, hidden_size))
    m_grouped_gemm_fp4_fp4_bf16_nt_nopad(
        down_input, down, None, down_output, expert_ids
    )
    post_reorder_deepgemm(
        down_output,
        output,
        src2dst,
        topk_ids,
        topk_weights,
        topk_ids.shape[1],
        num_tokens,
        hidden_size,
        routed_scaling_factor,
    )
