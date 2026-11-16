# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Adapted from vLLM models/step5/nvidia/ops/triton/router_bias.py.

Preserves sigmoid probabilities from the selected logits, ordered normalization,
and scale. This leaf uses one fixed launch configuration, not vLLM's optional
driver-launch cache or autotuner. Public metadata validation owns buffer safety.

One CTA per token row and one warp per CTA. Each selection round takes a plain
float max, then finds the smallest expert id among equal scores with an int32
min reduction. That keeps the left tie-break without an index-returning
reduction or a 64-bit key. The eight selected logits are loaded together after
selection, so global-load and sigmoid latency do not extend the dependent
selection chain. A single warp keeps every round inside warp shuffles rather
than a shared-memory round trip.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _step5_router_topk_kernel(
    logits,
    bias,
    weights_out,
    ids_out,
    logits_stride,
    weights_stride,
    ids_stride,
    num_token_non_padded,
    HAS_PADDING: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    expert = tl.arange(0, 512)
    valid = expert < 352
    gating = tl.load(logits + row * logits_stride + expert, mask=valid, other=0)
    probability = tl.sigmoid(gating.to(tl.float32))
    correction = tl.load(bias + expert, mask=valid, other=0).to(tl.float32)
    correction = tl.where(valid, correction, -float("inf"))
    scores = tl.where(valid, probability + correction, -float("inf"))
    has_bad = tl.max((gating != gating).to(tl.int32), axis=0) > 0

    # Folding -0.0 into +0.0 lets score equality implement the left tie-break:
    # the second reduction selects the smallest expert id among equal scores.
    scores = scores + 0.0

    indices = tl.zeros((8,), dtype=tl.int32)
    slot = tl.arange(0, 8)
    for k in tl.static_range(8):
        best = tl.max(scores, axis=0)
        index = tl.min(tl.where(scores == best, expert, 512), axis=0).to(tl.int32)
        indices = tl.where(slot == k, index, indices)
        scores = tl.where(expert == index, -float("inf"), scores)

    # NaN rows are overwritten below, but still execute the load. A padding
    # winner must never read outside the logical logits row.
    selected_logits = tl.load(
        logits + row * logits_stride + indices,
        mask=(indices >= 0) & (indices < 352),
        other=0,
    ).to(tl.float32)
    # Bias affects selection only. Subtracting it from the rounded score can
    # erase small probabilities and change normalized routing weights.
    weights = tl.sigmoid(selected_logits)
    total = 0.0
    for k in tl.static_range(8):
        total += tl.sum(tl.where(slot == k, weights, 0.0), axis=0)

    weights = weights / (total + 1e-20)
    weights = weights * 3.0
    weights = tl.where(has_bad, 0.0, weights)
    indices = tl.where(has_bad, 0, indices)
    tl.store(weights_out + row * weights_stride + slot, weights)
    if HAS_PADDING:
        indices = tl.where(row < tl.load(num_token_non_padded), indices, -1)
    tl.store(ids_out + row * ids_stride + slot, indices)


def _step5_router_topk(
    *,
    logits: torch.Tensor,
    bias: torch.Tensor,
    weights: torch.Tensor,
    ids: torch.Tensor,
    num_token_non_padded: torch.Tensor | None = None,
) -> None:
    _step5_router_topk_kernel[(logits.shape[0],)](
        logits,
        bias,
        weights,
        ids,
        logits.stride(0),
        weights.stride(0),
        ids.stride(0),
        num_token_non_padded,
        HAS_PADDING=num_token_non_padded is not None,
        num_warps=1,
        num_stages=1,
    )
