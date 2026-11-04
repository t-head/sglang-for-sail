# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Adapted from vLLM models/step5/nvidia/ops/triton/router_bias.py.

Preserves sigmoid probabilities from the selected logits, ordered normalization,
and scale. This leaf uses one fixed launch configuration, not vLLM's optional
driver-launch cache or autotuner. Public metadata validation owns buffer safety.

One CTA per token row and one warp per CTA, so the kernel is a single serial
latency chain: eight dependent reductions, each followed by a dependent reload
of the winning logit. Decode rows are few, so nothing hides that chain -- it is
what the duration is made of. Each round therefore reduces an int64 key that
packs the score's order-preserving bit pattern above the inverted expert id, so
one plain ``tl.max`` yields both the winner and its tie-broken index; the
``return_indices`` reduction computes the same thing far slower. A single warp
keeps every round inside a warp shuffle rather than a shared-memory round trip.
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

    # Order-preserving map from the score's bit pattern to an unsigned rank:
    # non-negative floats keep their order with the sign bit set, negative ones
    # are inverted. Adding 0.0 first folds -0.0 into +0.0, which compare equal
    # but rank differently once packed. The inverted expert id occupies the low
    # nine bits, so a tie resolves toward the smaller id exactly as the
    # left-tie-break reduction did, and every real key stays non-negative.
    scores = scores + 0.0
    bits = scores.to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF
    ordered = tl.where(bits >= 0x80000000, bits ^ 0xFFFFFFFF, bits | 0x80000000)
    key = (ordered << 9) | (511 - expert).to(tl.int64)

    weights = tl.zeros((8,), dtype=tl.float32)
    indices = tl.zeros((8,), dtype=tl.int32)
    total = 0.0
    slot = tl.arange(0, 8)
    for k in tl.static_range(8):
        best = tl.max(key, axis=0)
        index = (511 - (best & 511)).to(tl.int32)
        # NaN rows are overwritten below, but still execute the reduction. A
        # masked padding winner must never read outside the logical logits row.
        selected_logit = tl.load(
            logits + row * logits_stride + index,
            mask=(index >= 0) & (index < 352),
            other=0,
        ).to(tl.float32)
        # Bias affects selection only. Subtracting it from the rounded score
        # can erase small probabilities and change normalized routing weights.
        selected_probability = tl.sigmoid(selected_logit)
        weights = tl.where(slot == k, selected_probability, weights)
        indices = tl.where(slot == k, index, indices)
        total += selected_probability
        key = tl.where(expert == index, -1, key)

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
        num_stages=2,
    )
