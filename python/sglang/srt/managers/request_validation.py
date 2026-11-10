"""Validate client-controlled sizes before request expansion and inference."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from sglang.srt.sampling.sampling_params import TOP_K_ALL, SamplingParams

if TYPE_CHECKING:
    from sglang.srt.managers.io_struct import GenerateReqInput


def validate_request_limits(max_parallel_samples: int, max_batch_outputs: int) -> None:
    for name, value in (
        ("max-parallel-samples", max_parallel_samples),
        ("max-batch-outputs", max_batch_outputs),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"--{name} must be a positive integer.")


def validate_input_ids(
    input_ids, vocab_size: int, *, allow_empty: bool = False
) -> None:
    """Check raw client IDs, before multimodal processors introduce padding IDs."""
    if input_ids is None:
        return
    # A session may reconstruct the entire prompt from its existing history.
    if isinstance(input_ids, list) and not input_ids and allow_empty:
        return
    if not isinstance(input_ids, list) or not input_ids:
        raise ValueError(
            "input_ids must be a non-empty list of token IDs or sequences."
        )
    sequences = input_ids if isinstance(input_ids[0], list) else [input_ids]
    for sequence in sequences:
        if not isinstance(sequence, list) or not sequence:
            raise ValueError("Each input_ids sequence must be a non-empty list.")
        for token_id in sequence:
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise ValueError("input_ids must contain integers.")
            if not 0 <= token_id < vocab_size:
                raise ValueError(
                    f"input_ids contains token ID {token_id}; "
                    f"valid range is [0, {vocab_size})."
                )


def sampling_mask_error(
    *,
    top_k: int,
    is_disaggregated: bool,
    enabled: bool,
    capacity: int,
    speculative: bool,
    sampling_backend: str,
) -> str | None:
    if is_disaggregated and not enabled:
        return (
            "return_sampling_mask requires SGLANG_ENABLE_DISAGG_SAMPLING_MASK=1 "
            "on both prefill and decode servers."
        )
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        return "return_sampling_mask requires top_k to be a positive integer."
    if top_k == TOP_K_ALL:
        return (
            "return_sampling_mask requires finite top_k; top_p-only sampling "
            "is valid but can return huge masks in the tail, blowing up "
            "metadata, so we need a safety cap."
        )
    if speculative:
        return "return_sampling_mask is not supported with speculative decoding."
    if sampling_backend == "ascend":
        return "return_sampling_mask is not supported with the ascend sampling backend."
    # Match the scheduler's admission limit. Cutoff ties can still exceed it;
    # the sampler and PD metadata writer validate the realized support.
    if top_k > capacity:
        return (
            f"return_sampling_mask requires top_k <= {capacity}; got {top_k}. "
            "Lower top_k or increase --sampling-mask-max-tokens."
        )
    return None


def validate_generation_request(
    obj: GenerateReqInput,
    *,
    vocab_size: int,
    max_parallel_samples: int,
    max_batch_outputs: int,
    preferred_sampling_params: dict | None,
    is_disaggregated: bool,
    sampling_mask_enabled: bool,
    sampling_mask_capacity: int,
    speculative: bool,
    sampling_backend: str,
    sampling_params_class=SamplingParams,
) -> None:
    # Determine the unexpanded input count using the same rules as normalization.
    shape = copy.copy(obj)
    shape._validate_inputs()
    shape._determine_batch_size()
    if shape.batch_size < 1:
        raise ValueError("A generation request must contain at least one input.")
    params = obj.sampling_params
    if params is None:
        params = [{}]
    elif isinstance(params, dict):
        params = [params]
    elif (
        not isinstance(params, list)
        or shape.is_single
        or len(params) != shape.batch_size
        or not all(isinstance(p, dict) for p in params)
    ):
        raise ValueError("sampling_params must be a dict or one dict per batch input.")

    counts = [p.get("n", 1) for p in params]
    for n in counts:
        if (
            isinstance(n, bool)
            or not isinstance(n, int)
            or not 1 <= n <= max_parallel_samples
        ):
            raise ValueError(
                f"n must be an integer in [1, {max_parallel_samples}], got {n!r}. "
                "The server limit is --max-parallel-samples."
            )
    if any(n != counts[0] for n in counts):
        raise ValueError("n must be the same for all inputs in a batch.")
    outputs = shape.batch_size * counts[0]
    if outputs > max_batch_outputs:
        raise ValueError(
            f"batch_size * n is {outputs}, exceeding --max-batch-outputs="
            f"{max_batch_outputs}."
        )

    masks = obj.return_sampling_mask
    if masks is None:
        return
    if isinstance(masks, list):
        if shape.is_single or len(masks) != shape.batch_size:
            raise ValueError(
                "return_sampling_mask must contain one bool per batch input."
            )
        if counts[0] > 1:
            # Beam search returns n sequences without parallel-sample expansion.
            shape.parallel_sample_num = counts[0]
            try:
                parallel_samples = shape._handle_beam_search_parallel_sampling()
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid beam_width: {exc}") from exc
            if parallel_samples > 1:
                raise ValueError("Cannot use list return_sampling_mask with n > 1.")
    else:
        masks = [masks]
    if any(not isinstance(enabled, bool) for enabled in masks):
        raise ValueError("return_sampling_mask must be a bool or a list of bools.")
    if not any(masks):
        return
    for i in range(max(len(masks), len(params))):
        if not masks[i if len(masks) > 1 else 0]:
            continue
        kwargs = {
            **(preferred_sampling_params or {}),
            **params[i if len(params) > 1 else 0],
        }
        # Reuse normalization, including temperature=0 -> greedy top_k=1.
        try:
            sampling = sampling_params_class(**kwargs)
            sampling.verify(vocab_size)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"Invalid sampling_params for return_sampling_mask: {exc}"
            ) from exc
        error = sampling_mask_error(
            top_k=sampling.top_k,
            is_disaggregated=is_disaggregated,
            enabled=sampling_mask_enabled,
            capacity=sampling_mask_capacity,
            speculative=speculative,
            sampling_backend=sampling_backend,
        )
        if error:
            raise ValueError(error)
