"""Adapt PLA FlashQLA prefill to SGLang's indexed recurrent-state pool."""

from numbers import Integral

import torch


def pla_host_offsets(seq_lens_cpu, query_start_loc, num_tokens):
    """Reject stale/padded batch metadata without reading device offsets."""
    if isinstance(seq_lens_cpu, torch.Tensor) and seq_lens_cpu.device.type == "cpu":
        seq_lens_cpu = seq_lens_cpu.tolist()
    if not isinstance(seq_lens_cpu, (list, tuple)) or not seq_lens_cpu:
        return None
    if len(seq_lens_cpu) + 1 != query_start_loc.numel():
        return None
    offsets = [0]
    for length in seq_lens_cpu:
        if not isinstance(length, Integral) or isinstance(length, bool) or length < 0:
            return None
        offsets.append(offsets[-1] + int(length))
    return tuple(offsets) if offsets[-1] == num_tokens else None


def pla_gdn_prefill(
    q,
    k,
    v,
    g,
    beta,
    *,
    ssm_states,
    cache_indices,
    query_start_loc,
    raw_cu_seqlens,
):
    # Keep the optional native extension out of the default Triton import path.
    from pla.prefill.flashqla import chunk_gated_delta_rule_fwd

    # The indexed API uses [slot, H, V, K], unlike legacy non-indexed PLA
    # states [batch, H, K, V]. Preserve both that layout and the state dtype.
    needs_gather = not ssm_states.is_contiguous()
    state = ssm_states[cache_indices].contiguous() if needs_gather else ssm_states
    indices = cache_indices.to(dtype=torch.int32).contiguous()
    if needs_gather:
        indices = torch.arange(
            indices.numel(), dtype=torch.int32, device=indices.device
        )

    _, _, output, _, _ = chunk_gated_delta_rule_fwd(
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        g=g.contiguous(),
        beta=beta.contiguous(),
        initial_state=state,
        initial_state_indices=indices,
        cu_seqlens=query_start_loc.to(dtype=torch.int32).contiguous(),
        normalize_qk=True,
        raw_cu_seqlens=raw_cu_seqlens,
        output_final_state=False,
        final_state_dtype=ssm_states.dtype,
    )
    if needs_gather:
        ssm_states[cache_indices] = state
    # Indexed PLA writes final states in place. Its ht is NOT the per-chunk h
    # needed for prefix tracking; those batches are dispatched to Triton.
    return output, None, None
