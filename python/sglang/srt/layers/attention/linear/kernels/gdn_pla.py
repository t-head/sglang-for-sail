"""Adapt PLA FlashQLA prefill to SGLang's indexed recurrent-state pool."""

from numbers import Integral

import torch

from sglang.srt.layers.attention.linear.kernels.state_pool_io import (
    gather_pool_states,
    scatter_pool_states,
)


def pla_gdn_prefill_supported(q, k, v, g, beta, ssm_states):
    """Admit only FlashQLA's advertised geometry and precision domain."""
    if (
        any(t.ndim != 4 for t in (q, k, v))
        or q.shape[0] != 1
        or q.shape[1] == 0
        or q.shape[:2] != k.shape[:2]
        or q.shape[:2] != v.shape[:2]
        or q.shape[2] != k.shape[2]
        or any(t.shape[-1] != 128 for t in (q, k, v))
        or q.dtype not in (torch.float16, torch.bfloat16)
        or k.dtype != q.dtype
        or v.dtype != q.dtype
        or g is None
        or beta is None
        or g.dtype != torch.float32
        or beta.dtype != torch.float32
        or tuple(g.shape) != tuple(v.shape[:3])
        or tuple(beta.shape) != tuple(v.shape[:3])
        or ssm_states.ndim != 4
        or tuple(ssm_states.shape[1:]) != (v.shape[2], 128, 128)
        or ssm_states.dtype not in (torch.float32, torch.bfloat16)
        or any(t.device != q.device for t in (k, v, g, beta, ssm_states))
    ):
        return False

    # The optional installed PLA library owns its supported head combinations.
    # Unknown native failures remain visible once an input has been admitted.
    from pla.prefill.flashqla.ops import SUPPORTED_HEAD_CONFIGS

    return (v.shape[2], k.shape[2]) in SUPPORTED_HEAD_CONFIGS


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
    normalize_qk=True,
):
    # Keep the optional native extension out of the default Triton import path.
    # Requires indexed FlashQLA with raw_cu_seqlens/normalize_qk and
    # final_state_dtype support; g/beta are already activated by the caller.
    from pla.prefill.flashqla import chunk_gated_delta_rule_fwd

    # The indexed API uses [slot, H, V, K], unlike legacy non-indexed PLA
    # states [batch, H, K, V]. Preserve both that layout and the state dtype.
    # v1.0 pools reserve slot zero as the padding sink. Fixed-size device
    # gather/scatter gives every sequence an independent scratch state, including
    # negative/zero slots, without .any(), .item() or boolean compaction on GPU.
    pool_rows, live_rows, state = gather_pool_states(ssm_states, cache_indices)
    indices = torch.arange(
        cache_indices.numel(), dtype=torch.int32, device=cache_indices.device
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
        normalize_qk=normalize_qk,
        raw_cu_seqlens=raw_cu_seqlens,
        output_final_state=False,
        final_state_dtype=ssm_states.dtype,
    )
    scatter_pool_states(ssm_states, pool_rows, live_rows, state)
    # Indexed PLA writes final states in place. Its ht is NOT the per-chunk h
    # needed for prefix tracking; those batches are dispatched to Triton.
    return output, None, None
