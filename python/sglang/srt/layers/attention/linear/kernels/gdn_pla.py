"""Adapt PLA FlashQLA prefill to SGLang's indexed recurrent-state pool."""

from numbers import Integral

import torch


def gather_pool_states(states, indices):
    # v0.5.23 permits slot zero; only negative indices are padding.
    live = (indices >= 0).view(-1, *((1,) * (states.ndim - 1)))
    state = torch.where(live, states[indices.clamp_min(0)], 0).contiguous()
    return indices, live, state


def scatter_pool_states(states, indices, live, updates):
    if states.device.type == "cpu":
        # CPU contract/reference path; no device synchronization is involved.
        valid = live.reshape(-1)
        states[indices[valid]] = updates[valid].to(states.dtype)
    else:
        from sglang.kernels.ops.mamba.state_pool_rows import scatter_state_pool_rows_

        scatter_state_pool_rows_(states, updates, indices)


def pla_gdn_prefill_supported(
    packed_qkv, a, b, A_log, dt_bias, ssm_states, *, num_qk_heads, num_v_heads
):
    """Admit the packed Q|K|V and raw-gating ABI without materializing inputs."""
    if (
        packed_qkv.ndim != 2
        or packed_qkv.shape[0] == 0
        or num_qk_heads <= 0
        or num_v_heads <= 0
        or packed_qkv.shape[1] != (2 * num_qk_heads + num_v_heads) * 128
        or packed_qkv.dtype not in (torch.float16, torch.bfloat16)
        or packed_qkv.stride(1) != 1
        or packed_qkv.stride(0) < packed_qkv.shape[1]
        or a is None
        or b is None
        or A_log is None
        or dt_bias is None
        or any(
            t.ndim != 2
            or tuple(t.shape) != (packed_qkv.shape[0], num_v_heads)
            or t.dtype != packed_qkv.dtype
            or t.stride(1) != 1
            or t.stride(0) < num_v_heads
            for t in (a, b)
        )
        or any(
            t.ndim != 1
            or t.numel() != num_v_heads
            or t.dtype not in (torch.float32, torch.float16, torch.bfloat16)
            or not t.is_contiguous()
            for t in (A_log, dt_bias)
        )
        or ssm_states.ndim != 4
        or tuple(ssm_states.shape[1:]) != (num_v_heads, 128, 128)
        or ssm_states.dtype not in (torch.float32, torch.bfloat16)
        or any(
            t.device != packed_qkv.device for t in (a, b, A_log, dt_bias, ssm_states)
        )
    ):
        return False

    from pla.prefill.flashqla.ops import SUPPORTED_HEAD_CONFIGS

    return (num_v_heads, num_qk_heads) in SUPPORTED_HEAD_CONFIGS


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
    packed_qkv,
    a,
    b,
    *,
    A_log,
    dt_bias,
    num_qk_heads,
    num_v_heads,
    ssm_states,
    cache_indices,
    query_start_loc,
    raw_cu_seqlens,
):
    # This entry consumes post-convolution Q|K|V and UNACTIVATED a/b. PLA
    # applies -exp(A_log)*softplus(a+dt_bias), sigmoid(b), and Q/K L2 norm.
    # Do not feed activated g/beta here or normalize Q/K in the caller.
    from pla.prefill.flashqla import gdn_prefill_packed_qkv_fwd

    # Indexed PLA uses [slot,H,V,K] and requires contiguous scratch states.
    # Preserve v0.5.23's valid slot zero and independent negative-slot rows;
    # the original pool may be strided and is written back on the device.
    pool_rows, live_rows, state = gather_pool_states(ssm_states, cache_indices)
    # Advanced indexing/where can preserve a transposed pool's inner strides.
    # The native indexed-state ABI requires row-major [slot,H,V,K] scratch.
    state = state.contiguous()
    indices = torch.arange(
        cache_indices.numel(), dtype=torch.int32, device=cache_indices.device
    )
    _, _, output, _, _ = gdn_prefill_packed_qkv_fwd(
        packed_qkv=packed_qkv,
        raw_gate_a=a,
        raw_beta_b=b,
        # The original gating kernel converts these parameters to FP32 before
        # arithmetic. Widening FP16/BF16 here is exact; stored weights are not
        # changed. The packed ABI requires both parameter vectors in FP32.
        A_log=A_log.float(),
        dt_bias=dt_bias.float(),
        num_qk_heads=num_qk_heads,
        num_v_heads=num_v_heads,
        initial_state=state,
        initial_state_indices=indices,
        cu_seqlens=query_start_loc.to(dtype=torch.int32).contiguous(),
        raw_cu_seqlens=raw_cu_seqlens,
        output_final_state=False,
        final_state_dtype=ssm_states.dtype,
    )
    scatter_pool_states(ssm_states, pool_rows, live_rows, state)
    return output, None, None
