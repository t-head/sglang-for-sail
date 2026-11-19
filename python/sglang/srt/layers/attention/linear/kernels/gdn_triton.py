from collections import OrderedDict

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)
from sglang.srt.utils import is_cpu, is_npu, is_sail, is_xpu

if not is_cpu():
    from sglang.kernels.ops.attention.fla.chunk import chunk_gated_delta_rule
    from sglang.kernels.ops.attention.fla.fused_recurrent import (
        fused_recurrent_gated_delta_rule_packed_decode,
    )
    from sglang.kernels.ops.attention.linear.fused_recurrent_linear_replayssm import (
        fused_recurrent_gdn_replayssm_decode,
    )
    from sglang.kernels.ops.attention.linear.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )

if is_npu():
    from sgl_kernel_npu.fla.chunk import chunk_gated_delta_rule_npu
    from sgl_kernel_npu.fla.fused_sigmoid_gating_recurrent_decode_optimized import (
        fused_sigmoid_gating_delta_rule_update_decode_npu as fused_sigmoid_gating_delta_rule_update,
    )

    chunk_gated_delta_rule = chunk_gated_delta_rule_npu
elif is_cpu():
    from sgl_kernel.mamba import chunk_gated_delta_rule_cpu

    chunk_gated_delta_rule = chunk_gated_delta_rule_cpu
    fused_sigmoid_gating_delta_rule_update = (
        torch.ops.sgl_kernel.fused_sigmoid_gating_delta_rule_update_cpu
    )
elif is_xpu():
    from sglang.srt.hardware_backend.xpu.kernels.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )


class TritonGDNKernel(LinearAttnKernelBase):
    """Triton-based kernel for GDN (Gated Delta Network) linear attention."""

    supports_packed_decode: bool = not is_cpu() and not is_npu()
    supports_strided_target_verify_qkv: bool = True

    def __init__(self):
        self._triton_capture_signatures = OrderedDict()

    def _prepare_triton_capture(self, q, k, v, g, beta, ssm_states, query_start_loc):
        from sglang.kernels.ops.attention.fla.chunk import CHUNK_SIZE
        from sglang.kernels.ops.attention.fla.index import (
            prepare_chunk_indices,
            prepare_chunk_offsets,
        )

        # Populate the exact metadata cache keys used by direct capture. PLA's
        # own host plan does not populate these Triton caches.
        prepare_chunk_indices(query_start_loc, CHUNK_SIZE)
        prepare_chunk_offsets(query_start_loc, CHUNK_SIZE)
        output_chunk_size = min(CHUNK_SIZE, max(16, 1 << (q.shape[1] - 1).bit_length()))
        prepare_chunk_indices(query_start_loc, output_chunk_size)
        signature = (
            tuple(
                (t.device, t.dtype, tuple(t.shape), tuple(t.stride()))
                for t in (q, k, v, g, beta)
            ),
            ssm_states.dtype,
            query_start_loc.numel(),
        )
        if signature not in self._triton_capture_signatures:
            # Autotuning must also finish before capture. Use a scratch pool:
            # warmup must not apply the recurrence twice to the caller's state.
            count = query_start_loc.numel() - 1
            scratch = ssm_states.new_zeros((count, *ssm_states.shape[1:]))
            slots = torch.arange(count, dtype=query_start_loc.dtype, device=q.device)
            chunk_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=scratch,
                initial_state_indices=slots,
                cu_seqlens=query_start_loc,
                head_first=False,
                use_qk_l2norm_in_kernel=True,
                inplace_update=True,
            )
            self._triton_capture_signatures[signature] = None
            if len(self._triton_capture_signatures) > 64:
                self._triton_capture_signatures.popitem(last=False)
        else:
            self._triton_capture_signatures.move_to_end(signature)

    def packed_decode(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        num_v_heads: int,
        head_v_dim: int,
        **kwargs,
    ) -> torch.Tensor:
        """Packed decode fast path: fuse QKV extraction + gating + recurrent
        update into a single Triton kernel, eliminating intermediate tensors
        and extra kernel launches.

        Args:
            mixed_qkv: [B, qkv_dim] packed projection output after conv1d.
            a, b: [B, HV] gating inputs.
            A_log: [HV] log-space decay parameter.
            dt_bias: [HV] time-step bias.
            scale: attention scale factor (typically head_k_dim ** -0.5).
            ssm_states: [num_slots, HV, V, K] full state pool.
            cache_indices: [B] per-request state slot indices.
            num_v_heads: number of value heads (after TP sharding).
            head_v_dim: dimension per value head.

        Returns:
            output tensor of shape [1, B, HV, V] matching the existing
            decode kernel output layout.
        """
        B = mixed_qkv.shape[0]
        # Packed kernel expects output shape [B, 1, HV, V]
        out = mixed_qkv.new_empty(B, 1, num_v_heads, head_v_dim)

        # GDN ReplaySSM buffered decode (slice 1a). Drop-in for the packed
        # decode: same args plus the three per-layer ring caches and the
        # per-row write cursor. When any ring tensor / cursor is None (flag
        # off) we fall through to the byte-identical legacy path below.
        replayssm_d = kwargs.get("replayssm_d")
        replayssm_k = kwargs.get("replayssm_k")
        replayssm_g = kwargs.get("replayssm_g")
        replayssm_write_pos = kwargs.get("replayssm_write_pos")
        # GDN ReplaySSM (slice 2b): optional per-row force-flush (radix track
        # boundary). None when radix tracking is off / flag off; the kernel
        # treats None as "no forced flush" (byte-identical to slice 1a/1b).
        replayssm_force_flush = kwargs.get("replayssm_force_flush")
        if (
            replayssm_d is not None
            and replayssm_k is not None
            and replayssm_g is not None
            and replayssm_write_pos is not None
        ):
            fused_recurrent_gdn_replayssm_decode(
                mixed_qkv=mixed_qkv,
                a=a,
                b=b,
                A_log=A_log,
                dt_bias=dt_bias,
                scale=scale,
                initial_state=ssm_states,
                d_cache=replayssm_d,
                k_cache=replayssm_k,
                g_cache=replayssm_g,
                out=out,
                ssm_state_indices=cache_indices,
                write_pos=replayssm_write_pos,
                force_flush=replayssm_force_flush,
                use_qk_l2norm_in_kernel=True,
            )
            return out.transpose(0, 1)

        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            initial_state=ssm_states,
            out=out,
            ssm_state_indices=cache_indices,
            use_qk_l2norm_in_kernel=True,
        )

        # Convert [B, 1, HV, V] → [1, B, HV, V] to match existing output
        # layout. transpose() returns a view — zero cost.
        return out.transpose(0, 1)

    def decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            initial_state_source=ssm_states,
            initial_state_indices=cache_indices,
            cu_seqlens=query_start_loc,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
        )

    def extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        inplace_update: bool = True,
        **kwargs,
    ) -> tuple:
        if (
            envs.SGLANG_SAIL_PLA_CUDA.get()
            and is_sail()
            and q.is_cuda
            and inplace_update
            and not kwargs.get("requires_intermediate_states", False)
            and kwargs.get("track_state") is None
            and not kwargs.get("num_state_checkpoints", 0)
            # PLA plans chunks on the host. Keep direct Graph capture on the
            # existing Triton path; breakable graphs use their eager GDN break.
            and not torch.cuda.is_current_stream_capturing()
        ):
            from sglang.srt.layers.attention.linear.kernels.gdn_pla import (
                pla_gdn_prefill,
                pla_host_offsets,
            )

            offsets = pla_host_offsets(
                kwargs.get("seq_lens_cpu"), query_start_loc, q.shape[1]
            )
            if offsets is not None:
                self._prepare_triton_capture(
                    q, k, v, g, beta, ssm_states, query_start_loc
                )
                return pla_gdn_prefill(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    ssm_states=ssm_states,
                    cache_indices=cache_indices,
                    query_start_loc=query_start_loc,
                    raw_cu_seqlens=offsets,
                )

        recurrent_state = ssm_states
        recurrent_state_indices_args = {"initial_state_indices": cache_indices}
        inplace_update_args = {"inplace_update": inplace_update}
        if is_cpu():
            if not inplace_update:
                raise NotImplementedError(
                    "GDN multi-item scoring is not supported by the CPU chunk kernel"
                )
            inplace_update_args = {}
        elif is_npu():
            if not inplace_update:
                raise NotImplementedError(
                    "GDN multi-item scoring is not supported by the NPU chunk kernel"
                )
            recurrent_state = ssm_states[cache_indices]
            recurrent_state_indices_args = {}
            # The external NPU kernel does not expose the optional write-back
            # control. Its existing behavior is equivalent to True.
            inplace_update_args = {}

        return chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=recurrent_state,
            cu_seqlens=query_start_loc,
            head_first=False,
            use_qk_l2norm_in_kernel=True,
            **recurrent_state_indices_args,
            **inplace_update_args,
        )

    def target_verify(
        self,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        intermediate_states_buffer: torch.Tensor,
        intermediate_state_indices: torch.Tensor,
        cache_steps: int,
        retrieve_parent_token: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            initial_state_source=ssm_states,
            initial_state_indices=cache_indices,
            cu_seqlens=query_start_loc,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            is_kda=False,
            # target_verify specific parameters
            disable_state_update=True,
            intermediate_states_buffer=intermediate_states_buffer,
            intermediate_state_indices=intermediate_state_indices,
            cache_steps=cache_steps,
            retrieve_parent_token=retrieve_parent_token,
        )
