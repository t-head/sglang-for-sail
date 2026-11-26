from __future__ import annotations

from numbers import Integral
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)
from sglang.srt.runtime_context import get_exec
from sglang.srt.utils import is_sail, logger

if TYPE_CHECKING:
    from sglang.srt.layers.attention.mamba.mamba2_metadata import ForwardMetadata

# FlashKDA chunk size. Sequences shorter than this fall back to Triton.
_FLASHKDA_CHUNK_SIZE = 64

# FlashKDA's max sequence length, Batches whose longest sequence exceeds this
# fall back to Triton for the whole batch.
_FLASHKDA_MAX_SEQ_LEN = 2048


def ppu_pla_prefill_gate_open() -> bool:
    """Whether KDA prefill/extend may run on the PPU SAIL PLA kernel.

    Same gate as the decode fast-path: SGLANG_SAIL_PLA_CUDA (SAIL PLA master
    gate; SGLANG_SAIL_FLA_CUDA is its deprecated alias), which defaults to True
    on PPU. Off-PPU this is always False -- other platforms use the external
    ``flash_kda`` package instead of PLA.
    """
    if not is_sail():
        return False
    return envs.SGLANG_SAIL_PLA_CUDA.get()


def _pla_host_offsets(
    seq_lens_cpu: Optional[list],
    query_start_loc: torch.Tensor,
    num_tokens: int,
) -> Optional[tuple[int, ...]]:
    """Build trusted offsets only from the current packed batch's CPU lengths."""
    if not isinstance(seq_lens_cpu, (list, tuple)) or not seq_lens_cpu:
        return None
    if len(seq_lens_cpu) + 1 != query_start_loc.numel():
        return None

    offsets = [0]
    for length in seq_lens_cpu:
        # Do not coerce tensor scalars (possibly on device) or fractional values.
        if not isinstance(length, Integral) or isinstance(length, bool) or length < 0:
            return None
        offsets.append(offsets[-1] + int(length))
    # KDAAttnBackend trims physical padding before calling this adapter. If CPU
    # metadata describes another slice/batch, retain the device-only path.
    if offsets[-1] != num_tokens:
        return None
    return tuple(offsets)


def _load_flash_kda(*, ppu: bool = False):
    """Import the optional ``flash_kda`` module."""
    try:
        import flash_kda
    except ImportError as e:
        if ppu:
            raise ImportError(
                "Explicit FlashKDA on PPU SM89 requires the PPU FlashKDA wheel "
                "built from 0dfef21."
            ) from e
        raise ImportError(
            "The 'flashkda' KDA prefill backend requires the flash_kda module, "
            "which is not installed. Install it from source:\n"
            "    pip install git+https://github.com/MoonshotAI/FlashKDA.git"
        ) from e
    return flash_kda


def validate_ppu_flashkda_startup(
    *,
    decode: str,
    explicit_prefill: bool,
    verify: str,
    replayssm_mode: str,
    state_accum_dtype: str,
) -> None:
    """Fail during argument resolution if the selected PPU ABI is unavailable."""
    needs_verify = verify == "flashkda" and replayssm_mode != "native"
    if not (
        decode == "flashkda"
        or explicit_prefill
        or needs_verify
        or replayssm_mode == "flashkda"
    ):
        return

    flash_kda = _load_flash_kda(ppu=True)
    if decode == "flashkda" and not getattr(
        flash_kda, "DECODE_SUPPORTS_LOWER_BOUND", False
    ):
        raise ValueError(
            "PPU FlashKDA decode requires the bounded decode ABI from 0dfef21."
        )
    if explicit_prefill:
        capability = (
            "SUPPORTS_FP32_STATE_ACCUMULATION"
            if state_accum_dtype == "float32"
            else "SUPPORTS_BF16_STATE_ACCUMULATION"
        )
        if not getattr(flash_kda, capability, False):
            raise ValueError(
                "Explicit PPU FlashKDA prefill requires "
                f"{state_accum_dtype} state accumulation support from 0dfef21."
            )
    if needs_verify and not callable(getattr(flash_kda, "verify_strided", None)):
        raise ValueError("PPU FlashKDA verify requires the verify_strided ABI.")
    if replayssm_mode == "flashkda" and not callable(
        getattr(flash_kda, "verify_replayssm", None)
    ):
        raise ValueError(
            "FlashKDA ReplaySSM requires the verify_replayssm ABI from 0dfef21."
        )


def _triton_fallback(
    q,
    k,
    v,
    g,
    beta,
    ssm_states,
    cache_indices,
    query_start_loc,
    A_log=None,
    dt_bias=None,
    lower_bound=None,
    beta_is_raw=False,
    return_intermediate_states=False,
    track_state=None,
    track_chunk_idx=None,
):
    """Fall back to the Triton chunk_kda kernel (handles all preprocessing).

    `g` is the RAW gate; chunk_kda applies the gate activation internally when
    A_log is provided, so A_log/dt_bias/lower_bound must be threaded through too
    -- otherwise the fallback silently skips activation. chunk_kda updates the
    ssm state in-place via cache_indices and returns only the output tensor
    (or (output, h) when return_intermediate_states is set).
    """
    from sglang.kernels.ops.attention.fla.kda import chunk_kda

    return chunk_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=ssm_states,
        initial_state_indices=cache_indices,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=query_start_loc,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        beta_is_raw=beta_is_raw,
        output_intermediate_states=return_intermediate_states,
        track_state=track_state,
        track_chunk_idx=track_chunk_idx,
    )


class FlashKDAKernel(LinearAttnKernelBase):
    """FlashKDA (MoonshotAI) fully-fused CUTLASS KDA prefill backend.

    Wraps the external ``flash_kda`` package (https://github.com/MoonshotAI/FlashKDA).

    FlashKDA fuses q/k L2 norm, beta sigmoid, and the KDA gate *inside* the
    kernel, so we pass RAW tensors plus ``A_log``/``dt_bias``/``lower_bound``.
    It is prefill-only, bf16, K == V == 128, HV == H (no GVA), and requires the
    safe (bounded) gate (``lower_bound`` set). The non-safe path and sequences
    outside [chunk_size, max_seq_len] fall back to Triton ``chunk_kda``.
    Requires an SM90+ GPU with the ``flash_kda`` package.
    """

    # Tracked batches always take the Triton fallback, which forwards the
    # fp32 snapshot arguments (see _triton_fallback).
    supports_track_state_snapshot: bool = True

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
        raise NotImplementedError("FlashKDAKernel only supports prefill (extend)")

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
        A_log: Optional[torch.Tensor] = None,
        dt_bias: Optional[torch.Tensor] = None,
        lower_bound: Optional[float] = None,
        extend_seq_lens_cpu: Optional[list] = None,
        is_spec_decode: bool = False,
        beta_is_raw: bool = False,
        return_intermediate_states: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        # The fused kernel cannot expose per-chunk states (h), which the mamba
        # radix extra_buffer track path needs; route tracked batches through
        # the Triton chunk_kda fallback instead of silently skipping the
        # snapshot (that would corrupt prefix-cache restores).
        reason = self._fallback_reason(
            lower_bound,
            is_spec_decode,
            query_start_loc,
            extend_seq_lens_cpu,
            return_intermediate_states,
        )
        if reason is not None:
            logger.info_once(
                "KDA extend uses the community Triton chunk kernel: " + reason + "."
            )
            return _triton_fallback(
                q,
                k,
                v,
                g,
                beta,
                ssm_states,
                cache_indices,
                query_start_loc,
                A_log=A_log,
                dt_bias=dt_bias,
                lower_bound=lower_bound,
                beta_is_raw=beta_is_raw,
                return_intermediate_states=return_intermediate_states,
                track_state=kwargs.get("track_state"),
                track_chunk_idx=kwargs.get("track_chunk_idx"),
            )

        # Only reachable when return_intermediate_states is False (the True case
        # is served by the Triton fallback above), so honor the (out) vs (out, h)
        # return contract and hand back just the output tensor -- kda_nvidia and
        # kda_ptx do the same. Wrapping into (out, None) here made non-track
        # batches (track_ssm=False, which never unpack) receive a tuple and fail
        # on `.shape` downstream (o_norm / fused_norm_gate).
        return self._flashkda_extend(
            q,
            k,
            v,
            g,
            beta,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
            beta_is_raw=beta_is_raw,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            req_pool_indices_cpu=kwargs.get("req_pool_indices_cpu"),
            pla_prefill_metadata=kwargs.get("pla_prefill_metadata"),
        )

    def _fallback_reason(
        self,
        lower_bound: Optional[float],
        is_spec_decode: bool,
        query_start_loc: torch.Tensor,
        extend_seq_lens_cpu: Optional[list],
        return_intermediate_states: bool,
    ) -> Optional[str]:
        """Why this batch must stay on the Triton chunk kernel (None = fast path)."""
        if return_intermediate_states:
            return (
                "the mamba track path needs per-chunk fp32 states "
                "(why=intermediate-states), which the fused kernel does not emit"
            )
        if is_sail() and not ppu_pla_prefill_gate_open():
            return "the PPU SAIL PLA prefill gate is closed (SGLANG_SAIL_PLA_CUDA)"
        if not self._should_fall_back(
            lower_bound, is_spec_decode, query_start_loc, extend_seq_lens_cpu
        ):
            return None
        if lower_bound is None:
            return "the fused kernel requires the safe (lower-bounded) KDA gate"
        if is_spec_decode:
            return "draft-extend must stay rollback-able"
        return "the batch's sequence lengths are outside the fused kernel window"

    @staticmethod
    def _should_fall_back(
        lower_bound: Optional[float],
        is_spec_decode: bool,
        query_start_loc: torch.Tensor,
        extend_seq_lens_cpu: Optional[list],
    ) -> bool:
        """Whether to use the Triton chunk_kda path instead of the fused kernel."""
        # Safe-gate only: the fused kernel does not support the unbounded gate
        # (-exp(A_log)*softplus); those models leave lower_bound unset.
        if lower_bound is None:
            return True
        # FlashKDA writes the committed recurrent state back in place, so it is
        # unsafe for speculative verify / draft-extend forwards (which must stay
        # rollback-able). Those reach this backend through forward_extend, so
        # gate them here rather than relying on the decode/target_verify stubs.
        if is_spec_decode:
            return True
        if is_sail():
            return False
        # Short sequences (< chunk size) and long sequences (> the crossover
        # where Triton's chunked prefill wins) are faster on Triton. Read the
        # per-request lengths from the CPU-side extend_seq_lens to avoid a
        # GPU->CPU sync on every layer; derive from query_start_loc (one sync)
        # only if they are unavailable.
        if extend_seq_lens_cpu is not None:
            if torch.is_tensor(extend_seq_lens_cpu):
                lo = int(extend_seq_lens_cpu.min())
                hi = int(extend_seq_lens_cpu.max())
            else:
                lo = min(extend_seq_lens_cpu)
                hi = max(extend_seq_lens_cpu)
        else:
            seq_lens = query_start_loc[1:] - query_start_loc[:-1]
            lo_t, hi_t = torch.aminmax(seq_lens)
            lo, hi = int(lo_t), int(hi_t)
        return lo < _FLASHKDA_CHUNK_SIZE or hi > _FLASHKDA_MAX_SEQ_LEN

    def _flashkda_extend(
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
        A_log: Optional[torch.Tensor] = None,
        dt_bias: Optional[torch.Tensor] = None,
        lower_bound: Optional[float] = None,
        beta_is_raw: bool = False,
        extend_seq_lens_cpu: Optional[list] = None,
        req_pool_indices_cpu: Optional[torch.Tensor] = None,
        pla_prefill_metadata: Optional[ForwardMetadata] = None,
    ) -> torch.Tensor:
        # Input shapes (varlen, B == 1, matching chunk_kda's contract):
        #   q, k = [1, packed_seq, H, K]   v = [1, packed_seq, HV, V]
        #   g    = [1, packed_seq, HV, K]  beta = [1, packed_seq, H]
        # flash_kda wants these 4D tensors directly and RAW (it fuses l2norm /
        # beta sigmoid / gate activation in-kernel).
        num_heads = q.shape[2]
        head_dim = q.shape[3]
        scale = head_dim**-0.5

        # FlashKDA applies sigmoid internally; invert only the already-activated
        # Kimi beta path.
        if not beta_is_raw:
            beta = torch.logit(beta.float().clamp_(1e-7, 1.0 - 1e-7))
        beta = beta.to(torch.bfloat16)

        # flash_kda wants A_log [H] fp32 and dt_bias [H, K] fp32. The model
        # stores A_log as [1, 1, H, 1] and dt_bias as 1D [H*K], so reshape both.
        A_log = A_log.reshape(-1).float()
        if dt_bias is not None:
            dt_bias = dt_bias.reshape(num_heads, -1).float()

        if is_sail():
            return self._sail_flashkda_extend(
                q,
                k,
                v,
                g,
                beta,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                scale=scale,
                A_log=A_log,
                dt_bias=dt_bias,
                lower_bound=lower_bound,
                extend_seq_lens_cpu=extend_seq_lens_cpu,
                req_pool_indices_cpu=req_pool_indices_cpu,
                pla_prefill_metadata=pla_prefill_metadata,
            )

        # cu_seqlens must be int64 for flash_kda (FLA casts to long).
        cu_seqlens = query_start_loc.to(torch.int64)

        # flash_kda varlen state is [N, H, V, K] -- the SAME layout as sglang's
        # KDA pool, so no transpose is needed. Advanced indexing copies, so the
        # final state is written back in-place below (matching chunk_kda).
        initial_state = ssm_states[cache_indices]

        out_buf = torch.empty_like(v)
        final_state = torch.empty_like(initial_state)

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        g = g.contiguous()

        beta = beta.contiguous()
        A_log = A_log.contiguous()

        if dt_bias is not None:
            dt_bias = dt_bias.contiguous()

        initial_state = initial_state.contiguous()

        flash_kda = _load_flash_kda()
        flash_kda.fwd(
            q,
            k,
            v,
            g,
            beta,
            scale,
            out_buf,
            A_log,
            dt_bias,
            lower_bound,
            initial_state=initial_state,
            final_state=final_state,
            cu_seqlens=cu_seqlens,
        )

        ssm_states[cache_indices] = final_state

        # out_buf is already [1, packed_seq, HV, V].
        return out_buf

    @staticmethod
    def _sail_flashkda_extend(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        scale: float,
        A_log: torch.Tensor,
        dt_bias: Optional[torch.Tensor],
        lower_bound: Optional[float],
        extend_seq_lens_cpu: Optional[list],
        req_pool_indices_cpu: Optional[torch.Tensor],
        pla_prefill_metadata: Optional[ForwardMetadata],
    ) -> torch.Tensor:
        from pla.prefill.flashkdapro import flashkda_fwd

        logger.info_once(
            "USE PPU SAIL CUDA PLA kernel: flashkda_fwd (KDA prefill/extend)"
        )

        batch_size, seq_len, num_heads, head_dim = q.shape
        num_tokens = batch_size * seq_len
        beta = beta.reshape(num_tokens, num_heads)

        raw_offsets = None
        indexed_offsets = None
        # Capture must use device offsets and compact state on every replay.
        if not (query_start_loc.is_cuda and torch.cuda.is_current_stream_capturing()):
            if pla_prefill_metadata is not None:
                indexed_offsets = pla_prefill_metadata.pla_prefill_host_offsets
            if (
                indexed_offsets is not None
                and len(indexed_offsets) == query_start_loc.numel()
                and indexed_offsets[-1] == num_tokens
            ):
                raw_offsets = indexed_offsets
            else:
                indexed_offsets = None
                raw_offsets = _pla_host_offsets(
                    extend_seq_lens_cpu, query_start_loc, num_tokens
                )
                # Synthetic warmup rows have no host request indices. Padding
                # or repeated request rows do not own distinct state slots.
                if (
                    pla_prefill_metadata is not None
                    and req_pool_indices_cpu is not None
                    and raw_offsets is not None
                    and all(length > 0 for length in extend_seq_lens_cpu)
                ):
                    host_slots = req_pool_indices_cpu.tolist()
                    if (
                        len(set(host_slots)) == len(host_slots) == len(raw_offsets) - 1
                        and min(host_slots) > 0
                    ):
                        pla_prefill_metadata.pla_prefill_host_offsets = raw_offsets
                        indexed_offsets = raw_offsets

        use_indexed_state = (
            indexed_offsets is not None
            and ssm_states.is_contiguous()
            and cache_indices.dtype == torch.int32
            and cache_indices.is_contiguous()
            and cache_indices.numel() + 1 == len(indexed_offsets)
        )
        if use_indexed_state:
            state_kwargs = dict(state_pool=ssm_states, state_indices=cache_indices)
        else:
            initial_state = ssm_states[cache_indices].contiguous()
            final_state = torch.empty_like(initial_state)
            state_kwargs = dict(initial_state=initial_state, final_state=final_state)

        out_buf = torch.empty_like(v, memory_format=torch.contiguous_format)
        flashkda_fwd(
            q=q.view(batch_size * seq_len, num_heads, head_dim),
            k=k.view(batch_size * seq_len, num_heads, head_dim),
            v=v.view(batch_size * seq_len, num_heads, head_dim),
            g=g.view(batch_size * seq_len, num_heads, head_dim),
            beta=beta,
            A_log=A_log,
            dt_bias=dt_bias,
            out=out_buf.view(batch_size * seq_len, num_heads, head_dim),
            scale=scale,
            lower_bound=lower_bound,
            cu_seqlens=query_start_loc,
            raw_cu_seqlens=raw_offsets,
            **state_kwargs,
        )

        if not use_indexed_state:
            ssm_states[cache_indices] = final_state

        # out_buf is already [1, packed_seq, HV, V].
        return out_buf


class PPUFlashKDAKernel(FlashKDAKernel):
    """Explicit PPU SM89 backend using the external FlashKDA package, not PLA."""

    supports_packed_decode = True
    supports_packed_decode_safe_gate = True
    supports_safe_gate_verify = True

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
        lower_bound: Optional[float] = None,
        **kwargs,
    ) -> torch.Tensor:
        batch_size = mixed_qkv.shape[0]
        out = mixed_qkv.new_empty((batch_size, 1, num_v_heads, head_v_dim))
        if batch_size == 0:
            return out.transpose(0, 1)

        flash_kda = _load_flash_kda(ppu=True)
        if not getattr(flash_kda, "DECODE_SUPPORTS_LOWER_BOUND", False):
            raise RuntimeError(
                "PPU FlashKDA decode requires the bounded decode ABI (0dfef21)."
            )
        flash_kda.decode(
            mixed_qkv.contiguous(),
            a.reshape(batch_size, num_v_heads * head_v_dim).contiguous(),
            b.reshape(batch_size, num_v_heads).contiguous(),
            A_log.reshape(-1).float().contiguous(),
            dt_bias.reshape(-1).float().contiguous(),
            scale,
            ssm_states,
            cache_indices.to(torch.int32).contiguous(),
            out,
            lower_bound=lower_bound,
        )
        return out.transpose(0, 1)

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
        A_log: Optional[torch.Tensor] = None,
        dt_bias: Optional[torch.Tensor] = None,
        lower_bound: Optional[float] = None,
        is_spec_decode: bool = False,
        beta_is_raw: bool = False,
        return_intermediate_states: bool = False,
        track_state: Optional[torch.Tensor] = None,
        track_chunk_idx: Optional[torch.Tensor] = None,
        state_chunk_size: int = _FLASHKDA_CHUNK_SIZE,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, Optional[torch.Tensor]]:
        if (track_state is None) != (track_chunk_idx is None):
            raise ValueError("FlashKDA track_state and track_chunk_idx must be paired")
        if track_state is not None and state_chunk_size != _FLASHKDA_CHUNK_SIZE:
            raise ValueError("FlashKDA radix checkpoints require 64-token chunks")

        # Aligned rows only read final_state; unaligned rows consume the sparse
        # FP32 scratch instead of a full per-chunk intermediate state tensor.
        use_native = (
            lower_bound is not None
            and not is_spec_decode
            and (not return_intermediate_states or track_state is not None)
        )
        if use_native:
            flash_kda = _load_flash_kda(ppu=True)
            accum = get_exec().mamba.mamba_ssm_accum_dtype
            use_fp32 = accum == "float32"
            capability = "FP32" if use_fp32 else "BF16"
            if not getattr(
                flash_kda, f"SUPPORTS_{capability}_STATE_ACCUMULATION", False
            ):
                raise RuntimeError(
                    f"PPU FlashKDA prefill requires {capability} state accumulation "
                    "support (0dfef21)."
                )
            if track_state is not None:
                use_native = (
                    getattr(flash_kda, "SUPPORTS_FP32_TRACK_STATE", False)
                    if use_fp32
                    else flash_kda.supports_bf16_track_state(
                        q.shape[2], ssm_states.dtype
                    )
                )
        if not use_native:
            return _triton_fallback(
                q,
                k,
                v,
                g,
                beta,
                ssm_states,
                cache_indices,
                query_start_loc,
                A_log=A_log,
                dt_bias=dt_bias,
                lower_bound=lower_bound,
                beta_is_raw=beta_is_raw,
                return_intermediate_states=return_intermediate_states,
                track_state=track_state,
                track_chunk_idx=track_chunk_idx,
            )
        if not beta_is_raw:
            beta = torch.logit(beta.float().clamp_(1e-7, 1.0 - 1e-7))
        heads, head_dim = q.shape[-2:]
        # Keep the fused-QKV views: 0dfef21 accepts packed token-row strides.
        q, k, v = (
            (
                x
                if x.stride()[-2:] == (128, 1)
                and x.stride(1) >= heads * 128
                and x.stride(1) % 64 == 0
                else x.contiguous()
            )
            for x in (q, k, v)
        )
        out = v.new_empty(v.shape)
        initial_state = ssm_states[cache_indices].contiguous()
        final_state = torch.empty_like(initial_state)
        flash_kda.fwd(
            q,
            k,
            v,
            g.contiguous(),
            beta.to(torch.bfloat16).contiguous(),
            head_dim**-0.5,
            out,
            A_log.reshape(-1).float().contiguous(),
            dt_bias.reshape(heads, -1).float().contiguous(),
            lower_bound,
            initial_state=initial_state,
            final_state=final_state,
            cu_seqlens=query_start_loc.to(torch.int64),
            state_accum_dtype=getattr(torch, accum),
            track_state=track_state,
            track_chunk_idx=track_chunk_idx,
        )
        ssm_states[cache_indices] = final_state
        return (out, None) if return_intermediate_states else out

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
        intermediate_states_buffer: Optional[torch.Tensor],
        intermediate_state_indices: torch.Tensor,
        cache_steps: int,
        retrieve_parent_token: Optional[torch.Tensor],
        lower_bound: Optional[float] = None,
        native_correction: Optional[torch.Tensor] = None,
        native_key: Optional[torch.Tensor] = None,
        native_decay: Optional[torch.Tensor] = None,
        is_ragged_verify: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        batch_size = query_start_loc.numel() - 1
        _, seq_len, heads, head_dim = q.shape
        if a.ndim == 3 and a.shape == (1, seq_len, heads * head_dim):
            a = a.unflatten(-1, (heads, head_dim))
        dense = (
            not is_ragged_verify
            and retrieve_parent_token is None
            and lower_bound is not None
            and q.shape[0] == 1
            and seq_len == batch_size * cache_steps
            and head_dim == 128
            and ssm_states.is_contiguous()
            and all(
                x.shape == q.shape and x.dtype == torch.bfloat16 for x in (q, k, v, a)
            )
            and b.shape == (1, seq_len, heads)
            and b.dtype == torch.bfloat16
        )
        if native_correction is not None:
            if not dense:
                raise ValueError(
                    "FlashKDA ReplaySSM requires dense top-1 safe-gate verify"
                )
            flash_kda = _load_flash_kda(ppu=True)
            verify = getattr(flash_kda, "verify_replayssm", None)
            if not callable(verify):
                raise RuntimeError(
                    "PPU FlashKDA 0dfef21 with verify_replayssm is required."
                )
            verify_args = (native_correction, native_key, native_decay)
            logger.info_once("FlashKDA ReplaySSM verify active")
        elif intermediate_states_buffer is None:
            raise RuntimeError("FlashKDA verify cannot use a raw-input ReplaySSM ring")
        elif (
            dense
            and intermediate_states_buffer.is_contiguous()
            and intermediate_states_buffer.dtype == ssm_states.dtype
            and batch_size * heads <= 768
        ):
            verify = _load_flash_kda(ppu=True).verify_strided
            verify_args = (
                intermediate_states_buffer,
                intermediate_state_indices[:batch_size].to(torch.int32).contiguous(),
            )
        else:
            from sglang.srt.layers.attention.linear.kernels.kda_triton import (
                TritonKDAKernel,
            )

            return TritonKDAKernel().target_verify(
                A_log=A_log,
                dt_bias=dt_bias,
                q=q,
                k=k,
                v=v,
                a=a,
                b=b,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                intermediate_states_buffer=intermediate_states_buffer,
                intermediate_state_indices=intermediate_state_indices,
                cache_steps=cache_steps,
                retrieve_parent_token=retrieve_parent_token,
                lower_bound=lower_bound,
                **kwargs,
            )
        if batch_size == 0:
            return torch.empty_like(q)
        dense_shape = (batch_size, cache_steps, heads, 128)
        out = q.new_empty(dense_shape)
        verify(
            q.view(dense_shape),
            k.view(dense_shape),
            v.view(dense_shape),
            a.view(dense_shape),
            b.reshape(batch_size, cache_steps, heads).contiguous(),
            A_log.reshape(-1).float().contiguous(),
            dt_bias.reshape(heads, 128).float().contiguous(),
            128**-0.5,
            lower_bound,
            ssm_states,
            cache_indices[:batch_size].to(torch.int32).contiguous(),
            *verify_args,
            out,
        )
        return out.flatten(0, 1).unsqueeze(0)
