"""PPU SM89 KDA compact verification and accepted-prefix state recovery."""

from typing import Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.jit.utils.common import is_sail_runtime
from sglang.srt.utils.custom_op import register_custom_op


@cache_once
def _jit_module():
    if not is_sail_runtime() or torch.cuda.get_device_capability() != (8, 9):
        raise RuntimeError("Native KDA ReplaySSM requires PPU SM89")
    return load_jit(
        "kda_replayssm_ppu",
        cuda_files=["attention/kda_replayssm_ppu.cuh"],
        cuda_wrappers=[
            ("verify", "replay_verify"),
            ("verify_snapshot", "replay_verify_snapshot"),
            ("commit", "replay_commit"),
            ("commit_track", "replay_commit_track"),
        ],
        extra_cuda_cflags=["--use_fast_math"],
    )


@register_custom_op(mutates_args=["correction", "key", "decay", "out"])
def _verify(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    indices: torch.Tensor,
    correction: torch.Tensor,
    key: torch.Tensor,
    decay: torch.Tensor,
    out: torch.Tensor,
    lower_bound: float,
) -> None:
    _jit_module().verify(
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        state,
        indices,
        correction,
        key,
        decay,
        out,
        lower_bound,
    )


def target_verify_kda_replayssm_ppu(
    *,
    q,
    k,
    v,
    a,
    b,
    A_log,
    dt_bias,
    ssm_states,
    cache_indices,
    correction,
    key,
    decay,
    lower_bound,
    retrieve_parent_token=None,
):
    if retrieve_parent_token is not None or lower_bound is None:
        raise ValueError("Native PPU ReplaySSM requires a linear safe-gate chain")
    out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
    _verify(
        q,
        k,
        v,
        a,
        b.contiguous(),
        A_log.reshape(-1).float().contiguous(),
        dt_bias.reshape(q.shape[2], 128).float().contiguous(),
        ssm_states,
        cache_indices.contiguous(),
        correction,
        key,
        decay,
        out,
        float(lower_bound),
    )
    return out


@register_custom_op(mutates_args=["state"])
def commit_kda_replayssm_ppu(
    state: torch.Tensor,
    correction: torch.Tensor,
    key: torch.Tensor,
    decay: torch.Tensor,
    indices: torch.Tensor,
    accept_lens: torch.Tensor,
    tracks: Optional[torch.Tensor] = None,
    steps: Optional[torch.Tensor] = None,
) -> None:
    if (tracks is None) != (steps is None):
        raise ValueError("ReplaySSM track indices and steps must be paired")
    if tracks is None:
        _jit_module().commit(state, correction, key, decay, indices, accept_lens)
    else:
        _jit_module().commit_track(
            state, correction, key, decay, indices, accept_lens, tracks, steps
        )
