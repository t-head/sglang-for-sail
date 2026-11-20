"""Resolved rank-local bounds for optional PPU prefill preparation."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PrefillWarmupRuntime:
    chunked_prefill_size: object
    max_prefill_tokens: object
    page_size: int
    attn_dp_size: int
    attn_tp_size: int
    attn_cp_size: int
    tp_size: int
    moe_ep_size: int
    moe_a2a_backend: str
    mixed_chunk: bool
    dynamic_chunk: bool
    encoder_replay: bool
    prefill_graph_disabled: bool


def plan_dp_prefill_rows(server_args, *, model_type, is_sail, runtime):
    """Complete local row domain, only for the audited non-gather DP layout.

    max_prefill_tokens is a soft admission budget, not a hard row cap. The
    resolved positive chunk is the limit; never divide it by attention DP.
    """
    if runtime is None or not is_sail or model_type != "deepseek_v41":
        return ()
    if (
        "prefill_shapes" not in (server_args.warmups or ())
        or server_args.enable_prefill_cp
    ):
        return ()
    widths = (
        runtime.attn_dp_size,
        runtime.attn_tp_size,
        runtime.attn_cp_size,
        runtime.tp_size,
        runtime.moe_ep_size,
        runtime.page_size,
    )
    if any(type(n) is not int or n < 1 for n in widths):
        return ()
    if not (
        runtime.attn_dp_size > 1
        and runtime.attn_tp_size == 1
        and runtime.attn_cp_size == 1
        and runtime.attn_dp_size == runtime.moe_ep_size == runtime.tp_size
        and runtime.moe_a2a_backend == "megamoe"
    ):
        return ()
    if (
        runtime.mixed_chunk
        or runtime.dynamic_chunk
        or runtime.encoder_replay
        or runtime.prefill_graph_disabled is not True
    ):
        return ()
    chunk = runtime.chunked_prefill_size
    if type(chunk) is not int or not 0 < chunk <= 1 << 20:
        return ()
    if chunk % math.lcm(runtime.page_size, runtime.attn_tp_size):
        return ()
    return tuple(range(1, chunk + 1))
