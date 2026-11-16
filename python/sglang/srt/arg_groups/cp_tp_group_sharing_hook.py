# SPDX-License-Identifier: Apache-2.0
"""Model integration and topology checks for CP-TP group sharing: prefill CP
for hybrid linear-attention models where the CP group is the TP group."""

from typing import Any

from sglang.srt.arg_groups.overrides import (
    declare_resolution,
    resolved_view,
    resolving_view,
)
from sglang.srt.configs.hybrid_arch import mambaish_config
from sglang.srt.environ import envs
from sglang.srt.model_executor.cuda_graph_config import Backend, Phase, with_phase

# Architectures whose model code implements CP-TP group sharing: the CP group
# is the TP group, the residual stream / attention / indexer are CP-sharded,
# MoE and linear attention keep their TP partition (the linear attention
# gathers the sequence where its recurrence needs it). Model ports register
# here; every other model keeps the ordinary prefill CP path.
_SUPPORTED_MODELS: set[str] = {
    "Glm5NextForConditionalGeneration",
    "Qwen4ExpForConditionalGeneration",
    "KimiK31ForConditionalGeneration",
}
_INTERLEAVE_MODELS: set[str] = {
    "Glm5NextForConditionalGeneration",
    "KimiK31ForConditionalGeneration",
}


def _pd_spec_conflicts_with_cp_tp_sharing(cfg: Any, arch: str) -> bool:
    """Whether a PD + CP-TP-group-sharing Glm5Next node asks for speculation this
    combination cannot serve.

    Two sets are described separately on purpose: this predicate's own value
    domain, and what its caller actually lets through to it.

    Value domain: this function answers "no conflict" for
    {None, "DFLASH", "dflash"} on Glm5NextForConditionalGeneration, and for
    everything on any other architecture. None is the reason it exists --
    `speculative_algorithm` defaults to None when the flag is not given
    (arg_groups/fields/spec.py:31-34), and not speculating is a supported PD
    configuration, which the bare comparison it replaced treated as illegal.

    Caller's accepted set: it is narrower than the value domain. The only values
    that get past the whitelist further below in this hook (the one raising
    "CP/TP group sharing only supports DFLASH or native MTP with top-k 1") with
    speculation requested are the exact literal "DFLASH" and a native-MTP shape
    (an EAGLE/NEXTN spelling whose draft model and top-k match the target), and a
    native-MTP shape is then reported as a conflict here. Every other spelling --
    lower-case "dflash" included -- is rejected by that whitelist before the
    disaggregation branch is ever built, so "dflash" in the value domain is
    defensive rather than a supported configuration: no command-line spelling
    reaches the disaggregation branch through it, and it newly accepts nothing.
    It is kept because the matching guard in arg_groups/speculative_hook.py
    compares the settled value after `upper()`, and the two touchpoints should
    agree about what that value may be.

    Uppercasing is deliberately not generalized here: normalizing the accepted
    spellings would only ever widen what this branch tolerates, and that is a
    behavior change this fix has no evidence for. Anything else, including a
    malformed value, still reports a conflict. Glm5Next is the only architecture
    with this PD restriction; Qwen4Exp and KimiK31 keep their existing
    behavior."""
    if arch != "Glm5NextForConditionalGeneration":
        return False
    return cfg.speculative_algorithm not in (None, "DFLASH", "dflash")


def resolve_cp_tp_group_sharing(server_args: Any, model: Any) -> None:
    cfg, view = resolving_view(server_args), resolved_view(server_args)
    if (
        not cfg.enable_prefill_cp
        or view.attn_cp_size <= 1
        or model.hf_config.architectures[0] not in _SUPPORTED_MODELS
    ):
        return
    # Attention DP or attention TP would leave the CP group narrower than TP.
    if view.attn_cp_size != cfg.tp_size:
        raise ValueError(
            "CP-TP group sharing requires --attn-cp-size == --tp-size, got "
            f"attn_cp_size={view.attn_cp_size}, tp_size={cfg.tp_size}."
        )
    strategy = (
        "interleave"
        if model.hf_config.architectures[0] in _INTERLEAVE_MODELS
        else "zigzag"
    )
    if cfg.cp_strategy != strategy:
        raise ValueError(f"CP-TP group sharing requires --cp-strategy {strategy}.")

    arch = model.hf_config.architectures[0]
    # This hook runs before speculative defaults and NEXTN alias resolution.
    # Native MTP keeps the target's CP/TP placement. DFLASH consumes the
    # target's gathered hidden states and runs its draft with ordinary TP.
    # Only GLM-5.3-Flash has these speculative paths ported.
    if cfg.speculative_algorithm is not None:
        if arch == "KimiK31ForConditionalGeneration":
            if cfg.speculative_algorithm.upper() != "DFLASH":
                raise ValueError(
                    "Kimi-K3.1 CP-TP group sharing only supports --speculative-algorithm DFLASH."
                )
        elif model.hf_config.architectures[0] != "Glm5NextForConditionalGeneration":
            raise ValueError(
                "--speculative-algorithm is not supported with CP-TP group sharing."
            )
        if (
            arch != "KimiK31ForConditionalGeneration"
            and cfg.speculative_algorithm != "DFLASH"
            and (
                cfg.speculative_algorithm.upper() not in ("EAGLE", "NEXTN")
                or cfg.speculative_draft_model_path not in (None, cfg.model_path)
                or cfg.speculative_eagle_topk not in (None, 1)
            )
        ):
            raise ValueError(
                "CP/TP group sharing only supports DFLASH or native MTP with top-k 1."
            )

    unsupported = {
        "enable-dsa-cache-layer-split": cfg.enable_dsa_cache_layer_split,
        "dcp-size > 1": cfg.dcp_size > 1,
        "enable-mixed-chunk": cfg.enable_mixed_chunk,
        "moe-dp-size > 1": cfg.moe_dp_size > 1,
    }
    if arch == "KimiK31ForConditionalGeneration":
        unsupported["disaggregation-mode decode"] = cfg.disaggregation_mode == "decode"
    elif model.hf_config.architectures[0] == "Qwen4ExpForConditionalGeneration":
        unsupported |= {
            # The MoE runs TP experts on rows gathered over the TP group.
            # Waterfill turns on DeepEP and EP after this hook.
            "ep-size > 1": view.ep_size > 1,
            "moe-a2a-backend": view.moe_a2a_backend != "none",
            "enable-waterfill": cfg.enable_waterfill,
            "pp-size > 1": cfg.pp_size > 1,
            # Prefill only: a decode node serves plain TP.
            "disaggregation-mode decode": cfg.disaggregation_mode == "decode",
        }
    elif model.hf_config.architectures[0] == "Glm5NextForConditionalGeneration":
        if cfg.disaggregation_mode != "null":
            unsupported |= {
                "disaggregation-mode decode": cfg.disaggregation_mode == "decode",
                "disaggregation-transfer-backend != mooncake": (
                    cfg.disaggregation_transfer_backend != "mooncake"
                ),
                "speculative-algorithm != DFLASH": _pd_spec_conflicts_with_cp_tp_sharing(
                    cfg, arch
                ),
                "pp-size > 1": cfg.pp_size > 1,
                "dp-size > 1": cfg.dp_size > 1,
                "ep-size > 1": view.ep_size > 1,
            }
    for flag, enabled in unsupported.items():
        if enabled:
            raise ValueError(f"--{flag} is not supported with CP-TP group sharing.")
    # Decode sees this prefill as TP cp_size, while its sender stages by
    # attention TP 1, so the two would pick different staging layouts.
    if cfg.disaggregation_mode == "prefill" and envs.SGLANG_DISAGG_STAGING_BUFFER.get():
        raise ValueError(
            "SGLANG_DISAGG_STAGING_BUFFER is not supported with CP-TP group sharing."
        )

    linear_config = mambaish_config(model)
    if getattr(linear_config, "linear_num_key_heads", None) is not None:
        heads = {
            name: getattr(linear_config, name)
            for name in ("linear_num_key_heads", "linear_num_value_heads")
        }
    else:
        heads = {"num_heads": linear_config.linear_attn_config["num_heads"]}
    # Linear attention keeps its TP head partition over the whole TP group.
    for name, count in heads.items():
        if count % cfg.tp_size:
            raise ValueError(
                f"{name}={count} must be divisible by --tp-size={cfg.tp_size}."
            )

    fields = {"enable_cp_tp_group_sharing": True}
    if model.hf_config.architectures[0] == "Qwen4ExpForConditionalGeneration":
        # Qwen's sequence collectives still have batch-dependent sizes.
        fields["cuda_graph_config"] = with_phase(
            cfg.cuda_graph_config, Phase.PREFILL, backend=Backend.DISABLED
        )
    declare_resolution(server_args, "resolve_cp_tp_group_sharing", **fields)
