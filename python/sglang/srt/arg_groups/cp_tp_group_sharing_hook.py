# SPDX-License-Identifier: Apache-2.0
"""Model integration and topology checks for CP-TP group sharing: prefill CP
for hybrid linear-attention models where the CP group is the TP group."""

import os
from typing import Any

from sglang.srt.arg_groups.overrides import (
    declare_resolution,
    model_config_of,
    resolved_view,
    resolving_view,
)
from sglang.srt.configs.hybrid_arch import mambaish_config
from sglang.srt.environ import envs
from sglang.srt.model_executor.cuda_graph_config import Backend, Phase, with_phase
from sglang.srt.runtime_context import get_platform

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


def _ppu_mooncake_ep_admitted(cfg: Any, view: Any, raw_transfer_backend: Any) -> bool:
    """The one ported GLM PD EP layout: single-node PPU SM89, Mooncake RDMA,
    ep_size == attn_cp_size == 8, speculation limited to native MTP.

    PPU reports SM89 through device_sm; do not broaden this layout to CUDA.
    MC_FORCE_TCP is presence-sensitive upstream (any value, even "0" or the
    empty string, forces the TCP transport), so its mere presence rejects the
    layout. The raw transfer backend must be mooncake as well: a mooncake_tcp
    alias normalized to mooncake during resolution must not pass this gate.
    """
    return (
        cfg.disaggregation_mode == "prefill"
        and cfg.disaggregation_transfer_backend == "mooncake"
        and raw_transfer_backend == "mooncake"
        and get_platform().is_ppu
        and get_platform().device_sm == 89
        and envs.MOONCAKE_PROTOCOL.get().lower() == "rdma"
        and "MC_FORCE_TCP" not in os.environ
        and cfg.nnodes == 1
        and view.ep_size == view.attn_cp_size == 8
        and not cfg.enable_pd_role_switch
        and not cfg.enable_unified_memory
        and (cfg.speculative_algorithm or "").upper() in ("", "NEXTN", "EAGLE")
    )


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
    if cfg.speculative_algorithm is not None:
        if arch == "KimiK31ForConditionalGeneration":
            if cfg.speculative_algorithm.upper() != "DFLASH":
                raise ValueError(
                    "Kimi-K3.1 CP-TP group sharing only supports --speculative-algorithm DFLASH."
                )
        elif (
            model.hf_config.architectures[0] == "Qwen4ExpForConditionalGeneration"
            and cfg.speculative_algorithm.upper() == "DFLASH"
        ):
            raise ValueError("Qwen4 CP-TP group sharing supports native MTP only.")
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
            # Prefill only: a decode node serves plain TP.
            "disaggregation-mode decode": cfg.disaggregation_mode == "decode",
        }
    elif model.hf_config.architectures[0] == "Glm5NextForConditionalGeneration":
        if cfg.disaggregation_mode != "null":
            # Decode pairs with this prefill as plain TP whether it runs no
            # speculation or one of the algorithms the check above admits.
            ppu_mooncake_ep = _ppu_mooncake_ep_admitted(
                cfg, view, server_args.disaggregation_transfer_backend
            )
            unsupported |= {
                "disaggregation-mode decode": cfg.disaggregation_mode == "decode",
                "disaggregation-transfer-backend != mooncake": (
                    cfg.disaggregation_transfer_backend != "mooncake"
                ),
                "pp-size > 1": cfg.pp_size > 1,
                "dp-size > 1": cfg.dp_size > 1,
                "ep-size > 1": view.ep_size > 1 and not ppu_mooncake_ep,
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


def validate_cp_tp_group_sharing_final_ep(server_args: Any) -> None:
    """Re-run the GLM PD EP admission against the post-A2A EP topology.

    handle_context_parallelism runs before handle_a2a_moe, and the A2A pass
    expands ep_size to tp_size for DeepEP and the other EP-spanning backends.
    An EP1 input passes the admission hook's ep-size check trivially, so
    without this re-check an auto-expanded EP8 deployment would skip the
    ported-layout gate entirely. Runs after handle_a2a_moe in the pipeline.
    """
    view = resolved_view(server_args)
    if not view.enable_cp_tp_group_sharing or view.ep_size <= 1:
        return
    if view.disaggregation_mode == "null":
        return
    arch = model_config_of(server_args).hf_config.architectures[0]
    if arch != "Glm5NextForConditionalGeneration":
        return
    if not _ppu_mooncake_ep_admitted(
        view, view, server_args.disaggregation_transfer_backend
    ):
        raise ValueError(
            "--ep-size > 1 is not supported with CP-TP group sharing: the "
            f"resolved MoE topology (ep_size={view.ep_size}, "
            f"attn_cp_size={view.attn_cp_size}) is not the ported GLM PD "
            "layout (single-node PPU SM89, Mooncake RDMA, "
            "ep_size == attn_cp_size == 8, speculation limited to native MTP)."
        )
