from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

from sglang.srt.arg_groups.overrides import (
    declare_resolution,
    model_config_of,
    resolving_view,
)
from sglang.srt.connector import ConnectorType
from sglang.srt.environ import envs
from sglang.srt.runtime_context import attn_dp_enabled_of, get_platform
from sglang.srt.utils.common import ceil_align, ceil_div, parse_connector_type

logger = logging.getLogger(__name__)

# The PPU path drives DeepGEMM's MXFP4 MegaMoE kernels directly instead of the
# MoE runner, so the build has to expose all four entry points.
_PPU_MEGA_MOE_DEEP_GEMM_APIS = (
    "get_symm_buffer_for_mega_moe",
    "mega_moe_pre_dispatch",
    "fp4_fp4_mega_moe",
    "transform_weights_for_mega_moe",
)


def _require_ppu_mega_moe_deep_gemm_apis() -> None:
    try:
        import deep_gemm
    except ImportError as exc:
        raise ValueError(
            "--moe-a2a-backend megamoe on PPU requires the PPU DeepGEMM build"
        ) from exc

    missing = [
        name for name in _PPU_MEGA_MOE_DEEP_GEMM_APIS if not hasattr(deep_gemm, name)
    ]
    if missing:
        raise ValueError(
            "--moe-a2a-backend megamoe on PPU requires a DeepGEMM build with the "
            "PPU MegaMoE entry points; missing: " + ", ".join(missing)
        )


def handle_mega_moe(server_args: ServerArgs) -> None:
    handle_moe_runner_backend_alias(server_args)
    check_mega_moe_compat(server_args)


def check_mega_moe_compat(server_args: ServerArgs) -> None:
    cfg = resolving_view(server_args)
    if cfg.moe_a2a_backend != "megamoe":
        return
    if cfg.enable_two_batch_overlap or cfg.enable_single_batch_overlap:
        # The fused kernel has no dispatch_a / combine_a split for TBO / SBO.
        raise ValueError(
            "--moe-a2a-backend megamoe has no two-batch / single-batch overlap "
            "decomposition; disable --enable-two-batch-overlap and "
            "--enable-single-batch-overlap."
        )
    platform = get_platform()
    if platform.is_ppu:
        if platform.device_sm < 89:
            raise ValueError("--moe-a2a-backend megamoe on PPU requires SM89 or newer.")
        _require_ppu_mega_moe_deep_gemm_apis()
    elif not (
        (platform.is_cuda and (platform.is_sm90 or platform.is_sm100))
        or platform.is_hip
    ):
        raise ValueError(
            "--moe-a2a-backend megamoe needs a CUDA SM90 GPU (block-FP8 experts) "
            "or an SM100-class GPU (MXFP4 / NVFP4 experts); it runs DeepGEMM "
            "kernels over CUDA symmetric memory. On ROCm, set "
            "SGLANG_AMD_USE_FLYDSL_MEGA_MOE=1 to select aiter MegaMoEv2 instead."
        )


# MoE blocks with no fused-MoE fallback under megamoe.
MEGA_MOE_NO_FALLBACK_ARCHS = frozenset(
    {
        "DeepseekV4ForCausalLM",
        "GlmMoeDsaForCausalLM",
        "Glm5NextForConditionalGeneration",
        "KimiK31ForConditionalGeneration",
        "InternS2PreviewForConditionalGeneration",
        "MellumForCausalLM",
        "MiniMaxM3SparseForCausalLM",
        "MiniMaxM3SparseForConditionalGeneration",
        "Qwen2MoeForCausalLM",
        "Qwen3MoeForCausalLM",
        "Qwen3NextForCausalLM",
        "Qwen3VLMoeForConditionalGeneration",
        "Qwen3_5MoeForCausalLM",
        "Qwen3_5MoeForConditionalGeneration",
    }
)


def mega_moe_needs_token_budget(
    model_arch: str, hf_quant_config: dict, quantization: str | None
) -> bool:
    # NVFP4 experts are repacked into the mega layout at load: no fallback.
    nvfp4_experts = quantization in (
        "modelopt_fp4",
        "minimax_m3_nvfp4",
    ) or "FP4" in str(hf_quant_config.get("quant_algo", ""))
    return model_arch in MEGA_MOE_NO_FALLBACK_ARCHS or nvfp4_experts


_PPU_CHANNELWISE_FP8_ARCHS = frozenset(
    {
        "Glm5NextForConditionalGeneration",
        "MiniMaxM3SparseForCausalLM",
        "MiniMaxM3SparseForConditionalGeneration",
    }
)


def _ppu_channelwise_fp8_routed_experts(hf_quant_config: dict) -> bool:
    if hf_quant_config.get("quant_method") != "compressed-tensors":
        return False

    # Resolve the same expert projection targets and ignore rules used while
    # constructing FusedMoE, rather than accepting a dense-only FP8 group.
    from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
    from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )

    quant_config = CompressedTensorsConfig.from_config(hf_quant_config)
    quant_config._add_fused_moe_to_target_scheme_map()
    projection = object.__new__(FusedMoE)
    for prefix in (
        "model.layers.0.mlp.experts.0.",
        "model.language_model.layers.0.mlp.experts.0.",
    ):
        try:
            schemes = [
                quant_config.get_scheme_dict(projection, prefix + name)
                for name in ("gate_proj", "up_proj", "down_proj")
            ]
        except ValueError:
            continue
        if any(scheme is None for scheme in schemes) or any(
            scheme != schemes[0] for scheme in schemes[1:]
        ):
            continue
        weights = schemes[0].get("weights")
        activations = schemes[0].get("input_activations")
        if (
            weights is not None
            and weights.strategy == "channel"
            and weights.type == "float"
            and weights.num_bits == 8
            and activations is not None
            and activations.strategy == "token"
            and activations.type == "float"
            and activations.num_bits == 8
            and activations.dynamic is True
        ):
            return True
    return False


def _validate_ppu_channelwise_fp8_eager(cfg, channelwise_fp8: bool) -> None:
    if not channelwise_fp8 or not get_platform().is_ppu:
        return

    from sglang.srt.model_executor.cuda_graph_config import Backend

    graph = cfg.cuda_graph_config
    if (
        graph is None
        or graph.prefill.backend != Backend.DISABLED
        or graph.decode.backend != Backend.DISABLED
    ):
        raise ValueError(
            "PPU channelwise FP8 MegaMoE requires eager execution; "
            "use --disable-cuda-graph."
        )


def validate_mega_moe_token_budget_for_model(server_args: ServerArgs) -> None:
    # Runs after speculative resolution so the decode / verify bound is final.
    cfg = resolving_view(server_args)
    from sglang.srt.layers.moe.ultra_ep_ppu import mega_moe_runtime

    ultra_ep_enabled = mega_moe_runtime.is_enabled()
    if ultra_ep_enabled and cfg.moe_a2a_backend != "megamoe":
        raise ValueError("UltraEP MegaMoE requires --moe-a2a-backend megamoe.")
    if ultra_ep_enabled and cfg.encoder_only:
        raise ValueError("UltraEP MegaMoE does not support encoder-only models.")
    if cfg.moe_a2a_backend != "megamoe" or cfg.encoder_only:
        return
    validate_flydsl_decode_capacity(server_args)

    if parse_connector_type(cfg.model_path) == ConnectorType.INSTANCE:
        if ultra_ep_enabled:
            raise ValueError("UltraEP MegaMoE does not support instance connectors.")
        return
    model_config = model_config_of(server_args)
    model_arch = model_config.hf_config.architectures[0]
    channelwise_fp8 = model_arch in _PPU_CHANNELWISE_FP8_ARCHS and (
        _ppu_channelwise_fp8_routed_experts(model_config.hf_quant_config)
    )
    _validate_ppu_ultra_ep(
        cfg, model_arch, model_config.hf_quant_config, channelwise_fp8
    )
    _validate_ppu_channelwise_fp8_eager(cfg, channelwise_fp8)
    if mega_moe_needs_token_budget(
        model_arch, model_config.hf_quant_config, cfg.quantization
    ):
        validate_mega_moe_token_budget(
            server_args, model_arch, ppu_channelwise_fp8=channelwise_fp8
        )


def _validate_ppu_ultra_ep(
    cfg,
    model_arch: str,
    hf_quant_config: dict,
    channelwise_fp8: bool | None = None,
) -> None:
    from sglang.srt.layers.moe.ultra_ep_ppu import mega_moe_runtime

    if not mega_moe_runtime.is_enabled():
        return
    platform = get_platform()
    if not platform.is_ppu or platform.device_sm != 89:
        raise ValueError("UltraEP MegaMoE requires PPU SM89.")
    if (
        (
            cfg.disaggregation_mode,
            cfg.tp_size,
            cfg.ep_size,
            cfg.attn_cp_size,
        )
        not in (("null", 4, 4, 4), ("null", 8, 8, 8), ("prefill", 8, 8, 8))
        or cfg.nnodes != 1
        or cfg.pp_size != 1
        or not cfg.enable_prefill_cp
        or cfg.cp_strategy != "interleave"
    ):
        raise ValueError(
            "UltraEP MegaMoE requires single-node interleave CP: "
            "standalone TP4/CP4/EP4 or TP8/CP8/EP8, or TP8/CP8/EP8 PD prefill."
        )
    if not cfg.disable_overlap_schedule:
        raise ValueError("UltraEP MegaMoE requires --disable-overlap-schedule.")
    if channelwise_fp8 is None:
        channelwise_fp8 = _ppu_channelwise_fp8_routed_experts(hf_quant_config)
    if (
        model_arch != "Glm5NextForConditionalGeneration"
        or hf_quant_config.get("quant_method") != "compressed-tensors"
        or not channelwise_fp8
        or not cfg.disable_shared_experts_fusion
        or cfg.enforce_shared_experts_fusion
    ):
        raise ValueError(
            "UltraEP MegaMoE requires GLM-5.3-Flash channelwise FP8 and "
            "--disable-shared-experts-fusion."
        )
    if (
        cfg.enable_eplb
        or cfg.enable_eplb_async
        or cfg.enable_waterfill
        or cfg.ep_num_redundant_experts
        or cfg.ep_dispatch_algorithm is not None
        or cfg.init_expert_location != "trivial"
        or cfg.expert_distribution_recorder_mode is not None
    ):
        raise ValueError("UltraEP MegaMoE must be the only expert placement owner.")

    from sglang.srt.model_executor.cuda_graph_config import Backend

    graph = cfg.cuda_graph_config
    if (
        graph is None
        or graph.prefill.backend != Backend.DISABLED
        or graph.decode.backend != Backend.DISABLED
    ):
        raise ValueError(
            "UltraEP MegaMoE requires eager execution; use --disable-cuda-graph."
        )
    mega_moe_runtime.redundant_per_rank()


def mega_moe_decode_tokens_per_rank(cfg) -> int:
    cg_config = cfg.cuda_graph_config
    decode_max_bs = (cg_config.decode.max_bs if cg_config is not None else 0) or 0
    if cfg.max_running_requests is not None:
        per_rank_requests = ceil_align(
            cfg.max_running_requests // cfg.attn_dp_size,
            cfg.tp_size // cfg.attn_dp_size,
        )
        decode_max_bs = min(decode_max_bs, per_rank_requests)
    num_tokens_per_req = (
        (cfg.speculative_num_draft_tokens or 1) if cfg.speculative_algorithm else 1
    )
    return decode_max_bs * num_tokens_per_req


def flydsl_decode_tokens_per_rank(cfg) -> int | None:
    """Rows of the largest decode-step forward on one rank, or None while the
    request cap is still to be sized from memory.

    Eager batches and decode graph buckets are both capped at the per-rank
    request pool; a request runs at most max(draft tokens, top-k) rows
    (verify and draft extend, draft decode).
    """
    if cfg.max_running_requests is None:
        return None
    requests = cfg.max_running_requests
    if attn_dp_enabled_of(cfg):
        requests //= cfg.attn_dp_size
    tokens_per_req = 1
    if cfg.speculative_algorithm:
        tokens_per_req = max(
            cfg.speculative_num_draft_tokens or 1, cfg.speculative_eagle_topk or 1
        )
    return requests * tokens_per_req


def validate_flydsl_decode_capacity(server_args: ServerArgs) -> None:
    if not (get_platform().is_hip and envs.SGLANG_AMD_USE_FLYDSL_MEGA_MOE.get()):
        return
    decode_mtpr = envs.SGLANG_AMD_FLYDSL_MEGA_MOE_DECODE_MTPR.get()
    if decode_mtpr <= 0 or decode_mtpr >= envs.SGLANG_AMD_FLYDSL_MEGA_MOE_MTPR.get():
        # Off, one merged capacity, or rejected when the capacities are built.
        return
    cfg = resolving_view(server_args)
    if cfg.disaggregation_mode == "prefill":
        return
    if cfg.attn_cp_size > 1:
        raise ValueError(
            "SGLANG_AMD_FLYDSL_MEGA_MOE_DECODE_MTPR is not supported with "
            "attention context parallelism (--attn-cp-size > 1); set it to 0."
        )
    decode_tokens = flydsl_decode_tokens_per_rank(cfg)
    if decode_tokens is None:
        raise ValueError(
            "SGLANG_AMD_FLYDSL_MEGA_MOE_DECODE_MTPR needs --max-running-requests "
            "to bound the largest decode / verify forward on one rank."
        )
    if decode_tokens > decode_mtpr:
        raise ValueError(
            f"SGLANG_AMD_FLYDSL_MEGA_MOE_DECODE_MTPR={decode_mtpr} must cover the "
            "largest decode / verify forward on one rank: max running requests "
            f"per DP rank x tokens per request = {decode_tokens}. Raise it to a "
            f"power of two of at least {decode_tokens} or lower "
            "--max-running-requests."
        )


def validate_mega_moe_token_budget(
    server_args: ServerArgs,
    model_label: str,
    *,
    ppu_channelwise_fp8: bool = False,
) -> None:
    cfg = resolving_view(server_args)
    if cfg.moe_a2a_backend != "megamoe":
        return
    if get_platform().is_hip:
        return

    max_tokens_per_rank = (
        envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.get()
    )
    eager_and_graph_budget_models = [
        "KimiK31ForConditionalGeneration",
        "MiniMaxM3SparseForCausalLM",
        "MiniMaxM3SparseForConditionalGeneration",
    ]
    eager_and_graph_budget = (
        ppu_channelwise_fp8 or model_label in eager_and_graph_budget_models
    )
    if cfg.disaggregation_mode != "prefill":
        decode_tokens = mega_moe_decode_tokens_per_rank(cfg)
        if eager_and_graph_budget:
            # Repacked experts cannot fall back for oversized eager batches.
            if cfg.max_running_requests is None:
                raise ValueError(
                    f"{model_label} MegaMoE needs --max-running-requests to bound eager decode / verify rows."
                )
            eager_requests = ceil_align(
                cfg.max_running_requests // cfg.attn_dp_size,
                cfg.tp_size // cfg.attn_dp_size,
            )
            tokens_per_request = (
                (cfg.speculative_num_draft_tokens or 1)
                if cfg.speculative_algorithm
                else 1
            )
            decode_tokens = max(decode_tokens, eager_requests * tokens_per_request)
        if max_tokens_per_rank < decode_tokens:
            raise ValueError(
                f"{model_label} with MegaMoE requires "
                "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK to cover the "
                "largest decode / verify forward on one rank. Current values: "
                f"largest decode graph bucket x tokens per request = {decode_tokens}, "
                f"SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK="
                f"{max_tokens_per_rank}. Raise the env var to at least "
                f"{decode_tokens} or lower --cuda-graph-max-bs / "
                "--max-running-requests / --speculative-num-draft-tokens."
            )
    if cfg.disaggregation_mode == "decode":
        # decode node will skip the check because decode bs is not relevant with --chunk-prefill-size
        return

    if cfg.pp_size > 1 and cfg.enable_dynamic_chunking:
        return

    if cfg.chunked_prefill_size is None or cfg.chunked_prefill_size <= 0:
        raise ValueError(
            f"{model_label} with MegaMoE requires chunked prefill to be enabled. "
            "Set --chunked-prefill-size to a positive value; "
            "--chunked-prefill-size=-1 is unsafe because MegaMoE's per-rank "
            "token requirement would not have a strict prefill-forward bound."
        )

    prefill_tokens = cfg.chunked_prefill_size
    kimi = model_label == "KimiK31ForConditionalGeneration"
    if eager_and_graph_budget:
        prefill_tokens = max(prefill_tokens, getattr(cfg, "max_prefill_tokens", 0) or 0)
        graph = cfg.cuda_graph_config.prefill if cfg.cuda_graph_config else None
        if graph is not None and graph.backend != "disabled":
            prefill_tokens = max(
                prefill_tokens, graph.max_bs or 0, max(graph.bs or [0])
            )

    if cfg.enable_prefill_cp:
        token_partition_size = cfg.attn_cp_size
        token_partition_name = "attn_cp_size"
        token_alignment = token_partition_size if kimi else 1
        local_chunked_prefill_size = (
            prefill_tokens + token_partition_size - 1
        ) // token_partition_size
    elif attn_dp_enabled_of(cfg):
        token_partition_size = cfg.attn_dp_size
        token_partition_name = "attn_dp_size (already applied)"
        token_alignment = max(
            cfg.tp_size // cfg.attn_dp_size // cfg.attn_cp_size,
            1,
        )
        local_chunked_prefill_size = cfg.chunked_prefill_size
        # CP has its own padding and layout; keep its existing budget.
        if cfg.attn_cp_size == 1 and token_alignment > 1:
            graph = cfg.cuda_graph_config.prefill if cfg.cuda_graph_config else None
            if graph is not None and graph.backend != "disabled" and graph.bs:
                local_chunked_prefill_size = max(
                    local_chunked_prefill_size, max(graph.bs)
                )
            token_partition_size, token_alignment = token_alignment, 1
            token_partition_name = "attn_tp_size (after attention DP)"
            local_chunked_prefill_size = ceil_div(
                local_chunked_prefill_size, token_partition_size
            )
    else:
        # Pure TP and PP with static chunking are handled here.
        token_partition_size = 1
        token_partition_name = "none"
        # global_num_tokens will ceil_align to attn_tp_size so the validation needs to do alignment as well
        token_alignment = max(
            cfg.tp_size // token_partition_size // cfg.attn_cp_size,
            1,
        )
        local_chunked_prefill_size = prefill_tokens

    if local_chunked_prefill_size <= 0:
        raise ValueError(
            f"{model_label} with MegaMoE requires a positive effective per-rank "
            "chunked prefill size. "
            f"Current values: chunked_prefill_size="
            f"{cfg.chunked_prefill_size}, "
            f"token_partition={token_partition_name}, "
            f"token_partition_size={token_partition_size}."
        )

    required_tokens_per_rank = (
        (local_chunked_prefill_size + token_alignment - 1)
        // token_alignment
        * token_alignment
    )
    if max_tokens_per_rank < required_tokens_per_rank:
        raise ValueError(
            f"{model_label} with MegaMoE requires "
            "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK to "
            "cover each rank's effective prefill token budget. "
            f"Current values: chunked_prefill_size="
            f"{cfg.chunked_prefill_size}, "
            f"token_partition={token_partition_name}, "
            f"token_partition_size={token_partition_size}, "
            f"token_alignment={token_alignment}, "
            f"required_per_rank={required_tokens_per_rank}, "
            "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK="
            f"{max_tokens_per_rank}. Set "
            "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK to at "
            f"least {required_tokens_per_rank}, or lower "
            "--chunked-prefill-size until the effective per-rank budget fits."
        )


def handle_moe_runner_backend_alias(server_args: ServerArgs) -> None:
    cfg = resolving_view(server_args)
    if cfg.moe_runner_backend != "megamoe":
        return

    if cfg.moe_a2a_backend not in ("none", "megamoe"):
        logger.warning(
            "--moe-runner-backend megamoe is an alias for "
            "--moe-a2a-backend megamoe; overriding "
            "--moe-a2a-backend %s.",
            cfg.moe_a2a_backend,
        )
    declare_resolution(
        server_args,
        "handle_moe_runner_backend_alias",
        moe_runner_backend="auto",
        moe_a2a_backend="megamoe",
    )
