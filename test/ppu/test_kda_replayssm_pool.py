"""CPU regression for KDA pool selection, including non-Kimi model wrappers."""

from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import pytest

from sglang.srt.mem_cache import kv_cache_configurator as module
from sglang.srt.runtime_context import get_context, get_exec


@pytest.mark.parametrize(
    "enabled,kda,gdn,expected",
    [
        (True, True, False, True),
        (False, True, False, False),
        (True, False, True, True),
        (True, False, False, False),
    ],
)
def test_replayssm_pool_and_budget(enabled, kda, gdn, expected):
    params = Mock(is_kda=kda, layers=[0, 1], mamba_cache_per_req=1024)
    params.replayssm_ring_bytes_per_req.return_value = 2048
    config = module.KVCacheConfigurator.__new__(module.KVCacheConfigurator)
    config.mambaish_config = NS(mamba2_cache_params=params)
    config.hybrid_gdn_config = NS() if gdn else None
    config.hybrid_kda_config = NS() if kda else None
    config.model_config = NS(context_len=4096)
    config.layer_info = NS(start_layer=0, end_layer=2)
    config.device = "cpu"
    config.pp_size = 1
    config.attn_dp_size = 1
    config.extra_mamba_cache_bytes_per_req = 0
    config.extra_mamba_cache_bytes_per_request_row = 0
    config.spec_algorithm = NS(is_none=lambda: False)
    config.allocate_intermediate_ssm = True
    with (
        patch.object(
            module.KVCacheConfigurator, "fused_entry_bytes", return_value=None
        ),
        get_context().override_server_args(
            enable_linear_replayssm=False,
            enable_linear_replayssm_spec=enabled,
            linear_replayssm_cache_len=16,
            max_mamba_cache_size=8,
            max_running_requests=8,
            disable_overlap_schedule=True,
            speculative_algorithm="EAGLE",
            speculative_eagle_topk=1,
            speculative_num_steps=5,
            speculative_num_draft_tokens=6,
            enable_page_major_kv_layout=False,
            disaggregation_mode="null",
        ),
        get_exec().features.override(enable_memory_saver=False),
        get_exec().mamba.override(
            enable_mamba_extra_buffer=False,
            enable_mamba_extra_buffer_lazy=False,
        ),
        patch.object(
            module.KVCacheConfigurator, "_calculate_mamba_ratio", return_value=1
        ),
        patch.object(module, "HybridReqToTokenPool") as pool,
    ):
        config._build_hybrid_req_pool(max_num_reqs=8, extra_max_context_len=8)
        assert pool.call_args.kwargs["enable_linear_replayssm_spec"] is expected
        config._handle_max_mamba_cache(1.0)
    if expected:
        params.replayssm_ring_bytes_per_req.assert_called_once_with(
            record_len=16 if kda else 6
        )
    else:
        params.replayssm_ring_bytes_per_req.assert_not_called()


@pytest.mark.parametrize("mode,role_switch", [("decode", False), ("prefill", True)])
def test_pd_decode_extra_slots_keep_fixed_request_state(mode, role_switch):
    config = module.KVCacheConfigurator.__new__(module.KVCacheConfigurator)
    config.mambaish_config = NS(
        mamba2_cache_params=NS(layers=[0, 1], mamba_cache_per_req=1024)
    )
    config.hybrid_gdn_config = config.hybrid_kda_config = None
    config.pp_size = config.attn_dp_size = 1
    config.extra_mamba_cache_bytes_per_req = 0
    config.extra_mamba_cache_bytes_per_request_row = 128
    config.spec_algorithm = NS(is_none=lambda: True)
    config.allocate_intermediate_ssm = False
    with (
        get_context().override_server_args(
            model_path="dummy",
            max_mamba_cache_size=8,
            max_running_requests=8,
            disaggregation_mode=mode,
            enable_pd_role_switch=role_switch,
            disaggregation_decode_extra_slots=2,
            enable_unified_memory=False,
        ),
        get_exec().mamba.override(enable_mamba_extra_buffer=False),
        patch.object(
            module.KVCacheConfigurator, "fused_entry_bytes", return_value=None
        ),
        patch.object(
            module.KVCacheConfigurator, "_req_pool_pre_alloc_size", return_value=2
        ),
        patch.object(
            module.KVCacheConfigurator, "_calculate_mamba_ratio", return_value=1
        ),
    ):
        actual = config._handle_max_mamba_cache(1.0)
    expected_bytes = 11 * (1024 + 128)
    assert actual == 1.0 - expected_bytes / (1 << 30)
