"""Unit tests for allocation_sizing.py -- CPU only, no GPU required.

Covers the shared max_running_requests estimation and the req_to_token pool
byte deduction used by KVCacheConfigurator._resolve_memory_pool_config,
including the one-iteration fixed-point invariant (the deduction computed from
a provisional token capacity must always cover the req pool sized from the
final, smaller token capacity).
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.mem_cache.kv_cache_configurator import (
    KVCacheConfigurator,
    estimate_max_running_requests,
    estimate_req_to_token_pool_bytes,
    get_req_to_token_pool_num_slots,
)
from sglang.srt.runtime_context import get_context, get_disagg, get_memory, get_schedule
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_server_args(
    *,
    max_running_requests=None,
    disaggregation_mode="null",
    disaggregation_decode_extra_slots=None,
    speculative_algorithm=None,
    max_speculative_num_draft_tokens=None,
    page_size=1,
):
    return SimpleNamespace(
        max_running_requests=max_running_requests,
        disaggregation_mode=disaggregation_mode,
        disaggregation_decode_extra_slots=disaggregation_decode_extra_slots,
        speculative_algorithm=speculative_algorithm,
        max_speculative_num_draft_tokens=max_speculative_num_draft_tokens,
        page_size=page_size,
    )


@patch(
    "sglang.srt.mem_cache.kv_cache_configurator.resolving_view", new=lambda args: args
)
class TestEstimateMaxRunningRequests(CustomTestCase):
    def test_long_context_hits_2048_floor(self):
        # MiniMax-M3 shape: ~901k tokens over a 1M context -> estimate ~440,
        # clamped up to the 2048 floor.
        args = _make_server_args()
        self.assertEqual(
            estimate_max_running_requests(
                token_capacity=901_120,
                context_len=1_048_576,
                server_args=args,
                attn_dp_size=1,
            )[1],
            2048,
        )

    def test_short_context_hits_4096_ceiling(self):
        args = _make_server_args()
        self.assertEqual(
            estimate_max_running_requests(
                token_capacity=10_000_000,
                context_len=8192,
                server_args=args,
                attn_dp_size=1,
            )[1],
            4096,
        )

    def test_half_token_capacity_clamp(self):
        args = _make_server_args()
        self.assertEqual(
            estimate_max_running_requests(
                token_capacity=1000,
                context_len=8192,
                server_args=args,
                attn_dp_size=1,
            )[1],
            500,
        )

    def test_user_requested_and_dp_split(self):
        args = _make_server_args(max_running_requests=512)
        self.assertEqual(
            estimate_max_running_requests(
                token_capacity=901_120,
                context_len=1_048_576,
                server_args=args,
                attn_dp_size=1,
            )[1],
            512,
        )
        self.assertEqual(
            estimate_max_running_requests(
                token_capacity=901_120,
                context_len=1_048_576,
                server_args=args,
                attn_dp_size=2,
            )[1],
            256,
        )

    def test_user_requested_capped_by_half_token_capacity(self):
        args = _make_server_args(max_running_requests=4096)
        self.assertEqual(
            estimate_max_running_requests(
                token_capacity=1000,
                context_len=8192,
                server_args=args,
                attn_dp_size=1,
            )[1],
            500,
        )

    def test_dp_split_preserves_one_request(self):
        args = _make_server_args(max_running_requests=1)
        self.assertEqual(
            estimate_max_running_requests(
                token_capacity=1000,
                context_len=8192,
                server_args=args,
                attn_dp_size=8,
            ),
            (1, 1),
        )

    def test_mamba_req_cap(self):
        args = _make_server_args()
        self.assertEqual(
            estimate_max_running_requests(
                token_capacity=901_120,
                context_len=1_048_576,
                server_args=args,
                attn_dp_size=1,
                mamba_req_cap=300,
            )[1],
            300,
        )


@patch(
    "sglang.srt.mem_cache.kv_cache_configurator.resolving_view", new=lambda args: args
)
class TestGetReqToTokenPoolNumSlots(CustomTestCase):
    def test_default_padding_row(self):
        args = _make_server_args()
        self.assertEqual(get_req_to_token_pool_num_slots(2048, args), 2049)

    def test_decode_mode_extra_slots(self):
        args = _make_server_args(
            disaggregation_mode="decode", disaggregation_decode_extra_slots=10
        )
        self.assertEqual(get_req_to_token_pool_num_slots(2048, args), 2059)

    def test_decode_mode_unset_extra_slots(self):
        args = _make_server_args(
            disaggregation_mode="decode", disaggregation_decode_extra_slots=None
        )
        self.assertEqual(get_req_to_token_pool_num_slots(2048, args), 2049)


@patch(
    "sglang.srt.mem_cache.kv_cache_configurator.get_req_to_token_extra_context_len",
    new=lambda: 4,
)
@patch(
    "sglang.srt.mem_cache.kv_cache_configurator.resolving_view", new=lambda args: args
)
class TestEstimateReqToTokenPoolBytes(CustomTestCase):
    def test_minimax_m3_shape(self):
        # MiniMax-M3: 1M context, 2048-request floor -> ~8 GiB req_to_token map.
        args = _make_server_args()
        nbytes = estimate_req_to_token_pool_bytes(
            token_capacity=901_120,
            context_len=1_048_576,
            server_args=args,
            attn_dp_size=1,
        )
        # extra_context_len = 4 (no spec decoding) -> 2049 x (1048576 + 4) x 4B
        self.assertEqual(nbytes, 2049 * (1_048_576 + 4) * 4)
        self.assertGreater(nbytes, 8 * (1 << 30))

    def test_one_iteration_fixed_point_never_under_deducts(self):
        # Emulate KVCacheConfigurator._resolve_memory_pool_config:
        #   T0 = budget // cell (upper bound), R = deduction(T0),
        #   T1 = (budget - R) // cell (final).
        # The req pool actually allocated for T1 must fit within R for every
        # budget/cell/context combination, i.e. deduction(T1) <= R.
        args = _make_server_args()
        cell_size = 45_312  # MiniMax-M3 per-token KV bytes (tp16, bf16)
        context_len = 1_048_576
        for budget_gb in (5, 10, 20, 30, 38, 50, 100):
            budget = budget_gb * (1 << 30)
            t0 = budget // cell_size
            deduction = estimate_req_to_token_pool_bytes(
                token_capacity=t0,
                context_len=context_len,
                server_args=args,
                attn_dp_size=1,
            )
            t1 = max(budget - deduction, 0) // cell_size
            actual = estimate_req_to_token_pool_bytes(
                token_capacity=t1,
                context_len=context_len,
                server_args=args,
                attn_dp_size=1,
            )
            self.assertLessEqual(actual, deduction, f"budget={budget_gb} GB")


class TestLiveRequestPoolSizing(CustomTestCase):
    def test_runtime_overrides_drive_both_reservation_and_final_limit(self):
        configurator = KVCacheConfigurator.__new__(KVCacheConfigurator)
        configurator.server_args = _make_server_args(max_running_requests=999)
        configurator.model_config = SimpleNamespace(context_len=1_048_576)
        configurator.attn_dp_size = 2
        configurator.mambaish_config = None
        configurator._unified_sized = None
        pool_factory = SimpleNamespace(
            calculate_pool_sizes=lambda available, page_size: SimpleNamespace(
                max_total_num_tokens=8192
            )
        )
        with (
            get_context().override_server_args(model_path="dummy"),
            patch.object(
                KVCacheConfigurator,
                "_apply_token_constraints",
                lambda self, count: count,
            ),
            get_memory().override(enable_unified_memory=False),
            get_disagg().override(
                disaggregation_mode="decode",
                disaggregation_decode_extra_slots=7,
                enable_pd_role_switch=False,
            ),
            patch(
                "sglang.srt.model_executor.pool_configurator.create_memory_pool_configurator",
                return_value=pool_factory,
            ),
            patch(
                "sglang.srt.mem_cache.kv_cache_configurator.get_req_to_token_extra_context_len",
                return_value=0,
            ),
        ):
            for requests in (64, 128):
                with (
                    self.subTest(requests=requests),
                    get_schedule().override(max_running_requests=requests, page_size=1),
                ):
                    per_worker = requests // 2
                    self.assertEqual(
                        configurator.resolve_max_num_reqs(8192), per_worker
                    )
                    # Includes one padding row and seven decode-transfer rows.
                    self.assertEqual(
                        configurator._estimate_req_to_token_pool_bytes(1 << 30),
                        (per_worker + 1 + 7) * 1_048_576 * 4,
                    )


if __name__ == "__main__":
    unittest.main()
