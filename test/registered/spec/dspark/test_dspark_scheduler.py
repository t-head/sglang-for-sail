import functools
import types
import unittest
from contextlib import nullcontext
from unittest.mock import patch

import torch

from sglang.kernels.ops.speculative.dspark.dspark_schedule import (
    schedule_verify_lens_topk_from_survival,
)
from sglang.srt.speculative.dspark_components.dspark_planner import (
    DSparkScheduleConfig,
    HostConfidenceBudgetPlanner,
    VerifyBudgetDecision,
    choose_shared_verify_graph,
    compute_verify_token_budget,
    graph_budget_utilities,
    graph_tier_fill_budget,
)
from sglang.srt.speculative.dspark_components.dspark_sps import (
    SpsAdditiveCostTable,
    SpsCostTable,
)
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


def _flat_table(
    steps_per_sec: float = 1.0, max_batch_tokens: int = 4096
) -> SpsCostTable:
    return SpsCostTable(
        sample_batch_tokens=[1],
        sample_steps_per_sec=[steps_per_sec],
        max_batch_tokens=max_batch_tokens,
    )


def _cliff_table() -> SpsCostTable:
    return SpsCostTable(
        sample_batch_tokens=[1, 2, 3, 4, 5, 6, 7, 8],
        sample_steps_per_sec=[1.0, 1.0, 1.0, 0.5, 0.45, 0.44, 0.43, 0.42],
        max_batch_tokens=64,
    )


def _additive_table() -> SpsAdditiveCostTable:
    return SpsAdditiveCostTable(
        bias_seconds=0.01,
        bs_probes=[1, 100],
        alpha_seconds=[0.0, 0.05],
        m_probes=[1, 200],
        theta_seconds=[0.0, 0.02],
    )


class TestSharedGraphBudget(CustomTestCase):
    @staticmethod
    def _table():
        return SpsAdditiveCostTable(
            bias_seconds=0.04,
            bs_probes=[8],
            alpha_seconds=[0],
            m_probes=[48, 96],
            theta_seconds=[0, 0.04],
            m_axis="graph_tokens",
        )

    def test_uses_per_graph_capacity_and_joint_utility(self):
        budgets, tokens = graph_budget_utilities(
            survival=torch.full((12, 5), 0.5),
            grid=[48, 96],
            bs=12,
            cfg=DSparkScheduleConfig(gamma=5),
            verify_num_draft_tokens=6,
        )
        self.assertEqual(budgets, [36, 60])
        self.assertEqual(tokens, [30.0, 42.0])
        args = dict(
            grid=[48, 96],
            max_graph_tokens=96,
            max_bs=8,
            slots_for_graph=lambda rows: rows,
            table=self._table(),
        )
        self.assertEqual(
            choose_shared_verify_graph(**args, utilities=[[10, 11], [0, 0]])[0],
            0,
        )
        self.assertEqual(
            choose_shared_verify_graph(**args, utilities=[[10, 11], [5, 22], [0, 0]])[
                0
            ],
            1,
        )
        args["slots_for_graph"] = lambda rows: rows // 12
        self.assertEqual(
            choose_shared_verify_graph(**args, utilities=[[10, 11], [0, 0]])[0],
            1,
        )

    def test_busy_and_idle_agree_and_new_peer_falls_back(self):
        from sglang.srt.speculative.dspark_components import dspark_planner as mod

        planner = mod.DSparkVerifyPlanner.__new__(mod.DSparkVerifyPlanner)
        planner._graph_cost_table = self._table()
        planner._schedule_cfg = DSparkScheduleConfig(gamma=5)
        planner.verify_num_draft_tokens = 6
        runner = _fake_model_runner([48, 96], max_bs=16)
        graph_runner = runner.decode_cuda_graph_runner
        graph_runner.attn_backend = types.SimpleNamespace(
            supports_ragged_verify_graph=True
        )
        graph_runner.require_mlp_sync = False
        graph_runner._ragged_capture_slots = lambda rows: min(rows, 16)
        runner.req_to_token_pool = types.SimpleNamespace(
            req_generation=torch.ones(16, dtype=torch.int64)
        )
        planner.model_runner = runner
        host = HostConfidenceBudgetPlanner(
            sps_table=planner._graph_cost_table,
            cfg=planner._schedule_cfg,
            model_runner=runner,
            relay_lag_steps=1,
        )
        planner._budget_planner = host
        group = object()
        batch = types.SimpleNamespace(
            batch_size=lambda: 12,
            is_extend_in_batch=False,
            forward_mode=types.SimpleNamespace(
                is_idle=lambda: False, is_decode=lambda: True
            ),
            spec_info=types.SimpleNamespace(verify_token_budget=None),
            req_pool_indices_cpu=torch.arange(12),
        )
        idle = types.SimpleNamespace(
            batch_size=lambda: 0,
            is_extend_in_batch=False,
            spec_info=None,
            forward_mode=types.SimpleNamespace(is_idle=lambda: True),
        )
        resolved = types.SimpleNamespace(
            confidence=torch.full((12, 5), 0.8),
            generation=torch.ones(12, dtype=torch.int64),
        )
        future_map = types.SimpleNamespace(resolve_confidence_cpu=lambda _: resolved)
        busy_row = None
        peer_row = [0.0, 0.0, 0.0, 0.0, 0.0]
        gather_count = 0

        def gather(output, local, *, group):
            nonlocal gather_count, busy_row
            gather_count += 1
            if local[0]:
                busy_row = local.clone()
                peer = local.new_tensor(peer_row)
                output.copy_(torch.cat([local, peer]))
            else:
                output.copy_(torch.cat([busy_row, local]))

        with (
            patch.object(
                mod,
                "get_parallel",
                return_value=types.SimpleNamespace(
                    tp_group=types.SimpleNamespace(cpu_group=group)
                ),
            ),
            patch.object(torch.distributed, "get_world_size", return_value=2),
            patch.object(mod, "all_gather_single", side_effect=gather),
        ):
            planner.prepare_verify_budget(batch, future_map)
            planner.prepare_verify_budget(idle, future_map)
            self.assertEqual(batch.global_spec_verify_tier_num_tokens, [48])
            self.assertEqual(idle.global_spec_verify_tier_num_tokens, [48])
            self.assertEqual(batch.spec_info.verify_token_budget, 36)
            self.assertEqual(idle.global_spec_verify_num_reqs, 12)
            peer_row = [8, -1, 0, 0, 0]
            planner.prepare_verify_budget(batch, future_map)
            self.assertIsNone(batch.spec_info.verify_token_budget)
            self.assertIsNone(batch.global_spec_verify_tier_num_tokens)
            self.assertEqual(batch.global_spec_verify_num_reqs, 12)
            peer_row = [0, 0, 0, 0, 0]
            host.forced_budget_frac = 1.0
            planner.prepare_verify_budget(batch, future_map)
            self.assertEqual(batch.spec_info.verify_token_budget, 60)
            self.assertEqual(batch.global_spec_verify_tier_num_tokens, [72, 0])
            batch.is_extend_in_batch = True
            planner.prepare_verify_budget(batch, future_map)
            self.assertIsNone(batch.global_spec_verify_tier_num_tokens)
            self.assertEqual(gather_count, 4)


def _survival_from_confidence(confidence: torch.Tensor) -> torch.Tensor:
    return torch.cumprod(confidence, dim=1)


def _bruteforce_budget(
    *,
    history_survival_probs: torch.Tensor,
    sps_table: SpsCostTable,
    cfg: DSparkScheduleConfig,
) -> int:
    num_requests = history_survival_probs.shape[0]
    max_len = cfg.resolved_max_verify_len()
    candidates = history_survival_probs[:, :max_len].flatten()
    candidates = [float(x) for x in candidates.tolist() if float(x) >= cfg.survival_eps]
    candidates.sort(reverse=True)

    best_extra, best_theta = 0, float("-inf")
    for extra in range(len(candidates) + 1):
        tau_star = num_requests + sum(candidates[:extra])
        theta = tau_star * sps_table.lookup(num_requests + extra)
        if theta > best_theta:
            best_theta, best_extra = theta, extra
    return best_extra


def schedule_verify_lens_topk_vanilla(
    *,
    survival_probs: torch.Tensor,
    budget: int,
    cfg: DSparkScheduleConfig,
) -> torch.Tensor:
    cfg.validate()
    num_requests, _gamma = survival_probs.shape
    max_len = cfg.resolved_max_verify_len()
    device = survival_probs.device

    valid_rows = (survival_probs >= cfg.survival_eps).tolist()
    survival_rows = survival_probs.to(torch.float64).tolist()

    candidates: list[tuple[float, int, int]] = []
    for request in range(num_requests):
        for position in range(min(max_len, len(valid_rows[request]))):
            if valid_rows[request][position]:
                candidates.append((survival_rows[request][position], position, request))

    candidates.sort(key=lambda candidate: (-candidate[0], candidate[1], candidate[2]))

    selected_extra = [0] * num_requests
    for _survival, _position, request in candidates[: max(int(budget), 0)]:
        selected_extra[request] += 1

    lower_bound = max(cfg.min_verify_len, 1)
    verify_lens = [
        min(max(cfg.min_verify_len + extra, lower_bound), max_len)
        for extra in selected_extra
    ]
    return torch.tensor(verify_lens, dtype=torch.int32, device=device)


_TOPK_IMPLS = (
    schedule_verify_lens_topk_from_survival,
    schedule_verify_lens_topk_vanilla,
)


def _for_each_impl(test_method):
    @functools.wraps(test_method)
    def wrapper(self):
        for impl in _TOPK_IMPLS:
            with self.subTest(impl=impl.__name__):
                test_method(self, impl)

    return wrapper


class TestComputeVerifyTokenBudget(CustomTestCase):
    def test_budget_argmax_matches_bruteforce_scan_across_sps_cliffs(self):
        torch.manual_seed(1)
        cfg = DSparkScheduleConfig(gamma=7)
        table = _cliff_table()
        for _ in range(20):
            confidence = torch.rand(3, 7, dtype=torch.float32) * 0.5 + 0.45
            survival = _survival_from_confidence(confidence)
            expected = _bruteforce_budget(
                history_survival_probs=survival, sps_table=table, cfg=cfg
            )
            actual = compute_verify_token_budget(
                history_survival_probs=survival, sps_table=table, cfg=cfg
            ).budget
            self.assertEqual(actual, expected)

    def test_budget_pool_is_independent_of_min_verify_len(self):
        survival = torch.tensor([[0.9, 0.8, 0.7, 0.6]], dtype=torch.float32)
        cfg_no_min = DSparkScheduleConfig(gamma=4, min_verify_len=0)
        cfg_min2 = DSparkScheduleConfig(gamma=4, min_verify_len=2)
        table = _flat_table()
        budget_no_min = compute_verify_token_budget(
            history_survival_probs=survival, sps_table=table, cfg=cfg_no_min
        ).budget
        budget_min2 = compute_verify_token_budget(
            history_survival_probs=survival, sps_table=table, cfg=cfg_min2
        ).budget
        # The budget candidate pool spans all positions; the per-request floor
        # is enforced later at verify-lens scheduling, not at budget time.
        self.assertEqual(budget_no_min, 4)
        self.assertEqual(budget_min2, 4)

    def test_budget_drops_candidates_below_survival_eps(self):
        survival = torch.tensor([[0.9, 1e-9, 1e-12]], dtype=torch.float32)
        cfg = DSparkScheduleConfig(gamma=3, survival_eps=1e-6)
        table = _flat_table()
        budget = compute_verify_token_budget(
            history_survival_probs=survival, sps_table=table, cfg=cfg
        ).budget
        self.assertLessEqual(budget, 1)

    def test_decision_predicted_step_matches_additive_table_at_budget(self):
        survival = torch.tensor([[0.9, 0.8, 0.7, 0.6]], dtype=torch.float32)
        cfg = DSparkScheduleConfig(gamma=4)
        table = _additive_table()
        decision = compute_verify_token_budget(
            history_survival_probs=survival, sps_table=table, cfg=cfg
        )
        # Reference via the independent scalar interpolation path (step_time),
        # not the tensor helper the implementation itself uses.
        self.assertAlmostEqual(
            decision.predicted_step_seconds,
            table.step_time(num_reqs=1, budget=int(decision.budget)),
            places=5,
        )
        self.assertGreater(decision.predicted_theta, 0.0)


def _make_budget_planner() -> HostConfidenceBudgetPlanner:
    return HostConfidenceBudgetPlanner(
        sps_table=_flat_table(),
        cfg=DSparkScheduleConfig(gamma=4),
        model_runner=None,
    )


class TestBudgetDecisionLifecycle(CustomTestCase):
    def test_note_non_decode_step_clears_decision(self):
        planner = _make_budget_planner()
        planner.last_decision = VerifyBudgetDecision(budget=1)
        planner.note_non_decode_step()
        self.assertIsNone(planner.take_last_decision())


class TestScheduleVerifyLensTopk(CustomTestCase):
    @_for_each_impl
    def test_higher_confidence_admitted_first(self, impl):
        survival = torch.tensor(
            [[0.99, 0.98, 0.97, 0.96], [0.40, 0.30, 0.20, 0.10]],
            dtype=torch.float32,
        )
        cfg = DSparkScheduleConfig(gamma=4)
        verify_lens = impl(survival_probs=survival, budget=2, cfg=cfg)
        extra = verify_lens.to(torch.int64) - cfg.min_verify_len
        self.assertEqual(int(extra[0].item()), 2)
        self.assertEqual(int(extra[1].item()), 0)

    @_for_each_impl
    def test_min_and_max_enter_the_budget(self, impl):
        survival = torch.tensor([[0.99, 0.99, 0.99, 0.99, 0.99]], dtype=torch.float32)
        cfg = DSparkScheduleConfig(gamma=5, min_verify_len=1, max_verify_len=3)
        verify_lens = impl(survival_probs=survival, budget=100, cfg=cfg)
        self.assertGreaterEqual(int(verify_lens.min().item()), 1)
        self.assertLessEqual(int(verify_lens.max().item()), 3)

    @_for_each_impl
    def test_large_budget_selects_all_candidates(self, impl):
        survival = torch.tensor([[0.9, 0.8, 0.7]], dtype=torch.float32)
        cfg = DSparkScheduleConfig(gamma=3)
        verify_lens = impl(survival_probs=survival, budget=1000, cfg=cfg)
        # anchor (min_verify_len=1) + all 3 admitted drafts
        self.assertEqual(int(verify_lens[0].item()), 4)

    @_for_each_impl
    def test_tie_break_is_value_independent(self, impl):
        survival = torch.tensor([[0.8, 0.8, 0.8], [0.8, 0.8, 0.8]], dtype=torch.float32)
        cfg = DSparkScheduleConfig(gamma=3)
        floor = max(cfg.min_verify_len, 1)
        verify_lens = impl(survival_probs=survival, budget=3, cfg=cfg)
        total_extra = int((verify_lens.to(torch.int64) - floor).sum().item())
        self.assertEqual(total_extra, 3)


class TestVerifyLenAnchorContract(CustomTestCase):
    @_for_each_impl
    def test_explicit_zero_min_still_clamped_to_anchor(self, impl):
        survival = _survival_from_confidence(
            torch.tensor([[0.9, 0.8, 0.7], [0.6, 0.5, 0.4]], dtype=torch.float32)
        )
        cfg = DSparkScheduleConfig(gamma=3, min_verify_len=0)
        verify_lens = impl(survival_probs=survival, budget=0, cfg=cfg)
        self.assertGreaterEqual(int(verify_lens.min().item()), 1)
        self.assertTrue(
            torch.equal(verify_lens, torch.tensor([1, 1], dtype=torch.int32))
        )

    def test_non_flat_table_small_budget_feeds_ragged_layout(self):
        table = SpsCostTable(
            sample_batch_tokens=[2, 3],
            sample_steps_per_sec=[1.0, 0.1],
            max_batch_tokens=64,
        )
        cfg = DSparkScheduleConfig(gamma=3)
        survival = _survival_from_confidence(
            torch.tensor([[0.90, 0.80, 0.70], [0.85, 0.60, 0.40]], dtype=torch.float32)
        )
        budget = compute_verify_token_budget(
            history_survival_probs=survival, sps_table=table, cfg=cfg
        ).budget
        self.assertEqual(budget, 0)
        verify_lens = schedule_verify_lens_topk_from_survival(
            survival_probs=survival, budget=budget, cfg=cfg
        )
        self.assertGreaterEqual(int(verify_lens.min().item()), 1)
        verify_lens_cpu = verify_lens.to(torch.int64).tolist()
        layout = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=verify_lens_cpu,
            device=torch.device("cpu"),
            grid=[sum(verify_lens_cpu)],
        )
        self.assertEqual(layout.verify_lens_cpu, verify_lens_cpu)


class TestNonAnticipating(CustomTestCase):
    @_for_each_impl
    def test_lens_topk_non_anticipating_under_future_perturbation(self, impl):
        base = torch.tensor(
            [
                [0.95, 0.90, 0.80, 0.40],
                [0.92, 0.70, 0.30, 0.10],
                [0.99, 0.98, 0.50, 0.05],
            ],
            dtype=torch.float32,
        )
        cfg = DSparkScheduleConfig(gamma=4)
        budget = 5
        baseline = impl(survival_probs=base, budget=budget, cfg=cfg)

        request, cut = 1, 2
        for delta in (-0.05, -0.2, 0.05, 0.0):
            perturbed = base.clone()
            future = perturbed[request, cut:]
            perturbed[request, cut:] = torch.clamp(
                torch.minimum(future + delta, base[request, cut - 1]), min=0.0
            )
            verify_lens = impl(survival_probs=perturbed, budget=budget, cfg=cfg)
            admitted_prefix_unchanged = min(int(baseline[request].item()), cut) == min(
                int(verify_lens[request].item()), cut
            )
            self.assertTrue(
                admitted_prefix_unchanged,
                msg=f"prefix admission changed under future perturbation delta={delta}",
            )

    @_for_each_impl
    def test_other_requests_unaffected_by_one_request_future(self, impl):
        base = torch.tensor(
            [[0.95, 0.90, 0.20], [0.93, 0.88, 0.15]], dtype=torch.float32
        )
        cfg = DSparkScheduleConfig(gamma=3)
        budget = 2
        baseline = impl(survival_probs=base, budget=budget, cfg=cfg)
        perturbed = base.clone()
        perturbed[0, 2] = 0.01
        verify_lens = impl(survival_probs=perturbed, budget=budget, cfg=cfg)
        self.assertEqual(int(baseline[1].item()), int(verify_lens[1].item()))


class TestVanillaMatchesReference(CustomTestCase):
    def test_random_inputs_match_reference(self):
        torch.manual_seed(20260630)
        num_trials = 4000
        for trial in range(num_trials):
            num_requests = int(torch.randint(1, 6, ()).item())
            gamma = int(torch.randint(1, 9, ()).item())

            dtype = torch.float32 if trial % 2 == 0 else torch.float64
            confidence = torch.rand(num_requests, gamma, dtype=dtype)
            if trial % 3 == 0:
                confidence = (confidence * 4).round() / 4
            if trial % 7 == 0:
                confidence = torch.ones(num_requests, gamma, dtype=dtype)
            survival = torch.cumprod(confidence, dim=1)

            min_verify_len = int(torch.randint(0, gamma + 1, ()).item())
            if torch.rand(()).item() < 0.5:
                max_verify_len = 0
            else:
                max_verify_len = int(
                    torch.randint(min_verify_len, gamma + 1, ()).item()
                )
            survival_eps = float(
                [1e-6, 1e-3, 0.1, 0.5][int(torch.randint(0, 4, ()).item())]
            )
            budget = int(torch.randint(0, num_requests * gamma + 3, ()).item())

            cfg = DSparkScheduleConfig(
                gamma=gamma,
                min_verify_len=min_verify_len,
                max_verify_len=max_verify_len,
                survival_eps=survival_eps,
            )
            reference = schedule_verify_lens_topk_from_survival(
                survival_probs=survival, budget=budget, cfg=cfg
            )
            vanilla = schedule_verify_lens_topk_vanilla(
                survival_probs=survival, budget=budget, cfg=cfg
            )
            self.assertTrue(
                torch.equal(reference, vanilla),
                msg=(
                    f"mismatch on trial {trial}: budget={budget} "
                    f"min={min_verify_len} max={max_verify_len} eps={survival_eps} "
                    f"survival={survival.tolist()} "
                    f"reference={reference.tolist()} vanilla={vanilla.tolist()}"
                ),
            )


class TestGraphTierFillBudget(CustomTestCase):
    def test_feeding_budget_fills_topk_total_to_tier(self):
        """Feeding the fill budget to the top-k lifts the total to min(tier, bs*K)."""
        bs = 6
        cfg = DSparkScheduleConfig(gamma=6, min_verify_len=1)
        cap = cfg.resolved_max_verify_len()
        survival = torch.full((bs, cap), 0.99, dtype=torch.float32)
        for graph_num_tokens in (bs, bs * cap // 2, bs * cap, bs * cap + 12):
            budget = graph_tier_fill_budget(
                graph_num_tokens=graph_num_tokens,
                bs=bs,
                verify_num_draft_tokens=cap,
                min_verify_len=cfg.min_verify_len,
            )
            verify_lens = schedule_verify_lens_topk_from_survival(
                survival_probs=survival, budget=budget, cfg=cfg
            )
            total = int(verify_lens.to(torch.int64).sum().item())
            self.assertEqual(total, min(graph_num_tokens, bs * cap))


class _FakeRaggedRunner(types.SimpleNamespace):
    pass


def _fake_model_runner(capture_num_tokens, max_bs):
    runner = _FakeRaggedRunner(
        ragged_verify_mode=True,
        capture_num_tokens=capture_num_tokens,
        max_bs=max_bs,
    )
    return types.SimpleNamespace(decode_cuda_graph_runner=runner)


class TestBudgetTierSelection(CustomTestCase):
    def test_pd_preallocated_row_does_not_publish_confidence_for_reused_slot(self):
        from sglang.srt.managers import overlap_utils

        pool = types.SimpleNamespace(req_generation=torch.tensor([0, 1, 1]))
        with patch.object(overlap_utils, "_is_cuda", False):
            relay = overlap_utils.ConfidenceRelay(torch.device("cpu"), 3, pool)
            slots = torch.tensor([1, 2])
            relay.scatter(
                slots, torch.tensor([[0.125, 0.25], [0.5, 0.5]]), indices_cpu=slots
            )

        # KV preallocation reuses slot 1. Only the incumbent in slot 2
        # generates another confidence sample before the newcomer is admitted.
        pool.req_generation[1] += 1
        incumbent = torch.tensor([2])
        relay.scatter(incumbent, torch.tensor([[0.75, 0.5]]), indices_cpu=incumbent)
        depth = overlap_utils.CONFIDENCE_RELAY_RING_DEPTH
        relay.conf_ring = torch.empty((depth, 3, 2))
        relay.gen_ring = torch.zeros((depth, 3), dtype=torch.int64)
        ready = types.SimpleNamespace(query=lambda: True, record=lambda: None)
        relay.copy_done = [ready] * depth
        stream = types.SimpleNamespace(wait_event=lambda event: None)
        batch = types.SimpleNamespace(
            spec_info=types.SimpleNamespace(future_indices=slots),
            req_pool_indices=slots,
            req_pool_indices_cpu=slots,
        )

        def snapshot():
            with patch.object(
                torch,
                "get_device_module",
                return_value=types.SimpleNamespace(stream=lambda _: nullcontext()),
            ):
                for _ in range(overlap_utils.CONFIDENCE_RELAY_RING_LAG):
                    relay.issue_ring_copy(stream=stream, publish_ready=ready)
            return relay.resolve(batch, stream=stream, publish_ready=ready)

        def survival(resolved):
            planner = HostConfidenceBudgetPlanner(
                sps_table=_cliff_table(),
                cfg=DSparkScheduleConfig(gamma=2),
                model_runner=None,
            )
            return planner._two_steps_prior_survival(
                lagged_confidence=resolved.confidence,
                lagged_generation=resolved.generation,
                current_generation=pool.req_generation[slots],
            )

        # The pending row gets the no-confidence prior, not its old value.
        torch.testing.assert_close(
            survival(snapshot()), torch.tensor([[1.0, 1.0], [0.75, 0.375]])
        )
        new_slot = torch.tensor([1])
        relay.scatter(new_slot, torch.tensor([[0.5, 0.25]]), indices_cpu=new_slot)
        torch.testing.assert_close(
            survival(snapshot()), torch.tensor([[0.5, 0.125], [0.75, 0.375]])
        )

    def test_megamoe_tier_exchange_carries_request_counts_for_idle_rank(self):
        from sglang.srt.speculative.dspark_components import dspark_planner

        planner = dspark_planner.DSparkVerifyPlanner.__new__(
            dspark_planner.DSparkVerifyPlanner
        )
        planner._dp_tier_gather_enabled = True
        planner._ppu_megamoe_compact = True
        batch = types.SimpleNamespace(
            batch_size=lambda: 0, is_extend_in_batch=False, global_num_tokens=[0]
        )
        cpu_group = object()
        rows = torch.tensor([601, 256, 399, 128], dtype=torch.int64)

        def gather(output, local, *, group):
            output.copy_(torch.cat([local, rows]))

        with (
            patch.object(
                dspark_planner,
                "get_parallel",
                return_value=types.SimpleNamespace(
                    tp_group=types.SimpleNamespace(cpu_group=cpu_group)
                ),
            ),
            patch.object(torch.distributed, "get_world_size", return_value=3),
            patch.object(dspark_planner, "all_gather_single", side_effect=gather),
        ):
            planner._maybe_gather_dp_verify_tier(batch=batch, local_tier_num_tokens=0)
            self.assertEqual(batch.global_spec_verify_num_reqs, 256)
            self.assertEqual(batch.global_spec_verify_tier_num_tokens, [0, 601, 399])
            self.assertEqual(batch.global_num_tokens, [0])
            tier = dspark_planner.dp_global_verify_tier_num_tokens(
                global_tier_num_tokens=batch.global_spec_verify_tier_num_tokens
            )
            layout = dspark_planner.idle_ragged_layout(
                tier_num_reqs=batch.global_spec_verify_num_reqs,
                dp_tier_num_tokens=tier,
                device=torch.device("cpu"),
                verify_num_draft_tokens=6,
                model_runner=_fake_model_runner([384, 768, 1536], max_bs=256),
            )
            self.assertEqual(layout.graph_num_tokens, 768)

            # A newly admitted peer without confidence pins the common shape,
            # even though another active peer can still produce a small budget.
            rows[0] = -1
            planner._maybe_gather_dp_verify_tier(batch=batch, local_tier_num_tokens=0)
            tier = dspark_planner.dp_global_verify_tier_num_tokens(
                global_tier_num_tokens=batch.global_spec_verify_tier_num_tokens
            )
            self.assertIsNone(tier)
            layout = dspark_planner.idle_ragged_layout(
                tier_num_reqs=batch.global_spec_verify_num_reqs,
                dp_tier_num_tokens=tier,
                device=torch.device("cpu"),
                verify_num_draft_tokens=6,
                model_runner=_fake_model_runner([384, 768, 1536], max_bs=256),
            )
            self.assertEqual(layout.graph_num_tokens, 1536)

            # A global extend step must not reuse a preceding decode tier.
            batch.is_extend_in_batch = True
            planner._maybe_gather_dp_verify_tier(batch=batch, local_tier_num_tokens=0)
            self.assertIsNone(batch.global_spec_verify_num_reqs)
            self.assertIsNone(batch.global_spec_verify_tier_num_tokens)

    def test_floor_uses_tier_hint_capped_at_uniform_window(self):
        from sglang.srt.speculative.dspark_components.dspark_planner import (
            verify_layout_graph_num_tokens_floor,
        )
        from sglang.srt.speculative.ragged_verify import RaggedVerifyMode

        model_runner = _fake_model_runner([8, 16, 1024], max_bs=128)
        floor = verify_layout_graph_num_tokens_floor(
            num_reqs=100,
            ragged_verify_mode=RaggedVerifyMode.COMPACT,
            verify_num_draft_tokens=8,
            model_runner=model_runner,
            tier_num_tokens=150,
        )
        self.assertEqual(floor, 150)
        capped = verify_layout_graph_num_tokens_floor(
            num_reqs=10,
            ragged_verify_mode=RaggedVerifyMode.COMPACT,
            verify_num_draft_tokens=8,
            model_runner=model_runner,
            tier_num_tokens=150,
        )
        self.assertEqual(capped, 80)
        pinned = verify_layout_graph_num_tokens_floor(
            num_reqs=100,
            ragged_verify_mode=RaggedVerifyMode.COMPACT,
            verify_num_draft_tokens=8,
            model_runner=model_runner,
        )
        self.assertEqual(pinned, 800)

    def test_exceeds_gate_checks_slots_and_tier(self):
        from sglang.srt.speculative.dspark_components.dspark_planner import (
            ragged_layout_exceeds_captured_grid,
        )

        model_runner = _fake_model_runner([8, 16, 1024], max_bs=128)
        self.assertTrue(
            ragged_layout_exceeds_captured_grid(
                num_reqs=129,
                verify_num_draft_tokens=8,
                model_runner=model_runner,
                tier_tokens_hint=200,
            )
        )
        self.assertFalse(
            ragged_layout_exceeds_captured_grid(
                num_reqs=128,
                verify_num_draft_tokens=8,
                model_runner=model_runner,
                tier_tokens_hint=512,
            )
        )
        self.assertFalse(
            ragged_layout_exceeds_captured_grid(
                num_reqs=128,
                verify_num_draft_tokens=8,
                model_runner=model_runner,
            )
        )
        self.assertTrue(
            ragged_layout_exceeds_captured_grid(
                num_reqs=128,
                verify_num_draft_tokens=9,
                model_runner=model_runner,
            )
        )


if __name__ == "__main__":
    unittest.main()
