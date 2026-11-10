"""Context-length-aware DP routing without a server or model."""

import unittest
from types import SimpleNamespace

from sglang.srt.managers.context_bucket import (
    CONTEXT_BUCKET_BOUNDS,
    ContextBucketBudget,
    context_bucket_index,
    estimate_decode_context_length,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _load(rank, lengths, timestamp=1.0, waiting=0, histogram=True):
    counts = [0] * (len(CONTEXT_BUCKET_BOUNDS) + 1)
    for length in lengths:
        counts[context_bucket_index(length)] += 1
    return SimpleNamespace(
        dp_rank=rank,
        timestamp=timestamp,
        num_running_reqs=len(lengths) - waiting,
        num_waiting_reqs=waiting,
        num_context_tokens=sum(lengths),
        context_length_histogram=counts if histogram else None,
    )


class TestContextBuckets(CustomTestCase):
    def test_bucket_boundaries_and_overflow(self):
        self.assertEqual(context_bucket_index(0), 0)
        for index, bound in enumerate(CONTEXT_BUCKET_BOUNDS):
            with self.subTest(bound=bound):
                self.assertEqual(context_bucket_index(bound), index)
                self.assertEqual(context_bucket_index(bound + 1), index + 1)
        self.assertEqual(context_bucket_index(2**31), len(CONTEXT_BUCKET_BOUNDS))

    def test_decode_context_estimate_includes_first_token_and_prefers_embeddings(self):
        for input_embeds, input_ids, expected in (
            (None, range(32768), 32769),
            (range(8192), range(65536), 8193),
            (range(8192), None, 8193),
        ):
            with self.subTest(input_embeds=input_embeds, input_ids=input_ids):
                req = SimpleNamespace(input_embeds=input_embeds, input_ids=input_ids)
                self.assertEqual(estimate_decode_context_length(req), expected)


class TestContextBucketBudget(CustomTestCase):
    def test_equal_tokens_prefers_the_rank_with_fewer_same_bucket_requests(self):
        budget = ContextBucketBudget(num_dp_ranks=2)
        budget.update_budget(
            [
                _load(0, [65536, 49152, 8192, 8192]),
                _load(1, [114688, 8192, 4096, 4096]),
            ]
        )
        self.assertEqual(budget.context_tokens, [131072, 131072])
        self.assertEqual(budget.total_requests, [4, 4])
        self.assertEqual(budget.dispatch(65536, active_ranks=[0, 1]), 1)

    def test_short_requests_fill_a_long_context_rank_missing_the_short_bucket(self):
        budget = ContextBucketBudget(num_dp_ranks=2)
        budget.update_budget([_load(0, [1048576, 65536]), _load(1, [8192, 8192])])
        # Short requests must not all avoid the DP already holding long KV.
        self.assertEqual(budget.dispatch(8192, active_ranks=[0, 1]), 0)
        self.assertEqual(budget.dispatch(65536, active_ranks=[0, 1]), 1)

    def test_each_length_bucket_is_balanced_independently(self):
        budget = ContextBucketBudget(num_dp_ranks=2)
        budget.update_budget(
            [
                _load(0, [131072, 131072, 4096]),
                _load(1, [65536, 8192, 8192]),
            ]
        )
        # Rank 0 has no requests in the 64K bucket despite two longer ones.
        self.assertEqual(budget.dispatch(65536, active_ranks=[0, 1]), 0)

    def test_request_count_guard_rejects_a_worker_more_than_one_ahead(self):
        budget = ContextBucketBudget(num_dp_ranks=2)
        budget.update_budget([_load(0, [4096] * 6), _load(1, [131072] * 4)])
        self.assertEqual(budget.dispatch(65536, active_ranks=[0, 1]), 1)

    def test_request_count_guard_allows_one_request_of_slack(self):
        budget = ContextBucketBudget(num_dp_ranks=2)
        budget.update_budget([_load(0, [4096] * 5), _load(1, [131072] * 4)])
        self.assertEqual(budget.dispatch(65536, active_ranks=[0, 1]), 0)
        self.assertEqual(budget.total_requests, [6, 4])
        # One request of pre-dispatch slack allows two after this placement;
        # the next dispatch cannot widen that gap further.
        self.assertEqual(budget.dispatch(65536, active_ranks=[0, 1]), 1)
        self.assertEqual(budget.total_requests, [6, 5])

    def test_context_token_sum_breaks_equal_bucket_distributions(self):
        budget = ContextBucketBudget(num_dp_ranks=2)
        budget.update_budget([_load(0, [102400]), _load(1, [92160])])
        self.assertEqual(budget.context_histograms[0], budget.context_histograms[1])
        self.assertEqual(budget.dispatch(65536, active_ranks=[0, 1]), 1)

    def test_dispatch_reservations_spread_a_burst_without_new_snapshots(self):
        budget = ContextBucketBudget(num_dp_ranks=4)
        budget.update_budget([_load(rank, []) for rank in range(4)])
        targets = [budget.dispatch(65536, active_ranks=[0, 1, 2, 3]) for _ in range(12)]
        self.assertEqual(targets, [0, 1, 2, 3] * 3)
        self.assertEqual(budget.total_requests, [3] * 4)
        self.assertEqual(budget.context_tokens, [3 * 65536] * 4)
        for histogram in budget.context_histograms:
            self.assertEqual(sum(histogram), 3)
            self.assertEqual(histogram[context_bucket_index(65536)], 3)

    def test_periodic_mixed_lengths_spread_each_bucket_across_eight_ranks(self):
        budget = ContextBucketBudget(num_dp_ranks=8)
        ranks = list(range(8))
        budget.update_budget([_load(rank, []) for rank in ranks])
        for length in [8192, 65536, 1048576, 8192] * 32:
            budget.dispatch(length, active_ranks=ranks)
        for length in (8192, 65536, 1048576):
            with self.subTest(length=length):
                bucket = context_bucket_index(length)
                counts = [histogram[bucket] for histogram in budget.context_histograms]
                self.assertLessEqual(max(counts) - min(counts), 1)
        self.assertLessEqual(max(budget.total_requests) - min(budget.total_requests), 2)

    def test_decode_boundary_lengths_remain_mixed_across_snapshot_refreshes(self):
        # Requests arrive exactly at inclusive bucket boundaries. Decode starts
        # with a first token, so snapshots place them in the next bucket. A
        # prompt-only reservation can segregate short/long requests by rank
        # even while both ranks have identical request counts.
        for arrival_order in ((32768, 65536), (65536, 32768)):
            with self.subTest(arrival_order=arrival_order):
                budget = ContextBucketBudget(num_dp_ranks=2)
                active = [[], []]
                completed = 0
                for step in range(256):
                    for rank in range(2):
                        completed += sum(
                            req.finish_step <= step for req in active[rank]
                        )
                        active[rank] = [
                            req for req in active[rank] if req.finish_step > step
                        ]
                        for req in active[rank]:
                            req.context_length += 1

                    budget.update_budget(
                        [
                            _load(
                                rank,
                                [req.context_length for req in active[rank]],
                                timestamp=step + 1,
                            )
                            for rank in range(2)
                        ]
                    )
                    for prompt_length in arrival_order:
                        req = SimpleNamespace(
                            input_embeds=None, input_ids=range(prompt_length)
                        )
                        rank = budget.dispatch(
                            estimate_decode_context_length(req), active_ranks=[0, 1]
                        )
                        active[rank].append(
                            SimpleNamespace(
                                context_length=prompt_length + 1,
                                finish_step=step + 64,
                            )
                        )

                    actual_loads = [
                        _load(rank, [req.context_length for req in active[rank]])
                        for rank in range(2)
                    ]
                    for left, right in zip(
                        actual_loads[0].context_length_histogram,
                        actual_loads[1].context_length_histogram,
                    ):
                        self.assertLessEqual(abs(left - right), 2, f"step={step}")
                    self.assertLessEqual(
                        abs(len(active[0]) - len(active[1])), 2, f"step={step}"
                    )

                self.assertEqual(completed, 384)
                self.assertEqual(sum(map(len, active)), 128)

    def test_missing_histogram_falls_back_to_request_count_for_all_candidates(self):
        budget = ContextBucketBudget(num_dp_ranks=2)
        budget.update_budget(
            [_load(0, [131072] * 4), _load(1, [8192] * 5, histogram=False)]
        )
        self.assertEqual(budget.dispatch(65536, active_ranks=[0, 1]), 0)

    def test_startup_fallback_rotates_and_keeps_histograms_unknown(self):
        budget = ContextBucketBudget(num_dp_ranks=3)
        targets = [budget.dispatch(8192, active_ranks=[0, 1, 2]) for _ in range(6)]
        self.assertEqual(targets, [0, 1, 2] * 2)
        self.assertEqual(budget.total_requests, [2] * 3)
        self.assertEqual(budget.context_histograms, [None] * 3)

    def test_same_snapshot_does_not_erase_a_reservation(self):
        budget = ContextBucketBudget(num_dp_ranks=1)
        load = _load(0, [8192], waiting=1)
        budget.update_budget([load])
        budget.reserve(0, 65536)
        budget.update_budget([load])
        self.assertEqual(budget.total_requests, [2])
        self.assertEqual(budget.context_tokens, [8192 + 65536])
        self.assertEqual(sum(budget.context_histograms[0]), 2)
        self.assertEqual(sum(load.context_length_histogram), 1)

    def test_fresh_snapshot_replaces_reservations_and_can_clear_old_histogram(self):
        budget = ContextBucketBudget(num_dp_ranks=2)
        budget.update_budget([_load(0, [8192]), _load(1, [65536])])
        budget.reserve(0, 65536)
        budget.update_budget([_load(0, [], timestamp=2.0, histogram=False)])
        self.assertEqual(budget.total_requests, [0, 1])
        self.assertEqual(budget.context_tokens, [0, 65536])
        self.assertIsNone(budget.context_histograms[0])
        self.assertEqual(sum(budget.context_histograms[1]), 1)

    def test_inactive_workers_do_not_affect_candidate_count_floor(self):
        budget = ContextBucketBudget(num_dp_ranks=3)
        budget.update_budget(
            [_load(0, []), _load(1, [131072] * 5), _load(2, [8192] * 5)]
        )
        self.assertEqual(budget.dispatch(65536, active_ranks=[1, 2]), 2)
        self.assertEqual(budget.total_requests[0], 0)

    def test_full_workers_still_route_to_the_existing_scheduler_queue(self):
        budget = ContextBucketBudget(num_dp_ranks=2)
        budget.update_budget([_load(0, [65536] * 128), _load(1, [8192] * 128)])
        self.assertEqual(budget.dispatch(65536, active_ranks=[0, 1]), 1)
        self.assertEqual(budget.total_requests, [128, 129])

    def test_no_active_workers_raises_without_reserving(self):
        budget = ContextBucketBudget(num_dp_ranks=2)
        with self.assertRaises(RuntimeError):
            budget.dispatch(65536, active_ranks=[])
        self.assertEqual(budget.total_requests, [0, 0])


if __name__ == "__main__":
    unittest.main()
