import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_policy import (
    CacheAgnosticPolicy,
    CacheAwarePolicy,
    SchedulePolicy,
)
from sglang.srt.managers.scheduler_components.load_inquirer import (
    SchedulerLoadInquirer,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestSchedulePolicyWaitingQueueMatching(CustomTestCase):
    def make_policy(self, policy, supports_fast_match_prefix):
        schedule_policy = object.__new__(SchedulePolicy)
        schedule_policy.policy = policy
        schedule_policy.tree_cache = SimpleNamespace(
            supports_fast_match_prefix=lambda: supports_fast_match_prefix
        )
        return schedule_policy

    def test_cache_agnostic_policy_requires_fast_matching(self):
        policy = self.make_policy(CacheAgnosticPolicy.FCFS, False)
        self.assertFalse(policy.waiting_queue_prefix_matched([]))

        policy.tree_cache = SimpleNamespace(supports_fast_match_prefix=lambda: True)
        self.assertTrue(policy.waiting_queue_prefix_matched([]))

    def test_lpm_queue_limit_respects_fast_matching_capability(self):
        policy = self.make_policy(CacheAwarePolicy.LPM, False)
        self.assertTrue(policy.waiting_queue_prefix_matched([None] * 128))
        self.assertFalse(policy.waiting_queue_prefix_matched([None] * 129))

        policy.tree_cache = SimpleNamespace(supports_fast_match_prefix=lambda: True)
        self.assertTrue(policy.waiting_queue_prefix_matched([None] * 129))


class TestSchedulerLoadInquirer(CustomTestCase):
    def make_inquirer(self, waiting_queue_prefix_matched):
        waiting_req = SimpleNamespace(seqlen=100, num_matched_prefix_tokens=20)
        chunked_req = SimpleNamespace(seqlen=50, prefix_len=10)
        return SimpleNamespace(
            disaggregation_mode=DisaggregationMode.NULL,
            get_waiting_queue=lambda: [waiting_req],
            waiting_queue_prefix_matched=lambda: waiting_queue_prefix_matched,
            get_chunked_req=lambda: chunked_req,
            get_recent_cache_hit_rate=lambda: 0.75,
        )

    def test_waiting_tokens_are_estimated_when_prefix_matching_is_skipped(self):
        inquirer = self.make_inquirer(waiting_queue_prefix_matched=False)

        self.assertEqual(
            SchedulerLoadInquirer.get_num_waiting_uncached_tokens(inquirer),
            65,
        )

    def test_waiting_tokens_use_exact_match_when_prefix_matching_is_done(self):
        inquirer = self.make_inquirer(waiting_queue_prefix_matched=True)

        self.assertEqual(
            SchedulerLoadInquirer.get_num_waiting_uncached_tokens(inquirer),
            120,
        )


class TestContextLengthLoad(CustomTestCase):
    def make_inquirer(
        self,
        *,
        running=(),
        waiting=(),
        prealloc=(),
        transfer=(),
        retracted=(),
        bootstrap=(),
        mode=DisaggregationMode.DECODE,
        method="context_bucket",
    ):
        stats = SimpleNamespace(
            kv_transfer_speed_gb_s=0.0,
            kv_transfer_latency_ms=0.0,
            num_grammar_queue_reqs=0,
            num_paused_reqs=0,
            num_retracted_reqs=0,
            gen_throughput=0.0,
            cache_hit_rate=0.0,
            utilization=0.0,
        )
        return SchedulerLoadInquirer(
            disaggregation_mode=mode,
            server_args=SimpleNamespace(load_balance_method=method),
            max_total_num_tokens=1_000_000,
            max_running_requests=128,
            pool_stats_observer=SimpleNamespace(
                get_pool_stats=lambda: SimpleNamespace(
                    get_kv_token_stats=lambda: (100, 0.001)
                )
            ),
            tp_worker=SimpleNamespace(),
            token_to_kv_pool_allocator=SimpleNamespace(),
            spec_algorithm=SimpleNamespace(is_none=lambda: True),
            get_running_batch=lambda: SimpleNamespace(reqs=running),
            get_waiting_queue=lambda: waiting,
            waiting_queue_prefix_matched=lambda: False,
            get_recent_cache_hit_rate=lambda: 0.0,
            get_stats=lambda: stats,
            get_chunked_req=lambda: None,
            get_disagg_prefill_bootstrap_queue=lambda: SimpleNamespace(queue=bootstrap),
            get_disagg_prefill_inflight_queue=lambda: [],
            get_disagg_decode_prealloc_queue=lambda: SimpleNamespace(
                queue=prealloc,
                retracted_queue=retracted,
            ),
            get_disagg_decode_transfer_queue=lambda: SimpleNamespace(queue=transfer),
            get_spec_total_num_accept_tokens=lambda: 0,
            get_spec_total_num_forward_ct=lambda: 0,
            get_total_prefill_uncached_tokens=lambda: 0,
            get_total_prefill_busy_us=lambda: 0,
            get_decode_moment_totals=lambda: [0] * 6,
        )

    @staticmethod
    def req(length):
        return SimpleNamespace(seqlen=length, waiting_for_input=False)

    def get_loads(self, inquirer):
        with (
            patch(
                "sglang.srt.managers.scheduler_components.load_inquirer.get_lora",
                return_value=SimpleNamespace(enable_lora=False),
            ),
            patch(
                "sglang.srt.managers.scheduler_components.load_inquirer.get_parallel",
                return_value=SimpleNamespace(
                    load_balance_method=inquirer.server_args.load_balance_method,
                    dp_rank=0,
                ),
            ),
        ):
            return inquirer.get_loads()

    def test_decode_counts_each_lifecycle_queue_once(self):
        lengths = [8192, 8193, 16385, 32769, 65537]
        queues = ("running", "waiting", "prealloc", "transfer", "retracted")
        inquirer = self.make_inquirer(
            **{queue: [self.req(length)] for queue, length in zip(queues, lengths)}
        )
        load = self.get_loads(inquirer)
        self.assertEqual(load.context_length_histogram, [1, 1, 1, 1, 1, 0, 0, 0, 0])
        self.assertEqual(load.num_context_tokens, sum(lengths))
        self.assertEqual(sum(load.context_length_histogram), len(queues))
        self.assertEqual(load.num_running_reqs + load.num_waiting_reqs, len(queues))
        # Context load counts full lengths, independent of allocator usage.
        self.assertNotEqual(load.num_context_tokens, load.num_total_tokens)

    def test_current_generated_length_moves_request_to_next_bucket(self):
        class GrowingRequest:
            origin_input_ids = range(8192)
            output_ids = []

            @property
            def seqlen(self):
                return len(self.origin_input_ids) + len(self.output_ids)

        req = GrowingRequest()
        inquirer = self.make_inquirer(running=[req])
        load = self.get_loads(inquirer)
        self.assertEqual(load.context_length_histogram[:2], [1, 0])
        req.output_ids = [1]
        load = self.get_loads(inquirer)
        self.assertEqual(load.context_length_histogram[:2], [0, 1])
        self.assertEqual(load.num_context_tokens, 8193)

    def test_empty_context_bucket_load_is_available(self):
        load = self.get_loads(self.make_inquirer())
        self.assertEqual(load.context_length_histogram, [0] * 9)
        self.assertEqual(load.num_context_tokens, 0)

    def test_default_policies_do_not_read_running_sequence_lengths(self):
        class UnreadableRequest:
            @property
            def seqlen(self):
                raise AssertionError("default balancing must not scan running lengths")

        for method in ("round_robin", "total_requests", "total_tokens"):
            with self.subTest(method=method):
                load = self.get_loads(
                    self.make_inquirer(running=[UnreadableRequest()], method=method)
                )
                self.assertIsNone(load.context_length_histogram)
                self.assertEqual(load.num_context_tokens, 0)

    def test_prefill_bootstrap_and_null_waiting_are_included(self):
        for mode in (DisaggregationMode.PREFILL, DisaggregationMode.NULL):
            with self.subTest(mode=mode):
                load = self.get_loads(
                    self.make_inquirer(
                        running=[self.req(100)],
                        waiting=[self.req(200)],
                        bootstrap=[self.req(300)],
                        mode=mode,
                    )
                )
                expected = 3 if mode == DisaggregationMode.PREFILL else 2
                self.assertEqual(load.context_length_histogram, [expected] + [0] * 8)
                self.assertEqual(load.num_context_tokens, 600 if expected == 3 else 300)


if __name__ == "__main__":
    unittest.main()
