import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from sglang.srt.speculative.dspark_components.dspark_sps import (
    SpsAdditiveCostTable,
    SpsCostTable,
    build_uninitialized_sps_table,
    is_uninitialized_sps_table,
    load_sps_table_from_path,
    profile_sps_table,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


def _make_table() -> SpsCostTable:
    return SpsCostTable(
        sample_batch_tokens=[8, 16, 32, 64],
        sample_steps_per_sec=[1000.0, 950.0, 500.0, 480.0],
        max_batch_tokens=128,
    )


class TestSpsCostTableInvariants(CustomTestCase):
    def test_rejects_non_increasing_batch_tokens(self):
        with self.assertRaises(ValueError):
            SpsCostTable(
                sample_batch_tokens=[8, 8, 16],
                sample_steps_per_sec=[1.0, 2.0, 3.0],
                max_batch_tokens=16,
            )

    def test_rejects_unsorted_batch_tokens(self):
        with self.assertRaises(ValueError):
            SpsCostTable(
                sample_batch_tokens=[16, 8],
                sample_steps_per_sec=[1.0, 2.0],
                max_batch_tokens=16,
            )

    def test_rejects_length_mismatch(self):
        with self.assertRaises(ValueError):
            SpsCostTable(
                sample_batch_tokens=[8, 16],
                sample_steps_per_sec=[1.0],
                max_batch_tokens=16,
            )


class TestSpsCostTableLookup(CustomTestCase):
    def test_lookup_exact_probe_returns_that_sps(self):
        table = _make_table()
        self.assertEqual(table.lookup(8), 1000.0)
        self.assertEqual(table.lookup(16), 950.0)
        self.assertEqual(table.lookup(32), 500.0)
        self.assertEqual(table.lookup(64), 480.0)

    def test_lookup_below_first_probe_clamps_to_first(self):
        table = _make_table()
        self.assertEqual(table.lookup(1), 1000.0)
        self.assertEqual(table.lookup(7), 1000.0)

    def test_lookup_above_last_probe_clamps_to_last(self):
        table = _make_table()
        self.assertEqual(table.lookup(65), 480.0)
        self.assertEqual(table.lookup(10_000), 480.0)


class TestPhysicalGraphSps(CustomTestCase):
    def test_native_fit_uses_replayed_graph_instead_of_budget(self):
        from sglang.benchmark.dspark_sps_profiler import build_table_from_summaries

        summaries = [
            dict(
                batch_size_per_rank=bs,
                batch_tokens=logical,
                graph_tokens=physical,
                graph_cost_supported=True,
                frac=0.35,
                steps_per_sec=1 / seconds,
            )
            for bs, logical, physical, seconds in (
                (16, 22, 48, 0.04),
                (16, 90, 96, 0.05),
                (32, 44, 48, 0.043),
                (32, 88, 96, 0.053),
            )
        ]
        kwargs = dict(max_batch_tokens=None, offdiag=True)
        table = build_table_from_summaries(summaries=summaries, **kwargs)
        self.assertEqual(table.m_axis, "graph_tokens")
        self.assertEqual(table.m_probes, [48, 96])
        self.assertAlmostEqual(
            table.graph_step_time(num_reqs=16, graph_tokens=48), 0.04
        )
        self.assertAlmostEqual(
            table.graph_step_time(num_reqs=32, graph_tokens=96), 0.053
        )
        self.assertEqual(
            SpsAdditiveCostTable.from_json(table.to_json()).to_json(), table.to_json()
        )
        with self.assertRaisesRegex(ValueError, "physical graph size"):
            table.step_time(num_reqs=16, budget=22)
        del summaries[0]["graph_tokens"]
        with self.assertRaisesRegex(ValueError, "actually replayed"):
            build_table_from_summaries(summaries=summaries, **kwargs)
        del summaries[0]["graph_cost_supported"]
        with self.assertRaisesRegex(ValueError, "Cannot mix"):
            build_table_from_summaries(summaries=summaries, **kwargs)
        for summary in summaries[1:]:
            del summary["graph_cost_supported"]
        legacy = build_table_from_summaries(summaries=summaries, **kwargs)
        self.assertEqual(legacy.m_axis, "logical_tokens")

    def test_graph_capability_and_key_are_saved_by_collection(self):
        from sglang.benchmark.dspark_sps_profiler import (
            LoadInfo,
            SpsRow,
            postprocess_round,
            round_summary_dict,
        )

        def rows(replay_key):
            return [
                SpsRow(
                    forward_ct=i,
                    num_running_reqs=16,
                    num_verify_tokens=48,
                    step_time=0.04,
                    verify_tokens_graph_key=replay_key,
                )
                for i in range(20)
            ]

        args = dict(
            batch_size_per_rank=16,
            dp_size=2,
            verify_num_draft_tokens=6,
            min_steady_steps=8,
            load_info=LoadInfo(32, 64, 1.0, True),
            frac=0.35,
            graph_cost_supported=True,
        )
        outcome = postprocess_round(rank_rows=[rows(48), rows(48)], **args)
        summary = round_summary_dict(outcome=outcome, repeat=0)
        self.assertIs(summary["graph_cost_supported"], True)
        self.assertEqual(summary["graph_tokens"], 48)
        self.assertEqual(summary["batch_tokens"], 44)
        with self.assertRaisesRegex(RuntimeError, "actually replayed"):
            postprocess_round(rank_rows=[rows(48), rows(-1)], **args)


class TestPdSpsProfiler(CustomTestCase):
    @staticmethod
    def _flush_clock():
        elapsed = 0.0

        def sleep(seconds):
            nonlocal elapsed
            elapsed += seconds

        return Mock(
            monotonic=Mock(side_effect=lambda: elapsed), sleep=Mock(side_effect=sleep)
        )

    def test_pd_flush_retries_transport_errors_and_busy_response(self):
        from sglang.benchmark import dspark_sps_profiler as profiler

        clock = self._flush_clock()
        attempts = [
            profiler.requests.exceptions.ConnectionError(),
            profiler.requests.exceptions.Timeout(),
            Mock(status_code=400),
            Mock(status_code=200),
        ]
        with (
            patch.object(profiler, "time", clock),
            patch.object(profiler.requests, "post", side_effect=attempts) as post,
        ):
            profiler.flush_pd_cache(base_url="http://engine")
        self.assertEqual(post.call_count, 4)
        self.assertEqual(clock.sleep.call_count, 3)

    def test_pd_flush_does_not_retry_http_auth_failure(self):
        from sglang.benchmark import dspark_sps_profiler as profiler

        response = profiler.requests.Response()
        response.status_code = 403
        response.url = "http://engine/flush_cache"
        clock = self._flush_clock()
        with (
            patch.object(profiler, "time", clock),
            patch.object(profiler.requests, "post", return_value=response) as post,
            self.assertRaises(profiler.requests.exceptions.HTTPError) as error,
        ):
            profiler.flush_pd_cache(base_url="http://engine")
        self.assertIs(error.exception.response, response)
        post.assert_called_once()
        clock.sleep.assert_not_called()

    def test_pd_flush_timeout_fails_closed_and_caps_request_time(self):
        from sglang.benchmark import dspark_sps_profiler as profiler

        clock = self._flush_clock()
        with (
            patch.object(profiler, "time", clock),
            patch.object(profiler, "DEFAULT_TIMEOUT", 1.5),
            patch.object(
                profiler.requests, "post", return_value=Mock(status_code=400)
            ) as post,
            self.assertRaisesRegex(
                TimeoutError, "Could not confirm PD cache flush"
            ) as error,
        ):
            profiler.flush_pd_cache(base_url="http://engine")
        self.assertEqual(
            [call.kwargs["timeout"] for call in post.call_args_list], [1.5, 0.5]
        )
        self.assertEqual(
            [call.args[0] for call in clock.sleep.call_args_list], [1.0, 0.5]
        )
        self.assertEqual(error.exception.__cause__.response.status_code, 400)

    def test_fake_prefill_is_explicit_and_uses_the_base_url(self):
        from sglang.benchmark.dspark_sps_profiler import add_run_args

        parser = argparse.ArgumentParser()
        add_run_args(parser)
        colocate = parser.parse_args(["--base-url", "http://engine"])
        self.assertEqual(colocate.base_url, "http://engine")
        self.assertFalse(colocate.fake_prefill)
        decode = parser.parse_args(["--base-url", "http://decode", "--fake-prefill"])
        self.assertEqual(decode.base_url, "http://decode")
        self.assertTrue(decode.fake_prefill)
        with self.assertRaises(SystemExit):
            parser.parse_args(["--fake-prefill"])

    def test_fake_prefill_requires_decode_and_matches_attention_dp_streams(self):
        from sglang.benchmark import dspark_sps_profiler as profiler

        response = Mock()
        rank_record = dict(
            dp_size=1,
            effective_max_running_requests_per_dp=32,
            dspark_info_record=dict(
                mode="static",
                components=["core", "step_cpu_time"],
                simulate_acc_len=1.0,
                verify_num_draft_tokens=6,
            ),
        )
        info = dict(
            disaggregation_mode="decode",
            speculative_algorithm="DSPARK",
            attn_dp_size=2,
            tokenizer_path="tokenizer",
            internal_states=[rank_record, rank_record],
        )
        response.json.return_value = info
        args = dict(base_url="http://decode", local_tokenizer_path=None)
        with patch.object(profiler.requests, "get", return_value=response) as get:
            context = profiler.fetch_server_context(**args, fake_prefill=True)
            get.assert_called_once_with(
                "http://decode/server_info", timeout=profiler.DEFAULT_TIMEOUT
            )
            self.assertEqual(context.dp_size, 2)
            self.assertEqual(context.skip_max_running_requests_threshold, 64)
            self.assertTrue(context.fake_prefill)
            with self.assertRaisesRegex(ValueError, "pass --fake-prefill"):
                profiler.fetch_server_context(**args)
            info["attn_dp_size"] = 4
            with self.assertRaisesRegex(ValueError, "exposes 2 SPS streams"):
                profiler.fetch_server_context(**args, fake_prefill=True)
            info["disaggregation_mode"] = "prefill"
            with self.assertRaisesRegex(ValueError, "requires a PD Decode engine"):
                profiler.fetch_server_context(**args, fake_prefill=True)

    def test_fake_prefill_payloads_assign_exact_rank_and_room(self):
        from sglang.benchmark.dspark_sps_profiler import fake_prefill_payloads
        from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST

        args = dict(dp_size=2, first_room=501, temperature=1, max_new_tokens=100)
        payloads = fake_prefill_payloads(input_ids=[[10], [11], [20], [21]], **args)
        self.assertEqual(len(payloads), 2)
        for rank, payload in enumerate(payloads):
            self.assertEqual(payload["routed_dp_rank"], rank)
            self.assertEqual(
                payload["bootstrap_room"], [501 + 2 * rank, 502 + 2 * rank]
            )
            self.assertEqual(payload["bootstrap_host"], [FAKE_BOOTSTRAP_HOST] * 2)
            self.assertNotIn("bootstrap_port", payload)
            self.assertEqual(payload["input_ids"], [[10 + 10 * rank], [11 + 10 * rank]])
        with self.assertRaisesRegex(ValueError, "equal non-empty batch"):
            fake_prefill_payloads(input_ids=[[10]], **args)


class TestLoadSpsTableFromPath(CustomTestCase):
    def test_load_from_path_round_trips_table_and_lookup(self):
        table = _make_table()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sps.json"
            path.write_text(table.to_json(), encoding="utf-8")
            loaded = load_sps_table_from_path(str(path))
        self.assertEqual(loaded.sample_batch_tokens, table.sample_batch_tokens)
        self.assertEqual(loaded.sample_steps_per_sec, table.sample_steps_per_sec)
        self.assertEqual(loaded.max_batch_tokens, table.max_batch_tokens)
        for batch_tokens in (1, 8, 31, 64, 200):
            self.assertEqual(loaded.lookup(batch_tokens), table.lookup(batch_tokens))


class TestFlatTableLookupIsConstant(CustomTestCase):
    def test_flat_table_lookup_is_one_for_any_batch(self):
        flat = SpsCostTable(
            sample_batch_tokens=[1],
            sample_steps_per_sec=[1.0],
            max_batch_tokens=4096,
        )
        for batch_tokens in (0, 1, 2, 17, 256, 100_000):
            self.assertEqual(flat.lookup(batch_tokens), 1.0)


class TestProfileSpsTable(CustomTestCase):
    def test_profile_sorts_out_of_order_probes(self):
        table = profile_sps_table(
            probes=[(32, 500.0), (8, 1000.0), (16, 950.0)],
        )
        self.assertEqual(table.sample_batch_tokens, [8, 16, 32])
        self.assertEqual(table.sample_steps_per_sec, [1000.0, 950.0, 500.0])

    def test_profile_rejects_duplicate_batch_tokens(self):
        with self.assertRaises(ValueError):
            profile_sps_table(probes=[(8, 1000.0), (8, 900.0)])

    def test_profile_rejects_empty_probes(self):
        with self.assertRaises(ValueError):
            profile_sps_table(probes=[])

    def test_profile_max_batch_tokens_defaults_to_largest_probe(self):
        table = profile_sps_table(probes=[(8, 1000.0), (64, 480.0), (16, 950.0)])
        self.assertEqual(table.max_batch_tokens, 64)

    def test_profile_honors_explicit_max_batch_tokens(self):
        table = profile_sps_table(
            probes=[(8, 1000.0), (16, 950.0)], max_batch_tokens=256
        )
        self.assertEqual(table.max_batch_tokens, 256)


def _build_sps_cost_table_for(testcase, *, sps_table_path):
    from sglang.srt.runtime_context import get_context
    from sglang.srt.speculative.dspark_components.dspark_planner import (
        build_sps_cost_table,
    )

    # Both the table path and the bound come from the published bags, so the
    # case publishes them.
    override = get_context().override_server_args(
        speculative_dspark_sps_table_path=sps_table_path,
        max_running_requests=4,
    )
    override.install()
    testcase.addCleanup(override.restore)
    return build_sps_cost_table(verify_num_draft_tokens=5)


class TestBuildSpsCostTableContract(CustomTestCase):
    def test_unset_table_path_returns_flat_table(self):
        for sps_table_path in (None, ""):
            table = _build_sps_cost_table_for(self, sps_table_path=sps_table_path)
            self.assertEqual(table.sample_batch_tokens, [1])
            self.assertEqual(table.sample_steps_per_sec, [1.0])
            self.assertEqual(table.max_batch_tokens, 20)

    def test_real_path_loads_table(self):
        table = _make_table()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sps.json"
            path.write_text(table.to_json(), encoding="utf-8")
            loaded = _build_sps_cost_table_for(self, sps_table_path=str(path))
        self.assertEqual(loaded.sample_batch_tokens, table.sample_batch_tokens)
        self.assertEqual(loaded.sample_steps_per_sec, table.sample_steps_per_sec)
        self.assertEqual(loaded.max_batch_tokens, table.max_batch_tokens)


class TestIsUninitializedSpsTable(CustomTestCase):
    def test_additive_table_is_never_uninitialized(self):
        table = SpsAdditiveCostTable(
            bias_seconds=0.1,
            bs_probes=[128, 192, 256],
            alpha_seconds=[0.0, 0.008, 0.016],
            m_probes=[384, 512, 1024],
            theta_seconds=[0.0, 0.02, 0.1],
        )
        self.assertFalse(is_uninitialized_sps_table(table))

    def test_placeholder_diagonal_table_is_uninitialized(self):
        self.assertTrue(
            is_uninitialized_sps_table(
                build_uninitialized_sps_table(max_batch_tokens=128)
            )
        )

    def test_real_diagonal_table_is_initialized(self):
        self.assertFalse(is_uninitialized_sps_table(_make_table()))


if __name__ == "__main__":
    unittest.main()
