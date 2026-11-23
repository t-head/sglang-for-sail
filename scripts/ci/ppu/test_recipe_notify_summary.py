# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for the signed Markdown notification protocol."""

import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ppu_notify_summary as notify


class NotifySummaryTest(unittest.TestCase):
    def setUp(self):
        env = patch.dict(
            "os.environ",
            {
                "GITHUB_REPOSITORY": "example/framework",
                "GITHUB_REF_NAME": "main",
                "GITHUB_EVENT_NAME": "schedule",
                "GITHUB_RUN_ID": "123",
                "GITHUB_RUN_ATTEMPT": "2",
                "NOTIFY_REF": "main",
            },
            clear=True,
        )
        env.start()
        self.addCleanup(env.stop)

    def test_all_green_has_no_mentions(self):
        report = {"elapsed_s": 3701, "results": [{"model": "atlas", "status": "pass"}]}
        payload = notify.make_payload(report, "{invalid", "mon")
        self.assertNotIn("at", payload)
        self.assertNotIn("@member", payload["markdown"]["text"])
        self.assertIn("[✅ 全绿 1/1]", payload["markdown"]["title"])
        self.assertIn("1h 01m 41s", payload["markdown"]["text"])

    def test_partial_failure_routes_only_to_duty_and_failed_model_owner(self):
        report = {
            "results": [
                {"model": "atlas", "status": "pass"},
                {
                    "model": "borealis",
                    "status": "fail",
                    "phase": "gsm8k_score vs golden",
                    "score": 0.90,
                    "golden_score": 0.95,
                    "first_failing_sha": "b" * 40,
                },
            ]
        }
        roster = {
            "weekdays": {"tue": [{"user_id": "duty"}], "mon": [{"user_id": "other"}]},
            "models": {
                "atlas": [{"user_id": "green-owner"}],
                "borealis": [{"user_id": "duty"}, {"mobile": "+10000000000"}],
            },
        }
        payload = notify.make_payload(report, json.dumps(roster), "tue")
        self.assertEqual(
            {"atMobiles": ["+10000000000"], "atUserIds": ["duty"], "isAtAll": False},
            payload["at"],
        )
        self.assertIn("[❌ 1 项失败 · 待认领]", payload["markdown"]["title"])
        message = payload["markdown"]["text"]
        for word in (
            "@duty",
            "@+10000000000",
            "90.00% vs 95.00%",
            "b" * 40,
            "10:30",
            "/actions/runs/123",
        ):
            self.assertIn(word, message)
        self.assertNotIn("green-owner", message)
        self.assertNotIn("@other", message)

    def test_untrusted_fields_are_not_rendered(self):
        report = {
            "results": [
                {
                    "model": "atlas",
                    "status": "fail",
                    "phase": "PRIVATE-DIAGNOSTIC",
                    "score": "PRIVATE-SCORE",
                    "golden_score": 0.9,
                    "log": "PRIVATE-LOG",
                }
            ]
        }
        payload = notify.make_payload(report, "", "fri")
        self.assertIn("unknown", payload["markdown"]["text"])
        self.assertNotIn("PRIVATE", payload["markdown"]["text"])

    def test_hmac_signature_encodes_query_and_replaces_stale_signature(self):
        url = "https://example.test/hooks?key=alpha%2Bbeta&timestamp=old&sign=old"
        signed = notify.signed_url(url, "test-key", 1700000000000)
        self.assertEqual(
            {
                "key": ["alpha+beta"],
                "timestamp": ["1700000000000"],
                "sign": ["gPUhZFrGg8v77EqnloqIsaCjIZxY7xIrF2azCaF/3zU="],
            },
            parse_qs(urlsplit(signed).query),
        )
        self.assertIn("%3D", signed)
        self.assertEqual(url, notify.signed_url(url, "", 1700000000000))

    def test_no_webhook_skips_silently_before_reading_artifacts(self):
        result = subprocess.run(
            [sys.executable, str(Path(notify.__file__)), "--report", "absent.json"],
            capture_output=True,
            check=True,
        )
        self.assertEqual((b"", b""), (result.stdout, result.stderr))

    def test_timeout_http_5xx_and_webhook_rejection_are_warn_only(self):
        url = "https://example.test/hooks?key=fixture-key"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            path.write_text('{"results":[{"model":"atlas","status":"pass"}]}')
            for error in (
                TimeoutError("fixture-key"),
                HTTPError(url, 503, "fixture-key", {}, None),
                None,
            ):
                with (
                    self.subTest(error=type(error)),
                    patch.dict("os.environ", {notify.URL: url}),
                    patch.object(notify, "build_opener") as build,
                    redirect_stderr(io.StringIO()) as logs,
                ):
                    response = MagicMock()
                    response.__enter__.return_value.read.return_value = (
                        b'{"errcode":300,"errmsg":"fixture-key"}'
                    )
                    build.return_value.open.return_value = response
                    build.return_value.open.side_effect = error
                    self.assertEqual(0, notify.main(["--report", str(path)]))
                    self.assertEqual(1, build.return_value.open.call_count)
                    self.assertIn("::warning::", logs.getvalue())
                    self.assertNotIn("fixture-key", logs.getvalue())

    def test_one_successful_post(self):
        with (
            patch.dict("os.environ", {notify.URL: "https://example.test/hooks"}),
            patch.object(notify, "build_opener") as build,
        ):
            opener = build.return_value
            opener.open.return_value.__enter__.return_value.read.return_value = (
                b'{"errcode":0}'
            )
            self.assertTrue(
                notify.deliver(
                    "https://example.test/hooks", "", {"msgtype": "markdown"}
                )
            )
            self.assertEqual(1, opener.open.call_count)
            self.assertEqual("POST", opener.open.call_args.args[0].get_method())
            self.assertEqual(8, opener.open.call_args.kwargs["timeout"])

    def test_retry_uses_the_attempt_that_executed_each_job(self):
        prior = {
            "name": "prior",
            "id": 101,
            "run_attempt": 1,
            "conclusion": "success",
            "started_at": "2023-01-02T01:00:00Z",
            "completed_at": "2023-01-02T01:30:00Z",
        }
        retry = dict(prior, name="retry", id=102)
        retried = dict(
            retry,
            id=202,
            run_attempt=2,
            conclusion="failure",
            started_at="2023-01-02T03:00:00Z",
            completed_at="2023-01-02T03:00:30Z",
        )
        current = [
            {"jobs": [retried]},
            {"jobs": [dict(prior, id=201, run_attempt=2)]},
        ]
        old = [{"jobs": [retry, prior]}]

        def github_api(command, **_):
            endpoint = command[4]
            if endpoint.endswith("/attempts/2"):
                return b'{"run_started_at":"2023-01-02T03:00:00Z"}'
            if "/attempts/1/jobs?" in endpoint:
                return json.dumps(old)
            if "/attempts/2/jobs?" in endpoint:
                return json.dumps(current)
            raise AssertionError(endpoint)

        with (
            patch.object(notify.subprocess, "check_output", side_effect=github_api),
            patch.object(notify.time, "time", return_value=1672628460),
        ):
            jobs, elapsed = notify.run_jobs({"retry", "prior"})
        self.assertEqual({"retry": ("failure", 2), "prior": ("success", 1)}, jobs)
        self.assertEqual(60, elapsed)

        def original_evidence(attempt):
            if attempt != 1:
                raise FileNotFoundError
            return "pass", "unknown", None, None

        self.assertEqual(
            "pass",
            notify.model_result("prior", jobs["prior"], original_evidence)["status"],
        )
        self.assertEqual(
            "missing evidence",
            notify.model_result("retry", jobs["retry"], original_evidence)["phase"],
        )
        results = [
            notify.model_result(model, jobs[model], original_evidence)
            for model in ("prior", "retry")
        ]
        roster = '{"models":{"prior":[{"user_id":"prior-owner"}],"retry":[{"user_id":"retry-owner"}]}}'
        payload = notify.make_payload({"results": results}, roster, "mon")
        self.assertEqual(["retry-owner"], payload["at"]["atUserIds"])
        self.assertIn("[❌ 1 项失败 · 待认领]", payload["markdown"]["title"])

    def test_missing_evidence_and_conflicting_job_result_fail_closed(self):
        def missing(_attempt):
            raise FileNotFoundError

        result = notify.model_result("atlas", ("success", 2), missing)
        self.assertEqual(
            ("fail", "missing evidence"), (result["status"], result["phase"])
        )
        result = notify.model_result(
            "atlas", ("failure", 2), lambda _: ("pass", "unknown", None, None)
        )
        self.assertEqual("fail", result["status"])


class SGLangNotificationTest(unittest.TestCase):
    def test_scheduled_nightly_validates_attempt_and_source_and_supports_prebuilt(self):
        import ppu_notify_nightly as nightly

        model = "glm-5.3"
        source = "a" * 40
        environment = {
            "GITHUB_WORKFLOW_REF": "example/framework/.github/workflows/nightly-dispatch-ppu.yml@refs/heads/main",
            "NOTIFY_MATRIX": json.dumps({"include": [{"model": model}]}),
            "NOTIFY_SOURCE_SHA": source,
            "NOTIFY_GATE_RESULT": "success",
            "NOTIFY_PLAN_RESULT": "success",
            "NOTIFY_BUILD_RESULT": "skipped",
            "NOTIFY_EVAL_RESULT": "success",
            "NOTIFY_MATRIX_RESULT": "failure",
        }
        payload = {
            "schema": "ppu-sanity/v1",
            "model": model,
            "tier": "model_sanity",
            "source_sha": source,
            "route": "prebuilt-image",
            "private_retention": True,
            "outcome": "fail",
            "failure_phase": "gsm8k_eval",
            "gsm8k_score": 0.88,
            "golden_score": 0.97,
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence = root / f"model_sanity-{model}-2" / "recipe.json"
            evidence.parent.mkdir()
            evidence.write_text(json.dumps(payload))
            jobs = {f"release-v100 / Model Sanity ({model})": ("failure", 2)}
            with (
                patch.dict("os.environ", environment, clear=True),
                patch.object(nightly, "run_jobs", return_value=(jobs, 60)) as fetch,
            ):
                result = nightly.collect(root)["results"][0]
                fetch.assert_called_once_with(set(jobs))
                self.assertEqual("gsm8k_score vs golden", result["phase"])
                self.assertEqual(0.88, result["score"])
                evidence.write_text(json.dumps({**payload, "source_sha": "b" * 40}))
                self.assertEqual(
                    "missing evidence", nightly.collect(root)["results"][0]["phase"]
                )
                with patch.dict("os.environ", {"NOTIFY_BUILD_RESULT": "failure"}):
                    self.assertEqual(
                        "build", nightly.collect(root)["results"][0]["phase"]
                    )


if __name__ == "__main__":
    unittest.main()
