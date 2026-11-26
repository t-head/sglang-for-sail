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

import yaml

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

    def test_all_green_has_aligned_columns_and_no_mentions(self):
        report = {
            "elapsed_s": 3701,
            "results": [
                {
                    "model": "atlas",
                    "status": "pass",
                    "score": 0.9682,
                    "golden_score": 0.9682,
                    "job_elapsed_s": 3480,
                },
                {"model": "longer-model", "status": "pass", "job_elapsed_s": 24},
            ],
        }
        for version in ("v1.0.0", "v0.5.23"):
            with self.subTest(version=version):
                with patch.dict("os.environ", {"NOTIFY_VERSION": version}):
                    payload = notify.make_payload(report, "{invalid", "mon")
                title = f"[sglang | {version}] PPU Nightly 晨报"
                message = payload["markdown"]["text"]
                self.assertEqual(title, payload["markdown"]["title"])
                self.assertTrue(message.startswith(f"### {title}\n\n"))
                self.assertIn("通过 **2/2** · 失败 **0** · 总耗时 1h 01m 41s", message)
                self.assertNotIn("at", payload)
                self.assertNotIn("🔴", message)
                header, atlas, longer = message.split("```text\n")[1].splitlines()[:3]
                self.assertEqual(header.index("GSM8K"), atlas.index("96.82% (96.82%)"))
                self.assertEqual(header.index("GSM8K"), longer.index("—"))
                self.assertEqual(header.index("Time"), atlas.index("58m"))
                self.assertEqual(header.index("Time"), longer.index("24s"))

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
                    "job_id": 456,
                    "job_elapsed_s": 3480,
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
        message = payload["markdown"]["text"]
        self.assertLess(
            message.index("🔴 失败用例 (1)"), message.index("🟢 通过用例 (1)")
        )
        self.assertIn("\n\n#### 🔴 失败用例 (1)\n\n- `borealis`", message)
        self.assertIn("\n\n#### 🟢 通过用例 (1)\n\n```text", message)
        for word in (
            "@duty",
            "@+10000000000",
            "90.00% vs 95.00%",
            "b" * 40,
            "10:30",
            "耗时 58m",
            "[直达 Job](https://github.com/example/framework/actions/runs/123/job/456)",
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
                    "job_id": "456)/PRIVATE-JOB-URL",
                    "job_elapsed_s": "PRIVATE-TIME",
                    "first_failing_sha": "PRIVATE-SHA",
                }
            ]
        }
        with patch.dict(
            "os.environ",
            {
                "NOTIFY_REF": "PRIVATE-REF",
                "NOTIFY_VERSION": "PRIVATE-VERSION",
            },
        ):
            payload = notify.make_payload(report, "", "fri")
        self.assertIn("unknown · 耗时 —", payload["markdown"]["text"])
        self.assertIn(
            "[整轮 Run](https://github.com/example/framework/actions/runs/123)",
            payload["markdown"]["text"],
        )
        self.assertNotIn("直达 Job", payload["markdown"]["text"])
        self.assertNotIn("PRIVATE", str(payload))

    def test_workflow_limits_manual_notifications_to_selected_full_runs(self):
        workflow = (
            Path(__file__).resolve().parents[3]
            / ".github/workflows/nightly-test-ppu.yml"
        )
        job = yaml.safe_load(workflow.read_text())["jobs"]["notify-summary"]
        condition = " ".join(job["if"].split())
        self.assertTrue(condition.startswith("always() && !cancelled() &&"))
        self.assertIn(
            "(github.event_name == 'workflow_dispatch' && inputs.model_filter == '' && "
            "needs.pr-gate.result == 'success' && needs.pr-gate.outputs.approved == 'true' && "
            "needs.plan.result == 'success' &&",
            condition,
        )
        self.assertIn(
            "(github.event_name == 'schedule' && startsWith(github.workflow_ref, "
            "'t-head/sglang-for-sail/.github/workflows/nightly-dispatch-ppu.yml@refs/heads/')) ||",
            condition,
        )

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
        self.assertEqual(
            {
                "retry": notify.JobExecution("failure", 2, 202, 30),
                "prior": notify.JobExecution("success", 1, 101, 1800),
            },
            jobs,
        )
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
        self.assertIn("/actions/runs/123/job/202", payload["markdown"]["text"])

    def test_missing_evidence_and_conflicting_job_result_fail_closed(self):
        def missing(_attempt):
            raise FileNotFoundError

        result = notify.model_result(
            "atlas", notify.JobExecution("success", 2, 201, None), missing
        )
        self.assertEqual(
            ("fail", "missing evidence"), (result["status"], result["phase"])
        )
        result = notify.model_result(
            "atlas",
            notify.JobExecution("failure", 2, 201, 30),
            lambda _: ("pass", "unknown", None, None),
        )
        self.assertEqual("fail", result["status"])
        self.assertEqual(201, result["job_id"])
        self.assertEqual(30, result["job_elapsed_s"])

    def test_targets_json_routes_single_target_and_does_not_fallback_on_failure(self):
        targets = {
            "default": {
                "webhook_url": "https://example.com/robot/send?access_token=default-tok",
                "signing_secret": "SEC-default",
            },
            "routes": {
                "v0.5.23": {
                    "webhook_url": "https://example.com/robot/send?access_token=v0523-tok",
                    "signing_secret": "SEC-v0523",
                }
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            path.write_text('{"results":[{"model":"atlas","status":"pass"}]}')
            with (
                patch.dict(
                    "os.environ",
                    {
                        notify.TARGETS: json.dumps(targets),
                        notify.URL: "https://example.com/robot/send?access_token=legacy-tok",
                        notify.SECRET: "SEC-legacy",
                        "NOTIFY_REF": "release/v0.5.23",
                    },
                ),
                patch.object(notify, "build_opener") as build,
            ):
                opener = build.return_value
                opener.open.return_value.__enter__.return_value.read.return_value = (
                    b'{"errcode":0}'
                )
                self.assertEqual(0, notify.main(["--report", str(path)]))
                self.assertEqual(1, opener.open.call_count)
                req = opener.open.call_args.args[0]
                qs = parse_qs(urlsplit(req.full_url).query)
                self.assertEqual(["v0523-tok"], qs["access_token"])
                expected_signed = notify.signed_url(
                    targets["routes"]["v0.5.23"]["webhook_url"],
                    "SEC-v0523",
                    int(qs["timestamp"][0]),
                )
                self.assertEqual(expected_signed, req.full_url)
                body = json.loads(req.data.decode())
                self.assertTrue(
                    body["markdown"]["title"].startswith("[sglang | v0.5.23] ")
                )

            with (
                patch.dict(
                    "os.environ",
                    {
                        notify.TARGETS: json.dumps(targets),
                        notify.URL: "https://example.com/robot/send?access_token=legacy-tok",
                        notify.SECRET: "SEC-legacy",
                        "NOTIFY_REF": "release/v0.5.23",
                    },
                ),
                patch.object(notify, "build_opener") as build,
                redirect_stderr(io.StringIO()) as logs,
            ):
                build.return_value.open.side_effect = HTTPError(
                    "https://example.com/robot/send?access_token=v0523-tok",
                    503,
                    "v0523-tok",
                    {},
                    None,
                )
                self.assertEqual(0, notify.main(["--report", str(path)]))
                self.assertEqual(1, build.return_value.open.call_count)
                self.assertIn("::warning::", logs.getvalue())
                self.assertNotIn("v0523-tok", logs.getvalue())
                self.assertNotIn("default-tok", logs.getvalue())

    def test_targets_json_unregistered_version_uses_default_then_legacy_then_skips(
        self,
    ):
        targets_with_default = {
            "default": {
                "webhook_url": "https://example.com/robot/send?access_token=default-tok",
                "signing_secret": "SEC-default",
            },
            "routes": {
                "v0.5.23": {
                    "webhook_url": "https://example.com/robot/send?access_token=v0523-tok",
                }
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            path.write_text('{"results":[{"model":"atlas","status":"pass"}]}')
            with (
                patch.dict(
                    "os.environ",
                    {
                        notify.TARGETS: json.dumps(targets_with_default),
                        notify.URL: "https://example.com/robot/send?access_token=legacy-tok",
                        "NOTIFY_REF": "release/v1.1.0",
                    },
                ),
                patch.object(notify, "build_opener") as build,
            ):
                opener = build.return_value
                opener.open.return_value.__enter__.return_value.read.return_value = (
                    b'{"errcode":0}'
                )
                self.assertEqual(0, notify.main(["--report", str(path)]))
                self.assertEqual(1, opener.open.call_count)
                req = opener.open.call_args.args[0]
                self.assertEqual(
                    ["default-tok"],
                    parse_qs(urlsplit(req.full_url).query)["access_token"],
                )
                body = json.loads(req.data.decode())
                self.assertTrue(
                    body["markdown"]["title"].startswith("[sglang | v1.1.0] ")
                )

            routes_only = {
                "routes": {
                    "v0.5.23": {
                        "webhook_url": "https://example.com/robot/send?access_token=v0523-tok",
                    }
                }
            }
            with (
                patch.dict(
                    "os.environ",
                    {
                        notify.TARGETS: json.dumps(routes_only),
                        notify.URL: "https://example.com/robot/send?access_token=legacy-tok",
                        "NOTIFY_REF": "release/v1.1.0",
                    },
                ),
                patch.object(notify, "build_opener") as build,
            ):
                opener = build.return_value
                opener.open.return_value.__enter__.return_value.read.return_value = (
                    b'{"errcode":0}'
                )
                self.assertEqual(0, notify.main(["--report", str(path)]))
                self.assertEqual(1, opener.open.call_count)
                req = opener.open.call_args.args[0]
                self.assertEqual(
                    ["legacy-tok"],
                    parse_qs(urlsplit(req.full_url).query)["access_token"],
                )

            with (
                patch.dict(
                    "os.environ",
                    {
                        notify.TARGETS: json.dumps(routes_only),
                        "NOTIFY_REF": "release/v1.1.0",
                    },
                ),
                patch.object(notify, "build_opener") as build,
                redirect_stderr(io.StringIO()) as logs,
            ):
                self.assertEqual(0, notify.main(["--report", "absent.json"]))
                self.assertEqual(1, notify.main(["--smoke-test"]))
                self.assertEqual(0, build.return_value.open.call_count)
                self.assertIn("::warning::", logs.getvalue())

    def test_targets_json_invalid_fails_closed_without_fallback_or_secret_leak(self):
        secret_marker = "secret-marker-do-not-leak"
        invalid_payloads = [
            "{broken-json",
            "[]",
            "{}",
            '{"routes":{}}',
            json.dumps(
                {
                    "default": {"webhook_url": f"https://example.com/{secret_marker}"},
                    "extra": {},
                }
            ),
            json.dumps(
                {
                    "routes": {
                        "release/v0.5.23": {
                            "webhook_url": f"https://example.com/{secret_marker}"
                        }
                    }
                }
            ),
            json.dumps(
                {
                    "default": {
                        "webhook_url": f"http://example.com/{secret_marker}",
                        "signing_secret": secret_marker,
                    }
                }
            ),
            json.dumps(
                {
                    "default": {
                        "webhook_url": f"https://example.com/{secret_marker}#frag",
                    }
                }
            ),
            json.dumps(
                {
                    "default": {
                        "webhook_url": f"https://user@example.com/{secret_marker}",
                    }
                }
            ),
            json.dumps(
                {
                    "default": {
                        "webhook_url": f"https://example.com/{secret_marker}",
                        "unknown": 1,
                    }
                }
            ),
            json.dumps(
                {
                    "default": {
                        "webhook_url": "https://example.com/robot/send",
                        "signing_secret": 123,
                    }
                }
            ),
            "{"
            + f'"default":{{"webhook_url":"https://example.com/{secret_marker}"}}'
            + " " * (64 * 1024)
            + "}",
            " ",
            " " * 65537,
            "\t" * 65537,
            "\u2003",
            "\u2000" * 21846,
            json.dumps(
                {
                    "default": {
                        "webhook_url": f"https://example.com/{secret_marker}",
                    },
                    "routes": {
                        "v1.1.0": {
                            "webhook_url": f"https://example.com:notaport/{secret_marker}",
                        }
                    },
                }
            ),
            json.dumps(
                {
                    "default": {
                        "webhook_url": f"https://example.com/{secret_marker}",
                    },
                    "routes": {
                        "v1.1.0": {
                            "webhook_url": f"https://example.com:65536/{secret_marker}",
                        }
                    },
                }
            ),
            json.dumps(
                {
                    "default": {
                        "webhook_url": f"https://example.com/{secret_marker}",
                    },
                    "routes": {
                        "v1.1.0": {
                            "webhook_url": f"https://example.com:0/{secret_marker}",
                        }
                    },
                }
            ),
            json.dumps(
                {
                    "default": {
                        "webhook_url": f"https://example.com/{secret_marker}",
                    },
                    "routes": {
                        "v1.1.0": {
                            "webhook_url": f"https://example.com:/{secret_marker}",
                        }
                    },
                }
            ),
        ]
        for raw in invalid_payloads:
            with (
                self.subTest(raw=raw[:40]),
                patch.dict(
                    "os.environ",
                    {
                        notify.TARGETS: raw,
                        notify.URL: "https://example.com/robot/send?access_token=legacy-tok",
                    },
                ),
                patch.object(notify, "build_opener") as build,
                redirect_stderr(io.StringIO()) as logs,
            ):
                self.assertEqual(0, notify.main(["--report", "absent.json"]))
                self.assertEqual(1, notify.main(["--smoke-test"]))
                self.assertEqual(0, build.return_value.open.call_count)
                self.assertIn("::warning::", logs.getvalue())
                self.assertNotIn(secret_marker, logs.getvalue())

        base_json = json.dumps(
            {
                "default": {"webhook_url": "https://example.com:8443/robot/send"},
                "routes": {"v1.1.0": {"webhook_url": "https://[::1]:8443/robot/send"}},
            }
        )
        exact_limit = base_json[:-1] + " " * (64 * 1024 - len(base_json)) + "}"
        self.assertEqual(64 * 1024, len(exact_limit.encode("utf-8")))
        with (
            patch.dict("os.environ", {notify.TARGETS: exact_limit}),
            patch.object(notify, "build_opener") as build,
        ):
            build.return_value.open.return_value.__enter__.return_value.read.return_value = b'{"errcode":0}'
            self.assertEqual(0, notify.main(["--smoke-test"]))
            self.assertEqual(1, build.return_value.open.call_count)

    def test_run_jobs_strips_targets_and_notification_secrets(self):
        captured_envs = []

        def fake_check_output(command, **kwargs):
            captured_envs.append(kwargs["env"])
            endpoint = command[4]
            if endpoint.endswith("/attempts/2"):
                return b'{"run_started_at":"2023-01-02T03:00:00Z"}'
            raise AssertionError(endpoint)

        with (
            patch.dict(
                "os.environ",
                {
                    notify.URL: "https://example.com/u",
                    notify.SECRET: "s",
                    notify.ROSTER: "{}",
                    notify.TARGETS: '{"default":{"webhook_url":"https://example.com/t"}}',
                    "KEEP_ME": "1",
                },
            ),
            patch.object(
                notify.subprocess, "check_output", side_effect=fake_check_output
            ),
        ):
            _, _ = notify.run_jobs(set())
        self.assertEqual(1, len(captured_envs))
        for forbidden in (notify.URL, notify.SECRET, notify.ROSTER, notify.TARGETS):
            self.assertNotIn(forbidden, captured_envs[0])
        self.assertEqual("1", captured_envs[0]["KEEP_ME"])

    def test_normalize_version_and_dynamic_version_titles(self):
        cases = [
            ("", "main", "", "main"),
            ("", "refs/heads/main", "", "main"),
            ("", "releases/v0.35.0", "", "v0.35.0"),
            ("", "refs/heads/releases/v0.35.0", "", "v0.35.0"),
            ("", "release/v1.1.0", "", "v1.1.0"),
            ("", "refs/heads/release/v1.1.0", "", "v1.1.0"),
            ("", "yuhe/v1.0.0", "", "v1.0.0"),
            ("", "refs/heads/yuhe/v1.0.0", "", "v1.0.0"),
            ("", "v1.2.3-rc1", "", "v1.2.3-rc1"),
            ("v0.5.23", "release/v1.1.0", "v1.0.0", "v0.5.23"),
            ("", "a" * 40, "v0.5.23", "v0.5.23"),
            ("", "releases/main", "", ""),
            ("", "yuhe/main", "", ""),
            ("", "refs/heads/v0.5.23", "", ""),
            ("", "refs/tags/v0.5.23", "", ""),
            ("", "refs/heads/Main", "", ""),
            ("bad/ref", "release/v0.5.23", "v0.5.23", ""),
        ]
        for explicit, ref, fallback, expected in cases:
            with self.subTest(explicit=explicit, ref=ref, fallback=fallback):
                self.assertEqual(
                    expected, notify.normalize_version(explicit, ref, fallback)
                )

        report = {"results": [{"model": "atlas", "status": "pass"}]}
        for version, expected_prefix in (
            ("v1.1.0", "[sglang | v1.1.0] "),
            ("v0.5.23-rc1", "[sglang | v0.5.23-rc1] "),
            ("bad/version", "[example/framework] "),
        ):
            payload = notify.make_payload(report, "", "mon", version=version)
            self.assertTrue(payload["markdown"]["title"].startswith(expected_prefix))

    def test_cli_version_precedence_and_smoke_test_version_flag(self):
        targets = {
            "default": {
                "webhook_url": "https://example.com/robot/send?access_token=default-tok"
            },
            "routes": {
                "v0.5.23": {
                    "webhook_url": "https://example.com/robot/send?access_token=v0523-tok"
                },
                "v1.0.0": {
                    "webhook_url": "https://example.com/robot/send?access_token=v100-tok"
                },
            },
        }
        with (
            patch.dict(
                "os.environ",
                {
                    notify.TARGETS: json.dumps(targets),
                    "NOTIFY_REF": "yuhe/v1.0.0",
                    "NOTIFY_VERSION": "v0.5.23",
                },
            ),
            patch.object(notify, "build_opener") as build,
        ):
            opener = build.return_value
            opener.open.return_value.__enter__.return_value.read.return_value = (
                b'{"errcode":0}'
            )
            self.assertEqual(0, notify.main(["--smoke-test", "--version", "v0.5.23"]))
            req = opener.open.call_args.args[0]
            self.assertEqual(
                ["v0523-tok"], parse_qs(urlsplit(req.full_url).query)["access_token"]
            )
            body = json.loads(req.data.decode())
            self.assertEqual("[sglang | v0.5.23] PPU CI", body["markdown"]["title"])

            opener.open.reset_mock()
            with patch.dict("os.environ", {"NOTIFY_REF": "", "NOTIFY_VERSION": ""}):
                self.assertEqual(0, notify.main(["--smoke-test"]))
                req2 = opener.open.call_args.args[0]
                self.assertEqual(
                    ["default-tok"],
                    parse_qs(urlsplit(req2.full_url).query)["access_token"],
                )
                body2 = json.loads(req2.data.decode())
                self.assertEqual("PPU CI", body2["markdown"]["title"])

            opener.open.reset_mock()
            with redirect_stderr(io.StringIO()) as logs:
                self.assertEqual(
                    1, notify.main(["--smoke-test", "--version", "invalid/ver"])
                )
                self.assertEqual(0, opener.open.call_count)
                self.assertIn("::warning::", logs.getvalue())


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
            "NOTIFY_GATE_APPROVED": "true",
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
            evidence = root / f"v0523-model_sanity-{model}-2" / "recipe.json"
            evidence.parent.mkdir()
            evidence.write_text(json.dumps(payload))
            jobs = {
                f"release-v0523 / Model Sanity ({model})": notify.JobExecution(
                    "failure", 2, 456, 30
                )
            }
            with (
                patch.dict("os.environ", environment, clear=True),
                patch.object(nightly, "run_jobs", return_value=(jobs, 60)) as fetch,
            ):
                result = nightly.collect(root)["results"][0]
                fetch.assert_called_once_with(set(jobs))
                self.assertEqual("gsm8k_score vs golden", result["phase"])
                self.assertEqual(0.88, result["score"])
                self.assertEqual(0.97, result["golden_score"])
                self.assertEqual(456, result["job_id"])
                self.assertEqual(30, result["job_elapsed_s"])
                with patch.dict("os.environ", {"NOTIFY_GATE_APPROVED": "false"}):
                    self.assertEqual(
                        "approval", nightly.collect(root)["results"][0]["phase"]
                    )
                with patch.dict("os.environ", {"NOTIFY_PLAN_RESULT": "failure"}):
                    self.assertEqual(
                        "selection", nightly.collect(root)["results"][0]["phase"]
                    )
                evidence.write_text(json.dumps({**payload, "source_sha": "b" * 40}))
                self.assertEqual(
                    "missing evidence", nightly.collect(root)["results"][0]["phase"]
                )
                with patch.dict("os.environ", {"NOTIFY_BUILD_RESULT": "failure"}):
                    self.assertEqual(
                        "build", nightly.collect(root)["results"][0]["phase"]
                    )

    def test_workflow_forwards_targets_json_secret_and_gates_configuration(self):
        workflow_path = (
            Path(__file__).resolve().parents[3]
            / ".github/workflows/nightly-test-ppu.yml"
        )
        workflow_doc = yaml.safe_load(workflow_path.read_text())
        on_section = workflow_doc.get("on", workflow_doc.get(True))
        secrets_decl = on_section["workflow_call"]["secrets"]
        self.assertEqual(
            {"required": False}, secrets_decl["PPU_CI_NOTIFY_TARGETS_JSON"]
        )
        steps = workflow_doc["jobs"]["notify-summary"]["steps"]
        configured = next(step for step in steps if step.get("id") == "configured")
        send = next(
            step
            for step in steps
            if step.get("name") == "Send one Markdown notification"
        )
        self.assertEqual(
            "${{ secrets.PPU_CI_NOTIFY_TARGETS_JSON }}",
            configured["env"]["PPU_CI_NOTIFY_TARGETS_JSON"],
        )
        self.assertEqual(
            "${{ secrets.PPU_CI_NOTIFY_TARGETS_JSON }}",
            send["env"]["PPU_CI_NOTIFY_TARGETS_JSON"],
        )
        with tempfile.TemporaryDirectory() as tmp:
            for webhook, targets_json, expected in (
                ("https://example.com/robot/send", "", True),
                (
                    "",
                    '{"default":{"webhook_url":"https://example.com/robot/send"}}',
                    True,
                ),
                ("", "", False),
            ):
                output = Path(tmp) / "github_output.txt"
                output.unlink(missing_ok=True)
                subprocess.run(
                    ["bash", "-e", "-c", configured["run"]],
                    env={
                        "PATH": "/usr/bin:/bin:/opt/homebrew/bin",
                        "PPU_CI_NOTIFY_WEBHOOK_URL": webhook,
                        "PPU_CI_NOTIFY_TARGETS_JSON": targets_json,
                        "GITHUB_OUTPUT": str(output),
                    },
                    check=True,
                )
                enabled = output.read_text().strip() if output.exists() else ""
                self.assertEqual(expected, enabled == "enabled=true")


if __name__ == "__main__":
    unittest.main()
