# SPDX-License-Identifier: Apache-2.0

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ppu_recipe import (
    DEFAULT_POLICY_ROOT,
    REPO_ROOT,
    ValidationError,
    _load_yaml,
    resolve_recipe,
    validate_paths,
)
from recipe_bundle import sha256, wheel_inventory
from recipe_publish import publish, validate
from recipe_runner import metrics_zero, read_eval_reviews
from recipe_select import select


def fast_result():
    return {
        "schema": "ppu-sanity/v1",
        "model": "glm-5.3",
        "tier": "fast_sanity",
        "source_sha": "a" * 40,
        "recipe_tree_sha256": "0" * 64,
        "config_digest": "c" * 64,
        "outcome": "pass",
        "runtime_image": "download.t-head.cn/docker_test/sglang@sha256:" + "d" * 64,
        "route": "native-build + sglang-release-runtime",
        "parser_checked": True,
        "sse_verified": True,
        "private_retention": True,
        "queue_s": 2,
        "cold_start_s": 3.0,
        "layer_override": 2,
    }


class RecipeContractTest(unittest.TestCase):
    def test_default_fast_selects_both_models(self):
        self.assertEqual(
            select("fast_sanity", ""),
            {
                "include": [
                    {"model": "deepseek-v4-flash"},
                    {"model": "glm-5.3"},
                ]
            },
        )

    def test_nightly_exact_filter(self):
        self.assertEqual(
            select("model_sanity", "glm-5.3"), {"include": [{"model": "glm-5.3"}]}
        )
        with self.assertRaises(ValueError):
            select("model_sanity", "glm")
        with self.assertRaises(ValueError):
            select("model_sanity", "glm-5.3,glm-5.3")

    def test_policy_switches(self):
        policy = DEFAULT_POLICY_ROOT / "models/glm-5.3.yaml"
        recipe = REPO_ROOT / _load_yaml(policy)["recipe"]
        resolved = resolve_recipe(recipe, policy_path=policy, tier="fast_sanity")
        self.assertEqual(
            resolved["argv"][:5], ["python3", "-I", "-B", "-m", "sglang.launch_server"]
        )
        self.assertEqual(resolved["layer_override"], 2)
        self.assertIn('{"num_hidden_layers":2}', resolved["flags"])
        self.assertIn("dummy", resolved["flags"])
        self.assertNotIn("--disable-radix-cache", resolved["flags"])
        fallback = resolve_recipe(
            recipe, policy_path=policy, tier="fast_sanity", layer_override=None
        )
        self.assertNotIn("--json-model-override-args", fallback["flags"])
        self.assertNotEqual(fallback["config_digest"], resolved["config_digest"])
        perf = resolve_recipe(recipe, cache="disabled")
        self.assertIn("--disable-radix-cache", perf["flags"])
        self.assertNotIn("--load-format", perf["flags"])

    def test_required_parser_fails_if_runtime_unavailable(self):
        with patch("ppu_recipe._build_sglang_serve_parser", side_effect=ImportError):
            with self.assertRaises(ValidationError):
                validate_paths(
                    [REPO_ROOT / "test/ppu/recipes/zai-org/GLM-5.3.yaml"],
                    sglang_parser_mode="required",
                )

    def test_bundle_rejects_modified_wheel_and_source(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            source = "a" * 40
            files = (
                "source-sha.txt",
                "sglang-0.5-py3-none-any.whl",
                "sglang_kernel-0.1-cp312-cp312-linux_x86_64.whl",
            )
            for name in files:
                (directory / name).write_bytes(b"fixture")
            (directory / files[0]).write_text(source + "\n")
            (directory / "SHA256SUMS").write_text(
                "".join(f"{sha256(directory / name)}  {name}\n" for name in files)
            )
            wheel_inventory(directory, source)
            with self.assertRaises(ValueError):
                wheel_inventory(directory, "b" * 40)
            (directory / files[1]).write_bytes(b"changed")
            with self.assertRaises(ValueError):
                wheel_inventory(directory, source)


class RunEvidenceTest(unittest.TestCase):
    def test_metrics_require_present_and_zero_for_all_series(self):
        names = ("sglang:num_running_reqs", "sglang:num_queue_reqs")
        first, second = names
        idle = f'{first}{{rank="0"}} 0.0\n{first}{{rank="1"}} 0.0\n{second} 0\n'
        self.assertTrue(metrics_zero(idle, names))
        self.assertFalse(metrics_zero(f"{first} 0.0", names))
        self.assertFalse(metrics_zero(idle + f'{first}{{rank="2"}} 1.0', names))
        self.assertFalse(
            metrics_zero(idle.replace(f"{second} 0", f"{second} NaN"), names)
        )

    def test_eval_rows_count_score_and_uniqueness(self):
        with tempfile.TemporaryDirectory() as root:
            summary = Path(root) / "reviews/model/gsm8k_main.jsonl"
            summary.parent.mkdir(parents=True)
            rows = [
                {
                    "sample_score": {
                        "sample_id": number,
                        "score": {"status": "success", "value": int(number < 1278)},
                    }
                }
                for number in range(1319)
            ]
            summary.write_text("".join(json.dumps(row) + "\n" for row in rows))
            self.assertEqual(read_eval_reviews(Path(root), 1319, 0.9689, 1319)[1], 1278)
            with self.assertRaises(ValueError):
                read_eval_reviews(Path(root), 1319, 0.96, 1319)
            rows[-1] = rows[0]
            summary.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                read_eval_reviews(Path(root), 1319, 0.9689, 1319)

    def test_pass_requires_complete_nightly_evidence(self):
        result = fast_result()
        result["tier"] = "model_sanity"
        result.pop("layer_override")
        result.update(
            eval_s=16,
            bench_s=1,
            gsm8k_count=1319,
            gsm8k_correct=1278,
            gsm8k_score=0.9689,
            golden_score=0.9689,
            accuracy_within_band=True,
            baseline_established=True,
            kv_gauge_zero_verified=True,
            c1_passed=True,
            c16_passed=True,
            cache_mode="sanity_throughput_guard",
        )
        validate(result)
        for key in ("baseline_established", "kv_gauge_zero_verified", "c16_passed"):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    validate({**result, key: False})

    def test_public_positive_control(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            artifact = root / "artifact.json"
            summary = root / "summary"
            console = io.StringIO()
            publish(fast_result(), artifact, summary, console)
            self.assertEqual(json.loads(console.getvalue()), fast_result())
            self.assertEqual(artifact.read_text(), console.getvalue())
            self.assertIn(console.getvalue(), summary.read_text())

    def test_three_channels_fail_closed_on_injected_path_and_ip(self):
        simulated = (
            "/" + "nas" + "_aisw/example-secret " + ".".join(("10", "1", "2", "3"))
        )
        for key in (
            "diagnostic",
            "failure_phase",
            "runtime_image",
            "model",
            "source_sha",
        ):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as root:
                root = Path(root)
                candidate, artifact, summary = (
                    root / name
                    for name in ("candidate.json", "artifact.json", "summary")
                )
                candidate.write_text(
                    json.dumps({**fast_result(), "outcome": "fail", key: simulated})
                )
                process = subprocess.run(
                    [
                        sys.executable,
                        "-B",
                        str(REPO_ROOT / "scripts/ci/ppu/recipe_publish.py"),
                        str(candidate),
                        str(artifact),
                        str(summary),
                    ],
                    capture_output=True,
                    text=True,
                    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                )
                self.assertEqual(process.returncode, 1)
                self.assertEqual(process.stdout, "")
                for secret in simulated.split():
                    self.assertNotIn(secret, process.stderr)
                self.assertFalse(artifact.exists())
                self.assertFalse(summary.exists())


if __name__ == "__main__":
    unittest.main()
