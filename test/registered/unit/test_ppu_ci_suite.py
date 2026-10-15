import fnmatch
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from sglang.test.ci.ci_register import HWBackend, collect_tests, register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="stage-a-test-cpu-ppu")

RUNNER = Path(__file__).resolve().parents[2] / "run_suite.py"


class TestPPUCISuite(CustomTestCase):
    def test_cpu_checks_are_excluded_from_the_general_cpu_suite(self):
        files = list(Path(__file__).parent.glob("test_ppu_ci_*.py"))
        tests = collect_tests([str(path) for path in files])
        spec = importlib.util.spec_from_file_location("ppu_run_suite", RUNNER)
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        runner.validate_all_suites(tests)
        general, _ = runner.filter_tests(tests, HWBackend.CPU, ["base-a-test-cpu"])
        dedicated, skipped = runner.filter_tests(
            tests, HWBackend.CPU, ["stage-a-test-cpu-ppu"]
        )
        self.assertEqual(general, [])
        self.assertEqual(skipped, [])
        self.assertCountEqual(
            [Path(test.filename).name for test in dedicated],
            [
                "test_ppu_ci_dependencies.py",
                "test_ppu_ci_docker.py",
                "test_ppu_ci_suite.py",
            ],
        )

    def test_ppu_suite_contains_both_speculative_sanity_files(self):
        registered = RUNNER.parent / "registered"
        files = [
            str(path)
            for path in registered.rglob("*.py")
            if path.name not in ("conftest.py", "__init__.py")
        ]
        tests = collect_tests(files)
        ppu_tests = [
            test
            for test in tests
            if test.backend == HWBackend.PPU
            and test.effective_suite == "stage-a-test-1-gpu-ppu"
            and not test.disabled
        ]
        self.assertCountEqual(
            [
                Path(test.filename).relative_to(registered).as_posix()
                for test in ppu_tests
            ],
            ["core/test_basic_sanity_eagle3.py", "core/test_basic_sanity_dflash.py"],
        )

    def test_empty_ppu_suite_fails(self):
        for disabled in (False, True):
            with self.subTest(disabled=disabled), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                runner = root / "run_suite.py"
                runner.write_text(RUNNER.read_text())
                if disabled:
                    registered = root / "registered"
                    registered.mkdir()
                    (registered / "test_disabled.py").write_text(
                        'register_ppu_ci(est_time=1, suite="stage-a-test-1-gpu-ppu", '
                        'disabled="test fixture")\n'
                    )
                result = subprocess.run(
                    [
                        sys.executable,
                        str(runner),
                        "--hw",
                        "ppu",
                        "--suite",
                        "stage-a-test-1-gpu-ppu",
                    ],
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn(
                    "PPU suite requires at least one enabled test", result.stderr
                )


class TestPPUCIWorkflow(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        path = RUNNER.parent.parent / ".github/workflows/pr-test-ppu.yml"
        cls.jobs = yaml.safe_load(path.read_text())["jobs"]

    def test_shared_ci_changes_do_not_require_ppu_labels(self):
        """Shared CI edits must not opt unrelated PRs into the PPU label gate."""
        step = next(
            step
            for step in self.jobs["check-changes"]["steps"]
            if step.get("id") == "filter"
        )
        patterns = yaml.safe_load(step["with"]["filters"])["ppu"]
        for path, expected in (
            ("python/sglang/test/ci/ci_register.py", False),
            ("python/sglang/test/ci/ci_utils.py", False),
            ("test/run_suite.py", False),
            (".github/workflows/pr-test-ppu.yml", True),
            ("scripts/ci/ppu/run_test.sh", True),
            ("python/sglang/srt/platforms/sail.py", True),
            ("test/registered/unit/test_ppu_ci_suite.py", True),
        ):
            with self.subTest(path=path):
                self.assertEqual(
                    any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns),
                    expected,
                )

    def test_selftest_gates_device_jobs_without_waiting_for_wheels(self):
        """CPU checks must complete before allocating a device, not after a build."""
        selftest = self.jobs["ppu-ci-selftest"]
        self.assertEqual(selftest["runs-on"], "ubuntu-latest")
        self.assertCountEqual(selftest["needs"], ["check-changes", "pr-gate"])
        self.assertCountEqual(
            self.jobs["ppu-build"]["needs"], ["check-changes", "pr-gate"]
        )
        self.assertEqual(selftest["if"], "needs.check-changes.outputs.ppu == 'true'")
        for job in ("ppu-sanity", "pr-test-ppu-finish"):
            self.assertIn("ppu-ci-selftest", self.jobs[job]["needs"])
        self.assertEqual(
            self.jobs["ppu-sanity"]["if"], "needs.check-changes.outputs.ppu == 'true'"
        )
        self.assertEqual(
            self.jobs["pr-test-ppu-finish"]["if"],
            "always() && needs.check-changes.result != 'skipped'",
        )
        checkout = next(
            step
            for step in selftest["steps"]
            if step.get("uses", "").startswith("actions/checkout@")
        )
        self.assertEqual(
            checkout["with"]["ref"], "${{ needs.check-changes.outputs.source-sha }}"
        )
        self.assertIn("ignored-label-ppu-ci-selftest", selftest["name"])
        runtime = RUNNER.parent.parent / "scripts/ci/ppu/build_and_test.sh"
        self.assertNotIn("--hw cpu", runtime.read_text())

    def test_selftest_cache_preserves_wheels_and_tracks_dependencies(self):
        """Default uv cache pruning discards the downloaded CPU wheels."""
        steps = self.jobs["ppu-ci-selftest"]["steps"]
        caches = [
            step
            for step in steps
            if step.get("uses", "").startswith("astral-sh/setup-uv@")
        ]
        self.assertEqual(len(caches), 1)
        cache = caches[0]["with"]
        self.assertIs(cache["enable-cache"], True)
        self.assertIs(cache["prune-cache"], False)
        python = next(
            step
            for step in steps
            if step.get("uses", "").startswith("actions/setup-python@")
        )
        self.assertEqual(cache["python-version"], python["with"]["python-version"])
        self.assertTrue(
            any(
                fnmatch.fnmatchcase(
                    ".github/workflows/pr-test-ppu.yml", pattern.strip()
                )
                for pattern in cache["cache-dependency-glob"].splitlines()
            )
        )

    def test_finish_rejects_unsuccessful_selftest_including_skips(self):
        """A skipped prerequisite must not produce a successful PPU finish check."""
        finish = self.jobs["pr-test-ppu-finish"]["steps"][0]
        for status in ("success", "failure", "cancelled", "skipped"):
            for in_scope in (True, False):
                with self.subTest(status=status, in_scope=in_scope):
                    needs = {
                        name: {"result": "success"}
                        for name in self.jobs["pr-test-ppu-finish"]["needs"]
                    }
                    needs["ppu-ci-selftest"] = {"result": status}
                    result = subprocess.run(
                        ["bash", "-e", "-o", "pipefail", "-c", finish["run"]],
                        env=dict(
                            os.environ,
                            NEEDS_JSON=json.dumps(needs),
                            PPU_IN_SCOPE=str(in_scope).lower(),
                            SELFTEST_RESULT=status,
                            BUILD_RESULT="success",
                            SANITY_RESULT="success",
                        ),
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    expected = status == "success" or (
                        not in_scope and status == "skipped"
                    )
                    self.assertEqual(
                        result.returncode == 0, expected, result.stdout + result.stderr
                    )
        self.assertEqual(
            finish["env"]["SELFTEST_RESULT"], "${{ needs.ppu-ci-selftest.result }}"
        )

    def test_selftest_rejects_empty_suite_and_preserves_runner_failure(self):
        """An empty selection must not pass before its own collection checks run."""
        script = self.jobs["ppu-ci-selftest"]["steps"][-1]["run"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            python = root / "python3"
            python.write_text(
                '#!/bin/sh\nif [ "$1" != "-c" ]; then\n'
                '  printf "Test Summary: %s passed\\n" "$FIXTURE_SUMMARY"\n'
                '  exit "$FIXTURE_EXIT"\nfi\n'
            )
            python.chmod(0o755)
            for summary, exit_code, expected in (
                ("0/0", 0, False),
                ("2/3", 0, False),
                ("3/3", 0, True),
                ("3/3", 7, False),
            ):
                with self.subTest(summary=summary, exit_code=exit_code):
                    result = subprocess.run(
                        ["bash", "-e", "-o", "pipefail", "-c", script],
                        env=dict(
                            os.environ,
                            PATH=f"{root}{os.pathsep}{os.environ['PATH']}",
                            RUNNER_TEMP=str(root),
                            FIXTURE_SUMMARY=summary,
                            FIXTURE_EXIT=str(exit_code),
                        ),
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    self.assertEqual(result.returncode == 0, expected, result.stdout)

    def test_missing_flock_fails_before_selftests_can_skip(self):
        """Missing launcher prerequisites must fail the job rather than skip its tests."""
        prerequisite = next(
            step
            for step in self.jobs["ppu-ci-selftest"]["steps"]
            if step.get("id") == "prerequisites"
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            uname = root / "uname"
            uname.write_text("#!/bin/sh\nprintf 'Linux\\n'\n")
            uname.chmod(0o755)
            result = subprocess.run(
                [
                    shutil.which("bash"),
                    "-e",
                    "-o",
                    "pipefail",
                    "-c",
                    prerequisite["run"],
                ],
                env=dict(os.environ, PATH=str(root)),
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertNotEqual(result.returncode, 0, result.stdout)


class TestPPURuntimeCache(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        path = RUNNER.parent.parent / "scripts/ci/ppu/prepare_runtime.py"
        spec = importlib.util.spec_from_file_location("ppu_prepare_runtime", path)
        cls.prepare = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.prepare)

    def test_hub_ref_resolves_offline_without_writing_checkpoint(self):
        """A newline in refs/main makes offline hub lookups miss the snapshot."""
        from huggingface_hub import hf_hub_download

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "config.json").write_text('{"model_type": "llama"}')
            hub = root / "hub"
            self.prepare.prepare_model_cache(hub, {"test/model": checkpoint})
            resolved = hf_hub_download(
                "test/model", "config.json", cache_dir=hub, local_files_only=True
            )
            self.assertEqual(Path(resolved).resolve(), checkpoint / "config.json")
            self.assertEqual(list(checkpoint.iterdir()), [checkpoint / "config.json"])

    def test_eval_cache_copy_is_independent_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            (source / "mmlu").mkdir(parents=True)
            data = source / "mmlu/test.jsonl"
            marker = source / "mmlu/test.jsonl.sha256"
            content = b'{"question": "fixture"}\n'
            digest = hashlib.sha256(content).hexdigest()
            data.write_bytes(content)
            marker.write_text(digest + "\n")
            with patch.object(self.prepare, "MMLU_SHA256", digest):
                destination = root / "private"
                self.prepare.prepare_mmlu_cache(source, destination)
                copied = destination / "mmlu/test.jsonl"
                self.assertFalse(copied.is_symlink())
                self.assertFalse(copied.samefile(data))
                self.assertEqual(copied.read_bytes(), content)
                copied.write_text("private cache update")
                self.assertEqual(data.read_bytes(), content)
                for corrupt in (data, marker):
                    with self.subTest(corrupt=corrupt.name):
                        original = corrupt.read_bytes()
                        corrupt.write_text("corrupted")
                        with self.assertRaisesRegex(ValueError, "MMLU"):
                            self.prepare.prepare_mmlu_cache(source, root / "rejected")
                        self.assertFalse((root / "rejected").exists())
                        corrupt.write_bytes(original)
                marker.unlink()
                with self.assertRaises(FileNotFoundError):
                    self.prepare.prepare_mmlu_cache(source, root / "missing")


class TestPPUBuildProvenance(CustomTestCase):
    def test_build_rejects_wrong_revision_and_uncommitted_inputs(self):
        """HEAD alone cannot identify a build that also consumes uncommitted files."""
        source = RUNNER.parent.parent / "scripts/ci/ppu/build_and_test.sh"
        for state in ("revision", "tracked", "untracked", "ignored"):
            with self.subTest(state=state), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                repo = root / "repo"
                script = repo / "scripts/ci/ppu/build_and_test.sh"
                script.parent.mkdir(parents=True)
                script.write_bytes(source.read_bytes())
                (repo / ".gitignore").write_text("ignored\n")
                (repo / "tracked").write_text("committed")
                wheels = root / "wheels"
                wheels.mkdir()
                env = dict(
                    os.environ,
                    GIT_CONFIG_GLOBAL=os.devnull,
                    GIT_CONFIG_NOSYSTEM="1",
                    GIT_AUTHOR_NAME="Fixture",
                    GIT_AUTHOR_EMAIL="fixture@example.invalid",
                    GIT_COMMITTER_NAME="Fixture",
                    GIT_COMMITTER_EMAIL="fixture@example.invalid",
                    PPU_CI_ARTIFACT_DIR=str(root / "artifacts"),
                    PPU_CI_WHEEL_DIR=str(wheels),
                    PPU_SDK=str(root / "missing-sdk"),
                )
                for args in (
                    ("init", "-q"),
                    ("add", "."),
                    ("commit", "-qm", "fixture"),
                ):
                    subprocess.run(["git", "-C", str(repo), *args], env=env, check=True)
                sha = subprocess.check_output(
                    ["git", "-C", str(repo), "rev-parse", "HEAD"], env=env, text=True
                ).strip()
                env["PPU_CI_SOURCE_SHA"] = "0" * 40 if state == "revision" else sha
                if state != "revision":
                    (repo / state).write_text("uncommitted")
                result = subprocess.run(
                    ["bash", str(script), "build"],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Source checkout", result.stdout + result.stderr)

    def test_runtime_rejects_unlisted_wheel_before_installation(self):
        """A valid manifest must not leave additional installable wheels unchecked."""
        script = RUNNER.parent.parent / "scripts/ci/ppu/build_and_test.sh"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wheels = root / "wheels"
            (wheels / "runtime-deps").mkdir(parents=True)
            files = ("source-sha.txt", "./sglang.whl", "./runtime-deps/sailify.whl")
            manifest = []
            for name in files:
                content = b"fixture"
                (wheels / name).write_bytes(content)
                manifest.append(f"{hashlib.sha256(content).hexdigest()}  {name}\n")
            (wheels / "SHA256SUMS").write_text("".join(manifest))
            (wheels / "unchecked.whl").write_text("not covered by the manifest")
            result = subprocess.run(
                ["bash", str(script), "test"],
                env=dict(
                    os.environ,
                    PPU_CI_SOURCE_SHA="fixture",
                    PPU_CI_ARTIFACT_DIR=str(root / "artifacts"),
                    PPU_CI_WHEEL_DIR=str(wheels),
                ),
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Wheel manifest", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
