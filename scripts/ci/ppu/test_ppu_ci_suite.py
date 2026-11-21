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

RUNNER = Path(__file__).resolve().parents[3] / "test/run_suite.py"
REGISTRY_MODULE = "sglang.test.ci.ci_register"
registry_path = RUNNER.parent.parent / "python/sglang/test/ci/ci_register.py"
registry_spec = importlib.util.spec_from_file_location(REGISTRY_MODULE, registry_path)
registry = importlib.util.module_from_spec(registry_spec)
# The infrastructure tests need the real registration parser but no inference runtime.
with patch.dict(sys.modules, {REGISTRY_MODULE: registry}):
    registry_spec.loader.exec_module(registry)
    from run_speculative_sanity import select_tests


class TestPPUCISuite(unittest.TestCase):
    def test_manual_checks_are_not_ci_registered(self):
        files = list(Path(__file__).parent.glob("test_ppu_ci_*.py"))
        self.assertCountEqual(
            [path.name for path in files],
            [
                "test_ppu_ci_dependencies.py",
                "test_ppu_ci_docker.py",
                "test_ppu_ci_suite.py",
            ],
        )
        for path in files:
            with self.subTest(path=path.name):
                self.assertEqual(registry.ut_parse_one_file(str(path)), ([], True))
        self.assertEqual(
            list((RUNNER.parent / "registered").rglob("test_ppu_ci_*.py")), []
        )

    def test_ppu_suite_contains_both_speculative_sanity_files(self):
        registered = RUNNER.parent / "registered"
        ppu_tests = select_tests(RUNNER.parent.parent)
        self.assertCountEqual(
            [
                Path(test.filename).relative_to(registered).as_posix()
                for test in ppu_tests
            ],
            ["core/test_basic_sanity_eagle3.py", "core/test_basic_sanity_dflash.py"],
        )

    def test_runtime_requires_both_sanity_files_to_pass(self):
        source = Path(__file__).with_name("build_and_test.sh").read_text()
        marker = 'cd "$artifacts"\n'
        self.assertIn(marker, source)
        script = (
            "\n".join(source.splitlines()[:2])
            + "\n"
            + marker
            + source.split(marker, 1)[1]
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo with spaces"
            (root / "scripts/ci/ppu").mkdir(parents=True)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            (root / "scripts/ci/ppu/run_speculative_sanity.py").write_text(
                "import os, sys\n"
                "print(os.environ['FIXTURE_SUMMARY'], file=sys.stderr)\n"
                "sys.exit(int(os.environ['FIXTURE_EXIT']))\n"
            )
            for summary, exit_code, expected in (
                ("Test Summary: 2/2 passed", 0, 0),
                ("Test Summary: 0/0 passed", 0, 1),
                ("Test Summary: 1/1 passed", 0, 1),
                ("Test Summary: 1/2 passed", 0, 1),
                ("Test Summary: 3/3 passed", 0, 1),
                ("Test Summary: 2/2 passed", 7, 7),
                ("Test Summary: 2/2 passed (incomplete)", 0, 1),
                ("", 0, 1),
            ):
                with self.subTest(summary=summary, exit_code=exit_code):
                    result = subprocess.run(
                        ["bash", "-c", script],
                        env=dict(
                            os.environ,
                            repo=str(root),
                            artifacts=str(artifacts),
                            FIXTURE_SUMMARY=summary,
                            FIXTURE_EXIT=str(exit_code),
                        ),
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    self.assertEqual(
                        result.returncode, expected, result.stdout + result.stderr
                    )


class TestPPUCIWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = RUNNER.parent.parent / ".github/workflows/pr-test-ppu.yml"
        cls.workflow = yaml.safe_load(path.read_text())
        cls.jobs = cls.workflow["jobs"]

    def test_runtime_image_is_only_configured_by_runner_environment(self):
        path = RUNNER.parent.parent / ".github/workflows/pr-test-ppu.yml"
        workflow = yaml.load(path.read_text(), Loader=yaml.BaseLoader)
        for event in ("workflow_dispatch", "workflow_call"):
            self.assertNotIn("runtime_image", workflow["on"][event]["inputs"])
        self.assertNotIn("PPU_CI_RUNTIME_IMAGE", path.read_text())

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
            ("test/registered/core/test_basic_sanity_eagle3.py", False),
            ("test/registered/core/test_basic_sanity_dflash.py", False),
            (".github/workflows/pr-test-ppu.yml", True),
            ("scripts/ci/ppu/run_test.sh", True),
            ("python/sglang/srt/platforms/sail.py", True),
            ("scripts/ci/ppu/test_ppu_ci_suite.py", True),
        ):
            with self.subTest(path=path):
                self.assertEqual(
                    any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns),
                    expected,
                )

    def test_release_code_and_explicit_label_are_in_scope(self):
        check = self.jobs["check-changes"]
        step = next(step for step in check["steps"] if step.get("id") == "filter")
        patterns = yaml.safe_load(step["with"]["filters"])["release-code"]
        for path in (
            "python/sglang/srt/managers/scheduler.py",
            "python/sglang/kernels/aot/csrc/elementwise/topk.cu",
            "sgl-kernel/csrc/elementwise/example.cu",
        ):
            with self.subTest(path=path):
                self.assertTrue(
                    any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)
                )
        output = check["outputs"]["ppu"]
        self.assertIn("steps.filter.outputs.release-code == 'true'", output)
        self.assertIn("startsWith(github.base_ref, 'release/')", output)
        self.assertIn("needs.pr-gate.outputs.approved == 'true'", output)

    def test_cluster_jobs_use_fresh_approval_and_thead_cpu_runner(self):
        for name, cluster_jobs in (
            ("pr-test-ppu", ("ppu-cluster-config", "ppu-legacy-sanity")),
            ("nightly-test-ppu", ("cluster-preflight", "recipe-sanity")),
        ):
            with self.subTest(workflow=name):
                source = RUNNER.parent.parent / f".github/workflows/{name}.yml"
                workflow = yaml.load(source.read_text(), Loader=yaml.BaseLoader)
                self.assertIn("pull_request_target", workflow["on"])
                self.assertNotIn("pull_request", workflow["on"])
                jobs = workflow["jobs"]
                self.assertEqual(
                    jobs["pr-gate"]["uses"], "./.github/workflows/ppu-approval.yml"
                )
                for job in cluster_jobs:
                    self.assertEqual(jobs[job]["runs-on"], "k8s-runner-group-cpu-thead")

    def test_device_jobs_require_build_and_cpu_checks(self):
        selftests = self.jobs["ppu-ci-selftest"]
        self.assertEqual(selftests["runs-on"], "ubuntu-latest")
        self.assertCountEqual(selftests["needs"], ["check-changes", "pr-gate"])
        self.assertIn(
            "bash scripts/ci/ppu/run_selftests.sh", selftests["steps"][-1]["run"]
        )
        for job in ("ppu-build", "ppu-cluster-config"):
            self.assertCountEqual(
                self.jobs[job]["needs"], ["check-changes", "ppu-ci-selftest"]
            )
        self.assertCountEqual(
            self.jobs["ppu-legacy-sanity"]["needs"],
            ["check-changes", "pr-gate", "ppu-build", "ppu-cluster-config"],
        )
        self.assertCountEqual(
            self.jobs["pr-test-ppu-finish"]["needs"],
            [
                "pr-gate",
                "check-changes",
                "ppu-ci-selftest",
                "ppu-build",
                "ppu-cluster-config",
                "ppu-fast-sanity",
            ],
        )
        self.assertEqual(
            self.jobs["ppu-legacy-sanity"]["if"],
            "inputs.run_legacy_sanity == true && needs.check-changes.outputs.ppu == 'true'",
        )
        self.assertIn(
            "needs.pr-gate.result == 'failure'",
            self.jobs["pr-test-ppu-finish"]["if"],
        )
        runtime = RUNNER.parent.parent / "scripts/ci/ppu/build_and_test.sh"
        self.assertNotIn("--hw cpu", runtime.read_text())

    def test_jobs_check_out_the_triggering_ref_and_verify_the_pinned_sha(self):
        check = self.jobs["check-changes"]
        self.assertIn("github.sha", check["steps"][0]["with"]["ref"])
        self.assertIn(
            "github.event.pull_request.head.sha", check["outputs"]["source-sha"]
        )
        for name in ("ppu-build", "ppu-legacy-sanity"):
            with self.subTest(job=name):
                job = self.jobs[name]
                checkout = next(
                    step
                    for step in job["steps"]
                    if step.get("uses", "").startswith("actions/checkout@")
                )
                self.assertEqual(
                    checkout["with"]["ref"],
                    "${{ needs.check-changes.outputs.source-sha }}",
                )
                self.assertEqual(
                    job["env"]["PPU_CI_SOURCE_SHA"],
                    "${{ needs.check-changes.outputs.source-sha }}",
                )

    def test_sanity_runs_on_every_ppu_generation_with_private_retention(self):
        sanity = self.jobs["ppu-legacy-sanity"]
        self.assertEqual(sanity["runs-on"], "k8s-runner-group-cpu-thead")
        self.assertIs(sanity["strategy"]["fail-fast"], False)
        self.assertEqual(
            sanity["strategy"]["matrix"]["board"], ["ZW-M890P", "OAM-810E"]
        )
        self.assertIn("matrix.board", sanity["name"])
        uploads = [
            step
            for step in sanity["steps"]
            if step.get("uses", "").startswith("actions/upload-artifact@")
        ]
        self.assertEqual(len(uploads), 1)
        self.assertEqual(uploads[0]["with"]["path"], "ppu-public/legacy.json")
        self.assertEqual(
            uploads[0]["if"], "always() && steps.evidence.outputs.approved == 'true'"
        )
        steps = {step["id"]: step for step in sanity["steps"] if "id" in step}
        self.assertIn(
            "recipe_publish.py --verify-only ppu-public/legacy.json",
            steps["evidence"]["run"],
        )
        self.assertIn(
            "exec bash scripts/ci/ppu/run_recipe_cluster.sh", steps["cluster"]["run"]
        )

    def test_legacy_mode_is_explicit_and_separate_from_recipe_ci(self):
        path = RUNNER.parent.parent / ".github/workflows/pr-test-ppu.yml"
        workflow = yaml.load(path.read_text(), Loader=yaml.BaseLoader)
        for event in ("workflow_dispatch", "workflow_call"):
            self.assertEqual(
                workflow["on"][event]["inputs"]["run_legacy_sanity"]["default"], "false"
            )
        self.assertIn("inputs.run_legacy_sanity", workflow["concurrency"]["group"])
        self.assertIn(
            "inputs.run_legacy_sanity", self.jobs["check-changes"]["outputs"]["ppu"]
        )
        self.assertIn(
            "inputs.run_legacy_sanity != true", self.jobs["ppu-fast-sanity"]["if"]
        )
        self.assertIn(
            "legacy-only-pr-test-ppu-finish", self.jobs["pr-test-ppu-finish"]["name"]
        )

    def test_finish_requires_each_recipe_dependency_when_in_scope(self):
        job = self.jobs["pr-test-ppu-finish"]
        finish = job["steps"][0]
        for dependency in job["needs"]:
            for status in ("success", "failure", "cancelled", "skipped"):
                for in_scope in (True, False):
                    with self.subTest(
                        dependency=dependency, status=status, in_scope=in_scope
                    ):
                        needs = {name: {"result": "success"} for name in job["needs"]}
                        needs[dependency] = {"result": status}
                        result = subprocess.run(
                            ["bash", "-e", "-o", "pipefail", "-c", finish["run"]],
                            env=dict(
                                os.environ,
                                NEEDS_JSON=json.dumps(needs),
                                PREFLIGHT_ONLY="false",
                                SANITY_RESULT="success",
                                PPU_IN_SCOPE=str(in_scope).lower(),
                            ),
                            capture_output=True,
                            text=True,
                            timeout=10,
                        )
                        allowed = {"success"} if in_scope else {"success", "skipped"}
                        self.assertEqual(
                            result.returncode == 0,
                            status in allowed,
                            result.stdout + result.stderr,
                        )


class TestPPUManualSelftests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repo = self.root / "repo with spaces"
        directory = self.repo / "scripts/ci/ppu"
        directory.mkdir(parents=True)
        self.script = directory / "run_selftests.sh"
        source = Path(__file__).with_name("run_selftests.sh")
        self.assertTrue(source.is_file(), "Manual selftest entry point is missing")
        self.script.write_bytes(source.read_bytes())
        self.files = [
            "test_ppu_ci_dependencies.py",
            "test_ppu_ci_docker.py",
            "test_ppu_ci_suite.py",
        ]
        self.result_path = self.root / "executed.jsonl"
        self.env = dict(
            os.environ,
            FIXTURE_RESULTS=str(self.result_path),
            FIXTURE_FAILURE="",
            SGLANG_IS_IN_CI="1",
            SGLANG_TEST_MAX_RETRY="9",
            PYTHONDONTWRITEBYTECODE="0",
            HF_HUB_OFFLINE="0",
            PYTHONPATH=str(self.root / "installed"),
        )
        for folder, value in (
            (self.repo / "python", "source"),
            (self.root / "installed", "installed"),
        ):
            folder.mkdir()
            (folder / "selftest_fixture.py").write_text(f"ORIGIN = {value!r}\n")
        for name in self.files:
            (directory / name).write_text(
                "import json, os, unittest\n"
                "from pathlib import Path\n"
                "from selftest_fixture import ORIGIN\n"
                "class Check(unittest.TestCase):\n"
                "    def test_environment(self):\n"
                "        with open(os.environ['FIXTURE_RESULTS'], 'a') as output:\n"
                "            output.write(json.dumps(Path(__file__).name) + '\\n')\n"
                "        self.assertEqual(ORIGIN, 'source')\n"
                "        for key, value in {'SGLANG_IS_IN_CI': '0', 'SGLANG_TEST_MAX_RETRY': '0',\n"
                "                           'HF_HUB_OFFLINE': '1',\n"
                "                           'PYTHONDONTWRITEBYTECODE': '1'}.items():\n"
                "            self.assertEqual(os.environ[key], value)\n"
                "        self.assertNotEqual(Path(__file__).name, os.environ['FIXTURE_FAILURE'])\n"
                "unittest.main()\n"
            )

    def run_entry(self, env=None):
        return subprocess.run(
            [shutil.which("bash"), str(self.script)],
            cwd=self.root,
            env=self.env if env is None else env,
            capture_output=True,
            text=True,
            timeout=15,
        )

    def test_runs_all_files_from_unrelated_working_directory(self):
        result = self.run_entry()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(
            [json.loads(line) for line in self.result_path.read_text().splitlines()],
            self.files,
        )
        self.assertEqual(result.stderr.count("Ran 1 test"), 3)
        self.assertEqual(list(self.repo.rglob("__pycache__")), [])

    def test_stops_and_propagates_test_failure_or_missing_file(self):
        target = self.script.parent / self.files[1]
        original = target.read_bytes()
        for failure, code in (("assertion", 1), ("exit", 7), ("missing", 2)):
            with self.subTest(failure=failure):
                target.write_bytes(original)
                self.result_path.write_text("")
                self.env["FIXTURE_FAILURE"] = target.name
                if failure == "exit":
                    target.write_text("import sys\nsys.exit(7)\n")
                elif failure == "missing":
                    target.unlink()
                result = self.run_entry()
                self.assertEqual(result.returncode, code, result.stdout + result.stderr)
                executed = self.result_path.read_text()
                self.assertIn(self.files[0], executed)
                self.assertNotIn(self.files[2], executed)

    def test_missing_prerequisites_fail_before_running_tests(self):
        tools = self.root / "tools"
        tools.mkdir()
        uname = tools / "uname"
        uname.write_text("#!/bin/sh\nprintf '%s\\n' Linux\n")
        uname.chmod(0o755)
        commands = ("flock", "git", "jq", "python3")
        for name in commands:
            (tools / name).symlink_to(shutil.which(name))
        for missing in ("Linux", *commands):
            with self.subTest(missing=missing):
                if missing == "Linux":
                    uname.write_text("#!/bin/sh\nprintf '%s\\n' Darwin\n")
                else:
                    (tools / missing).unlink()
                result = self.run_entry(dict(self.env, PATH=str(tools)))
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn(missing, result.stderr)
                self.assertFalse(self.result_path.exists())
                if missing == "Linux":
                    uname.write_text("#!/bin/sh\nprintf '%s\\n' Linux\n")
                else:
                    (tools / missing).symlink_to(shutil.which(missing))


class TestPPURuntimeCache(unittest.TestCase):
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


class TestPPUBuildProvenance(unittest.TestCase):
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
