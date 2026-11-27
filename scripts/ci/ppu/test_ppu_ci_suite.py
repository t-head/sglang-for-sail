import contextlib
import fnmatch
import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

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

    def test_protected_recipe_job_forwards_matrix_device_count(self):
        source = RUNNER.parent.parent / ".github/workflows/nightly-test-ppu.yml"
        workflow = yaml.load(source.read_text(), Loader=yaml.BaseLoader)
        job = workflow["jobs"]["recipe-sanity"]
        self.assertEqual(job["env"]["RECIPE_PPU_COUNT"], "${{ matrix.ppu_count || 8 }}")

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

    def test_fast_model_filter_is_available_only_for_manual_dispatch(self):
        path = RUNNER.parent.parent / ".github/workflows/pr-test-ppu.yml"
        workflow = yaml.load(path.read_text(), Loader=yaml.BaseLoader)
        triggers = workflow["on"]
        dispatch_filter = triggers["workflow_dispatch"]["inputs"]["model_filter"]
        self.assertEqual(dispatch_filter["type"], "string")
        self.assertEqual(dispatch_filter["default"], "")
        self.assertNotIn("model_filter", triggers["workflow_call"]["inputs"])
        self.assertEqual(
            self.jobs["ppu-fast-sanity"]["with"]["model_filter"],
            "${{ github.event_name == 'workflow_dispatch' && inputs.model_filter || '' }}",
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


def load_pd_fixture(path):
    """Load infrastructure/test code without importing the inference runtime."""
    spec = importlib.util.spec_from_file_location("ppu_fixture_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestPPUPDEnvironment(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        with patch.dict(sys.modules, {"sglang": None}):
            self.checker = load_pd_fixture(
                Path(__file__).with_name("check_ppu_environment.py")
            )
            self.pd = load_pd_fixture(Path(__file__).with_name("ppu_pd.py"))
        self.config = self.pd.load_pd_config()

    def run_main(self, args=(), env=None, failure=None, config=None):
        values = {
            "print_host_summary": None,
            "load_pd_config": self.config if config is None else config,
            "check_ppu_inventory": 16,
            "check_device_nodes": None,
            "check_ppu_sdk": None,
            "check_torch_compute": 2,
            "check_visibility_consistency": None,
            "check_memory_limit": 900 * 1024**3,
            "check_model": 2,
            "check_rdma": ["mlx5_0"],
            "check_pd_devices": None,
            "check_pd_runtime": {"code_source": "installed"},
        }
        checks = Mock()
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(sys, "argv", ["checker", *args]))
            stack.enter_context(patch.object(sys, "path", list(sys.path)))
            stack.enter_context(patch.dict(os.environ, env or {}, clear=True))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            for name, value in values.items():
                module = self.pd if name == "load_pd_config" else self.checker
                mock = stack.enter_context(
                    patch.object(module, name, return_value=value)
                )
                checks.attach_mock(mock, name)
                if name == failure:
                    mock.side_effect = self.checker.CheckFailure("fixture failure")
            # An accidental SGLang import is a regression even if no PD helper runs.
            stack.enter_context(
                patch.dict(sys.modules, {"sglang": None, "ppu_pd": self.pd})
            )
            self.last_checks = checks
            result = self.checker.main()
        return result, checks

    def test_default_checker_keeps_original_sequence_and_ignores_pd_environment(self):
        result, checks = self.run_main(
            env={
                "PD_TEST_CODE_SOURCE": "invalid",
                "PD_TEST_DUMMY_WEIGHTS": "1",
                "MODEL_PATH": "/missing",
                "RECIPE_PPU_COUNT": "2",
            },
            failure="load_pd_config",
        )
        self.assertEqual(result, 0)
        self.assertEqual(
            checks.mock_calls,
            [
                call.print_host_summary(),
                call.check_ppu_inventory(),
                call.check_device_nodes(),
                call.check_ppu_sdk(),
                call.check_torch_compute(),
                call.check_visibility_consistency(16, 2),
            ],
        )

    def test_pd_opt_in_sequence_and_runtime_artifact(self):
        env = {
            "MODEL_PATH": str(self.root / "model"),
            "IB_DEVICE": "mlx5_0",
            "NCCL_IB_GID_INDEX": "3",
            "PD_TEST_ARTIFACT_DIR": str(self.root),
        }
        result, checks = self.run_main(["--pd-case", "deepseek-v41-flash-1p1d"], env)
        self.assertEqual(result, 0)
        self.assertEqual(
            checks.mock_calls,
            [
                call.print_host_summary(),
                call.load_pd_config(),
                call.check_memory_limit(900 * 1024**3),
                call.check_model(self.root / "model"),
                call.check_rdma("mlx5_0", 3),
                call.check_ppu_inventory(),
                call.check_device_nodes(),
                call.check_ppu_sdk(),
                call.check_torch_compute(),
                call.check_visibility_consistency(16, 2),
                call.check_pd_devices(8),
                call.check_pd_runtime(),
            ],
        )
        self.assertEqual(
            json.loads((self.root / "runtime.json").read_text()),
            {
                "case": "deepseek-v41-flash-1p1d",
                "code_source": "installed",
                "config_digest": self.pd.config_digest(self.config),
                "memory_limit_bytes": 900 * 1024**3,
                "weight_shards": 2,
                "rdma_devices": ["mlx5_0"],
            },
        )

    def test_pd_checker_uses_configured_memory_devices_and_rdma_defaults(self):
        config = deepcopy(self.config)
        config["nodes"]["worker"]["devices"] = 4
        config["ppu"]["topology"]["devices_per_node"] = 4
        config["roles"]["prefill"]["devices"] = [0, 2]
        config["roles"]["decode"]["devices"] = [1, 3]
        config["environment"].update(
            memory_limit_gib=12, ib_device="mlx5_fixture", gid_index=2
        )
        result, checks = self.run_main(
            ["--pd-case", config["case_id"], "--pd-config", str(self.root / "case.yaml")],
            {"MODEL_PATH": str(self.root), "PD_TEST_ARTIFACT_DIR": str(self.root)},
            config=config,
        )
        self.assertEqual(result, 0)
        checks.load_pd_config.assert_called_once_with(self.root / "case.yaml")
        checks.check_memory_limit.assert_called_once_with(12 * 1024**3)
        checks.check_pd_devices.assert_called_once_with(4)
        checks.check_rdma.assert_called_once_with("mlx5_fixture", 2)
        runtime = json.loads((self.root / "runtime.json").read_text())
        self.assertEqual(runtime["config_digest"], self.pd.config_digest(config))
        self.assertNotEqual(runtime["config_digest"], self.pd.config_digest(self.config))

    def test_pd_preflight_failure_stops_without_artifact(self):
        for failure in (
            "load_pd_config",
            "check_memory_limit",
            "check_model",
            "check_rdma",
            "check_pd_devices",
            "check_pd_runtime",
            "dummy",
        ):
            with self.subTest(failure=failure):
                result, checks = self.run_main(
                    ["--pd-case", "deepseek-v41-flash-1p1d"],
                    {
                        "MODEL_PATH": str(self.root),
                        "PD_TEST_ARTIFACT_DIR": str(self.root),
                        "PD_TEST_DUMMY_WEIGHTS": "1" if failure == "dummy" else "0",
                    },
                    failure,
                )
                self.assertEqual(result, 1)
                self.assertFalse((self.root / "runtime.json").exists())
                expected_last = "print_host_summary" if failure == "dummy" else failure
                self.assertEqual(checks.mock_calls[-1][0], expected_last)

    def test_checker_rejects_unknown_case_and_arguments_before_checks(self):
        for args in (["--pd-case", "future-multinode"], ["--pd-case"], ["pd_test"]):
            with self.subTest(args=args):
                with self.assertRaises(SystemExit) as error:
                    self.run_main(args)
                self.assertEqual(error.exception.code, 2)
                self.assertEqual(self.last_checks.mock_calls, [])

    def test_memory_limit_and_both_no_swap_paths(self):
        cgroup = self.root / "cgroup"
        cgroup.mkdir()
        meminfo = self.root / "meminfo"
        limit_file = cgroup / "memory.max"
        swap_file = cgroup / "memory.swap.max"
        max_bytes = self.config["environment"]["memory_limit_gib"] * 1024**3
        for limit in (1, max_bytes):
            for swap in ("0", "max", "1024"):
                with self.subTest(limit=limit, swap=swap):
                    limit_file.write_text(str(limit) + "\n")
                    swap_file.write_text(swap + "\n")
                    if swap == "0":
                        # The container path must not need /proc/meminfo at all.
                        if meminfo.exists():
                            meminfo.unlink()
                    else:
                        meminfo.write_text("MemTotal: 123 kB\nSwapTotal: 0 kB\n")
                    self.assertEqual(
                        self.checker.check_memory_limit(
                            max_bytes, cgroup=cgroup, meminfo=meminfo
                        ),
                        limit,
                    )
        for limit in ("max", "0", "-1", "1.5", str(max_bytes + 1)):
            with self.subTest(limit=limit):
                limit_file.write_text(limit)
                with self.assertRaisesRegex(self.checker.CheckFailure, "memory limit"):
                    self.checker.check_memory_limit(
                        max_bytes, cgroup=cgroup, meminfo=meminfo
                    )
        limit_file.write_text(str(max_bytes))
        swap_file.write_text("0")
        with self.assertRaisesRegex(self.checker.CheckFailure, "memory limit"):
            self.checker.check_memory_limit(
                max_bytes - 1, cgroup=cgroup, meminfo=meminfo
            )
        for swap in ("max", "1024"):
            for host in ("SwapTotal: 1 kB\n", "SwapTotal: 00 kB\n", "MemTotal: 123 kB\n"):
                with self.subTest(swap=swap, host=host):
                    swap_file.write_text(swap)
                    meminfo.write_text(host)
                    with self.assertRaisesRegex(self.checker.CheckFailure, "Disable swap"):
                        self.checker.check_memory_limit(
                            max_bytes, cgroup=cgroup, meminfo=meminfo
                        )
        limit_file.unlink()
        with self.assertRaises(FileNotFoundError):
            self.checker.check_memory_limit(max_bytes, cgroup=cgroup, meminfo=meminfo)

    def test_model_shards_missing_empty_and_unsafe_index_paths(self):
        model = self.root / "model"
        model.mkdir()
        files = ("config.json", "tokenizer.json", "model.safetensors.index.json")
        for name in files:
            (model / name).write_text("{}")
        shard = model / "weights/part.safetensors"
        shard.parent.mkdir()
        shard.write_bytes(b"fixture")
        index = model / files[-1]
        index.write_text(json.dumps({"weight_map": {"a": "weights/part.safetensors",
                                                   "b": "weights/part.safetensors"}}))
        self.assertEqual(self.checker.check_model(model), 1)
        for name in files:
            with self.subTest(missing=name):
                path = model / name
                content = path.read_bytes()
                path.unlink()
                with self.assertRaisesRegex(self.checker.CheckFailure, "Missing model file"):
                    self.checker.check_model(model)
                path.write_bytes(content)
        for names, message in (([], "no weight shards"), ([str(shard)], "Invalid"),
                               (["../outside"], "Invalid"), (["weights/../part"], "Invalid"),
                               ([7], "Invalid"), (["missing.safetensors"], "Missing or empty")):
            with self.subTest(names=names):
                index.write_text(json.dumps({"weight_map": dict(enumerate(names))}))
                with self.assertRaisesRegex(self.checker.CheckFailure, message):
                    self.checker.check_model(model)
        index.write_text(json.dumps({"weight_map": {"a": "weights/part.safetensors"}}))
        shard.write_bytes(b"")
        with self.assertRaisesRegex(self.checker.CheckFailure, "empty shard"):
            self.checker.check_model(model)

    def test_model_accepts_huggingface_snapshot_links_to_external_blobs(self):
        model, blobs = self.root / "snapshots/revision", self.root / "blobs"
        model.mkdir(parents=True)
        blobs.mkdir()
        content = {
            "config.json": b"{}", "tokenizer.json": b"{}",
            "model.safetensors.index.json": b'{"weight_map":{"a":"part.safetensors"}}',
            "part.safetensors": b"fixture",
        }
        for name, value in content.items():
            (blobs / name).write_bytes(value)
            (model / name).symlink_to(Path("../../blobs") / name)
        self.assertEqual(self.checker.check_model(model), 1)
        (blobs / "part.safetensors").unlink()
        with self.assertRaisesRegex(self.checker.CheckFailure, "Missing or empty"):
            self.checker.check_model(model)

    def test_rdma_requires_distinct_devices_active_gid_and_open_verbs(self):
        sysfs, dev = self.root / "sys", self.root / "dev"
        dev.mkdir()
        for number in range(2):
            port = sysfs / f"infiniband/mlx5_{number}/ports/1"
            (port / "gids").mkdir(parents=True)
            (port / "state").write_text("4: ACTIVE\n")
            (port / "gids/3").write_text("0000:0000:0000:0000:0000:0000:0000:0001\n")
            verbs = sysfs / f"infiniband_verbs/uverbs{number}"
            verbs.mkdir(parents=True)
            (verbs / "ibdev").write_text(f"mlx5_{number}\n")
            (dev / f"uverbs{number}").write_bytes(b"fixture")
        check = lambda names="mlx5_0,mlx5_1", gid=3: self.checker.check_rdma(names, gid, sysfs, dev)
        self.assertEqual(check(), ["mlx5_0", "mlx5_1"])
        for names, gid in (("mlx5_0,mlx5_0", 3), ("../mlx5_0", 3), ("mlx5_0,", 3),
                           ("mlx5_0, mlx5_1", 3), ("", 3), ("mlx5_0", -1), ("mlx5_2", 3)):
            with self.subTest(names=names, gid=gid), self.assertRaises(self.checker.CheckFailure):
                check(names, gid)
        for relative, value, message in (
            ("infiniband/mlx5_1/ports/1/state", "2: INIT", "active port"),
            ("infiniband/mlx5_1/ports/1/gids/3", "0000:0000:0000:0000", "active port"),
            ("infiniband_verbs/uverbs1/ibdev", "other_device", "verbs device"),
        ):
            with self.subTest(path=relative):
                path = sysfs / relative
                old = path.read_text()
                path.write_text(value)
                with self.assertRaisesRegex(self.checker.CheckFailure, message):
                    check()
                path.write_text(old)
        (dev / "uverbs1").unlink()
        with self.assertRaises(FileNotFoundError):
            check()
        with patch.object(self.checker.os, "open", side_effect=PermissionError("fixture verbs denied")) as open_verbs:
            with self.assertRaises(PermissionError):
                check("mlx5_0")
            open_verbs.assert_called_once_with(dev / "uverbs0", os.O_RDWR)

    def test_pd_devices_computes_on_configured_width_and_rejects_other_allocations(self):
        torch = Mock()
        torch.ones.return_value.__add__ = Mock(
            return_value=SimpleNamespace(item=lambda: 2)
        )
        with patch.dict(sys.modules, {"torch": torch}):
            for width in (8, 4):
                with self.subTest(width=width):
                    torch.ones.reset_mock()
                    torch.cuda.device_count.return_value = width
                    self.checker.check_pd_devices(width)
                    self.assertEqual(
                        torch.ones.call_args_list,
                        [call(1, device=f"cuda:{i}") for i in range(width)],
                    )
                    for count in (0, 2, 16):
                        with self.subTest(count=count):
                            torch.ones.reset_mock()
                            torch.cuda.device_count.return_value = count
                            with self.assertRaisesRegex(
                                self.checker.CheckFailure, f"{width} visible PPUs"
                            ):
                                self.checker.check_pd_devices(width)
                            torch.ones.assert_not_called()
            torch.cuda.device_count.return_value = 8
            torch.ones.return_value.__add__.return_value = SimpleNamespace(item=lambda: 3)
            with self.assertRaisesRegex(self.checker.CheckFailure, "PPU 0 compute failed"):
                self.checker.check_pd_devices(8)

    def test_pd_yaml_round_trip_and_digest(self):
        self.assertEqual(
            self.pd.PD_CONFIG_PATH,
            RUNNER.parent / "ppu/recipes/deepseek-ai/DeepSeek-V4.1-Flash-PD.yaml",
        )
        path = self.root / "pd.yaml"
        path.write_text(yaml.safe_dump(self.config, sort_keys=False))
        self.assertEqual(self.pd.load_pd_config(path), self.config)
        reordered = dict(reversed(list(self.config.items())))
        self.assertEqual(
            self.pd.config_digest(reordered),
            hashlib.sha256(json.dumps(self.config, sort_keys=True).encode()).hexdigest(),
        )

    def test_pd_yaml_rejects_multinode_and_remote_hosts_explicitly(self):
        path = self.root / "pd.yaml"
        for host, extra_node in (("10.0.0.1", False), ("localhost", False), ("127.0.0.1", True)):
            with self.subTest(host=host, extra_node=extra_node):
                config = deepcopy(self.config)
                config["nodes"]["worker"]["host"] = host
                if extra_node:
                    config["nodes"]["other"] = {"host": "127.0.0.1", "devices": 8}
                    config["ppu"]["topology"]["nodes"] = 2
                    config["roles"]["decode"]["node"] = "other"
                path.write_text(yaml.safe_dump(config))
                with self.assertRaisesRegex(NotImplementedError, "remote nodes"):
                    self.pd.load_pd_config(path)

    def test_pd_yaml_rejects_invalid_schema_allocations_ports_and_types(self):
        invalid = [
            (("schema_version",), "ppu-pd-test/v2", "Unsupported"),
            (("case_id",), "future-multinode", "Unsupported"),
            (("roles",), {"prefill": self.config["roles"]["prefill"]}, "1P1D"),
            (("roles", "decode", "node"), "missing", "unavailable node"),
            (("router", "node"), "missing", "unavailable node"),
            (("roles", "decode", "devices"), [3, 5, 6, 7], "distinct"),
            (("roles", "decode", "devices"), [4, 5, 6], "cover"),
            (("roles", "prefill", "devices"), [0, 0, 2, 3], "distinct"),
            (("roles", "prefill", "devices"), [], "indices"),
            (("roles", "prefill", "devices"), [-1, 1, 2, 3], "indices"),
            (("roles", "decode", "devices"), [4, 5, 6, 8], "indices"),
            (("roles", "prefill", "devices"), [False, 1, 2, 3], "indices"),
            (("router", "metrics_port"), 8008, "ports"),
            (("roles", "decode", "nccl_port"), 2030, "ports"),
            (("roles", "decode", "bootstrap_port"), 2000, "ports"),
        ]
        invalid += [
            (keys, value, "ports")
            for keys in (
                ("router", "port"),
                ("router", "metrics_port"),
                ("roles", "prefill", "port"),
                ("roles", "decode", "nccl_port"),
                ("roles", "prefill", "bootstrap_port"),
            )
            for value in (0, -1, 65536, "8008", True)
        ]
        invalid += [
            (("nodes", "worker", "devices"), value, "device count")
            for value in (0, -1, "8", True)
        ]
        invalid += [
            (path, value, "strings")
            for path in (("common_env",), ("roles", "prefill", "env"), ("roles", "decode", "env"))
            for value in ({"FLAG": 1}, {"FLAG": False}, {1: "value"}, ["FLAG=1"])
        ]
        invalid += [
            (path, value, "arguments")
            for path in (("common_args",), ("roles", "prefill", "args"), ("roles", "decode", "args"), ("router", "args"))
            for value in ("--flag", ["--flag", 4], [False], [""])
        ]
        invalid += [
            ((key,), value, "timeouts")
            for key in ("ready_timeout_s", "request_timeout_s")
            for value in (0, -1, "60", True)
        ]
        invalid += [
            (("environment", "memory_limit_gib"), value, "memory limit")
            for value in (0, -1, "900", True)
        ]
        path = self.root / "pd.yaml"
        for keys, value, message in invalid:
            with self.subTest(keys=keys, value=value):
                config = deepcopy(self.config)
                target = config
                for key in keys[:-1]:
                    target = target[key]
                target[keys[-1]] = value
                path.write_text(yaml.safe_dump(config))
                with self.assertRaisesRegex(ValueError, message):
                    self.pd.load_pd_config(path)

    def test_pd_yaml_preserves_original_deployment_fingerprint(self):
        deployment = {
            "MODEL_PATH": self.config["environment"]["model_path"],
            "COMMON_ENV": self.config["common_env"],
            "COMMON_ARGS": self.config["common_args"],
            "PREFILL_ENV": self.config["roles"]["prefill"]["env"],
            "DECODE_ENV": self.config["roles"]["decode"]["env"],
            "PREFILL_ARGS": self.config["roles"]["prefill"]["args"],
            "DECODE_ARGS": self.config["roles"]["decode"]["args"],
        }
        self.assertEqual(
            hashlib.sha256(json.dumps(deployment, sort_keys=True).encode()).hexdigest(),
            "5c3fc010f4ff70f1d5fb2981ef9fd7f412f544592ffd7b4c2cff22aa38095ad1",
        )

    def test_runtime_provenance_native_imports_router_and_metadata(self):
        repo = RUNNER.parent.parent
        native = ("sgl_kernel", "deep_gemm", "flash_mla", "mooncake.engine", "sglang_router")
        flags = ("--mini-lb", "--pd-disaggregation", "--prefill", "--decode")
        for source in ("source", "installed"):
            origin = (repo / "python" if source == "source" else self.root / "site") / "sglang/__init__.py"
            platform = Mock()
            platform.is_sail.return_value = True
            modules = {"sglang": SimpleNamespace(__file__=str(origin)),
                       "sglang.srt.platforms": SimpleNamespace(current_platform=platform)}
            def version(name):
                if name == "sgl-kernel":
                    raise self.checker.metadata.PackageNotFoundError(name)
                return "1.2.3"
            with self.subTest(source=source), contextlib.ExitStack() as stack:
                stack.enter_context(patch.dict(os.environ, {"PD_TEST_CODE_SOURCE": source,
                                                           "PPU_CI_RECIPE_JIT_DIR": str(self.root)}, clear=True))
                stack.enter_context(patch.object(sys, "path", list(sys.path)))
                original_path = list(sys.path)
                stack.enter_context(patch.dict(sys.modules, modules))
                imports = stack.enter_context(patch.object(self.checker.importlib, "import_module"))
                router = stack.enter_context(patch.object(self.checker.subprocess, "run", return_value=SimpleNamespace(stdout=" ".join(flags))))
                stack.enter_context(patch.object(self.checker.metadata, "version", side_effect=version))
                with patch.object(Path, "read_bytes", side_effect=FileNotFoundError("E2E script unavailable")) as read_bytes:
                    runtime = self.checker.check_pd_runtime()
                    read_bytes.assert_not_called()
                self.assertEqual(imports.call_args_list, [call(name) for name in native])
                router.assert_called_once_with(
                    [sys.executable, "-I", "-B", "-m", "sglang_router.launch_router", "--help"],
                    check=True, capture_output=True, text=True, timeout=60,
                )
                self.assertEqual(sys.path, [str(repo / "python"), *original_path] if source == "source" else original_path)
                self.assertEqual(runtime, {
                    "code_source": source, "sglang_import": str(origin.resolve()),
                    "versions": {name: "1.2.3" for name in ("sglang", "sglang-kernel", "torch", "sglang-router", "deep-gemm")},
                    "jit_dir": str(self.root),
                })
                platform.is_sail.return_value = False
                with self.assertRaisesRegex(self.checker.CheckFailure, "SAIL"):
                    self.checker.check_pd_runtime()
                platform.is_sail.return_value = True
                modules["sglang"].__file__ = str(self.root / "wrong/__init__.py" if source == "source" else repo / "python/sglang/__init__.py")
                with self.assertRaisesRegex(self.checker.CheckFailure, "does not match"):
                    self.checker.check_pd_runtime()
                modules["sglang"].__file__ = str(origin)
                for missing in native:
                    with self.subTest(missing=missing):
                        def import_native(name):
                            if name == missing:
                                raise ImportError(name)
                        imports.side_effect = import_native
                        with self.assertRaises(ImportError):
                            self.checker.check_pd_runtime()
                imports.side_effect = None
                for missing in flags:
                    with self.subTest(missing=missing):
                        router.return_value.stdout = " ".join(flag for flag in flags if flag != missing)
                        with self.assertRaisesRegex(self.checker.CheckFailure, missing):
                            self.checker.check_pd_runtime()
                router.side_effect = subprocess.CalledProcessError(7, "router")
                with self.assertRaises(subprocess.CalledProcessError):
                    self.checker.check_pd_runtime()


class TestPPURecipeContainer(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.source = self.root / "repo with spaces"
        scripts = self.source / "scripts/ci/ppu"
        scripts.mkdir(parents=True)
        self.script = scripts / "recipe_container.sh"
        self.script.write_bytes(Path(__file__).with_name("recipe_container.sh").read_bytes())
        self.artifacts, self.wheels, self.tools = (self.root / name for name in ("artifacts", "wheel house", "tools"))
        for directory in (self.artifacts, self.wheels, self.tools):
            directory.mkdir()
        (self.wheels / "runtime-deps").mkdir()
        (self.wheels / "sglang-fixture.whl").write_bytes(b"not a real wheel")
        self.events = self.root / "events.jsonl"
        # Fake only external Python calls. Execute the real inline platform and
        # executor wrappers, with injected modules and a harmless executor.
        fake = self.tools / "python3"
        fake.write_text(f"#!{sys.executable}\n" + r'''
import json
import os
import sys
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

argv = sys.argv[1:]
args = argv[2:]
code = sys.stdin.read() if args[0] == "-" else None
if args[0] == "-":
    event = "platform" if len(args) == 2 else "executor"
elif args[:2] == ["-m", "pip"]:
    event = "install"
elif args[0].endswith("check_dependencies.py"):
    event = args[1]
    if event == "snapshot":
        event += "-before" if args[2].endswith("before.json") else "-after"
elif args[0].endswith("ppu_recipe.py"):
    event = "validation"
else:
    raise AssertionError(args)
keys = ("PD_TEST_CODE_SOURCE", "PD_TEST_DUMMY_WEIGHTS", "RECIPE_PPU_COUNT",
        "MODEL_PATH", "PD_TEST_ARTIFACT_DIR", "PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV",
        "HF_HUB_OFFLINE", "SGLANG_DG_CACHE_DIR", "SGLANG_CACHE_DIR", "SGLANG_JIT_CACHE_DIR")
with open(os.environ["FIXTURE_EVENTS"], "a") as output:
    output.write(json.dumps({"event": event, "argv": argv, "cwd": os.getcwd(),
                            "phase": (Path(os.environ["PPU_CI_ARTIFACT_DIR"]) / "worker-phase.txt").read_text().strip(),
                            "env": {key: os.environ.get(key) for key in keys}}) + "\n")
if event == os.environ.get("FIXTURE_FAILURE"):
    sys.exit(7)
if event == "platform":
    platform = SimpleNamespace(is_sail=lambda: os.environ.get("FIXTURE_SAIL", "true") == "true")
    sys.modules.update({"sglang": SimpleNamespace(__version__="0.5.18"),
                       "torch": SimpleNamespace(__version__="2.13.0", cuda=SimpleNamespace(device_count=lambda: int(os.environ["FIXTURE_VISIBLE_COUNT"]))),
                       "sglang.srt.platforms": SimpleNamespace(current_platform=platform)})
    def version(name):
        versions = {"sglang": "0.5.18", "torch": "2.13.0", "sglang-kernel": "0.4.9"}
        if name not in versions:
            raise metadata.PackageNotFoundError(name)
        return versions[name]
    metadata.version = version
    sys.argv = ["-", *args[1:]]
    exec(compile(code, "platform-wrapper", "exec"), {})
elif event == "executor":
    sys.argv = ["-", *args[1:]]
    exec(compile(code, "executor-wrapper", "exec"), {})
''')
        fake.chmod(0o755)
        (scripts / "recipe_runner.py").write_text(
            "import json, os, sys\n"
            "with open(os.environ['FIXTURE_EVENTS'], 'a') as output:\n"
            "    output.write(json.dumps({'event': 'runner', 'argv': sys.argv, 'executor_path': sys.path[0]}) + '\\n')\n"
            "sys.exit(int(os.environ.get('FIXTURE_RUNNER_EXIT', '0')))\n"
        )
        # Do not inherit caller PD settings or Python startup configuration.
        self.env = {
            "PATH": str(self.tools) + os.pathsep + os.defpath,
            "PPU_CI_ARTIFACT_DIR": str(self.artifacts), "PPU_CI_WHEEL_DIR": str(self.wheels),
            "PPU_CI_RECIPE_SOURCE_DIR": str(self.source), "PPU_CI_RECIPE_JIT_DIR": str(self.root / "jit"),
            "PPU_RECIPE_CONTAINER_MODEL": str(self.root / "mounted model"),
            "FIXTURE_EVENTS": str(self.events), "FIXTURE_VISIBLE_COUNT": "8",
            "PYTHONPATH": "/untrusted/source", "PYTHONHOME": "/untrusted/home", "VIRTUAL_ENV": "/untrusted/venv",
        }

    def run_entry(self, args, **env):
        self.events.write_text("")
        result = subprocess.run(
            ["/bin/bash", str(self.script), *args], cwd=self.root,
            env={**self.env, **env}, capture_output=True, text=True, timeout=15,
        )
        records = [json.loads(line) for line in self.events.read_text().splitlines()]
        return result, records

    def assert_sequence(self, records, names):
        self.assertEqual([record["event"] for record in records], names)
        for record in records:
            if "phase" not in record:
                continue
            self.assertEqual(record["argv"][:2], ["-I", "-B"])
            self.assertEqual(record["cwd"], str(self.source))
            for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
                self.assertIsNone(record["env"][key])
            self.assertEqual(record["env"]["HF_HUB_OFFLINE"], "1")

    def test_original_validation_and_executor_keep_non_eight_allocations(self):
        install = ["snapshot-before", "install", "missing-wheels", "snapshot-after", "check"]
        for width in (2, 4, 8, 16):
            for args in (["validate"], ["fixture-model", "fast_sanity"]):
                with self.subTest(width=width, args=args):
                    result, records = self.run_entry(
                        args, RECIPE_PPU_COUNT=str(width), FIXTURE_VISIBLE_COUNT=str(width),
                        PD_TEST_CODE_SOURCE="invalid", PD_TEST_DUMMY_WEIGHTS="1",
                    )
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    tail = ["validation"] if args == ["validate"] else ["validation", "platform", "executor", "runner"]
                    self.assert_sequence(records, install + tail)
                    self.assertEqual(records[5]["argv"][2:], [
                        "scripts/ci/ppu/ppu_recipe.py", "validate", "--all", "--require-sglang-parser",
                    ])
                    self.assertFalse((self.artifacts / "pd").exists())
                    if args != ["validate"]:
                        self.assertEqual(records[-1]["argv"], [str(self.source / "scripts/ci/ppu/recipe_runner.py"), *args])
                        self.assertEqual(records[-1]["executor_path"], str(self.source / "scripts/ci/ppu"))
        args = ["test/ppu/policy/models/deepseek-v41-flash-pd.yaml", "model_sanity"]
        result, records = self.run_entry(args, RECIPE_PPU_COUNT="8")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assert_sequence(records, install + ["validation", "platform", "executor", "runner"])
        self.assertEqual(records[-1]["argv"], [str(self.source / "scripts/ci/ppu/recipe_runner.py"), *args])
        self.assertEqual(records[-1]["executor_path"], str(self.source / "scripts/ci/ppu"))
        result, records = self.run_entry(["fixture-model", "fast_sanity"], PPU_CI_SKIP_WHEEL_INSTALL="true", FIXTURE_RUNNER_EXIT="9")
        self.assertEqual(result.returncode, 9)
        self.assert_sequence(records, ["snapshot-before", "validation", "platform", "executor", "runner"])


if __name__ == "__main__":
    unittest.main()
