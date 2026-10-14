import hashlib
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import HWBackend, collect_tests, register_cpu_ci
from sglang.test.test_utils import CustomTestCase, popen_launch_server

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

RUNNER = Path(__file__).resolve().parents[2] / "run_suite.py"


class TestPPUCISuite(CustomTestCase):
    def test_explicit_offline_launch_cannot_recover_by_going_online(self):
        """An explicitly offline launch must not succeed with downloaded replacements."""
        failed = MagicMock()
        failed.poll.return_value = 1
        ready = MagicMock()
        ready.poll.return_value = None
        with (
            patch(
                "sglang.test.test_utils.subprocess.Popen", side_effect=[failed, ready]
            ),
            patch("sglang.test.test_utils.requests.Session") as session,
            patch("sglang.test.test_utils.kill_process_tree"),
        ):
            session.return_value.__enter__.return_value.get.return_value.status_code = (
                200
            )
            self.assertRaisesRegex(
                Exception,
                "Server process exited with code 1",
                popen_launch_server,
                "test/model",
                "http://127.0.0.1:30000",
                timeout=1,
                device="cpu",
                env={"HF_HUB_OFFLINE": "1"},
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
