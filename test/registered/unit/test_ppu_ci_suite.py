import hashlib
import importlib.util
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


if __name__ == "__main__":
    unittest.main()
