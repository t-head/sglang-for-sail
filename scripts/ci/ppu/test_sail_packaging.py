import os
import runpy
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
SETUP = ROOT / "python/sglang/kernels/aot/setup_sail.py"


class TestSailPackaging(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.aot = self.root / "aot"
        self.aot.mkdir()
        shutil.copy2(SETUP, self.aot / "setup_sail.py")
        for filename in ("pyproject_sail.toml", "LICENSE", "README.md"):
            shutil.copy2(SETUP.parent / filename, self.aot / filename)
        shutil.copy2(self.aot / "pyproject_sail.toml", self.aot / "pyproject.toml")
        kernel = self.aot / "python/sgl_kernel"
        kernel.mkdir(parents=True)
        (kernel / "__init__.py").write_text("")
        self.third_party = self.aot / "build/_deps"
        self.triton = self.third_party / "triton/python/triton_kernels"
        mxfp = self.triton / "triton_kernels/numerics_details/mxfp.py"
        mxfp.parent.mkdir(parents=True)
        mxfp.write_text("def downcast_to_mxfp(value):\n    return value\n")
        (self.triton / "tests").mkdir()
        (self.triton / "tests/test_not_shipped.py").write_text("")
        license_file = self.aot / "build/_deps/triton/LICENSE"
        license_file.write_text("MIT License\nCopyright Triton contributors\n")

    def package_metadata(self):
        torch = types.ModuleType("torch")
        torch.version = types.SimpleNamespace(cuda="13.0")
        torch._C = types.SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=True)
        extension = types.ModuleType("torch.utils.cpp_extension")
        extension.BuildExtension = type(
            "BuildExtension", (), {"with_options": classmethod(lambda cls, **kw: cls)}
        )
        extension.CUDAExtension = lambda **kwargs: None
        old_cwd = Path.cwd()
        try:
            os.chdir(self.aot)
            with (
                patch.dict(
                    sys.modules,
                    {"torch": torch, "torch.utils.cpp_extension": extension},
                ),
                patch.dict(
                    os.environ,
                    {
                        "SGL_KERNEL_SKIP_THIRD_PARTY": "1",
                        "SGL_KERNEL_SKIP_SAILIFY": "1",
                        "SGL_KERNEL_THIRD_PARTY_DIR": str(self.third_party),
                    },
                ),
                patch("setuptools.setup") as setup,
            ):
                scope = runpy.run_path(str(self.aot / "setup_sail.py"))
            return setup.call_args.kwargs, scope
        finally:
            os.chdir(old_cwd)

    def run_python(self, python, *args, cwd=None, expected_returncode=0):
        result = subprocess.run(
            [str(python), "-I", "-B", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "PIP_NO_INDEX": "1",
                "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            },
        )
        self.assertEqual(
            result.returncode, expected_returncode, result.stdout + result.stderr
        )
        return result

    def test_kernel_wheel_is_self_contained_after_old_distribution_uninstall(self):
        self.assert_self_contained_wheel()

    def test_external_third_party_sources_preserve_both_licenses(self):
        external = self.root / "external"
        shutil.move(self.third_party, external)
        self.third_party = external
        self.triton = external / "triton/python/triton_kernels"
        self.assert_self_contained_wheel()

    def assert_self_contained_wheel(self):
        metadata, scope = self.package_metadata()
        self.assertIn("triton_kernels.numerics_details", metadata["packages"])
        self.assertNotIn("tests", metadata["packages"])
        triton_repo = scope["_TRITON_REPO"]
        self.assertIn(triton_repo, scope["ALL_REPOS"])
        self.assertEqual(
            triton_repo.git_tag, "5f3f125e8f63c24613f1f73b937442864f263f94"
        )
        wheels = self.root / "wheels"
        wheels.mkdir()
        metadata = {
            key: metadata[key]
            for key in ("name", "version", "packages", "package_dir", "license_files")
        }
        self.run_python(
            sys.executable,
            "-c",
            "from setuptools import setup; setup(**" + repr(metadata) + ")",
            "bdist_wheel",
            "--dist-dir",
            str(wheels),
            cwd=self.aot,
        )
        wheel = next(wheels.glob("*.whl"))
        with zipfile.ZipFile(wheel) as archive:
            self.assertIn("triton_kernels/numerics_details/mxfp.py", archive.namelist())
            licenses = {
                archive.read(name)
                for name in archive.namelist()
                if Path(name).name.startswith("LICENSE")
            }
            self.assertIn(b"MIT License\nCopyright Triton contributors\n", licenses)
            self.assertIn((self.aot / "LICENSE").read_bytes(), licenses)
            self.assertFalse(
                any(name.startswith("tests/") for name in archive.namelist())
            )
        old = self.root / "old"
        old.mkdir()
        self.run_python(
            sys.executable,
            "-c",
            "from setuptools import setup; setup(name='sgl-kernel', version='0.4.8', "
            "packages=['triton_kernels', 'triton_kernels.numerics_details'], "
            f"package_dir={{'': {str(self.triton)!r}}})",
            "bdist_wheel",
            "--dist-dir",
            str(old),
            cwd=old,
        )
        env = self.root / "venv"
        self.run_python(sys.executable, "-m", "venv", str(env))
        python = env / "bin/python"
        self.run_python(
            python, "-m", "pip", "install", "--no-deps", str(next(old.glob("*.whl")))
        )
        self.run_python(python, "-m", "pip", "uninstall", "-y", "sgl-kernel")
        missing = self.run_python(
            python,
            "-c",
            "from triton_kernels.numerics_details.mxfp import downcast_to_mxfp",
            expected_returncode=1,
        )
        self.assertIn("ModuleNotFoundError", missing.stderr)
        self.run_python(python, "-m", "pip", "install", "--no-deps", str(wheel))
        self.run_python(
            python,
            "-c",
            "from triton_kernels.numerics_details.mxfp import downcast_to_mxfp; assert downcast_to_mxfp(7) == 7",
        )

    def test_missing_triton_source_fails_before_packaging(self):
        (self.triton / "triton_kernels/numerics_details/mxfp.py").unlink()
        with self.assertRaisesRegex(RuntimeError, "triton_kernels"):
            self.package_metadata()

    def test_ci_installers_do_not_depend_on_runtime_backup(self):
        for filename in ("build_and_test.sh", "recipe_container.sh"):
            with self.subTest(filename=filename):
                text = (ROOT / "scripts/ci/ppu" / filename).read_text()
                self.assertNotIn("backup-triton-kernels", text)
                self.assertNotIn("restore-triton-kernels", text)
                self.assertIn("check-triton-kernels", text)


if __name__ == "__main__":
    unittest.main()
