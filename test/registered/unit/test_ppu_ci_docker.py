"""Host launcher protocol tests; Docker/git boundaries need no PPU or daemon."""

import hashlib
import json
import os
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

if sys.platform == "linux":
    import fcntl

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/ci/ppu/run_test.sh"
SHA = "a" * 40
IMAGE_ID = "sha256:" + "b" * 64

# Stub Git to avoid modifying the checkout under test.
GIT = r"""
import os
import sys
from pathlib import Path

args = sys.argv[1:]
root = Path(os.environ["FIXTURE_SOURCE"])
assert args[:2] == ["-C", str(root)] or args[:2] == ["-C", str(root / "scripts/ci/ppu")], args
args = args[2:]
if args == ["rev-parse", "--show-toplevel"]:
    print(root)
elif args == ["rev-parse", "HEAD"]:
    print("a" * 40)
elif args[0] == "status":
    assert any(arg.startswith("--porcelain") for arg in args), args
    if (root / "dirty-tracked").exists():
        print(" M tracked.py")
    if (root / "dirty-untracked").exists():
        print("?? untracked.py")
    if (root / "dirty-ignored").exists() and any(arg.startswith("--ignored") for arg in args):
        print("!! ignored.whl")
else:
    raise AssertionError(args)
"""

DOCKER = r"""
import fcntl
import json
import os
import signal
import sys
from pathlib import Path

args = sys.argv[1:]
root = Path(os.environ["FIXTURE_ROOT"])
lock = Path(os.environ["FIXTURE_LOCK"])
held = False
if lock.exists():
    with lock.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            held = True
with (root / "docker.jsonl").open("a") as log:
    log.write(json.dumps({"args": args, "lock_held": held}) + "\n")
state = root / "container.json"
if args[:2] == ["image", "inspect"]:
    if os.environ.get("MISSING_IMAGE") and not (root / "pulled").exists():
        sys.exit(1)
    print("sha256:" + "b" * 64)
elif args[0] == "pull":
    (root / "pulled").touch()
elif args[0] == "create":
    name = args[args.index("--name") + 1]
    assert not state.exists(), "another container was overwritten"
    state.write_text(json.dumps({"name": name}))
    print("fixture-container-id")
elif args[0] == "start":
    data = json.loads(state.read_text())
    assert args[-1] == data["name"]
    data["pid"] = os.getpid()
    state.write_text(json.dumps(data))
    if os.environ.get("BLOCK_TEST"):
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
        with (root / "ready.fifo").open("w") as ready:
            ready.write("started\n")
        signal.pause()
    sys.exit(int(os.environ.get("TEST_EXIT", "0")))
elif args[0] == "stop":
    data = json.loads(state.read_text())
    assert args[-1] == data["name"], "stopped someone else's container"
    if "pid" in data:
        try:
            os.kill(data["pid"], signal.SIGTERM)
        except ProcessLookupError:
            pass
elif args[0] == "rm":
    assert args[-1] == json.loads(state.read_text())["name"]
    state.unlink()
elif args[0] == "run":
    assert not state.exists(), "ownership recovery ran before container removal"
    assert "chown" in args, args
    mounts = [args[i + 1] for i, arg in enumerate(args) if arg == "--mount"]
    assert len(mounts) == 1, "ownership helper exposed shared inputs"
    mount = dict(item.split("=", 1) for item in mounts[0].split(","))
    assert mount["target"] == "/artifacts"
    private = Path(mount["source"])
    (private / "ownership-recovered").touch()
else:
    raise AssertionError(args)
"""


@unittest.skipUnless(
    sys.platform == "linux" and shutil.which("flock"),
    "The Docker host launcher requires Linux and util-linux flock",
)
class TestPPUCIDocker(CustomTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ppu-docker-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / "source with spaces"
        script_dir = self.source / "scripts/ci/ppu"
        script_dir.mkdir(parents=True)
        self.launcher = script_dir / "run_test.sh"
        if SCRIPT.exists():
            shutil.copyfile(SCRIPT, self.launcher)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        for name, body in (("docker", DOCKER), ("git", GIT)):
            path = self.bin / name
            path.write_text(f"#!{sys.executable}\n" + body)
            path.chmod(0o700)
        self.wheels = self.root / "wheel house"
        (self.wheels / "runtime-deps").mkdir(parents=True)
        (self.wheels / "source-sha.txt").write_text(SHA + "\n")
        (self.wheels / "sglang-1-py3-none-any.whl").write_bytes(b"main wheel")
        (self.wheels / "runtime-deps/sailify-1-py3-none-any.whl").write_bytes(
            b"dependency"
        )
        self.manifest()
        self.artifacts = self.root / "artifacts"
        self.artifacts.mkdir()
        self.models = [self.root / name for name in ("target", "eagle3", "dflash")]
        for model in self.models:
            model.mkdir()
            (model / "config.json").write_text("{}")
        self.cache = self.root / "eval-cache"
        (self.cache / "mmlu").mkdir(parents=True)
        data = b'{"question": "fixture"}\n'
        (self.cache / "mmlu/test.jsonl").write_bytes(data)
        (self.cache / "mmlu/test.jsonl.sha256").write_text(
            hashlib.sha256(data).hexdigest() + "\n"
        )
        self.device = str(uuid.uuid4().int)
        self.lock = Path(f"/tmp/sglang-ppu-ci-device-{self.device}.lock")
        # LIFO cleanup keeps the fixture lock until launcher processes exit.
        self.addCleanup(self.lock.unlink, missing_ok=True)
        self.env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("PPU_CI_") and key != "NVIDIA_VISIBLE_DEVICES"
        }
        self.env.update(
            PATH=f"{self.bin}:{os.environ['PATH']}",
            FIXTURE_ROOT=str(self.root),
            FIXTURE_SOURCE=str(self.source),
            FIXTURE_LOCK=str(self.lock),
            PPU_CI_RUNTIME_IMAGE="example.invalid/ppu/runtime:fixture",
            PPU_CI_SOURCE_SHA=SHA,
            PPU_CI_DEVICE=self.device,
            PPU_CI_WHEEL_DIR=str(self.wheels),
            PPU_CI_ARTIFACT_DIR=str(self.artifacts),
            PPU_CI_TARGET_MODEL_PATH=str(self.models[0]),
            PPU_CI_EAGLE3_MODEL_PATH=str(self.models[1]),
            PPU_CI_DFLASH_MODEL_PATH=str(self.models[2]),
            PPU_CI_EVAL_CACHE=str(self.cache),
        )
        os.mkfifo(self.root / "ready.fifo")
        self.ready = os.open(self.root / "ready.fifo", os.O_RDWR | os.O_NONBLOCK)
        self.addCleanup(os.close, self.ready)

    def manifest(self):
        names = ["source-sha.txt"] + [
            "./" + str(path.relative_to(self.wheels))
            for path in sorted(self.wheels.rglob("*.whl"))
        ]
        (self.wheels / "SHA256SUMS").write_text(
            "".join(
                f"{hashlib.sha256((self.wheels / name).read_bytes()).hexdigest()}  {name}\n"
                for name in names
            )
        )

    def run_launcher(self, **updates):
        return subprocess.run(
            ["bash", str(self.launcher)],
            cwd=self.root,
            env={**self.env, **updates},
            capture_output=True,
            text=True,
            timeout=15,
        )

    def events(self):
        path = self.root / "docker.jsonl"
        return (
            [json.loads(line) for line in path.read_text().splitlines()]
            if path.exists()
            else []
        )

    def rejected(self, message, **updates):
        result = self.run_launcher(**updates)
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn(message, result.stderr)
        self.assertEqual(self.events(), [], "validation must precede Docker")

    def test_device_is_explicit_and_single_numeric(self):
        """Never turn absent, invalid or multi-card runner assignments into card 0."""
        for device in ("", "all", "void", "1,2", "-1", "1x"):
            with self.subTest(device=device):
                self.rejected("numeric device", PPU_CI_DEVICE=device)
        self.rejected("numeric device", PPU_CI_DEVICE="bad", NVIDIA_VISIBLE_DEVICES="2")

    def test_source_identity_includes_untracked_and_ignored_inputs(self):
        """A clean tracked diff alone is insufficient provenance for wheel builds."""
        self.rejected("HEAD", PPU_CI_SOURCE_SHA="c" * 40)
        for kind in ("tracked", "untracked", "ignored"):
            with self.subTest(kind=kind):
                marker = self.source / f"dirty-{kind}"
                marker.touch()
                self.rejected("pristine")
                marker.unlink()
        (self.wheels / "source-sha.txt").write_text("c" * 40 + "\n")
        self.manifest()
        self.rejected("source-sha.txt")

    def test_checksum_covers_content_and_exact_installable_inventory(self):
        """Checking listed hashes alone permits unlisted packages to be installed."""
        for name in ("injected.whl", ".hidden.whl", "runtime-deps/injected.whl"):
            with self.subTest(name=name):
                injected = self.wheels / name
                injected.write_bytes(b"injected")
                self.rejected("inventory")
                injected.unlink()
        for wheel in self.wheels.rglob("*.whl"):
            with self.subTest(wheel=wheel.name):
                original = wheel.read_bytes()
                wheel.write_bytes(b"tampered")
                self.rejected("checksum")
                wheel.write_bytes(original)
        manifest = self.wheels / "SHA256SUMS"
        original = manifest.read_text()
        manifest.write_text(original + original.splitlines(keepends=True)[0])
        self.rejected("inventory")
        manifest.write_text(original.replace("source-sha.txt", "../source-sha.txt"))
        self.rejected("inventory")

    def test_mounts_require_valid_inputs_and_no_source_overlap(self):
        """Canonicalize before containment checks; no writable mount hides source."""
        for path in (
            self.models[0] / "config.json",
            self.cache / "mmlu/test.jsonl.sha256",
        ):
            original = path.read_bytes()
            path.unlink()
            self.rejected(path.name)
            path.write_bytes(original)
        self.rejected("absolute", PPU_CI_WHEEL_DIR="wheel house")
        alias = self.root / "alias"
        alias.symlink_to(self.source, target_is_directory=True)
        self.rejected("outside source", PPU_CI_ARTIFACT_DIR=str(alias))
        self.rejected("outside source", PPU_CI_WHEEL_DIR=str(alias))
        comma = self.root / "bad,mount"
        comma.mkdir()
        self.rejected("commas", PPU_CI_ARTIFACT_DIR=str(comma))

    def test_model_and_cache_symlinks_stay_within_each_mount(self):
        """Docker remaps each root, so host-valid links can still break mounts."""
        for root in [*self.models, self.cache]:
            nested = root / "nested"
            nested.mkdir()
            (root / "weights.bin").write_bytes(b"internal")
            outside = root.with_name(root.name + "-outside")
            outside.mkdir()
            (outside / "weights.bin").write_bytes(b"external")
            other_mount = self.cache if root != self.cache else self.models[0]
            cases = (
                ("file-escape", os.path.relpath(outside / "weights.bin", nested)),
                ("absolute-file", str(root / "weights.bin")),
                ("absolute-directory", str(root)),
                ("directory-escape", os.path.relpath(other_mount, nested)),
                ("dangling", "../missing.bin"),
            )
            for kind, destination in cases:
                with self.subTest(root=root.name, link=kind):
                    link = nested / kind
                    link.symlink_to(destination)
                    try:
                        self.rejected("symlink")
                    finally:
                        link.unlink()
            (nested / "internal-file").symlink_to("../weights.bin")
            (nested / "internal-directory").symlink_to("..", target_is_directory=True)
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assert_lifecycle()

    def assert_lifecycle(self):
        events = self.events()
        self.assertTrue(events)
        self.assertTrue(all(event["lock_held"] for event in events), events)
        commands = [event["args"][0] for event in events]
        self.assertLess(commands.index("stop"), commands.index("rm"))
        self.assertLess(commands.index("rm"), commands.index("run"))
        create = next(event["args"] for event in events if event["args"][0] == "create")
        name = create[create.index("--name") + 1]
        for event in events:
            if event["args"][0] in ("stop", "rm"):
                self.assertEqual(event["args"][-1], name)
        ownership = events[-1]["args"]
        self.assertEqual(ownership[0], "run")
        self.assertIn(f"{os.getuid()}:{os.getgid()}", ownership)
        self.assertIn("--runtime=runc", ownership)
        self.assertIn("NVIDIA_VISIBLE_DEVICES=void", ownership)
        self.assertIn(IMAGE_ID, ownership)
        self.assertFalse((self.root / "container.json").exists())
        self.assertEqual(len(list(self.artifacts.glob("*/ownership-recovered"))), 1)
        with self.lock.open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return create

    def test_nonzero_device_uses_readonly_inputs_and_pinned_image_id(self):
        """A tag may move after inspect; an NVIDIA default runtime must not add GPUs."""
        result = self.run_launcher(MISSING_IMAGE="1")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        create = self.assert_lifecycle()
        self.assertIn("--runtime=runc", create)
        self.assertIn("NVIDIA_VISIBLE_DEVICES=void", create)
        self.assertIn("CUDA_VISIBLE_DEVICES=0", create)
        self.assertEqual(
            {arg for arg in create if arg.startswith("--device=")},
            {
                f"--device=/dev/{name}"
                for name in (
                    "alixpu",
                    "alixpu_ctl",
                    "alixpu_sep",
                    f"alixpu_ppu{self.device}",
                )
            },
        )
        self.assertEqual(
            create[-3:], [IMAGE_ID, "/source/scripts/ci/ppu/build_and_test.sh", "test"]
        )
        self.assertNotIn(self.env["PPU_CI_RUNTIME_IMAGE"], create)
        mounts = [create[i + 1] for i, arg in enumerate(create) if arg == "--mount"]
        expected = {
            "/source": self.source,
            "/wheels": self.wheels,
            "/models/target": self.models[0],
            "/models/eagle3": self.models[1],
            "/models/dflash": self.models[2],
            "/eval-cache": self.cache,
        }
        self.assertEqual(len(mounts), 7)
        for target, source in expected.items():
            self.assertIn(f"type=bind,source={source},target={target},readonly", mounts)
        private = next(mount for mount in mounts if "target=/artifacts" in mount)
        self.assertNotIn("readonly", private)
        self.assertIn(f"source={self.artifacts}/", private)
        for forbidden in (
            "--privileged",
            "--ipc=host",
            "--network=host",
            "docker.sock",
        ):
            self.assertNotIn(forbidden, " ".join(create))
        records = list(self.artifacts.glob("*/runtime-image-id.txt"))
        self.assertEqual([path.read_text().strip() for path in records], [IMAGE_ID])
        self.assertEqual(sum(event["args"][0] == "pull" for event in self.events()), 1)

    def test_numeric_runner_fallback_and_failure_status(self):
        """Cleanup and ownership recovery must not turn failed tests green."""
        result = self.run_launcher(
            PPU_CI_DEVICE="", NVIDIA_VISIBLE_DEVICES=self.device, TEST_EXIT="29"
        )
        self.assertEqual(result.returncode, 29, result.stdout + result.stderr)
        create = self.assert_lifecycle()
        self.assertIn(f"--device=/dev/alixpu_ppu{self.device}", create)
        self.assertFalse(any(event["args"][0] == "pull" for event in self.events()))

    def start_blocked(self):
        process = subprocess.Popen(
            ["bash", str(self.launcher)],
            cwd=self.root,
            env={
                **self.env,
                "BLOCK_TEST": "1",
                "RUNNER_TEMP": str(self.root / "runner-a"),
            },
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        def cleanup():
            if process.poll() is None:
                process.terminate()
                process.communicate(timeout=10)

        self.addCleanup(cleanup)
        readable, _, _ = select.select([self.ready], [], [], 10)
        self.assertTrue(readable, "launcher did not reach attached container start")
        self.assertEqual(os.read(self.ready, 100), b"started\n")
        return process

    def test_contention_does_not_touch_other_container(self):
        """Different RUNNER_TEMP values must still serialize the same physical card."""
        first = self.start_blocked()
        before = self.events()
        result = self.run_launcher(RUNNER_TEMP=str(self.root / "runner-b"))
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("busy", result.stderr)
        self.assertEqual(self.events(), before)
        self.assertIsNone(first.poll())
        first.terminate()
        first.communicate(timeout=10)
        self.assertEqual(first.returncode, 143)
        self.assert_lifecycle()

    def test_cancellation_cleans_up_before_unlocking(self):
        """The host must handle signals while Docker is attached, not after inference."""
        for sig, code in ((signal.SIGTERM, 143), (signal.SIGINT, 130)):
            with self.subTest(signal=sig):
                process = self.start_blocked()
                process.send_signal(sig)
                output, errors = process.communicate(timeout=10)
                self.assertEqual(process.returncode, code, output + errors)
                self.assert_lifecycle()
                (self.root / "docker.jsonl").unlink()
                for private in self.artifacts.iterdir():
                    shutil.rmtree(private)


if __name__ == "__main__":
    unittest.main()
