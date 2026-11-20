# SPDX-License-Identifier: Apache-2.0
"""Load task-scoped Cluster Run configuration without emitting its values."""

import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

BACKEND_SHA = "96e25147971f5d2a04d828f8ad508cae38a4fa8a"
CONFIG_NAME = ".sglang-recipe-ci-v0523"
SETTINGS = {
    "PPU_CI_K8S_STAGE_ROOT",
    "PPU_CI_K8S_PRIVATE_WORKER_ROOT",
    "PPU_CI_K8S_RUNTIME_DIGEST",
    "PPU_CI_RUNTIME_IMAGE",
    "PPU_CI_JIT_ROOT",
    "PPU_CI_HF_CACHE_ROOT",
    "PPU_CI_GLM53_MODEL_REVISION",
    "PPU_CI_DSV4_FLASH_MODEL_REVISION",
    "PPU_CI_GLM53_MODEL_PATH",
    "PPU_CI_DSV4_FLASH_MODEL_PATH",
    "PPU_CI_GSM8K_PATH",
    "PPU_CI_TARGET_MODEL_PATH",
    "PPU_CI_EAGLE3_MODEL_PATH",
    "PPU_CI_DFLASH_MODEL_PATH",
    "PPU_CI_EVAL_CACHE",
}


def staging_root(backend):
    # The pinned Action defines the shared staging mount. Read its contract
    # locally so the mount path never needs to appear in this repository.
    manifest = (backend / "cluster-run/action.yml").read_text(encoding="utf-8")
    (root,) = re.findall(
        r'^\s*STAGE_TARBALL="(/[A-Za-z0-9_./-]+)/\$\{GITHUB_RUN_ID\}-',
        manifest,
        re.MULTILINE,
    )
    return Path(root)


def load_settings(directory, expected):
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("invalid configuration digest")
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        if os.fstat(fd).st_mode & 0o007:
            raise ValueError("configuration directory is accessible to others")
        # Keep existing workflow runs pinned to their original configuration.
        try:
            handle = os.open(
                f"config.{expected}.json", os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd
            )
        except FileNotFoundError:
            handle = os.open("config.json", os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
        with os.fdopen(handle, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o007:
                raise ValueError("invalid configuration permissions")
            payload = stream.read(8193)
    finally:
        os.close(fd)
    if len(payload) > 8192 or hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError("configuration digest mismatch")
    document = json.loads(payload)
    values = document["settings"]
    if set(values) - SETTINGS or not all(isinstance(v, str) for v in values.values()):
        raise ValueError("unknown configuration key")
    return values


def handoff(directory, expected, uid, gid):
    """Give the CPU runner only the three provisioned task objects."""
    if not directory.is_absolute() or directory.name != CONFIG_NAME or uid != 1001:
        raise ValueError("invalid private configuration handoff")
    if gid < 0:
        raise ValueError("invalid runner group")
    load_settings(directory, expected)
    parent = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        config = os.open("config.json", os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
        try:
            output = os.open(
                "output", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent
            )
            try:
                metadata = [os.fstat(fd) for fd in (config, output, parent)]
                if [stat.S_IMODE(info.st_mode) for info in metadata] != [
                    0o600,
                    0o700,
                    0o700,
                ]:
                    raise ValueError("handoff expects the original task-private modes")
                if len({info.st_uid for info in metadata} - {uid}) > 1:
                    raise ValueError("handoff has unexpected mixed ownership")
                with open(config, "rb", closefd=False) as stream:
                    if hashlib.sha256(stream.read(8193)).hexdigest() != expected:
                        raise ValueError("handoff digest mismatch")
                for fd in (config, output, parent):
                    os.fchown(fd, uid, gid)
            finally:
                os.close(output)
        finally:
            os.close(config)
    finally:
        os.close(parent)


def stage_cache_config(directory, current_digest, selection_digest, uid, gid):
    """Stage a verified, independently addressable task configuration."""
    if (
        not directory.is_absolute()
        or directory.name != CONFIG_NAME
        or uid != 1001
        or gid < 0
        or not re.fullmatch(r"[0-9a-f]{64}", selection_digest)
    ):
        raise ValueError("invalid task configuration staging parameters")
    values = load_settings(directory, current_digest)
    parent = os.open(
        directory.with_name(CONFIG_NAME + "-operator"),
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    try:
        operator = os.fstat(parent)
        if stat.S_IMODE(operator.st_mode) != 0o700:
            raise ValueError("operator directory is not private")
        selected = os.open(
            "hf-cache-selection.json", os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent
        )
        with os.fdopen(selected, "rb") as stream:
            info = os.fstat(stream.fileno())
            choice = stream.read(8193)
            if (
                not stat.S_ISREG(info.st_mode)
                or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_uid != operator.st_uid
                or len(choice) > 8192
                or hashlib.sha256(choice).hexdigest() != selection_digest
            ):
                raise ValueError("operator selection did not match its seal")
    finally:
        os.close(parent)

    selection = json.loads(choice)["settings"]
    revision_keys = {"PPU_CI_GLM53_MODEL_REVISION", "PPU_CI_DSV4_FLASH_MODEL_REVISION"}
    if set(selection) != revision_keys | {"PPU_CI_HF_CACHE_ROOT"} or not all(
        re.fullmatch(r"[0-9a-f]{40}", selection[k]) for k in revision_keys
    ):
        raise ValueError("invalid cache selection")
    cache = Path(selection["PPU_CI_HF_CACHE_ROOT"])
    if not cache.is_absolute() or ".." in cache.parts or not (cache / "hub").is_dir():
        raise ValueError("invalid cache root")
    previous_paths = {
        name: values.pop(key)
        for name, key in (
            ("glm-5.3", "PPU_CI_GLM53_MODEL_PATH"),
            ("deepseek-v4-flash", "PPU_CI_DSV4_FLASH_MODEL_PATH"),
        )
    }
    values.update(selection)
    payload = json.dumps({"settings": values}, sort_keys=True).encode()
    digest = hashlib.sha256(payload).hexdigest()
    target = directory / f"config.{digest}.json"
    fd, staged = tempfile.mkstemp(prefix=".config.", dir=directory)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
            os.fchown(output.fileno(), uid, gid)
        try:
            os.link(staged, target, follow_symlinks=False)
        except FileExistsError:
            pass
        if load_settings(directory, digest) != values:
            raise ValueError("staged configuration failed digest validation")
    finally:
        os.unlink(staged)
    # Return only the two old checkpoint paths to the authorized task operator.
    destination = directory.with_name(CONFIG_NAME + "-operator")
    parent = os.open(destination, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        info = os.fstat(parent)
        if info.st_uid != operator.st_uid or stat.S_IMODE(info.st_mode) != 0o700:
            raise ValueError("operator directory changed")
        receipt = "hf-cache-previous-paths.json"
        evidence = json.dumps(previous_paths, sort_keys=True).encode()
        try:
            fd = os.open(
                receipt,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=parent,
            )
        except FileExistsError:
            fd = os.open(receipt, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
            with os.fdopen(fd, "rb") as stream:
                if stream.read(8193) != evidence:
                    raise ValueError("previous checkpoint receipt already differs")
        else:
            with os.fdopen(fd, "wb") as stream:
                stream.write(evidence)
                stream.flush()
                os.fchown(stream.fileno(), operator.st_uid, operator.st_gid)
    finally:
        os.close(parent)
    print(digest)


def main():
    if len(sys.argv) == 7 and sys.argv[1] == "--stage-cache-config":
        stage_cache_config(
            Path(sys.argv[2]),
            sys.argv[3],
            sys.argv[4],
            int(sys.argv[5]),
            int(sys.argv[6]),
        )
        return
    if len(sys.argv) == 6 and sys.argv[1] == "--handoff":
        handoff(Path(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))
        return
    backend = Path(os.environ["GITHUB_WORKSPACE"]) / ".ppu-infra"
    head = subprocess.check_output(
        ["git", "-C", str(backend), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
    )
    if head.decode().strip() != BACKEND_SHA:
        raise ValueError("Cluster Run revision mismatch")
    directory = staging_root(backend) / CONFIG_NAME
    expected = os.environ["PPU_CI_K8S_CONFIG_SHA256"]
    try:
        values = load_settings(directory, expected)
    except PermissionError:
        # The operator initially owns the task directory. The runner container's
        # documented sudo capability transfers only that authenticated directory,
        # its manifest and its output parent once.
        subprocess.run(
            [
                "sudo",
                "-n",
                "--",
                sys.executable,
                "-I",
                "-B",
                str(Path(__file__).resolve()),
                "--handoff",
                str(directory),
                expected,
                str(os.geteuid()),
                str(os.getegid()),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        values = load_settings(directory, expected)
    retained = directory / "output/retained"
    retained.mkdir(mode=0o700, exist_ok=True)
    info = retained.lstat()
    if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o700:
        raise ValueError("invalid private results directory")
    if info.st_uid != os.geteuid():
        raise ValueError("the runner does not own the private results directory")
    if sys.argv[1:] == ["--check"]:
        from recipe_cluster_diagnostics import publish_recent_failures

        print("PPU cluster private configuration verified")
        publish_recent_failures(retained, int(os.environ["GITHUB_RUN_ID"]))
        selection_digest = os.environ.get("PPU_CACHE_SELECTION_SHA256")
        if selection_digest:
            digest = subprocess.check_output(
                [
                    "sudo",
                    "-n",
                    "--",
                    sys.executable,
                    "-I",
                    "-B",
                    str(Path(__file__).resolve()),
                    "--stage-cache-config",
                    str(directory),
                    expected,
                    selection_digest,
                    str(os.geteuid()),
                    str(os.getegid()),
                ],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError("staged configuration seal was invalid")
            load_settings(directory, digest)
            print(f"Staged PPU task configuration digest: {digest}")
        return
    os.execvpe(
        "bash",
        ["bash", "scripts/ci/ppu/run_recipe_cluster.sh"],
        {
            **os.environ,
            **values,
            "PPU_CI_PRIVATE_RESULTS_ROOT": str(retained),
            "PPU_CI_K8S_LOADED_CONFIG_SHA256": expected,
        },
    )


if __name__ == "__main__":
    try:
        main()
    except (KeyError, TypeError, ValueError, OSError, subprocess.SubprocessError):
        print(
            "::error::PPU cluster private configuration is unavailable or invalid",
            file=sys.stderr,
        )
        raise SystemExit(2) from None
