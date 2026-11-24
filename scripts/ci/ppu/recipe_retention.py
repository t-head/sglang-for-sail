# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm-project/vllm-sail@0163924 (Apache-2.0).

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from pathlib import Path

RESULT_FILES = ("results/summary.json", "provenance.json")
OPTIONAL_FILES = (
    "runtime.log",
    "scheduler-output",
    "scheduler-env",
    "scheduler-summary",
    "pod-status.json",
    "ownership.log",
    "image-pull.log",
    "ppu-memory-before.log",
    "ppu-memory-after.log",
    "ppu-inventory.log",
    "ppu-before.log",
    "ppu-processes-before.log",
    "ppu-after.log",
    "ppu-processes-after.log",
    "dependencies-before.json",
    "dependencies-after.json",
    *(
        f"results/bench-{phase}-{concurrency}.{ext}"
        for phase in ("warmup", "measure")
        for concurrency in (1, 16)
        for ext in ("log", "jsonl")
    ),
    "recipe-phase.txt",
    "worker-phase.txt",
    "results/chat-probe-response.json",
    "results/evalscope.log",
    "results/gsm8k-result.json",
    "results/gsm8k_per_question.jsonl",
    "results/sglang_serve_fast_2l.log",
    "results/sglang_serve_fast_full.log",
    "results/sglang_serve_model_sanity.log",
)


RETENTION_SUBPATH = ("sglang", "logs", "sglang-for-sail")


def _check_dir_mode(directory: Path, *allowed_modes: int) -> None:
    mode = directory.lstat().st_mode
    if not stat.S_ISDIR(mode) or stat.S_IMODE(mode) not in allowed_modes:
        raise ValueError("unexpected retention directory type or mode")


def validate_retention_root(raw_root: str, environ) -> Path:
    if not raw_root or any(ord(char) < 32 for char in raw_root):
        raise ValueError("expected an absolute retention root")
    root = Path(raw_root)
    if not root.is_absolute() or ".." in root.parts:
        raise ValueError("expected an absolute retention root")
    _check_dir_mode(root, 0o700, 0o755)
    root = root.resolve(strict=True)
    archive_base = root.joinpath(*RETENTION_SUBPATH)
    for key in ("RUNNER_TEMP", "GITHUB_WORKSPACE"):
        raw_ephemeral = environ.get(key)
        if not raw_ephemeral:
            raise ValueError("missing runner workspace path")
        ephemeral = Path(raw_ephemeral).resolve()
        if (
            ephemeral == root
            or ephemeral in root.parents
            or ephemeral == archive_base
            or ephemeral in archive_base.parents
            or archive_base in ephemeral.parents
        ):
            raise ValueError("retention root must not be job-local")
    return root


def ensure_retention_base(root: Path) -> Path:
    parent = root
    for component in RETENTION_SUBPATH:
        parent /= component
        try:
            _check_dir_mode(parent, 0o700, 0o755)
        except FileNotFoundError:
            parent.mkdir(mode=0o700, exist_ok=True)
            _check_dir_mode(parent, 0o700, 0o755)
    return parent


def probe_retention_writable(base_dir: Path) -> None:
    fd, probe_name = tempfile.mkstemp(prefix=".probe-", dir=base_dir)
    probe_path = Path(probe_name)
    closed = False
    try:
        os.fchmod(fd, 0o600)
        if os.write(fd, b"ok\n") != 3:
            raise OSError("short retention probe write")
        os.close(fd)
        closed = True
    finally:
        if not closed:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            probe_path.unlink()
        except FileNotFoundError:
            pass


def retain(environ) -> None:
    index = {
        "checkout_sha": environ["PPU_CI_SOURCE_SHA"],
        "tier": environ["TIER"],
        "model_key": environ["MODEL_KEY"],
        "run_id": environ["GITHUB_RUN_ID"],
        "attempt": environ["GITHUB_RUN_ATTEMPT"],
    }
    if (
        index["tier"] not in {"fast_sanity", "model_sanity"}
        or not re.fullmatch(r"[a-z0-9][a-z0-9.-]*", index["model_key"])
        or not re.fullmatch(r"[0-9a-f]{40}", index["checkout_sha"])
        or not all(re.fullmatch(r"[0-9]+", index[key]) for key in ("run_id", "attempt"))
    ):
        raise ValueError("invalid retention metadata")

    root = validate_retention_root(environ["PPU_CI_PRIVATE_RESULTS_ROOT"], environ)
    parent = ensure_retention_base(root)
    for component in (
        f"{index['run_id']}-{index['attempt']}",
        index["tier"],
    ):
        parent /= component
        try:
            _check_dir_mode(parent, 0o700)
        except FileNotFoundError:
            parent.mkdir(mode=0o700, exist_ok=True)
            _check_dir_mode(parent, 0o700)

    destination = parent / index["model_key"]
    if os.path.lexists(destination):
        raise FileExistsError("retention destination already exists")

    source_dir = Path(environ["PPU_CI_ARTIFACT_DIR"])
    partial = environ.get("RECIPE_OUTCOME") in ("failure", "cancelled")
    try:
        if not stat.S_ISDIR((source_dir / "results").lstat().st_mode):
            raise ValueError("expected a recipe results directory")
    except FileNotFoundError:
        if not partial:
            raise
    required = {"launcher.log"}
    if not partial:
        required.update(RESULT_FILES)
    entries = ["launcher.log", *RESULT_FILES, *OPTIONAL_FILES]
    review_root = source_dir / "results/evalscope"
    if review_root.is_dir() and not review_root.is_symlink():
        entries.extend(
            path.relative_to(source_dir).as_posix()
            for path in sorted(review_root.rglob("*"))
            if path.is_file() and path.suffix in (".json", ".jsonl", ".log")
        )
    with tempfile.TemporaryDirectory(prefix=".staging-", dir=parent) as staging_dir:
        staging = Path(staging_dir)
        published = staging / index["model_key"]
        published.mkdir(mode=0o700)
        files = {}
        for relative_path in entries:
            source = source_dir / relative_path
            name = (
                relative_path
                if relative_path.startswith("results/evalscope/")
                else source.name
            )
            try:
                mode = source.lstat().st_mode
            except FileNotFoundError:
                if relative_path not in required:
                    continue
                raise
            if not stat.S_ISREG(mode):
                raise ValueError("expected a regular recipe output")
            copied = published / name
            copied.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            shutil.copyfile(source, copied)
            copied.chmod(0o600)
            digest = hashlib.sha256()
            with copied.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            files[name] = digest.hexdigest()
        index_path = published / "index.json"
        index_path.write_text(
            json.dumps({**index, "files": files}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        index_path.chmod(0o600)
        published.rename(destination)
