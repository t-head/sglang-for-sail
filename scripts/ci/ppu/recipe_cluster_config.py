# SPDX-License-Identifier: Apache-2.0
"""Load task-scoped Cluster Run configuration without emitting its values."""

import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

BACKEND_SHA = "96e25147971f5d2a04d828f8ad508cae38a4fa8a"
CONFIG_NAME = ".sglang-recipe-ci-v0523"
SETTINGS = {
    "PPU_CI_K8S_STAGE_ROOT",
    "PPU_CI_K8S_PRIVATE_WORKER_ROOT",
    "PPU_CI_PRIVATE_RESULTS_ROOT",
    "PPU_CI_K8S_RUNTIME_DIGEST",
    "PPU_CI_RUNTIME_IMAGE",
    "PPU_CI_RECIPE_IMAGE_REPO",
    "PPU_CI_RECIPE_ALLOWED_IMAGE_REPOS",
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


def resolve_retention_root(environ, values, fallback):
    env_root = (environ.get("PPU_CI_PRIVATE_RESULTS_ROOT") or "").strip()
    if env_root:
        return env_root, "env"
    config_root = (values.get("PPU_CI_PRIVATE_RESULTS_ROOT") or "").strip()
    if config_root:
        return config_root, "config"
    return str(fallback), "fallback"


def _verify_retention_root(retention_root_raw, retention_source, environ):
    from recipe_retention import (
        ensure_retention_base,
        probe_retention_writable,
        validate_retention_root,
    )

    status = {
        "external_retention_root": retention_source != "fallback",
        "retention_root_valid": False,
        "retention_source": retention_source,
        "retention_writable": False,
    }
    try:
        validated_root = validate_retention_root(retention_root_raw, environ)
        status["retention_root_valid"] = True
        retention_base = ensure_retention_base(validated_root)
        probe_retention_writable(retention_base)
        status["retention_writable"] = True
    except (KeyError, TypeError, ValueError, OSError):
        print(json.dumps({"private_retention_check": status}, sort_keys=True))
        raise
    return validated_root, status


def main():
    backend = Path(os.environ["GITHUB_WORKSPACE"]) / ".ppu-infra"
    head = subprocess.check_output(
        ["git", "-C", str(backend), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
    )
    if head.decode().strip() != BACKEND_SHA:
        raise ValueError("Cluster Run revision mismatch")
    directory = staging_root(backend) / CONFIG_NAME
    expected = os.environ["PPU_CI_K8S_CONFIG_SHA256"]
    values = load_settings(directory, expected)
    retained = directory / "output/retained"
    retained.mkdir(mode=0o700, exist_ok=True)
    info = retained.lstat()
    if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o700:
        raise ValueError("invalid private results directory")
    if info.st_uid != os.geteuid():
        raise ValueError("the runner does not own the private results directory")
    retention_root_raw, retention_source = resolve_retention_root(
        os.environ, values, retained
    )
    if sys.argv[1:] == ["--check"]:
        from recipe_cluster_diagnostics import publish_recent_failures

        validated_root, status = _verify_retention_root(
            retention_root_raw, retention_source, os.environ
        )
        print("PPU cluster private configuration verified")
        print(json.dumps({"private_retention_check": status}, sort_keys=True))
        publish_recent_failures(
            validated_root,
            int(os.environ["GITHUB_RUN_ID"]),
            fallback_root=retained.resolve(strict=True),
        )
        return
    if os.environ.get("TIER") != "legacy":
        _verify_retention_root(retention_root_raw, retention_source, os.environ)
    os.execvpe(
        "bash",
        ["bash", "scripts/ci/ppu/run_recipe_cluster.sh"],
        {
            **os.environ,
            **values,
            "PPU_CI_K8S_HOST_SCRATCH_ROOT": str(retained),
            "PPU_CI_PRIVATE_RESULTS_ROOT": retention_root_raw,
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
