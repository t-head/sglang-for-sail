# SPDX-License-Identifier: Apache-2.0
"""Report fixed diagnostic codes from the task's retained private evidence."""

import json
import os
import re
import stat

from recipe_publish import MODEL_POLICY_ROOT
from recipe_retention import RETENTION_SUBPATH

RULES = {
    "image_pull": r"ErrImagePull|ImagePullBackOff|failed to pull image|manifest unknown",
    "registry_auth": r"status code: (401|403)|pull access denied|authorization failed|insufficient_scope",
    "checkout_ownership": r"detected dubious ownership",
    "source_archive": r"tar: .*Cannot open|source archive.*(missing|not found)",
    "permission_denied": r"permission denied|operation not permitted",
    "capacity_ppu": r"Insufficient .*ppu",
    "capacity_cpu": r"Insufficient cpu",
    "capacity_memory": r"Insufficient memory",
    "worker_oom": r"OOMKilled|out of memory",
    "wheel_install": r"not a supported wheel|No matching distribution|Failed building wheel",
    "python_import": r"ModuleNotFoundError|ImportError|undefined symbol",
    "invalid_server_args": r"unrecognized arguments|argument .+: invalid choice",
    "startup_timeout": r"server did not become ready|startup timed out",
    "worker_exit": r"container.+exit.?code|worker pods? failed|Command failed",
    "deadline": r"DeadlineExceeded|PPU scheduling timed out|workload timeout",
    "configuration_key": r"KeyError: 'PPU_CI_[A-Z0-9_]+'",
}


def diagnostic_codes(text):
    return [name for name, pattern in RULES.items() if re.search(pattern, text, re.I)]


def _safe_subdir(root, parts):
    current = root
    for part in parts:
        current = current / part
        try:
            if current.is_symlink() or not current.is_dir():
                return None
        except OSError:
            return None
    return current


def recent_failures(root, current_run, fallback_root=None):
    roots = [root]
    if fallback_root is not None and fallback_root != root:
        roots.append(fallback_root)
    previous = []
    for candidate_root in roots:
        try:
            if candidate_root.is_symlink() or not candidate_root.is_dir():
                continue
        except OSError:
            continue
        for parts in (RETENTION_SUBPATH, ("sglang-for-sail",)):
            parent = _safe_subdir(candidate_root, parts)
            if parent is None:
                continue
            try:
                runs = list(parent.iterdir())
            except OSError:
                continue
            for run in runs:
                try:
                    if run.is_symlink() or not run.is_dir():
                        continue
                except OSError:
                    continue
                if not re.fullmatch(r"[0-9]+-[0-9]+", run.name):
                    continue
                run_id, attempt = map(int, run.name.split("-"))
                if run_id != current_run:
                    previous.append((run_id, attempt, run))
            if previous:
                break
        if previous:
            break
    if not previous:
        return []
    models = sorted(
        policy.stem
        for policy in MODEL_POLICY_ROOT.glob("*.yaml")
        if re.fullmatch(r"[a-z0-9][a-z0-9.-]*", policy.stem)
    )
    for run_id, _, run in sorted(previous, reverse=True):
        records = []
        for tier in ("fast_sanity", "model_sanity"):
            for model in models:
                directory = _safe_subdir(run, (tier, model))
                if directory is None:
                    continue
                index_path = directory / "index.json"
                try:
                    index_info = index_path.lstat()
                except OSError:
                    continue
                if not stat.S_ISREG(index_info.st_mode):
                    continue
                code = []
                for filename in ("launcher.log", "runtime.log", "pod-status.json"):
                    path = directory / filename
                    try:
                        pre_info = path.lstat()
                    except OSError:
                        continue
                    if not stat.S_ISREG(pre_info.st_mode):
                        continue
                    try:
                        fd = os.open(
                            path,
                            os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0),
                        )
                    except OSError:
                        continue
                    try:
                        info = os.fstat(fd)
                        if not stat.S_ISREG(info.st_mode):
                            os.close(fd)
                            continue
                        stream = os.fdopen(fd, "rb")
                    except OSError:
                        os.close(fd)
                        continue
                    with stream:
                        stream.seek(max(0, info.st_size - 262144))
                        code.extend(
                            diagnostic_codes(
                                stream.read(262144).decode(errors="replace")
                            )
                        )
                if code:
                    records.append(
                        {
                            "run_id": run_id,
                            "tier": tier,
                            "model": model,
                            "codes": sorted(set(code)),
                        }
                    )
        if records:
            return records
    return []


def publish_recent_failures(root, current_run, fallback_root=None):
    records = recent_failures(root, current_run, fallback_root=fallback_root)
    if records:
        print(json.dumps({"prior_cluster_diagnostics": records}, sort_keys=True))
