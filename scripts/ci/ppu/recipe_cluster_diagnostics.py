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
            if not stat.S_ISDIR(current.lstat().st_mode):
                return None
        except OSError:
            return None
    return current


def _previous_runs(parent, current_run):
    try:
        children = list(parent.iterdir())
    except OSError:
        return []
    previous = []
    for child in children:
        if not re.fullmatch(r"[0-9]+-[0-9]+", child.name):
            continue
        run_id, attempt = map(int, child.name.split("-"))
        if run_id != current_run and _safe_subdir(parent, (child.name,)) is not None:
            previous.append((run_id, attempt, child))
    return sorted(previous, reverse=True)


def _archive_roots(run):
    yield run
    try:
        children = list(run.iterdir())
    except OSError:
        return
    for child in children:
        if re.fullmatch(r"[0-9a-f]{12}", child.name):
            scope = _safe_subdir(run, (child.name,))
            if scope is not None:
                yield scope


def _retained_models(run, models):
    for scope in _archive_roots(run):
        for tier in ("fast_sanity", "model_sanity"):
            tier_dir = _safe_subdir(scope, (tier,))
            if tier_dir is None:
                continue
            for model in models:
                directory = _safe_subdir(tier_dir, (model,))
                if directory is None:
                    continue
                try:
                    index_info = (directory / "index.json").lstat()
                except OSError:
                    continue
                if stat.S_ISREG(index_info.st_mode):
                    yield tier, model, directory


def _archive_codes(directory):
    codes = set()
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
            codes.update(diagnostic_codes(stream.read(262144).decode(errors="replace")))
    return codes


def recent_failures(root, current_run, fallback_root=None):
    roots = [root]
    if fallback_root is not None and fallback_root != root:
        roots.append(fallback_root)
    models = sorted(
        policy.stem
        for policy in MODEL_POLICY_ROOT.glob("*.yaml")
        if re.fullmatch(r"[a-z0-9][a-z0-9.-]*", policy.stem)
    )
    for candidate_root in roots:
        try:
            if not stat.S_ISDIR(candidate_root.lstat().st_mode):
                continue
        except OSError:
            continue
        for parts in (RETENTION_SUBPATH, ("sglang-for-sail",)):
            parent = _safe_subdir(candidate_root, parts)
            if parent is None:
                continue
            found_archive = False
            for run_id, _, run in _previous_runs(parent, current_run):
                codes_by_model = {}
                for tier, model, directory in _retained_models(run, models):
                    found_archive = True
                    codes = _archive_codes(directory)
                    if codes:
                        codes_by_model.setdefault((tier, model), set()).update(codes)
                if codes_by_model:
                    return [
                        {
                            "run_id": run_id,
                            "tier": tier,
                            "model": model,
                            "codes": sorted(codes),
                        }
                        for (tier, model), codes in sorted(codes_by_model.items())
                    ]
            if found_archive:
                return []
    return []


def publish_recent_failures(root, current_run, fallback_root=None):
    records = recent_failures(root, current_run, fallback_root=fallback_root)
    if records:
        print(json.dumps({"prior_cluster_diagnostics": records}, sort_keys=True))
