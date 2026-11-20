# SPDX-License-Identifier: Apache-2.0
"""Report fixed diagnostic codes from the task's retained private evidence."""

import json
import re

from recipe_publish import MODEL_POLICY_ROOT

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
    "deep_select_topk_unavailable": (
        r"deep_select[^\n]*(?:has no attribute ['\"]topk|cannot import[^\n]*topk)"
        r"|cannot import name ['\"]topk['\"][^\n]*deep_select"
    ),
    "device_kernel": r"no kernel image is available|invalid device function|illegal memory access",
    "model_assertion": r"\bAssertionError\b",
    "startup_timeout": r"server did not become ready|startup timed out",
    "worker_exit": r"container.+exit.?code|worker pods? failed|Command failed",
    "deadline": r"DeadlineExceeded|PPU scheduling timed out|workload timeout",
    "configuration_key": r"KeyError: 'PPU_CI_[A-Z0-9_]+'",
}


def diagnostic_codes(text):
    return [name for name, pattern in RULES.items() if re.search(pattern, text, re.I)]


def recent_failures(root, current_run):
    parent = root / "sglang-for-sail"
    if not parent.is_dir():
        return []
    previous = []
    for run in parent.iterdir():
        if not re.fullmatch(r"[0-9]+-[0-9]+", run.name):
            continue
        run_id, attempt = map(int, run.name.split("-"))
        if run_id != current_run:
            previous.append((run_id, attempt, run))
    models = sorted(
        policy.stem
        for policy in MODEL_POLICY_ROOT.glob("*.yaml")
        if re.fullmatch(r"[a-z0-9][a-z0-9.-]*", policy.stem)
    )
    for run_id, _, run in sorted(previous, reverse=True):
        records = []
        for tier in ("fast_sanity", "model_sanity"):
            for model in models:
                directory = run / tier / model
                if not (directory / "index.json").is_file():
                    continue
                code = []
                for filename in (
                    "launcher.log",
                    "runtime.log",
                    "pod-status.json",
                    "sglang_serve_fast_2l.log",
                    "sglang_serve_fast_full.log",
                    "sglang_serve_model_sanity.log",
                ):
                    path = directory / filename
                    if path.is_file() and not path.is_symlink():
                        with path.open("rb") as stream:
                            stream.seek(max(0, path.stat().st_size - 262144))
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


def publish_recent_failures(root, current_run):
    records = recent_failures(root, current_run)
    if records:
        print(json.dumps({"prior_cluster_diagnostics": records}, sort_keys=True))
