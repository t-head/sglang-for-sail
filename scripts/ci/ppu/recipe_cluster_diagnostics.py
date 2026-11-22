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
    "cuda_oom": r"CUDA out of memory|OutOfMemoryError|Not enough memory",
    "wheel_install": r"not a supported wheel|No matching distribution|Failed building wheel",
    "python_import": r"ModuleNotFoundError|ImportError|undefined symbol|No module named",
    "invalid_server_args": r"unrecognized arguments|argument .+: invalid choice",
    "deepep_dispatch_limit": r"num_max_dispatch_tokens_per_rank|SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK",
    "deepep_error": r"deep_ep|deepep|DeepEP",
    "watchdog": r"watchdog|Watchdog",
    "nccl_error": r"NCCL error|DistBackendError|TCPStore|ProcessGroupNCCL",
    "assertion_error": r"AssertionError",
    "value_error": r"ValueError:",
    "runtime_error": r"RuntimeError:",
    "attribute_error": r"AttributeError:",
    "type_error": r"TypeError:",
    "key_error": r"KeyError:",
    "sigkill": r"Signals\.SIGKILL|exitcode: -9|exit code 137",
    "sigabrt": r"Signals\.SIGABRT|exitcode: -6|Segmentation fault|exit code (134|139)",
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
                    "sglang_serve_fast_full.log",
                    "sglang_serve_fast_2l.log",
                    "sglang_serve_model_sanity.log",
                    "bench-warmup-1.log",
                    "bench-measure-1.log",
                    "bench-warmup-16.log",
                    "bench-measure-16.log",
                    "evalscope.log",
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
