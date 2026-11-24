# SPDX-License-Identifier: Apache-2.0
"""Describe failure phases with fixed public codes only."""

PHASES = {
    "runner_preflight",
    "cluster_submit",
    "cluster_worker",
    "wheel_verification",
    "image_resolution",
    "multinode_network",
    "runtime_dependencies",
    "eval_environment",
    "platform_check",
    "recipe_validation",
    "model_cache_check",
    "page_cache_warm",
    "server_startup",
    "server_exit",
    "worker_exit",
    "readiness_timeout",
    "chat_probe",
    "gsm8k_eval",
    "cache_flush",
    "micro_e2e",
    "cleanup",
    "private_retention",
}


def _read_phase(path):
    if path.is_file() and not path.is_symlink() and path.stat().st_size < 64:
        phase = path.read_text(encoding="ascii", errors="replace").strip()
        if phase in PHASES:
            return phase
    return None


def _worker_rank_phase(source):
    ranks_dir = source / "ranks"
    if not ranks_dir.is_dir() or ranks_dir.is_symlink():
        return None
    for rank_dir in sorted(ranks_dir.glob("rank-*")):
        if not rank_dir.is_dir() or rank_dir.is_symlink():
            continue
        found = _read_phase(rank_dir / "worker-phase.txt")
        if found:
            return found
    return None


def failure_reason(source):
    launcher = _read_phase(source / "recipe-phase.txt")
    if launcher in {"cleanup", "image_resolution"}:
        return launcher
    return (
        _read_phase(source / "worker-phase.txt")
        or _worker_rank_phase(source)
        or launcher
        or "unknown"
    )
