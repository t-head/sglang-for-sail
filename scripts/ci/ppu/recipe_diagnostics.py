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


_IN_PROGRESS_PHASES = {
    "wheel_verification",
    "runtime_dependencies",
    "eval_environment",
    "platform_check",
    "recipe_validation",
    "model_cache_check",
    "page_cache_warm",
    "server_startup",
    "worker_exit",
}


def _read_status(path):
    if path.is_file() and not path.is_symlink() and path.stat().st_size < 32:
        return path.read_text(encoding="ascii", errors="replace").strip()
    return None


def _failed_worker_rank_phase(source):
    ranks_dir = source / "ranks"
    if not ranks_dir.is_dir() or ranks_dir.is_symlink():
        return None
    barrier_dir = ranks_dir / "barrier"
    failed_ranks = set()
    if barrier_dir.is_dir() and not barrier_dir.is_symlink():
        for marker in sorted(barrier_dir.glob("fail-rank-*.json")):
            if not marker.is_file() or marker.is_symlink():
                continue
            rank_str = marker.stem.removeprefix("fail-rank-")
            if rank_str.isdigit() and int(rank_str) > 0:
                failed_ranks.add(int(rank_str))
    for status_file in sorted(ranks_dir.glob("rank-*.status")):
        rank_str = status_file.stem.removeprefix("rank-")
        if not rank_str.isdigit() or int(rank_str) <= 0:
            continue
        val = _read_status(status_file)
        if val and val != "0":
            failed_ranks.add(int(rank_str))
    for rank in sorted(failed_ranks):
        rank_dir = ranks_dir / f"rank-{rank}"
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
    rank0_phase = _read_phase(source / "worker-phase.txt")
    peer_failed = _failed_worker_rank_phase(source)
    if peer_failed and (rank0_phase is None or rank0_phase in _IN_PROGRESS_PHASES):
        return peer_failed
    return rank0_phase or _worker_rank_phase(source) or launcher or "unknown"
