# SPDX-License-Identifier: Apache-2.0
"""Describe failure phases with fixed public codes only."""

PHASES = {
    "runner_preflight",
    "cluster_submit",
    "cluster_worker",
    "wheel_verification",
    "image_resolution",
    "runtime_dependencies",
    "eval_environment",
    "platform_check",
    "recipe_validation",
    "model_cache_check",
    "server_startup",
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


def failure_reason(source):
    launcher = _read_phase(source / "recipe-phase.txt")
    if launcher in {"cleanup", "image_resolution"}:
        return launcher
    return _read_phase(source / "worker-phase.txt") or launcher or "unknown"
