# SPDX-License-Identifier: Apache-2.0
"""Describe failure phases with fixed public codes only."""

PHASES = {
    "runner_preflight",
    "cluster_submit",
    "cluster_worker",
    "wheel_verification",
    "image_resolution",
    "runtime_dependencies",
    "platform_check",
    "recipe_validation",
    "server_startup",
    "chat_probe",
    "gsm8k_eval",
    "cache_flush",
    "micro_e2e",
    "cleanup",
    "private_retention",
}


def failure_reason(source):
    path = source / "recipe-phase.txt"
    if path.is_file() and not path.is_symlink() and path.stat().st_size < 64:
        phase = path.read_text(encoding="ascii", errors="replace").strip()
        if phase in PHASES:
            return phase
    return "unknown"
