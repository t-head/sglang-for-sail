# SPDX-License-Identifier: Apache-2.0
"""Prepare the public candidate result from private run files."""

import json
import os
import sys
from pathlib import Path

from recipe_diagnostics import failure_reason
from recipe_retention import retain


def main():
    env = os.environ.copy()
    source = Path(env["PPU_CI_ARTIFACT_DIR"])
    outcome = env["RECIPE_OUTCOME"]
    try:
        retain(env)
        retained = True
    except (KeyError, OSError, ValueError, RuntimeError):
        retained = False
    result = {}
    runner_summary = source / "results/summary.json"
    if runner_summary.is_file() and not runner_summary.is_symlink():
        if runner_summary.stat().st_size > 8192:
            raise ValueError("unexpected result size")
        result = json.loads(runner_summary.read_text())
    if outcome == "success" and result.get("outcome") != "pass":
        outcome = "failure"
    if outcome != "success" or not retained:
        result["failure_phase"] = (
            failure_reason(source) if retained else "private_retention"
        )
    result.update(
        schema="ppu-sanity/v1",
        tier=env["TIER"],
        model=env["MODEL_KEY"],
        source_sha=env["PPU_CI_SOURCE_SHA"],
        private_retention=retained,
        route="native-build + sglang-release-runtime",
        outcome="pass" if outcome == "success" and retained else "fail",
    )
    provenance = source / "provenance.json"
    if provenance.is_file() and not provenance.is_symlink():
        record = json.loads(provenance.read_text())
        for key in ("runtime_image", "recipe_tree_sha256"):
            result[key] = record[key]
    Path(sys.argv[1]).write_text(json.dumps(result, allow_nan=False) + "\n")
    return int(result["outcome"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
