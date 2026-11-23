# SPDX-License-Identifier: Apache-2.0
"""Join SGLang Nightly job conclusions to the checked public Recipe evidence."""

import json
import os
from functools import partial

from ppu_notify_summary import MODEL, json_file, model_result, run_jobs
from recipe_publish import validate

STARTUP = ("server_startup", "server_exit", "readiness_timeout", "chat_probe")
SETUP = (
    "runner_preflight",
    "cluster_submit",
    "cluster_worker",
    "wheel_verification",
    "image_resolution",
    "runtime_dependencies",
    "eval_environment",
    "platform_check",
    "recipe_validation",
)


def evidence(root, model, attempt):
    report = json_file(root / f"model_sanity-{model}-{attempt}" / "recipe.json")
    validate(report)
    if (
        report.get("model") != model
        or report.get("tier") != "model_sanity"
        or report.get("source_sha") != os.environ.get("NOTIFY_SOURCE_SHA")
    ):
        raise ValueError("Recipe identity mismatch")
    phase = report.get("failure_phase", "unknown")
    if phase in STARTUP:
        phase = "startup"
    elif phase == "gsm8k_eval" or report.get("accuracy_within_band") is False:
        phase = "gsm8k_score vs golden"
    elif phase in SETUP:
        phase = "runtime_setup"
    return (
        report["outcome"],
        phase,
        report.get("gsm8k_score"),
        report.get("golden_score"),
    )


def failure(model, phase):
    return {"model": model, "status": "fail", "phase": phase}


def collect(root):
    matrix = json.loads(os.environ.get("NOTIFY_MATRIX") or '{"include":[]}')
    models = [item["model"] for item in matrix["include"]]
    if len(models) != len(set(models)) or any(
        not MODEL.fullmatch(key) for key in models
    ):
        raise ValueError("Invalid model selection")

    # The scheduled caller names its reusable workflow job "release-v100";
    # direct dispatch exposes the inner job name without that prefix.
    workflow = os.environ.get("GITHUB_WORKFLOW_REF", "")
    prefix = "release-v100 / " if "/nightly-dispatch-ppu.yml@" in workflow else ""
    names = {f"{prefix}Model Sanity ({model})": model for model in models}
    jobs, elapsed = run_jobs(set(names))
    if os.environ["NOTIFY_GATE_RESULT"] != "success":
        results = [failure("approval", "approval")]
    elif os.environ["NOTIFY_PLAN_RESULT"] != "success":
        results = [failure("plan", "selection")]
    elif os.environ["NOTIFY_BUILD_RESULT"] not in ("success", "skipped"):
        results = [failure(model, "build") for model in models] or [
            failure("build", "build")
        ]
    elif os.environ["NOTIFY_EVAL_RESULT"] != "success":
        results = [failure(model, "runtime_setup") for model in models] or [
            failure("evalscope-wheels", "runtime_setup")
        ]
    else:
        results = [
            model_result(model, jobs.get(name), partial(evidence, root, model))
            for name, model in names.items()
        ]
        if not results or (
            all(row["status"] == "pass" for row in results)
            and os.environ["NOTIFY_MATRIX_RESULT"] == "failure"
        ):
            results.append(failure("model-matrix", "matrix"))
    return {"elapsed_s": elapsed, "results": results}
