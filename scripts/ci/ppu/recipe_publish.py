# SPDX-License-Identifier: Apache-2.0
"""Publish the same allowlisted evidence to Console, Artifact, and Job Summary."""

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

from recipe_diagnostics import PHASES

MODEL_POLICY_ROOT = Path(__file__).resolve().parents[3] / "test/ppu/policy/models"
ALLOWED_ROUTES = {
    "native-build + sglang-release-runtime",
    "prebuilt-image",
}
VERSION_KEYS = {"sglang", "sgl_kernel", "torch"}
VERSION_RE = re.compile(
    r"\d+(?:\.\d+){1,3}(?:(?:a|b|rc|\.post|\.dev)\d+){0,2}(?:\+[A-Za-z0-9_-]{1,24})?"
)
LEGACY_SCHEMA = "ppu-legacy-sanity/v1"
SM89_BOARD = "ZW-M890P"
CORE_LEGACY_CASES = {
    "test_basic_sanity_dflash.py": "dflash",
    "test_basic_sanity_eagle3.py": "eagle3",
}
NATIVE_Q8_LEGACY_CASES = {
    "test_dsa_native_q8_ppu.py": "native_q8",
    "bench_dsa_native_q8_ppu.py": "native_q8_benchmark",
}
LEGACY_CASES = CORE_LEGACY_CASES | NATIVE_Q8_LEGACY_CASES


def legacy_cases_for_board(board):
    return LEGACY_CASES if board == SM89_BOARD else CORE_LEGACY_CASES


FAST = {"layer_override", "fallback_reason"}
NIGHTLY = {
    "gsm8k_score",
    "gsm8k_correct",
    "gsm8k_count",
    "golden_score",
    "accuracy_within_band",
    "baseline_established",
    "kv_gauge_zero_verified",
    "c1_passed",
    "c16_passed",
    "eval_s",
    "bench_s",
    "cache_mode",
}
BASE = {
    "outcome",
    "queue_s",
    "cold_start_s",
    "config_digest",
    "parser_checked",
    "sse_verified",
}


def validate_legacy(payload):
    if set(payload) != {
        "schema",
        "source_sha",
        "board",
        "outcome",
        "private_retention",
        "cases",
    } or (
        payload["schema"] != LEGACY_SCHEMA
        or not re.fullmatch(r"[0-9a-f]{40}", payload["source_sha"])
        or payload["board"] not in {"ZW-M890P", "OAM-810E"}
        or payload["outcome"] not in {"pass", "fail"}
        or type(payload["private_retention"]) is not bool
        or type(payload["cases"]) is not dict
        or set(payload["cases"]) != set(legacy_cases_for_board(payload["board"]).values())
    ):
        raise ValueError("invalid legacy run identity")
    for case in payload["cases"].values():
        if type(case) is not dict or set(case) != {"status", "duration_s"}:
            raise ValueError("invalid legacy case")
        status = case["status"]
        duration = case["duration_s"]
        if status == "not_run" and duration is None:
            continue
        if (
            status not in {"pass", "fail"}
            or type(duration) not in (int, float)
            or not math.isfinite(duration)
            or not 0 <= duration <= 3600
        ):
            raise ValueError("invalid legacy case evidence")
    if payload["outcome"] == "pass" and (
        not payload["private_retention"]
        or any(case["status"] != "pass" for case in payload["cases"].values())
    ):
        raise ValueError("legacy pass lacks evidence")


def validate(payload):
    if type(payload) is not dict:
        raise ValueError("invalid result")
    if payload.get("schema") == LEGACY_SCHEMA:
        validate_legacy(payload)
        return
    meta = {
        "schema",
        "model",
        "tier",
        "source_sha",
        "runtime_image",
        "runtime_versions",
        "route",
        "failure_phase",
        "private_retention",
        "recipe_tree_sha256",
    }
    tier = payload.get("tier")
    tier_fields = {"fast_sanity": FAST, "model_sanity": NIGHTLY}.get(tier)
    if tier_fields is None or set(payload) - (meta | BASE | tier_fields):
        raise ValueError("unexpected result field")
    if (
        payload["schema"] != "ppu-sanity/v1"
        or not re.fullmatch(r"[a-z0-9][a-z0-9.-]*", payload["model"])
        or not (MODEL_POLICY_ROOT / f"{payload['model']}.yaml").is_file()
        or payload["route"] not in ALLOWED_ROUTES
        or not re.fullmatch(r"[0-9a-f]{40}", payload["source_sha"])
        or payload["outcome"] not in {"pass", "fail"}
        or type(payload["private_retention"]) is not bool
    ):
        raise ValueError("invalid run identity")
    for key in ("recipe_tree_sha256", "config_digest"):
        if key in payload and not re.fullmatch(r"[0-9a-f]{64}", payload[key]):
            raise ValueError("invalid digest")
    if "runtime_image" in payload and not re.fullmatch(
        r"@sha256:[0-9a-f]{64}", payload["runtime_image"]
    ):
        raise ValueError("invalid image identity")
    if "runtime_versions" in payload:
        versions = payload["runtime_versions"]
        if (
            type(versions) is not dict
            or not versions
            or set(versions) - VERSION_KEYS
            or any(
                type(val) is not str or not VERSION_RE.fullmatch(val)
                for val in versions.values()
            )
        ):
            raise ValueError("invalid runtime versions")
    for key in (
        "queue_s",
        "cold_start_s",
        "eval_s",
        "bench_s",
        "gsm8k_score",
        "golden_score",
    ):
        if key in payload:
            number = payload[key]
            ceiling = 1 if "score" in key else 86400
            if (
                type(number) not in (int, float)
                or not math.isfinite(number)
                or not 0 <= number <= ceiling
            ):
                raise ValueError("invalid numeric evidence")
    for key in (
        "parser_checked",
        "sse_verified",
        "accuracy_within_band",
        "baseline_established",
        "kv_gauge_zero_verified",
        "c1_passed",
        "c16_passed",
    ):
        if key in payload and type(payload[key]) is not bool:
            raise ValueError("invalid Boolean evidence")
    for key, allowed in {
        "layer_override": (None, 2),
        "fallback_reason": ("server_exit", "readiness_timeout"),
        "cache_mode": ("sanity_throughput_guard",),
        "failure_phase": (*sorted(PHASES), "unknown"),
    }.items():
        if key in payload and payload[key] not in allowed:
            raise ValueError("invalid status")
    if "gsm8k_count" in payload and (
        type(payload["gsm8k_count"]) is not int or payload["gsm8k_count"] != 1319
    ):
        raise ValueError("invalid GSM8K count")
    if "gsm8k_correct" in payload and (
        type(payload["gsm8k_correct"]) is not int
        or not 0 <= payload["gsm8k_correct"] <= 1319
    ):
        raise ValueError("invalid correct count")
    if payload["outcome"] == "pass":
        required = (
            (meta - {"failure_phase", "runtime_versions"})
            | BASE
            | (tier_fields - {"fallback_reason"})
        )
        if payload["route"] == "prebuilt-image":
            required = required | {"runtime_versions"}
        if (
            not required.issubset(payload)
            or "failure_phase" in payload
            or not all(
                payload[key]
                for key in ("private_retention", "parser_checked", "sse_verified")
            )
        ):
            raise ValueError("passing result lacks evidence")
        if (
            payload["route"] == "prebuilt-image"
            and "sglang" not in payload["runtime_versions"]
        ):
            raise ValueError("prebuilt-image pass lacks sglang version fingerprint")
        if (
            tier == "fast_sanity"
            and "fallback_reason" in payload
            and payload["layer_override"] is not None
        ):
            raise ValueError("fallback result retained a layer override")
        if tier == "model_sanity" and (
            not all(
                payload[key]
                for key in (
                    "accuracy_within_band",
                    "baseline_established",
                    "kv_gauge_zero_verified",
                    "c1_passed",
                    "c16_passed",
                )
            )
            or abs(payload["gsm8k_score"] - payload["golden_score"]) > 0.02
            or payload["gsm8k_correct"] != round(payload["gsm8k_score"] * 1319)
        ):
            raise ValueError("passing nightly result did not meet gate")


def publish(payload, artifact, summary, console):
    validate(payload)
    canonical = json.dumps(payload, sort_keys=True, allow_nan=False)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    with artifact.open("x", encoding="utf-8") as output:
        output.write(canonical + "\n")
    with summary.open("a", encoding="utf-8") as output:
        title = (
            "PPU Single-Device Sanity"
            if payload["schema"] == LEGACY_SCHEMA
            else "PPU Recipe Sanity"
        )
        output.write(f"## {title}\n\n```json\n" + canonical + "\n```\n")
    print(canonical, file=console)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("artifact", nargs="?", type=Path)
    parser.add_argument("summary", nargs="?", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not args.verify_only and (args.artifact is None or args.summary is None):
        parser.error("publishing requires the artifact and summary paths")
    try:
        if args.source.stat().st_size > 8192 or args.source.is_symlink():
            raise ValueError("invalid result file")
        record = json.loads(args.source.read_text())
        if args.verify_only:
            validate(record)
            if record["source_sha"] != os.environ["PPU_CI_SOURCE_SHA"]:
                raise ValueError("unmatched evidence source revision")
            if (
                record["schema"] == LEGACY_SCHEMA
                and record["board"] != os.environ["PPU_CI_BOARD"]
            ):
                raise ValueError("unmatched legacy board")
        else:
            publish(record, args.artifact, args.summary, sys.stdout)
    except (ValueError, OSError, KeyError, TypeError, OverflowError, RecursionError):
        print("PPU result rejected by the public evidence gate", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
