# SPDX-License-Identifier: Apache-2.0
"""Publish the same allowlisted evidence to Console, Artifact, and Job Summary."""

import argparse
import json
import math
import re
import sys
from pathlib import Path

from recipe_diagnostics import PHASES

REGISTRY = "download.t-head.cn/docker_test/sglang"
MODELS = {"glm-5.3", "deepseek-v4-flash"}
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


def validate(payload):
    if type(payload) is not dict:
        raise ValueError("invalid result")
    meta = {
        "schema",
        "model",
        "tier",
        "source_sha",
        "runtime_image",
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
        or payload["model"] not in MODELS
        or payload["route"] != "native-build + sglang-release-runtime"
        or not re.fullmatch(r"[0-9a-f]{40}", payload["source_sha"])
        or payload["outcome"] not in {"pass", "fail"}
        or type(payload["private_retention"]) is not bool
    ):
        raise ValueError("invalid run identity")
    for key in ("recipe_tree_sha256", "config_digest"):
        if key in payload and not re.fullmatch(r"[0-9a-f]{64}", payload[key]):
            raise ValueError("invalid digest")
    if "runtime_image" in payload and not re.fullmatch(
        re.escape(REGISTRY) + r"@sha256:[0-9a-f]{64}", payload["runtime_image"]
    ):
        raise ValueError("invalid image identity")
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
            (meta - {"failure_phase"}) | BASE | (tier_fields - {"fallback_reason"})
        )
        if (
            not required.issubset(payload)
            or "failure_phase" in payload
            or not all(
                payload[key]
                for key in ("private_retention", "parser_checked", "sse_verified")
            )
        ):
            raise ValueError("passing result lacks evidence")
        if tier == "fast_sanity" and (payload["layer_override"] is None) != (
            "fallback_reason" in payload
        ):
            raise ValueError("fallback result lacks reason")
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
        output.write("## PPU Recipe Sanity\n\n```json\n" + canonical + "\n```\n")
    print(canonical, file=console)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    try:
        if args.source.stat().st_size > 8192 or args.source.is_symlink():
            raise ValueError("invalid result file")
        publish(
            json.loads(args.source.read_text()), args.artifact, args.summary, sys.stdout
        )
    except (ValueError, OSError, KeyError, TypeError, OverflowError, RecursionError):
        print("PPU result rejected by the public evidence gate", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
