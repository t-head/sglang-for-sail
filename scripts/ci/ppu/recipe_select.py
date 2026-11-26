# SPDX-License-Identifier: Apache-2.0
"""Select every policy declaring the tier, optionally narrowed by exact model keys."""

import argparse
import json
import re

from ppu_recipe import (
    DEFAULT_POLICY_ROOT,
    REPO_ROOT,
    _load_yaml,
    resolve_recipe,
    validate_paths,
)

LIGHT_PERF_BOARDS = {
    "zhenwu_m890": ("ZW-M890P", {2, 4, 8}),
    "zhenwu_810e": ("OAM-810E", {1}),
}


def cluster_allocation(policy_path, tier):
    policy = _load_yaml(policy_path)
    recipe = REPO_ROOT / policy["recipe"]
    topology = _load_yaml(recipe)["ppu"]["topology"]
    if tier == "light_perf":
        settings = policy["tiers"][tier]
        _, allowed_widths = LIGHT_PERF_BOARDS[settings["hardware"]]
        plans = [
            resolve_recipe(
                recipe, policy_path=policy_path, tier=tier, preset=case["preset"]
            )
            for case in settings["cases"]
        ]
        allocations = {(plan["ppu_count"], plan["nodes"]) for plan in plans}
        if len(allocations) != 1:
            raise ValueError("light_perf cases must use the same allocation")
        ((total, nodes),) = allocations
        if nodes != 1 and (
            nodes != 2
            or settings["hardware"] != "zhenwu_m890"
            or total != 16
            or any(plan["benchmark_engine"] != "one_batch_server" for plan in plans)
        ):
            raise ValueError("multi-node light_perf requires one_batch_server on 2×8 890P")
        width = total // nodes
    else:
        width = resolve_recipe(recipe, policy_path=policy_path, tier=tier)["ppu_count"]
        nodes = 1
        allowed_widths = {topology["devices_per_node"]} & {2, 4, 8}
    if (
        topology["kind"] != "single_node"
        or topology["nodes"] != 1
        or width not in allowed_widths
    ):
        raise ValueError(f"unsupported cluster topology for policy {policy_path.stem}")
    return width, nodes


def cluster_width(policy_path, tier):
    return cluster_allocation(policy_path, tier)[0]


def select(tier, model_filter):
    validate_paths(
        sorted(DEFAULT_POLICY_ROOT.rglob("*.yaml")), sglang_parser_mode="skip"
    )
    models = {}
    for path in sorted((DEFAULT_POLICY_ROOT / "models").glob("*.yaml")):
        if tier in _load_yaml(path).get("tiers", {}):
            if not re.fullmatch(r"[a-z0-9][a-z0-9.-]*", path.stem):
                raise ValueError("invalid policy model key")
            models[path.stem] = path
    requested = (
        [key.strip() for key in model_filter.split(",")]
        if model_filter
        else list(models)
    )
    if (
        not requested
        or len(requested) != len(set(requested))
        or set(requested) - set(models)
    ):
        raise ValueError(
            "model filter must contain distinct policy model keys for this tier"
        )
    include = []
    for key in requested:
        path = models[key]
        width, nodes = cluster_allocation(path, tier)
        entry = {"model": key, "ppu_count": width}
        if nodes > 1:
            entry["nodes"] = nodes
        if tier == "light_perf":
            hardware = _load_yaml(path)["tiers"][tier]["hardware"]
            entry["board"] = LIGHT_PERF_BOARDS[hardware][0]
        include.append(entry)
    return {"include": include}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tier", choices=("fast_sanity", "model_sanity", "light_perf"))
    parser.add_argument("--model-filter", default="")
    args = parser.parse_args()
    print(json.dumps(select(args.tier, args.model_filter), separators=(",", ":")))


if __name__ == "__main__":
    main()
