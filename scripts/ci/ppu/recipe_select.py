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


def cluster_width(policy_path, tier):
    recipe = REPO_ROOT / _load_yaml(policy_path)["recipe"]
    topology = _load_yaml(recipe)["ppu"]["topology"]
    width = resolve_recipe(recipe, policy_path=policy_path, tier=tier)["ppu_count"]
    if (
        topology["kind"] != "single_node"
        or topology["nodes"] != 1
        or topology["devices_per_node"] != width
        or width not in {2, 4, 8}
    ):
        raise ValueError(f"unsupported cluster topology for policy {policy_path.stem}")
    return width


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
    return {
        "include": [
            {"model": key, "ppu_count": cluster_width(models[key], tier)}
            for key in requested
        ]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tier", choices=("fast_sanity", "model_sanity"))
    parser.add_argument("--model-filter", default="")
    args = parser.parse_args()
    print(json.dumps(select(args.tier, args.model_filter), separators=(",", ":")))


if __name__ == "__main__":
    main()
