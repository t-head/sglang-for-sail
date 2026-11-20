# SPDX-License-Identifier: Apache-2.0
"""Select every policy declaring the tier, optionally narrowed by exact model keys."""

import argparse
import json
import re

from ppu_recipe import DEFAULT_POLICY_ROOT, _load_yaml, validate_paths
from recipe_cluster_config import MODEL_PATH_SETTINGS


def select(tier, model_filter):
    validate_paths(
        sorted(DEFAULT_POLICY_ROOT.rglob("*.yaml")), sglang_parser_mode="skip"
    )
    models = {}
    for path in sorted((DEFAULT_POLICY_ROOT / "models").glob("*.yaml")):
        if tier in _load_yaml(path).get("tiers", {}):
            if not re.fullmatch(r"[a-z0-9][a-z0-9.-]*", path.stem):
                raise ValueError("invalid policy model key")
            if path.stem not in MODEL_PATH_SETTINGS:
                raise ValueError(f"missing checkpoint setting for policy {path.stem}")
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
    return {"include": [{"model": key} for key in requested]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tier", choices=("fast_sanity", "model_sanity"))
    parser.add_argument("--model-filter", default="")
    args = parser.parse_args()
    print(json.dumps(select(args.tier, args.model_filter), separators=(",", ":")))


if __name__ == "__main__":
    main()
