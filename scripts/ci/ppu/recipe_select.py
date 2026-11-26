# SPDX-License-Identifier: Apache-2.0
"""Select every single-node CI policy declaring the tier, optionally narrowed by exact model keys."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from ppu_recipe import REPO_ROOT, _load_yaml, resolve_recipe, validate_paths

SINGLE_NODE_MAX_DEVICES_BY_HARDWARE: dict[str, int] = {
    "zhenwu_m890": 8,
}
SINGLE_NODE_CLUSTER_WIDTHS = frozenset({2, 4, 8})


def cluster_width(
    policy_path: Path, tier: str, *, max_devices: int | None = None
) -> int:
    policy_doc = _load_yaml(policy_path)
    recipe = REPO_ROOT / policy_doc["recipe"]
    recipe_doc = _load_yaml(recipe)
    resolved = resolve_recipe(recipe, policy_path=policy_path, tier=tier)
    raw_ppu_topo = recipe_doc.get("ppu", {}).get("topology")
    topology = (
        raw_ppu_topo
        if set(recipe_doc) == {"ppu"} and isinstance(raw_ppu_topo, dict)
        else resolved["topology"]
    )
    width = int(resolved["ppu_count"])
    allowed_widths = (
        SINGLE_NODE_CLUSTER_WIDTHS
        if max_devices is None
        else {w for w in (2, 4, 8, 16) if w <= max_devices}
        | ({width} if 1 <= width <= max_devices else set())
    )
    if (
        topology["kind"] != "single_node"
        or topology["nodes"] != 1
        or topology["devices_per_node"] != width
        or width not in allowed_widths
    ):
        raise ValueError(f"unsupported cluster topology for policy {policy_path.stem}")
    return width


def select(
    tier: str,
    model_filter: str = "",
    *,
    hardware: str | None = "zhenwu_m890",
    max_devices: int | None = None,
) -> dict[str, list[dict[str, int | str]]]:
    policy_root = REPO_ROOT / "test" / "ppu" / "policy"
    validate_paths(sorted(policy_root.rglob("*.yaml")), sglang_parser_mode="skip")
    models: dict[str, Path] = {}
    for path in sorted((policy_root / "models").glob("*.yaml")):
        if not re.fullmatch(r"[a-z0-9][a-z0-9.-]*", path.stem):
            raise ValueError("invalid policy model key")
        if tier in _load_yaml(path).get("tiers", {}):
            models[path.stem] = path

    eff_max_devices = (
        max_devices
        if max_devices is not None
        else (
            SINGLE_NODE_MAX_DEVICES_BY_HARDWARE.get(hardware)
            if hardware is not None
            else None
        )
    )

    if model_filter:
        requested = [key.strip() for key in model_filter.split(",")]
        if (
            not requested
            or any(not key for key in requested)
            or len(requested) != len(set(requested))
            or set(requested) - set(models)
        ):
            raise ValueError(
                "model filter must contain distinct policy model keys for this tier"
            )
        include: list[dict[str, int | str]] = []
        for key in requested:
            pol_path = models[key]
            pol_doc = _load_yaml(pol_path)
            resolved = resolve_recipe(
                REPO_ROOT / pol_doc["recipe"], policy_path=pol_path, tier=tier
            )
            topo = resolved["topology"]
            if topo["nodes"] > 1 or topo["kind"] == "pd_disagg":
                raise ValueError(
                    f"recipe policy {key!r} requires topology {topo['kind']!r} "
                    f"(nodes={topo['nodes']}), which is not runnable on single-node CI"
                )
            if hardware is not None and resolved["hardware"] != hardware:
                raise ValueError(
                    f"recipe policy {key!r} targets hardware {resolved['hardware']!r}, "
                    f"which is not runnable on single-node {hardware!r} CI"
                )
            if (
                eff_max_devices is not None
                and int(resolved["ppu_count"]) > eff_max_devices
            ):
                raise ValueError(
                    f"recipe policy {key!r} requires {resolved['ppu_count']} devices, "
                    f"exceeding single-node {hardware or resolved['hardware']!r} CI runner limit ({eff_max_devices})"
                )
            include.append(
                {
                    "model": key,
                    "ppu_count": cluster_width(pol_path, tier, max_devices=max_devices),
                }
            )
        return {"include": include}

    include = []
    for key, pol_path in models.items():
        pol_doc = _load_yaml(pol_path)
        resolved = resolve_recipe(
            REPO_ROOT / pol_doc["recipe"], policy_path=pol_path, tier=tier
        )
        topo = resolved["topology"]
        if topo["nodes"] > 1 or topo["kind"] == "pd_disagg":
            continue
        if hardware is not None and resolved["hardware"] != hardware:
            continue
        if eff_max_devices is not None and int(resolved["ppu_count"]) > eff_max_devices:
            continue
        include.append(
            {
                "model": key,
                "ppu_count": cluster_width(pol_path, tier, max_devices=max_devices),
            }
        )
    if not include:
        raise ValueError("no runnable single-node policies found for this tier")
    return {"include": include}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tier", choices=("fast_sanity", "model_sanity"))
    parser.add_argument("--model-filter", default="")
    args = parser.parse_args()
    print(json.dumps(select(args.tier, args.model_filter), separators=(",", ":")))


if __name__ == "__main__":
    main()
