# SPDX-License-Identifier: Apache-2.0
"""Select policies declaring a tier, with capacity isolation for multi-node flagships."""

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


def cluster_topology(policy_path, tier):
    recipe = REPO_ROOT / _load_yaml(policy_path)["recipe"]
    topology = _load_yaml(recipe)["ppu"]["topology"]
    total_width = resolve_recipe(recipe, policy_path=policy_path, tier=tier)[
        "ppu_count"
    ]
    kind = topology["kind"]
    nodes = topology["nodes"]
    devices_per_node = topology["devices_per_node"]
    if (
        kind == "single_node"
        and nodes == 1
        and devices_per_node == total_width
        and devices_per_node in {2, 4, 8}
    ):
        return (1, devices_per_node)
    if (
        kind in {"multi_node", "multi_node_ep"}
        and type(nodes) is int
        and nodes >= 2
        and devices_per_node == 8
        and nodes * devices_per_node == total_width
    ):
        return (nodes, devices_per_node)
    raise ValueError(f"unsupported cluster topology for policy {policy_path.stem}")


def cluster_width(policy_path, tier):
    nodes, devices_per_node = cluster_topology(policy_path, tier)
    if nodes != 1:
        raise ValueError(f"unsupported cluster topology for policy {policy_path.stem}")
    return devices_per_node


def _matrix_entry(key, nodes, devices_per_node):
    if nodes == 1:
        return {"model": key, "ppu_count": devices_per_node}
    return {"model": key, "ppu_count": devices_per_node, "nnodes": nodes}


def select(tier, model_filter, *, topology="all"):
    if topology not in {"all", "single_node", "multi_node"}:
        raise ValueError(f"unsupported topology filter: {topology!r}")
    validate_paths(
        sorted(DEFAULT_POLICY_ROOT.rglob("*.yaml")), sglang_parser_mode="skip"
    )
    models = {}
    topologies = {}
    for path in sorted((DEFAULT_POLICY_ROOT / "models").glob("*.yaml")):
        if tier in _load_yaml(path).get("tiers", {}):
            if not re.fullmatch(r"[a-z0-9][a-z0-9.-]*", path.stem):
                raise ValueError("invalid policy model key")
            models[path.stem] = path
            topologies[path.stem] = cluster_topology(path, tier)

    if model_filter:
        requested = [key.strip() for key in model_filter.split(",")]
        if (
            not requested
            or len(requested) != len(set(requested))
            or set(requested) - set(models)
        ):
            raise ValueError(
                "model filter must contain distinct policy model keys for this tier"
            )
    else:
        # Capacity isolation: multi-node flagships never enter the default PR
        # fast_sanity matrix; they run only via explicit model_filter or the
        # dedicated serial multi-node Nightly slot.
        if tier == "fast_sanity":
            requested = [key for key in models if topologies[key][0] == 1]
        else:
            requested = list(models)

    include = []
    for key in requested:
        nodes, devices_per_node = topologies[key]
        if topology == "single_node" and nodes != 1:
            continue
        if topology == "multi_node" and nodes < 2:
            continue
        include.append(_matrix_entry(key, nodes, devices_per_node))
    return {"include": include}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tier", choices=("fast_sanity", "model_sanity"))
    parser.add_argument("--model-filter", default="")
    parser.add_argument(
        "--topology",
        choices=("all", "single_node", "multi_node"),
        default="all",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            select(args.tier, args.model_filter, topology=args.topology),
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
