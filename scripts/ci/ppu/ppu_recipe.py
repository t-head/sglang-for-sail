#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm-project/vllm-sail@c67bf6a (Apache-2.0).
"""Thin recipe/policy resolver and validator for ``ppu-serving/v1``."""

from __future__ import annotations

import argparse
import contextlib
import copy
import difflib
import hashlib
import importlib
import json
import math
import os
import posixpath
import re
import shlex
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
_RECIPES_REL = Path("test") / "ppu" / "recipes"
_POLICY_REL = Path("test") / "ppu" / "policy"
DEFAULT_RECIPES_ROOT = REPO_ROOT / _RECIPES_REL
DEFAULT_POLICY_ROOT = REPO_ROOT / _POLICY_REL
DEFAULT_MATRIX_DOC = REPO_ROOT / "test" / "ppu" / "README.md"
DEFAULT_AUTHORING_DOC = REPO_ROOT / "test" / "ppu" / "AUTHORING.md"
GSM8K_GOLDEN_BAND_ABS = 0.02

SERVING_SCHEMA = "ppu-serving/v1"
DATASET_SCHEMA = "ppu-dataset/v1"
POLICY_SCHEMA = "ppu-test-policy/v1"
VARIANT_ISOLATED_FIELDS = (
    "assets",
    "hf_model_id",
    "golden_model_key",
    "served_name",
)
_STRAY_SERVING_SCHEMA_RE = re.compile(
    r"""(?:^\s*|[{\[,]\s*)['"]?schema_version['"]?\s*:\s*(['"]?)([^\r\n#,}\]]+?)\1(?=\s*(?:[,}\]]|#|$))""",
    re.MULTILINE,
)
_YAML_ESCAPE_RE = re.compile(
    r"\\(?:u([0-9a-fA-F]{4})|U([0-9a-fA-F]{8})|x([0-9a-fA-F]{2}))"
)
_STRAY_SCAN_PRUNE_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".mypy_cache",
        ".nox",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "target",
        "venv",
    }
)
POLICY_TIERS = frozenset({"fast_sanity", "model_sanity", "nightly", "release"})
SUPPORTED_HARDWARE = frozenset({"zhenwu_m890", "zhenwu_810e"})
_M890_PRECISION_RE = re.compile(
    r"^(?:(?:mx)?fp4(?:_fp8)?|fp8(?:_[a-z0-9]+)?|w4a4_mxfp4|w8a8_fp8)$"
)
_810E_PRECISION_RE = re.compile(r"^(?:int8|w[48]a8_int8(?:_[a-z0-9]+)?)$")
MULTI_NODE_LAUNCH_FLAGS = frozenset(
    {
        "--dist-init-addr",
        "--elastic-ep-initial-size",
        "--elastic-ep-join-mode",
        "--elastic-ep-join-rank-offset",
        "--nccl-init-addr",
        "--nnodes",
        "--node-rank",
    }
)
PD_DISAGG_FLAGS = frozenset(
    {
        "--disaggregation-bootstrap-port",
        "--disaggregation-decode-dp",
        "--disaggregation-decode-tp",
        "--disaggregation-ib-device",
        "--disaggregation-mode",
        "--disaggregation-transfer-backend",
        "--kv-transfer-config",
    }
)
SUPPORTED_DISAGG_BACKENDS = frozenset(
    {"mooncake", "mooncake_tcp", "nixl", "ascend", "mori", "fake"}
)
_UNSET: Any = object()

ALLOWED_ENV_PREFIXES = (
    "VLLM_",
    "SGLANG_",
    "SGL_",
    "NCCL_",
    "TORCH_",
    "OMP_",
    "DG_",
    "HGGC_",
    "MC_",
    "MOONCAKE_",
)
FORBIDDEN_RECIPE_ENV = frozenset(
    {
        "CUDA_VISIBLE_DEVICES",
        "HF_DATASETS_CACHE",
        "HF_DATASETS_OFFLINE",
        "HF_HOME",
        "HF_HUB_OFFLINE",
        "HF_MODULES_CACHE",
        "NVIDIA_VISIBLE_DEVICES",
        "SGLANG_USE_MODELSCOPE",
        "SGLANG_WARMUP_TIMEOUT",
        "TRANSFORMERS_OFFLINE",
        "VLLM_SERVER_DEV_MODE",
        "VLLM_USE_MODELSCOPE",
        "VLLM_USE_RUST_BENCH",
        "VLLM_USE_RUST_FRONTEND",
    }
)
EXTERNAL_ENV_PREFIXES = ("DG_", "HGGC_")
EXTERNAL_ENV_EXACT = frozenset({"SGLANG_NSA_DUAL_STREAM"})
JSON_FLAGS = frozenset(
    {
        "--attention-config",
        "--compilation-config",
        "--engram-config",
        "--json-model-override-args",
        "--kv-transfer-config",
        "--override-generation-config",
        "--reasoning-config",
        "--speculative-config",
    }
)
INVERSE_BOOLEAN_GROUPS: tuple[tuple[str, ...], ...] = (
    ("--trust-remote-code", "--no-trust-remote-code"),
    ("--enable-prefix-caching", "--no-enable-prefix-caching"),
)
FORBIDDEN_RECIPE_FLAGS = frozenset(
    {
        "--config",
        "--disable-radix-cache",
        "--enable-metrics",
        "--enable-prefix-caching",
        "--host",
        "--load-format",
        "--model",
        "--model-path",
        "--no-enable-prefix-caching",
        "--port",
        "--served-model-name",
        "--tokenizer-path",
    }
)
ALLOWED_PRESET_SLOTS = frozenset(
    {
        "decode",
        "decode_ep",
        "decode_nographs",
        "dp_ep_low_latency",
        "eval_longctx",
        "model_sanity",
        "pd_decode",
        "pd_prefill",
        "prefill",
        "prefill_long",
        "tp8",
    }
)
CANONICAL_PRESET_TOPOLOGY_KIND: dict[str, str] = {
    "prefill": "single_node",
    "decode": "single_node",
    "decode_ep": "multi_node_ep",
    "pd_prefill": "pd_disagg",
    "pd_decode": "pd_disagg",
}
TOPOLOGY_PLACEHOLDERS: dict[str, tuple[str, Any]] = {
    "master_addr": ("127.0.0.1", "127.0.0.1"),
    "master_port": ("29500", 29500),
    "node_rank": ("0", 0),
    "kv_ip": ("127.0.0.1", "127.0.0.1"),
    "kv_port": ("8998", 8998),
    "ib_devices": ("rdma0", "rdma0"),
}
MULTI_NODE_REQUIRED_PLACEHOLDERS = frozenset(
    {"master_addr", "master_port", "node_rank"}
)
PD_DISAGG_PREFILL_REQUIRED_PLACEHOLDERS = frozenset({"kv_ip", "kv_port", "ib_devices"})
PD_DISAGG_DECODE_REQUIRED_PLACEHOLDERS = frozenset({"kv_ip", "ib_devices"})
PD_DISAGG_REQUIRED_PLACEHOLDERS = PD_DISAGG_PREFILL_REQUIRED_PLACEHOLDERS
_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
_ADJACENT_NUMERIC_PLACEHOLDER_RE = re.compile(
    r"\d\{(?:node_rank|master_port|kv_port)\}|\{(?:node_rank|master_port|kv_port)\}\d"
)
ALLOWED_CACHE_MODES = frozenset({"sanity_throughput_guard", "prefix_reuse_bench"})

_PROHIBITED_TOKEN_A = "".join(chr(c) for c in (98, 116, 118))
_PROHIBITED_TOKEN_B = "".join(chr(c) for c in (102, 108, 121, 116, 105, 103, 101, 114))
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_SENSITIVE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("internal_domain", re.compile(r"\.(?:alibaba-inc\.com|eng\.t-head\.cn)\b", re.I)),
    ("internal_mirror", re.compile(r"\brsproxy\.cn\b", re.I)),
    (
        "local_path",
        re.compile(
            r"(?:^|[\s'\"=])(?:/nas_aisw|/ppusw|/cpfs/|/mnt/|/Users/|/home/|/root/|/data/)"
        ),
    ),
    (
        "credential",
        re.compile(
            r"\b(?:ghp|gho|ghu|ghs)_[A-Za-z0-9_]{8,}|github_pat_|AKIA[0-9A-Z]{8,}|QODER_PAT\b"
        ),
    ),
    (
        "nic_device",
        re.compile(r"\bmlx5_(?:bond_)?\d+\b", re.I),
    ),
    (
        "prohibited_token",
        re.compile(rf"(?:{_PROHIBITED_TOKEN_A}|{_PROHIBITED_TOKEN_B})", re.I),
    ),
)


class ValidationError(ValueError):
    """Raised when recipe or policy validation fails."""


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValidationError(f"{path}: failed to read YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ValidationError(f"{path}: top-level YAML must be a mapping")
    return data


def _check_keys(
    mapping: Any,
    *,
    allowed: set[str] | frozenset[str],
    required: set[str] | frozenset[str] = frozenset(),
    context: str,
) -> dict[str, Any]:
    if not isinstance(mapping, dict):
        raise ValidationError(
            f"{context}: expected mapping, got {type(mapping).__name__}"
        )
    unknown = sorted(set(mapping) - set(allowed))
    if unknown:
        raise ValidationError(f"{context}: unknown key(s): {', '.join(unknown)}")
    missing = sorted(set(required) - set(mapping))
    if missing:
        raise ValidationError(
            f"{context}: missing required key(s): {', '.join(missing)}"
        )
    return mapping


def _check_str_list(value: Any, *, context: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValidationError(f"{context}: expected list of strings")
    return value


def _check_str_dict(value: Any, *, context: str) -> dict[str, str]:
    if not isinstance(value, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in value.items()
    ):
        raise ValidationError(f"{context}: expected string-to-string mapping")
    return value


def _deep_merge_dicts(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _inverse_map() -> dict[str, frozenset[str]]:
    pairs: dict[str, frozenset[str]] = {}
    for group in INVERSE_BOOLEAN_GROUPS:
        pos = group[0]
        negs = frozenset(group[1:])
        pairs[pos] = negs
        for neg in negs:
            pairs[neg] = frozenset({pos, *(n for n in negs if n != neg)})
    return pairs


_INVERSE_FLAGS = _inverse_map()


def _parse_flag_entries(tokens: list[str]) -> list[tuple[str, str | None, bool]]:
    """Split raw CLI tokens into ``(flag_name, value_or_none, has_equals)``."""
    entries: list[tuple[str, str | None, bool]] = []
    idx = 0
    while idx < len(tokens):
        tok = tokens[idx]
        if not isinstance(tok, str) or not tok.startswith("--"):
            raise ValidationError(f"expected --flag token, got {tok!r}")
        if "=" in tok:
            flag, val = tok.split("=", 1)
            has_eq = True
            step = 1
        elif idx + 1 < len(tokens) and not str(tokens[idx + 1]).startswith("--"):
            flag, val = tok, str(tokens[idx + 1])
            has_eq = False
            step = 2
        else:
            flag, val = tok, None
            has_eq = False
            step = 1
        opt_head = flag.split(".", 1)[0]
        if len(opt_head) <= 2 or "_" in opt_head:
            raise ValidationError(
                f"CLI flag {flag!r} must use canonical hyphenated spelling "
                f"({opt_head.replace('_', '-')!r})"
            )
        entries.append((flag, val, has_eq))
        idx += step
    return entries


def _apply_flag_entries(
    ordered: list[tuple[str, str | None, bool]],
    new_tokens: list[str],
) -> list[tuple[str, str | None, bool]]:
    """Append flags with last-wins deduplication and inverse-flag clearing."""
    result = list(ordered)
    for flag, val, has_eq in _parse_flag_entries(new_tokens):
        inverses = _INVERSE_FLAGS.get(flag)
        result = [
            item
            for item in result
            if item[0] != flag and (inverses is None or item[0] not in inverses)
        ]
        result.append((flag, val, has_eq))
    return result


def _remove_flag_entries(
    ordered: list[tuple[str, str | None, bool]],
    remove_flags: list[str],
) -> list[tuple[str, str | None, bool]]:
    to_drop = set(remove_flags)
    return [item for item in ordered if item[0] not in to_drop]


def _merge_json_flag(
    ordered: list[tuple[str, str | None, bool]],
    flag: str,
    patch: dict[str, Any],
) -> list[tuple[str, str | None, bool]]:
    existing_obj: dict[str, Any] = {}
    for item_flag, val, _ in ordered:
        if item_flag == flag and val is not None:
            parsed = json.loads(val)
            if not isinstance(parsed, dict):
                raise ValidationError(f"cannot json-merge non-object flag {flag}")
            existing_obj = parsed
    merged = _deep_merge_dicts(existing_obj, patch)
    encoded = json.dumps(merged, separators=(",", ":"), sort_keys=True)
    return _apply_flag_entries(ordered, [flag, encoded])


def _render_entries(ordered: list[tuple[str, str | None, bool]]) -> list[str]:
    tokens: list[str] = []
    for flag, val, has_eq in ordered:
        if val is None:
            tokens.append(flag)
        elif has_eq:
            tokens.append(f"{flag}={val}")
        else:
            tokens.extend([flag, val])
    return tokens


def _flag_value(
    ordered: Sequence[tuple[str, str | None, bool]], *flags: str
) -> str | None:
    target = frozenset(flags)
    for item_flag, val, _ in reversed(ordered):
        if item_flag in target:
            return val
    return None


def _load_assets_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    raw = path.read_text(encoding="utf-8")
    loaded = yaml.safe_load(raw)
    if not isinstance(loaded, dict):
        raise ValidationError(f"assets-map {path}: must be a mapping")
    if "models" in loaded and isinstance(loaded["models"], dict):
        return {str(k): str(v) for k, v in loaded["models"].items()}
    return {str(k): str(v) for k, v in loaded.items()}


def _canonical_flags(ordered: list[tuple[str, str | None, bool]]) -> list[str]:
    canonical: list[str] = []
    for flag, val, _ in ordered:
        if val is None:
            canonical.append(flag)
        elif flag in JSON_FLAGS:
            obj = json.loads(val)
            canonical.extend(
                [flag, json.dumps(obj, separators=(",", ":"), sort_keys=True)]
            )
        else:
            canonical.extend([flag, val])
    return canonical


def _normalize_precision(precision: Any, *, context: str) -> str:
    if not isinstance(precision, str) or not precision.strip():
        raise ValidationError(f"{context}: precision must be a non-empty string")
    return precision.strip().lower()


def _variant_compatible_hardware(
    var_cfg: Mapping[str, Any], *, context: str
) -> frozenset[str]:
    explicit = var_cfg.get("hardware")
    if explicit is not None:
        if explicit not in SUPPORTED_HARDWARE:
            raise ValidationError(
                f"{context}.hardware invalid: {explicit!r}; "
                f"expected one of {sorted(SUPPORTED_HARDWARE)}"
            )
        return frozenset({str(explicit)})
    prec = _normalize_precision(var_cfg.get("precision"), context=context)
    if prec == "bf16":
        return SUPPORTED_HARDWARE
    if _810E_PRECISION_RE.fullmatch(prec):
        return frozenset({"zhenwu_810e"})
    return frozenset({"zhenwu_m890"})


def _infer_variant_hardware(var_cfg: Mapping[str, Any], *, context: str) -> str:
    explicit = var_cfg.get("hardware")
    if explicit is not None:
        if explicit not in SUPPORTED_HARDWARE:
            raise ValidationError(
                f"{context}.hardware invalid: {explicit!r}; "
                f"expected one of {sorted(SUPPORTED_HARDWARE)}"
            )
        return str(explicit)
    prec = _normalize_precision(var_cfg.get("precision"), context=context)
    if _810E_PRECISION_RE.fullmatch(prec):
        return "zhenwu_810e"
    return "zhenwu_m890"


def _validate_hardware_precision(
    *,
    hardware: str,
    precision: Any,
    context: str,
) -> None:
    if hardware not in SUPPORTED_HARDWARE:
        raise ValidationError(
            f"{context}: unknown hardware {hardware!r}; "
            f"expected one of {sorted(SUPPORTED_HARDWARE)}"
        )
    prec = _normalize_precision(precision, context=context)
    if prec == "bf16":
        return
    if hardware == "zhenwu_m890":
        if not _M890_PRECISION_RE.fullmatch(prec):
            raise ValidationError(
                f"{context}: hardware 'zhenwu_m890' only allows bf16 or mxfp4/fp4/fp8 "
                f"precision variants, got {prec!r}"
            )
    elif hardware == "zhenwu_810e":
        if not _810E_PRECISION_RE.fullmatch(prec):
            raise ValidationError(
                f"{context}: hardware 'zhenwu_810e' only allows bf16 or int8 precision "
                f"variants (e.g. w8a8_int8, w4a8_int8), got {prec!r}"
            )


def _iter_variant_preset_targets(
    recipe: Mapping[str, Any], *, context: str
) -> list[tuple[str, str | None, str]]:
    """Return deterministic ``(variant, preset, hardware)`` expansion tuples."""
    variants_blk = recipe.get("variants", {})
    presets_blk = recipe.get("presets", {})
    targets: list[tuple[str, str | None, str]] = []
    for vname, vcfg in variants_blk.items():
        vctx = f"{context}.variants.{vname}"
        compat_hw = _variant_compatible_hardware(vcfg, context=vctx)
        default_hw = _infer_variant_hardware(vcfg, context=vctx)
        baseline_hws = [default_hw] + [
            hw for hw in sorted(compat_hw) if hw != default_hw
        ]
        for hw in baseline_hws:
            targets.append((vname, None, hw))
        for pname, pcfg in presets_blk.items():
            if pcfg.get("variants") is not None and vname not in pcfg["variants"]:
                continue
            preset_hw = pcfg.get("hardware")
            if preset_hw is not None:
                if preset_hw not in compat_hw:
                    continue
                targets.append((vname, pname, str(preset_hw)))
            else:
                for hw in baseline_hws:
                    targets.append((vname, pname, hw))
    return targets


def _validate_topology_block(
    topo: Any,
    *,
    context: str,
    allow_role: bool = False,
    expected_role: str | None = None,
) -> dict[str, Any]:
    allowed = {"kind", "nodes", "devices_per_node"}
    if allow_role:
        allowed = allowed | {"role"}
    checked = _check_keys(
        topo,
        allowed=allowed,
        required={"kind", "nodes", "devices_per_node"},
        context=context,
    )
    kind = checked["kind"]
    if kind not in {"single_node", "multi_node_ep", "pd_disagg"}:
        raise ValidationError(f"{context}.kind invalid: {kind!r}")
    nodes = checked["nodes"]
    devices = checked["devices_per_node"]
    if type(nodes) is not int or nodes <= 0:
        raise ValidationError(f"{context}.nodes must be positive int")
    if type(devices) is not int or devices <= 0:
        raise ValidationError(f"{context}.devices_per_node must be positive int")
    if kind == "single_node" and nodes != 1:
        raise ValidationError(
            f"{context}: single_node topology requires nodes=1, got {nodes}"
        )
    if kind == "multi_node_ep" and nodes < 2:
        raise ValidationError(
            f"{context}: multi_node_ep topology requires nodes >= 2, got {nodes}"
        )
    if "role" in checked:
        if kind != "pd_disagg":
            raise ValidationError(
                f"{context}.role is only allowed for pd_disagg topology"
            )
        if checked["role"] not in {"prefill", "decode"}:
            raise ValidationError(
                f"{context}.role must be 'prefill' or 'decode', got {checked['role']!r}"
            )
        if expected_role is not None and checked["role"] != expected_role:
            raise ValidationError(
                f"{context}.role must be {expected_role!r}, got {checked['role']!r}"
            )
    return checked


def _find_placeholders_in_obj(obj: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(obj, str):
        found.update(_PLACEHOLDER_RE.findall(obj))
    elif isinstance(obj, list):
        for elem in obj:
            found.update(_find_placeholders_in_obj(elem))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            found.update(_find_placeholders_in_obj(k))
            found.update(_find_placeholders_in_obj(v))
    return found


def _extract_placeholders_from_flags_and_env(
    flags: Sequence[str], server_env: Mapping[str, str]
) -> set[str]:
    found: set[str] = set()
    for flag, val, _ in _parse_flag_entries(list(flags)):
        if val is None:
            continue
        if flag in JSON_FLAGS:
            try:
                parsed = json.loads(val)
            except json.JSONDecodeError as exc:
                raise ValidationError(
                    f"JSON flag {flag} has invalid JSON: {exc}"
                ) from exc
            found.update(_find_placeholders_in_obj(parsed))
        else:
            found.update(_PLACEHOLDER_RE.findall(val))
    for val in server_env.values():
        found.update(_PLACEHOLDER_RE.findall(val))
    return found


def _build_placeholder_table(
    values: Mapping[str, Any] | None = None,
) -> dict[str, tuple[str, Any]]:
    table = dict(TOPOLOGY_PLACEHOLDERS)
    if values is not None:
        unknown = sorted(set(values) - set(TOPOLOGY_PLACEHOLDERS))
        if unknown:
            raise ValidationError(
                f"unknown topology placeholder override(s): {', '.join(unknown)}"
            )
        for name, raw in values.items():
            table[name] = (str(raw), raw)
    return table


def _substitute_placeholders_in_str(
    text: str, table: Mapping[str, tuple[str, Any]] = TOPOLOGY_PLACEHOLDERS
) -> str:
    def _repl(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in table:
            raise ValidationError(f"unknown topology placeholder '{{{name}}}'")
        return table[name][0]

    return _PLACEHOLDER_RE.sub(_repl, text)


def _substitute_placeholders_in_json_obj(
    obj: Any, table: Mapping[str, tuple[str, Any]] = TOPOLOGY_PLACEHOLDERS
) -> Any:
    if isinstance(obj, str):
        exact = _PLACEHOLDER_RE.fullmatch(obj)
        if exact:
            name = exact.group(1)
            if name not in table:
                raise ValidationError(f"unknown topology placeholder '{{{name}}}'")
            return table[name][1]
        return _substitute_placeholders_in_str(obj, table)
    if isinstance(obj, list):
        return [_substitute_placeholders_in_json_obj(elem, table) for elem in obj]
    if isinstance(obj, dict):
        return {
            _substitute_placeholders_in_str(
                str(k), table
            ): _substitute_placeholders_in_json_obj(v, table)
            for k, v in obj.items()
        }
    return obj


def substitute_validation_placeholders(
    flags: Sequence[str],
    server_env: Mapping[str, str] | None = None,
    *,
    values: Mapping[str, Any] | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Substitute topology placeholders with valid dummy or caller-supplied values."""
    table = _build_placeholder_table(values)
    sub_entries: list[tuple[str, str | None, bool]] = []
    for flag, val, has_eq in _parse_flag_entries(list(flags)):
        if val is None:
            sub_entries.append((flag, None, has_eq))
        elif flag in JSON_FLAGS:
            try:
                parsed = json.loads(val)
            except json.JSONDecodeError as exc:
                raise ValidationError(
                    f"JSON flag {flag} has invalid JSON: {exc}"
                ) from exc
            if not isinstance(parsed, dict):
                raise ValidationError(f"JSON flag {flag} must decode to a JSON object")
            substituted = _substitute_placeholders_in_json_obj(parsed, table)
            encoded = json.dumps(substituted, separators=(",", ":"), sort_keys=True)
            sub_entries.append((flag, encoded, has_eq))
        else:
            sub_entries.append(
                (flag, _substitute_placeholders_in_str(val, table), has_eq)
            )
    sub_env = {
        str(k): _substitute_placeholders_in_str(str(v), table)
        for k, v in (server_env or {}).items()
    }
    return _render_entries(sub_entries), sub_env


def _validate_port_value(raw: str, *, context: str) -> None:
    try:
        port = int(raw)
    except ValueError as exc:
        raise ValidationError(
            f"{context}: port must be an integer in [1, 65535], got {raw!r}"
        ) from exc
    if not 1 <= port <= 65535:
        raise ValidationError(f"{context}: port must be in [1, 65535], got {port}")


def _parse_host_port(
    address: str, *, context: str, label: str = "address"
) -> tuple[str, int]:
    """Parse ``host:port`` or ``[ipv6]:port`` matching SGLang ``NetworkAddress.parse``."""
    if not address or any(ch.isspace() for ch in address) or "/" in address:
        raise ValidationError(
            f"{context}: {label} must be 'host:port' or '[ipv6]:port' "
            f"(no scheme, path, or whitespace), got {address!r}"
        )
    if address.startswith("["):
        if "]" not in address:
            raise ValidationError(
                f"{context}: {label} has unclosed IPv6 bracket: {address!r}"
            )
        host_part, rest = address.rsplit("]", 1)
        host = host_part[1:]
        if not host or "[" in host or "]" in host or not rest.startswith(":"):
            raise ValidationError(
                f"{context}: {label} must be '[ipv6]:port', got {address!r}"
            )
        port_raw = rest[1:]
        if ":" in port_raw:
            raise ValidationError(
                f"{context}: {label} must be '[ipv6]:port', got {address!r}"
            )
    else:
        parts = address.split(":")
        if len(parts) != 2:
            raise ValidationError(
                f"{context}: {label} must be 'host:port', got {address!r}"
            )
        host, port_raw = parts
    if not host:
        raise ValidationError(
            f"{context}: {label} requires non-empty host, got {address!r}"
        )
    _validate_port_value(port_raw, context=context)
    return host, int(port_raw)


def _parse_pos_int_flag(flag_name: str, raw: str | None, *, context: str) -> int | None:
    if raw is None:
        return None
    try:
        val = int(raw)
    except ValueError as exc:
        raise ValidationError(
            f"{context}: {flag_name} must be a positive int, got {raw!r}"
        ) from exc
    if val <= 0:
        raise ValidationError(
            f"{context}: {flag_name} must be a positive int, got {raw!r}"
        )
    return val


def _validate_multinode_tp_pp_layout(
    *,
    tp_val: int,
    pp_val: int,
    topo_nodes: int,
    devices_per_node: int | None,
    context: str,
) -> None:
    """Validate SGLang ``_calculate_rank_ranges`` per-node TP/PP partition."""
    pp_size_per_node = max(pp_val // topo_nodes, 1)
    nnodes_per_pp_rank = max(topo_nodes // pp_val, 1)
    if (
        (pp_val >= topo_nodes and pp_val % topo_nodes != 0)
        or (topo_nodes > pp_val and topo_nodes % pp_val != 0)
        or tp_val % nnodes_per_pp_rank != 0
    ):
        raise ValidationError(
            f"{context}: multi-node SGLang rank assignment incompatible for "
            f"nodes={topo_nodes}, --tp-size={tp_val}, --pp-size={pp_val}"
        )
    tp_size_per_node = tp_val // nnodes_per_pp_rank
    workers_per_node = pp_size_per_node * tp_size_per_node
    if devices_per_node is not None and workers_per_node > devices_per_node:
        raise ValidationError(
            f"{context}: multi-node SGLang assigns {workers_per_node} workers per node "
            f"(pp_per_node={pp_size_per_node} * tp_per_node={tp_size_per_node}), "
            f"exceeding topology.devices_per_node ({devices_per_node})"
        )
    seen_ranks: list[tuple[int, int]] = []
    for node_rank in range(topo_nodes):
        pp_start = (node_rank // nnodes_per_pp_rank) * pp_size_per_node
        tp_start = (node_rank % nnodes_per_pp_rank) * tp_size_per_node
        for pp_r in range(pp_start, pp_start + pp_size_per_node):
            for tp_r in range(tp_start, tp_start + tp_size_per_node):
                seen_ranks.append((pp_r, tp_r))
    expected_ranks = {(p, t) for p in range(pp_val) for t in range(tp_val)}
    if len(seen_ranks) != len(expected_ranks) or set(seen_ranks) != expected_ranks:
        raise ValidationError(
            f"{context}: multi-node SGLang rank ranges do not partition "
            f"pp_size={pp_val} x tp_size={tp_val} across nodes={topo_nodes}"
        )


def _compute_sglang_parallelism(
    entries: Sequence[tuple[str, str | None, bool]],
    *,
    topo_nodes: int,
    devices_per_node: int | None = None,
    context: str,
) -> tuple[int, int, int, int, int, bool]:
    """Return ``(ppu_count, dp_size, tp_size, pp_size, attn_dp_size, ep_enabled)`` matching SGLang ``ServerArgs``."""
    if any(flag == "--enable-dp-attention" for flag, _, _ in entries):
        raise ValidationError(
            f"{context}: deprecated flag --enable-dp-attention is forbidden in recipes; "
            f"set --attn-dp-size explicitly"
        )

    dp_val = (
        _parse_pos_int_flag(
            "--dp-size",
            _flag_value(entries, "--dp-size", "--data-parallel-size"),
            context=context,
        )
        or 1
    )
    tp_val = (
        _parse_pos_int_flag(
            "--tp-size",
            _flag_value(entries, "--tp-size", "--tensor-parallel-size"),
            context=context,
        )
        or 1
    )
    pp_val = (
        _parse_pos_int_flag(
            "--pp-size",
            _flag_value(entries, "--pp-size", "--pipeline-parallel-size"),
            context=context,
        )
        or 1
    )
    ep_val = (
        _parse_pos_int_flag(
            "--ep-size",
            _flag_value(entries, "--ep-size", "--expert-parallel-size", "--ep"),
            context=context,
        )
        or 1
    )
    attn_dp_val = (
        _parse_pos_int_flag(
            "--attn-dp-size",
            _flag_value(entries, "--attn-dp-size", "--attention-data-parallel-size"),
            context=context,
        )
        or 1
    )
    attn_cp_val = (
        _parse_pos_int_flag(
            "--attn-cp-size",
            _flag_value(entries, "--attn-cp-size", "--attention-context-parallel-size"),
            context=context,
        )
        or 1
    )

    if dp_val > 1 and attn_dp_val > 1:
        raise ValidationError(
            f"{context}: --dp-size ({dp_val}) and --attn-dp-size ({attn_dp_val}) "
            f"cannot both be greater than 1 in SGLang"
        )
    if topo_nodes > 1 and dp_val > 1:
        raise ValidationError(
            f"{context}: multi-node SGLang topology (nodes={topo_nodes}) requires "
            f"--dp-size 1 (got {dp_val}); scale attention DP via --tp-size and --attn-dp-size"
        )
    if (attn_dp_val > 1 or attn_cp_val > 1) and tp_val % (
        attn_dp_val * attn_cp_val
    ) != 0:
        raise ValidationError(
            f"{context}: --tp-size ({tp_val}) must be divisible by "
            f"--attn-dp-size * --attn-cp-size ({attn_dp_val} * {attn_cp_val})"
        )
    if ep_val > 1 and tp_val % ep_val != 0:
        raise ValidationError(
            f"{context}: --tp-size ({tp_val}) must be divisible by --ep-size ({ep_val})"
        )

    ppu_count = dp_val * tp_val * pp_val
    if topo_nodes > 1:
        if ppu_count % topo_nodes != 0:
            raise ValidationError(
                f"{context}: topology.nodes ({topo_nodes}) must evenly divide effective "
                f"SGLang world size ({ppu_count})"
            )
        _validate_multinode_tp_pp_layout(
            tp_val=tp_val,
            pp_val=pp_val,
            topo_nodes=topo_nodes,
            devices_per_node=devices_per_node,
            context=context,
        )

    moe_a2a = (_flag_value(entries, "--moe-a2a-backend") or "none").strip().lower()
    ep_enabled = ep_val > 1 or moe_a2a != "none"
    return ppu_count, dp_val, tp_val, pp_val, attn_dp_val, ep_enabled


def _validate_resolved_topology_and_placeholders(
    resolved: Mapping[str, Any],
    *,
    context: str,
) -> tuple[list[str], dict[str, str]]:
    """Validate topology/placeholder invariants and return dummy-substituted flags/env."""
    flags = list(resolved["flags"])
    server_env = dict(resolved["server_env"])
    topo = resolved["topology"]
    topo_kind = topo["kind"]
    topo_nodes = int(topo["nodes"])
    topo_devices_per_node = int(topo["devices_per_node"])
    topo_capacity = topo_nodes * topo_devices_per_node
    preset_name = resolved.get("preset")

    expected_topo_kind = (
        CANONICAL_PRESET_TOPOLOGY_KIND.get(str(preset_name))
        if preset_name is not None
        else None
    )
    if expected_topo_kind is not None and topo_kind != expected_topo_kind:
        raise ValidationError(
            f"{context}: preset {preset_name!r} requires topology.kind={expected_topo_kind!r}, "
            f"got {topo_kind!r}"
        )
    if topo_kind == "pd_disagg" and preset_name not in {"pd_prefill", "pd_decode"}:
        raise ValidationError(
            f"{context}: pd_disagg topology must use paired 'pd_prefill' and 'pd_decode' preset slots"
        )

    entries = _parse_flag_entries(flags)
    flag_names = {name for name, _, _ in entries}
    calc_ppu_count, _, _, _, _, ep_enabled = _compute_sglang_parallelism(
        entries,
        topo_nodes=topo_nodes,
        devices_per_node=topo_devices_per_node,
        context=context,
    )
    ppu_count = max(int(resolved["ppu_count"]), calc_ppu_count)

    if ppu_count > topo_capacity:
        raise ValidationError(
            f"{context}: resolved parallelism requires {ppu_count} devices, "
            f"exceeding declared topology capacity ({topo_nodes}x{topo['devices_per_node']}={topo_capacity})"
        )
    if topo_nodes > 1 and ppu_count % topo_nodes != 0:
        raise ValidationError(
            f"{context}: topology.nodes ({topo_nodes}) must evenly divide effective "
            f"SGLang world size ({ppu_count})"
        )
    local_gpu_count = ppu_count // topo_nodes

    placeholders = _extract_placeholders_from_flags_and_env(flags, server_env)
    unknown = sorted(placeholders - set(TOPOLOGY_PLACEHOLDERS))
    if unknown:
        raise ValidationError(
            f"{context}: unknown topology placeholder(s): {', '.join(unknown)}"
        )
    for loc, raw_str in _walk_strings({"flags": flags, "env": server_env}, context):
        if _ADJACENT_NUMERIC_PLACEHOLDER_RE.search(raw_str):
            raise ValidationError(
                f"{loc}: numeric topology placeholder must not be concatenated with digits: {raw_str!r}"
            )

    def _validate_ib_device_wiring(ib_dev: str | None, *, backend: str) -> None:
        if not isinstance(ib_dev, str):
            raise ValidationError(
                f"{context}: {backend} disaggregation requires "
                f"--disaggregation-ib-device to wire '{{ib_devices}}'"
            )
        if ib_dev == "{ib_devices}":
            return
        if ib_dev.startswith("{"):
            try:
                parsed_ib = json.loads(ib_dev)
            except json.JSONDecodeError as exc:
                raise ValidationError(
                    f"{context}: --disaggregation-ib-device invalid JSON map: {exc}"
                ) from exc
            if isinstance(parsed_ib, dict) and parsed_ib:
                if not all(
                    isinstance(k, str)
                    and k.isdigit()
                    and str(int(k)) == k
                    and v == "{ib_devices}"
                    for k, v in parsed_ib.items()
                ):
                    raise ValidationError(
                        f"{context}: per-GPU --disaggregation-ib-device JSON map must map "
                        f"canonical numeric GPU ID strings to exact '{{ib_devices}}', got {ib_dev!r}"
                    )
                expected_gpus = set(range(local_gpu_count))
                configured_gpus = {int(k) for k in parsed_ib}
                if configured_gpus != expected_gpus:
                    missing_gpus = sorted(expected_gpus - configured_gpus)
                    extra_gpus = sorted(configured_gpus - expected_gpus)
                    raise ValidationError(
                        f"{context}: per-GPU --disaggregation-ib-device JSON map must cover "
                        f"all {local_gpu_count} local GPUs (0..{local_gpu_count - 1}); "
                        f"missing={missing_gpus}, extra={extra_gpus}"
                    )
                return
        raise ValidationError(
            f"{context}: {backend} disaggregation requires --disaggregation-ib-device "
            f"to be exact '{{ib_devices}}' (or a complete per-GPU JSON map of '{{ib_devices}}'), got {ib_dev!r}"
        )

    def _require_multinode_wiring() -> None:
        nnodes_raw = _flag_value(entries, "--nnodes")
        if nnodes_raw is None:
            raise ValidationError(
                f"{context}: {topo_kind} topology (nodes={topo_nodes}) requires "
                f"--nnodes matching topology.nodes ({topo_nodes})"
            )
        try:
            nnodes_val = int(nnodes_raw)
        except ValueError as exc:
            raise ValidationError(
                f"{context}: --nnodes must be an integer, got {nnodes_raw!r}"
            ) from exc
        if nnodes_val != topo_nodes:
            raise ValidationError(
                f"{context}: --nnodes ({nnodes_val}) must match topology.nodes ({topo_nodes})"
            )

        node_rank_raw = _flag_value(entries, "--node-rank")
        if node_rank_raw != "{node_rank}":
            raise ValidationError(
                f"{context}: {topo_kind} topology requires --node-rank to be exact '{{node_rank}}', "
                f"got {node_rank_raw!r}"
            )
        for probe_rank in (0, topo_nodes - 1):
            sub_rank_str = _substitute_placeholders_in_str(
                node_rank_raw, _build_placeholder_table({"node_rank": probe_rank})
            )
            try:
                parsed_rank = int(sub_rank_str)
            except ValueError as exc:
                raise ValidationError(
                    f"{context}: --node-rank must resolve to an integer, got {sub_rank_str!r}"
                ) from exc
            if parsed_rank != probe_rank:
                raise ValidationError(
                    f"{context}: --node-rank must preserve exact identity for rank {probe_rank}"
                )

        dist_addr_raw = _flag_value(entries, "--dist-init-addr", "--nccl-init-addr")
        if dist_addr_raw not in {
            "{master_addr}:{master_port}",
            "[{master_addr}]:{master_port}",
        }:
            raise ValidationError(
                f"{context}: {topo_kind} topology requires --dist-init-addr "
                f"to be exact '{{master_addr}}:{{master_port}}' or '[{{master_addr}}]:{{master_port}}', "
                f"got {dist_addr_raw!r}"
            )
        probe_host = "::1" if dist_addr_raw.startswith("[") else "127.0.0.1"
        sub_dist_addr = _substitute_placeholders_in_str(
            dist_addr_raw,
            _build_placeholder_table({"master_addr": probe_host, "master_port": 29500}),
        )
        _parse_host_port(
            sub_dist_addr,
            context=f"{context}.dist_init_addr",
            label="--dist-init-addr",
        )

    if topo_kind == "single_node":
        if placeholders:
            raise ValidationError(
                f"{context}: single_node topology must not contain placeholders: "
                f"{', '.join(sorted(placeholders))}"
            )
        bad_mn = sorted(flag_names & MULTI_NODE_LAUNCH_FLAGS)
        if bad_mn:
            raise ValidationError(
                f"{context}: single_node topology must not set multi-node flag(s): "
                f"{', '.join(bad_mn)}"
            )
        bad_pd = sorted(flag_names & PD_DISAGG_FLAGS)
        if bad_pd:
            raise ValidationError(
                f"{context}: single_node topology must not set disaggregation flag(s) "
                f"({', '.join(bad_pd)}); declare paired pd_disagg presets instead"
            )
    elif topo_kind == "multi_node_ep":
        bad_pd = sorted(flag_names & PD_DISAGG_FLAGS)
        if bad_pd:
            raise ValidationError(
                f"{context}: multi_node_ep topology must not set disaggregation flag(s) "
                f"({', '.join(bad_pd)}); use paired pd_disagg presets instead"
            )
        missing = sorted(MULTI_NODE_REQUIRED_PLACEHOLDERS - placeholders)
        if missing:
            raise ValidationError(
                f"{context}: multi_node_ep topology missing required placeholder(s): "
                f"{', '.join(missing)}"
            )
        _require_multinode_wiring()
        if ppu_count != topo_capacity:
            raise ValidationError(
                f"{context}: multi_node_ep parallelism ({ppu_count}) must match "
                f"declared topology capacity ({topo_nodes}x{topo['devices_per_node']}={topo_capacity})"
            )
        if not ep_enabled:
            raise ValidationError(
                f"{context}: multi_node_ep topology requires effective expert parallelism "
                f"(--ep-size > 1 or --moe-a2a-backend != 'none')"
            )
    elif topo_kind == "pd_disagg":
        if "--kv-transfer-config" in flag_names:
            raise ValidationError(
                f"{context}: SGLang pd_disagg presets must use --disaggregation-mode "
                f"and --disaggregation-* flags rather than vLLM --kv-transfer-config"
            )
        for dead_vllm_env in ("VLLM_HOST_IP", "VLLM_MOONCAKE_BOOTSTRAP_PORT"):
            if dead_vllm_env in server_env:
                raise ValidationError(
                    f"{context}: SGLang pd_disagg does not read {dead_vllm_env}; "
                    f"wire {{kv_ip}} via SGLANG_HOST_IP and {{kv_port}} via --disaggregation-bootstrap-port"
                )
        role = topo.get("role")
        if role is None:
            if preset_name == "pd_prefill":
                role = "prefill"
            elif preset_name == "pd_decode":
                role = "decode"
        disagg_mode = _flag_value(entries, "--disaggregation-mode")
        if disagg_mode is None or disagg_mode == "null":
            raise ValidationError(
                f"{context}: pd_disagg preset requires --disaggregation-mode "
                f"{role or 'prefill|decode'!r}"
            )
        if role is not None and disagg_mode != role:
            raise ValidationError(
                f"{context}: pd_disagg role {role!r} requires "
                f"--disaggregation-mode={role!r}, got {disagg_mode!r}"
            )
        transfer_backend = (
            _flag_value(entries, "--disaggregation-transfer-backend") or "mooncake"
        ).strip()
        if transfer_backend not in SUPPORTED_DISAGG_BACKENDS:
            raise ValidationError(
                f"{context}: unsupported --disaggregation-transfer-backend {transfer_backend!r}"
            )
        if role == "prefill" and transfer_backend == "fake":
            raise ValidationError(
                f"{context}: SGLang pd_prefill does not support --disaggregation-transfer-backend "
                f"'fake' (fake transfer backend is decode-only)"
            )

        required_pd_ph = (
            PD_DISAGG_DECODE_REQUIRED_PLACEHOLDERS
            if role == "decode"
            else PD_DISAGG_PREFILL_REQUIRED_PLACEHOLDERS
        )
        if transfer_backend == "mooncake_tcp":
            required_pd_ph = required_pd_ph - {"ib_devices"}
        missing = sorted(required_pd_ph - placeholders)
        if missing:
            raise ValidationError(
                f"{context}: pd_disagg topology missing required placeholder(s): "
                f"{', '.join(missing)}"
            )

        if transfer_backend in {"mooncake", "mooncake_tcp", "nixl"}:
            host_ip_env = server_env.get("SGLANG_HOST_IP")
            if host_ip_env != "{kv_ip}":
                raise ValidationError(
                    f"{context}: {transfer_backend} disaggregation requires env "
                    f"SGLANG_HOST_IP to be exact '{{kv_ip}}', got {host_ip_env!r}"
                )
            if transfer_backend in {"mooncake", "nixl"}:
                _validate_ib_device_wiring(
                    _flag_value(entries, "--disaggregation-ib-device"),
                    backend=transfer_backend,
                )
            bootstrap_port_raw = _flag_value(entries, "--disaggregation-bootstrap-port")
            if role == "prefill":
                if bootstrap_port_raw != "{kv_port}":
                    raise ValidationError(
                        f"{context}: {transfer_backend} pd_prefill requires "
                        f"--disaggregation-bootstrap-port to be exact '{{kv_port}}', got {bootstrap_port_raw!r}"
                    )
                _validate_port_value(
                    _substitute_placeholders_in_str(bootstrap_port_raw),
                    context=f"{context}.disaggregation_bootstrap_port",
                )
            elif role == "decode":
                if bootstrap_port_raw is not None or "kv_port" in placeholders:
                    raise ValidationError(
                        f"{context}: {transfer_backend} pd_decode must not set "
                        f"--disaggregation-bootstrap-port or {{kv_port}}; decoders receive "
                        f"prefill bootstrap endpoints per request via sglang_router"
                    )

        if topo_nodes == 1:
            bad_mn = sorted(flag_names & MULTI_NODE_LAUNCH_FLAGS)
            if bad_mn:
                raise ValidationError(
                    f"{context}: single-node pd_disagg topology must not set multi-node flag(s): "
                    f"{', '.join(bad_mn)}"
                )
            bad_mn_ph = sorted(placeholders & MULTI_NODE_REQUIRED_PLACEHOLDERS)
            if bad_mn_ph:
                raise ValidationError(
                    f"{context}: single-node pd_disagg topology must not contain multi-node placeholder(s): "
                    f"{', '.join(bad_mn_ph)}"
                )
        else:
            missing_mn = sorted(MULTI_NODE_REQUIRED_PLACEHOLDERS - placeholders)
            if missing_mn:
                raise ValidationError(
                    f"{context}: multi-node pd_disagg topology missing required placeholder(s): "
                    f"{', '.join(missing_mn)}"
                )
            _require_multinode_wiring()

    sub_flags, sub_env = substitute_validation_placeholders(flags, server_env)
    _validate_flag_list(sub_flags, context=context, allow_policy_flags=True)
    for loc, s in _walk_strings({"flags": sub_flags, "env": sub_env}, context):
        findings = _scan_text(s, context=loc, allow_loopback=True)
        if findings:
            raise ValidationError("; ".join(findings))
    return sub_flags, sub_env


def _validate_assets_map_variant_isolation(
    ppu_blk: Mapping[str, Any],
    model_blk: Mapping[str, Any],
    variants_blk: Mapping[str, Any],
    assets_map: Mapping[str, str],
    *,
    context: str,
) -> None:
    """Reject multi-variant assets_map collisions that merge distinct variants into one checkpoint."""
    if not assets_map or len(variants_blk) <= 1:
        return
    top_id = _effective_variant_identity(ppu_blk, model_blk, {})
    eff_ids = {
        vn: _effective_variant_identity(ppu_blk, model_blk, vc)
        for vn, vc in variants_blk.items()
    }

    def _canonicalize_ckpt_ref(raw: str) -> str:
        val = raw.strip()
        if not val:
            return val
        norm = posixpath.normpath(val)
        if norm.startswith("//") and not norm.startswith("///"):
            norm = "/" + norm.lstrip("/")
        if os.path.exists(norm):
            return os.path.realpath(norm)
        return norm

    def _resolved_ckpt(ident: Mapping[str, Any]) -> str:
        asset_key = ident["assets"]["model"]
        return _canonicalize_ckpt_ref(assets_map.get(asset_key, ident["hf_model_id"]))

    top_asset = top_id["assets"]["model"]
    top_ckpt = _resolved_ckpt(top_id)
    for vname, vcfg in variants_blk.items():
        if vname == "default":
            continue
        exempt = set(vcfg.get("allow_shared_fields", []))
        if "assets" in exempt:
            continue
        vid = eff_ids[vname]
        v_asset = vid["assets"]["model"]
        v_ckpt = _resolved_ckpt(vid)
        if (v_asset in assets_map or top_asset in assets_map) and v_ckpt == top_ckpt:
            raise ValidationError(
                f"{context}.variants.{vname}: assets_map resolves model asset {_safe_diag_repr(v_asset)} "
                f"to {_safe_diag_repr(v_ckpt)}, colliding with default/top-level checkpoint "
                "(use distinct mapped paths or declare allow_shared_fields: [assets])"
            )
        for other_name, other_id in eff_ids.items():
            if other_name == vname:
                continue
            o_asset = other_id["assets"]["model"]
            o_ckpt = _resolved_ckpt(other_id)
            if (v_asset in assets_map or o_asset in assets_map) and v_ckpt == o_ckpt:
                target = (
                    "default/top-level checkpoint"
                    if other_name == "default"
                    else f"variant {other_name!r}"
                )
                raise ValidationError(
                    f"{context}.variants.{vname}: assets_map resolves model asset {_safe_diag_repr(v_asset)} "
                    f"to {_safe_diag_repr(v_ckpt)}, colliding with {target} "
                    "(use distinct mapped paths or declare allow_shared_fields: [assets])"
                )


def _effective_variant_identity(
    ppu_blk: Mapping[str, Any],
    model_blk: Mapping[str, Any],
    var_blk: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute effective runtime asset/model identity using resolver fallback order."""
    eff_assets = {**ppu_blk["assets"], **(var_blk.get("assets") or {})}
    asset_name = eff_assets["model"]
    hf_model_id = (
        var_blk.get("hf_model_id")
        or ppu_blk.get("hf_model_id")
        or model_blk["model_id"]
    )
    served_name = var_blk.get("served_name") or ppu_blk.get("served_name") or asset_name
    golden_model_key = (
        var_blk.get("golden_model_key") or ppu_blk.get("golden_model_key") or asset_name
    )
    return {
        "assets": eff_assets,
        "hf_model_id": hf_model_id,
        "served_name": served_name,
        "golden_model_key": golden_model_key,
    }


def _resolve_recipe_from_doc(
    recipe: Mapping[str, Any],
    *,
    recipe_label: str,
    policy: Mapping[str, Any] | None = None,
    policy_label: str = "",
    tier: str | None = None,
    variant: str | None = None,
    hardware: str | None = None,
    preset: str | None = None,
    cache: str | None = None,
    dummy_weights: bool = False,
    layer_override: Any = _UNSET,
    assets_map_path: Path | None = None,
    cache_revision: str | None = None,
    host: str = "127.0.0.1",
    port: int = 9985,
    served_model_name: str | None = None,
    include_runner_flags: bool = True,
) -> dict[str, Any]:
    """Resolve an already structurally validated recipe dict (+ optional policy dict)."""
    ppu_blk = recipe["ppu"]
    if ppu_blk["framework"] != "sglang":
        raise ValidationError(
            f"{recipe_label}: executor only resolves framework='sglang', got {ppu_blk['framework']!r}"
        )

    eff_cache = cache
    eff_dummy = dummy_weights
    eff_layers: int | str | None = None if layer_override is _UNSET else layer_override
    eff_preset = preset
    eff_variant = (
        variant
        if variant is not None
        else (policy.get("variant") if policy is not None else None) or "default"
    )

    if policy is not None:
        if tier is None:
            raise ValidationError("--tier is required when --policy is supplied")
        policy_model_id = policy.get("model_id")
        recipe_model_id = recipe["model"]["model_id"]
        if policy_model_id != recipe_model_id:
            raise ValidationError(
                f"{policy_label or recipe_label}: policy model_id {policy_model_id!r} "
                f"does not match recipe model.model_id {recipe_model_id!r}"
            )
        tiers = policy.get("tiers", {})
        if tier not in tiers:
            raise ValidationError(f"{policy_label}: tier {tier!r} not found")
        tier_cfg = tiers[tier]
        if tier == "fast_sanity":
            if eff_cache is None:
                eff_cache = tier_cfg.get("cache", "enabled")
            if not dummy_weights:
                eff_dummy = bool(tier_cfg.get("dummy_weights", True))
            if layer_override is _UNSET and "layer_override" in tier_cfg:
                eff_layers = tier_cfg["layer_override"]
            if eff_preset is None:
                eff_preset = tier_cfg.get("preset")
        elif tier == "model_sanity":
            stages = tier_cfg.get("stages", [])
            if eff_cache is None and stages:
                eff_cache = stages[0].get("cache", "enabled")
            for st in stages:
                if eff_preset is None and st.get("preset"):
                    eff_preset = st["preset"]
                    break

    variants_blk = recipe.get("variants", {})
    if eff_variant not in variants_blk:
        raise ValidationError(f"{recipe_label}: unknown variant {eff_variant!r}")
    var_blk = variants_blk[eff_variant]

    presets_blk = recipe.get("presets", {})
    preset_blk: Mapping[str, Any] = {}
    if eff_preset is not None:
        if eff_preset not in presets_blk:
            raise ValidationError(f"{recipe_label}: unknown preset {eff_preset!r}")
        preset_blk = presets_blk[eff_preset]
        allowed_vars = preset_blk.get("variants")
        if allowed_vars is not None and eff_variant not in allowed_vars:
            raise ValidationError(
                f"{recipe_label}: preset {eff_preset!r} does not support variant {eff_variant!r}"
            )

    var_hw = _infer_variant_hardware(
        var_blk, context=f"{recipe_label}.variants.{eff_variant}"
    )
    preset_hw = preset_blk.get("hardware")
    policy_hw = policy.get("hardware") if policy is not None else None
    if hardware is not None:
        eff_hardware = hardware
        if policy_hw is not None and policy_hw != eff_hardware:
            raise ValidationError(
                f"{recipe_label}: requested hardware {eff_hardware!r} conflicts with policy hardware {policy_hw!r}"
            )
    elif policy_hw is not None:
        eff_hardware = str(policy_hw)
    elif preset_hw is not None:
        eff_hardware = str(preset_hw)
    else:
        eff_hardware = var_hw

    if var_blk.get("hardware") is not None and var_blk["hardware"] != eff_hardware:
        raise ValidationError(
            f"{recipe_label}: variant {eff_variant!r} targets hardware "
            f"{var_blk['hardware']!r}, incompatible with {eff_hardware!r}"
        )
    if preset_hw is not None and preset_hw != eff_hardware:
        raise ValidationError(
            f"{recipe_label}: preset {eff_preset!r} targets hardware "
            f"{preset_hw!r}, incompatible with {eff_hardware!r}"
        )
    _validate_hardware_precision(
        hardware=eff_hardware,
        precision=var_blk["precision"],
        context=f"{recipe_label}.variants.{eff_variant}",
    )

    ordered: list[tuple[str, str | None, bool]] = []
    server_env: dict[str, str] = {}

    # 1. base_args / base_env
    model_blk = recipe["model"]
    ordered = _apply_flag_entries(ordered, list(model_blk.get("base_args", [])))
    server_env.update(model_blk.get("base_env", {}))

    # 2. variants.<variant>
    ordered = _apply_flag_entries(ordered, list(var_blk.get("extra_args", [])))
    server_env.update(var_blk.get("extra_env", {}))

    # 3. hardware_overrides.<hw>
    hw_blk = recipe.get("hardware_overrides", {}).get(eff_hardware, {})
    ordered = _apply_flag_entries(ordered, list(hw_blk.get("extra_args", [])))
    server_env.update(hw_blk.get("extra_env", {}))

    # 4. enabled features (ppu.default_features overridden by preset.select.features)
    enabled_features = {name: True for name in ppu_blk.get("default_features", [])}
    select_features = preset_blk.get("select", {}).get("features", {})
    for feat_name, flag_on in select_features.items():
        enabled_features[feat_name] = bool(flag_on)

    features_blk = recipe.get("features", {})
    active_features: list[str] = []
    for feat_name, is_enabled in enabled_features.items():
        if not is_enabled:
            continue
        if feat_name not in features_blk:
            raise ValidationError(f"{recipe_label}: unknown feature {feat_name!r}")
        active_features.append(feat_name)
        feat = features_blk[feat_name]
        if "modes" in feat:
            mode_name = feat.get("default_mode") or next(iter(feat["modes"]))
            mode_cfg = feat["modes"][mode_name]
            ordered = _apply_flag_entries(ordered, list(mode_cfg.get("args", [])))
            server_env.update(mode_cfg.get("env", {}))
        else:
            ordered = _apply_flag_entries(ordered, list(feat.get("args", [])))
            server_env.update(feat.get("env", {}))

    # 5. presets.<slot>
    if preset_blk:
        if preset_blk.get("remove_args"):
            ordered = _remove_flag_entries(ordered, list(preset_blk["remove_args"]))
        if preset_blk.get("tp") is not None:
            ordered = _apply_flag_entries(ordered, ["--tp-size", str(preset_blk["tp"])])
        for json_flag, json_patch in preset_blk.get("json_args", {}).items():
            ordered = _merge_json_flag(ordered, json_flag, json_patch)
        ordered = _apply_flag_entries(ordered, list(preset_blk.get("args", [])))
        server_env.update(preset_blk.get("env", {}))

    eff_topology = copy.deepcopy(
        preset_blk.get("topology") or var_blk.get("topology") or ppu_blk["topology"]
    )
    if eff_topology["kind"] == "pd_disagg" and "role" not in eff_topology:
        if eff_preset == "pd_prefill":
            eff_topology["role"] = "prefill"
        elif eff_preset == "pd_decode":
            eff_topology["role"] = "decode"

    # 6. policy switches (prefix cache, dummy load format, layer override)
    if eff_cache == "enabled":
        ordered = _remove_flag_entries(ordered, ["--disable-radix-cache"])
    elif eff_cache == "disabled":
        ordered = _apply_flag_entries(ordered, ["--disable-radix-cache"])
    elif eff_cache is not None:
        raise ValidationError(f"unsupported cache value: {eff_cache!r}")

    if include_runner_flags:
        ordered = _apply_flag_entries(ordered, ["--enable-metrics"])
    if eff_dummy:
        ordered = _apply_flag_entries(ordered, ["--load-format", "dummy"])
    if eff_layers is not None:
        if type(eff_layers) is not int or eff_layers < 1:
            raise ValidationError(
                f"layer_override must be a positive int or None, got {eff_layers!r}"
            )
        ordered = _merge_json_flag(
            ordered,
            "--json-model-override-args",
            {"num_hidden_layers": eff_layers},
        )

    if (
        any(
            flag == "--json-model-override-args"
            and "num_hidden_layers" in json.loads(value)
            for flag, value, _ in ordered
            if value is not None
        )
        and eff_layers is None
    ):
        raise ValidationError("layer override must come from the test policy")

    # 7. assets-map + runtime endpoint injection
    assets_map = _load_assets_map(assets_map_path)
    _validate_assets_map_variant_isolation(
        ppu_blk,
        model_blk,
        variants_blk,
        assets_map,
        context=recipe_label,
    )
    eff_identity = _effective_variant_identity(ppu_blk, model_blk, var_blk)
    eff_assets = eff_identity["assets"]
    asset_name = eff_assets["model"]
    hf_model_id = eff_identity["hf_model_id"]
    resolved_model = assets_map.get(asset_name, hf_model_id)
    resolved_tokenizer = assets_map.get(eff_assets["tokenizer"], resolved_model)
    effective_served_name = served_model_name or eff_identity["served_name"]
    golden_model_key = eff_identity["golden_model_key"]

    flags_only = _render_entries(ordered)
    argv = [
        "python3",
        "-I",
        "-B",
        "-m",
        "sglang.launch_server",
        "--model-path",
        resolved_model,
        "--tokenizer-path",
        resolved_tokenizer,
        "--served-model-name",
        effective_served_name,
        "--host",
        host,
        "--port",
        str(port),
        *flags_only,
    ]

    ppu_count, dp_val, _, _, attn_dp_val, _ = _compute_sglang_parallelism(
        ordered,
        topo_nodes=int(eff_topology["nodes"]),
        devices_per_node=int(eff_topology["devices_per_node"]),
        context=recipe_label,
    )

    digest_payload: dict[str, Any] = {
        "framework": ppu_blk["framework"],
        "model_id": model_blk["model_id"],
        "hf_model_id": hf_model_id,
        "variant": eff_variant,
        "hardware": eff_hardware,
        "preset": eff_preset,
        "tier": tier,
        "cache": eff_cache,
        "dummy_weights": eff_dummy,
        "layer_override": eff_layers,
        "flags": _canonical_flags(ordered),
        "server_env": dict(sorted(server_env.items())),
    }
    if eff_assets != ppu_blk["assets"]:
        digest_payload["assets"] = dict(sorted(eff_assets.items()))
    if resolved_tokenizer != resolved_model:
        digest_payload["distinct_tokenizer"] = resolved_tokenizer
    if cache_revision is not None:
        digest_payload["cache_revision"] = cache_revision
    config_digest = hashlib.sha256(
        json.dumps(digest_payload, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )
    ).hexdigest()

    result = {
        "argv": argv,
        "flags": flags_only,
        "server_env": server_env,
        "ppu_count": ppu_count,
        "dp_size": dp_val,
        "attn_dp_size": attn_dp_val,
        "model_id": model_blk["model_id"],
        "hf_model_id": hf_model_id,
        "assets": copy.deepcopy(eff_assets),
        "resolved_model": resolved_model,
        "resolved_tokenizer": resolved_tokenizer,
        "golden_model_key": golden_model_key,
        "served_name": effective_served_name,
        "config_digest": config_digest,
        "tier": tier,
        "preset": eff_preset,
        "variant": eff_variant,
        "precision": _normalize_precision(
            var_blk["precision"], context=f"{recipe_label}.variants.{eff_variant}"
        ),
        "hardware": eff_hardware,
        "topology": eff_topology,
        "enabled_features": active_features,
        "layer_override": eff_layers,
        "dummy_weights": eff_dummy,
    }
    _validate_resolved_topology_and_placeholders(
        result, context=f"{recipe_label}:{eff_variant}:{eff_preset or 'baseline'}"
    )
    return result


def _validate_pd_disagg_pair(
    prefill_res: Mapping[str, Any],
    decode_res: Mapping[str, Any],
    *,
    context: str,
) -> None:
    """Validate runtime wire compatibility between paired ``pd_prefill`` and ``pd_decode`` roles."""
    p_entries = _parse_flag_entries(list(prefill_res["flags"]))
    d_entries = _parse_flag_entries(list(decode_res["flags"]))

    p_backend = (
        _flag_value(p_entries, "--disaggregation-transfer-backend") or "mooncake"
    ).strip()
    d_backend = (
        _flag_value(d_entries, "--disaggregation-transfer-backend") or "mooncake"
    ).strip()
    if p_backend != d_backend:
        raise ValidationError(
            f"{context}: pd_prefill and pd_decode must use the same "
            f"--disaggregation-transfer-backend, got {p_backend!r} vs {d_backend!r}"
        )

    if p_backend == "mooncake":
        p_proto = (
            prefill_res["server_env"].get("MOONCAKE_PROTOCOL", "rdma").strip().lower()
        )
        d_proto = (
            decode_res["server_env"].get("MOONCAKE_PROTOCOL", "rdma").strip().lower()
        )
        if p_proto != d_proto:
            raise ValidationError(
                f"{context}: pd_prefill and pd_decode must agree on MOONCAKE_PROTOCOL, "
                f"got {p_proto!r} vs {d_proto!r}"
            )

    p_page = (
        _parse_pos_int_flag(
            "--page-size", _flag_value(p_entries, "--page-size"), context=context
        )
        or 1
    )
    d_page = (
        _parse_pos_int_flag(
            "--page-size", _flag_value(d_entries, "--page-size"), context=context
        )
        or 1
    )
    if p_page != d_page:
        raise ValidationError(
            f"{context}: pd_prefill and pd_decode --page-size mismatch "
            f"({p_page} vs {d_page}); SGLang disaggregation requires identical KV page sizes"
        )

    p_kv_dtype = (_flag_value(p_entries, "--kv-cache-dtype") or "auto").strip().lower()
    d_kv_dtype = (_flag_value(d_entries, "--kv-cache-dtype") or "auto").strip().lower()
    if p_kv_dtype != d_kv_dtype:
        raise ValidationError(
            f"{context}: pd_prefill and pd_decode --kv-cache-dtype mismatch "
            f"({p_kv_dtype!r} vs {d_kv_dtype!r})"
        )

    p_dsv4_attn = (
        (_flag_value(p_entries, "--dsv4-attn-backend") or "auto").strip().lower()
    )
    d_dsv4_attn = (
        (_flag_value(d_entries, "--dsv4-attn-backend") or "auto").strip().lower()
    )
    if p_dsv4_attn != d_dsv4_attn:
        raise ValidationError(
            f"{context}: pd_prefill and pd_decode --dsv4-attn-backend mismatch "
            f"({p_dsv4_attn!r} vs {d_dsv4_attn!r})"
        )

    p_dsv4_layout = prefill_res["server_env"].get("SGLANG_DSV4_KV_LAYOUT", "").strip()
    d_dsv4_layout = decode_res["server_env"].get("SGLANG_DSV4_KV_LAYOUT", "").strip()
    if p_dsv4_layout != d_dsv4_layout:
        raise ValidationError(
            f"{context}: pd_prefill and pd_decode SGLANG_DSV4_KV_LAYOUT mismatch "
            f"({p_dsv4_layout!r} vs {d_dsv4_layout!r})"
        )


def resolve_recipe(
    recipe_path: Path,
    *,
    policy_path: Path | None = None,
    tier: str | None = None,
    variant: str | None = None,
    hardware: str | None = None,
    preset: str | None = None,
    cache: str | None = None,
    dummy_weights: bool = False,
    layer_override: Any = _UNSET,
    assets_map_path: Path | None = None,
    cache_revision: str | None = None,
    host: str = "127.0.0.1",
    port: int = 9985,
    served_model_name: str | None = None,
) -> dict[str, Any]:
    """Resolve recipe + optional policy tier into effective SGLang argv, env, and digest."""
    recipe = _load_yaml(recipe_path)
    validate_serving_recipe_structure(
        recipe, context=str(recipe_path), validate_expansions=False
    )

    policy: dict[str, Any] | None = None
    if policy_path is not None:
        policy = _load_yaml(policy_path)
        validate_policy_structure(policy, context=str(policy_path), base_dir=REPO_ROOT)

    resolved = _resolve_recipe_from_doc(
        recipe,
        recipe_label=str(recipe_path),
        policy=policy,
        policy_label=str(policy_path) if policy_path is not None else "",
        tier=tier,
        variant=variant,
        hardware=hardware,
        preset=preset,
        cache=cache,
        dummy_weights=dummy_weights,
        layer_override=layer_override,
        assets_map_path=assets_map_path,
        cache_revision=cache_revision,
        host=host,
        port=port,
        served_model_name=served_model_name,
    )
    if resolved["preset"] in {"pd_prefill", "pd_decode"}:
        peer_preset = (
            "pd_decode" if resolved["preset"] == "pd_prefill" else "pd_prefill"
        )
        peer_resolved = _resolve_recipe_from_doc(
            recipe,
            recipe_label=str(recipe_path),
            variant=resolved["variant"],
            hardware=resolved["hardware"],
            preset=peer_preset,
            cache=cache,
            dummy_weights=dummy_weights,
            layer_override=layer_override,
            assets_map_path=assets_map_path,
            cache_revision=cache_revision,
            host=host,
            port=port,
            served_model_name=served_model_name,
        )
        prefill_res, decode_res = (
            (resolved, peer_resolved)
            if resolved["preset"] == "pd_prefill"
            else (peer_resolved, resolved)
        )
        _validate_pd_disagg_pair(
            prefill_res,
            decode_res,
            context=f"{recipe_path}:{resolved['variant']}:{resolved['hardware']}:pd_pair",
        )
    return resolved


def _scan_text(text: str, *, context: str, allow_loopback: bool = False) -> list[str]:
    findings: list[str] = []
    for label, pattern in _SENSITIVE_PATTERNS:
        if pattern.search(text):
            findings.append(f"{context}: matched {label}")
    for match in _IPV4_RE.finditer(text):
        ip = match.group(0)
        if allow_loopback and ip == "127.0.0.1":
            continue
        findings.append(f"{context}: matched ipv4_literal")
    return findings


def _safe_diag_repr(value: str) -> str:
    """Return ``repr(value)`` unless ``value`` contains sensitive text, else ``'<redacted>'``."""
    if _scan_text(value, context="diag", allow_loopback=True):
        return "<redacted>"
    return repr(value)


def _redact_error_text(text: str) -> str:
    """Scrub secret tokens, internal domains, prohibited tokens, and private IPs from error text."""
    scrubbed = text
    for label, pattern in _SENSITIVE_PATTERNS:
        if label != "local_path":
            scrubbed = pattern.sub("<redacted>", scrubbed)
    return _IPV4_RE.sub(
        lambda m: m.group(0) if m.group(0) == "127.0.0.1" else "<redacted-ip>",
        scrubbed,
    )


def _walk_strings(obj: Any, prefix: str = "") -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    if isinstance(obj, str):
        items.append((prefix, obj))
    elif isinstance(obj, list):
        for idx, elem in enumerate(obj):
            items.extend(_walk_strings(elem, f"{prefix}[{idx}]"))
    elif isinstance(obj, dict):
        for key, val in obj.items():
            items.extend(_walk_strings(key, f"{prefix}.{key}(key)"))
            items.extend(_walk_strings(val, f"{prefix}.{key}"))
    return items


def redact_and_verify_public_payload(
    payload: Any, *, allow_loopback: bool = True
) -> None:
    """Fail closed if a public output dict/text contains internal paths, hosts, IPs, or tokens."""
    text = (
        payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    )
    findings = _scan_text(text, context="public_payload", allow_loopback=allow_loopback)
    if findings:
        raise ValidationError("redaction gate blocked output: " + "; ".join(findings))


def _validate_env_block(
    env: Any,
    *,
    external_env: set[str],
    context: str,
) -> None:
    checked = _check_str_dict(env, context=context)
    for name, val in checked.items():
        if name in FORBIDDEN_RECIPE_ENV:
            raise ValidationError(
                f"{context}: runtime-mode env {name!r} must not be set in recipe"
            )
        if not name.startswith(ALLOWED_ENV_PREFIXES):
            raise ValidationError(
                f"{context}: env {name!r} does not match allowed prefixes"
            )
        if (name.startswith(EXTERNAL_ENV_PREFIXES) or name in EXTERNAL_ENV_EXACT) and (
            name not in external_env
        ):
            raise ValidationError(
                f"{context}: external env {name!r} must be listed in ppu.external_env"
            )
        unknown_ph = sorted(
            set(_PLACEHOLDER_RE.findall(val)) - set(TOPOLOGY_PLACEHOLDERS)
        )
        if unknown_ph:
            raise ValidationError(
                f"{context}: env {name!r} has unknown placeholder(s): "
                f"{', '.join(unknown_ph)}"
            )


_HF_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


def _validate_hf_model_id(value: Any, *, context: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{context} must be non-empty string")
    if not _HF_REPO_ID_RE.fullmatch(value):
        raise ValidationError(
            f"{context} must be a valid owner/name Hugging Face repo ID: {value!r}"
        )
    owner, repo = value.split("/", 1)
    if owner.endswith(("-", ".")) or repo.endswith(("-", ".")) or len(repo) > 96:
        raise ValidationError(
            f"{context} violates Hugging Face repo ID rules (max 96 chars for name): {value!r}"
        )


def _validate_flag_list(
    tokens: Any, *, context: str, allow_policy_flags: bool = False
) -> None:
    raw = _check_str_list(tokens, context=context)
    for flag, val, _ in _parse_flag_entries(raw):
        if not allow_policy_flags and flag in FORBIDDEN_RECIPE_FLAGS:
            raise ValidationError(f"{context}: framework runner owns {flag}")
        if flag == "--mem-fraction-static":
            if val is None:
                raise ValidationError(
                    f"{context}: --mem-fraction-static requires a float in (0.0, 1.0]"
                )
            try:
                util = float(val)
            except ValueError as exc:
                raise ValidationError(
                    f"{context}: --mem-fraction-static must be a float in (0.0, 1.0], got {val!r}"
                ) from exc
            if not math.isfinite(util) or not (0.0 < util <= 1.0):
                raise ValidationError(
                    f"{context}: --mem-fraction-static must be in (0.0, 1.0], got {val!r}"
                )
        if flag in JSON_FLAGS:
            if val is None:
                raise ValidationError(f"{context}: JSON flag {flag} requires a value")
            if "`" in val or "$(" in val:
                raise ValidationError(
                    f"{context}: JSON flag {flag} must not contain shell substitution"
                )
            try:
                parsed = json.loads(val)
            except json.JSONDecodeError as exc:
                raise ValidationError(
                    f"{context}: JSON flag {flag} invalid JSON: {exc}"
                ) from exc
            if not isinstance(parsed, dict):
                raise ValidationError(
                    f"{context}: JSON flag {flag} must decode to a JSON object"
                )
            unknown_ph = sorted(
                _find_placeholders_in_obj(parsed) - set(TOPOLOGY_PLACEHOLDERS)
            )
            if unknown_ph:
                raise ValidationError(
                    f"{context}: JSON flag {flag} has unknown placeholder(s): "
                    f"{', '.join(unknown_ph)}"
                )
        elif val is not None:
            unknown_ph = sorted(
                set(_PLACEHOLDER_RE.findall(val)) - set(TOPOLOGY_PLACEHOLDERS)
            )
            if unknown_ph:
                raise ValidationError(
                    f"{context}: flag {flag} has unknown placeholder(s): "
                    f"{', '.join(unknown_ph)}"
                )


def validate_serving_recipe_structure(
    recipe: dict[str, Any],
    *,
    context: str,
    allow_internal_visibility: bool = False,
    validate_expansions: bool = True,
) -> None:
    """Validate structure, env allowlist, JSON flags, and redaction for ``ppu-serving/v1``."""
    _check_keys(
        recipe,
        allowed={
            "schema_version",
            "ppu",
            "model",
            "features",
            "variants",
            "hardware_overrides",
            "presets",
            "notes",
        },
        required={"schema_version", "ppu", "model"},
        context=context,
    )
    if recipe["schema_version"] != SERVING_SCHEMA:
        raise ValidationError(f"{context}: schema_version must be {SERVING_SCHEMA!r}")

    ppu = _check_keys(
        recipe["ppu"],
        allowed={
            "framework",
            "visibility",
            "upstream_schema",
            "assets",
            "hf_model_id",
            "golden_model_key",
            "served_name",
            "topology",
            "default_features",
            "external_env",
        },
        required={"framework", "visibility", "assets", "topology"},
        context=f"{context}.ppu",
    )
    if ppu["framework"] not in {"vllm", "sglang"}:
        raise ValidationError(f"{context}.ppu.framework invalid: {ppu['framework']!r}")
    if ppu["visibility"] not in {"public", "internal_restricted"}:
        raise ValidationError(
            f"{context}.ppu.visibility invalid: {ppu['visibility']!r}"
        )
    if ppu["visibility"] == "internal_restricted" and not allow_internal_visibility:
        raise ValidationError(
            f"{context}.ppu.visibility='internal_restricted' is forbidden in public repo"
        )
    assets = _check_keys(
        ppu["assets"],
        allowed={"model", "tokenizer"},
        required={"model", "tokenizer"},
        context=f"{context}.ppu.assets",
    )
    _check_str_dict(assets, context=f"{context}.ppu.assets")

    _validate_topology_block(ppu["topology"], context=f"{context}.ppu.topology")

    if "hf_model_id" in ppu:
        _validate_hf_model_id(ppu["hf_model_id"], context=f"{context}.ppu.hf_model_id")
    external_env = set(
        _check_str_list(
            ppu.get("external_env", []), context=f"{context}.ppu.external_env"
        )
    )

    model = _check_keys(
        recipe["model"],
        allowed={"model_id", "base_args", "base_env"},
        required={"model_id", "base_args"},
        context=f"{context}.model",
    )
    if not isinstance(model["model_id"], str) or not model["model_id"].strip():
        raise ValidationError(f"{context}.model.model_id must be non-empty string")
    _validate_flag_list(model["base_args"], context=f"{context}.model.base_args")
    _validate_env_block(
        model.get("base_env", {}),
        external_env=external_env,
        context=f"{context}.model.base_env",
    )

    features = recipe.get("features", {})
    if not isinstance(features, dict):
        raise ValidationError(f"{context}.features must be mapping")
    for fname in ppu.get("default_features", []):
        if fname not in features:
            raise ValidationError(
                f"{context}.ppu.default_features references unknown feature {fname!r}"
            )
    for fname, fcfg in features.items():
        if not isinstance(fcfg, dict):
            raise ValidationError(f"{context}.features.{fname} must be mapping")
        if "modes" in fcfg:
            _check_keys(
                fcfg,
                allowed={"default_mode", "modes"},
                required={"modes"},
                context=f"{context}.features.{fname}",
            )
            if not isinstance(fcfg["modes"], dict) or not fcfg["modes"]:
                raise ValidationError(
                    f"{context}.features.{fname}.modes must be non-empty mapping"
                )
            for mname, mcfg in fcfg["modes"].items():
                _check_keys(
                    mcfg,
                    allowed={"args", "env"},
                    context=f"{context}.features.{fname}.modes.{mname}",
                )
                _validate_flag_list(
                    mcfg.get("args", []),
                    context=f"{context}.features.{fname}.modes.{mname}.args",
                )
                _validate_env_block(
                    mcfg.get("env", {}),
                    external_env=external_env,
                    context=f"{context}.features.{fname}.modes.{mname}.env",
                )
        else:
            _check_keys(
                fcfg, allowed={"args", "env"}, context=f"{context}.features.{fname}"
            )
            _validate_flag_list(
                fcfg.get("args", []), context=f"{context}.features.{fname}.args"
            )
            _validate_env_block(
                fcfg.get("env", {}),
                external_env=external_env,
                context=f"{context}.features.{fname}.env",
            )

    if ppu["framework"] == "sglang":
        raw_variants = recipe.get("variants")
        if not isinstance(raw_variants, dict) or not raw_variants:
            raise ValidationError(
                f"{context}.variants must be a non-empty mapping for sglang serving recipes"
            )
        variants_blk = raw_variants
    else:
        variants_blk = recipe.get("variants", {})
        if not isinstance(variants_blk, dict):
            raise ValidationError(f"{context}.variants must be mapping")

    variant_compat_hw_map: dict[str, frozenset[str]] = {}
    for vname, vcfg in variants_blk.items():
        _check_keys(
            vcfg,
            allowed={
                "precision",
                "hardware",
                "topology",
                "extra_args",
                "extra_env",
                "assets",
                "hf_model_id",
                "golden_model_key",
                "served_name",
                "allow_shared_fields",
            },
            required={"precision"},
            context=f"{context}.variants.{vname}",
        )
        if "allow_shared_fields" in vcfg:
            shared = _check_str_list(
                vcfg["allow_shared_fields"],
                context=f"{context}.variants.{vname}.allow_shared_fields",
            )
            unknown_shared = sorted(set(shared) - set(VARIANT_ISOLATED_FIELDS))
            if unknown_shared:
                raise ValidationError(
                    f"{context}.variants.{vname}.allow_shared_fields has unsupported field(s): "
                    f"{', '.join(unknown_shared)}"
                )
            if len(shared) != len(set(shared)):
                raise ValidationError(
                    f"{context}.variants.{vname}.allow_shared_fields must not contain duplicates"
                )
        compat_hw = _variant_compatible_hardware(
            vcfg, context=f"{context}.variants.{vname}"
        )
        for hw in sorted(compat_hw):
            _validate_hardware_precision(
                hardware=hw,
                precision=vcfg["precision"],
                context=f"{context}.variants.{vname}",
            )
        variant_compat_hw_map[vname] = compat_hw
        if "topology" in vcfg:
            _validate_topology_block(
                vcfg["topology"], context=f"{context}.variants.{vname}.topology"
            )
        if "assets" in vcfg:
            v_assets = _check_keys(
                vcfg["assets"],
                allowed={"model", "tokenizer"},
                required={"model", "tokenizer"},
                context=f"{context}.variants.{vname}.assets",
            )
            _check_str_dict(v_assets, context=f"{context}.variants.{vname}.assets")
        if "hf_model_id" in vcfg:
            _validate_hf_model_id(
                vcfg["hf_model_id"], context=f"{context}.variants.{vname}.hf_model_id"
            )
        for key in ("golden_model_key", "served_name"):
            if key in vcfg and (
                not isinstance(vcfg[key], str) or not vcfg[key].strip()
            ):
                raise ValidationError(
                    f"{context}.variants.{vname}.{key} must be non-empty string"
                )
        _validate_flag_list(
            vcfg.get("extra_args", []), context=f"{context}.variants.{vname}.extra_args"
        )
        _validate_env_block(
            vcfg.get("extra_env", {}),
            external_env=external_env,
            context=f"{context}.variants.{vname}.extra_env",
        )

    if len(variants_blk) > 1:
        top_id = _effective_variant_identity(ppu, model, {})
        eff_ids = {
            vn: _effective_variant_identity(ppu, model, vc)
            for vn, vc in variants_blk.items()
        }
        for vname, vcfg in variants_blk.items():
            if vname == "default":
                continue
            exempt = set(vcfg.get("allow_shared_fields", []))
            vid = eff_ids[vname]
            if "assets" not in exempt:
                if "assets" not in vcfg:
                    raise ValidationError(
                        f"{context}.variants.{vname}: multi-variant recipe requires explicit "
                        f"'assets' on non-default variant {vname!r} "
                        "(or declare allow_shared_fields: [assets])"
                    )
                v_model_asset = vid["assets"]["model"]
                if v_model_asset == top_id["assets"]["model"]:
                    raise ValidationError(
                        f"{context}.variants.{vname}.assets.model ({_safe_diag_repr(v_model_asset)}) "
                        "duplicates default/top-level value "
                        "(set distinct assets or declare allow_shared_fields: [assets])"
                    )
                for other_name, other_id in eff_ids.items():
                    if other_name == vname:
                        continue
                    if v_model_asset == other_id["assets"]["model"]:
                        target = (
                            "default/top-level value"
                            if other_name == "default"
                            else f"variant {other_name!r}"
                        )
                        raise ValidationError(
                            f"{context}.variants.{vname}.assets.model ({_safe_diag_repr(v_model_asset)}) "
                            f"duplicates {target} "
                            "(set distinct assets or declare allow_shared_fields: [assets])"
                        )

            for field in ("hf_model_id", "golden_model_key", "served_name"):
                if field in exempt:
                    continue
                if field not in vcfg:
                    raise ValidationError(
                        f"{context}.variants.{vname}: multi-variant recipe requires explicit "
                        f"{field!r} on non-default variant {vname!r} "
                        f"(or declare allow_shared_fields: [{field}])"
                    )
                val = vid[field]
                if val == top_id[field]:
                    raise ValidationError(
                        f"{context}.variants.{vname}.{field} ({_safe_diag_repr(val)}) duplicates "
                        f"default/top-level value (set distinct {field} or declare "
                        f"allow_shared_fields: [{field}])"
                    )
                for other_name, other_id in eff_ids.items():
                    if other_name == vname:
                        continue
                    if val == other_id[field]:
                        target = (
                            "default/top-level value"
                            if other_name == "default"
                            else f"variant {other_name!r}"
                        )
                        raise ValidationError(
                            f"{context}.variants.{vname}.{field} ({_safe_diag_repr(val)}) duplicates "
                            f"{target} (set distinct {field} or declare "
                            f"allow_shared_fields: [{field}])"
                        )

    for hname, hcfg in recipe.get("hardware_overrides", {}).items():
        if hname not in SUPPORTED_HARDWARE:
            raise ValidationError(
                f"{context}.hardware_overrides: unknown hardware {hname!r}"
            )
        _check_keys(
            hcfg,
            allowed={"extra_args", "extra_env"},
            context=f"{context}.hardware_overrides.{hname}",
        )
        _validate_flag_list(
            hcfg.get("extra_args", []),
            context=f"{context}.hardware_overrides.{hname}.extra_args",
        )
        _validate_env_block(
            hcfg.get("extra_env", {}),
            external_env=external_env,
            context=f"{context}.hardware_overrides.{hname}.extra_env",
        )

    presets_blk = recipe.get("presets", {})
    if not isinstance(presets_blk, dict):
        raise ValidationError(f"{context}.presets must be mapping")
    if ("pd_prefill" in presets_blk) != ("pd_decode" in presets_blk):
        raise ValidationError(
            f"{context}.presets: pd_prefill and pd_decode must be declared as a pair"
        )
    for pname, pcfg in presets_blk.items():
        if pname not in ALLOWED_PRESET_SLOTS:
            raise ValidationError(f"{context}.presets: unknown preset slot {pname!r}")
        _check_keys(
            pcfg,
            allowed={
                "hardware",
                "variants",
                "topology",
                "tp",
                "select",
                "args",
                "env",
                "remove_args",
                "json_args",
            },
            context=f"{context}.presets.{pname}",
        )
        if "hardware" in pcfg and pcfg["hardware"] not in SUPPORTED_HARDWARE:
            raise ValidationError(
                f"{context}.presets.{pname}.hardware invalid: {pcfg['hardware']!r}"
            )
        if "variants" in pcfg:
            p_variants = _check_str_list(
                pcfg["variants"], context=f"{context}.presets.{pname}.variants"
            )
            if not p_variants:
                raise ValidationError(
                    f"{context}.presets.{pname}.variants must be non-empty"
                )
            for vname in p_variants:
                if vname not in variants_blk:
                    raise ValidationError(
                        f"{context}.presets.{pname}.variants references unknown variant {vname!r}"
                    )
                if (
                    "hardware" in pcfg
                    and pcfg["hardware"] not in variant_compat_hw_map[vname]
                ):
                    raise ValidationError(
                        f"{context}.presets.{pname}: variant {vname!r} compatible hardware "
                        f"{sorted(variant_compat_hw_map[vname])} conflicts with preset hardware {pcfg['hardware']!r}"
                    )
        expected_role = (
            "prefill"
            if pname == "pd_prefill"
            else ("decode" if pname == "pd_decode" else None)
        )
        expected_topo_kind = CANONICAL_PRESET_TOPOLOGY_KIND.get(pname)
        if "topology" in pcfg:
            ptopo = _validate_topology_block(
                pcfg["topology"],
                context=f"{context}.presets.{pname}.topology",
                allow_role=True,
                expected_role=expected_role,
            )
            if expected_topo_kind is not None and ptopo["kind"] != expected_topo_kind:
                raise ValidationError(
                    f"{context}.presets.{pname}.topology.kind must be {expected_topo_kind!r}, "
                    f"got {ptopo['kind']!r}"
                )
            if ptopo["kind"] == "pd_disagg" and pname not in {
                "pd_prefill",
                "pd_decode",
            }:
                raise ValidationError(
                    f"{context}.presets.{pname}: pd_disagg topology must use "
                    f"paired 'pd_prefill' and 'pd_decode' preset slots"
                )
        elif expected_topo_kind is not None:
            candidate_topos = [
                vcfg.get("topology") or ppu["topology"]
                for vname, vcfg in variants_blk.items()
                if (pcfg.get("variants") is None or vname in pcfg["variants"])
                and (
                    pcfg.get("hardware") is None
                    or pcfg["hardware"] in variant_compat_hw_map[vname]
                )
            ] or [ppu["topology"]]
            for inherited_topo in candidate_topos:
                if inherited_topo["kind"] != expected_topo_kind:
                    raise ValidationError(
                        f"{context}.presets.{pname}: preset {pname!r} requires "
                        f"topology.kind={expected_topo_kind!r}, got inherited {inherited_topo['kind']!r}"
                    )
        if "tp" in pcfg and (type(pcfg["tp"]) is not int or pcfg["tp"] <= 0):
            raise ValidationError(f"{context}.presets.{pname}.tp must be positive int")
        if "select" in pcfg:
            sel = _check_keys(
                pcfg["select"],
                allowed={"features"},
                context=f"{context}.presets.{pname}.select",
            )
            for fname, enabled in sel.get("features", {}).items():
                if fname not in features or not isinstance(enabled, bool):
                    raise ValidationError(
                        f"{context}.presets.{pname}.select.features.{fname} invalid"
                    )
        _check_str_list(
            pcfg.get("remove_args", []),
            context=f"{context}.presets.{pname}.remove_args",
        )
        for jflag, jval in pcfg.get("json_args", {}).items():
            if jflag not in JSON_FLAGS or not isinstance(jval, dict):
                raise ValidationError(
                    f"{context}.presets.{pname}.json_args.{jflag} invalid"
                )
            unknown_jph = sorted(
                _find_placeholders_in_obj(jval) - set(TOPOLOGY_PLACEHOLDERS)
            )
            if unknown_jph:
                raise ValidationError(
                    f"{context}.presets.{pname}.json_args.{jflag} has unknown placeholder(s): "
                    f"{', '.join(unknown_jph)}"
                )
        _validate_flag_list(
            pcfg.get("args", []), context=f"{context}.presets.{pname}.args"
        )
        _validate_env_block(
            pcfg.get("env", {}),
            external_env=external_env,
            context=f"{context}.presets.{pname}.env",
        )
        if variants_blk:
            matched_variants = [
                vname
                for vname in variants_blk
                if (pcfg.get("variants") is None or vname in pcfg["variants"])
                and (
                    pcfg.get("hardware") is None
                    or pcfg["hardware"] in variant_compat_hw_map[vname]
                )
            ]
            if not matched_variants:
                raise ValidationError(
                    f"{context}.presets.{pname}: matches no compatible variant in recipe"
                )

    for loc, s in _walk_strings(recipe, context):
        findings = _scan_text(s, context=loc, allow_loopback=False)
        if findings:
            raise ValidationError("; ".join(findings))

    if "pd_prefill" in presets_blk and "pd_decode" in presets_blk and variants_blk:
        all_targets = _iter_variant_preset_targets(recipe, context=context)
        prefill_targets = {
            (vname, hw) for vname, pname, hw in all_targets if pname == "pd_prefill"
        }
        decode_targets = {
            (vname, hw) for vname, pname, hw in all_targets if pname == "pd_decode"
        }
        if not prefill_targets or prefill_targets != decode_targets:
            raise ValidationError(
                f"{context}.presets: pd_prefill and pd_decode must resolve to identical "
                f"(variant, hardware) targets, got pd_prefill={sorted(prefill_targets)} "
                f"vs pd_decode={sorted(decode_targets)}"
            )

    if validate_expansions and ppu["framework"] == "sglang":
        pd_prefill_by_key: dict[tuple[str, str], dict[str, Any]] = {}
        pd_decode_by_key: dict[tuple[str, str], dict[str, Any]] = {}
        for vname, pname, target_hw in _iter_variant_preset_targets(
            recipe, context=context
        ):
            resolved_exp = _resolve_recipe_from_doc(
                recipe,
                recipe_label=context,
                variant=vname,
                hardware=target_hw,
                preset=pname,
                cache=None,
                include_runner_flags=False,
            )
            if pname == "pd_prefill":
                pd_prefill_by_key[(vname, target_hw)] = resolved_exp
            elif pname == "pd_decode":
                pd_decode_by_key[(vname, target_hw)] = resolved_exp
        for pair_key in sorted(set(pd_prefill_by_key) | set(pd_decode_by_key)):
            _validate_pd_disagg_pair(
                pd_prefill_by_key[pair_key],
                pd_decode_by_key[pair_key],
                context=f"{context}:{pair_key[0]}:{pair_key[1]}:pd_pair",
            )


def validate_dataset_structure(dataset: dict[str, Any], *, context: str) -> None:
    """Validate structure and redaction for ``ppu-dataset/v1``."""
    _check_keys(
        dataset,
        allowed={
            "schema_version",
            "id",
            "version",
            "dataset_asset",
            "subset_list",
            "few_shot_num",
            "few_shot_random",
            "num_expected",
            "scoring",
            "filters",
            "sandbox",
        },
        required={"schema_version", "id", "version", "dataset_asset", "scoring"},
        context=context,
    )
    if dataset["schema_version"] != DATASET_SCHEMA:
        raise ValidationError(f"{context}: schema_version must be {DATASET_SCHEMA!r}")
    if "sandbox" in dataset:
        sb = _check_keys(
            dataset["sandbox"], allowed={"enabled"}, context=f"{context}.sandbox"
        )
        if not isinstance(sb.get("enabled"), bool):
            raise ValidationError(f"{context}.sandbox.enabled must be bool")
    for loc, s in _walk_strings(dataset, context):
        findings = _scan_text(s, context=loc, allow_loopback=False)
        if findings:
            raise ValidationError("; ".join(findings))


def _validate_stage_cache_rules(
    stage: dict[str, Any],
    *,
    resolved_flags: list[str],
    context: str,
) -> None:
    kind = stage.get("kind")
    cache = stage.get("cache")
    cache_mode = stage.get("cache_mode")
    if kind in {"sanity", "evalscope"}:
        if cache != "enabled":
            raise ValidationError(f"{context}: kind={kind!r} requires cache='enabled'")
        if any(
            f in resolved_flags for f in ("--disable-cuda-graph", "--enforce-eager")
        ):
            raise ValidationError(
                f"{context}: kind={kind!r} cannot run with --disable-cuda-graph or --enforce-eager"
            )
    elif kind == "bench":
        if cache == "enabled" and cache_mode not in ALLOWED_CACHE_MODES:
            raise ValidationError(
                f"{context}: bench with cache='enabled' requires cache_mode in {sorted(ALLOWED_CACHE_MODES)}"
            )
        if cache not in {"enabled", "disabled"}:
            raise ValidationError(f"{context}: invalid bench cache={cache!r}")
        bench_input = stage.get("input")
        if isinstance(bench_input, dict) and "range_ratio" in bench_input:
            ratio = bench_input["range_ratio"]
            if (
                isinstance(ratio, bool)
                or not isinstance(ratio, int | float)
                or not (0.0 <= float(ratio) <= 1.0)
            ):
                raise ValidationError(
                    f"{context}: input.range_ratio must be in [0.0, 1.0], got {ratio!r}"
                )


def validate_policy_structure(
    policy: dict[str, Any],
    *,
    context: str,
    base_dir: Path = REPO_ROOT,
) -> None:
    """Validate structure, cache guardrails, and redaction for ``ppu-test-policy/v1``."""
    _check_keys(
        policy,
        allowed={
            "schema_version",
            "model_id",
            "recipe",
            "variant",
            "hardware",
            "frameworks",
            "measurement",
            "accuracy_gate",
            "eval_defaults",
            "replaces_legacy_cases",
            "stages",
            "tiers",
        },
        required={"schema_version", "model_id", "hardware"},
        context=context,
    )
    if policy["schema_version"] != POLICY_SCHEMA:
        raise ValidationError(f"{context}: schema_version must be {POLICY_SCHEMA!r}")
    if policy["hardware"] not in SUPPORTED_HARDWARE:
        raise ValidationError(
            f"{context}.hardware invalid: {policy['hardware']!r}; "
            f"expected one of {sorted(SUPPORTED_HARDWARE)}"
        )
    if "tiers" not in policy and "stages" not in policy:
        raise ValidationError(f"{context}: policy must define 'tiers' or 'stages'")
    if not isinstance(policy["model_id"], str) or not policy["model_id"].strip():
        raise ValidationError(f"{context}.model_id must be non-empty string")
    if "variant" in policy and (
        not isinstance(policy["variant"], str) or not policy["variant"].strip()
    ):
        raise ValidationError(f"{context}.variant must be non-empty string")
    policy_variant = policy.get("variant", "default")

    recipe_path: Path | None = None
    recipes_root = (base_dir / "test" / "ppu" / "recipes").resolve()
    if "recipe" in policy:
        rel = Path(policy["recipe"])
        recipe_path = rel if rel.is_absolute() else (base_dir / rel)
        if (
            not recipe_path.resolve().is_relative_to(recipes_root)
            or not recipe_path.is_file()
        ):
            raise ValidationError(
                f"{context}: referenced recipe not found: {policy['recipe']}"
            )
        rec_doc = _load_yaml(recipe_path)
        recipe_model_blk = rec_doc.get("model")
        recipe_model_id = (
            recipe_model_blk.get("model_id")
            if isinstance(recipe_model_blk, Mapping)
            else None
        )
        if not isinstance(recipe_model_id, str) or not recipe_model_id.strip():
            raise ValidationError(
                f"{context}: referenced recipe {policy['recipe']} is missing model.model_id"
            )
        if policy["model_id"] != recipe_model_id:
            raise ValidationError(
                f"{context}: policy model_id {policy['model_id']!r} does not match "
                f"recipe {policy['recipe']} model.model_id {recipe_model_id!r}"
            )
        variants_blk = rec_doc.get("variants", {})
        if policy_variant not in variants_blk:
            raise ValidationError(
                f"{context}: unknown variant {policy_variant!r} in {policy['recipe']}"
            )
        var_cfg = variants_blk[policy_variant]
        _validate_hardware_precision(
            hardware=policy["hardware"],
            precision=var_cfg.get("precision"),
            context=f"{context} ({policy['recipe']}#{policy_variant})",
        )
        var_explicit_hw = var_cfg.get("hardware")
        if var_explicit_hw is not None and var_explicit_hw != policy["hardware"]:
            raise ValidationError(
                f"{context}: variant {policy_variant!r} in {policy['recipe']} targets hardware "
                f"{var_explicit_hw!r}, incompatible with policy hardware {policy['hardware']!r}"
            )

    tiers = policy.get("tiers", {})
    if tiers:
        _check_keys(
            tiers,
            allowed=POLICY_TIERS,
            context=f"{context}.tiers",
        )
    if "replaces_legacy_cases" in policy:
        cases = policy["replaces_legacy_cases"]
        if (
            not isinstance(cases, list)
            or not cases
            or any(not isinstance(case, str) or not case.strip() for case in cases)
            or len(set(cases)) != len(cases)
        ):
            raise ValidationError(
                f"{context}.replaces_legacy_cases must be a nonempty list of unique case IDs"
            )
        if "model_sanity" not in tiers:
            raise ValidationError(
                f"{context}.replaces_legacy_cases requires tiers.model_sanity"
            )
    if tiers:
        if "fast_sanity" in tiers:
            fs = _check_keys(
                tiers["fast_sanity"],
                allowed={
                    "dummy_weights",
                    "layer_override",
                    "fallback_full_layers",
                    "cache",
                    "preset",
                    "probe",
                },
                required={"dummy_weights", "cache", "probe"},
                context=f"{context}.tiers.fast_sanity",
            )
            lo = fs.get("layer_override")
            if (
                fs["dummy_weights"] is not True
                or (
                    "layer_override" in fs
                    and lo is not None
                    and (type(lo) is not int or lo <= 0)
                )
                or (
                    "fallback_full_layers" in fs
                    and type(fs["fallback_full_layers"]) is not bool
                )
            ):
                raise ValidationError(
                    f"{context}: invalid fast sanity weight or layer policy"
                )
            if fs["cache"] not in ("enabled", "disabled"):
                raise ValidationError(
                    f"{context}.tiers.fast_sanity.cache must be 'enabled' or 'disabled'"
                )
            if recipe_path is not None:
                resolve_recipe(
                    recipe_path,
                    variant=policy_variant,
                    hardware=policy["hardware"],
                    preset=fs.get("preset"),
                    cache=fs.get("cache"),
                    tier="fast_sanity",
                )
        if "model_sanity" in tiers:
            ms = _check_keys(
                tiers["model_sanity"],
                allowed={"stages"},
                required={"stages"},
                context=f"{context}.tiers.model_sanity",
            )
            if not isinstance(ms["stages"], list):
                raise ValidationError(
                    f"{context}.tiers.model_sanity.stages must be a list"
                )
            seen_kinds: set[str] = set()
            for idx, stage in enumerate(ms["stages"]):
                _check_keys(
                    stage,
                    allowed={
                        "id",
                        "kind",
                        "cache",
                        "cache_mode",
                        "preset",
                        "dataset",
                        "limit",
                        "seed",
                        "eval_batch_size",
                        "generation",
                        "probe",
                        "engine",
                        "input",
                        "concurrencies",
                        "request_count_by_concurrency",
                        "warmup_count_by_concurrency",
                    },
                    required={"id", "kind", "cache"},
                    context=f"{context}.tiers.model_sanity.stages[{idx}]",
                )
                stage_kind = stage["kind"]
                if stage_kind not in {"sanity", "evalscope", "bench"}:
                    raise ValidationError(
                        f"{context}.tiers.model_sanity.stages[{idx}]: invalid kind {stage_kind!r}"
                    )
                if stage_kind in seen_kinds:
                    raise ValidationError(
                        f"{context}.tiers.model_sanity: duplicate stage kind {stage_kind!r}"
                    )
                seen_kinds.add(stage_kind)
                flags: list[str] = []
                if recipe_path is not None:
                    resolved = resolve_recipe(
                        recipe_path,
                        variant=policy_variant,
                        hardware=policy["hardware"],
                        preset=stage.get("preset"),
                        cache=stage.get("cache"),
                        tier="model_sanity",
                    )
                    flags = resolved["flags"]
                _validate_stage_cache_rules(
                    stage,
                    resolved_flags=flags,
                    context=f"{context}.tiers.model_sanity.stages[{idx}]",
                )
                if stage.get("cache") != "enabled":
                    raise ValidationError(
                        f"{context}.tiers.model_sanity.stages[{idx}]: requires cache='enabled' for shared model_sanity server"
                    )
                dataset_ref = stage.get("dataset")
                if dataset_ref is not None:
                    if not isinstance(dataset_ref, str) or "@" not in dataset_ref:
                        raise ValidationError(
                            f"{context}.tiers.model_sanity.stages[{idx}].dataset must be <id>@<version>"
                        )
                    ds_id, ds_ver = dataset_ref.split("@", 1)
                    ds_path = (
                        base_dir
                        / "test"
                        / "ppu"
                        / "policy"
                        / "datasets"
                        / f"{ds_id}.yaml"
                    )
                    if not ds_path.is_file():
                        raise ValidationError(
                            f"{context}: referenced dataset definition not found: {dataset_ref}"
                        )
                    ds_doc = _load_yaml(ds_path)
                    if (
                        ds_doc.get("id") != ds_id
                        or str(ds_doc.get("version")) != ds_ver
                    ):
                        raise ValidationError(
                            f"{context}: dataset {dataset_ref} version mismatch in {ds_path}"
                        )
            missing_kinds = sorted({"sanity", "evalscope", "bench"} - seen_kinds)
            if missing_kinds:
                raise ValidationError(
                    f"{context}.tiers.model_sanity: missing required stage kind(s): {', '.join(missing_kinds)}"
                )
            stage_presets = {stage.get("preset") for stage in ms["stages"]}
            if len(stage_presets) > 1:
                raise ValidationError(
                    f"{context}.tiers.model_sanity: all shared-server stages must agree on preset, "
                    f"got {sorted(str(p) for p in stage_presets)}"
                )

    for idx, stage in enumerate(policy.get("stages", [])):
        _validate_stage_cache_rules(
            stage, resolved_flags=[], context=f"{context}.stages[{idx}]"
        )

    for loc, s in _walk_strings(policy, context):
        findings = _scan_text(s, context=loc, allow_loopback=False)
        if findings:
            raise ValidationError("; ".join(findings))


def _build_sglang_serve_parser() -> argparse.ArgumentParser:
    """Register the installed candidate's actual SGLang options."""
    server_args = importlib.import_module("sglang.srt.server_args").ServerArgs
    parser = argparse.ArgumentParser(prog="sglang", allow_abbrev=False)
    server_args.add_cli_args(parser)
    return parser


def validate_flags_with_parser(
    parser: argparse.ArgumentParser,
    flags: list[str],
    *,
    context: str,
) -> None:
    """Validate every ``--flag`` and parsed types/choices against an argparse parser."""
    option_actions = getattr(parser, "_option_string_actions", {})
    for flag, _, _ in _parse_flag_entries(flags):
        if flag not in option_actions:
            raise ValidationError(f"{context}: unknown SGLang serve flag {flag!r}")

    test_argv = ["--model-path", "placeholder/model", *flags]
    orig_error = parser.error
    errors: list[str] = []

    def _capture_error(message: str) -> None:
        errors.append(message)
        raise SystemExit(2)

    parser.error = _capture_error  # type: ignore[method-assign]
    try:
        parser.parse_args(test_argv)
    except SystemExit as exc:
        detail = "; ".join(errors) if errors else f"exit={exc.code}"
        raise ValidationError(f"{context}: parser rejected args: {detail}") from exc
    finally:
        parser.error = orig_error  # type: ignore[method-assign]


def iter_recipe_catalog_entries(
    recipe_path: Path,
    recipe: dict[str, Any] | None = None,
    *,
    allow_internal_visibility: bool = False,
) -> list[dict[str, Any]]:
    """Expand every compatible ``variant x preset`` entry (including each variant baseline)."""
    rdoc = recipe if recipe is not None else _load_yaml(recipe_path)
    validate_serving_recipe_structure(
        rdoc,
        context=str(recipe_path),
        allow_internal_visibility=allow_internal_visibility,
        validate_expansions=False,
    )
    entries: list[dict[str, Any]] = []
    pd_prefill_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    pd_decode_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for vname, pname, target_hw in _iter_variant_preset_targets(
        rdoc, context=str(recipe_path)
    ):
        resolved = _resolve_recipe_from_doc(
            rdoc,
            recipe_label=str(recipe_path),
            variant=vname,
            hardware=target_hw,
            preset=pname,
            cache=None,
            include_runner_flags=False,
        )
        if pname == "pd_prefill":
            pd_prefill_by_key[(vname, target_hw)] = resolved
        elif pname == "pd_decode":
            pd_decode_by_key[(vname, target_hw)] = resolved
        entries.append(
            {
                "recipe_path": recipe_path,
                "variant": vname,
                "preset": pname,
                "hardware": target_hw,
                "resolved": resolved,
            }
        )
    for pair_key in sorted(set(pd_prefill_by_key) | set(pd_decode_by_key)):
        _validate_pd_disagg_pair(
            pd_prefill_by_key[pair_key],
            pd_decode_by_key[pair_key],
            context=f"{recipe_path}:{pair_key[0]}:{pair_key[1]}:pd_pair",
        )
    return entries


def _collect_all_recipe_flag_sets(
    recipe_path: Path,
    recipe: dict[str, Any],
    *,
    catalog_entries: Sequence[dict[str, Any]] | None = None,
) -> list[tuple[str, list[str]]]:
    sets: list[tuple[str, list[str]]] = []
    entries = (
        list(catalog_entries)
        if catalog_entries is not None
        else iter_recipe_catalog_entries(recipe_path, recipe)
    )
    for item in entries:
        vname = item["variant"]
        pname = item["preset"]
        hw = item["hardware"]
        label = (
            f"{recipe_path.name}:{vname}:{hw}:baseline"
            if pname is None
            else f"{recipe_path.name}:{vname}:{hw}:{pname}"
        )
        sub_flags, _ = _validate_resolved_topology_and_placeholders(
            item["resolved"], context=label
        )
        sets.append((label, sub_flags))
        if (
            pname is None
            and vname == "default"
            and item["resolved"]["topology"]["kind"] == "single_node"
        ):
            fs_res = _resolve_recipe_from_doc(
                recipe,
                recipe_label=str(recipe_path),
                variant=vname,
                hardware=hw,
                cache="enabled",
                dummy_weights=True,
                layer_override=2,
            )
            fs_flags, _ = _validate_resolved_topology_and_placeholders(
                fs_res, context=f"{recipe_path.name}:{hw}:fast_sanity_2l"
            )
            sets.append((f"{recipe_path.name}:{hw}:fast_sanity_2l", fs_flags))

    for feat_name, feat_cfg in recipe.get("features", {}).items():
        if "modes" in feat_cfg:
            for mode_name, mode_cfg in feat_cfg["modes"].items():
                sub_f, _ = substitute_validation_placeholders(
                    list(mode_cfg.get("args", [])), {}
                )
                sets.append(
                    (f"{recipe_path.name}:feature:{feat_name}.{mode_name}", sub_f)
                )
        else:
            sub_f, _ = substitute_validation_placeholders(
                list(feat_cfg.get("args", [])), {}
            )
            sets.append((f"{recipe_path.name}:feature:{feat_name}", sub_f))
    return sets


def _recipes_root(repo_root: Path | None = None) -> Path:
    return (repo_root or REPO_ROOT) / _RECIPES_REL


def _policy_root(repo_root: Path | None = None) -> Path:
    return (repo_root or REPO_ROOT) / _POLICY_REL


def _is_canonical_recipe_path(path: Path, *, repo_root: Path) -> bool:
    abs_path = path.resolve()
    recipes_root = _recipes_root(repo_root).resolve()
    return (
        abs_path.is_relative_to(recipes_root)
        and abs_path.suffix == ".yaml"
        and not abs_path.name.startswith(".")
    )


def _iter_repo_yaml_files(repo_root: Path) -> list[Path]:
    root = repo_root.resolve()
    if (root / ".git").exists():
        proc = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={root}",
                "-C",
                str(root),
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
                "--",
                "*.yaml",
                "*.yml",
                ":(glob)**/*.yaml",
                ":(glob)**/*.yml",
            ],
            capture_output=True,
            check=False,
        )
        if proc.returncode == 0:
            found: list[Path] = []
            for raw in proc.stdout.split(b"\0"):
                if not raw:
                    continue
                candidate = root / os.fsdecode(raw)
                if candidate.is_file():
                    found.append(candidate)
            return sorted(set(found))

    walked: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d
            for d in dirnames
            if d not in _STRAY_SCAN_PRUNE_DIRS and not d.startswith(".venv")
        )
        for fname in sorted(filenames):
            if fname.endswith((".yaml", ".yml")):
                candidate = Path(dirpath) / fname
                if candidate.is_file():
                    walked.append(candidate)
    return walked


def _unescape_yaml_scalar_text(raw: str) -> str:
    return _YAML_ESCAPE_RE.sub(
        lambda m: chr(int(m.group(1) or m.group(2) or m.group(3), 16)),
        raw,
    )


def _text_has_stray_serving_schema(text: str) -> bool:
    for match in _STRAY_SERVING_SCHEMA_RE.finditer(text):
        raw_val = match.group(2).strip()
        if len(raw_val) >= 2 and raw_val[0] == raw_val[-1] and raw_val[0] in {"'", '"'}:
            raw_val = raw_val[1:-1]
        if (
            raw_val == SERVING_SCHEMA
            or _unescape_yaml_scalar_text(raw_val) == SERVING_SCHEMA
        ):
            return True
    return False


def _mapping_node_has_serving_schema(
    node: yaml.Node,
    seen: set[int] | None = None,
) -> bool:
    if not isinstance(node, yaml.MappingNode):
        return False
    visited = seen if seen is not None else set()
    node_id = id(node)
    if node_id in visited:
        return False
    visited.add(node_id)
    for k_node, v_node in node.value:
        if (
            isinstance(k_node, yaml.ScalarNode)
            and k_node.value == "schema_version"
            and isinstance(v_node, yaml.ScalarNode)
            and v_node.value == SERVING_SCHEMA
        ):
            return True
        if (
            isinstance(k_node, yaml.ScalarNode)
            and k_node.tag == "tag:yaml.org,2002:merge"
        ):
            if isinstance(
                v_node, yaml.MappingNode
            ) and _mapping_node_has_serving_schema(v_node, visited):
                return True
            if isinstance(v_node, yaml.SequenceNode) and any(
                _mapping_node_has_serving_schema(item, visited) for item in v_node.value
            ):
                return True
    return False


def scan_stray_serving_recipes(repo_root: Path | None = None) -> list[Path]:
    """Fail closed if any ``ppu-serving/v1`` YAML exists outside ``test/ppu/recipes/*.yaml``."""
    root = (repo_root or REPO_ROOT).resolve()
    stray: list[Path] = []
    for path in _iter_repo_yaml_files(root):
        if _is_canonical_recipe_path(path, repo_root=root):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        matched = False
        try:
            for doc_node in yaml.compose_all(text):
                if doc_node is not None and _mapping_node_has_serving_schema(doc_node):
                    matched = True
                    break
        except yaml.YAMLError:
            for chunk in re.split(r"(?m)^(?:---|\.\.\.)(?=$|[\s#])", text):
                try:
                    doc_node = yaml.compose(chunk)
                except yaml.YAMLError:
                    continue
                if doc_node is not None and _mapping_node_has_serving_schema(doc_node):
                    matched = True
                    break
        if not matched and _text_has_stray_serving_schema(text):
            matched = True
        if matched:
            stray.append(path)
    if stray:
        rels = ", ".join(
            p.resolve().relative_to(root).as_posix() for p in sorted(stray)
        )
        raise ValidationError(
            f"stray {SERVING_SCHEMA!r} recipe file(s) outside {_RECIPES_REL.as_posix()}/*.yaml: "
            f"{rels}"
        )
    return []


def validate_paths(
    paths: list[Path],
    *,
    sglang_parser_mode: str = "auto",
    parser: argparse.ArgumentParser | None = None,
    allow_internal_visibility: bool = False,
    check_repo_stray_recipes: bool = False,
) -> dict[str, Any]:
    """Validate recipe, dataset, and policy YAML files."""
    if sglang_parser_mode not in {"auto", "required", "skip"}:
        raise ValidationError(f"invalid sglang_parser_mode: {sglang_parser_mode!r}")

    if check_repo_stray_recipes:
        scan_stray_serving_recipes(REPO_ROOT)

    repo_root_res = REPO_ROOT.resolve()
    recipe_items: list[tuple[Path, dict[str, Any]]] = []
    validated: list[str] = []

    for path in paths:
        doc = _load_yaml(path)
        schema = doc.get("schema_version")
        if schema == SERVING_SCHEMA:
            abs_path = path.resolve()
            if abs_path.is_relative_to(repo_root_res) and not _is_canonical_recipe_path(
                abs_path, repo_root=repo_root_res
            ):
                raise ValidationError(
                    f"{path}: {SERVING_SCHEMA!r} files inside repository must reside under "
                    f"{_RECIPES_REL.as_posix()}/*.yaml"
                )
            validate_serving_recipe_structure(
                doc,
                context=str(path),
                allow_internal_visibility=allow_internal_visibility,
                validate_expansions=False,
            )
            recipe_items.append((path, doc))
        elif schema == DATASET_SCHEMA:
            validate_dataset_structure(doc, context=str(path))
        elif schema == POLICY_SCHEMA:
            validate_policy_structure(doc, context=str(path), base_dir=REPO_ROOT)
        else:
            raise ValidationError(f"{path}: unsupported schema_version {schema!r}")
        validated.append(str(path))

    expanded_flag_sets: list[tuple[str, list[str]]] = []
    checked_combinations = 0
    for rpath, rdoc in recipe_items:
        if rdoc["ppu"]["framework"] != "sglang":
            continue
        entries = iter_recipe_catalog_entries(
            rpath, rdoc, allow_internal_visibility=allow_internal_visibility
        )
        checked_combinations += len(entries)
        expanded_flag_sets.extend(
            _collect_all_recipe_flag_sets(rpath, rdoc, catalog_entries=entries)
        )

    parser_checked = False
    checked_flag_sets = 0
    if recipe_items and sglang_parser_mode != "skip":
        active_parser = parser
        if active_parser is None:
            try:
                active_parser = _build_sglang_serve_parser()
            except ImportError as exc:
                if sglang_parser_mode == "required":
                    raise ValidationError(
                        f"SGLang parser check required but sglang is not importable: {exc}"
                    ) from exc
                active_parser = None
        if active_parser is not None:
            for label, flag_set in expanded_flag_sets:
                validate_flags_with_parser(active_parser, flag_set, context=label)
                checked_flag_sets += 1
            parser_checked = True

    return {
        "validated_files": validated,
        "parser_checked": parser_checked,
        "checked_flag_sets": checked_flag_sets,
        "checked_combinations": checked_combinations,
    }


def _discover_default_paths(repo_root: Path | None = None) -> list[Path]:
    paths: list[Path] = []
    for root in (_recipes_root(repo_root), _policy_root(repo_root)):
        if root.is_dir():
            paths.extend(
                sorted(p for p in root.rglob("*.yaml") if not p.name.startswith("."))
            )
    if not paths:
        raise ValidationError("no recipe or policy YAML files found under test/ppu")
    return paths


def _md_cell(value: str) -> str:
    return (
        value.strip().replace("|", "&#124;").replace("\r\n", "\n").replace("\n", "<br>")
    )


def _md_code(value: str) -> str:
    longest = max(
        (len(match.group(0)) for match in re.finditer(r"`+", value)), default=0
    )
    if longest:
        fence = "`" * (longest + 1)
        return f"{fence} {value} {fence}"
    return f"`{value}`"


def _md_link(label: str, repo_relative_path: str, *, base_dir: Path) -> str:
    rel = os.path.relpath(
        (REPO_ROOT / repo_relative_path).resolve(), base_dir.resolve()
    )
    posix_rel = Path(rel).as_posix()
    return f"[{_md_code(label)}]({posix_rel})"


def _format_preset(resolved: dict[str, Any]) -> str:
    preset = resolved.get("preset")
    variant = resolved.get("variant", "default")
    preset_text = _md_code(preset) if preset else f"{_md_code('default')} (none)"
    if variant != "default":
        return f"variant {_md_code(variant)}, {preset_text}"
    return preset_text


def _format_features(resolved: dict[str, Any]) -> str:
    features = resolved.get("enabled_features") or []
    if not features:
        return _md_code("none")
    return ", ".join(_md_code(name) for name in features)


def _format_flags_and_env(flags: list[str], server_env: dict[str, str]) -> str:
    parts: list[str] = []
    if flags:
        parts.append(_md_code(shlex.join(flags)))
    else:
        parts.append(_md_code("none"))
    if server_env:
        env_items = ", ".join(
            _md_code(f"{k}={v}") for k, v in sorted(server_env.items())
        )
        parts.append(f"env: {env_items}")
    return "<br>".join(parts)


def _format_fast_sanity_weights(
    policy_doc: dict[str, Any], resolved: dict[str, Any]
) -> str:
    weight = _md_code("dummy") if resolved.get("dummy_weights") else "real"
    layer_override = resolved.get("layer_override")
    if layer_override is None:
        return f"{weight}, full layers"
    fs_cfg = policy_doc.get("tiers", {}).get("fast_sanity", {})
    suffix = " (fallback: full)" if fs_cfg.get("fallback_full_layers") else ""
    return f"{weight}, {layer_override} layers{suffix}"


def _format_model_sanity_stages(policy_doc: dict[str, Any]) -> str:
    stages = policy_doc.get("tiers", {}).get("model_sanity", {}).get("stages", [])
    items: list[str] = []
    for stage in stages:
        stage_id = str(stage["id"])
        kind = stage["kind"]
        if kind == "evalscope":
            limit = stage.get("limit")
            scope = "full" if limit is None else f"limit={limit}"
            items.append(f"{_md_code(stage_id)} ({scope}, ±{GSM8K_GOLDEN_BAND_ABS})")
        elif kind == "bench":
            concs = stage.get("concurrencies") or []
            if concs:
                conc_str = ",".join(str(c) for c in concs)
                items.append(f"{_md_code(stage_id)} ({_md_code(f'C={conc_str}')})")
            else:
                items.append(_md_code(stage_id))
        else:
            items.append(_md_code(stage_id))
    return ", ".join(items) if items else _md_code("none")


def _format_golden(resolved: dict[str, Any], *, base_dir: Path) -> str:
    golden_key = resolved.get("golden_model_key")
    if not golden_key:
        return _md_code("none")
    rel_path = f"test/ppu/golden/accuracy_{golden_key}_890p.json"
    if (REPO_ROOT / rel_path).is_file():
        return _md_link(golden_key, rel_path, base_dir=base_dir)
    return _md_code(golden_key)


def _render_md_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(_md_cell(h) for h in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_md_cell(cell) for cell in row) + " |")
    return lines


def _format_topology_summary(resolved: Mapping[str, Any]) -> str:
    hw = _md_code(str(resolved["hardware"]))
    cards = int(resolved["ppu_count"])
    topo = resolved["topology"]
    kind = str(topo["kind"])
    if kind == "multi_node_ep":
        return (
            f"{hw} / {cards} / {_md_code(kind)} "
            f"({topo['nodes']}x{topo['devices_per_node']})"
        )
    if kind == "pd_disagg":
        role = topo.get("role")
        if not role:
            preset = resolved.get("preset")
            role = (
                "prefill"
                if preset == "pd_prefill"
                else ("decode" if preset == "pd_decode" else "disagg")
            )
        return f"{hw} / {cards} / {_md_code(kind)} ({role})"
    return f"{hw} / {cards} / {_md_code(kind)}"


def _format_catalog_gate_tiers(tiers: Sequence[str]) -> str:
    if not tiers:
        return f"{_md_code('static')} (catalog only)"
    ordered = ["static", *[t for t in ("fast_sanity", "model_sanity") if t in tiers]]
    return ", ".join(_md_code(t) for t in ordered)


def render_ci_matrix_markdown(*, base_dir: Path | None = None) -> str:
    """Render the PR, Nightly, and Catalog PPU CI recipe matrix Markdown document."""
    ci_dir = str(Path(__file__).resolve().parent)
    if ci_dir not in sys.path:
        sys.path.insert(0, ci_dir)
    selector = importlib.import_module("recipe_select")
    doc_dir = (
        base_dir
        if base_dir is not None
        else (REPO_ROOT / "test" / "ppu" / "README.md").parent
    )
    authoring_rel = Path(
        os.path.relpath(
            (REPO_ROOT / "test" / "ppu" / "AUTHORING.md").resolve(),
            doc_dir.resolve(),
        )
    ).as_posix()

    gate_tiers_by_key: dict[tuple[str, str, str, str | None], set[str]] = {}

    fast_keys: list[str] = []
    fast_rows: list[list[str]] = []
    for entry in selector.select("fast_sanity", "")["include"]:
        model_key = entry["model"]
        fast_keys.append(model_key)
        policy_rel = f"test/ppu/policy/models/{model_key}.yaml"
        policy_path = REPO_ROOT / policy_rel
        pol_doc = _load_yaml(policy_path)
        recipe_rel = Path(pol_doc["recipe"]).as_posix()
        resolved = resolve_recipe(
            REPO_ROOT / recipe_rel, policy_path=policy_path, tier="fast_sanity"
        )
        gate_key = (
            recipe_rel,
            resolved["variant"],
            resolved["hardware"],
            resolved["preset"],
        )
        gate_tiers_by_key.setdefault(gate_key, set()).add("fast_sanity")
        fast_rows.append(
            [
                _md_link(model_key, policy_rel, base_dir=doc_dir),
                _md_link(Path(recipe_rel).name, recipe_rel, base_dir=doc_dir),
                f"{_md_code(resolved['hardware'])} / {resolved['ppu_count']}",
                _format_preset(resolved),
                _format_features(resolved),
                _format_fast_sanity_weights(pol_doc, resolved),
                _format_flags_and_env(resolved["flags"], resolved["server_env"]),
            ]
        )

    nightly_keys: list[str] = []
    nightly_rows: list[list[str]] = []
    for entry in selector.select("model_sanity", "")["include"]:
        model_key = entry["model"]
        nightly_keys.append(model_key)
        policy_rel = f"test/ppu/policy/models/{model_key}.yaml"
        policy_path = REPO_ROOT / policy_rel
        pol_doc = _load_yaml(policy_path)
        recipe_rel = Path(pol_doc["recipe"]).as_posix()
        resolved = resolve_recipe(
            REPO_ROOT / recipe_rel, policy_path=policy_path, tier="model_sanity"
        )
        gate_key = (
            recipe_rel,
            resolved["variant"],
            resolved["hardware"],
            resolved["preset"],
        )
        gate_tiers_by_key.setdefault(gate_key, set()).add("model_sanity")
        nightly_rows.append(
            [
                _md_link(model_key, policy_rel, base_dir=doc_dir),
                _md_link(Path(recipe_rel).name, recipe_rel, base_dir=doc_dir),
                f"{_md_code(resolved['hardware'])} / {resolved['ppu_count']}",
                _format_preset(resolved),
                _format_features(resolved),
                _format_model_sanity_stages(pol_doc),
                _format_golden(resolved, base_dir=doc_dir),
                _format_flags_and_env(resolved["flags"], resolved["server_env"]),
            ]
        )

    catalog_rows: list[list[str]] = []
    recipes_root = REPO_ROOT / "test" / "ppu" / "recipes"
    if recipes_root.is_dir():
        recipe_files = sorted(
            (p for p in recipes_root.rglob("*.yaml") if not p.name.startswith(".")),
            key=lambda p: p.relative_to(REPO_ROOT).as_posix(),
        )
        for rec_path in recipe_files:
            rec_doc = _load_yaml(rec_path)
            if rec_doc.get("schema_version") != SERVING_SCHEMA:
                continue
            if rec_doc.get("ppu", {}).get("framework") != "sglang":
                continue
            rec_rel = rec_path.relative_to(REPO_ROOT).as_posix()
            for item in iter_recipe_catalog_entries(rec_path, rec_doc):
                resolved = item["resolved"]
                preset_name = item["preset"]
                gate_key = (
                    rec_rel,
                    item["variant"],
                    item["hardware"],
                    preset_name,
                )
                tiers = sorted(gate_tiers_by_key.get(gate_key, set()))
                preset_cell = (
                    _md_code(preset_name)
                    if preset_name is not None
                    else f"{_md_code('default')} (baseline)"
                )
                catalog_rows.append(
                    [
                        _md_link(rec_path.name, rec_rel, base_dir=doc_dir),
                        f"{_md_code(item['variant'])} ({_md_code(resolved['precision'])})",
                        preset_cell,
                        _format_topology_summary(resolved),
                        _format_catalog_gate_tiers(tiers),
                        _format_flags_and_env(
                            resolved["flags"], resolved["server_env"]
                        ),
                    ]
                )

    fast_only = [key for key in fast_keys if key not in set(nightly_keys)]

    lines = [
        "<!-- Generated by `python3 -B scripts/ci/ppu/ppu_recipe.py render-matrix --output test/ppu/README.md`. Do not edit manually. -->",
        "",
        "# SGLang PPU Recipe CI Matrix",
        "",
        "This document is generated from `test/ppu/recipes/` and `test/ppu/policy/models/`",
        "using the same selector logic (`scripts/ci/ppu/recipe_select.py`) as GitHub Actions",
        f"({_md_link('.github/workflows/pr-test-ppu.yml', '.github/workflows/pr-test-ppu.yml', base_dir=doc_dir)}).",
        f"For recipe/preset authoring rules, hardware-precision bindings, and CI gate admission criteria, see [AUTHORING.md]({authoring_rel}).",
        f"For runner orchestration, cluster provisioning, and CI labels, see {_md_link('scripts/ci/ppu/README.md', 'scripts/ci/ppu/README.md', base_dir=doc_dir)}.",
        "",
        "## Directory Layout",
        "",
        *_render_md_table(
            ["Path", "Schema / Kind", "Purpose"],
            [
                [
                    _md_link("AUTHORING.md", "test/ppu/AUTHORING.md", base_dir=doc_dir),
                    "Authoring Guide",
                    "Recipe boundaries, hardware-precision bindings, canonical preset slots (`prefill`, `decode`, `decode_ep`, `pd_prefill`/`pd_decode`), topology placeholders, and 3-tier CI admission rules.",
                ],
                [
                    _md_link("recipes/", "test/ppu/recipes", base_dir=doc_dir),
                    "`ppu-serving/v1` YAML",
                    "Declarative serve config per model (`base_args`, `base_env`, `features`, `variants`, `hardware_overrides`, `presets`, topology, and HF checkpoint id).",
                ],
                [
                    _md_link(
                        "policy/models/", "test/ppu/policy/models", base_dir=doc_dir
                    ),
                    "`ppu-test-policy/v1` YAML",
                    "Per-model CI policy (`hardware`, `variant`, `accuracy_gate`, and `fast_sanity` / `model_sanity` tiers).",
                ],
                [
                    _md_link(
                        "policy/datasets/", "test/ppu/policy/datasets", base_dir=doc_dir
                    ),
                    "`ppu-dataset/v1` YAML",
                    "Eval dataset definitions referenced as `<id>@<version>`.",
                ],
                [
                    _md_link("golden/", "test/ppu/golden", base_dir=doc_dir),
                    "`schema_version: 2` JSON",
                    "Reference GSM8K score and tolerance band per `golden_model_key` (`accuracy_<key>_890p.json`).",
                ],
                [
                    f"{_md_link('e2e/', 'test/ppu/e2e', base_dir=doc_dir)}, {_md_link('unit/', 'test/ppu/unit', base_dir=doc_dir)}, {_md_link('module/', 'test/ppu/module', base_dir=doc_dir)}",
                    "pytest / `unittest`",
                    "Device-facing regression suites (independent of the recipe/policy resolver; skip when no PPU device is present).",
                ],
            ],
        ),
        "",
        "## Configuration Resolution Rules",
        "",
        "- **Resolution order**: `model.base_args` / `model.base_env` -> `variants.<variant>` (default: `default`) -> `hardware_overrides.<hardware>` -> enabled `features` (`ppu.default_features` overridden by `presets.<preset>.select.features`) -> optional `presets.<preset>` (`remove_args`, `tp`, `json_args`, `args`, `env`) -> policy switches (`--disable-radix-cache`, `--load-format dummy`, `--json-model-override-args`) -> runner-owned flags (`--enable-metrics`, `--model-path`, `--tokenizer-path`, `--served-model-name`, `--host`, `--port`).",
        "- **Default preset behavior**: When a policy does not specify `preset`, CI runs the recipe's default baseline configuration. Named presets (such as `prefill`, `decode`, `decode_ep`, `pd_prefill`/`pd_decode`, `tp8`, or `model_sanity`) are only applied when explicitly selected by a policy tier/stage or `--preset`.",
        "- **Weights & layers in `fast_sanity`**: Policies declare `dummy_weights` and optional `layer_override`; current models start directly with full-layer dummy weights (`--load-format dummy`) unless `layer_override` is set.",
        "- **Shared server in `model_sanity`**: All stages in `model_sanity` (`sanity`, `evalscope`, `bench`) run against one shared server process with real weights, matching `preset`, and `cache: enabled`.",
        "- **Regenerate / verify**:",
        "  ```bash",
        "  python3 -B scripts/ci/ppu/ppu_recipe.py render-matrix --output test/ppu/README.md",
        "  python3 -B scripts/ci/ppu/ppu_recipe.py render-matrix --check",
        "  python3 -B scripts/ci/ppu/ppu_recipe.py validate --all --sglang-parser skip",
        "  python3 -B -m unittest discover -s scripts/ci/ppu -p 'test_recipe*.py'",
        "  ```",
        "",
        "## PR CI (`fast_sanity`)",
        "",
    ]
    if fast_rows:
        lines.extend(
            _render_md_table(
                [
                    "Policy",
                    "Recipe",
                    "Hardware / Cards",
                    "Preset",
                    "Features",
                    "Weights / Layers",
                    "Effective serve flags",
                ],
                fast_rows,
            )
        )
    else:
        lines.append("_None._")

    lines.extend(["", "## Nightly CI (`model_sanity`)", ""])
    if nightly_rows:
        lines.extend(
            _render_md_table(
                [
                    "Policy",
                    "Recipe",
                    "Hardware / Cards",
                    "Preset",
                    "Features",
                    "Stages",
                    "Golden",
                    "Effective serve flags",
                ],
                nightly_rows,
            )
        )
    else:
        lines.append("_None._")

    if fast_only:
        fast_only_list = ", ".join(_md_code(key) for key in fast_only)
        lines.extend(
            [
                "",
                f"> Policies declaring only `fast_sanity` (no `model_sanity` tier): {fast_only_list}.",
            ]
        )

    lines.extend(
        [
            "",
            "## Registered Serving Presets (Benchmark & Disagg Catalog)",
            "",
            "Every compatible `variant x preset` expansion below is statically validated by",
            "`ppu_recipe.py validate --all` (and real `sglang.srt.server_args.ServerArgs` argparse",
            "in PPU container runtime checks with topology placeholders substituted). Policy- and",
            "runner-injected switches such as `--disable-radix-cache`, `--load-format dummy`, and",
            "`--enable-metrics` are intentionally omitted from serving recipes.",
            "",
        ]
    )
    if catalog_rows:
        lines.extend(
            _render_md_table(
                [
                    "Recipe",
                    "Variant (Precision)",
                    "Preset",
                    "Hardware / Cards / Topology",
                    "Gate Tier",
                    "Effective `sglang.launch_server` args",
                ],
                catalog_rows,
            )
        )
    else:
        lines.append("_None._")

    lines.extend(
        [
            "",
            "## Adding or Changing a Model",
            "",
            f"1. Follow [AUTHORING.md]({authoring_rel}) to add or edit the serving recipe under `test/ppu/recipes/` (`ppu-serving/v1`).",
            "2. Add or edit the policy under `test/ppu/policy/models/` (`ppu-test-policy/v1`) and, for `model_sanity`, add `test/ppu/golden/accuracy_<golden_model_key>_890p.json`.",
            "3. Regenerate this document with `python3 -B scripts/ci/ppu/ppu_recipe.py render-matrix --output test/ppu/README.md`.",
            "4. Verify with `python3 -B scripts/ci/ppu/ppu_recipe.py validate --all --sglang-parser skip`, `python3 -B scripts/ci/ppu/ppu_recipe.py render-matrix --check`, and `python3 -B -m unittest discover -s scripts/ci/ppu -p 'test_recipe*.py'`.",
        ]
    )

    return "\n".join(lines) + "\n"


@contextlib.contextmanager
def _clean_diff_env() -> Iterator[None]:
    """Strip host cache/framework variables so diff resolution is deterministic."""
    prefixes = (
        "HF_",
        "HUGGINGFACE_",
        "TRANSFORMERS_",
        "PPU_",
        "VLLM_",
        "SGLANG_",
        "NCCL_",
        "MC_",
        "MOONCAKE_",
    )
    saved = {k: os.environ.pop(k) for k in list(os.environ) if k.startswith(prefixes)}
    try:
        yield
    finally:
        os.environ.update(saved)


def _run_git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    root = repo_root.resolve()
    return subprocess.run(
        ["git", "-c", f"safe.directory={root}", "-C", str(root), *args],
        capture_output=True,
        check=False,
    )


def _resolve_git_merge_base(
    repo_root: Path,
    base_ref: str,
    head_ref: str | None = None,
    *,
    use_merge_base: bool = True,
) -> str:
    head_target = head_ref or "HEAD"
    if use_merge_base:
        mb = _run_git(repo_root, "merge-base", base_ref, head_target)
        if mb.returncode == 0 and mb.stdout.strip():
            return os.fsdecode(mb.stdout.strip())
    rev = _run_git(repo_root, "rev-parse", "--verify", f"{base_ref}^{{tree}}")
    if rev.returncode != 0:
        err = os.fsdecode(rev.stderr).strip()
        raise ValidationError(f"git ref {base_ref!r} not found: {err}")
    return base_ref


def _materialize_git_tree_snapshot(repo_root: Path, ref: str, dest_root: Path) -> None:
    verify = _run_git(repo_root, "rev-parse", "--verify", f"{ref}^{{tree}}")
    if verify.returncode != 0:
        err = os.fsdecode(verify.stderr).strip()
        raise ValidationError(f"git ref {ref!r} not found: {err}")
    ls = _run_git(
        repo_root,
        "ls-tree",
        "-r",
        "-z",
        "--name-only",
        ref,
        "--",
        _RECIPES_REL.as_posix(),
        _POLICY_REL.as_posix(),
    )
    if ls.returncode != 0:
        err = os.fsdecode(ls.stderr).strip()
        raise ValidationError(f"failed to list recipe/policy tree at {ref!r}: {err}")
    rel_paths = [os.fsdecode(p) for p in ls.stdout.split(b"\0") if p]
    if not rel_paths:
        return
    for rel in rel_paths:
        show = _run_git(repo_root, "show", f"{ref}:{rel}")
        if show.returncode != 0:
            err = os.fsdecode(show.stderr).strip()
            raise ValidationError(f"failed to read {ref}:{rel}: {err}")
        out_path = dest_root / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(show.stdout)


def _snapshot_yaml_map(snapshot_root: Path) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    for rel_dir in (_RECIPES_REL, _POLICY_REL):
        root = snapshot_root / rel_dir
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if (
                path.is_file()
                and path.suffix in {".yaml", ".yml"}
                and not path.name.startswith(".")
            ):
                rel = path.relative_to(snapshot_root).as_posix()
                files[rel] = path.read_bytes()
    return files


def _collect_resolved_diff_entries(
    snapshot_root: Path, *, strict: bool
) -> dict[tuple[str, str], dict[str, Any]]:
    entries: dict[tuple[str, str], dict[str, Any]] = {}
    gate_tiers_by_key: dict[tuple[str, str, str, str | None], set[str]] = {}
    recipes_dir = _recipes_root(snapshot_root)
    policy_models_dir = _policy_root(snapshot_root) / "models"

    with _clean_diff_env():
        if policy_models_dir.is_dir():
            for pol_path in sorted(
                p
                for p in policy_models_dir.glob("*.yaml")
                if not p.name.startswith(".")
            ):
                pol_rel = pol_path.relative_to(snapshot_root).as_posix()
                slug = pol_path.stem
                try:
                    pol_doc = _load_yaml(pol_path)
                    if strict:
                        validate_policy_structure(
                            pol_doc, context=pol_rel, base_dir=snapshot_root
                        )
                    rec_rel = str(pol_doc["recipe"])
                    rec_path = snapshot_root / rec_rel
                    rec_doc = _load_yaml(rec_path)
                    if rec_doc.get("ppu", {}).get("framework") != "sglang":
                        continue
                    tiers_blk = pol_doc.get("tiers") or {}
                    for tier_name in sorted(tiers_blk):
                        if tier_name not in {"fast_sanity", "model_sanity"}:
                            continue
                        resolved = _resolve_recipe_from_doc(
                            rec_doc,
                            recipe_label=rec_rel,
                            policy=pol_doc,
                            policy_label=pol_rel,
                            tier=tier_name,
                        )
                        gate_key = (
                            rec_rel,
                            resolved["variant"],
                            resolved["hardware"],
                            resolved["preset"],
                        )
                        gate_tiers_by_key.setdefault(gate_key, set()).add(tier_name)
                        env = dict(resolved.get("server_env", {}))
                        entries[("policy", f"{slug}:{tier_name}")] = {
                            "label": f"policy:{slug}",
                            "tier": tier_name,
                            "argv": list(resolved["argv"]),
                            "env": dict(sorted(env.items())),
                        }
                except ValidationError as exc:
                    if strict:
                        raise
                    entries[("policy", f"{slug}:error")] = {
                        "label": f"policy:{slug}",
                        "tier": "invalid",
                        "argv": [f"<unresolved: {_redact_error_text(str(exc))}>"],
                        "env": {},
                    }

        if recipes_dir.is_dir():
            for rec_path in sorted(
                (p for p in recipes_dir.rglob("*.yaml") if not p.name.startswith(".")),
                key=lambda p: p.relative_to(snapshot_root).as_posix(),
            ):
                rec_rel = rec_path.relative_to(snapshot_root).as_posix()
                try:
                    rec_doc = _load_yaml(rec_path)
                    if (
                        rec_doc.get("schema_version") != SERVING_SCHEMA
                        or rec_doc.get("ppu", {}).get("framework") != "sglang"
                    ):
                        continue
                    for item in iter_recipe_catalog_entries(rec_path, rec_doc):
                        resolved = item["resolved"]
                        gate_key = (
                            rec_rel,
                            item["variant"],
                            item["hardware"],
                            item["preset"],
                        )
                        tiers = sorted(gate_tiers_by_key.get(gate_key, set()))
                        tier_str = ",".join(tiers) if tiers else "catalog"
                        preset_name = item["preset"] or "baseline"
                        target_id = f"{rec_rel} ({item['variant']}, {item['hardware']}, {preset_name})"
                        env = dict(resolved.get("server_env", {}))
                        entries[("recipe", target_id)] = {
                            "label": target_id,
                            "tier": tier_str,
                            "argv": list(resolved["argv"]),
                            "env": dict(sorted(env.items())),
                        }
                except ValidationError as exc:
                    if strict:
                        raise
                    entries[("recipe", f"{rec_rel} (error)")] = {
                        "label": rec_rel,
                        "tier": "invalid",
                        "argv": [f"<unresolved: {_redact_error_text(str(exc))}>"],
                        "env": {},
                    }
    return entries


def _group_argv_units(argv: list[str]) -> list[str]:
    units: list[str] = []
    idx = 0
    while idx < len(argv):
        tok = argv[idx]
        if (
            tok.startswith("-")
            and "=" not in tok
            and idx + 1 < len(argv)
            and not argv[idx + 1].startswith("-")
        ):
            units.append(shlex.join([tok, argv[idx + 1]]))
            idx += 2
        else:
            units.append(shlex.quote(tok))
            idx += 1
    return units


def _format_argv_diff(base_argv: list[str] | None, head_argv: list[str] | None) -> str:
    if base_argv is None and head_argv is None:
        return "—"
    if base_argv is None:
        assert head_argv is not None
        return _md_code("+ " + shlex.join(head_argv))
    if head_argv is None:
        return _md_code("- " + shlex.join(base_argv))
    if base_argv == head_argv:
        return "unchanged"
    base_units = _group_argv_units(base_argv)
    head_units = _group_argv_units(head_argv)
    matcher = difflib.SequenceMatcher(a=base_units, b=head_units)
    delta_parts: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in {"replace", "delete"}:
            delta_parts.append("- " + " ".join(base_units[i1:i2]))
        if tag in {"replace", "insert"}:
            delta_parts.append("+ " + " ".join(head_units[j1:j2]))
    delta_parts.append("head: " + shlex.join(head_argv))
    return _md_cell("\n".join(_md_code(part) for part in delta_parts))


def _format_env_diff(
    base_env: dict[str, str] | None, head_env: dict[str, str] | None
) -> str:
    b_env = base_env or {}
    h_env = head_env or {}
    if b_env == h_env:
        return "—"
    parts: list[str] = []
    for key in sorted(set(b_env) | set(h_env)):
        if key not in h_env:
            parts.append(_md_code(f"- {key}={b_env[key]}"))
        elif key not in b_env:
            parts.append(_md_code(f"+ {key}={h_env[key]}"))
        elif b_env[key] != h_env[key]:
            parts.append(_md_code(f"- {key}={b_env[key]}"))
            parts.append(_md_code(f"+ {key}={h_env[key]}"))
    return _md_cell("\n".join(parts)) if parts else "—"


def render_resolved_diff_markdown(
    *,
    base_ref: str | None = None,
    head_ref: str | None = None,
    base_root: Path | None = None,
    head_root: Path | None = None,
    repo_root: Path | None = None,
    use_merge_base: bool = True,
    only_when_changed: bool = False,
    include_signoff_hint: bool = False,
) -> str:
    """Render a Markdown table of ``base -> head`` resolved ``argv``, ``env``, and ``tier`` changes."""
    if (base_ref is None) == (base_root is None):
        raise ValidationError("specify exactly one of base_ref or base_root")
    if head_ref is not None and head_root is not None:
        raise ValidationError("specify at most one of head_ref or head_root")

    active_repo = (repo_root or REPO_ROOT).resolve()
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        if base_ref is not None:
            resolved_base_ref = _resolve_git_merge_base(
                active_repo, base_ref, head_ref, use_merge_base=use_merge_base
            )
            eff_base_root = tmp_path / "base"
            eff_base_root.mkdir()
            _materialize_git_tree_snapshot(
                active_repo, resolved_base_ref, eff_base_root
            )
        else:
            assert base_root is not None
            eff_base_root = base_root.resolve()

        if head_ref is not None:
            eff_head_root = tmp_path / "head"
            eff_head_root.mkdir()
            _materialize_git_tree_snapshot(active_repo, head_ref, eff_head_root)
        elif head_root is not None:
            eff_head_root = head_root.resolve()
        else:
            eff_head_root = active_repo

        base_files = _snapshot_yaml_map(eff_base_root)
        head_files = _snapshot_yaml_map(eff_head_root)
        changed_files = sorted(
            path
            for path in (set(base_files) | set(head_files))
            if base_files.get(path) != head_files.get(path)
        )
        if only_when_changed and not changed_files:
            return ""

        base_entries = _collect_resolved_diff_entries(eff_base_root, strict=False)
        head_entries = _collect_resolved_diff_entries(eff_head_root, strict=True)

    rows: list[list[str]] = []
    for key in sorted(set(base_entries) | set(head_entries)):
        b_item = base_entries.get(key)
        h_item = head_entries.get(key)
        if (
            b_item is not None
            and h_item is not None
            and b_item["tier"] == h_item["tier"]
            and b_item["argv"] == h_item["argv"]
            and b_item["env"] == h_item["env"]
        ):
            continue
        label = (h_item or b_item)["label"]  # type: ignore[index]
        b_tier = b_item["tier"] if b_item is not None else "—"
        h_tier = h_item["tier"] if h_item is not None else "—"
        tier_cell = (
            _md_code(h_tier)
            if b_tier == h_tier
            else f"{_md_code(b_tier)} -> {_md_code(h_tier)}"
        )
        rows.append(
            [
                _md_code(label),
                tier_cell,
                _format_argv_diff(
                    b_item["argv"] if b_item is not None else None,
                    h_item["argv"] if h_item is not None else None,
                ),
                _format_env_diff(
                    b_item["env"] if b_item is not None else None,
                    h_item["env"] if h_item is not None else None,
                ),
            ]
        )

    lines = ["## Resolved Command Diff (base -> head)", ""]
    if changed_files:
        files_summary = ", ".join(_md_code(p) for p in changed_files)
        lines.append(f"Changed recipe/policy files: {files_summary}")
        lines.append("")
    if rows:
        lines.extend(
            _render_md_table(
                ["Target", "Tier (base -> head)", "Argv Changes", "Env Changes"],
                rows,
            )
        )
    else:
        lines.append(
            f"_No resolved argv, env, or tier changes across {len(head_entries)} expanded target(s)._"
        )

    if include_signoff_hint:
        lines.extend(
            [
                "",
                "### Model Recipe Sign-off Checklist",
                "",
                "- [ ] **Preserved capabilities & intentional command diff**: Required serving capabilities (topology/parallelism, quantization/KV dtype, tool-call & reasoning parsers, speculative decoding) are preserved, and every `base -> head` change above is intentional.",
                "- [ ] **Behavioral assertion beyond HTTP 200**: Verified non-empty generation (`content` or `reasoning_content`), complete SSE `[DONE]` framing, and/or bounded GSM8K accuracy.",
                "- [ ] **Legacy deduplication & verification links**: Updated `replaces_legacy_cases`, ran `ppu_recipe.py validate --all` + `render-matrix --check`, and linked verification evidence.",
            ]
        )

    output = "\n".join(lines) + "\n"
    findings = _scan_text(output, context="resolved_command_diff", allow_loopback=True)
    if findings:
        raise ValidationError(
            "redaction gate blocked diff output: " + "; ".join(findings)
        )
    return output


def _parse_layer_override_cli(raw: str) -> int | None:
    lowered = raw.strip().lower()
    if lowered in {"none", "null", "full"}:
        return None
    try:
        parsed = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "layer-override must be a positive int or 'full'"
        ) from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("layer-override must be >= 1")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resolve and validate ppu-serving/v1")
    sub = parser.add_subparsers(dest="command", required=True)

    res = sub.add_parser("resolve", help="Resolve a recipe (+ optional policy/tier)")
    res.add_argument("--recipe", type=Path, required=True)
    res.add_argument("--policy", type=Path, default=None)
    res.add_argument("--tier", choices=sorted(POLICY_TIERS), default=None)
    res.add_argument("--variant", default=None)
    res.add_argument("--hardware", default=None)
    res.add_argument("--preset", default=None)
    res.add_argument("--cache", choices=["enabled", "disabled"], default=None)
    res.add_argument("--dummy-weights", action="store_true")
    res.add_argument("--layer-override", type=_parse_layer_override_cli, default=_UNSET)
    res.add_argument("--assets-map", type=Path, default=None)
    res.add_argument("--cache-revision", default=None)
    res.add_argument("--host", default="127.0.0.1")
    res.add_argument("--port", type=int, default=9985)
    res.add_argument("--served-model-name", default=None)

    val = sub.add_parser("validate", help="Validate recipe, dataset, and policy YAMLs")
    val.add_argument("paths", nargs="*", type=Path)
    val.add_argument(
        "--all", action="store_true", help="Validate all test/ppu YAML files"
    )
    val.add_argument(
        "--sglang-parser",
        choices=["auto", "required", "skip"],
        default="auto",
    )
    val.add_argument("--require-sglang-parser", action="store_true")

    mat = sub.add_parser(
        "render-matrix",
        help="Render or check the PR/Nightly CI recipe matrix Markdown doc",
    )
    mat.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write rendered Markdown to this path (default: stdout, or test/ppu/README.md with --check)",
    )
    mat.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if the target Markdown doc is missing or out of date",
    )

    diff_cmd = sub.add_parser(
        "diff",
        help="Render a Markdown diff of resolved argv, env, and tier between base and head",
    )
    diff_cmd.add_argument("--base-ref", default=None)
    diff_cmd.add_argument("--head-ref", default=None)
    diff_cmd.add_argument("--base-root", type=Path, default=None)
    diff_cmd.add_argument("--head-root", type=Path, default=None)
    diff_cmd.add_argument("--no-merge-base", action="store_true")
    diff_cmd.add_argument("--only-when-changed", action="store_true")
    diff_cmd.add_argument("--include-signoff-hint", action="store_true")
    diff_cmd.add_argument("--output", type=Path, default=None)

    args = parser.parse_args(argv)
    try:
        if args.command == "resolve":
            result = resolve_recipe(
                args.recipe,
                policy_path=args.policy,
                tier=args.tier,
                variant=args.variant,
                hardware=args.hardware,
                preset=args.preset,
                cache=args.cache,
                dummy_weights=args.dummy_weights,
                layer_override=args.layer_override,
                assets_map_path=args.assets_map,
                cache_revision=args.cache_revision,
                host=args.host,
                port=args.port,
                served_model_name=args.served_model_name,
            )
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0

        if args.command == "diff":
            rendered = render_resolved_diff_markdown(
                base_ref=args.base_ref,
                head_ref=args.head_ref,
                base_root=args.base_root,
                head_root=args.head_root,
                use_merge_base=not args.no_merge_base,
                only_when_changed=args.only_when_changed,
                include_signoff_hint=args.include_signoff_hint,
            )
            if args.output is None:
                print(rendered, end="")
            else:
                args.output.write_text(rendered, encoding="utf-8")
            return 0

        if args.command == "render-matrix":
            target = args.output
            expected = render_ci_matrix_markdown()
            if args.check:
                check_path = target if target is not None else DEFAULT_MATRIX_DOC
                if not check_path.is_file():
                    print(
                        f"matrix doc missing: {check_path}. Regenerate with:\n"
                        "  python3 -B scripts/ci/ppu/ppu_recipe.py render-matrix "
                        "--output test/ppu/README.md",
                        file=sys.stderr,
                    )
                    return 1
                actual = check_path.read_text(encoding="utf-8")
                if actual != expected:
                    print(
                        f"matrix doc out of date: {check_path}. Regenerate with:\n"
                        "  python3 -B scripts/ci/ppu/ppu_recipe.py render-matrix "
                        "--output test/ppu/README.md",
                        file=sys.stderr,
                    )
                    return 1
                return 0
            if target is None:
                print(expected, end="")
            else:
                target.write_text(expected, encoding="utf-8")
            return 0

        target_paths = list(args.paths)
        check_stray = bool(args.all or not target_paths)
        if check_stray:
            target_paths = _discover_default_paths()
        parser_mode = "required" if args.require_sglang_parser else args.sglang_parser
        report = validate_paths(
            target_paths,
            sglang_parser_mode=parser_mode,
            check_repo_stray_recipes=check_stray,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    except ValidationError as exc:
        print(f"validation error: {_redact_error_text(str(exc))}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
