#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm-project/vllm-sail@0163924 (Apache-2.0).
"""Thin recipe/policy resolver and validator for ``ppu-serving/v1``."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RECIPES_ROOT = REPO_ROOT / "test" / "ppu" / "recipes"
DEFAULT_POLICY_ROOT = REPO_ROOT / "test" / "ppu" / "policy"

SERVING_SCHEMA = "ppu-serving/v1"
DATASET_SCHEMA = "ppu-dataset/v1"
POLICY_SCHEMA = "ppu-test-policy/v1"
POLICY_TIERS = frozenset({"fast_sanity", "model_sanity", "nightly", "release"})
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
INVERSE_BOOLEAN_GROUPS: tuple[tuple[str, str], ...] = (
    ("--trust-remote-code", "--no-trust-remote-code"),
)
ALLOWED_PRESET_SLOTS = frozenset(
    {
        "decode_nographs",
        "dp_ep_low_latency",
        "eval_longctx",
        "prefill_long",
        "tp8",
    }
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
            r"(?:^|[\s'\"=])(?:/nas_aisw|/ppusw|/mnt/|/Users/|/home/|/root/|/data/)"
        ),
    ),
    (
        "credential",
        re.compile(
            r"\b(?:ghp|gho|ghu|ghs)_[A-Za-z0-9_]{8,}|github_pat_|AKIA[0-9A-Z]{8,}|QODER_PAT\b"
        ),
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


def _inverse_map() -> dict[str, str]:
    pairs: dict[str, str] = {}
    for pos, neg in INVERSE_BOOLEAN_GROUPS:
        pairs[pos] = neg
        pairs[neg] = pos
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
            entries.append((flag, val, True))
            idx += 1
        elif idx + 1 < len(tokens) and not str(tokens[idx + 1]).startswith("--"):
            entries.append((tok, str(tokens[idx + 1]), False))
            idx += 2
        else:
            entries.append((tok, None, False))
            idx += 1
    return entries


def _apply_flag_entries(
    ordered: list[tuple[str, str | None, bool]],
    new_tokens: list[str],
) -> list[tuple[str, str | None, bool]]:
    """Append flags with last-wins deduplication and inverse-flag clearing."""
    result = list(ordered)
    for flag, val, has_eq in _parse_flag_entries(new_tokens):
        inverse = _INVERSE_FLAGS.get(flag)
        result = [
            item
            for item in result
            if item[0] != flag and (inverse is None or item[0] != inverse)
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


def _flag_value(ordered: list[tuple[str, str | None, bool]], flag: str) -> str | None:
    for item_flag, val, _ in reversed(ordered):
        if item_flag == flag:
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


def resolve_recipe(
    recipe_path: Path,
    *,
    policy_path: Path | None = None,
    tier: str | None = None,
    variant: str = "default",
    hardware: str = "zhenwu_m890",
    preset: str | None = None,
    cache: str | None = None,
    dummy_weights: bool = False,
    layer_override: Any = _UNSET,
    assets_map_path: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 9985,
    served_model_name: str | None = None,
) -> dict[str, Any]:
    """Resolve recipe + optional policy tier into effective SGLang argv, env, and digest."""
    recipe = _load_yaml(recipe_path)
    validate_serving_recipe_structure(recipe, context=str(recipe_path))

    ppu_blk = recipe["ppu"]
    if ppu_blk["framework"] != "sglang":
        raise ValidationError(
            f"{recipe_path}: executor only resolves framework='sglang', got {ppu_blk['framework']!r}"
        )

    policy: dict[str, Any] | None = None
    if policy_path is not None:
        policy = _load_yaml(policy_path)
        validate_policy_structure(policy, context=str(policy_path), base_dir=REPO_ROOT)

    eff_cache = cache
    eff_dummy = dummy_weights
    eff_layers: int | None = None if layer_override is _UNSET else layer_override
    eff_preset = preset

    if policy is not None:
        if tier is None:
            raise ValidationError("--tier is required when --policy is supplied")
        tiers = policy.get("tiers", {})
        if tier not in tiers:
            raise ValidationError(f"{policy_path}: tier {tier!r} not found")
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
        if policy.get("hardware") and hardware == "zhenwu_m890":
            hardware = policy["hardware"]

    ordered: list[tuple[str, str | None, bool]] = []
    server_env: dict[str, str] = {}

    # 1. base_args / base_env
    model_blk = recipe["model"]
    ordered = _apply_flag_entries(ordered, list(model_blk.get("base_args", [])))
    server_env.update(model_blk.get("base_env", {}))

    # 2. variants.<variant>
    variants_blk = recipe.get("variants", {})
    if variant not in variants_blk:
        raise ValidationError(f"{recipe_path}: unknown variant {variant!r}")
    var_blk = variants_blk[variant]
    ordered = _apply_flag_entries(ordered, list(var_blk.get("extra_args", [])))
    server_env.update(var_blk.get("extra_env", {}))

    # 3. hardware_overrides.<hw>
    hw_blk = recipe.get("hardware_overrides", {}).get(hardware, {})
    ordered = _apply_flag_entries(ordered, list(hw_blk.get("extra_args", [])))
    server_env.update(hw_blk.get("extra_env", {}))

    # 4. enabled features (ppu.default_features overridden by preset.select.features)
    presets_blk = recipe.get("presets", {})
    preset_blk: dict[str, Any] = {}
    if eff_preset is not None:
        if eff_preset not in presets_blk:
            raise ValidationError(f"{recipe_path}: unknown preset {eff_preset!r}")
        preset_blk = presets_blk[eff_preset]

    enabled_features = {name: True for name in ppu_blk.get("default_features", [])}
    select_features = preset_blk.get("select", {}).get("features", {})
    for feat_name, flag_on in select_features.items():
        enabled_features[feat_name] = bool(flag_on)

    features_blk = recipe.get("features", {})
    for feat_name, is_enabled in enabled_features.items():
        if not is_enabled:
            continue
        if feat_name not in features_blk:
            raise ValidationError(f"{recipe_path}: unknown feature {feat_name!r}")
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

    # 6. policy switches (prefix cache, dummy load format, layer override)
    if eff_cache == "enabled":
        ordered = _remove_flag_entries(ordered, ["--disable-radix-cache"])
    elif eff_cache == "disabled":
        ordered = _apply_flag_entries(ordered, ["--disable-radix-cache"])
    elif eff_cache is not None:
        raise ValidationError(f"unsupported cache value: {eff_cache!r}")

    ordered = _apply_flag_entries(ordered, ["--enable-metrics"])
    if eff_dummy:
        ordered = _apply_flag_entries(ordered, ["--load-format", "dummy"])
    if eff_layers is not None:
        ordered = _merge_json_flag(
            ordered,
            "--json-model-override-args",
            {"num_hidden_layers": int(eff_layers)},
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
    asset_name = ppu_blk["assets"]["model"]
    hf_model_id = ppu_blk.get("hf_model_id") or model_blk["model_id"]
    resolved_model = assets_map.get(asset_name, hf_model_id)
    resolved_tokenizer = assets_map.get(ppu_blk["assets"]["tokenizer"], resolved_model)
    effective_served_name = (
        served_model_name or ppu_blk.get("served_name") or asset_name
    )

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

    dp_raw = _flag_value(ordered, "--dp-size")
    tp_raw = _flag_value(ordered, "--tp-size")
    if dp_raw is not None:
        ppu_count = int(dp_raw) * (int(tp_raw) if tp_raw is not None else 1)
    elif tp_raw is not None:
        ppu_count = int(tp_raw)
    else:
        ppu_count = int(ppu_blk["topology"]["devices_per_node"])

    digest_payload = {
        "framework": ppu_blk["framework"],
        "model_id": model_blk["model_id"],
        "hf_model_id": hf_model_id,
        "variant": variant,
        "hardware": hardware,
        "preset": eff_preset,
        "tier": tier,
        "cache": eff_cache,
        "dummy_weights": eff_dummy,
        "layer_override": eff_layers,
        "flags": _canonical_flags(ordered),
        "server_env": dict(sorted(server_env.items())),
    }
    config_digest = hashlib.sha256(
        json.dumps(digest_payload, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )
    ).hexdigest()

    return {
        "argv": argv,
        "flags": flags_only,
        "server_env": server_env,
        "ppu_count": ppu_count,
        "model_id": model_blk["model_id"],
        "hf_model_id": hf_model_id,
        "resolved_model": resolved_model,
        "resolved_tokenizer": resolved_tokenizer,
        "golden_model_key": ppu_blk.get("golden_model_key", asset_name),
        "served_name": effective_served_name,
        "config_digest": config_digest,
        "tier": tier,
        "preset": eff_preset,
        "variant": variant,
        "hardware": hardware,
        "layer_override": eff_layers,
        "dummy_weights": eff_dummy,
    }


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
    for name in checked:
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


def _validate_flag_list(tokens: Any, *, context: str) -> None:
    raw = _check_str_list(tokens, context=context)
    managed = {
        "--model-path",
        "--model",
        "--tokenizer-path",
        "--served-model-name",
        "--host",
        "--port",
        "--disable-radix-cache",
        "--load-format",
        "--enable-metrics",
        "--config",
    }
    for flag, val, _ in _parse_flag_entries(raw):
        if flag in managed:
            raise ValidationError(f"{context}: framework runner owns {flag}")
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


def validate_serving_recipe_structure(
    recipe: dict[str, Any],
    *,
    context: str,
    allow_internal_visibility: bool = False,
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

    topo = _check_keys(
        ppu["topology"],
        allowed={"kind", "nodes", "devices_per_node"},
        required={"kind", "nodes", "devices_per_node"},
        context=f"{context}.ppu.topology",
    )
    if topo["kind"] not in {"single_node", "multi_node_ep", "pd_disagg"}:
        raise ValidationError(f"{context}.ppu.topology.kind invalid: {topo['kind']!r}")
    if not isinstance(topo["nodes"], int) or topo["nodes"] <= 0:
        raise ValidationError(f"{context}.ppu.topology.nodes must be positive int")
    if not isinstance(topo["devices_per_node"], int) or topo["devices_per_node"] <= 0:
        raise ValidationError(
            f"{context}.ppu.topology.devices_per_node must be positive int"
        )

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
    if not isinstance(model["model_id"], str) or not model["model_id"]:
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

    for vname, vcfg in recipe.get("variants", {}).items():
        _check_keys(
            vcfg,
            allowed={"precision", "extra_args", "extra_env"},
            required={"precision"},
            context=f"{context}.variants.{vname}",
        )
        _validate_flag_list(
            vcfg.get("extra_args", []), context=f"{context}.variants.{vname}.extra_args"
        )
        _validate_env_block(
            vcfg.get("extra_env", {}),
            external_env=external_env,
            context=f"{context}.variants.{vname}.extra_env",
        )

    for hname, hcfg in recipe.get("hardware_overrides", {}).items():
        if hname not in {"zhenwu_m890", "zhenwu_810e"}:
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

    for pname, pcfg in recipe.get("presets", {}).items():
        if pname not in ALLOWED_PRESET_SLOTS:
            raise ValidationError(f"{context}.presets: unknown preset slot {pname!r}")
        _check_keys(
            pcfg,
            allowed={
                "hardware",
                "tp",
                "select",
                "args",
                "env",
                "remove_args",
                "json_args",
            },
            context=f"{context}.presets.{pname}",
        )
        if "tp" in pcfg and (not isinstance(pcfg["tp"], int) or pcfg["tp"] <= 0):
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
        _validate_flag_list(
            pcfg.get("args", []), context=f"{context}.presets.{pname}.args"
        )
        _validate_env_block(
            pcfg.get("env", {}),
            external_env=external_env,
            context=f"{context}.presets.{pname}.env",
        )

    for loc, s in _walk_strings(recipe, context):
        findings = _scan_text(s, context=loc, allow_loopback=False)
        if findings:
            raise ValidationError("; ".join(findings))


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
    if "tiers" not in policy and "stages" not in policy:
        raise ValidationError(f"{context}: policy must define 'tiers' or 'stages'")

    recipe_path: Path | None = None
    if "recipe" in policy:
        rel = Path(policy["recipe"])
        recipe_path = rel if rel.is_absolute() else (base_dir / rel)
        if (
            not recipe_path.resolve().is_relative_to(DEFAULT_RECIPES_ROOT.resolve())
            or not recipe_path.is_file()
        ):
            raise ValidationError(
                f"{context}: referenced recipe not found: {policy['recipe']}"
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
            if (
                fs["dummy_weights"] is not True
                or (
                    "layer_override" in fs
                    and (
                        type(fs["layer_override"]) is not int
                        or fs["layer_override"] <= 0
                    )
                )
                or (
                    "fallback_full_layers" in fs
                    and type(fs["fallback_full_layers"]) is not bool
                )
            ):
                raise ValidationError(
                    f"{context}: invalid fast sanity weight or layer policy"
                )
            if fs["cache"] != "enabled":
                raise ValidationError(
                    f"{context}.tiers.fast_sanity.cache must be 'enabled'"
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
                        hardware=policy["hardware"],
                        preset=stage.get("preset"),
                        cache=stage.get("cache"),
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


def _collect_all_recipe_flag_sets(
    recipe_path: Path, recipe: dict[str, Any]
) -> list[tuple[str, list[str]]]:
    sets: list[tuple[str, list[str]]] = []
    default_res = resolve_recipe(recipe_path, cache="enabled")
    sets.append((f"{recipe_path.name}:default", default_res["flags"]))
    sets.append(
        (
            f"{recipe_path.name}:fast_sanity_2l",
            resolve_recipe(
                recipe_path, cache="enabled", dummy_weights=True, layer_override=2
            )["flags"],
        )
    )
    for feat_name, feat_cfg in recipe.get("features", {}).items():
        if "modes" in feat_cfg:
            for mode_name, mode_cfg in feat_cfg["modes"].items():
                sets.append(
                    (
                        f"{recipe_path.name}:feature:{feat_name}.{mode_name}",
                        list(mode_cfg.get("args", [])),
                    )
                )
        else:
            sets.append(
                (
                    f"{recipe_path.name}:feature:{feat_name}",
                    list(feat_cfg.get("args", [])),
                )
            )
    for var_name in recipe.get("variants", {}):
        sets.append(
            (
                f"{recipe_path.name}:variant:{var_name}",
                resolve_recipe(recipe_path, variant=var_name)["flags"],
            )
        )
    for preset_name in recipe.get("presets", {}):
        sets.append(
            (
                f"{recipe_path.name}:preset:{preset_name}",
                resolve_recipe(recipe_path, preset=preset_name, cache="enabled")[
                    "flags"
                ],
            )
        )
    return sets


def validate_paths(
    paths: list[Path],
    *,
    sglang_parser_mode: str = "auto",
    parser: argparse.ArgumentParser | None = None,
    allow_internal_visibility: bool = False,
) -> dict[str, Any]:
    """Validate recipe, dataset, and policy YAML files."""
    if sglang_parser_mode not in {"auto", "required", "skip"}:
        raise ValidationError(f"invalid sglang_parser_mode: {sglang_parser_mode!r}")

    recipe_items: list[tuple[Path, dict[str, Any]]] = []
    validated: list[str] = []

    for path in paths:
        doc = _load_yaml(path)
        schema = doc.get("schema_version")
        if schema == SERVING_SCHEMA:
            validate_serving_recipe_structure(
                doc,
                context=str(path),
                allow_internal_visibility=allow_internal_visibility,
            )
            recipe_items.append((path, doc))
        elif schema == DATASET_SCHEMA:
            validate_dataset_structure(doc, context=str(path))
        elif schema == POLICY_SCHEMA:
            validate_policy_structure(doc, context=str(path), base_dir=REPO_ROOT)
        else:
            raise ValidationError(f"{path}: unsupported schema_version {schema!r}")
        validated.append(str(path))

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
            for rpath, rdoc in recipe_items:
                if rdoc["ppu"]["framework"] != "sglang":
                    continue
                for label, flag_set in _collect_all_recipe_flag_sets(rpath, rdoc):
                    validate_flags_with_parser(active_parser, flag_set, context=label)
                    checked_flag_sets += 1
            parser_checked = True

    return {
        "validated_files": validated,
        "parser_checked": parser_checked,
        "checked_flag_sets": checked_flag_sets,
    }


def _discover_default_paths() -> list[Path]:
    paths: list[Path] = []
    for root in (DEFAULT_RECIPES_ROOT, DEFAULT_POLICY_ROOT):
        if root.is_dir():
            paths.extend(
                sorted(p for p in root.rglob("*.yaml") if not p.name.startswith("."))
            )
    if not paths:
        raise ValidationError("no recipe or policy YAML files found under test/ppu")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resolve and validate ppu-serving/v1")
    sub = parser.add_subparsers(dest="command", required=True)

    res = sub.add_parser("resolve", help="Resolve a recipe into argv/env/digest")
    res.add_argument("--recipe", type=Path, required=True)
    res.add_argument("--policy", type=Path, default=None)
    res.add_argument("--tier", choices=["fast_sanity", "model_sanity"], default=None)
    res.add_argument("--variant", default="default")
    res.add_argument("--hardware", default="zhenwu_m890")
    res.add_argument("--preset", default=None)
    res.add_argument("--cache", choices=["enabled", "disabled"], default=None)
    res.add_argument("--dummy-weights", action="store_true")
    res.add_argument("--layer-override", type=int, default=argparse.SUPPRESS)
    res.add_argument("--assets-map", type=Path, default=None)
    res.add_argument("--host", default="127.0.0.1")
    res.add_argument("--port", type=int, default=9985)
    res.add_argument("--served-model-name", default=None)

    val = sub.add_parser("validate", help="Validate recipe/policy/dataset files")
    val.add_argument("paths", nargs="*", type=Path)
    val.add_argument("--all", action="store_true")
    val.add_argument(
        "--sglang-parser",
        choices=["auto", "required", "skip"],
        default="auto",
    )
    val.add_argument("--require-sglang-parser", action="store_true")

    args = parser.parse_args(argv)
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
            layer_override=getattr(args, "layer_override", _UNSET),
            assets_map_path=args.assets_map,
            host=args.host,
            port=args.port,
            served_model_name=args.served_model_name,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    target_paths = list(args.paths)
    if args.all or not target_paths:
        target_paths = _discover_default_paths()
    parser_mode = "required" if args.require_sglang_parser else args.sglang_parser
    report = validate_paths(target_paths, sglang_parser_mode=parser_mode)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as error:
        print(f"validation error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
