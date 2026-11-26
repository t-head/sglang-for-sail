# SPDX-License-Identifier: Apache-2.0
"""Load task-scoped Cluster Run configuration without emitting its values."""

import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

BACKEND_SHA = "96e25147971f5d2a04d828f8ad508cae38a4fa8a"
CONFIG_NAME = ".sglang-recipe-ci-v0523"
DEFAULT_MXFP4_REVISION_KEYS = {
    "glm-5.3": "PPU_CI_GLM53_MODEL_REVISION",
    "glm-5.3-tp4": "PPU_CI_GLM53_MODEL_REVISION",
    "deepseek-v4-flash": "PPU_CI_DSV4_FLASH_MODEL_REVISION",
    "deepseek-v4-flash-tp8": "PPU_CI_DSV4_FLASH_MODEL_REVISION",
}
SETTINGS = {
    "PPU_CI_K8S_STAGE_ROOT",
    "PPU_CI_K8S_PRIVATE_WORKER_ROOT",
    "PPU_CI_PRIVATE_RESULTS_ROOT",
    "PPU_CI_K8S_RUNTIME_DIGEST",
    "PPU_CI_RUNTIME_IMAGE",
    "PPU_CI_RECIPE_IMAGE_REPO",
    "PPU_CI_RECIPE_ALLOWED_IMAGE_REPOS",
    "PPU_CI_JIT_ROOT",
    "PPU_CI_HF_CACHE_ROOT",
    "PPU_CI_GLM53_MODEL_REVISION",
    "PPU_CI_DSV4_FLASH_MODEL_REVISION",
    "PPU_CI_GLM53_MODEL_PATH",
    "PPU_CI_DSV4_FLASH_MODEL_PATH",
    "PPU_CI_GSM8K_PATH",
    "PPU_CI_TARGET_MODEL_PATH",
    "PPU_CI_EAGLE3_MODEL_PATH",
    "PPU_CI_DFLASH_MODEL_PATH",
    "PPU_CI_EVAL_CACHE",
}


def staging_root(backend):
    # The pinned Action defines the shared staging mount. Read its contract
    # locally so the mount path never needs to appear in this repository.
    manifest = (backend / "cluster-run/action.yml").read_text(encoding="utf-8")
    (root,) = re.findall(
        r'^\s*STAGE_TARBALL="(/[A-Za-z0-9_./-]+)/\$\{GITHUB_RUN_ID\}-',
        manifest,
        re.MULTILINE,
    )
    return Path(root)


def load_settings(directory, expected):
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("invalid configuration digest")
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        if os.fstat(fd).st_mode & 0o007:
            raise ValueError("configuration directory is accessible to others")
        # Keep existing workflow runs pinned to their original configuration.
        try:
            handle = os.open(
                f"config.{expected}.json", os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd
            )
        except FileNotFoundError:
            handle = os.open("config.json", os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
        with os.fdopen(handle, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o007:
                raise ValueError("invalid configuration permissions")
            payload = stream.read(8193)
    finally:
        os.close(fd)
    if len(payload) > 8192 or hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError("configuration digest mismatch")
    document = json.loads(payload)
    values = document["settings"]
    if set(values) - SETTINGS or not all(isinstance(v, str) for v in values.values()):
        raise ValueError("unknown configuration key")
    return values


def resolve_retention_root(environ, values, fallback):
    env_root = (environ.get("PPU_CI_PRIVATE_RESULTS_ROOT") or "").strip()
    if env_root:
        return env_root, "env"
    config_root = (values.get("PPU_CI_PRIVATE_RESULTS_ROOT") or "").strip()
    if config_root:
        return config_root, "config"
    return str(fallback), "fallback"


def _parse_yaml_scalar_token(raw_val: str) -> str | None:
    s = raw_val.strip()
    if not s or s.startswith("#"):
        return None
    if s[0] in ("'", '"'):
        quote = s[0]
        out: list[str] = []
        i = 1
        while i < len(s):
            ch = s[i]
            if quote == '"' and ch == "\\" and i + 1 < len(s):
                out.append(s[i + 1])
                i += 2
                continue
            if quote == "'" and ch == "'" and i + 1 < len(s) and s[i + 1] == "'":
                out.append("'")
                i += 2
                continue
            if ch == quote:
                rest = s[i + 1 :].strip()
                if rest and not rest.startswith("#"):
                    return None
                return "".join(out).strip()
            out.append(ch)
            i += 1
        return None
    if s[0] in ("|", ">", "[", "{", "&", "*", "!"):
        return None
    s = re.split(r"\s+#", s, maxsplit=1)[0].strip()
    return s or None


def _stdlib_extract_policy_variant_and_recipe(
    raw_text: str,
) -> tuple[str | None, str | None]:
    body_lines = [
        line
        for line in raw_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#") and line.strip() != "---"
    ]
    joined = "\n".join(body_lines).strip()
    if joined.startswith("{"):
        try:
            parsed = json.loads(joined)
        except ValueError:
            if not joined.endswith("}"):
                return None, None
            inner = joined[1:-1]
            variant_val = "default"
            recipe_val: str | None = None
            for match in re.finditer(
                r'(?:^|,)\s*["\']?([A-Za-z0-9_-]+)["\']?\s*:\s*([^,{}]+)', inner
            ):
                k = match.group(1)
                v = _parse_scalar_token_or_none(match.group(2))
                if k == "variant":
                    if not v:
                        return None, None
                    variant_val = v
                elif k == "recipe":
                    recipe_val = v
            return variant_val, recipe_val
        if not isinstance(parsed, dict):
            return None, None
        raw_var = parsed.get("variant")
        if raw_var is None:
            var_name = "default"
        elif isinstance(raw_var, str) and raw_var.strip():
            var_name = raw_var.strip()
        else:
            return None, None
        rec = parsed.get("recipe")
        return var_name, rec if isinstance(rec, str) else None

    seen_variants: list[str] = []
    recipe_rel: str | None = None
    for raw_line in raw_text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#") or stripped in ("---", "..."):
            continue
        if raw_line[0].isspace():
            continue
        key, sep, rest = raw_line.partition(":")
        if not sep:
            continue
        k = key.strip()
        if k == "variant":
            val = _parse_yaml_scalar_token(rest)
            if not val:
                return None, None
            seen_variants.append(val)
        elif k == "recipe":
            recipe_rel = _parse_yaml_scalar_token(rest)
    if len(seen_variants) > 1:
        return None, None
    return (seen_variants[0] if seen_variants else "default"), recipe_rel


def _parse_scalar_token_or_none(raw_val: str) -> str | None:
    return _parse_yaml_scalar_token(raw_val)


def _stdlib_recipe_default_matches_top_hf_id(raw_text: str) -> bool:
    ppu_hf: str | None = None
    default_hf: str | None = None
    lvl0: str | None = None
    lvl1: str | None = None
    lvl1_indent: int | None = None
    for raw_line in raw_text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#") or stripped in ("---", "..."):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        key, sep, rest = stripped.partition(":")
        if not sep:
            continue
        k = key.strip()
        if indent == 0:
            lvl0 = k
            lvl1 = None
            lvl1_indent = None
            continue
        if lvl0 == "ppu" and k == "hf_model_id" and lvl1 is None:
            ppu_hf = _parse_yaml_scalar_token(rest)
        elif lvl0 == "variants":
            if lvl1_indent is None or indent <= lvl1_indent:
                lvl1 = k
                lvl1_indent = indent
                flow = rest.strip()
                if lvl1 == "default" and flow.startswith("{") and flow.endswith("}"):
                    m = re.search(
                        r'(?:^|[{,])\s*["\']?hf_model_id["\']?\s*:\s*([^,}]+)', flow
                    )
                    if m:
                        default_hf = _parse_yaml_scalar_token(m.group(1))
                        if default_hf is None:
                            return False
            elif lvl1 == "default" and k == "hf_model_id":
                default_hf = _parse_yaml_scalar_token(rest)
                if default_hf is None:
                    return False
    return default_hf is None or default_hf == ppu_hf


def resolve_pinned_model_revision_key(
    model_key: str, workspace: Path | str = Path(".")
) -> str | None:
    """Return the MXFP4 revision env key only when ``model_key`` uses ``default`` MXFP4."""
    revision_key = DEFAULT_MXFP4_REVISION_KEYS.get(model_key)
    if revision_key is None:
        return None
    root = Path(workspace)
    policy_path = root / "test/ppu/policy/models" / f"{model_key}.yaml"
    raw_policy = policy_path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore[import-not-found]
    except (ModuleNotFoundError, ImportError):
        variant_name, recipe_rel = _stdlib_extract_policy_variant_and_recipe(raw_policy)
        if variant_name != "default":
            return None
        if recipe_rel:
            recipe_path = root / recipe_rel
            if recipe_path.is_file() and not _stdlib_recipe_default_matches_top_hf_id(
                recipe_path.read_text(encoding="utf-8")
            ):
                return None
        return revision_key

    doc = yaml.safe_load(raw_policy) or {}
    if not isinstance(doc, dict):
        return None
    raw_variant = doc.get("variant")
    if raw_variant is None:
        variant_name = "default"
    elif isinstance(raw_variant, str) and raw_variant.strip():
        variant_name = raw_variant.strip()
    else:
        return None
    if variant_name != "default":
        return None
    recipe_rel = doc.get("recipe")
    if isinstance(recipe_rel, str) and recipe_rel.strip():
        recipe_path = root / recipe_rel
        if recipe_path.is_file():
            rdoc = yaml.safe_load(recipe_path.read_text(encoding="utf-8")) or {}
            if not isinstance(rdoc, dict):
                return None
            ppu_hf = (rdoc.get("ppu") or {}).get("hf_model_id")
            var_doc = (rdoc.get("variants") or {}).get("default") or {}
            if not isinstance(var_doc, dict):
                return None
            eff_hf = var_doc.get("hf_model_id") or ppu_hf
            if eff_hf != ppu_hf:
                return None
    return revision_key


def _verify_retention_root(retention_root_raw, retention_source, environ):
    from recipe_retention import (
        ensure_retention_base,
        probe_retention_writable,
        validate_retention_root,
    )

    status = {
        "external_retention_root": retention_source != "fallback",
        "retention_root_valid": False,
        "retention_source": retention_source,
        "retention_writable": False,
    }
    try:
        validated_root = validate_retention_root(retention_root_raw, environ)
        status["retention_root_valid"] = True
        retention_base = ensure_retention_base(validated_root)
        probe_retention_writable(retention_base)
        status["retention_writable"] = True
    except (KeyError, TypeError, ValueError, OSError):
        print(json.dumps({"private_retention_check": status}, sort_keys=True))
        raise
    return validated_root, status


def main():
    backend = Path(os.environ["GITHUB_WORKSPACE"]) / ".ppu-infra"
    head = subprocess.check_output(
        ["git", "-C", str(backend), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
    )
    if head.decode().strip() != BACKEND_SHA:
        raise ValueError("Cluster Run revision mismatch")
    directory = staging_root(backend) / CONFIG_NAME
    expected = os.environ["PPU_CI_K8S_CONFIG_SHA256"]
    values = load_settings(directory, expected)
    retained = directory / "output/retained"
    retained.mkdir(mode=0o700, exist_ok=True)
    info = retained.lstat()
    if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o700:
        raise ValueError("invalid private results directory")
    if info.st_uid != os.geteuid():
        raise ValueError("the runner does not own the private results directory")
    retention_root_raw, retention_source = resolve_retention_root(
        os.environ, values, retained
    )
    if sys.argv[1:] == ["--check"]:
        from recipe_cluster_diagnostics import publish_recent_failures

        validated_root, status = _verify_retention_root(
            retention_root_raw, retention_source, os.environ
        )
        print("PPU cluster private configuration verified")
        print(json.dumps({"private_retention_check": status}, sort_keys=True))
        publish_recent_failures(
            validated_root,
            int(os.environ["GITHUB_RUN_ID"]),
            fallback_root=retained.resolve(strict=True),
        )
        return
    if os.environ.get("TIER") != "legacy":
        _verify_retention_root(retention_root_raw, retention_source, os.environ)
    os.execvpe(
        "bash",
        ["bash", "scripts/ci/ppu/run_recipe_cluster.sh"],
        {
            **os.environ,
            **values,
            "PPU_CI_K8S_HOST_SCRATCH_ROOT": str(retained),
            "PPU_CI_PRIVATE_RESULTS_ROOT": retention_root_raw,
            "PPU_CI_K8S_LOADED_CONFIG_SHA256": expected,
        },
    )


if __name__ == "__main__":
    try:
        main()
    except (KeyError, TypeError, ValueError, OSError, subprocess.SubprocessError):
        print(
            "::error::PPU cluster private configuration is unavailable or invalid",
            file=sys.stderr,
        )
        raise SystemExit(2) from None
