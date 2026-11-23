import argparse
import hashlib
import importlib
import importlib.util
import json
import re
import shutil
import sys
from importlib import metadata
from pathlib import Path

BUILD_TARGETS = {"sglang", "sgl-kernel", "sglang-kernel"}
COMPAT_ADDITIONS = {"sailify", "blake3"}
PINNED_ADDITIONS = {
    "sailify": "1.0.0",
    "blake3": "1.0.0",
    "sgl-eval": "0.1.2",
    "math-verify": "0.9.0",
    "latex2sympy2-extended": "1.11.0",
    "antlr4-python3-runtime": "4.9.3",
    "editdistance": "0.8.1",
}


def normalize_name(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def snapshot():
    packages = {}
    for dist in metadata.distributions():
        name = normalize_name(dist.metadata["Name"])
        if name not in packages:
            packages[name] = {
                "version": dist.version,
                "record": hashlib.sha256(
                    (dist.read_text("RECORD") or "").encode()
                ).hexdigest(),
            }
    return packages


def validate_evaluators(packages):
    for name, version in PINNED_ADDITIONS.items():
        if name in COMPAT_ADDITIONS or name not in packages:
            continue
        installed = packages[name]["version"]
        if installed != version:
            raise ValueError(
                f"Preinstalled {name}=={installed} does not match required "
                f"{name}=={version}; refusing to replace it."
            )


def missing_wheels(wheel_dir, packages):
    if not wheel_dir.is_dir():
        raise ValueError(f"Wheel directory does not exist: {wheel_dir}")
    available = {}
    for path in wheel_dir.resolve().glob("*.whl"):
        match = re.fullmatch(
            r"([^-]+)-([^-]+)-(?:[0-9][^-]*-)?[^-]+-[^-]+-[^-]+\.whl", path.name
        )
        if path.is_file() and match:
            key = (normalize_name(match[1]), match[2])
            available.setdefault(key, []).append(path)
    selected = []
    for name, version in PINNED_ADDITIONS.items():
        if name in packages:
            continue
        matches = available.get((name, version), [])
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one wheel for {name}=={version} in {wheel_dir}; "
                f"found {len(matches)}."
            )
        selected.append(str(matches[0]))
    return selected


def find_triton_kernels_dir():
    spec = importlib.util.find_spec("triton_kernels")
    if spec is None or not spec.submodule_search_locations:
        return None
    for raw in spec.submodule_search_locations:
        candidate = Path(raw)
        if (
            candidate.is_dir()
            and not candidate.is_symlink()
            and any(candidate.rglob("*.py"))
        ):
            return candidate.resolve(strict=True)
    return None


def backup_triton_kernels(backup_dir):
    source = find_triton_kernels_dir()
    if source is None:
        return
    backup_dir.mkdir(parents=True, exist_ok=True)
    dest = backup_dir / "triton_kernels"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(
        source,
        dest,
        symlinks=False,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )
    (backup_dir / "meta.json").write_text(
        json.dumps({"target": str(source)}) + "\n", encoding="utf-8"
    )


def restore_triton_kernels(backup_dir):
    meta_path = backup_dir / "meta.json"
    payload_dir = backup_dir / "triton_kernels"
    if not meta_path.is_file() or not payload_dir.is_dir():
        return
    importlib.invalidate_caches()
    if find_triton_kernels_dir() is not None:
        return
    target = Path(json.loads(meta_path.read_text(encoding="utf-8"))["target"])
    if (
        target.name != "triton_kernels"
        or not target.parent.is_dir()
        or target.is_symlink()
    ):
        raise ValueError(
            f"Refusing to restore triton_kernels into invalid path: {target}"
        )
    shutil.copytree(payload_dir, target, symlinks=False, dirs_exist_ok=True)
    importlib.invalidate_caches()
    if find_triton_kernels_dir() is None:
        raise RuntimeError(f"Failed to restore triton_kernels into {target}")


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("snapshot")
    capture.add_argument("output", type=Path)
    check = commands.add_parser("check")
    check.add_argument("before", type=Path)
    check.add_argument("after", type=Path)
    commands.add_parser("requirements")
    wheels = commands.add_parser("missing-wheels")
    wheels.add_argument("wheel_dir", type=Path)
    backup_tk = commands.add_parser("backup-triton-kernels")
    backup_tk.add_argument("backup_dir", type=Path)
    restore_tk = commands.add_parser("restore-triton-kernels")
    restore_tk.add_argument("backup_dir", type=Path)
    args = parser.parse_args()
    if args.command == "backup-triton-kernels":
        backup_triton_kernels(args.backup_dir)
        return 0
    if args.command == "restore-triton-kernels":
        restore_triton_kernels(args.backup_dir)
        return 0
    if args.command == "snapshot":
        args.output.write_text(json.dumps(snapshot(), indent=2, sort_keys=True) + "\n")
        return 0
    if args.command == "requirements":
        for name, version in PINNED_ADDITIONS.items():
            if name != "sailify":
                print(f"{name}=={version}")
        return 0
    if args.command == "missing-wheels":
        try:
            packages = snapshot()
            validate_evaluators(packages)
            lines = missing_wheels(args.wheel_dir, packages)
        except ValueError as error:
            print(error, file=sys.stderr)
            return 1
        for line in lines:
            print(line)
        return 0

    before = json.loads(args.before.read_text())
    after = json.loads(args.after.read_text())
    changed = {
        name: {"before": before.get(name), "after": after.get(name)}
        for name in before.keys() | after.keys()
        if name not in BUILD_TARGETS
        and not (
            name not in before
            and name in PINNED_ADDITIONS
            and after[name]["version"] == PINNED_ADDITIONS[name]
        )
        and before.get(name) != after.get(name)
    }
    if changed:
        print("Unexpected dependency changes:", file=sys.stderr)
        print(json.dumps(changed, indent=2, sort_keys=True), file=sys.stderr)
        return 1
    print("All preinstalled dependencies other than build targets are unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
