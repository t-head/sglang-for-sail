# SPDX-License-Identifier: Apache-2.0
"""Bind native wheels and recipes to the same source revision."""

import argparse
import gzip
import hashlib
import io
import json
import re
import subprocess
import tarfile
from pathlib import Path

ROOT = "test/ppu/"
SHA = re.compile(r"[0-9a-f]{40}")
HEX = re.compile(r"[0-9a-f]{64}")
WHEEL = re.compile(r"(?:runtime-deps/)?[A-Za-z0-9][A-Za-z0-9_.+-]*\.whl")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wheel_inventory(bundle, source):
    manifest = bundle / "SHA256SUMS"
    tag = bundle / "source-sha.txt"
    if (
        not SHA.fullmatch(source)
        or tag.is_symlink()
        or tag.read_text() != source + "\n"
    ):
        raise ValueError("wheel source mismatch")
    if manifest.is_symlink() or (bundle / "runtime-deps").is_symlink():
        raise ValueError("wheel manifest paths must be regular")
    files = {"source-sha.txt": tag}
    for path in bundle.rglob("*.whl"):
        key = path.relative_to(bundle).as_posix()
        if not WHEEL.fullmatch(key) or not path.is_file() or path.is_symlink():
            raise ValueError("invalid wheel file")
        files[key] = path
    native_wheels = {
        name.split("-")[0]
        for name in files
        if name.endswith(".whl") and "/" not in name
    }
    if native_wheels != {"sglang", "sglang_kernel"} and native_wheels != {
        "sglang",
        "sgl_kernel",
    }:
        raise ValueError("expected native framework and kernel wheels")
    entries = {}
    for line in manifest.read_text().splitlines():
        recorded, path = line.split("  ", 1)
        path = path.removeprefix("./")
        if path in entries or not HEX.fullmatch(recorded):
            raise ValueError("malformed wheel manifest")
        entries[path] = recorded
    if entries != {key: sha256(path) for key, path in files.items()}:
        raise ValueError("wheel inventory checksum mismatch")


def _tar_recipe_digest(recipes):
    files = {}
    for member in recipes:
        if member.isdir():
            continue
        name = member.name
        if (
            not member.isfile()
            or not name.startswith(ROOT)
            or ".." in name.split("/")
            or name in files
            or member.size > 1024 * 1024
        ):
            raise ValueError("unsafe recipe archive entry")
        files[name] = hashlib.sha256(recipes.extractfile(member).read()).hexdigest()
    if not files:
        raise ValueError("empty recipe tree")
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def recipe_digest(archive):
    with tarfile.open(archive, "r:gz") as recipes:
        return _tar_recipe_digest(recipes)


def git_tree_digest(source):
    if not SHA.fullmatch(source):
        raise ValueError("expected immutable 40-char source SHA")
    content = subprocess.check_output(["git", "archive", source, ROOT])
    with tarfile.open(fileobj=io.BytesIO(content), mode="r:") as recipes:
        return _tar_recipe_digest(recipes)


def checkout_digest(source):
    entries = {}
    raw = subprocess.check_output(["git", "ls-tree", "-rz", source, "--", ROOT])
    for record in raw.split(b"\0"):
        if not record:
            continue
        meta, name = record.decode().split("\t", 1)
        if not meta.startswith(("100644 blob ", "100755 blob ")):
            raise ValueError("recipe files must be regular")
        path = Path(name)
        if path.is_symlink() or not path.is_file():
            raise ValueError("recipe files must be regular")
        entries[name] = sha256(path)
    root_dir = Path(ROOT)
    if not root_dir.is_dir() or root_dir.is_symlink():
        raise ValueError("recipe directory must be a regular directory")
    disk_files = set()
    for path in root_dir.rglob("*"):
        if path.is_symlink():
            raise ValueError("recipe tree must not contain symlinks")
        if path.is_file():
            disk_files.add(path.as_posix())
    if disk_files != set(entries):
        raise ValueError("recipe working tree contains unexpected or missing files")
    return hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=("pack", "check", "cache-key", "checkout-digest")
    )
    parser.add_argument("bundle", type=Path)
    parser.add_argument("source", nargs="?")
    args = parser.parse_args()
    if args.mode == "checkout-digest":
        source = str(args.bundle)
        if not SHA.fullmatch(source):
            raise ValueError("expected immutable 40-char source SHA")
        expected = git_tree_digest(source)
        actual = checkout_digest(source)
        if actual != expected:
            raise ValueError("recipe working tree differs from committed source")
        print(actual)
        return
    if not args.source:
        parser.error("source SHA is required")
    bundle = args.bundle.resolve()
    source = args.source
    if args.mode == "cache-key":
        manifest = json.loads((bundle / "recipe-manifest.json").read_text())
        key = hashlib.sha256(
            (
                manifest["wheel_manifest_sha256"] + manifest["recipe_tree_sha256"]
            ).encode()
        ).hexdigest()
        print(key)
        return
    wheel_inventory(bundle, source)
    archive = bundle / "recipes.tar.gz"
    index = bundle / "recipe-manifest.json"
    if args.mode == "pack":
        content = subprocess.check_output(["git", "archive", source, ROOT])
        archive.write_bytes(gzip.compress(content, mtime=0))
    if archive.is_symlink() or index.is_symlink() or not archive.is_file():
        raise ValueError("recipe manifest paths must be regular")
    expected = {
        "source_sha": source,
        "recipe_archive_sha256": sha256(archive),
        "recipe_tree_sha256": recipe_digest(archive),
        "wheel_manifest_sha256": sha256(bundle / "SHA256SUMS"),
    }
    if checkout_digest(source) != expected["recipe_tree_sha256"]:
        raise ValueError("recipe archive differs from candidate checkout")
    if args.mode == "pack":
        index.write_text(json.dumps(expected, sort_keys=True) + "\n")
    elif json.loads(index.read_text()) != expected:
        raise ValueError("wheel and recipe manifest mismatch")


if __name__ == "__main__":
    main()
