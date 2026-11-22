# SPDX-License-Identifier: Apache-2.0
"""Validate and resolve allowlisted PPU runtime images without exposing private registries."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

TAG_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$")
DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
PUBLIC_DIGEST_RE = re.compile(r"^@sha256:[0-9a-f]{64}$")
REPO_RE = re.compile(r"^[a-z0-9](?:[a-z0-9._:/-]*[a-z0-9])?$")


def validate_tag(tag: str) -> str:
    if not isinstance(tag, str) or not tag or ".." in tag or not TAG_RE.fullmatch(tag):
        raise ValueError(
            "image_tag must be a valid tag without registry or repository path"
        )
    return tag


def validate_digest(digest: str) -> str:
    if not isinstance(digest, str) or not DIGEST_RE.fullmatch(digest):
        raise ValueError("image_digest must match sha256:<64-hex>")
    return digest


def validate_repo(repo: str) -> str:
    if (
        not isinstance(repo, str)
        or not repo
        or ".." in repo
        or "//" in repo
        or "@" in repo
        or not REPO_RE.fullmatch(repo)
    ):
        raise ValueError("invalid runtime image repository")
    last_segment = repo.rsplit("/", 1)[-1]
    if ":" in last_segment:
        raise ValueError("runtime image repository must not include a tag")
    return repo


def extract_repo(image_ref: str) -> str:
    if (
        not isinstance(image_ref, str)
        or not image_ref
        or any(c.isspace() for c in image_ref)
    ):
        raise ValueError("invalid runtime image reference")
    no_digest = image_ref.split("@", 1)[0]
    head, sep, tail = no_digest.rpartition("/")
    if ":" in tail:
        tail = tail.split(":", 1)[0]
    repo = f"{head}{sep}{tail}" if sep else tail
    return validate_repo(repo)


def _parse_allowlist(env: dict[str, str], default_repo: str) -> list[str]:
    raw = env.get("PPU_CI_RECIPE_ALLOWED_IMAGE_REPOS", "").strip()
    if not raw:
        return [default_repo]
    entries = []
    for item in raw.split(","):
        cleaned = item.strip()
        if not cleaned:
            continue
        if cleaned.endswith("/"):
            validate_repo(cleaned[:-1])
            entries.append(cleaned)
        else:
            validate_repo(cleaned)
            entries.append(cleaned)
    if not entries:
        raise ValueError("empty runtime image repository allowlist")
    return entries


def repo_allowed(candidate: str, allowlist: list[str]) -> bool:
    validate_repo(candidate)
    for entry in allowlist:
        if entry.endswith("/"):
            if candidate.startswith(entry) and candidate != entry:
                return True
        elif candidate == entry:
            return True
    return False


def _default_image_and_repo(env: dict[str, str]) -> tuple[str, str]:
    default_ref = (
        env.get("PPU_CI_K8S_RUNTIME_DIGEST", "").strip()
        or env.get("PPU_CI_RUNTIME_IMAGE", "").strip()
    )
    if not default_ref:
        explicit_repo = env.get("PPU_CI_RECIPE_IMAGE_REPO", "").strip()
        if not explicit_repo:
            raise ValueError("runner runtime image configuration is missing")
        return "", validate_repo(explicit_repo)
    return default_ref, extract_repo(default_ref)


def resolve_target(
    env: dict[str, str], *, image_tag: str = "", image_digest: str = ""
) -> dict[str, str]:
    default_ref, default_repo = _default_image_and_repo(env)
    allowlist = _parse_allowlist(env, default_repo)
    target_repo = env.get("PPU_CI_RECIPE_IMAGE_REPO", "").strip() or default_repo
    validate_repo(target_repo)
    if not repo_allowed(target_repo, allowlist):
        raise ValueError("runtime image repository is not in the runner allowlist")

    if image_tag:
        validate_tag(image_tag)
    if image_digest:
        validate_digest(image_digest)

    if not image_tag and not image_digest:
        if not default_ref:
            raise ValueError("default runtime image is not configured")
        if not repo_allowed(default_repo, allowlist):
            raise ValueError("default runtime image repository is not allowed")
        if "@" in default_ref:
            repo_part, digest_part = default_ref.split("@", 1)
            validate_repo(repo_part)
            validate_digest(digest_part)
            return {
                "pull_ref": default_ref,
                "expected_repo": repo_part,
                "expected_digest": digest_part,
                "image_key": digest_part.split(":", 1)[1],
            }
        return {
            "pull_ref": default_ref,
            "expected_repo": default_repo,
            "expected_digest": "",
            "image_key": hashlib.sha256(
                f"ref:{default_ref}".encode("utf-8")
            ).hexdigest(),
        }

    if image_digest:
        pull_ref = f"{target_repo}@{image_digest}"
        return {
            "pull_ref": pull_ref,
            "expected_repo": target_repo,
            "expected_digest": image_digest,
            "image_key": image_digest.split(":", 1)[1],
        }

    pull_ref = f"{target_repo}:{image_tag}"
    return {
        "pull_ref": pull_ref,
        "expected_repo": target_repo,
        "expected_digest": "",
        "image_key": hashlib.sha256(f"tag:{pull_ref}".encode("utf-8")).hexdigest(),
    }


def verify_pullable_image_id(
    image_id: str,
    env: dict[str, str],
    *,
    expected_repo: str = "",
    expected_digest: str = "",
) -> str:
    if not isinstance(image_id, str) or not image_id:
        raise ValueError("missing resolved image ID")
    ref = image_id.removeprefix("docker-pullable://")
    if "@" not in ref:
        raise ValueError("resolved image ID is missing an immutable digest")
    repo, digest = ref.split("@", 1)
    validate_repo(repo)
    validate_digest(digest)
    _, default_repo = _default_image_and_repo(env)
    allowlist = _parse_allowlist(env, default_repo)
    if not repo_allowed(repo, allowlist):
        raise ValueError("resolved image repository is not in the runner allowlist")
    if expected_repo and repo != expected_repo:
        raise ValueError(
            "resolved image repository does not match requested repository"
        )
    if expected_digest and digest != expected_digest:
        raise ValueError("resolved image digest does not match requested digest")
    return f"@{digest}"


def verify_pod_status(
    pod_status_path: Path,
    env: dict[str, str],
    *,
    expected_repo: str = "",
    expected_digest: str = "",
) -> str:
    document = json.loads(pod_status_path.read_text(encoding="utf-8"))
    statuses = document.get("status", {}).get("containerStatuses", [])
    worker_ids = [
        item.get("imageID", "")
        for item in statuses
        if isinstance(item, dict) and item.get("name") == "worker"
    ]
    if len(worker_ids) != 1:
        raise ValueError("worker container image status is missing")
    return verify_pullable_image_id(
        worker_ids[0],
        env,
        expected_repo=expected_repo,
        expected_digest=expected_digest,
    )


def docker_pull_and_verify(
    env: dict[str, str],
    pull_log: Path,
    *,
    image_tag: str = "",
    image_digest: str = "",
) -> dict[str, str]:
    target = resolve_target(env, image_tag=image_tag, image_digest=image_digest)
    pull_ref = target["pull_ref"]
    expected_repo = target["expected_repo"]
    expected_digest = target["expected_digest"]

    with pull_log.open("wb") as log_handle:
        pull = subprocess.run(
            ["docker", "pull", pull_ref],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if pull.returncode != 0:
        raise ValueError("failed to pull requested PPU runtime image")

    inspect_id = subprocess.run(
        ["docker", "image", "inspect", pull_ref, "--format", "{{.Id}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    image_id = inspect_id.stdout.strip()
    if inspect_id.returncode != 0 or not DIGEST_RE.fullmatch(image_id):
        raise ValueError("failed to inspect immutable runtime image ID")

    inspect_digests = subprocess.run(
        [
            "docker",
            "image",
            "inspect",
            pull_ref,
            "--format",
            "{{range .RepoDigests}}{{println .}}{{end}}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    resolved_digest = ""
    if inspect_digests.returncode == 0:
        prefix = f"{expected_repo}@"
        for line in inspect_digests.stdout.splitlines():
            line = line.strip()
            if line.startswith(prefix):
                candidate = line[len(prefix) :]
                if DIGEST_RE.fullmatch(candidate):
                    resolved_digest = candidate
                    break
    if not resolved_digest and pull_log.is_file():
        for line in pull_log.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split()
            if (
                len(parts) >= 2
                and parts[0] == "Digest:"
                and DIGEST_RE.fullmatch(parts[1])
            ):
                resolved_digest = parts[1]
    if not DIGEST_RE.fullmatch(resolved_digest):
        raise ValueError("docker pull did not produce a verified sha256 digest")
    if expected_digest and resolved_digest != expected_digest:
        raise ValueError("pulled runtime image digest does not match requested digest")

    public_digest = verify_pullable_image_id(
        f"{expected_repo}@{resolved_digest}",
        env,
        expected_repo=expected_repo,
        expected_digest=expected_digest,
    )
    return {
        "image_id": image_id,
        "public_digest": public_digest,
        "image_key": resolved_digest.split(":", 1)[1],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_val = sub.add_parser("validate-inputs")
    p_val.add_argument("--image-tag", default="")
    p_val.add_argument("--image-digest", default="")

    p_res = sub.add_parser("resolve")
    p_res.add_argument("--image-tag", default="")
    p_res.add_argument("--image-digest", default="")

    p_pod = sub.add_parser("verify-pod")
    p_pod.add_argument("pod_status", type=Path)
    p_pod.add_argument("--expected-repo", default="")
    p_pod.add_argument("--expected-digest", default="")

    p_pull = sub.add_parser("docker-pull")
    p_pull.add_argument("pull_log", type=Path)
    p_pull.add_argument("--image-tag", default="")
    p_pull.add_argument("--image-digest", default="")

    args = parser.parse_args()
    try:
        if args.command == "validate-inputs":
            if args.image_tag:
                validate_tag(args.image_tag)
            if args.image_digest:
                validate_digest(args.image_digest)
            return 0
        if args.command == "resolve":
            result = resolve_target(
                os.environ,
                image_tag=args.image_tag,
                image_digest=args.image_digest,
            )
            print(json.dumps(result))
            return 0
        if args.command == "verify-pod":
            public_digest = verify_pod_status(
                args.pod_status,
                os.environ,
                expected_repo=args.expected_repo,
                expected_digest=args.expected_digest,
            )
            print(public_digest)
            return 0
        if args.command == "docker-pull":
            result = docker_pull_and_verify(
                os.environ,
                args.pull_log,
                image_tag=args.image_tag,
                image_digest=args.image_digest,
            )
            print(json.dumps(result))
            return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"PPU runtime image validation failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
