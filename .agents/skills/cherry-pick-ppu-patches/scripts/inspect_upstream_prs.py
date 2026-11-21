"""Check PR merge ancestry against a specified local target, not semantic coverage."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from migration_ledger import resolve


def inspect_pr(repo: Path, target: str, metadata: dict) -> dict:
    result = {
        "number": metadata["number"],
        "url": metadata.get("url"),
        "merged": metadata.get("merged"),
        "state": metadata.get("state"),
        "merge_commit_sha": metadata.get("merge_commit_sha"),
        "head_sha": metadata.get("head_sha"),
        "roadmap_scope": metadata.get("roadmap_scope"),
        "semantic_coverage": "not_reviewed",
    }
    if metadata.get("merged") is False:
        result.update(
            history_presence="not_merged",
            reason="PR is not merged; ignore any tentative merge SHA",
        )
        return result
    if metadata.get("merged") is not True or not metadata.get("merge_commit_sha"):
        result.update(
            history_presence="unknown",
            reason="Missing merged status or real merge commit",
        )
        return result
    merge = metadata["merge_commit_sha"]
    try:
        if resolve(repo, merge) != merge:
            raise ValueError("merge_commit_sha must be a full commit SHA")
    except ValueError as exc:
        result.update(history_presence="unknown", reason=str(exc))
        return result
    check = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", merge, target],
        capture_output=True,
        text=True,
    )
    if check.returncode == 0:
        result.update(
            history_presence="included",
            reason="Real merge commit is an ancestor of the fixed target; review live code/reverts",
        )
    elif check.returncode == 1:
        result.update(
            history_presence="not_in_target",
            reason="Not in target ancestry; check equivalent backports separately",
        )
    else:
        result.update(history_presence="unknown", reason=check.stderr.strip())
    return result


def inspect(repo: Path, target_ref: str, entries: list[dict]) -> dict:
    target = resolve(repo, target_ref)
    shallow = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--is-shallow-repository"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if shallow == "true":
        raise ValueError(
            "Shallow history: fetch complete required history before ancestry checks"
        )
    numbers = [entry["number"] for entry in entries]
    if len(numbers) != len(set(numbers)):
        raise ValueError("Duplicate PR numbers")
    results = [inspect_pr(repo, target, entry) for entry in entries]
    return {
        "target_ref": target_ref,
        "target_base_sha": target,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "counts": dict(Counter(item["history_presence"] for item in results)),
        "prs": results,
        "limitation": "History inclusion only; no code-equivalence or migration/SKIP decision",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--target", required=True)
    parser.add_argument("--prs", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        data = json.loads(args.prs.read_text(encoding="utf-8"))
        result = inspect(args.repo_root, args.target, data["prs"])
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (
        ValueError,
        KeyError,
        TypeError,
        OSError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
