"""Flag added server_args reads where declaration/getter contracts need review."""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

from migration_ledger import git, require_ancestor, resolve


def added_lines(diff: str) -> set[int]:
    lines = set()
    current = 0
    for line in diff.splitlines():
        if line.startswith("@@"):
            match = re.search(r"\+(\d+)(?:,\d+)? @@", line)
            if not match:
                raise ValueError("Cannot parse diff hunk")
            current = int(match.group(1))
        elif line.startswith(("+++", "---")):
            continue
        elif line.startswith("+"):
            lines.add(current)
            current += 1
        elif line.startswith(" "):
            current += 1
    return lines


def resolvable_fields(source: str) -> set[str]:
    tree = ast.parse(source)
    fields = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        if any(
            isinstance(item, ast.keyword)
            and item.arg == "resolvable"
            and isinstance(item.value, ast.Constant)
            and item.value.value is True
            for item in ast.walk(node.annotation)
        ):
            fields.add(node.target.id)
    return fields


def audit(repo, base, tip, definitions_path):
    base, tip = resolve(repo, base), resolve(repo, tip)
    require_ancestor(repo, base, tip)
    fields = resolvable_fields(git(repo, "show", f"{tip}:{definitions_path}"))
    paths = git(
        repo,
        "diff",
        "--name-only",
        "--diff-filter=ACMRT",
        "-z",
        base,
        tip,
    ).split("\0")
    findings = []
    checked = 0
    for path in paths:
        if not path.endswith(".py"):
            continue
        source = git(repo, "show", f"{tip}:{path}")
        tree = ast.parse(source, filename=path)
        added = added_lines(git(repo, "diff", "--unified=0", base, tip, "--", path))
        getters = {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id.startswith("get_")
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute) or not isinstance(
                node.ctx, ast.Load
            ):
                continue
            receiver = node.value
            raw = (isinstance(receiver, ast.Name) and receiver.id == "server_args") or (
                isinstance(receiver, ast.Attribute) and receiver.attr == "server_args"
            )
            if not raw or not (set(range(node.lineno, node.end_lineno + 1)) & added):
                continue
            checked += 1
            reasons = []
            if node.attr in fields:
                reasons.append("resolvable=True")
            if node.attr in getters:
                reasons.append("same-file getter convention")
            if reasons:
                findings.append(
                    f"REVIEW {path}:{node.lineno} {node.attr}: {', '.join(reasons)}"
                )
    return fields, checked, findings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--base", required=True)
    parser.add_argument("--tip", required=True)
    parser.add_argument("--definitions", default="python/sglang/srt/server_args.py")
    args = parser.parse_args(argv)
    try:
        fields, checked, findings = audit(
            args.repo_root, args.base, args.tip, args.definitions
        )
        for finding in findings:
            print(finding)
        applicability = (
            "DECLARATIONS_FOUND"
            if fields
            else "NOT_APPLICABLE(no resolvable declarations)"
        )
        print(f"SERVERARGS checked={checked} review={len(findings)} {applicability}")
        return 1 if findings else 0
    except (ValueError, OSError, SyntaxError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
