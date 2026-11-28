"""Compare newly added Python members using explicit source/target SHAs."""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

from migration_ledger import git, parents, resolve


def parent_of(repo: Path, sha: str, explicit: str | None) -> str:
    choices = parents(repo, sha)
    if explicit:
        parent = resolve(repo, explicit)
        if parent not in choices:
            raise ValueError(f"{explicit} is not a parent of {sha}")
        return parent
    if len(choices) != 1:
        raise ValueError(f"{sha} requires an explicit parent (root/merge commit)")
    return choices[0]


def definitions(source: str) -> dict[tuple[str, str], int]:
    result = {}

    def visit(body, owner=""):
        for node in body:
            if isinstance(node, ast.ClassDef):
                visit(node.body, f"{owner}.{node.name}".strip("."))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                result[(owner or "<module>", node.name)] = node.lineno
            else:
                # Class definitions may be conditional, but nested function locals
                # must not be mistaken for methods of their surrounding class.
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, ast.stmt):
                        visit([child], owner)

    visit(ast.parse(source).body)
    return result


def read_blob(repo: Path, sha: str, path: str) -> str:
    # A new or deleted path is legitimate; other Git/read failures are errors.
    available = git(repo, "ls-tree", "-r", "--name-only", sha, "--", path).splitlines()
    return git(repo, "show", f"{sha}:{path}") if path in available else ""


def audit(repo, source, target, source_parent=None, target_parent=None, file_map=None):
    source, target = resolve(repo, source), resolve(repo, target)
    before_source = parent_of(repo, source, source_parent)
    before_target = parent_of(repo, target, target_parent)
    mapping = file_map or {}
    paths = git(repo, "diff", "--name-only", "-z", before_source, source).split("\0")
    findings = []
    checked = 0
    for path in paths:
        if not path.endswith(".py"):
            continue
        old = definitions(read_blob(repo, before_source, path))
        new = definitions(read_blob(repo, source, path))
        added = new.keys() - old.keys()
        if not added:
            continue
        target_path = mapping.get(path, path)
        landed = definitions(read_blob(repo, target, target_path))
        target_old = definitions(read_blob(repo, before_target, target_path))
        for owner, name in sorted(added):
            checked += 1
            hits = [(o, n) for o, n in landed if n == name]
            if (owner, name) in landed and (owner, name) not in target_old:
                continue
            if len(hits) == 1:
                detail = f"{owner}.{name} -> {hits[0][0]}.{name}"
                if hits[0] in target_old:
                    detail += (
                        " (already existed before target; check upstream coverage)"
                    )
            elif not hits:
                detail = (
                    f"{owner}.{name}: not found; check rename/extraction/file mapping"
                )
            else:
                detail = (
                    f"{owner}.{name}: ambiguous target owners {[o for o, _ in hits]}"
                )
            findings.append(f"REVIEW {path} -> {target_path}: {detail}")
    return checked, findings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--source-parent")
    parser.add_argument("--target-parent")
    parser.add_argument("--file-map", action="append", default=[], metavar="OLD=NEW")
    args = parser.parse_args(argv)
    try:
        mapping = {}
        for entry in args.file_map:
            old, sep, new = entry.partition("=")
            if not sep or not old or not new or old in mapping:
                raise ValueError(f"Invalid/duplicate file mapping: {entry}")
            mapping[old] = new
        checked, findings = audit(
            args.repo_root,
            args.source,
            args.target,
            args.source_parent,
            args.target_parent,
            mapping,
        )
        for finding in findings:
            print(finding)
        print(f"CLASS_PLACEMENT checked={checked} review={len(findings)}")
        return 1 if findings else 0
    except (ValueError, OSError, SyntaxError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
