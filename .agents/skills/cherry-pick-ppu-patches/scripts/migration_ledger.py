"""Inventory source-only commits and check a migration ledger; never mutate Git."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

APPLIED = {"DIRECT", "RESOLVED", "ADAPTED"}
UNFINISHED = {"PENDING", "BLOCKED"}
DECISIONS = APPLIED | UNFINISHED | {"SKIP", "EXCLUDED"}
CLASSIFICATIONS = {"ppu", "dependency", "community_fix", "unrelated"}
COVERAGE = {"full", "partial", "absent", "obsolete", "unknown"}


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
    )
    if result.returncode:
        raise ValueError(f"git {' '.join(args)}: {result.stderr.strip()}")
    return result.stdout.rstrip("\n")


def resolve(repo: Path, ref: str) -> str:
    return git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}")


def require_ancestor(repo: Path, base: str, tip: str) -> None:
    git(repo, "merge-base", "--is-ancestor", base, tip)


def candidates(repo: Path, base: str, tip: str) -> list[str]:
    require_ancestor(repo, base, tip)
    return git(
        repo, "rev-list", "--reverse", "--topo-order", f"{base}..{tip}"
    ).splitlines()


def parents(repo: Path, sha: str) -> list[str]:
    return git(repo, "show", "-s", "--format=%P", sha).split()


def initialize(
    repo: Path, source_base: str, source_tip: str, target_base: str | None = None
) -> dict:
    if git(repo, "rev-parse", "--is-shallow-repository") == "true":
        raise ValueError("Shallow history: fetch the required complete history first")
    refs = {
        key: {"ref": ref, "sha": resolve(repo, ref)}
        for key, ref in (
            ("source_base", source_base),
            ("source_tip", source_tip),
            ("target_base", target_base),
        )
        if ref is not None
    }
    refs.setdefault("target_base", None)
    source_shas = candidates(
        repo, refs["source_base"]["sha"], refs["source_tip"]["sha"]
    )
    records = []
    for sha in source_shas:
        records.append(
            {
                "source_sha": sha,
                "parents": parents(repo, sha),
                "subject": git(repo, "show", "-s", "--format=%s", sha),
                "classification": "unclassified",
                "intent": "",
                "impact": [],
                "dependencies": [],
                "external_dependencies": [],
                "coverage": {
                    "state": "unknown",
                    "evidence": [],
                    "remaining": [],
                    "target_base_sha": refs["target_base"]["sha"]
                    if refs["target_base"]
                    else None,
                    "basis_sha": None,
                },
                "decision": "PENDING",
                "reason": "",
                "target_before": None,
                "target_sha": None,
                "conflicts": [],
                "adjustments": [],
                "validation": [],
                "review": "pending",
                "operations": [],
                "notes": "",
            }
        )
    remotes = {
        name: git(repo, "remote", "get-url", name)
        for name in git(repo, "remote").splitlines()
    }
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository": str(repo.resolve()),
        "remotes": remotes,
        "refs": refs,
        "candidate_count": len(records),
        "records": records,
        "integration_commits": [],
    }


def check(data: dict, repo: Path, target_tip: str | None, complete: bool) -> Counter:
    def need(condition, message):
        if not condition:
            raise ValueError(message)

    need(data.get("schema_version") == 1, "Unsupported ledger schema")
    refs = data["refs"]
    fixed = {}
    for key in ("source_base", "source_tip"):
        sha = refs[key]["sha"]
        need(resolve(repo, sha) == sha, f"{key} must contain a full commit SHA")
        fixed[key] = sha
    target_ref = refs.get("target_base")
    fixed["target_base"] = target_ref["sha"] if target_ref else None
    if fixed["target_base"]:
        need(
            resolve(repo, fixed["target_base"]) == fixed["target_base"],
            "target_base must contain a full commit SHA",
        )
    else:
        need(
            not target_tip and not complete,
            "Inventory-only ledger needs a fixed target_base before migration",
        )
    need(
        git(repo, "rev-parse", "--is-shallow-repository") == "false", "Shallow history"
    )
    expected = candidates(repo, fixed["source_base"], fixed["source_tip"])
    records = data["records"]
    need(
        [r["source_sha"] for r in records] == expected,
        "Records must preserve every source candidate in original topological order",
    )
    need(data["candidate_count"] == len(expected), "candidate_count mismatch")
    order = {sha: i for i, sha in enumerate(expected)}
    tip = resolve(repo, target_tip) if target_tip else None
    if complete:
        need(tip is not None, "--complete requires --target-tip")
    if tip:
        require_ancestor(repo, fixed["target_base"], tip)
    mapped = []
    counts = Counter()
    evidence_root = Path(data.get("_evidence_root", "."))

    def check_validation(items, label):
        need(
            items,
            f"{label}: validation record required (including NOT_RUN if unavailable)",
        )
        for validation in items:
            need(
                validation.get("command") and validation.get("evidence"),
                f"{label}: validation lacks command/evidence",
            )
            need(
                validation.get("result") in {"PASS", "FAIL", "NOT_RUN", "N/A"},
                f"{label}: invalid validation result",
            )
            for artifact in validation.get("artifacts", []):
                need(
                    (evidence_root / artifact).is_file(),
                    f"{label}: validation artifact missing: {artifact}",
                )

    for i, r in enumerate(records):
        sha = r["source_sha"]
        label = sha[:12]
        need(r["parents"] == parents(repo, sha), f"{label}: source parents changed")
        decision = r["decision"]
        need(decision in DECISIONS, f"{label}: invalid decision")
        if not fixed["target_base"]:
            need(
                decision not in APPLIED | {"SKIP"},
                f"{label}: migration decision requires target_base",
            )
        counts[decision] += 1
        state = r["coverage"]["state"]
        need(state in COVERAGE, f"{label}: invalid coverage state")
        for dep in r["dependencies"]:
            need(
                dep in order and order[dep] < i,
                f"{label}: missing/later source dependency {dep}",
            )
        target = r.get("target_sha")
        if decision in {"SKIP", "EXCLUDED"}:
            need(target is None, f"{label}: {decision} must not create a target commit")
        if decision == "SKIP":
            need(
                state in {"full", "obsolete"},
                f"{label}: SKIP needs full/obsolete coverage",
            )
            need(r["coverage"]["evidence"], f"{label}: SKIP needs coverage evidence")
        if decision in APPLIED:
            need(target, f"{label}: applied record needs target_sha")
            need(
                state in {"absent", "partial"},
                f"{label}: applied record has incompatible coverage",
            )
            need(
                not (decision == "DIRECT" and state == "partial"),
                f"{label}: partial coverage needs adaptation",
            )
            need(
                resolve(repo, target) == target,
                f"{label}: target_sha must be a full SHA",
            )
            before = r.get("target_before")
            need(
                before and resolve(repo, before) == before,
                f"{label}: target_before must be a full SHA",
            )
            need(
                parents(repo, target) == [before],
                f"{label}: target_before must be the actual sole parent",
            )
            require_ancestor(repo, fixed["target_base"], before)
            if tip:
                require_ancestor(repo, target, tip)
            mapped.append(target)
            need(
                bool(r["conflicts"]) == (decision == "RESOLVED"),
                f"{label}: conflict/status mismatch",
            )
            if decision == "ADAPTED" or state == "partial":
                need(
                    r["adjustments"],
                    f"{label}: adapted/partial record needs adjustment details",
                )
        if decision in UNFINISHED:
            need(not complete, f"{label}: unfinished {decision}")
            continue
        if decision in APPLIED | {"SKIP"}:
            coverage = r["coverage"]
            need(
                coverage.get("target_base_sha") == fixed["target_base"],
                f"{label}: coverage target_base_sha must match the fixed target",
            )
            basis = coverage.get("basis_sha")
            need(
                basis
                and resolve(repo, basis) == basis
                and basis == r.get("target_before"),
                f"{label}: coverage basis_sha must be the full reviewed target_before SHA",
            )
            require_ancestor(repo, fixed["target_base"], basis)
            if tip:
                require_ancestor(repo, basis, tip)
            if r.get("code_split"):
                need(
                    r["code_split"].get("basis_sha") == basis,
                    f"{label}: code_split basis_sha must match the reviewed coverage basis",
                )
        need(
            r["classification"] in CLASSIFICATIONS, f"{label}: classification required"
        )
        need(r["intent"] and r["reason"], f"{label}: intent/reason required")
        need(
            r["impact"] and r["operations"],
            f"{label}: impact and operation history required",
        )
        need(r["review"] == "complete", f"{label}: semantic review incomplete")
        if decision in APPLIED:
            check_validation(r["validation"], label)
            for conflict in r["conflicts"]:
                for field in (
                    "path",
                    "symbol",
                    "evidence",
                    "cause",
                    "source_intent",
                    "target_intent",
                    "resolution",
                    "verification",
                ):
                    need(conflict.get(field), f"{label}: conflict missing {field}")
                for evidence in conflict["evidence"]:
                    need(
                        (evidence_root / evidence).is_file(),
                        f"{label}: conflict evidence file missing: {evidence}",
                    )
        if state == "partial":
            need(
                r["coverage"]["evidence"] and r["coverage"]["remaining"],
                f"{label}: partial coverage lacks existing/residual analysis",
            )
            if decision in APPLIED:
                split = r.get("code_split", {})
                need(
                    isinstance(split.get("community_covered"), list)
                    and split["community_covered"]
                    and isinstance(split.get("retain"), list)
                    and split["retain"],
                    f"{label}: partial coverage needs community_covered and retain code-unit lists",
                )
    need(len(mapped) == len(set(mapped)), "Target commit mapped more than once")
    for earlier, later in zip(mapped, mapped[1:]):
        require_ancestor(repo, earlier, later)
        need(earlier != later, "Duplicate target commit")
    extras = []
    for entry in data.get("integration_commits", []):
        need(fixed["target_base"], "Integration commit requires target_base")
        sha = entry["target_sha"]
        need(resolve(repo, sha) == sha, "Integration commit must have full SHA")
        need(
            entry.get("reason")
            and entry.get("source_shas")
            and entry.get("validation"),
            "Integration commit needs reason, source_shas and validation",
        )
        need(
            all(s in order for s in entry["source_shas"]), "Unknown integration source"
        )
        check_validation(entry["validation"], f"integration {sha[:12]}")
        extras.append(sha)
    need(
        len(mapped + extras) == len(set(mapped + extras)),
        "Duplicate target/integration mapping",
    )
    if tip:
        actual = set(candidates(repo, fixed["target_base"], tip))
        need(set(mapped + extras) <= actual, "Mapped target outside migration range")
        if complete:
            need(
                set(mapped + extras) == actual,
                "Unaccounted target commits or missing mappings",
            )
    return counts


def stale_coverage_warnings(data: dict) -> list[str]:
    target = (data.get("refs", {}).get("target_base") or {}).get("sha")
    if not target:
        return []
    warnings = []
    for record in data.get("records", []):
        if record.get("decision") not in UNFINISHED:
            continue
        coverage = record.get("coverage", {})
        reviewed_target = coverage.get("target_base_sha")
        has_review = (
            coverage.get("state") != "unknown"
            or coverage.get("basis_sha")
            or coverage.get("evidence")
        )
        if has_review and reviewed_target and reviewed_target != target:
            warnings.append(
                f"{record['source_sha'][:12]}: stale coverage for another target; re-review before migration"
            )
    prs = data.get("target_upstream_prs")
    if isinstance(prs, dict) and prs.get("target_base_sha") != target:
        warnings.append(
            "stale target_upstream_prs inventory; regenerate for the fixed target"
        )
    return warnings


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    init = sub.add_parser("init")
    init.add_argument("--repo-root", type=Path, default=Path.cwd())
    for flag in ("--source-base", "--source-tip"):
        init.add_argument(flag, required=True)
    init.add_argument(
        "--target-base", help="Omit for inventory-only; fix this ref before migration"
    )
    init.add_argument("--output", type=Path, required=True)
    verify = sub.add_parser("check")
    verify.add_argument("ledger", type=Path)
    verify.add_argument("--repo-root", type=Path)
    verify.add_argument("--target-tip")
    verify.add_argument("--complete", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.action == "init":
            data = initialize(
                args.repo_root, args.source_base, args.source_tip, args.target_base
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(data, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            print(f"Inventoried {data['candidate_count']} candidates: {args.output}")
        else:
            data = json.loads(args.ledger.read_text(encoding="utf-8"))
            data["_evidence_root"] = str(args.ledger.resolve().parent)
            counts = check(
                data,
                args.repo_root or Path(data["repository"]),
                args.target_tip,
                args.complete,
            )
            print(
                json.dumps(
                    {
                        "structural_check": "PASS",
                        "counts": dict(counts),
                        "warnings": stale_coverage_warnings(data),
                    },
                    ensure_ascii=False,
                )
            )
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
