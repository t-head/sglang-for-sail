"""Behavior tests in isolated Git repositories; no PPU or network required."""

import copy
import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import audit_class_placement as placement
import audit_serverargs_reads as reads
import inspect_upstream_prs as upstream
import migration_ledger as ledger


class GitFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # Synthetic repos must not inherit the developer's signing/hooks settings.
        self.environment = patch.dict(
            os.environ,
            {
                "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_CONFIG_NOSYSTEM": "1",
            },
        )
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.git("init", "-q", "--initial-branch=base")
        self.git("config", "user.name", "Skill Test")
        self.git("config", "user.email", "skill-test@example.invalid")

    def git(self, *args):
        return ledger.git(self.root, *args)

    def write(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def commit(self, name, content, message="update"):
        self.write(name, content)
        self.git("add", "--", name)
        self.git("commit", "-q", "-m", message)
        return self.git("rev-parse", "HEAD")

    def start_migration(self):
        base = self.commit("module.py", "class A:\n    pass\n", "community old")
        self.git("checkout", "-q", "-b", "ppu")
        source = self.commit(
            "module.py", "class A:\n    def run(self):\n        return 1\n", "update"
        )
        self.git("checkout", "-q", "-b", "target", base)
        data = ledger.initialize(self.root, base, source, base)
        return base, source, data

    def finish(self, record, target, before):
        record.update(
            {
                "classification": "ppu",
                "intent": "support serving",
                "reason": "missing on target",
                "impact": ["module.py:A.run"],
                "operations": ["fixture applied and reviewed"],
                "decision": "DIRECT",
                "target_before": before,
                "target_sha": target,
                "review": "complete",
                "validation": [
                    {
                        "command": "fixture comparison",
                        "result": "PASS",
                        "evidence": "compared final content",
                    }
                ],
            }
        )
        record["coverage"]["state"] = "absent"
        record["coverage"]["basis_sha"] = before

    def test_inventory_includes_unlabelled_and_merge_commits(self):
        base = self.commit("base.txt", "base", "community")
        self.git("checkout", "-q", "-b", "ppu")
        first = self.commit("a.txt", "a", "update")
        self.git("checkout", "-q", "-b", "side", base)
        side = self.commit("b.txt", "b", "fix build")
        self.git("checkout", "-q", "ppu")
        self.git("merge", "--no-ff", "-m", "merge adaptation", "side")
        merge = self.git("rev-parse", "HEAD")
        data = ledger.initialize(self.root, base, "ppu", base)
        records = {r["source_sha"]: r for r in data["records"]}
        self.assertEqual(set(records), {first, side, merge})
        self.assertEqual(len(records[merge]["parents"]), 2)
        self.assertEqual(data["records"][-1]["source_sha"], merge)

    def test_wrong_old_baseline_and_unknown_ref_fail(self):
        base, source, _ = self.start_migration()
        other = self.commit("other.txt", "other")
        with self.assertRaises(ValueError):
            ledger.initialize(self.root, other, source, base)
        with self.assertRaises(ValueError):
            ledger.initialize(self.root, "missing-ref", source, base)

    def test_inventory_does_not_require_new_target_version(self):
        base, source, _ = self.start_migration()
        data = ledger.initialize(self.root, base, source)
        self.assertIsNone(data["refs"]["target_base"])
        self.assertEqual(ledger.check(data, self.root, None, False)["PENDING"], 1)
        with self.assertRaisesRegex(ValueError, "fixed target_base"):
            ledger.check(data, self.root, base, True)
        data["refs"]["target_base"] = {"ref": base, "sha": base}
        ledger.check(data, self.root, None, False)

    def test_direct_mapping_and_missing_candidate(self):
        base, source, data = self.start_migration()
        self.git("cherry-pick", "-x", source)
        target = self.git("rev-parse", "HEAD")
        self.finish(data["records"][0], target, base)
        self.assertEqual(ledger.check(data, self.root, target, True)["DIRECT"], 1)
        bad = copy.deepcopy(data)
        bad["records"] = []
        with self.assertRaisesRegex(ValueError, "every source candidate"):
            ledger.check(bad, self.root, target, True)

    def test_complete_requires_tip_and_no_pending(self):
        base, _, data = self.start_migration()
        with self.assertRaisesRegex(ValueError, "target-tip"):
            ledger.check(data, self.root, None, True)
        with self.assertRaisesRegex(ValueError, "unfinished"):
            ledger.check(data, self.root, base, True)
        self.assertEqual(ledger.check(data, self.root, None, False)["PENDING"], 1)

    def test_skip_requires_evidence_and_full_coverage(self):
        base, _, data = self.start_migration()
        record = data["records"][0]
        record.update(
            classification="ppu",
            intent="already supported",
            decision="SKIP",
            reason="upstream covers it",
            review="complete",
            impact=["module.py:A.run"],
            operations=["checked upstream coverage"],
            target_before=base,
        )
        record["coverage"]["basis_sha"] = base
        record["coverage"]["state"] = "partial"
        with self.assertRaisesRegex(ValueError, "full/obsolete"):
            ledger.check(data, self.root, base, True)
        record["coverage"]["state"] = "full"
        with self.assertRaisesRegex(ValueError, "coverage evidence"):
            ledger.check(data, self.root, base, True)
        record["coverage"]["evidence"] = ["fixture reference and semantic comparison"]
        self.assertEqual(ledger.check(data, self.root, base, True)["SKIP"], 1)

    def test_unaccounted_target_commit_and_integration_mapping(self):
        base, source, data = self.start_migration()
        self.git("cherry-pick", "-x", source)
        target = self.git("rev-parse", "HEAD")
        self.finish(data["records"][0], target, base)
        extra = self.commit("integration.txt", "integration")
        with self.assertRaisesRegex(ValueError, "Unaccounted"):
            ledger.check(data, self.root, extra, True)
        data["integration_commits"] = [
            {
                "target_sha": extra,
                "reason": "fixture integration",
                "source_shas": [source],
                "validation": [
                    {
                        "command": "fixture check",
                        "result": "PASS",
                        "evidence": "checked",
                    }
                ],
            }
        ]
        ledger.check(data, self.root, extra, True)
        data["integration_commits"][0]["validation"][0]["artifacts"] = [
            str(self.root / "missing.log")
        ]
        with self.assertRaisesRegex(ValueError, "validation artifact missing"):
            ledger.check(data, self.root, extra, True)

    def test_finished_record_requires_impact_and_history(self):
        base, source, data = self.start_migration()
        self.git("cherry-pick", "-x", source)
        target = self.git("rev-parse", "HEAD")
        record = data["records"][0]
        self.finish(record, target, base)
        record["operations"] = []
        with self.assertRaisesRegex(ValueError, "operation history"):
            ledger.check(data, self.root, target, True)

    def test_missing_validation_log_is_rejected(self):
        base, source, data = self.start_migration()
        self.git("cherry-pick", "-x", source)
        target = self.git("rev-parse", "HEAD")
        record = data["records"][0]
        self.finish(record, target, base)
        log = self.root / "validation.log"
        record["validation"][0]["artifacts"] = [str(log)]
        with self.assertRaisesRegex(ValueError, "validation artifact missing"):
            ledger.check(data, self.root, target, True)
        log.write_text("fixture output", encoding="utf-8")
        ledger.check(data, self.root, target, True)

    def test_resolved_requires_real_conflict_evidence(self):
        base, source, data = self.start_migration()
        self.git("cherry-pick", "-x", source)
        target = self.git("rev-parse", "HEAD")
        record = data["records"][0]
        self.finish(record, target, base)
        record["decision"] = "RESOLVED"
        with self.assertRaisesRegex(ValueError, "conflict/status"):
            ledger.check(data, self.root, target, True)
        record["conflicts"] = [
            {
                "path": "module.py",
                "symbol": "A.run",
                "evidence": [str(self.root / "conflict.diff")],
                "cause": "fixture conflict",
                "source_intent": "source behavior",
                "target_intent": "target behavior",
                "resolution": "semantic resolution",
                "verification": "fixture",
            }
        ]
        with self.assertRaisesRegex(ValueError, "evidence file missing"):
            ledger.check(data, self.root, target, True)
        self.write("conflict.diff", "fixture conflict content")
        ledger.check(data, self.root, target, True)

    def test_initialize_does_not_overwrite_existing_ledger(self):
        base, source, _ = self.start_migration()
        report = self.root / "ledger.json"
        report.write_text("preserve this", encoding="utf-8")
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            code = ledger.main(
                [
                    "init",
                    "--repo-root",
                    str(self.root),
                    "--source-base",
                    base,
                    "--source-tip",
                    source,
                    "--target-base",
                    base,
                    "--output",
                    str(report),
                ]
            )
        self.assertEqual(code, 2)
        self.assertEqual(report.read_text(), "preserve this")

    def test_class_placement_explicit_source_without_subject_matching(self):
        base, source, _ = self.start_migration()
        wrong = self.commit(
            "module.py",
            "class B:\n    def run(self):\n        return 1\n",
            "different title",
        )
        checked, findings = placement.audit(self.root, source, wrong)
        self.assertEqual(checked, 1)
        self.assertIn("A.run -> B.run", findings[0])
        self.git("checkout", "-q", "-b", "correct", base)
        correct = self.commit(
            "module.py",
            "class A:\n    def run(self):\n        return 1\n",
            "another title",
        )
        self.assertEqual(placement.audit(self.root, source, correct), (1, []))

    def test_nested_functions_are_not_class_members(self):
        found = placement.definitions(
            "class A:\n    def run(self):\n        def local():\n            pass\n"
        )
        self.assertEqual(set(found), {("A", "run")})

    def test_class_placement_missing_ref_is_error(self):
        _, source, _ = self.start_migration()
        with self.assertRaises(ValueError):
            placement.audit(self.root, "missing-ref", source)

    def test_config_new_file_and_parse_time_candidate(self):
        config = "class ServerArgs:\n    page_size: A[int, Param(resolvable=True)]\n"
        base = self.commit("python/sglang/srt/server_args.py", config)
        tip = self.commit(
            "new.py", "def run(server_args):\n    return server_args.page_size\n"
        )
        fields, checked, findings = reads.audit(
            self.root, base, tip, "python/sglang/srt/server_args.py"
        )
        self.assertEqual(fields, {"page_size"})
        self.assertEqual(checked, 1)
        self.assertIn("resolvable=True", findings[0])

    def test_config_ignores_comments_and_cli_fields(self):
        config = "class ServerArgs:\n    page_size: A[int, Param(resolvable=True)]\n    port: int = 30000\n"
        base = self.commit("python/sglang/srt/server_args.py", config)
        tip = self.commit(
            "new.py",
            "# server_args.page_size\ndef run(server_args):\n    return server_args.port\n",
        )
        _, checked, findings = reads.audit(
            self.root, base, tip, "python/sglang/srt/server_args.py"
        )
        self.assertEqual((checked, findings), (1, []))

    def test_config_no_declarations_is_explicitly_inapplicable(self):
        base = self.commit(
            "python/sglang/srt/server_args.py",
            "class ServerArgs:\n    port: int = 30000\n",
        )
        tip = self.commit(
            "new.py", "def run(server_args):\n    return server_args.port\n"
        )
        with redirect_stdout(io.StringIO()) as output:
            code = reads.main(
                ["--repo-root", str(self.root), "--base", base, "--tip", tip]
            )
        self.assertEqual(code, 0)
        self.assertIn("NOT_APPLICABLE", output.getvalue())

    def test_config_unknown_ref_fails(self):
        base = self.commit(
            "python/sglang/srt/server_args.py", "class ServerArgs:\n    pass\n"
        )
        with self.assertRaises(ValueError):
            reads.audit(
                self.root, base, "missing-ref", "python/sglang/srt/server_args.py"
            )

    def test_pr_in_main_is_not_automatically_in_release_target(self):
        base, _, _ = self.start_migration()
        self.git("checkout", "-q", "-b", "community-main", base)
        merged = self.commit("feature.py", "enabled = True\n", "upstream PR merge")
        metadata = {
            "number": 1,
            "merged": True,
            "state": "closed",
            "merge_commit_sha": merged,
        }
        release = upstream.inspect(self.root, base, [metadata])
        latest = upstream.inspect(self.root, "community-main", [metadata])
        self.assertEqual(release["prs"][0]["history_presence"], "not_in_target")
        self.assertEqual(latest["prs"][0]["history_presence"], "included")
        self.assertEqual(release["target_base_sha"], base)
        self.assertEqual(release["prs"][0]["semantic_coverage"], "not_reviewed")

    def test_equivalent_backport_does_not_fake_original_merge_ancestry(self):
        base, _, _ = self.start_migration()
        self.git("checkout", "-q", "-b", "community-main", base)
        merged = self.commit("feature.py", "enabled = True\n", "upstream PR")
        self.git("checkout", "-q", "target")
        backport = self.commit("feature.py", "enabled = True\n", "release backport")
        self.assertEqual(self.git("diff", merged, backport), "")
        result = upstream.inspect(
            self.root,
            backport,
            [{"number": 1, "merged": True, "merge_commit_sha": merged}],
        )
        self.assertEqual(result["prs"][0]["history_presence"], "not_in_target")
        self.assertEqual(result["prs"][0]["semantic_coverage"], "not_reviewed")

    def test_unmerged_pr_tentative_merge_sha_is_ignored(self):
        base, _, _ = self.start_migration()
        result = upstream.inspect(
            self.root,
            base,
            [
                {
                    "number": 1,
                    "merged": False,
                    "state": "open",
                    "merge_commit_sha": base,
                },
                {
                    "number": 2,
                    "merged": False,
                    "state": "closed",
                    "merge_commit_sha": base,
                },
            ],
        )
        self.assertEqual(result["counts"], {"not_merged": 2})

    def test_missing_pr_commit_or_metadata_is_unknown(self):
        base, _, _ = self.start_migration()
        result = upstream.inspect(
            self.root,
            base,
            [
                {"number": 1, "merged": True, "merge_commit_sha": "0" * 40},
                {"number": 2, "merged": True},
                {"number": 3},
            ],
        )
        self.assertEqual(result["counts"], {"unknown": 3})
        with self.assertRaises(ValueError):
            upstream.inspect(self.root, "missing-target", [])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            upstream.inspect(self.root, base, [{"number": 1}, {"number": 1}])

    def test_reverted_pr_history_is_not_semantic_coverage(self):
        self.start_migration()
        merged = self.commit("feature.py", "enabled = True\n", "upstream feature")
        reverted = self.commit("feature.py", "enabled = False\n", "revert feature")
        result = upstream.inspect(
            self.root,
            reverted,
            [{"number": 1, "merged": True, "merge_commit_sha": merged}],
        )
        self.assertEqual(result["prs"][0]["history_presence"], "included")
        self.assertEqual(result["prs"][0]["semantic_coverage"], "not_reviewed")

    def test_skip_cannot_use_another_branch_as_coverage_basis(self):
        base, _, data = self.start_migration()
        self.git("checkout", "-q", "-b", "community-main", base)
        other = self.commit(
            "module.py",
            "class A:\n    def run(self):\n        return 1\n",
            "merged on main",
        )
        record = data["records"][0]
        record.update(
            classification="ppu",
            intent="already supported",
            decision="SKIP",
            reason="main PR merged",
            review="complete",
            target_before=base,
            impact=["module.py:A.run"],
            operations=["reviewed main"],
        )
        record["coverage"].update(state="full", evidence=["main PR"], basis_sha=other)
        with self.assertRaisesRegex(ValueError, "coverage basis_sha"):
            ledger.check(data, self.root, base, True)

    def test_changed_target_invalidates_old_coverage_base(self):
        base, _, data = self.start_migration()
        newer = self.commit("release.txt", "new release", "new target")
        data["refs"]["target_base"] = {"ref": "target", "sha": newer}
        record = data["records"][0]
        record.update(
            classification="ppu",
            intent="covered",
            decision="SKIP",
            reason="review from old target",
            review="complete",
            target_before=newer,
            impact=["module.py:A.run"],
            operations=["old review"],
        )
        record["coverage"].update(
            state="full", evidence=["old review"], basis_sha=newer
        )
        with self.assertRaisesRegex(ValueError, "coverage target_base_sha"):
            ledger.check(data, self.root, newer, True)

    def test_pending_target_change_keeps_history_but_warns_stale(self):
        base, _, data = self.start_migration()
        data["records"][0]["coverage"].update(
            state="partial", evidence=["old target analysis"], basis_sha=base
        )
        data["target_upstream_prs"] = {"target_base_sha": base, "prs": []}
        newer = self.commit("release.txt", "new version", "new target")
        data["refs"]["target_base"] = {"ref": "target", "sha": newer}
        self.assertEqual(ledger.check(data, self.root, None, False)["PENDING"], 1)
        warnings = ledger.stale_coverage_warnings(data)
        self.assertEqual(len(warnings), 2)
        self.assertIn("stale coverage", warnings[0])
        self.assertIn("target_upstream_prs", warnings[1])
        data["records"][0]["coverage"].update(
            state="unknown", basis_sha=None, evidence=[]
        )
        self.assertEqual(len(ledger.stale_coverage_warnings(data)), 1)

    def test_partial_migration_requires_both_code_unit_lists(self):
        base = self.commit("module.py", "class A:\n    pass\n", "old community")
        self.git("checkout", "-q", "-b", "ppu")
        complete = "class A:\n    def run(self):\n        return 1\n    def extra(self):\n        return 2\n"
        source = self.commit("module.py", complete, "PPU feature group")
        self.git("checkout", "-q", "-b", "target", base)
        target_base = self.commit(
            "module.py",
            "class A:\n    def run(self):\n        return 1\n",
            "run upstreamed",
        )
        data = ledger.initialize(self.root, base, source, target_base)
        target = self.commit("module.py", complete, "port remaining extra method")
        record = data["records"][0]
        self.finish(record, target, target_base)
        record["decision"] = "ADAPTED"
        record["adjustments"] = ["reused run; added extra only"]
        record["coverage"].update(
            state="partial",
            evidence=[target_base + ":module.py:A.run"],
            remaining=["A.extra"],
        )
        with self.assertRaisesRegex(ValueError, "code-unit lists"):
            ledger.check(data, self.root, target, True)
        record["code_split"] = {
            "basis_sha": base,
            "community_covered": [
                {
                    "id": "C1",
                    "source_file": "module.py",
                    "symbols": ["A.run"],
                    "target_file": "module.py",
                    "target_symbols": ["A.run"],
                }
            ],
            "retain": [
                {
                    "id": "R1",
                    "source_file": "module.py",
                    "symbols": ["A.extra"],
                    "reason": "absent from target",
                }
            ],
        }
        with self.assertRaisesRegex(ValueError, "code_split basis_sha"):
            ledger.check(data, self.root, target, True)
        record["code_split"]["basis_sha"] = target_base
        self.assertEqual(ledger.check(data, self.root, target, True)["ADAPTED"], 1)


if __name__ == "__main__":
    unittest.main()
