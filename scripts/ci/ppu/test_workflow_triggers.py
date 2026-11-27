# SPDX-License-Identifier: Apache-2.0
import copy
import json
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
BRANCHES = {"release-v0523": "release/v0.5.23", "yuhe-v100": "yuhe/v1.0.0"}


class WorkflowLoader(yaml.BaseLoader):
    # Keep the Actions key "on" as a string and reject duplicate mapping keys.
    def construct_mapping(self, node, deep=False):
        mapping = super().construct_mapping(node, deep=deep)
        if len(mapping) != len(node.value):
            raise ValueError(f"Duplicate YAML key at {node.start_mark}")
        return mapping


class TestWorkflowTriggers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workflows = {
            name: yaml.load(
                (ROOT / ".github/workflows" / f"{name}-test-ppu.yml").read_text(),
                Loader=WorkflowLoader,
            )
            for name in ("pr", "nightly")
        }

    def assert_events(self, events):
        pr, nightly = events["pr"], events["nightly"]
        for workflow in (pr, nightly):
            self.assertCountEqual(
                workflow["pull_request_target"]["branches"], BRANCHES.values()
            )
        self.assertCountEqual(
            pr["pull_request_target"]["types"],
            "opened synchronize reopened labeled unlabeled ready_for_review converted_to_draft".split(),
        )
        self.assertCountEqual(
            nightly["pull_request_target"]["types"],
            "labeled synchronize reopened ready_for_review".split(),
        )
        self.assertCountEqual(
            pr["workflow_dispatch"]["inputs"],
            "preflight_only model_filter run_model_sanity run_all_tests run_legacy_sanity".split(),
        )
        self.assertCountEqual(
            pr["workflow_call"]["inputs"],
            "legacy_call ref run_all_tests run_legacy_sanity".split(),
        )
        self.assertEqual(
            pr["workflow_call"]["inputs"]["legacy_call"]["default"], "true"
        )
        self.assertCountEqual(
            nightly["workflow_dispatch"]["inputs"],
            "target_ref model_filter image_tag image_digest skip_wheel_build wheel_artifact_run_id".split(),
        )

    def test_live_workflow_events_and_delegation(self):
        self.assert_events(
            {name: workflow["on"] for name, workflow in self.workflows.items()}
        )
        for name, workflow in self.workflows.items():
            jobs = workflow["jobs"]
            self.assertEqual(
                jobs["pr-gate"]["uses"], "./.github/workflows/ppu-approval.yml"
            )
            for job, branch in BRANCHES.items():
                self.assertEqual(
                    jobs[job]["uses"],
                    f"t-head/sglang-for-sail/.github/workflows/{name}-test-ppu.yml@{branch}",
                )
        self.assertIn("pr-test-ppu-finish", self.workflows["pr"]["jobs"])
        expected_secrets = {
            key: f"${{{{ secrets.{key} }}}}"
            for key in (
                "PPU_CI_K8S_RUNTIME_DIGEST",
                "PPU_CI_RECIPE_IMAGE_REPO",
                "PPU_CI_RECIPE_ALLOWED_IMAGE_REPOS",
            )
        }
        for name, workflow in self.workflows.items():
            for job in BRANCHES:
                with self.subTest(workflow=name, job=job):
                    secrets = workflow["jobs"][job]["secrets"]
                    self.assertNotEqual(secrets, "inherit")
                    self.assertEqual(secrets, expected_secrets)
        tiers = json.loads((ROOT / "scripts/ci/opt_in_ci_tiers.json").read_text())
        self.assertEqual(
            tiers["ppu-extra"],
            {
                "label": "run-ci-ppu-extra",
                "workflows": [".github/workflows/pr-test-ppu-extra.yml"],
            },
        )

    def test_missing_value_is_detected_on_its_own_event(self):
        for name, event, prop, required in (
            ("pr", "pull_request_target", "types", "synchronize"),
            ("nightly", "pull_request_target", "types", "synchronize"),
            ("pr", "workflow_dispatch", "inputs", "run_legacy_sanity"),
        ):
            with self.subTest(workflow=name, event=event, required=required):
                events = copy.deepcopy(
                    {key: value["on"] for key, value in self.workflows.items()}
                )
                values = events[name][event][prop]
                if isinstance(values, dict):
                    del values[required]
                else:
                    values.remove(required)
                with self.assertRaises(AssertionError):
                    self.assert_events(events)

    def test_equivalent_yaml_and_duplicate_keys(self):
        compact = "on:\n  pull_request_target:\n    types: [opened, synchronize]\n"
        spaced = "'on' :\n  pull_request_target :\n    types:  [ \"opened\", 'synchronize' ]\n"
        self.assertEqual(
            yaml.load(compact, Loader=WorkflowLoader),
            yaml.load(spaced, Loader=WorkflowLoader),
        )
        with self.assertRaises(ValueError):
            yaml.load(compact + "  pull_request_target: {}\n", Loader=WorkflowLoader)


if __name__ == "__main__":
    unittest.main()
