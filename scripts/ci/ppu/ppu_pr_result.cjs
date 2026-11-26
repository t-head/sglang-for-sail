// SPDX-License-Identifier: Apache-2.0
const approval = require("./ppu_approval.cjs");

const targets = new Map([
  ["release/v0.5.23", "release-v0523"],
  ["yuhe/v1.0.0", "yuhe-v100"],
]);

module.exports = async function requireFastSanity({github, context, needs}) {
  const pr = context.payload.pull_request;
  const target = targets.get(pr.base.ref);
  const gate = needs["pr-gate"];
  if (gate.result !== "success" || gate.outputs.approved !== "true" ||
      gate.outputs.sha !== pr.head.sha || !/^[0-9a-f]{40}$/.test(pr.head.sha)) {
    throw new Error("Fast Sanity requires run-ci and fresh run-ci-ppu approval for the current PR head.");
  }
  if (!target || needs[target]?.result !== "success") {
    throw new Error("The target branch's PPU workflow did not succeed.");
  }

  // The version workflows do not export a workflow_call result. On a partial
  // rerun, successful upstream jobs may belong to an earlier attempt of this run.
  const jobs = await github.paginate(github.rest.actions.listJobsForWorkflowRun, {
    ...context.repo, run_id: context.runId, filter: "all", per_page: 100,
  });
  const latestByName = new Map();
  for (const job of jobs) {
    if (job.name !== target && !job.name.startsWith(target + " / ")) continue;
    if (!latestByName.has(job.name) || job.run_attempt > latestByName.get(job.name).run_attempt) {
      latestByName.set(job.name, job);
    }
  }
  const passed = job => job?.status === "completed" && job.conclusion === "success" &&
    job.head_sha === pr.head.sha;
  const sanity = target + " / ppu-fast-sanity";
  const plan = latestByName.get(sanity + " / plan");
  const summary = latestByName.get(target + " / pr-test-ppu-finish");
  // A skipped reusable workflow is recorded under its bare name, with no child jobs.
  if (!passed(summary) || !passed(plan) || summary.run_attempt < plan.run_attempt ||
      [target, sanity].some(name => latestByName.get(name)?.run_attempt >= plan.run_attempt)) {
    throw new Error("The delegated workflow did not complete Fast Sanity for this PR head.");
  }
  const models = [...latestByName.values()].filter(job =>
    job.name.startsWith(sanity + " / Fast Sanity (") && job.run_attempt >= plan.run_attempt
  );
  const ranModel = job => passed(job) && job.run_attempt <= summary.run_attempt &&
    job.steps?.some(step => step.name === "Orchestrate PPU Pod with private logs" && step.conclusion === "success");
  if (!models.length || !models.every(ranModel)) {
    throw new Error("Expected successful PPU execution for every Fast Sanity model on this PR head.");
  }

  const current = await approval.resolve({github, context, label: "run-ci-ppu", ref: pr.head.sha});
  if (!current.approved || current.sha !== pr.head.sha) {
    throw new Error("PR approval changed while Fast Sanity was running.");
  }
};
