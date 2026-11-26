// SPDX-License-Identifier: Apache-2.0
const approval = require("./ppu_approval.cjs");

const targets = new Map([
  ["release/v0.5.23", "release-v0523"],
  ["yuhe/v1.0.0", "yuhe-v100"],
]);

function latestExecutions(jobs, target) {
  const latest = new Map();
  const firstAttempt = new Map();
  for (const job of [...jobs].sort((a, b) => a.run_attempt - b.run_attempt)) {
    if (job.name !== target && !job.name.startsWith(target + " / ")) continue;
    // GitHub copies a successful job into later attempts with a new ID but the
    // same execution interval. Its wheel and model evidence belong to the first one.
    let executedIn = null;
    if (job.started_at && job.completed_at) {
      const interval = JSON.stringify([job.name, job.started_at, job.completed_at]);
      executedIn = firstAttempt.get(interval) ?? job.run_attempt;
      firstAttempt.set(interval, executedIn);
    } else if (job.run_attempt === 1) {
      executedIn = 1;
    }
    latest.set(job.name, {...job, executedIn});
  }
  return latest;
}

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
  const latestByName = latestExecutions(jobs, target);
  const passed = job => job?.status === "completed" && job.conclusion === "success" && Number.isInteger(job.executedIn);
  const stepPassed = (job, name) => job.steps?.some(step => step.name === name && step.conclusion === "success");
  const sanity = target + " / ppu-fast-sanity";
  const build = latestByName.get(target + " / ppu-build");
  const plan = latestByName.get(sanity + " / plan");
  const summary = latestByName.get(target + " / pr-test-ppu-finish");
  // A skipped reusable workflow is recorded under its bare name, with no child jobs.
  if (!passed(build) || !stepPassed(build, "Upload PPU wheels") || !passed(plan) || !passed(summary) ||
      build.executedIn > plan.executedIn || summary.executedIn < plan.executedIn ||
      [target, sanity].some(name => latestByName.get(name)?.run_attempt >= plan.run_attempt)) {
    throw new Error("The delegated workflow did not complete Fast Sanity for this PR head.");
  }
  // A new plan can change the matrix. Validate every model listed since that
  // plan, including jobs GitHub carried forward from an older execution.
  const models = [...latestByName.values()].filter(job =>
    job.name.startsWith(sanity + " / Fast Sanity (") && job.run_attempt >= plan.executedIn
  );
  const ranModel = job => passed(job) && job.executedIn >= plan.executedIn && job.executedIn <= summary.executedIn &&
    stepPassed(job, "Fetch exact candidate bundle") && stepPassed(job, "Upload checked public evidence");
  if (!models.length || !models.every(ranModel)) {
    throw new Error("Expected successful PPU execution for every Fast Sanity model on this PR head.");
  }

  // Both callees upload an immutable wheel bundle named for the checkout SHA
  // and pass its artifact ID to the model jobs. The run's head_sha is event metadata.
  const name = `ppu-wheels-${context.runId}-${build.executedIn}-${pr.head.sha}`;
  const wheels = await github.paginate(github.rest.actions.listWorkflowRunArtifacts, {
    ...context.repo, run_id: context.runId, name, per_page: 100,
  });
  if (!wheels.some(artifact => artifact.name === name && artifact.expired === false && artifact.size_in_bytes > 0)) {
    throw new Error("The Fast Sanity wheel bundle is missing for the approved PR head and build attempt.");
  }

  const current = await approval.resolve({github, context, label: "run-ci-ppu", ref: pr.head.sha});
  if (!current.approved || current.sha !== pr.head.sha) {
    throw new Error("PR approval changed while Fast Sanity was running.");
  }
};
