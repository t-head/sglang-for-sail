// SPDX-License-Identifier: Apache-2.0
const assert = require("node:assert/strict");
const {test} = require("node:test");
const approved = require("./ppu_approval.cjs");
const requireFastSanity = require("./ppu_pr_result.cjs");

const head = "a".repeat(40);
const other = "b".repeat(40);
const yuheSha = "c".repeat(40);
const base = "release/v0.5.23";

function fixture(label = "run-ci-ppu") {
  const repo = {owner: "t-head", repo: "sglang-for-sail", defaultBranch: "main"};
  const pr = {
    number: 268, state: "open", draft: false, base: {ref: base},
    head: {sha: head, ref: "feature/topic", repo: {id: 123, owner: {login: repo.owner}, name: repo.repo}},
    labels: [{name: "run-ci"}, {name: label}],
  };
  const context = {
    repo: {owner: repo.owner, repo: repo.repo},
    eventName: "pull_request_target", actor: "github-actions[bot]",
    ref: "refs/heads/" + base, sha: other,
    payload: {action: "synchronize", repository: {id: 123}, pull_request: structuredClone(pr)},
  };
  const timeline = [{
    event: "labeled", label: {name: label}, actor: {login: "maintainer"},
    created_at: "2026-11-20T10:01:00Z",
  }];
  const change = {after: head, ref: "refs/heads/feature/topic", timestamp: "2026-11-20T10:00:00Z"};
  const branchActivity = {
    "refs/heads/sail_ci_cd": {after: other, ref: "refs/heads/sail_ci_cd", actor: {login: "maintainer"}},
    "refs/heads/yuhe/v1.0.0": {after: yuheSha, ref: "refs/heads/yuhe/v1.0.0", actor: {login: "maintainer"}},
  };
  const github = {
    rest: {
      pulls: {get: async () => ({data: pr})},
      issues: {listEventsForTimeline: Symbol("timeline")},
      repos: {
        get: async () => ({data: {default_branch: repo.defaultBranch}}),
        getCollaboratorPermissionLevel: async ({username}) => ({
          data: {permission: username === "maintainer" ? "maintain" : "triage"},
        }),
        getBranch: async ({branch}) => ({
          data: {
            name: branch,
            protected: branch === base || branch === "release/v1.0.0",
            commit: {sha: branch === "yuhe/v1.0.0" ? yuheSha : head},
          },
        }),
        listBranches: Symbol("listBranches"),
      },
    },
    paginate: async (route, args) => {
      if (route === github.rest.repos.listBranches) {
        assert.deepEqual(args, {owner: repo.owner, repo: repo.repo, protected: true, per_page: 100});
        return [
          {name: "main", protected: true, commit: {sha: other}},
          {name: base, protected: true, commit: {sha: head}},
          {name: "release/v1.0.0", protected: true, commit: {sha: head}},
        ];
      }
      assert.equal(route, github.rest.issues.listEventsForTimeline);
      assert.deepEqual(args, {owner: repo.owner, repo: repo.repo, issue_number: 268, per_page: 100});
      return timeline;
    },
    request: async (route, args) => {
      assert.equal(route, "GET /repos/{owner}/{repo}/activity");
      if (args.owner === repo.owner && args.repo === repo.repo && branchActivity[args.ref]) {
        return {data: [branchActivity[args.ref]]};
      }
      assert.deepEqual(args, {
        owner: repo.owner, repo: repo.repo, ref: "refs/heads/feature/topic",
        direction: "desc", per_page: 1,
      });
      return {data: [change]};
    },
  };
  return {github, context, label, pr, timeline, change, repo, branchActivity};
}

test("the same approved SHA works for either label notification and bot reruns", async () => {
  for (const label of ["run-ci-ppu", "run-ci-ppu-nightly"]) {
    const f = fixture(label);
    for (const notification of ["run-ci", label]) {
      f.context.payload.action = "labeled";
      f.context.payload.label = {name: notification};
      assert.equal(await approved(f), true);
    }
    f.context.payload.action = "synchronize";
    delete f.context.payload.label;
    assert.equal(await approved(f), true);
  }
});

test("maintainer-approved fork PRs work for both label workflows and stay SHA-pinned", async () => {
  for (const label of ["run-ci-ppu", "run-ci-ppu-nightly"]) {
    const f = fixture(label);
    const activityCalls = [];
    const writeCalls = [];
    f.pr.head.repo = {id: 456, owner: {login: "contributor"}, name: "sglang-fork"};
    f.context.payload.pull_request.head.repo = structuredClone(f.pr.head.repo);
    f.github.request = async (route, args) => {
      assert.equal(route, "GET /repos/{owner}/{repo}/activity");
      activityCalls.push({owner: args.owner, repo: args.repo, ref: args.ref});
      return {data: [f.change]};
    };
    f.github.rest.repos.getCollaboratorPermissionLevel = async ({owner, repo, username}) => {
      writeCalls.push({owner, repo, username});
      return {data: {permission: username === "maintainer" ? "maintain" : "triage"}};
    };

    assert.deepEqual(await approved.resolve(f), {approved: true, sha: head});
    assert.deepEqual(activityCalls, [
      {owner: "contributor", repo: "sglang-fork", ref: "refs/heads/feature/topic"},
    ]);
    assert.deepEqual(writeCalls, [
      {owner: "t-head", repo: "sglang-for-sail", username: "maintainer"},
    ]);

    // Pushing a new commit (or force-pushing) after labeling invalidates approval.
    f.change.timestamp = "2026-11-20T10:02:00Z";
    assert.deepEqual(await approved.resolve(f), {approved: false, sha: ""});
    f.change.timestamp = "2026-11-20T10:00:00Z";

    // Missing/empty fork activity or 404 fails closed without throwing.
    f.github.request = async () => ({data: []});
    assert.deepEqual(await approved.resolve(f), {approved: false, sha: ""});
    f.github.request = async () => {
      const err = new Error("Not Found");
      err.status = 404;
      throw err;
    };
    assert.deepEqual(await approved.resolve(f), {approved: false, sha: ""});

    // Non-writer labeler on base repo is denied.
    f.github.request = async () => ({data: [f.change]});
    f.timeline[0].actor.login = "contributor";
    assert.deepEqual(await approved.resolve(f), {approved: false, sha: ""});

    // Deleted fork repo (head.repo = null) fails closed.
    f.timeline[0].actor.login = "maintainer";
    f.pr.head.repo = null;
    assert.deepEqual(await approved.resolve(f), {approved: false, sha: ""});
  }
});

test("an unrequested yuhe check is optional; release and revoked requests stay required", async () => {
  for (const branch of [base, "yuhe/v1.0.0"]) {
    const f = fixture();
    f.pr.base.ref = f.context.payload.pull_request.base.ref = branch;
    f.pr.labels = f.context.payload.pull_request.labels = [];
    for (const action of ["opened", "synchronize", "reopened", "ready_for_review", "labeled"]) {
      f.context.payload.action = action;
      const result = await approved.resolve(f);
      assert.equal(result.approved, false);
      assert.equal(result.skipOptionalCheck === true, branch === "yuhe/v1.0.0");
    }
    for (const action of ["unlabeled", "converted_to_draft"]) {
      f.context.payload.action = action;
      assert.deepEqual(await approved.resolve(f), {approved: false, sha: ""});
    }
    f.context.payload.action = "synchronize";
    f.pr.labels = [{name: "run-ci"}, {name: f.label}];
    assert.deepEqual(await approved.resolve(f), {approved: true, sha: head}); // Approved since the event.
    f.pr.labels = [];
    f.context.payload.pull_request.labels = [{name: "run-ci"}, {name: f.label}];
    assert.deepEqual(await approved.resolve(f), {approved: false, sha: ""}); // Withdrawn since the event.
    f.pr.labels = f.context.payload.pull_request.labels;
    f.change.timestamp = f.timeline[0].created_at;
    assert.deepEqual(await approved.resolve(f), {approved: false, sha: ""}); // Stale approval.
  }
});

function fastSanityFixture(branch = base) {
  const f = fixture();
  const target = branch === base ? "release-v0523" : "yuhe-v100";
  f.pr.base.ref = f.context.payload.pull_request.base.ref = branch;
  f.context.runId = 821;
  const jobs = ["pr-test-ppu-finish", "ppu-fast-sanity / plan", "ppu-fast-sanity / Fast Sanity (qwen)", "ppu-build"]
    .map(name => ({
      name: target + " / " + name, run_attempt: 1, head_sha: other, status: "completed", conclusion: "success",
    }));
  jobs[2].steps = ["Fetch exact candidate bundle", "Upload checked public evidence"]
    .map(name => ({name, conclusion: "success"}));
  jobs[3].steps = [{name: "Upload PPU wheels", conclusion: "success"}];
  const artifacts = [{name: "ppu-wheels-821-1-" + head, expired: false, size_in_bytes: 2048}];
  f.github.rest.actions = {
    listJobsForWorkflowRun: Symbol("run jobs"),
    listWorkflowRunArtifacts: Symbol("run artifacts"),
  };
  const paginate = f.github.paginate;
  f.github.paginate = async (route, args) => {
    if (route === f.github.rest.actions.listWorkflowRunArtifacts) {
      const {name, ...options} = args;
      assert.deepEqual(options, {...f.context.repo, run_id: 821, per_page: 100});
      assert.equal(typeof name, "string");
      return artifacts.filter(artifact => artifact.name === name);
    }
    if (route !== f.github.rest.actions.listJobsForWorkflowRun) return paginate(route, args);
    assert.deepEqual(args, {...f.context.repo, run_id: 821, filter: "all", per_page: 100});
    return jobs;
  };
  f.needs = {
    "pr-gate": {result: "success", outputs: {approved: "true", sha: head}},
    [target]: {result: "success"},
  };
  return {...f, jobs, artifacts, target};
}

test("both target branches require successful PPU execution on the approved head", async () => {
  for (const branch of [base, "yuhe/v1.0.0"]) {
    await requireFastSanity(fastSanityFixture(branch));
  }
});

test("retrying a failed model consumes the wheel from the build's original attempt", async () => {
  const f = fastSanityFixture();
  const [summary, plan, model, build] = f.jobs;
  // Times from Actions run 44685005679. Carried jobs have new IDs and identical execution intervals.
  Object.assign(build, {id: 136117838321, started_at: "2026-11-20T06:07:51Z", completed_at: "2026-11-20T06:16:57Z"});
  Object.assign(plan, {started_at: "2026-11-20T06:21:19Z", completed_at: "2026-11-20T06:21:25Z"});
  Object.assign(model, {started_at: "2026-11-20T07:20:52Z", completed_at: "2026-11-20T07:29:33Z", conclusion: "failure"});
  Object.assign(summary, {started_at: "2026-11-20T07:54:57Z", completed_at: "2026-11-20T07:55:00Z", conclusion: "failure"});
  const passedModel = {
    ...model, name: f.target + " / ppu-fast-sanity / Fast Sanity (deepseek-v4-flash)",
    started_at: "2026-11-20T06:28:13Z", completed_at: "2026-11-20T06:38:08Z", conclusion: "success",
  };
  f.jobs.push(passedModel);
  f.jobs.unshift(
    {...summary, run_attempt: 2, started_at: "2026-11-20T08:49:49Z", completed_at: "2026-11-20T08:49:51Z", conclusion: "success"},
    {...model, run_attempt: 2, started_at: "2026-11-20T08:39:21Z", completed_at: "2026-11-20T08:48:11Z", conclusion: "success"},
    {...plan, run_attempt: 2},
    {...build, id: 136173070353, run_attempt: 2},
    {...passedModel, run_attempt: 2},
  );
  await requireFastSanity(f);
  // Rerunning only the top-level report keeps all delegated results and the same wheel.
  f.jobs.push(...f.jobs.filter(job => job.run_attempt === 2).map(job => ({...job, run_attempt: 3})));
  await requireFastSanity(f);
});

test("a fresh build requires matching wheels and fresh model results for its plan", async () => {
  const f = fastSanityFixture();
  for (const job of f.jobs) {
    job.started_at = "2026-11-20T06:07:51Z";
    job.completed_at = "2026-11-20T06:16:57Z";
  }
  const oldModel = {...f.jobs[2], name: f.jobs[2].name.replace("qwen", "minimax")};
  f.jobs.push(oldModel);
  const rerun = f.jobs.map(job => ({
    ...job, run_attempt: 2, started_at: "2026-11-20T08:39:21Z", completed_at: "2026-11-20T08:48:11Z",
  }));
  f.jobs.push(...rerun);
  await assert.rejects(requireFastSanity(f), /wheel bundle is missing/);
  f.artifacts.push({...f.artifacts[0], name: "ppu-wheels-821-2-" + head});
  await requireFastSanity(f);
  Object.assign(rerun.at(-1), {started_at: oldModel.started_at, completed_at: oldModel.completed_at});
  await assert.rejects(requireFastSanity(f), /every Fast Sanity model/);
});

test("approval and the delegated job alone cannot satisfy the release check", async () => {
  for (const change of [
    f => { f.needs["pr-gate"].outputs.approved = "false"; },
    f => { f.needs["pr-gate"].result = "cancelled"; },
    f => { f.needs["pr-gate"].outputs.sha = other; },
    f => { f.needs[f.target].result = "skipped"; },
    f => { f.jobs.length = 0; },
    f => { f.jobs.splice(2, 1); },
    f => { f.jobs[0].conclusion = "skipped"; },
    f => { f.jobs[3].conclusion = "skipped"; },
    f => { f.jobs[3].steps[0].conclusion = "skipped"; },
    f => { f.artifacts.length = 0; },
    f => { f.artifacts[0].name = "ppu-wheels-821-1-" + other; },
    f => { f.artifacts[0].name = "ppu-wheels-821-2-" + head; },
    f => { f.artifacts[0].expired = true; },
    f => { f.artifacts[0].size_in_bytes = 0; },
    f => { f.jobs[2].steps[0].conclusion = "skipped"; },
    f => { f.jobs[2].steps[1].conclusion = "skipped"; },
    f => { f.jobs[2].name = f.jobs[2].name.replace("Fast Sanity", "Model Sanity"); },
    f => { f.jobs.push({...f.jobs[2], run_attempt: 2, conclusion: "failure"}); },
    f => { f.jobs[2].run_attempt = 2; }, // The old summary predates the new model job.
    f => { f.jobs.push({...f.jobs[0], name: f.target + " / ppu-fast-sanity", run_attempt: 2, conclusion: "skipped"}); },
    f => { f.jobs[0].run_attempt = f.jobs[1].run_attempt = 2; }, // A new plan has no model results yet.
    f => { f.pr.head.sha = other; },
    f => { f.pr.labels.pop(); },
  ]) {
    const f = fastSanityFixture();
    change(f);
    await assert.rejects(requireFastSanity(f), /PPU|Fast Sanity|approval/, String(change));
  }
});

test("reject stale, missing and withdrawn approvals", async () => {
  for (const update of [
    f => { f.pr.draft = true; },
    f => { f.pr.state = "closed"; },
    f => { f.pr.labels.shift(); },
    f => { f.pr.labels.pop(); },
    f => { f.pr.base.ref = "main"; },
    f => { f.pr.head.sha = other; },
    f => { f.pr.head.repo.id++; },
    f => { f.ref = other; },
    f => { f.change.after = other; },
    f => { f.change.timestamp = f.timeline[0].created_at; },
    f => { f.change.timestamp = "2026-11-21T00:00:00Z"; },
    f => { f.change.timestamp = null; },
    f => { f.timeline.length = 0; },
    f => { f.timeline.push({...f.timeline[0], event: "unlabeled"}); },
    f => { f.timeline.push({...f.timeline[0], actor: {login: "contributor"}}); },
  ]) {
    const f = fixture();
    update(f);
    assert.equal(await approved(f), false, String(update));
  }
});

test("same-repo dispatch binds the tested revision and checks both actors", async t => {
  const f = fixture();
  f.context.eventName = "workflow_dispatch";
  f.context.ref = "refs/heads/validation";
  f.context.sha = head;
  f.context.actor = "maintainer";
  // Node's test runner isolates this file's process.
  const original = process.env.GITHUB_TRIGGERING_ACTOR;
  t.after(() => {
    if (original === undefined) delete process.env.GITHUB_TRIGGERING_ACTOR;
    else process.env.GITHUB_TRIGGERING_ACTOR = original;
  });
  process.env.GITHUB_TRIGGERING_ACTOR = "maintainer";
  assert.equal(await approved({...f, ref: head}), true);
  assert.equal(await approved({...f, ref: other}), false);
  process.env.GITHUB_TRIGGERING_ACTOR = "contributor";
  assert.equal(await approved(f), false);
  f.context.actor = "contributor";
  process.env.GITHUB_TRIGGERING_ACTOR = "maintainer";
  assert.equal(await approved(f), false);
});

test("a protected main schedule can call the protected release workflow", async t => {
  const f = fixture();
  const original = process.env.GITHUB_REF_PROTECTED;
  t.after(() => {
    if (original === undefined) delete process.env.GITHUB_REF_PROTECTED;
    else process.env.GITHUB_REF_PROTECTED = original;
  });
  f.context.ref = "refs/heads/main";
  f.context.eventName = "schedule";
  process.env.GITHUB_REF_PROTECTED = "true";
  assert.equal(await approved({...f, ref: base}), true);
  assert.equal(await approved({...f, ref: "release/unprotected"}), false);
  f.context.eventName = "push";
  assert.equal(await approved({...f, ref: f.context.sha}), true);
  assert.equal(await approved({...f, ref: base}), false);
  process.env.GITHUB_REF_PROTECTED = "false";
  assert.equal(await approved(f), false);
  for (const eventName of ["issue_comment", "pull_request", "workflow_run"]) {
    f.context.eventName = eventName;
    assert.equal(await approved(f), false);
  }
});

test("default branch sail_ci_cd schedule can call yuhe/v1.0.0 and protected release branches", async t => {
  const f = fixture("run-ci-ppu-nightly");
  const original = process.env.GITHUB_REF_PROTECTED;
  t.after(() => {
    if (original === undefined) delete process.env.GITHUB_REF_PROTECTED;
    else process.env.GITHUB_REF_PROTECTED = original;
  });
  f.repo.defaultBranch = "sail_ci_cd";
  f.context.ref = "refs/heads/sail_ci_cd";
  f.context.eventName = "schedule";
  process.env.GITHUB_REF_PROTECTED = "false";
  assert.equal(await approved({...f, ref: "yuhe/v1.0.0"}), true);
  assert.equal(await approved({...f, ref: "release/v1.0.0"}), true);
  assert.equal(await approved({...f, ref: "release/unprotected"}), false);
  assert.equal(await approved({...f, ref: "feature/untrusted"}), false);

  // Reject when sail_ci_cd is not the repository default branch.
  f.repo.defaultBranch = "main";
  assert.equal(await approved({...f, ref: "yuhe/v1.0.0"}), false);
  f.repo.defaultBranch = "sail_ci_cd";

  // Reject when yuhe/v1.0.0 activity actor lacks write access or SHA mismatches.
  f.branchActivity["refs/heads/yuhe/v1.0.0"].actor.login = "contributor";
  assert.equal(await approved({...f, ref: "yuhe/v1.0.0"}), false);
  f.branchActivity["refs/heads/yuhe/v1.0.0"].actor.login = "maintainer";
  f.branchActivity["refs/heads/yuhe/v1.0.0"].after = head;
  assert.equal(await approved({...f, ref: "yuhe/v1.0.0"}), false);
});

test("resolve returns pinned target SHA and admits nested schedule SHA but rejects unapproved SHA", async t => {
  const f = fixture("run-ci-ppu");
  const original = process.env.GITHUB_REF_PROTECTED;
  t.after(() => {
    if (original === undefined) delete process.env.GITHUB_REF_PROTECTED;
    else process.env.GITHUB_REF_PROTECTED = original;
  });
  f.repo.defaultBranch = "sail_ci_cd";
  f.context.ref = "refs/heads/sail_ci_cd";
  f.context.eventName = "schedule";
  process.env.GITHUB_REF_PROTECTED = "false";

  const parentGate = await approved.resolve({...f, ref: "yuhe/v1.0.0"});
  assert.deepEqual(parentGate, {approved: true, sha: yuheSha});
  const nestedGate = await approved.resolve({...f, ref: parentGate.sha});
  assert.deepEqual(nestedGate, {approved: true, sha: yuheSha});

  const releaseGate = await approved.resolve({...f, ref: "release/v1.0.0"});
  assert.deepEqual(releaseGate, {approved: true, sha: head});
  const nestedReleaseGate = await approved.resolve({...f, ref: releaseGate.sha});
  assert.deepEqual(nestedReleaseGate, {approved: true, sha: head});

  // Unapproved 40-char SHA must fail closed.
  const unapprovedSha = "d".repeat(40);
  assert.deepEqual(await approved.resolve({...f, ref: unapprovedSha}), {approved: false, sha: ""});
});

test("default branch sail_ci_cd workflow_dispatch can target yuhe/v1.0.0 and protected release branches for write actors only", async t => {
  const f = fixture("run-ci-ppu-nightly");
  const original = process.env.GITHUB_TRIGGERING_ACTOR;
  t.after(() => {
    if (original === undefined) delete process.env.GITHUB_TRIGGERING_ACTOR;
    else process.env.GITHUB_TRIGGERING_ACTOR = original;
  });
  f.repo.defaultBranch = "sail_ci_cd";
  f.context.eventName = "workflow_dispatch";
  f.context.ref = "refs/heads/sail_ci_cd";
  f.context.sha = other;
  f.context.actor = "maintainer";
  process.env.GITHUB_TRIGGERING_ACTOR = "maintainer";

  assert.deepEqual(await approved.resolve({...f, ref: "yuhe/v1.0.0"}), {
    approved: true,
    sha: yuheSha,
  });
  assert.deepEqual(await approved.resolve({...f, ref: "release/v1.0.0"}), {
    approved: true,
    sha: head,
  });
  assert.deepEqual(await approved.resolve({...f, ref: "feature/untrusted"}), {
    approved: false,
    sha: "",
  });
  assert.deepEqual(await approved.resolve({...f, ref: "d".repeat(40)}), {
    approved: false,
    sha: "",
  });

  // Non-default branch dispatch cannot cross-target yuhe/v1.0.0.
  f.repo.defaultBranch = "main";
  assert.deepEqual(await approved.resolve({...f, ref: "yuhe/v1.0.0"}), {
    approved: false,
    sha: "",
  });
  f.repo.defaultBranch = "sail_ci_cd";

  // Unauthorized actor or triggering actor must be denied.
  f.context.actor = "contributor";
  assert.deepEqual(await approved.resolve({...f, ref: "yuhe/v1.0.0"}), {
    approved: false,
    sha: "",
  });
  f.context.actor = "maintainer";
  process.env.GITHUB_TRIGGERING_ACTOR = "contributor";
  assert.deepEqual(await approved.resolve({...f, ref: "yuhe/v1.0.0"}), {
    approved: false,
    sha: "",
  });
});

test("allowlisted GitHub App bots can dispatch trusted release targets without weakening PR or feature-branch gates", async t => {
  const f = fixture("run-ci-ppu-nightly");
  const originalActor = process.env.GITHUB_TRIGGERING_ACTOR;
  const originalBots = process.env.PPU_CI_ALLOWED_DISPATCH_BOTS;
  t.after(() => {
    if (originalActor === undefined) delete process.env.GITHUB_TRIGGERING_ACTOR;
    else process.env.GITHUB_TRIGGERING_ACTOR = originalActor;
    if (originalBots === undefined) delete process.env.PPU_CI_ALLOWED_DISPATCH_BOTS;
    else process.env.PPU_CI_ALLOWED_DISPATCH_BOTS = originalBots;
  });

  const bot = "trusted-devops-app[bot]";
  // GitHub collaborator permission API returns none/404 for GitHub App bots.
  f.github.rest.repos.getCollaboratorPermissionLevel = async ({username}) => ({
    data: {permission: username === "maintainer" ? "maintain" : "none"},
  });
  f.repo.defaultBranch = "sail_ci_cd";
  f.context.eventName = "workflow_dispatch";
  f.context.ref = "refs/heads/sail_ci_cd";
  f.context.sha = other;
  f.context.actor = bot;
  process.env.GITHUB_TRIGGERING_ACTOR = bot;
  delete process.env.PPU_CI_ALLOWED_DISPATCH_BOTS;

  // Denied by default when not allowlisted.
  assert.deepEqual(await approved.resolve({...f, ref: "yuhe/v1.0.0"}), {
    approved: false,
    sha: "",
  });

  // Allowed for cross-branch trusted target when explicitly allowlisted (case-insensitive).
  const allowlist = ` ${bot.toUpperCase()} , contributor , * `;
  assert.deepEqual(
    await approved.resolve({...f, ref: "yuhe/v1.0.0", allowedDispatchBots: allowlist}),
    {approved: true, sha: yuheSha},
  );
  assert.deepEqual(
    await approved.resolve({...f, ref: "release/v1.0.0", allowedDispatchBots: allowlist}),
    {approved: true, sha: head},
  );
  // Still denied for untrusted cross-branch target.
  assert.deepEqual(
    await approved.resolve({...f, ref: "feature/untrusted", allowedDispatchBots: allowlist}),
    {approved: false, sha: ""},
  );

  // Maintainer rerun of bot-triggered dispatch succeeds; non-write user rerun fails.
  process.env.GITHUB_TRIGGERING_ACTOR = "maintainer";
  assert.deepEqual(
    await approved.resolve({...f, ref: "yuhe/v1.0.0", allowedDispatchBots: allowlist}),
    {approved: true, sha: yuheSha},
  );
  process.env.GITHUB_TRIGGERING_ACTOR = "contributor";
  assert.deepEqual(
    await approved.resolve({...f, ref: "yuhe/v1.0.0", allowedDispatchBots: allowlist}),
    {approved: false, sha: ""},
  );

  // Non-[bot] entries or wildcards in allowlist must never grant access to humans or unknown bots.
  f.context.actor = "contributor";
  process.env.GITHUB_TRIGGERING_ACTOR = "contributor";
  assert.deepEqual(
    await approved.resolve({...f, ref: "yuhe/v1.0.0", allowedDispatchBots: allowlist}),
    {approved: false, sha: ""},
  );
  f.context.actor = "unknown-app[bot]";
  process.env.GITHUB_TRIGGERING_ACTOR = "unknown-app[bot]";
  assert.deepEqual(
    await approved.resolve({...f, ref: "yuhe/v1.0.0", allowedDispatchBots: allowlist}),
    {approved: false, sha: ""},
  );

  // Allowlisted bot dispatching directly (ownRef) is permitted on trusted yuhe/v1.0.0
  // and protected release/* branches, but denied on non-release branches even if protected.
  f.context.actor = bot;
  process.env.GITHUB_TRIGGERING_ACTOR = bot;
  f.context.ref = "refs/heads/yuhe/v1.0.0";
  f.context.sha = yuheSha;
  assert.deepEqual(
    await approved.resolve({...f, ref: "", allowedDispatchBots: allowlist}),
    {approved: true, sha: yuheSha},
  );
  f.context.ref = "refs/heads/release/v1.0.0";
  f.context.sha = head;
  assert.deepEqual(
    await approved.resolve({...f, ref: "", allowedDispatchBots: allowlist}),
    {approved: true, sha: head},
  );
  f.context.ref = "refs/heads/feature/untrusted";
  f.context.sha = head;
  assert.deepEqual(
    await approved.resolve({...f, ref: "", allowedDispatchBots: allowlist}),
    {approved: false, sha: ""},
  );
  const origGetBranch = f.github.rest.repos.getBranch;
  f.github.rest.repos.getBranch = async ({branch}) => ({
    data: {name: branch, protected: true, commit: {sha: head}},
  });
  f.context.ref = "refs/heads/feature/protected-topic";
  f.context.sha = head;
  assert.deepEqual(
    await approved.resolve({...f, ref: "", allowedDispatchBots: allowlist}),
    {approved: false, sha: ""},
  );
  f.context.actor = "maintainer";
  process.env.GITHUB_TRIGGERING_ACTOR = bot;
  assert.deepEqual(
    await approved.resolve({...f, ref: "", allowedDispatchBots: allowlist}),
    {approved: false, sha: ""},
  );
  f.github.rest.repos.getBranch = origGetBranch;

  // pull_request_target must never accept bot labels even when bot is in dispatch allowlist.
  const prFixture = fixture("run-ci-ppu");
  prFixture.timeline[0].actor.login = bot;
  prFixture.github.rest.repos.getCollaboratorPermissionLevel = async () => ({
    data: {permission: "none"},
  });
  process.env.PPU_CI_ALLOWED_DISPATCH_BOTS = bot;
  assert.deepEqual(await approved.resolve({...prFixture, allowedDispatchBots: bot}), {
    approved: false,
    sha: "",
  });
});
