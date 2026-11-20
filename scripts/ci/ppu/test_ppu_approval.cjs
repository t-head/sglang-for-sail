// SPDX-License-Identifier: Apache-2.0
const assert = require("node:assert/strict");
const {test} = require("node:test");
const approved = require("./ppu_approval.cjs");

const head = "a".repeat(40);
const other = "b".repeat(40);
const base = "release/v0.5.23";

function fixture(label = "run-ci-ppu") {
  const repo = {owner: "t-head", repo: "sglang-for-sail"};
  const pr = {
    number: 268, state: "open", draft: false, base: {ref: base},
    head: {sha: head, ref: "feature/topic", repo: {id: 123, owner: {login: "contributor"}, name: "fork"}},
    labels: [{name: "run-ci"}, {name: label}],
  };
  const context = {
    repo, eventName: "pull_request_target", actor: "github-actions[bot]",
    ref: "refs/heads/" + base, sha: other,
    payload: {action: "synchronize", pull_request: structuredClone(pr)},
  };
  const timeline = [{
    event: "labeled", label: {name: label}, actor: {login: "maintainer"},
    created_at: "2026-11-20T10:01:00Z",
  }];
  const change = {after: head, ref: "refs/heads/feature/topic", timestamp: "2026-11-20T10:00:00Z"};
  const github = {
    rest: {
      pulls: {get: async () => ({data: pr})},
      issues: {listEventsForTimeline: Symbol("timeline")},
      repos: {
        getCollaboratorPermissionLevel: async ({username}) => ({
          data: {permission: username === "maintainer" ? "maintain" : "triage"},
        }),
        getBranch: async ({branch}) => ({data: {name: branch, protected: branch === base}}),
      },
    },
    paginate: async (route, args) => {
      assert.equal(route, github.rest.issues.listEventsForTimeline);
      assert.deepEqual(args, {...repo, issue_number: 268, per_page: 100});
      return timeline;
    },
    request: async (route, args) => {
      assert.equal(route, "GET /repos/{owner}/{repo}/activity");
      assert.deepEqual(args, {
        owner: "contributor", repo: "fork", ref: "refs/heads/feature/topic",
        direction: "desc", per_page: 1,
      });
      return {data: [change]};
    },
  };
  return {github, context, label, pr, timeline, change};
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
