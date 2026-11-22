// SPDX-License-Identifier: Apache-2.0
// This module runs on a GitHub-hosted runner from the trusted workflow revision.

async function canWrite(github, repo, username) {
  if (!username) return false;
  try {
    const {data} = await github.rest.repos.getCollaboratorPermissionLevel({
      ...repo, username,
    });
    return ["write", "maintain", "admin"].includes(data.permission);
  } catch (error) {
    if (error.status === 404) return false;
    throw error;
  }
}

async function trustedBranchHead(github, repo, branch, sha) {
  if (!sha) return false;
  const branchRef = "refs/heads/" + branch;
  const {data: [change]} = await github.request("GET /repos/{owner}/{repo}/activity", {
    ...repo, ref: branchRef, direction: "desc", per_page: 1,
  });
  if (change?.ref !== branchRef || change.after !== sha) return false;
  return canWrite(github, repo, change.actor?.login);
}

async function approvedPullRequest(github, context, label, ref) {
  const eventPR = context.payload.pull_request;
  if (!eventPR || (ref && ref !== eventPR.head.sha)) return false;
  const {data: pr} = await github.rest.pulls.get({
    ...context.repo, pull_number: eventPR.number,
  });
  const labels = new Set(pr.labels.map(item => item.name));
  if (pr.state !== "open" || pr.draft ||
      pr.base.ref !== eventPR.base.ref || pr.head.sha !== eventPR.head.sha ||
      !pr.head.repo || pr.head.repo.id !== eventPR.head.repo?.id ||
      !labels.has("run-ci") || !labels.has(label)) return false;

  const timeline = await github.paginate(github.rest.issues.listEventsForTimeline, {
    ...context.repo, issue_number: pr.number, per_page: 100,
  });
  const lastLabel = timeline.findLast(item =>
    (item.event === "labeled" || item.event === "unlabeled") &&
    item.label?.name === label
  );
  if (lastLabel?.event !== "labeled") return false;

  // Repository activity timestamps are server receipt times. Commit dates can
  // predate a push or stay unchanged when a branch is force-pushed back.
  const branchRef = "refs/heads/" + pr.head.ref;
  const {data: [change]} = await github.request("GET /repos/{owner}/{repo}/activity", {
    owner: pr.head.repo.owner.login, repo: pr.head.repo.name,
    ref: branchRef, direction: "desc", per_page: 1,
  });
  if (change?.ref !== branchRef || change.after !== pr.head.sha ||
      !(Date.parse(lastLabel.created_at) > Date.parse(change.timestamp))) return false;
  return canWrite(github, context.repo, lastLabel.actor?.login);
}

module.exports = async function ppuApproved({github, context, label, ref = ""}) {
  const {owner, repo} = context.repo;
  if (owner.toLowerCase() !== "t-head" || repo.toLowerCase() !== "sglang-for-sail" ||
      !["run-ci-ppu", "run-ci-ppu-nightly"].includes(label)) return false;

  if (context.eventName === "pull_request_target") {
    return approvedPullRequest(github, context, label, ref);
  }
  if (!context.ref.startsWith("refs/heads/")) return false;
  const branch = context.ref.slice("refs/heads/".length);
  const ownRef = !ref || ref === context.sha || ref === context.ref || ref === branch;
  if (context.eventName === "workflow_dispatch") {
    if (!ownRef || !await canWrite(github, context.repo, context.actor)) return false;
    const rerunner = process.env.GITHUB_TRIGGERING_ACTOR;
    if (!rerunner) return false;
    return rerunner === context.actor || canWrite(github, context.repo, rerunner);
  }
  const isProtectedCaller = process.env.GITHUB_REF_PROTECTED === "true";
  if (context.eventName === "push") {
    if (!ownRef) return false;
    if (isProtectedCaller) return true;
    return branch === "sail_ci_cd" && trustedBranchHead(github, context.repo, branch, context.sha);
  }
  if (context.eventName !== "schedule" || !["main", "sail_ci_cd"].includes(branch)) return false;
  const {data: repository} = await github.rest.repos.get(context.repo);
  if (repository.default_branch !== branch) return false;
  if (!isProtectedCaller &&
      !(branch === "sail_ci_cd" && await trustedBranchHead(github, context.repo, branch, context.sha))) {
    return false;
  }
  if (ownRef) return true;
  // The default-branch dispatcher calls the release workflow with a release branch.
  if (ref !== "yuhe/v1.0.0" && !ref.startsWith("release/")) return false;
  const {data: target} = await github.rest.repos.getBranch({...context.repo, branch: ref});
  if (target.name !== ref) return false;
  if (target.protected === true) return true;
  return ref === "yuhe/v1.0.0" && trustedBranchHead(github, context.repo, ref, target.commit?.sha);
};
