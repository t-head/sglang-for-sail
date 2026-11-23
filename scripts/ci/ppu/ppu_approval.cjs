// SPDX-License-Identifier: Apache-2.0
// This module runs on a GitHub-hosted runner from the trusted workflow revision.

const DENIED = Object.freeze({approved: false, sha: ""});
const BOT_LOGIN_RE = /^[a-z0-9](?:[a-z0-9-]{0,37}[a-z0-9])?\[bot\]$/;

function isSha(value) {
  return typeof value === "string" && /^[0-9a-f]{40}$/.test(value);
}

function parseAllowedDispatchBots(raw) {
  const bots = new Set();
  if (typeof raw !== "string" || !raw.trim()) return bots;
  for (const entry of raw.split(",")) {
    const normalized = entry.trim().toLowerCase();
    if (BOT_LOGIN_RE.test(normalized)) {
      bots.add(normalized);
    }
  }
  return bots;
}

async function canDispatch(github, repo, username, allowedBots) {
  if (!username) return {allowed: false, isBot: false};
  if (allowedBots.has(String(username).toLowerCase())) {
    return {allowed: true, isBot: true};
  }
  if (await canWrite(github, repo, username)) {
    return {allowed: true, isBot: false};
  }
  return {allowed: false, isBot: false};
}

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

async function trustedBranchHead(github, repo, branch, sha, perPage = 1) {
  if (!isSha(sha)) return false;
  const branchRef = "refs/heads/" + branch;
  const {data: changes} = await github.request("GET /repos/{owner}/{repo}/activity", {
    ...repo, ref: branchRef, direction: "desc", per_page: perPage,
  });
  if (!Array.isArray(changes)) return false;
  const match = perPage === 1
    ? (changes[0]?.ref === branchRef && changes[0].after === sha ? changes[0] : null)
    : changes.find(item => item?.ref === branchRef && item.after === sha);
  if (!match) return false;
  return canWrite(github, repo, match.actor?.login);
}

async function approvedPullRequest(github, context, label, ref) {
  const eventPR = context.payload.pull_request;
  if (!eventPR || (ref && ref !== eventPR.head.sha)) return DENIED;
  const {data: pr} = await github.rest.pulls.get({
    ...context.repo, pull_number: eventPR.number,
  });
  const labels = new Set(pr.labels.map(item => item.name));
  if (pr.state !== "open" || pr.draft || !isSha(pr.head.sha) ||
      pr.base.ref !== eventPR.base.ref || pr.head.sha !== eventPR.head.sha ||
      !pr.head.repo || pr.head.repo.id !== eventPR.head.repo?.id ||
      !labels.has("run-ci") || !labels.has(label)) return DENIED;

  const timeline = await github.paginate(github.rest.issues.listEventsForTimeline, {
    ...context.repo, issue_number: pr.number, per_page: 100,
  });
  const lastLabel = timeline.findLast(item =>
    (item.event === "labeled" || item.event === "unlabeled") &&
    item.label?.name === label
  );
  if (lastLabel?.event !== "labeled") return DENIED;

  // Repository activity timestamps are server receipt times. Commit dates can
  // predate a push or stay unchanged when a branch is force-pushed back.
  const branchRef = "refs/heads/" + pr.head.ref;
  const {data: [change]} = await github.request("GET /repos/{owner}/{repo}/activity", {
    owner: pr.head.repo.owner.login, repo: pr.head.repo.name,
    ref: branchRef, direction: "desc", per_page: 1,
  });
  if (change?.ref !== branchRef || change.after !== pr.head.sha ||
      !(Date.parse(lastLabel.created_at) > Date.parse(change.timestamp))) return DENIED;
  return await canWrite(github, context.repo, lastLabel.actor?.login)
    ? {approved: true, sha: pr.head.sha}
    : DENIED;
}

async function approvedTargetBranch(github, repo, branch) {
  let target;
  try {
    ({data: target} = await github.rest.repos.getBranch({...repo, branch}));
  } catch (error) {
    if (error.status === 404) return DENIED;
    throw error;
  }
  const sha = target.commit?.sha || "";
  if (target.name !== branch || !isSha(sha)) return DENIED;
  if (target.protected === true) return {approved: true, sha};
  return branch === "yuhe/v1.0.0" && await trustedBranchHead(github, repo, branch, sha)
    ? {approved: true, sha}
    : DENIED;
}

function isTrustedReleaseBranchName(branch) {
  return branch === "yuhe/v1.0.0" || branch.startsWith("release/");
}

async function approvedCrossBranchTarget(github, repo, ref, workflowSha) {
  // The default-branch dispatcher calls the release workflow with a release branch
  // or forwards a previously pinned 40-char SHA into a nested reusable workflow.
  if (isTrustedReleaseBranchName(ref)) {
    return approvedTargetBranch(github, repo, ref);
  }
  if (!isSha(ref)) return DENIED;
  const yuhe = await approvedTargetBranch(github, repo, "yuhe/v1.0.0");
  if (yuhe.approved && yuhe.sha === ref) return yuhe;
  if (workflowSha === ref && await trustedBranchHead(github, repo, "yuhe/v1.0.0", ref, 30)) {
    return {approved: true, sha: ref};
  }
  const protectedBranches = await github.paginate(github.rest.repos.listBranches, {
    ...repo, protected: true, per_page: 100,
  });
  return protectedBranches.some(item => item.name.startsWith("release/") && item.commit?.sha === ref)
    ? {approved: true, sha: ref}
    : DENIED;
}

async function resolve({github, context, label, ref = "", workflowSha = "", allowedDispatchBots = ""}) {
  const {owner, repo} = context.repo;
  if (owner.toLowerCase() !== "t-head" || repo.toLowerCase() !== "sglang-for-sail" ||
      !["run-ci-ppu", "run-ci-ppu-nightly"].includes(label)) return DENIED;

  if (context.eventName === "pull_request_target") {
    return approvedPullRequest(github, context, label, ref);
  }
  if (!context.ref.startsWith("refs/heads/") || !isSha(context.sha)) return DENIED;
  const branch = context.ref.slice("refs/heads/".length);
  const ownRef = !ref || ref === context.sha || ref === context.ref || ref === branch;
  if (context.eventName === "workflow_dispatch") {
    const allowedBots = parseAllowedDispatchBots(
      allowedDispatchBots || process.env.PPU_CI_ALLOWED_DISPATCH_BOTS || "",
    );
    const actorAuth = await canDispatch(github, context.repo, context.actor, allowedBots);
    if (!actorAuth.allowed) return DENIED;
    const rerunner = process.env.GITHUB_TRIGGERING_ACTOR;
    if (!rerunner) return DENIED;
    let rerunnerIsBot = actorAuth.isBot;
    if (rerunner !== context.actor) {
      const rerunnerAuth = await canDispatch(github, context.repo, rerunner, allowedBots);
      if (!rerunnerAuth.allowed) return DENIED;
      rerunnerIsBot = rerunnerAuth.isBot;
    }
    if (ownRef) {
      if (!actorAuth.isBot && !rerunnerIsBot) {
        return {approved: true, sha: context.sha};
      }
      if (!isTrustedReleaseBranchName(branch)) return DENIED;
      const target = await approvedTargetBranch(github, context.repo, branch);
      return target.approved && target.sha === context.sha
        ? {approved: true, sha: context.sha}
        : DENIED;
    }
    if (!["main", "sail_ci_cd"].includes(branch)) return DENIED;
    const {data: repository} = await github.rest.repos.get(context.repo);
    if (repository.default_branch !== branch) return DENIED;
    return approvedCrossBranchTarget(github, context.repo, ref, workflowSha);
  }
  const isProtectedCaller = process.env.GITHUB_REF_PROTECTED === "true";
  if (context.eventName === "push") {
    if (!ownRef) return DENIED;
    if (isProtectedCaller || (branch === "sail_ci_cd" && await trustedBranchHead(github, context.repo, branch, context.sha))) {
      return {approved: true, sha: context.sha};
    }
    return DENIED;
  }
  if (context.eventName !== "schedule" || !["main", "sail_ci_cd"].includes(branch)) return DENIED;
  const {data: repository} = await github.rest.repos.get(context.repo);
  if (repository.default_branch !== branch) return DENIED;
  if (!isProtectedCaller &&
      !(branch === "sail_ci_cd" && await trustedBranchHead(github, context.repo, branch, context.sha))) {
    return DENIED;
  }
  if (ownRef) return {approved: true, sha: context.sha};
  return approvedCrossBranchTarget(github, context.repo, ref, workflowSha);
}

async function ppuApproved(args) {
  return (await resolve(args)).approved;
}

ppuApproved.resolve = resolve;
module.exports = ppuApproved;
