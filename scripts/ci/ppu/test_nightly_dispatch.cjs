// SPDX-License-Identifier: Apache-2.0

const assert = require("node:assert/strict");
const {readFileSync} = require("node:fs");
const {resolve} = require("node:path");
const test = require("node:test");
const {selectRefs} = require("./nightly_dispatch.cjs");

function extractPlannerScript() {
  const workflow = readFileSync(
    resolve(__dirname, "../../../.github/workflows/nightly-dispatch-ppu.yml"),
    "utf8",
  );
  const lines = workflow.split("\n");
  const step = lines.findIndex(line => /^\s*id:\s*refs\s*$/.test(line));
  assert.notEqual(step, -1, "Missing refs step in nightly-dispatch-ppu.yml");
  for (let i = step + 1; i < lines.length; i++) {
    const header = /^( *)script: \|\s*$/.exec(lines[i]);
    if (!header) continue;
    const body = [];
    let indent;
    for (const line of lines.slice(i + 1)) {
      if (!line.trim()) {
        body.push("");
        continue;
      }
      const spaces = line.match(/^ */)[0].length;
      if (spaces <= header[1].length) break;
      indent ??= spaces;
      body.push(line.slice(indent));
    }
    return body.join("\n").trimEnd();
  }
  assert.fail("Missing planner script block in nightly-dispatch-ppu.yml");
}

test("scheduled dispatch defaults to both versions and supports release cutover", async () => {
  assert.deepEqual(await selectRefs({event: "schedule"}), {
    v0523: "release/v0.5.23", v100: "yuhe/v1.0.0",
  });
  assert.deepEqual(await selectRefs({
    event: "schedule", activeRefs: "release/v0.5.23, release/v1.0.0",
  }), {v0523: "release/v0.5.23", v100: "release/v1.0.0"});
  assert.deepEqual(await selectRefs({
    event: "schedule", activeRefs: "release/v1.0.0",
  }), {v0523: "", v100: "release/v1.0.0"});
});

test("scheduled dispatch rejects duplicates, competing v1 branches and unknown refs", async () => {
  for (const activeRefs of [
    "release/v0.5.23,release/v0.5.23", "yuhe/v1.0.0,release/v1.0.0", "main",
  ]) {
    await assert.rejects(() => selectRefs({event: "schedule", activeRefs}), /ACTIVE_REFS/);
  }
});

test("DevOps dispatch resolves one version independently of the cron configuration", async () => {
  const event = "workflow_dispatch";
  const sha = "a".repeat(40);
  const getHead = async ref => ref === "release/v0.5.23" ? sha : "b".repeat(40);
  assert.deepEqual(await selectRefs({event, target: "release/v0.5.23"}), {
    v0523: "release/v0.5.23", v100: "",
  });
  assert.deepEqual(await selectRefs({event, target: "release/v1.0.0", activeRefs: "main"}), {
    v0523: "", v100: "release/v1.0.0",
  });
  assert.deepEqual(await selectRefs({event, target: sha, getHead}), {v0523: sha, v100: ""});
  await assert.rejects(() => selectRefs({event, target: "main", getHead}), /target_ref/);
});

test("missing earlier branch (404/null) does not block later head SHA lookup, while API errors propagate", async () => {
  const event = "workflow_dispatch";
  const v1Sha = "c".repeat(40);
  const staleSha = "d".repeat(40);
  const notFound = () => Object.assign(new Error("Branch not found"), {status: 404});
  const getHead404 = async ref => {
    if (ref === "release/v1.0.0") return v1Sha;
    throw notFound();
  };
  assert.deepEqual(
    await selectRefs({event, target: v1Sha, getHead: getHead404}),
    {v0523: "", v100: v1Sha},
  );
  await assert.rejects(
    () => selectRefs({event, target: staleSha, getHead: getHead404}),
    /target_ref/,
  );

  const serverError = Object.assign(new Error("Internal Server Error"), {status: 500});
  await assert.rejects(
    () => selectRefs({event, target: v1Sha, getHead: async () => { throw serverError; }}),
    /Internal Server Error/,
  );
});

test("workflow planner script treats 404 branch lookup as absent and rethrows other API errors", async t => {
  const script = extractPlannerScript();
  const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
  const runPlanner = new AsyncFunction("github", "context", "core", "require", script);
  const repoRoot = resolve(__dirname, "../../..");
  const localRequire = id => require(resolve(repoRoot, id));

  const prevTarget = process.env.TARGET;
  const prevActive = process.env.ACTIVE_REFS;
  t.after(() => {
    if (prevTarget === undefined) delete process.env.TARGET;
    else process.env.TARGET = prevTarget;
    if (prevActive === undefined) delete process.env.ACTIVE_REFS;
    else process.env.ACTIVE_REFS = prevActive;
  });

  const v1Sha = "e".repeat(40);
  process.env.TARGET = v1Sha;
  delete process.env.ACTIVE_REFS;

  const outputs = {};
  const core = {setOutput: (k, v) => { outputs[k] = v; }};
  const context = {
    eventName: "workflow_dispatch",
    repo: {owner: "t-head", repo: "sglang-for-sail"},
  };
  const github404 = {
    rest: {
      repos: {
        getBranch: async ({branch}) => {
          if (branch === "release/v1.0.0") return {data: {commit: {sha: v1Sha}}};
          throw Object.assign(new Error("Not Found"), {status: 404});
        },
      },
    },
  };
  await runPlanner(github404, context, core, localRequire);
  assert.deepEqual(outputs, {v0523: "", v100: v1Sha});

  const github503 = {
    rest: {
      repos: {
        getBranch: async () => {
          throw Object.assign(new Error("Service Unavailable"), {status: 503});
        },
      },
    },
  };
  await assert.rejects(
    () => runPlanner(github503, context, core, localRequire),
    /Service Unavailable/,
  );
});

function extractJobSecrets(workflowText, jobName) {
  const lines = workflowText.split(/\r?\n/);
  const meaningful = line => {
    const trimmed = line.trim();
    return trimmed !== "" && !trimmed.startsWith("#");
  };
  const indentOf = line => line.match(/^ */)[0].length;

  let jobsLine = -1;
  for (let i = 0; i < lines.length; i++) {
    if (!meaningful(lines[i])) continue;
    if (indentOf(lines[i]) === 0 && /^jobs:\s*(?:#.*)?$/.test(lines[i])) {
      assert.equal(jobsLine, -1, "Duplicate top-level jobs section");
      jobsLine = i;
    }
  }
  assert.notEqual(jobsLine, -1, "Missing top-level jobs section");

  let jobIndent = null;
  let targetStart = -1;
  let targetEnd = lines.length;
  for (let i = jobsLine + 1; i < lines.length; i++) {
    const line = lines[i];
    if (!meaningful(line)) continue;
    const spaces = indentOf(line);
    if (spaces === 0) break;
    jobIndent ??= spaces;
    if (spaces === jobIndent) {
      if (targetStart !== -1) {
        targetEnd = i;
        break;
      }
      if (new RegExp(`^\\s*${jobName}:\\s*(?:#.*)?$`).test(line)) {
        targetStart = i;
      }
    }
  }
  assert.notEqual(targetStart, -1, `Missing job ${jobName}`);
  for (let i = targetEnd + 1; i < lines.length; i++) {
    const line = lines[i];
    if (!meaningful(line)) continue;
    const spaces = indentOf(line);
    if (spaces === 0) break;
    if (spaces === jobIndent && new RegExp(`^\\s*${jobName}:\\s*(?:#.*)?$`).test(line)) {
      assert.fail(`Duplicate job ${jobName}`);
    }
  }

  let propIndent = null;
  let secretsStart = -1;
  let secretsEnd = targetEnd;
  for (let i = targetStart + 1; i < targetEnd; i++) {
    const line = lines[i];
    if (!meaningful(line)) continue;
    const spaces = indentOf(line);
    if (spaces <= jobIndent) break;
    propIndent ??= spaces;
    if (spaces === propIndent) {
      if (secretsStart !== -1 && secretsEnd === targetEnd) {
        secretsEnd = i;
      }
      if (/^\s*secrets:\s*(?:#.*)?$/.test(line)) {
        assert.equal(secretsStart, -1, `Duplicate secrets section in job ${jobName}`);
        secretsStart = i;
        secretsEnd = targetEnd;
      }
    }
  }
  if (secretsStart === -1) return Object.create(null);

  const secrets = Object.create(null);
  let entryIndent = null;
  for (let i = secretsStart + 1; i < secretsEnd; i++) {
    const line = lines[i];
    if (!meaningful(line)) continue;
    const spaces = indentOf(line);
    if (spaces <= propIndent) break;
    entryIndent ??= spaces;
    assert.equal(spaces, entryIndent, `Unexpected indentation in ${jobName}.secrets: ${line}`);
    const match = /^\s*([A-Za-z0-9_]+):\s*(.+?)\s*(?:#.*)?$/.exec(line);
    assert.ok(match, `Malformed secret entry in ${jobName}.secrets: ${line}`);
    const [, key, value] = match;
    assert.equal(Object.hasOwn(secrets, key), false, `Duplicate secret ${key} in ${jobName}`);
    secrets[key] = value;
  }
  return secrets;
}

test("nightly-dispatch-ppu forwards PPU_CI_NOTIFY_TARGETS_JSON on each named release job", () => {
  const workflow = readFileSync(
    resolve(__dirname, "../../../.github/workflows/nightly-dispatch-ppu.yml"),
    "utf8",
  );
  for (const jobName of ["release-v0523", "release-v100"]) {
    const secrets = extractJobSecrets(workflow, jobName);
    assert.equal(
      secrets.PPU_CI_NOTIFY_TARGETS_JSON,
      "${{ secrets.PPU_CI_NOTIFY_TARGETS_JSON }}",
      `${jobName} must forward PPU_CI_NOTIFY_TARGETS_JSON`,
    );
    assert.equal(
      secrets.PPU_CI_NOTIFY_WEBHOOK_URL,
      "${{ secrets.PPU_CI_NOTIFY_WEBHOOK_URL }}",
      `${jobName} must forward PPU_CI_NOTIFY_WEBHOOK_URL`,
    );
    assert.equal(
      secrets.PPU_CI_NOTIFY_SIGNING_SECRET,
      "${{ secrets.PPU_CI_NOTIFY_SIGNING_SECRET }}",
      `${jobName} must forward PPU_CI_NOTIFY_SIGNING_SECRET`,
    );
    assert.equal(
      secrets.PPU_CI_NOTIFY_ROSTER_JSON,
      "${{ secrets.PPU_CI_NOTIFY_ROSTER_JSON }}",
      `${jobName} must forward PPU_CI_NOTIFY_ROSTER_JSON`,
    );
  }
});

test("extractJobSecrets detects missing binding moved to another job or plan.env", () => {
  const workflow = readFileSync(
    resolve(__dirname, "../../../.github/workflows/nightly-dispatch-ppu.yml"),
    "utf8",
  );
  const v1Section = workflow.slice(workflow.indexOf("  release-v100:"));
  const removedFromV1 =
    workflow.slice(0, workflow.indexOf("  release-v100:")) +
    v1Section.replace(
      /^\s+PPU_CI_NOTIFY_TARGETS_JSON:\s+\$\{\{\s*secrets\.PPU_CI_NOTIFY_TARGETS_JSON\s*\}\}\r?\n/m,
      "",
    );

  const movedToDisabledJob =
    removedFromV1.trimEnd() +
    "\n\n  release-disabled:\n    if: false\n    uses: t-head/sglang-for-sail/.github/workflows/nightly-test-ppu.yml@main\n    secrets:\n      PPU_CI_NOTIFY_TARGETS_JSON: ${{ secrets.PPU_CI_NOTIFY_TARGETS_JSON }}\n";
  assert.equal(
    extractJobSecrets(movedToDisabledJob, "release-v0523").PPU_CI_NOTIFY_TARGETS_JSON,
    "${{ secrets.PPU_CI_NOTIFY_TARGETS_JSON }}",
  );
  assert.equal(
    extractJobSecrets(movedToDisabledJob, "release-v100").PPU_CI_NOTIFY_TARGETS_JSON,
    undefined,
  );

  const movedToPlanEnv = removedFromV1.replace(
    "        env:\n",
    "        env:\n          PPU_CI_NOTIFY_TARGETS_JSON: ${{ secrets.PPU_CI_NOTIFY_TARGETS_JSON }}\n",
  );
  assert.equal(
    extractJobSecrets(movedToPlanEnv, "release-v100").PPU_CI_NOTIFY_TARGETS_JSON,
    undefined,
  );

  const duplicatedInJob = workflow.replace(
    "      PPU_CI_NOTIFY_TARGETS_JSON: ${{ secrets.PPU_CI_NOTIFY_TARGETS_JSON }}\n",
    "      PPU_CI_NOTIFY_TARGETS_JSON: ${{ secrets.PPU_CI_NOTIFY_TARGETS_JSON }}\n      PPU_CI_NOTIFY_TARGETS_JSON: ${{ secrets.PPU_CI_NOTIFY_TARGETS_JSON }}\n",
  );
  assert.throws(() => extractJobSecrets(duplicatedInJob, "release-v0523"), /Duplicate secret/);
});

test("default branch registers pull_request_target label triggers for PR and Nightly PPU workflows", () => {
  const prWorkflow = readFileSync(
    resolve(__dirname, "../../../.github/workflows/pr-test-ppu.yml"),
    "utf8",
  );
  assert.match(prWorkflow, /^\s*pull_request_target:\s*$/m);
  assert.match(prWorkflow, /branches:\s*\['yuhe\/v1\.0\.0',\s*'release\/\*\*'\]/);
  for (const eventType of ["opened", "synchronize", "reopened", "labeled", "ready_for_review"]) {
    assert.match(prWorkflow, new RegExp(`\\b${eventType}\\b`), `pr-test-ppu.yml must register ${eventType}`);
  }
  for (const inputName of ["preflight_only", "model_filter", "run_model_sanity", "run_all_tests", "run_legacy_sanity"]) {
    assert.match(prWorkflow, new RegExp(`^\\s*${inputName}:\\s*$`, "m"), `pr-test-ppu.yml missing input ${inputName}`);
  }

  const nightlyWorkflow = readFileSync(
    resolve(__dirname, "../../../.github/workflows/nightly-test-ppu.yml"),
    "utf8",
  );
  assert.match(nightlyWorkflow, /^\s*pull_request_target:\s*$/m);
  assert.match(nightlyWorkflow, /branches:\s*\['yuhe\/v1\.0\.0',\s*'release\/\*\*'\]/);
  for (const eventType of ["labeled", "synchronize", "reopened", "ready_for_review"]) {
    assert.match(nightlyWorkflow, new RegExp(`\\b${eventType}\\b`), `nightly-test-ppu.yml must register ${eventType}`);
  }

  const tiers = JSON.parse(
    readFileSync(resolve(__dirname, "../opt_in_ci_tiers.json"), "utf8"),
  );
  assert.deepEqual(tiers["ppu-extra"], {
    label: "run-ci-ppu-extra",
    workflows: [".github/workflows/pr-test-ppu-extra.yml"],
  });
});
