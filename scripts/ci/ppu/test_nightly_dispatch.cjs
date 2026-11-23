// SPDX-License-Identifier: Apache-2.0

const assert = require("node:assert/strict");
const test = require("node:test");
const {selectRefs} = require("./nightly_dispatch.cjs");

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
