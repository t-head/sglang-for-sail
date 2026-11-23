// SPDX-License-Identifier: Apache-2.0

const v0523 = "release/v0.5.23";
const transition = "yuhe/v1.0.0";
const v100 = "release/v1.0.0";
const supported = [v0523, transition, v100];

async function selectRefs({event, activeRefs, target, getHead}) {
  if (event === "schedule") {
    const refs = (activeRefs || `${v0523},${transition}`).split(",").map(ref => ref.trim());
    if (refs.some(ref => !supported.includes(ref)) || new Set(refs).size !== refs.length ||
        (refs.includes(transition) && refs.includes(v100))) {
      throw new Error("PPU_NIGHTLY_ACTIVE_REFS must select at most one branch per supported version");
    }
    return {v0523: refs.includes(v0523) ? v0523 : "", v100: refs.find(ref => ref !== v0523) || ""};
  }

  if (event !== "workflow_dispatch") throw new Error("Unsupported Nightly dispatch event");
  const requested = target || transition;
  if (supported.includes(requested)) {
    return requested === v0523 ? {v0523: requested, v100: ""} : {v0523: "", v100: requested};
  }

  if (/^[0-9a-f]{40}$/.test(requested)) {
    if (await getHead(v0523) === requested) return {v0523: requested, v100: ""};
    if (await getHead(transition) === requested || await getHead(v100) === requested) {
      return {v0523: "", v100: requested};
    }
  }
  throw new Error("target_ref must identify a supported Nightly branch or its current head SHA");
}

module.exports = {selectRefs};
