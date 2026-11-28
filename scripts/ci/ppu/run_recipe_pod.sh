#!/usr/bin/env bash
# The device plugin assigns the requested cards to this isolated worker Pod.
set -euo pipefail
umask 077
export PPU_CI_ARTIFACT_DIR=${PPU_CI_WORKER_ARTIFACT_DIR:?}
[[ -d "$PPU_CI_ARTIFACT_DIR" && ! -L "$PPU_CI_ARTIFACT_DIR" &&
    $(stat -c %a -- "$PPU_CI_ARTIFACT_DIR") == 700 ]] || exit 2
exec > "$PPU_CI_ARTIFACT_DIR/runtime.log" 2>&1
# The runtime image may use a different UID than the CPU runner.
# Return ownership before the runner validates and retains private evidence.
finish() {
    code=$?
    trap - EXIT
    if [[ "$(stat -c %u -- "$PPU_CI_ARTIFACT_DIR")" != "$(id -u)" ]]; then
        chown -Rh --reference="$PPU_CI_ARTIFACT_DIR" "$PPU_CI_ARTIFACT_DIR" || code=1
    fi
    exit "$code"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
cd /workspace/source
# Cluster Run preserves the CPU runner's checkout ownership in the archive.
# Trust this one unpacked checkout, then verify its candidate revision.
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0="$PWD"
export PPU_CI_WHEEL_DIR="$PWD/wheels" PPU_CI_RECIPE_SOURCE_DIR="$PWD"
[[ $(git rev-parse HEAD) == "${PPU_CI_SOURCE_SHA:?}" ]]
if [[ "$TIER" == legacy ]]; then
    export SGLANG_TEST_METRICS_FILE="$PPU_CI_ARTIFACT_DIR/suite-metrics.jsonl"
    bash scripts/ci/ppu/build_and_test.sh legacy
    exit
fi
printf '%s\n' wheel_verification > "$PPU_CI_ARTIFACT_DIR/worker-phase.txt"
skip_wheel_install="${PPU_CI_SKIP_WHEEL_INSTALL:-false}"
[[ "$skip_wheel_install" == true || "$skip_wheel_install" == false ]] || exit 2
if [[ "$skip_wheel_install" == true ]]; then
    [[ "${PPU_CI_RECIPE_TREE_SHA256:?}" =~ ^[0-9a-f]{64}$ ]]
    actual_tree=$(python3 -B scripts/ci/ppu/recipe_bundle.py checkout-digest "$PPU_CI_SOURCE_SHA")
    [[ "$actual_tree" == "$PPU_CI_RECIPE_TREE_SHA256" ]]
    cache_key="$actual_tree"
else
    python3 -B scripts/ci/ppu/recipe_bundle.py check "$PPU_CI_WHEEL_DIR" "$PPU_CI_SOURCE_SHA"
    cache_key=$(python3 -B scripts/ci/ppu/recipe_bundle.py cache-key "$PPU_CI_WHEEL_DIR" "$PPU_CI_SOURCE_SHA")
fi
[[ -d "${PPU_CI_HF_CACHE_ROOT:?}/hub" && -d /recipe-jit ]]
[[ "${RECIPE_PPU_COUNT:?}" =~ ^(2|4|8)$ ]]
if [[ "$TIER" == model_sanity ]]; then
    [[ -d "${PPU_CI_GSM8K_PATH:?}" ]]
    export PPU_CI_EVAL_WHEEL_DIR="$PWD/evalscope-wheels"
fi
[[ "${PPU_CI_RECIPE_IMAGE_KEY:?}" =~ ^[0-9a-f]{64}$ ]]
export PPU_CI_SKIP_WHEEL_INSTALL="$skip_wheel_install"
export PPU_CI_RECIPE_JIT_DIR="/recipe-jit/sglang-ci/$PPU_CI_RECIPE_IMAGE_KEY/$cache_key/$MODEL_KEY"
bash scripts/ci/ppu/recipe_container.sh "test/ppu/policy/models/$MODEL_KEY.yaml" "$TIER" --queue-s 0
