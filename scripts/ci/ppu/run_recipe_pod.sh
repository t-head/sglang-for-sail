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
export PPU_CI_WHEEL_DIR="$PWD/wheels" PPU_CI_RECIPE_SOURCE_DIR="$PWD"
[[ $(git rev-parse HEAD) == "${PPU_CI_SOURCE_SHA:?}" ]]
if [[ "$TIER" == legacy ]]; then
    bash scripts/ci/ppu/build_and_test.sh test
    exit
fi
printf '%s\n' wheel_verification > "$PPU_CI_ARTIFACT_DIR/recipe-phase.txt"
python3 -B scripts/ci/ppu/recipe_bundle.py check "$PPU_CI_WHEEL_DIR" "$PPU_CI_SOURCE_SHA"
case "$MODEL_KEY" in
    glm-5.3) export PPU_RECIPE_CONTAINER_MODEL=${PPU_CI_GLM53_MODEL_PATH:?} ;;
    deepseek-v4-flash) export PPU_RECIPE_CONTAINER_MODEL=${PPU_CI_DSV4_FLASH_MODEL_PATH:?} ;;
    *) exit 2 ;;
esac
[[ -f "$PPU_RECIPE_CONTAINER_MODEL/config.json" && -d /recipe-jit ]]
if [[ "$TIER" == model_sanity ]]; then
    [[ -d "${PPU_CI_GSM8K_PATH:?}" ]]
fi
cache_key=$(python3 -B scripts/ci/ppu/recipe_bundle.py cache-key "$PPU_CI_WHEEL_DIR" "$PPU_CI_SOURCE_SHA")
[[ "${PPU_CI_RECIPE_IMAGE_KEY:?}" =~ ^([0-9a-f]{64}|run-[0-9]+-[0-9]+)$ ]]
export PPU_CI_RECIPE_JIT_DIR="/recipe-jit/sglang-ci/$PPU_CI_RECIPE_IMAGE_KEY/$cache_key/$MODEL_KEY"
bash scripts/ci/ppu/recipe_container.sh "test/ppu/policy/models/$MODEL_KEY.yaml" "$TIER" --queue-s 0
