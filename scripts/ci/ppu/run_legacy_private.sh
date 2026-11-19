#!/usr/bin/env bash
set -euo pipefail
umask 077

root=${PPU_CI_PRIVATE_RESULTS_ROOT:-"${HOME:?}/.sglang-ppu-ci-results"}
mkdir -p -- "$root"
[[ -d "$root" && ! -L "$root" && $(stat -c '%a' "$root") == 700 ]]
root=$(realpath -e -- "$root")
[[ "$root" != "$GITHUB_WORKSPACE" && "$root" != "$GITHUB_WORKSPACE/"* ]]
[[ "$root" != "$RUNNER_TEMP" && "$root" != "$RUNNER_TEMP/"* ]]
destination=$(mktemp -d "$root/legacy.${GITHUB_RUN_ID:?}.${GITHUB_RUN_ATTEMPT:?}.XXXXXXXX")
export PPU_CI_ARTIFACT_DIR="$destination"
export PPU_CI_RETAIN_LEGACY=1
exec bash scripts/ci/ppu/run_test.sh > "$destination/launcher.log" 2>&1
