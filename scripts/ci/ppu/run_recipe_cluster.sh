#!/usr/bin/env bash
# Run the pinned cluster-run backend with private scheduler streams.
set -euo pipefail
umask 077

if [[ -z "${PPU_CI_PRIVATE_RESULTS_ROOT:-}" && -n "${PPU_CI_K8S_CONFIG_SHA256:-}" ]]; then
    exec python3 -B scripts/ci/ppu/recipe_cluster_config.py
fi

visible_output=${GITHUB_OUTPUT:?}
mode=${TIER:?}
root=${PPU_CI_PRIVATE_RESULTS_ROOT:-}
pod_root=${PPU_CI_K8S_PRIVATE_WORKER_ROOT:-}
if [[ "$root" != /* || "$pod_root" != /* || ! -d "$root" || -L "$root" ]]; then
    echo "::error::PPU cluster private storage is not configured"
    exit 2
fi
if ! raw=$(mktemp -d "$root/cluster.${GITHUB_RUN_ID}.${GITHUB_RUN_ATTEMPT}.XXXXXXXX" 2>/dev/null); then
    echo "::error::PPU cluster private storage is unavailable"
    exit 2
fi

exec 3>&1 > "$raw/launcher.log" 2>&1
export PPU_CI_ARTIFACT_DIR="$raw"
export PATH="/opt/bin:$PATH"
infra="$GITHUB_WORKSPACE/.ppu-infra/cluster-run/scripts"
export INPUT_NAMESPACE=ppu-sched INPUT_NNODES=0 INPUT_LAUNCH_MODE=each-node
export INPUT_SOURCE_DIR=/workspace/source INPUT_DMESG_POLICY=never INPUT_CLEANUP_POLICY=always
export GITHUB_OUTPUT="$raw/scheduler-output" GITHUB_ENV="$raw/scheduler-env"
export GITHUB_STEP_SUMMARY="$raw/scheduler-summary"
export GITHUB_PATH="$raw/scheduler-path" GITHUB_STATE="$raw/scheduler-state"
: > "$GITHUB_OUTPUT"

read_output() {
    awk -F= -v key="$1" '$1 == key {print substr($0, length(key)+2)}' "$GITHUB_OUTPUT" | tail -1
}

export JOB_NAME="" SOURCE_STAGE_DIR="" INPUT_JOB_STATUS=failed INPUT_WORKFLOW_STATUS=failure
phase() { printf '%s\n' "$1" > "$raw/recipe-phase.txt"; }
child=""
run_backend() {
    local rc=0
    bash "$1" &
    child=$!
    wait "$child" || rc=$?
    child=""
    return "$rc"
}
finish() {
    local rc=$?
    trap - EXIT
    trap '' TERM INT
    set +e
    if [[ -n "$child" ]]; then
        kill -TERM "$child" 2>/dev/null
        wait "$child" 2>/dev/null
    fi
    if [[ -n "$JOB_NAME" && -n "$SOURCE_STAGE_DIR" ]]; then
        if ! bash "$infra/cleanup.sh" ||
            ! remaining=$(timeout 15s kubectl -n "$INPUT_NAMESPACE" get pods -l "ppu-job=$JOB_NAME" -o name) ||
            [[ -n "$remaining" || -e "$SOURCE_STAGE_DIR" ]]; then
            rc=1
            phase cleanup
        fi
    fi
    if [[ "$mode" == legacy ]]; then
        (
            cd "$raw" || exit 1
            find . -type d -exec chmod 700 '{}' +
            find . -type f -exec chmod 600 '{}' +
            sums=$(mktemp)
            find . -type f -print0 | sort -z | xargs -0 -r sha256sum > "$sums"
            mv -- "$sums" SHA256SUMS
        ) || rc=1
        if (( rc != 0 )); then
            echo "::error::Legacy PPU cluster job failed" >&3
        fi
        exit "$rc"
    fi
    export RECIPE_OUTCOME=failure
    (( rc == 0 )) && export RECIPE_OUTCOME=success
    python3 -B scripts/ci/ppu/recipe_report.py "$raw/public-candidate.json" || rc=1
    if [[ -f "$raw/public-candidate.json" ]] &&
        python3 -B scripts/ci/ppu/recipe_publish.py "$raw/public-candidate.json" \
            ppu-public/recipe.json "$public_summary" >&3; then
        printf 'approved=true\n' >> "$visible_output"
    else
        echo "::error::PPU cluster result could not be published" >&3
        rc=1
    fi
    exit "$rc"
}
public_summary=${PPU_PUBLIC_SUMMARY:?}
trap finish EXIT
trap 'INPUT_WORKFLOW_STATUS=cancelled; exit 143' TERM
trap 'INPUT_WORKFLOW_STATUS=cancelled; exit 130' INT

phase runner_preflight
[[ "$(git rev-parse HEAD)" == "${PPU_CI_SOURCE_SHA:?}" ]]
[[ "$(git -C "$infra" rev-parse HEAD)" == 96e25147971f5d2a04d828f8ad508cae38a4fa8a ]]
command -v kubectl >/dev/null
stage_root=${PPU_CI_K8S_STAGE_ROOT:?}
[[ "$stage_root" == /* && -d "$stage_root" && ! -L "$stage_root" ]]
[[ -d wheels && ! -L wheels ]]
case "$mode" in
    fast_sanity|model_sanity)
        [[ "$MODEL_KEY" == glm-5.3 || "$MODEL_KEY" == deepseek-v4-flash ]]
        INPUT_NODE_SELECTOR=board-type=ZW-M890P
        INPUT_NPROC_PER_NODE=8
        INPUT_TIMEOUT_MINUTES=330
        [[ "$mode" != fast_sanity ]] || INPUT_TIMEOUT_MINUTES=30
        INPUT_IMAGE=${PPU_CI_RECIPE_RUNTIME_IMAGE:?}
        registry=download.t-head.cn/docker_test/sglang
        if [[ "$INPUT_IMAGE" == "$registry:release-v0.5.23-20261116-cf244d4" &&
            -n "${PPU_CI_K8S_RUNTIME_DIGEST:-}" ]]; then
            INPUT_IMAGE=$PPU_CI_K8S_RUNTIME_DIGEST
            [[ "$INPUT_IMAGE" == *@sha256:* ]]
        fi
        [[ "$INPUT_IMAGE" == "$registry:release-v0.5.23-20261116-cf244d4" ||
            "$INPUT_IMAGE" =~ ^download\.t-head\.cn/docker_test/sglang@sha256:[0-9a-f]{64}$ ]]
        [[ "${PPU_CI_JIT_ROOT:?}" == /* && "$PPU_CI_JIT_ROOT" != *[:,$'\r\n']* ]]
        export INPUT_HOST_VOLUMES="$PPU_CI_JIT_ROOT:/recipe-jit"
        if [[ "$INPUT_IMAGE" == *@sha256:* ]]; then
            PPU_CI_RECIPE_IMAGE_KEY=${INPUT_IMAGE##*:}
        else
            PPU_CI_RECIPE_IMAGE_KEY="run-$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT"
        fi
        export PPU_CI_RECIPE_IMAGE_KEY
        export CONTAINER_OPTIONS='shm_size: "128Gi"'
        phase wheel_verification
        python3 -B scripts/ci/ppu/recipe_bundle.py check wheels "$PPU_CI_SOURCE_SHA"
        ;;
    legacy)
        [[ "${PPU_CI_BOARD:?}" == ZW-M890P || "$PPU_CI_BOARD" == OAM-810E ]]
        INPUT_NODE_SELECTOR="board-type=$PPU_CI_BOARD"
        INPUT_NPROC_PER_NODE=1
        INPUT_TIMEOUT_MINUTES=45
        INPUT_IMAGE=${PPU_CI_RUNTIME_IMAGE:?}
        export CONTAINER_OPTIONS='shm_size: "16Gi"'
        ;;
    *) exit 2 ;;
esac
export INPUT_NODE_SELECTOR INPUT_NPROC_PER_NODE INPUT_TIMEOUT_MINUTES INPUT_IMAGE
export INPUT_COMMAND="exec bash /workspace/source/scripts/ci/ppu/run_recipe_pod.sh"
PPU_CI_WORKER_ARTIFACT_DIR="${pod_root%/}/$(basename -- "$raw")"
export PPU_CI_WORKER_ARTIFACT_DIR
USER_ENV_JSON=$(python3 -B - <<'PY'
import json
import os
from pathlib import PurePosixPath

keys = ["TIER", "PPU_CI_SOURCE_SHA", "PPU_CI_WORKER_ARTIFACT_DIR"]
if os.environ["TIER"] == "legacy":
    keys += [
        "PPU_CI_TARGET_MODEL_PATH", "PPU_CI_EAGLE3_MODEL_PATH",
        "PPU_CI_DFLASH_MODEL_PATH", "PPU_CI_EVAL_CACHE",
    ]
else:
    keys += ["MODEL_KEY", "PPU_CI_RECIPE_IMAGE_KEY"]
    model = os.environ["MODEL_KEY"]
    keys += ["PPU_CI_GLM53_MODEL_PATH" if model == "glm-5.3" else "PPU_CI_DSV4_FLASH_MODEL_PATH"]
    if os.environ["TIER"] == "model_sanity":
        keys += ["PPU_CI_GSM8K_PATH"]
values = {key: os.environ[key] for key in keys}
for key, value in values.items():
    if not value or any(ord(char) < 32 for char in value):
        raise ValueError("invalid worker input")
    if key.startswith("PPU_CI_") and key not in ("PPU_CI_SOURCE_SHA", "PPU_CI_RECIPE_IMAGE_KEY"):
        if not PurePosixPath(value).is_absolute() or ".." in PurePosixPath(value).parts:
            raise ValueError("expected an absolute worker path")
print(json.dumps(values))
PY
)
export USER_ENV_JSON

# Transfer the candidate revision, its Git metadata and checked wheel bundle.
# Cluster Run translates the CPU runner's staging mount into the worker view.
phase cluster_submit
job_hash=$(printf '%s' "$raw" | sha256sum | cut -c1-16)
export JOB_NAME="sglang-recipe-$job_hash"
stage_dir=$(mktemp -d "${stage_root%/}/$JOB_NAME.XXXXXXXX")
export SOURCE_STAGE_DIR="$stage_dir/source.tar.gz"
tar -czf "$SOURCE_STAGE_DIR" --exclude='./.ppu-infra' --exclude='./.git/config' \
    --exclude='./.git/config.worktree' --exclude='./.git/hooks' \
    --exclude='./.git/modules' --exclude='./.git/worktrees' -C "$GITHUB_WORKSPACE" .
run_backend "$infra/submit.sh"
[[ $(read_output job_status) == scheduled ]]
export JOB_START_TIME JOB_DEADLINE
JOB_START_TIME=$(read_output start_time)
JOB_DEADLINE=$(read_output deadline)
[[ "$JOB_START_TIME" =~ ^[0-9]+$ && "$JOB_DEADLINE" =~ ^[0-9]+$ ]]
queue_s=$(read_output queue_duration_seconds)
[[ "$queue_s" =~ ^[0-9]+$ ]] || exit 1

phase cluster_worker
run_backend "$infra/monitor.sh"
INPUT_JOB_STATUS=$(read_output job_status)
[[ "$INPUT_JOB_STATUS" == succeeded ]]
if [[ "$mode" != legacy ]]; then
    # The kubelet reports the worker's resolved pullable image reference.
    phase image_resolution
    image_id=$(kubectl -n "$INPUT_NAMESPACE" get pod "$JOB_NAME-worker-0" -o json |
        jq -er '.status.containerStatuses[] | select(.name == "worker") | .imageID')
    digest=${image_id#docker-pullable://}
    [[ "$digest" =~ ^download\.t-head\.cn/docker_test/sglang@sha256:[0-9a-f]{64}$ ]]
    [[ "$INPUT_IMAGE" != *@sha256:* || "$digest" == "$INPUT_IMAGE" ]]
    jq --arg image "$digest" '. + {runtime_image:$image}' wheels/recipe-manifest.json > "$raw/provenance.json"
    jq --argjson queue "$queue_s" '.queue_s = $queue' "$raw/results/summary.json" > "$raw/summary.tmp"
    mv -- "$raw/summary.tmp" "$raw/results/summary.json"
fi
INPUT_WORKFLOW_STATUS=success
