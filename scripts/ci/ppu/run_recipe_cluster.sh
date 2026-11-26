#!/usr/bin/env bash
# Run the pinned cluster-run backend with private scheduler streams.
set -euo pipefail
umask 077

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

if [[ -n "${PPU_CI_K8S_CONFIG_SHA256:-}" &&
    "${PPU_CI_K8S_LOADED_CONFIG_SHA256:-}" != "$PPU_CI_K8S_CONFIG_SHA256" ]]; then
    exec python3 -B "$SCRIPT_DIR/recipe_cluster_config.py"
fi

if [[ -z "${PPU_CI_POLICY_ROOT:-}" && -d "${GITHUB_WORKSPACE:-$PWD}/test/ppu/policy/models" && ! -L "${GITHUB_WORKSPACE:-$PWD}/test/ppu/policy/models" ]]; then
    export PPU_CI_POLICY_ROOT="${GITHUB_WORKSPACE:-$PWD}/test/ppu/policy/models"
fi

mode=${TIER:?}
root=${PPU_CI_K8S_HOST_SCRATCH_ROOT:-${PPU_CI_PRIVATE_RESULTS_ROOT:-}}
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

WORKER_POD_COUNT=1

collect_pod_statuses() {
    local count="${WORKER_POD_COUNT:-1}"
    local i out
    local collected=0
    for (( i = 0; i < count; i++ )); do
        out="$raw/pod-status-worker-$i.json"
        if timeout 15s kubectl -n "$INPUT_NAMESPACE" get pod "$JOB_NAME-worker-$i" -o json > "$out" 2>/dev/null && [[ -s "$out" ]]; then
            collected=$(( collected + 1 ))
        else
            rm -f -- "$out"
        fi
    done
    if [[ -s "$raw/pod-status-worker-0.json" ]]; then
        cp -- "$raw/pod-status-worker-0.json" "$raw/pod-status.json" || true
    fi
    (( collected == count ))
}

record_runtime_image() {
    local count="${WORKER_POD_COUNT:-1}"
    local i
    local files=()
    for (( i = 0; i < count; i++ )); do
        [[ -s "$raw/pod-status-worker-$i.json" ]] || return 1
        files+=("$raw/pod-status-worker-$i.json")
    done
    local public_digest
    public_digest=$(python3 -B "$SCRIPT_DIR/recipe_image.py" verify-pods "${files[@]}" \
        --expected-repo "$EXPECTED_IMAGE_REPO" \
        --expected-digest "$EXPECTED_IMAGE_DIGEST") || return 1
    if [[ "$SKIP_WHEEL_INSTALL" == true ]]; then
        jq -n --arg image "$public_digest" --arg tree "$RECIPE_TREE_SHA256" \
            '{runtime_image:$image, recipe_tree_sha256:$tree, install_mode:"prebuilt_image"}' \
            > "$raw/provenance.tmp" &&
            mv -- "$raw/provenance.tmp" "$raw/provenance.json"
    else
        jq --arg image "$public_digest" '. + {runtime_image:$image, install_mode:"wheel_overlay"}' \
            wheels/recipe-manifest.json > "$raw/provenance.tmp" &&
            mv -- "$raw/provenance.tmp" "$raw/provenance.json"
    fi
}

verify_cluster_cleanup() {
    local remaining remaining_svc remaining_pg stage_parent
    remaining=$(timeout 15s kubectl -n "$INPUT_NAMESPACE" get pods -l "ppu-job=$JOB_NAME" -o name) || return 1
    [[ -z "$remaining" ]] || return 1
    if (( INPUT_NNODES > 0 )); then
        remaining_svc=$(timeout 15s kubectl -n "$INPUT_NAMESPACE" get svc "$JOB_NAME" --ignore-not-found -o name) || return 1
        remaining_pg=$(timeout 15s kubectl -n "$INPUT_NAMESPACE" get podgroups.scheduling.x-k8s.io "$JOB_NAME" --ignore-not-found -o name) || return 1
        [[ -z "$remaining_svc" && -z "$remaining_pg" ]] || return 1
    fi
    [[ ! -e "$SOURCE_STAGE_DIR" ]] || return 1
    stage_parent=$(dirname -- "$SOURCE_STAGE_DIR")
    if [[ "$(basename -- "$SOURCE_STAGE_DIR")" == "source.tar.gz" &&
        "$(basename -- "$stage_parent")" == "${JOB_NAME}."* ]]; then
        [[ ! -e "$stage_parent" ]] || return 1
    fi
}

export JOB_NAME="" SOURCE_STAGE_DIR="" INPUT_JOB_STATUS=failed INPUT_WORKFLOW_STATUS=failure
cleanup_state="${RUNNER_TEMP:-/tmp}/ppu-cleanup-state.json"
cleanup_verified_marker="${RUNNER_TEMP:-/tmp}/ppu-cleanup-verified.json"
rm -f -- "$cleanup_verified_marker"
phase() { printf '%s\n' "$1" > "$raw/recipe-phase.txt"; }
write_cleanup_verified_marker() {
    local source_kind=$1
    mkdir -p -- "$(dirname -- "$cleanup_verified_marker")" || return 1
    local tmp="${cleanup_verified_marker}.tmp.$$"
    printf '{"cleanup_verified": true, "schema": "ppu-cleanup/v1", "source": "%s"}\n' "$source_kind" > "$tmp" &&
        mv -- "$tmp" "$cleanup_verified_marker"
}
cleanup_staging_fallback() {
    [[ -n "$JOB_NAME" && -n "$SOURCE_STAGE_DIR" ]] || return 0
    local parent
    parent=$(dirname -- "$SOURCE_STAGE_DIR")
    if [[ "$(basename -- "$SOURCE_STAGE_DIR")" == "source.tar.gz" &&
        "$(basename -- "$parent")" == "${JOB_NAME}."* ]]; then
        rm -f -- "$SOURCE_STAGE_DIR"
        rmdir -- "$parent" 2>/dev/null || true
    fi
}
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
        if [[ "$INPUT_WORKFLOW_STATUS" != cancelled ]] && collect_pod_statuses && [[ "$mode" != legacy && ! -f "$raw/provenance.json" ]]; then
            record_runtime_image || true
        fi
        local cleanup_ok=0
        bash "$infra/cleanup.sh" || cleanup_ok=1
        cleanup_staging_fallback
        if (( cleanup_ok == 0 )) && verify_cluster_cleanup; then
            write_cleanup_verified_marker orchestrator_trap
            rm -f -- "$cleanup_state"
        else
            rc=1
            phase cleanup
        fi
    else
        write_cleanup_verified_marker no_cluster_resources
    fi
    if [[ "$mode" == legacy ]]; then
        export PPU_LEGACY_RETAINED=true RECIPE_OUTCOME=failure
        (
            cd "$raw" || exit 1
            find . -type d -exec chmod 700 '{}' +
            find . -type f -exec chmod 600 '{}' +
            sums=$(mktemp)
            find . -type f -print0 | sort -z | xargs -0 -r sha256sum > "$sums"
            mv -- "$sums" SHA256SUMS
        ) || { rc=1; export PPU_LEGACY_RETAINED=false; }
        (( rc == 0 )) && export RECIPE_OUTCOME=success
        local candidate
        if ! candidate=$(mktemp 2>/dev/null) ||
            ! python3 -B "$SCRIPT_DIR/recipe_report.py" "$candidate" 2>/dev/null ||
            ! python3 -B "$SCRIPT_DIR/recipe_publish.py" "$candidate" \
                ppu-public/legacy.json "$public_summary" >&3 2>/dev/null; then
            echo "::error::Single-device evidence could not be published" >&3
            rc=1
        fi
        if [[ -n "$candidate" ]]; then
            rm -- "$candidate" 2>/dev/null || rc=1
        fi
        if (( rc != 0 )); then
            echo "::error::Legacy PPU cluster job failed" >&3
        fi
        exit "$rc"
    fi
    export RECIPE_OUTCOME=failure
    (( rc == 0 )) && export RECIPE_OUTCOME=success
    python3 -B "$SCRIPT_DIR/recipe_report.py" "$raw/public-candidate.json" || rc=1
    if [[ ! -f "$raw/public-candidate.json" ]] ||
        ! python3 -B "$SCRIPT_DIR/recipe_publish.py" "$raw/public-candidate.json" \
            ppu-public/recipe.json "$public_summary" >&3; then
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
SKIP_WHEEL_INSTALL=${PPU_CI_SKIP_WHEEL_INSTALL:-false}
[[ "$SKIP_WHEEL_INSTALL" == true || "$SKIP_WHEEL_INSTALL" == false ]]
EXPECTED_IMAGE_REPO=""
EXPECTED_IMAGE_DIGEST=""
RECIPE_TREE_SHA256=""
RECIPE_NNODES=${RECIPE_NNODES:-1}
[[ "$RECIPE_NNODES" =~ ^[1-9][0-9]*$ ]]
case "$mode" in
    fast_sanity|model_sanity)
        [[ "$MODEL_KEY" =~ ^[a-z0-9][a-z0-9.-]*$ && -f "test/ppu/policy/models/$MODEL_KEY.yaml" ]]
        INPUT_NODE_SELECTOR=board-type=ZW-M890P
        WORKER_POD_COUNT=$RECIPE_NNODES
        if (( RECIPE_NNODES > 1 )); then
            if [[ "${PPU_CI_ALLOW_MULTINODE_HOST_ACCESS:-false}" != true ]]; then
                echo "multi-node host access is not authorized for this run" >&2
                exit 2
            fi
            [[ "${RECIPE_PPU_COUNT:?}" == 8 ]] || exit 2
            INPUT_NNODES=$RECIPE_NNODES
            mkdir -p "$raw/ranks/barrier"
            for (( _rank = 1; _rank < RECIPE_NNODES; _rank++ )); do
                mkdir -p "$raw/ranks/rank-$_rank/results"
            done
            export CONTAINER_OPTIONS=$'privileged: true\nhost_ipc: true\nhost_network: true\nshm_size: "128Gi"'
            INPUT_TIMEOUT_MINUTES=330
            [[ "$mode" != fast_sanity ]] || INPUT_TIMEOUT_MINUTES=60
        else
            [[ "${RECIPE_PPU_COUNT:?}" =~ ^(2|4|8)$ ]] || exit 2
            INPUT_NNODES=0
            export CONTAINER_OPTIONS='shm_size: "128Gi"'
            INPUT_TIMEOUT_MINUTES=330
            [[ "$mode" != fast_sanity ]] || INPUT_TIMEOUT_MINUTES=30
        fi
        INPUT_NPROC_PER_NODE=$RECIPE_PPU_COUNT
        resolved_image_json=$(python3 -B "$SCRIPT_DIR/recipe_image.py" resolve \
            --image-tag "${PPU_CI_RUNTIME_IMAGE_TAG:-}" \
            --image-digest "${PPU_CI_RUNTIME_IMAGE_DIGEST:-}")
        INPUT_IMAGE=$(jq -er '.pull_ref' <<< "$resolved_image_json")
        EXPECTED_IMAGE_REPO=$(jq -er '.expected_repo' <<< "$resolved_image_json")
        EXPECTED_IMAGE_DIGEST=$(jq -r '.expected_digest // ""' <<< "$resolved_image_json")
        PPU_CI_RECIPE_IMAGE_KEY=$(jq -er '.image_key' <<< "$resolved_image_json")
        [[ "$PPU_CI_RECIPE_IMAGE_KEY" =~ ^[0-9a-f]{64}$ ]]
        [[ "${PPU_CI_JIT_ROOT:?}" == /* && "$PPU_CI_JIT_ROOT" != *[:,$'\r\n']* ]]
        export INPUT_HOST_VOLUMES="$PPU_CI_JIT_ROOT:/recipe-jit"
        export PPU_CI_RECIPE_IMAGE_KEY
        export RECIPE_NNODES
        phase wheel_verification
        if [[ "$SKIP_WHEEL_INSTALL" == true ]]; then
            [[ -n "${PPU_CI_RUNTIME_IMAGE_TAG:-}${PPU_CI_RUNTIME_IMAGE_DIGEST:-}" ]]
            [[ ! -e wheels ]]
            RECIPE_TREE_SHA256=$(python3 -B "$SCRIPT_DIR/recipe_bundle.py" checkout-digest "$PPU_CI_SOURCE_SHA")
            [[ "$RECIPE_TREE_SHA256" =~ ^[0-9a-f]{64}$ ]]
            export PPU_CI_RECIPE_TREE_SHA256="$RECIPE_TREE_SHA256"
        else
            [[ -d wheels && ! -L wheels ]]
            python3 -B "$SCRIPT_DIR/recipe_bundle.py" check wheels "$PPU_CI_SOURCE_SHA"
        fi
        export PPU_CI_SKIP_WHEEL_INSTALL="$SKIP_WHEEL_INSTALL"
        if [[ "$mode" == model_sanity ]]; then
            (cd evalscope-wheels && sha256sum -- *.whl | cmp -s SHA256SUMS -)
        fi
        ;;
    legacy)
        [[ "$SKIP_WHEEL_INSTALL" == false ]]
        [[ "$RECIPE_NNODES" == 1 ]]
        [[ -d wheels && ! -L wheels ]]
        [[ "${PPU_CI_BOARD:?}" == ZW-M890P || "$PPU_CI_BOARD" == OAM-810E ]]
        INPUT_NODE_SELECTOR="board-type=$PPU_CI_BOARD"
        INPUT_NPROC_PER_NODE=1
        INPUT_TIMEOUT_MINUTES=45
        INPUT_IMAGE=${PPU_CI_RUNTIME_IMAGE:?}
        export CONTAINER_OPTIONS='shm_size: "16Gi"'
        ;;
    *) exit 2 ;;
esac
export INPUT_NNODES INPUT_NODE_SELECTOR INPUT_NPROC_PER_NODE INPUT_TIMEOUT_MINUTES INPUT_IMAGE
export INPUT_COMMAND="exec bash /workspace/source/scripts/ci/ppu/run_recipe_pod.sh"
PPU_CI_WORKER_ARTIFACT_DIR="${pod_root%/}/$(basename -- "$raw")"
export PPU_CI_WORKER_ARTIFACT_DIR
USER_ENV_JSON=$(python3 -B - <<'PY'
import json
import os
import re
from pathlib import PurePosixPath

keys = ["TIER", "PPU_CI_SOURCE_SHA", "PPU_CI_WORKER_ARTIFACT_DIR"]
if os.environ["TIER"] == "legacy":
    keys += [
        "PPU_CI_TARGET_MODEL_PATH", "PPU_CI_EAGLE3_MODEL_PATH",
        "PPU_CI_DFLASH_MODEL_PATH", "PPU_CI_EVAL_CACHE",
    ]
else:
    keys += [
        "MODEL_KEY",
        "RECIPE_PPU_COUNT",
        "RECIPE_NNODES",
        "PPU_CI_RECIPE_IMAGE_KEY",
        "PPU_CI_HF_CACHE_ROOT",
        "PPU_CI_SKIP_WHEEL_INSTALL",
    ]
    if os.environ.get("PPU_CI_SKIP_WHEEL_INSTALL") == "true":
        keys += ["PPU_CI_RECIPE_TREE_SHA256"]
    if os.environ["TIER"] == "model_sanity":
        keys += ["PPU_CI_GSM8K_PATH"]
values = {key: os.environ[key] for key in keys}
if os.environ["TIER"] != "legacy":
    revision_key = {
        "glm-5.3": "PPU_CI_GLM53_MODEL_REVISION",
        "glm-5.3-tp4": "PPU_CI_GLM53_MODEL_REVISION",
        "deepseek-v4-flash": "PPU_CI_DSV4_FLASH_MODEL_REVISION",
        "deepseek-v4-flash-tp8": "PPU_CI_DSV4_FLASH_MODEL_REVISION",
    }.get(os.environ["MODEL_KEY"])
    if revision_key is not None:
        revision = os.environ[revision_key]
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("expected a pinned model snapshot")
        values["PPU_CI_HF_MODEL_REVISION"] = revision
for key, value in values.items():
    if not value or any(ord(char) < 32 for char in value):
        raise ValueError("invalid worker input")
    if key.startswith("PPU_CI_") and key not in (
        "PPU_CI_SOURCE_SHA",
        "PPU_CI_RECIPE_IMAGE_KEY",
        "PPU_CI_HF_MODEL_REVISION",
        "PPU_CI_SKIP_WHEEL_INSTALL",
        "PPU_CI_RECIPE_TREE_SHA256",
    ):
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
# Persist the cleanup record BEFORE writing the staging tarball or submitting
# so a hard kill during tar/submit always leaves durable state for failsafe cleanup.
mkdir -p -- "$(dirname -- "$cleanup_state")"
jq -n \
    --arg job_name "$JOB_NAME" \
    --arg namespace "$INPUT_NAMESPACE" \
    --argjson nnodes "$INPUT_NNODES" \
    --arg source_stage_dir "$SOURCE_STAGE_DIR" \
    '{job_name:$job_name, namespace:$namespace, nnodes:$nnodes, source_stage_dir:$source_stage_dir}' \
    > "${cleanup_state}.tmp"
mv -- "${cleanup_state}.tmp" "$cleanup_state"
[[ -s "$cleanup_state" ]]
tar -czf "$SOURCE_STAGE_DIR" --exclude='./.ppu-infra' --exclude='./.ppu-trusted-ci' \
    --exclude='./.git/config' \
    --exclude='./.git/config.worktree' --exclude='./.git/hooks' \
    --exclude='./.git/modules' --exclude='./.git/worktrees' -C "$GITHUB_WORKSPACE" .
run_backend "$infra/submit.sh"
[[ $(read_output job_status) == scheduled ]]
export JOB_START_TIME JOB_DEADLINE
JOB_START_TIME=$(read_output start_time)
JOB_DEADLINE=$(read_output deadline)
[[ "$JOB_START_TIME" =~ ^[0-9]+$ && "$JOB_DEADLINE" =~ ^[0-9]+$ ]]

phase cluster_worker
run_backend "$infra/monitor.sh"
INPUT_JOB_STATUS=$(read_output job_status)
[[ "$INPUT_JOB_STATUS" == succeeded ]]
queue_s=$(read_output queue_duration_seconds)
[[ "$queue_s" =~ ^[0-9]+$ ]] || exit 1
if [[ "$mode" != legacy ]]; then
    # Kubelet reports the resolved pullable image reference for every worker Pod.
    phase image_resolution
    collect_pod_statuses
    record_runtime_image
    jq --argjson queue "$queue_s" '.queue_s = $queue' "$raw/results/summary.json" > "$raw/summary.tmp"
    mv -- "$raw/summary.tmp" "$raw/results/summary.json"
fi
INPUT_WORKFLOW_STATUS=success
