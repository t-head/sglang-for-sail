#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Failsafe cluster cleanup executed from an `if: always()` workflow step when
# the main orchestrator step is cancelled or killed before its EXIT trap finishes.
set -euo pipefail
umask 077

state_file="${RUNNER_TEMP:-/tmp}/ppu-cleanup-state.json"
verified_marker="${RUNNER_TEMP:-/tmp}/ppu-cleanup-verified.json"
if [[ ! -f "$state_file" ]]; then
    if [[ -f "$verified_marker" && ! -L "$verified_marker" ]]; then
        cat -- "$verified_marker"
        exit 0
    fi
    printf '{"cleanup_verified": false, "schema": "ppu-cleanup/v1", "source": "missing_state"}\n'
    echo "::error::Cluster cleanup state is missing and no verified cleanup marker was recorded"
    exit 1
fi

export PATH="/opt/bin:$PATH"
infra="${GITHUB_WORKSPACE:-$PWD}/.ppu-infra/cluster-run/scripts"
[[ ! -L "${GITHUB_WORKSPACE:-$PWD}/.ppu-infra" && -f "$infra/cleanup.sh" && ! -L "$infra/cleanup.sh" ]] || {
    echo "::error::Cluster cleanup backend missing for failsafe cleanup"
    exit 1
}

read_state() {
    python3 -I -B - "$state_file" "$1" <<'PY'
import json, re, sys
data = json.loads(open(sys.argv[1], encoding="utf-8").read())
key = sys.argv[2]
val = str(data.get(key, ""))
if key == "job_name" and not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,62}", val):
    raise SystemExit(2)
if key == "namespace" and not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", val):
    raise SystemExit(2)
if key == "nnodes" and not re.fullmatch(r"[0-9]+", val):
    raise SystemExit(2)
if key == "source_stage_dir" and val and (not val.startswith("/") or ".." in val.split("/")):
    raise SystemExit(2)
print(val)
PY
}

export JOB_NAME INPUT_NAMESPACE INPUT_NNODES SOURCE_STAGE_DIR
JOB_NAME=$(read_state job_name)
INPUT_NAMESPACE=$(read_state namespace)
INPUT_NNODES=$(read_state nnodes)
SOURCE_STAGE_DIR=$(read_state source_stage_dir)
export INPUT_LAUNCH_MODE=each-node INPUT_CLEANUP_POLICY=always
export INPUT_JOB_STATUS=cancelled INPUT_WORKFLOW_STATUS=cancelled
export GITHUB_OUTPUT="${RUNNER_TEMP:-/tmp}/ppu-cleanup-failsafe.out"
: > "$GITHUB_OUTPUT"

private_log="${RUNNER_TEMP:-/tmp}/ppu-cleanup-failsafe.log"
cleanup_rc=0
bash "$infra/cleanup.sh" > "$private_log" 2>&1 || cleanup_rc=$?

count_lines() {
    if [[ -z "$1" ]]; then
        printf '0\n'
    else
        printf '%s\n' "$1" | wc -l | tr -d ' '
    fi
}

run_kubectl() {
    if command -v timeout >/dev/null 2>&1; then
        timeout 15s kubectl "$@"
    else
        kubectl "$@"
    fi
}

rem_pods=$(run_kubectl -n "$INPUT_NAMESPACE" get pods -l "ppu-job=$JOB_NAME" -o name 2>/dev/null || printf '__error__')
leftover_pods=1
if [[ "$rem_pods" != "__error__" ]]; then
    leftover_pods=$(count_lines "$rem_pods")
fi

leftover_services=0
leftover_podgroups=0
if (( INPUT_NNODES > 0 )); then
    rem_svc=$(run_kubectl -n "$INPUT_NAMESPACE" get svc "$JOB_NAME" --ignore-not-found -o name 2>/dev/null || printf '__error__')
    rem_pg=$(run_kubectl -n "$INPUT_NAMESPACE" get podgroups.scheduling.x-k8s.io "$JOB_NAME" --ignore-not-found -o name 2>/dev/null || printf '__error__')
    leftover_services=1
    leftover_podgroups=1
    [[ "$rem_svc" != "__error__" ]] && leftover_services=$(count_lines "$rem_svc")
    [[ "$rem_pg" != "__error__" ]] && leftover_podgroups=$(count_lines "$rem_pg")
fi

leftover_staging=0
stage_parent=""
stage_parent_is_job_dir=false
if [[ -n "$SOURCE_STAGE_DIR" ]]; then
    stage_parent=$(dirname -- "$SOURCE_STAGE_DIR")
    if [[ "$(basename -- "$SOURCE_STAGE_DIR")" == "source.tar.gz" &&
        "$(basename -- "$stage_parent")" == "${JOB_NAME}."* ]]; then
        stage_parent_is_job_dir=true
        rm -f -- "$SOURCE_STAGE_DIR" || true
        rmdir -- "$stage_parent" 2>/dev/null || true
    fi
    if [[ -e "$SOURCE_STAGE_DIR" ]] || [[ "$stage_parent_is_job_dir" == true && -e "$stage_parent" ]]; then
        leftover_staging=1
    fi
fi

verified=false
if (( cleanup_rc == 0 && leftover_pods == 0 && leftover_services == 0 && leftover_podgroups == 0 && leftover_staging == 0 )); then
    marker_tmp="${verified_marker}.tmp.$$"
    if printf '{"cleanup_verified": true, "leftover_podgroups": 0, "leftover_pods": 0, "leftover_services": 0, "leftover_staging": 0, "nnodes": %d, "schema": "ppu-cleanup/v1", "source": "always_failsafe"}\n' \
        "$INPUT_NNODES" > "$marker_tmp" && mv -- "$marker_tmp" "$verified_marker"; then
        verified=true
        rm -f -- "$state_file"
    else
        rm -f -- "$marker_tmp" 2>/dev/null || true
    fi
fi

printf '{"cleanup_verified": %s, "leftover_podgroups": %d, "leftover_pods": %d, "leftover_services": %d, "leftover_staging": %d, "nnodes": %d, "schema": "ppu-cleanup/v1", "source": "always_failsafe"}\n' \
    "$verified" "$leftover_podgroups" "$leftover_pods" "$leftover_services" "$leftover_staging" "$INPUT_NNODES"

if [[ "$verified" != true ]]; then
    echo "::error::Failsafe cluster cleanup left resources behind"
    exit 1
fi
