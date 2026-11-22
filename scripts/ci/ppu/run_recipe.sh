#!/usr/bin/env bash
# The workflow redirects all output of this host launcher to a private file.
set -euo pipefail
umask 077

private=${PPU_CI_ARTIFACT_DIR:?}
wheel_dir=${PPU_CI_WHEEL_DIR:?}
tier=${TIER:?}
model=${MODEL_KEY:?}
source_sha=${PPU_CI_SOURCE_SHA:?}
host_python=${PPU_CI_HOST_PYTHON:-python3}
phase() { printf '%s\n' "$1" > "$private/recipe-phase.txt"; }

phase runner_preflight
[[ $(git rev-parse HEAD) == "$source_sha" ]]
[[ $(git status --porcelain=v1 --ignored --ignore-submodules=none) == "" ]]
case "$model" in
    glm-5.3) model_path=${PPU_CI_GLM53_MODEL_PATH:?} ;;
    deepseek-v4-flash) model_path=${PPU_CI_DSV4_FLASH_MODEL_PATH:?} ;;
    *) exit 2 ;;
esac
[[ "$tier" == fast_sanity || "$tier" == model_sanity ]]
repo=$(pwd -P)
model_path=$(realpath -e -- "$model_path")
dataset=""
if [[ "$tier" == model_sanity ]]; then
    dataset=$(realpath -e -- "${PPU_CI_GSM8K_PATH:?}")
    [[ -d "$dataset" ]]
fi
jit_root=$(realpath -e -- "${PPU_CI_JIT_ROOT:?}")
[[ -d "$jit_root" && -f "$model_path/config.json" ]]
for path in "$repo" "$wheel_dir" "$private" "$model_path" "$dataset" "$jit_root"; do
    [[ "$path" != *','* && "$path" != *$'\n'* && "$path" != *$'\r'* ]]
done

phase wheel_verification
"$host_python" -B scripts/ci/ppu/recipe_bundle.py check "$wheel_dir" "$source_sha"
manifest="$wheel_dir/recipe-manifest.json"

phase image_resolution
image=${PPU_CI_RECIPE_RUNTIME_IMAGE:-download.t-head.cn/docker_test/sglang:release-v0.5.23-20261116-cf244d4}
registry=download.t-head.cn/docker_test/sglang
if [[ "$image" != "$registry:release-v0.5.23-20261116-cf244d4" &&
    ! "$image" =~ ^download\.t-head\.cn/docker_test/sglang@sha256:[0-9a-f]{64}$ ]]; then
    exit 2
fi
if [[ "$image" == *@sha256:* ]]; then
    digest=$image
    if ! docker image inspect "$digest" >/dev/null 2>&1; then
        docker pull "$digest"
    fi
else
    docker pull "$image" > "$private/image-pull.log" 2>&1
    pulled=$(awk '$1 == "Digest:" {print $2}' "$private/image-pull.log")
    [[ "$pulled" =~ ^sha256:[0-9a-f]{64}$ ]]
    digest="$registry@$pulled"
fi
# Use the immutable local image identity from this inspected digest.
image_id=$(docker image inspect "$digest" --format '{{.Id}}')
[[ "$image_id" =~ ^sha256:[0-9a-f]{64}$ ]]
"$host_python" -B - "$manifest" "$digest" "$private/provenance.json" <<'PY'
import json
import sys
from pathlib import Path
manifest, digest, output = sys.argv[1:]
data = json.loads(Path(manifest).read_text())
Path(output).write_text(json.dumps({**data, "runtime_image": digest}) + "\n")
PY

phase runner_preflight
queue_start=$(date +%s)
# A multi-card owner takes the host lock and every legacy per-device lock.
# This coordinates with earlier one-card jobs that only take the latter.
exec 9>>/tmp/sglang-ppu-ci-host.lock
flock -w 900 9
device_args=(--device=/dev/alixpu --device=/dev/alixpu_ctl --device=/dev/alixpu_sep)
for device in {0..7}; do
    exec {lock_fd}>>"/tmp/sglang-ppu-ci-device-$device.lock"
    flock -w 900 "$lock_fd"
    node="/dev/alixpu_ppu$device"
    [[ -c "$node" ]]
    device_args+=(--device="$node")
done
queue_s=$(( $(date +%s) - queue_start + ${PPU_CI_JOB_QUEUE_S:-0} ))
ppu-smi -L > "$private/ppu-inventory.log"
[[ $(grep -ci 'M890P' "$private/ppu-inventory.log") == 8 ]]
[[ $(grep -c '^PPU [0-9]' "$private/ppu-inventory.log") == 8 ]]
ppu-smi > "$private/ppu-before.log"
ppu-smi --query-ppu=memory.used --format=csv,noheader,nounits > "$private/ppu-memory-before.log"
ppu-smi --query-compute-apps=pid --format=csv,noheader,nounits > "$private/ppu-processes-before.log"
if grep -Eq '[[:digit:]]' "$private/ppu-processes-before.log"; then
    exit 2
fi

# Mount the model read-only. Hugging Face snapshots need their sibling blobs.
model_root=$model_path
container_model=/assets/model
if [[ $(basename -- "$(dirname -- "$model_path")") == snapshots ]]; then
    model_root=$(dirname -- "$(dirname -- "$model_path")")
    container_model="/assets/model/snapshots/$(basename -- "$model_path")"
fi
model_mount=(--mount "type=bind,source=$model_root,target=/assets/model,readonly")
dataset_mount=()
if [[ -n "$dataset" ]]; then
    dataset_mount=(--mount "type=bind,source=$dataset,target=/assets/gsm8k,readonly")
fi

# Use an image-, candidate-recipe- and model-specific cache on the runner's local disk.
cache_key=$("$host_python" -B scripts/ci/ppu/recipe_bundle.py cache-key "$wheel_dir" "$source_sha")
jit="$jit_root/sglang-ci/${digest##*:}/$cache_key/$model"
mkdir -p -- "$jit"
job_tag=$(basename -- "$private")
[[ "$job_tag" =~ ^recipe\.[A-Za-z0-9]+$ ]]
container="sglang-ppu-$job_tag"
uid=$(id -u)
gid=$(id -g)
attached=""
cleanup() {
    code=$?
    trap - EXIT
    trap '' INT TERM
    set +e
    cleanup_failed=0
    if docker container inspect "$container" >/dev/null 2>&1; then
        docker stop --time=3 "$container" >/dev/null 2>&1
        docker rm --force "$container" >/dev/null 2>&1 || cleanup_failed=1
        docker run --rm --name "$container-ownership" --runtime=runc --network=none \
            --user 0:0 --env NVIDIA_VISIBLE_DEVICES=void --entrypoint chown \
            --mount "type=bind,source=$private,target=/artifacts" \
            "$image_id" -hR -- "$uid:$gid" /artifacts >> "$private/ownership.log" 2>&1 || cleanup_failed=1
    fi
    if [[ -n "$attached" ]]; then
        kill "$attached" 2>/dev/null
        wait "$attached" 2>/dev/null
    fi
    ppu-smi > "$private/ppu-after.log" || cleanup_failed=1
    ppu-smi --query-compute-apps=pid --format=csv,noheader,nounits > "$private/ppu-processes-after.log" || cleanup_failed=1
    ppu-smi --query-ppu=memory.used --format=csv,noheader,nounits > "$private/ppu-memory-after.log" || cleanup_failed=1
    if grep -Eq '[[:digit:]]' "$private/ppu-processes-after.log" ||
        ! cmp -s "$private/ppu-memory-before.log" "$private/ppu-memory-after.log"; then
        cleanup_failed=1
    fi
    if (( cleanup_failed != 0 )); then
        phase cleanup
        code=1
    fi
    exit "$code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

docker create --name "$container" --runtime=runc --network=none --shm-size=128g \
    "${device_args[@]}" "${model_mount[@]}" "${dataset_mount[@]}" \
    --mount "type=bind,source=$repo,target=/source,readonly" \
    --mount "type=bind,source=$wheel_dir,target=/wheels,readonly" \
    --mount "type=bind,source=$private,target=/artifacts" \
    --mount "type=bind,source=$jit,target=/jit" \
    --env NVIDIA_VISIBLE_DEVICES=void --env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    --env NCCL_SOCKET_IFNAME=lo --env GLOO_SOCKET_IFNAME=lo \
    --env "PPU_RECIPE_CONTAINER_MODEL=$container_model" \
    --entrypoint bash "$image_id" /source/scripts/ci/ppu/recipe_container.sh \
    "test/ppu/policy/models/$model.yaml" "$tier" --queue-s "$queue_s"
docker start --attach "$container" > "$private/runtime.log" 2>&1 &
attached=$!
wait "$attached"
attached=""
