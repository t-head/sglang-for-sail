#!/usr/bin/env bash
set -euo pipefail

: "${PPU_CI_ARTIFACT_DIR:?Set a directory for build and test logs}"
mkdir -p "$PPU_CI_ARTIFACT_DIR"
PPU_CI_ARTIFACT_DIR=$(realpath "$PPU_CI_ARTIFACT_DIR")
exec > >(tee "$PPU_CI_ARTIFACT_DIR/ci.log") 2>&1

: "${PPU_CI_BUILD_IMAGE:?Set the native cuda-free PPU build image on the runner}"
: "${PPU_CI_RUNTIME_IMAGE:?Set the CUDA-compatible PPU runtime image on the runner}"
: "${PPU_CI_TARGET_MODEL_PATH:?Set the target checkpoint path on the Docker host}"
: "${PPU_CI_EAGLE3_MODEL_PATH:?Set the EAGLE3 draft checkpoint path on the Docker host}"
: "${PPU_CI_DFLASH_MODEL_PATH:?Set the DFlash draft checkpoint path on the Docker host}"
: "${PPU_CI_EVAL_CACHE:?Set the shared sgl-eval 0.1.2 cache path on the Docker host}"
PPU_CI_DEVICE=${PPU_CI_DEVICE:-${NVIDIA_VISIBLE_DEVICES:-}}
if [[ ! "$PPU_CI_DEVICE" =~ ^[0-9]+$ ]]; then
    echo "PPU_CI_DEVICE must select exactly one numeric device" >&2
    exit 2
fi
for model in "$PPU_CI_TARGET_MODEL_PATH" "$PPU_CI_EAGLE3_MODEL_PATH" "$PPU_CI_DFLASH_MODEL_PATH"; do
    [[ "$model" == /* && -f "$model/config.json" ]]
done
[[ "$PPU_CI_EVAL_CACHE" == /* && -f "$PPU_CI_EVAL_CACHE/mmlu/test.jsonl" ]]

repo=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
for path in "$repo" "$PPU_CI_TARGET_MODEL_PATH" "$PPU_CI_EAGLE3_MODEL_PATH" "$PPU_CI_DFLASH_MODEL_PATH" "$PPU_CI_EVAL_CACHE" "$PPU_CI_ARTIFACT_DIR"; do
    if [[ "$path" == *,* ]]; then
        echo "Docker mount paths must not contain commas" >&2
        exit 2
    fi
done
wheel_dir="$PPU_CI_ARTIFACT_DIR/wheels"
mkdir -p "$wheel_dir"
for wheel in "$wheel_dir"/*.whl "$wheel_dir"/.*.whl; do
    if [[ -e "$wheel" || -L "$wheel" ]]; then
        echo "Wheel output must be fresh; found preexisting wheel: $wheel" >&2
        exit 2
    fi
done
for device in alixpu alixpu_ctl alixpu_sep "alixpu_ppu$PPU_CI_DEVICE"; do
    [[ -c "/dev/$device" ]] || { echo "Missing PPU device: /dev/$device" >&2; exit 1; }
done

container_prefix="sglang-ppu-ci-$(id -u)-$$-$RANDOM"
container=""
cleanup() {
    code=$?
    trap - EXIT
    # Successful --rm runs are gone; only our active container can remain.
    if [[ -n "$container" ]] && docker container inspect "$container" >/dev/null 2>&1; then
        docker stop --time=30 "$container" >/dev/null 2>&1 || true
    fi
    exit "$code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

source_sha=$(git -C "$repo" rev-parse HEAD)
printf '%s\n' "$source_sha" > "$PPU_CI_ARTIFACT_DIR/source-sha.txt"
git -C "$repo" diff HEAD > "$PPU_CI_ARTIFACT_DIR/source.patch"
docker image inspect "$PPU_CI_BUILD_IMAGE" --format '{{.Id}}' > "$PPU_CI_ARTIFACT_DIR/build-image-id.txt"
docker image inspect "$PPU_CI_RUNTIME_IMAGE" --format '{{.Id}}' > "$PPU_CI_ARTIFACT_DIR/runtime-image-id.txt"

# Torch stays image-local; runtime additions never replace preinstalled dependencies.
container="$container_prefix-build"
docker run --rm --name "$container" --shm-size=16g \
    --mount "type=bind,source=$repo,target=/source,readonly" \
    --mount "type=bind,source=$wheel_dir,target=/wheels" \
    --mount "type=bind,source=$PPU_CI_ARTIFACT_DIR,target=/artifacts" \
    --env NVIDIA_VISIBLE_DEVICES=void \
    --env CUDA_VISIBLE_DEVICES= \
    --env PPU_CI_WHEEL_DIR=/wheels \
    --env PPU_CI_ARTIFACT_DIR=/artifacts \
    --env SGLANG_IS_IN_CI=1 \
    --entrypoint bash "$PPU_CI_BUILD_IMAGE" /source/scripts/ci/ppu/build_and_test.sh build

container="$container_prefix-test"
mkdir -p "$PPU_CI_ARTIFACT_DIR/runtime"
docker run --rm --name "$container" --shm-size=16g \
    --device=/dev/alixpu --device=/dev/alixpu_ctl --device=/dev/alixpu_sep \
    --device="/dev/alixpu_ppu$PPU_CI_DEVICE" \
    --mount "type=bind,source=$repo,target=/source,readonly" \
    --mount "type=bind,source=$wheel_dir,target=/wheels,readonly" \
    --mount "type=bind,source=$PPU_CI_TARGET_MODEL_PATH,target=/models/target,readonly" \
    --mount "type=bind,source=$PPU_CI_EAGLE3_MODEL_PATH,target=/models/eagle3,readonly" \
    --mount "type=bind,source=$PPU_CI_DFLASH_MODEL_PATH,target=/models/dflash,readonly" \
    --mount "type=bind,source=$PPU_CI_EVAL_CACHE,target=/eval-cache,readonly" \
    --mount "type=bind,source=$PPU_CI_ARTIFACT_DIR/runtime,target=/artifacts" \
    --env CUDA_VISIBLE_DEVICES=0 \
    --env PPU_CI_WHEEL_DIR=/wheels \
    --env PPU_CI_TARGET_MODEL_PATH=/models/target \
    --env PPU_CI_EAGLE3_MODEL_PATH=/models/eagle3 \
    --env PPU_CI_DFLASH_MODEL_PATH=/models/dflash \
    --env PPU_CI_EVAL_CACHE=/eval-cache \
    --env PPU_CI_ARTIFACT_DIR=/artifacts \
    --env SGLANG_IS_IN_CI=1 \
    --entrypoint bash "$PPU_CI_RUNTIME_IMAGE" /source/scripts/ci/ppu/build_and_test.sh test
container=""
