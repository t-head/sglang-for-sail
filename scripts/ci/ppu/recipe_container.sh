#!/usr/bin/env bash
set -euo pipefail
umask 077
artifact_dir=${PPU_CI_ARTIFACT_DIR:-/artifacts}
wheel_dir=${PPU_CI_WHEEL_DIR:-/wheels}
source_dir=${PPU_CI_RECIPE_SOURCE_DIR:-/source}
jit_dir=${PPU_CI_RECIPE_JIT_DIR:-/jit}

phase() { printf '%s\n' "$1" > "$artifact_dir/worker-phase.txt"; }
phase runtime_dependencies
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
export HF_HOME="$artifact_dir/hf-cache" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
if [[ -n "${PPU_CI_HF_CACHE_ROOT:-}" ]]; then
    export HF_HUB_CACHE="$PPU_CI_HF_CACHE_ROOT/hub"
    export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
fi
export TRANSFORMERS_OFFLINE=1 SGLANG_IS_IN_CI=1 PYTHONDONTWRITEBYTECODE=1
export SGLANG_DG_CACHE_DIR="$jit_dir/sglang" DG_JIT_CACHE_DIR="$jit_dir/deepgemm"
export DG_CACHE_DIR="$jit_dir/deepgemm" TRITON_CACHE_DIR="$jit_dir/triton"
mkdir -p "$SGLANG_DG_CACHE_DIR" "$DG_JIT_CACHE_DIR" "$TRITON_CACHE_DIR"
cd "$source_dir"
checker=scripts/ci/ppu/check_dependencies.py
python3 -I -B "$checker" snapshot "$artifact_dir/dependencies-before.json"
# The release image owns the SDK, PyTorch, DeepGEMM, FlashMLA and EvalScope.
python3 -I -B -m pip uninstall -y sgl-kernel >/dev/null 2>&1 || true
python3 -I -B -m pip install --no-deps --no-index --force-reinstall "$wheel_dir/"*.whl
missing=$(python3 -I -B "$checker" missing-wheels "$wheel_dir/runtime-deps")
if [[ -n "$missing" ]]; then
    mapfile -t extra <<< "$missing"
    python3 -I -B -m pip install --no-deps --no-index "${extra[@]}"
fi
python3 -I -B "$checker" snapshot "$artifact_dir/dependencies-after.json"
python3 -I -B "$checker" check "$artifact_dir/dependencies-before.json" "$artifact_dir/dependencies-after.json"
if [[ "${2:-}" == model_sanity && -n "${PPU_CI_EVAL_WHEEL_DIR:-}" ]]; then
    phase eval_environment
    (cd "$PPU_CI_EVAL_WHEEL_DIR" && sha256sum -- *.whl | cmp -s SHA256SUMS -)
    eval_dir=$(mktemp -d)
    trap 'rm -rf -- "$eval_dir"' EXIT
    python3 -I -B -m pip install --no-index --ignore-installed --target "$eval_dir" \
        --find-links "$PPU_CI_EVAL_WHEEL_DIR" 'evalscope==1.15.0'
    export PPU_CI_EVAL_SITE="$eval_dir"
    python3 -I -S -B -c 'import sys; sys.path.insert(0, sys.argv[1]); from evalscope import TaskConfig, run_task' "$eval_dir"
fi
phase recipe_validation
python3 -I -B scripts/ci/ppu/ppu_recipe.py validate --all --require-sglang-parser
if [[ "${1:-}" == validate ]]; then
    exit 0
fi
phase platform_check
python3 -I -B - <<'PY'
import os

import torch
from sglang.srt.platforms import current_platform

expected_width = int(os.environ.get("RECIPE_PPU_COUNT", "8"))
assert current_platform.is_sail(), "candidate must select the SAIL platform"
assert torch.cuda.device_count() == expected_width, "candidate device count must match the Recipe allocation"
PY
# The wrapper adds only the checked-in executor directory to the isolated path.
python3 -I -B - "$@" <<'PY'
import runpy
import sys
from pathlib import Path
executor = Path.cwd() / "scripts/ci/ppu"
sys.path.insert(0, str(executor))
runpy.run_path(str(executor / "recipe_runner.py"), run_name="__main__")
PY
