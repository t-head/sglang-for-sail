#!/usr/bin/env bash
set -euo pipefail
umask 077

phase() { printf '%s\n' "$1" > /artifacts/recipe-phase.txt; }
phase runtime_dependencies
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
export HF_HOME=/artifacts/hf-cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1 SGLANG_IS_IN_CI=1 PYTHONDONTWRITEBYTECODE=1
export SGLANG_DG_CACHE_DIR=/jit/sglang DG_JIT_CACHE_DIR=/jit/deepgemm
export DG_CACHE_DIR=/jit/deepgemm TRITON_CACHE_DIR=/jit/triton
mkdir -p "$SGLANG_DG_CACHE_DIR" "$DG_JIT_CACHE_DIR" "$TRITON_CACHE_DIR"
cd /source
checker=scripts/ci/ppu/check_dependencies.py
python3 -I -B "$checker" snapshot /artifacts/dependencies-before.json
# The release image owns the SDK, PyTorch, DeepGEMM, FlashMLA and EvalScope.
python3 -I -B -m pip install --no-deps --no-index --force-reinstall /wheels/*.whl
missing=$(python3 -I -B "$checker" missing-wheels /wheels/runtime-deps)
if [[ -n "$missing" ]]; then
    mapfile -t extra <<< "$missing"
    python3 -I -B -m pip install --no-deps --no-index "${extra[@]}"
fi
python3 -I -B "$checker" snapshot /artifacts/dependencies-after.json
python3 -I -B "$checker" check /artifacts/dependencies-before.json /artifacts/dependencies-after.json
phase recipe_validation
python3 -I -B scripts/ci/ppu/ppu_recipe.py validate --all --require-sglang-parser
if [[ "${1:-}" == validate ]]; then
    exit 0
fi
phase platform_check
python3 -I -B - <<'PY'
import torch
from sglang.srt.platforms import current_platform
assert current_platform.is_sail(), "candidate must select the SAIL platform"
assert torch.cuda.device_count() == 8, "candidate must see eight devices"
PY
# The wrapper adds only the checked-in executor directory to the isolated path.
python3 -I -B - "$@" <<'PY'
import runpy
import sys
sys.path.insert(0, "/source/scripts/ci/ppu")
runpy.run_path("/source/scripts/ci/ppu/recipe_runner.py", run_name="__main__")
PY
