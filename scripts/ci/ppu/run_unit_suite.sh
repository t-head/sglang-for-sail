#!/usr/bin/env bash
set -euo pipefail

backend=${1:?Expected cpu or ppu}
shard=${2:?Expected shard 0 or 1}
case "$backend" in
    cpu) suite=base-a-test-cpu-ppu ;;
    ppu) suite=extra-unit-test-1-gpu-ppu ;;
    *) exit 2 ;;
esac
[[ "$shard" =~ ^[01]$ ]]

export SGLANG_IS_IN_CI=1 PYTHONDONTWRITEBYTECODE=1
log=$(mktemp)
trap 'rm -f -- "$log"' EXIT
python3 -B test/run_suite.py --hw "$backend" --suite "$suite" \
    --auto-partition-id "$shard" --auto-partition-size 2 --continue-on-error \
    2>&1 | tee "$log"
# The suite runner can succeed when a shard selects no files.
grep -Eq '^Test Summary: [1-9][0-9]*/[1-9][0-9]* passed$' "$log"
