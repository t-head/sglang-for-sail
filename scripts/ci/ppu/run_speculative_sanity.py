# SPDX-License-Identifier: Apache-2.0
"""Run a selected registered single-device PPU smoke or FP8 operator suite."""

import os
from pathlib import Path

from sglang.test.ci.ci_register import HWBackend, collect_tests

FILES = (
    "test/registered/core/test_basic_sanity_eagle3.py",
    "test/registered/core/test_basic_sanity_dflash.py",
)
FP8_FILES = (
    "test/registered/kernels/ops/moe/test_fp8_mega_moe_ppu.py",
    "test/registered/kernels/benchmark/moe/bench_fp8_mega_moe_ppu.py",
)
SUITE = "stage-a-test-1-gpu-ppu"


def select_tests(repo):
    files = FP8_FILES if os.environ.get("PPU_CI_FP8_ONLY") == "1" else FILES
    paths = {str(repo / path) for path in files}
    tests = [
        test
        for test in collect_tests(sorted(paths))
        if test.backend == HWBackend.PPU
        and test.effective_suite == SUITE
        and not test.disabled
        and not test.nightly
    ]
    if len(tests) != len(paths) or {test.filename for test in tests} != paths:
        raise ValueError(
            "all selected single-device files must be registered for the PPU lane"
        )
    return tests


def main():
    from sglang.test.ci.ci_utils import run_unittest_files

    repo = Path(__file__).resolve().parents[3]
    return run_unittest_files(
        select_tests(repo), timeout_per_file=1200, continue_on_error=True
    )


if __name__ == "__main__":
    raise SystemExit(main())
