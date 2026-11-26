# SPDX-License-Identifier: Apache-2.0
"""Run the registered single-device PPU smoke and native-Q8 files."""

import os
from pathlib import Path

from sglang.test.ci.ci_register import HWBackend, collect_tests

SMOKE_FILES = (
    "test/registered/core/test_basic_sanity_eagle3.py",
    "test/registered/core/test_basic_sanity_dflash.py",
)
NATIVE_Q8_FILES = (
    "test/registered/kernels/ops/attention/test_dsa_native_q8_ppu.py",
    "test/registered/kernels/benchmark/attention/bench_dsa_native_q8_ppu.py",
)
SM89_BOARD = "ZW-M890P"
SUITE = "stage-a-test-1-gpu-ppu"


def selected_files(board):
    return SMOKE_FILES + (NATIVE_Q8_FILES if board == SM89_BOARD else ())


def select_tests(repo, board=None):
    board = os.environ.get("PPU_CI_BOARD", SM89_BOARD) if board is None else board
    paths = {str(repo / path) for path in selected_files(board)}
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
            "all selected single-device smoke and native-Q8 files must be "
            "registered for the PPU lane"
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
