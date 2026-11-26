# SPDX-License-Identifier: Apache-2.0
"""Run the registered single-device PPU model, numerical and benchmark files."""

from pathlib import Path

from sglang.test.ci.ci_register import HWBackend, collect_tests

FILES = (
    "test/registered/core/test_basic_sanity_eagle3.py",
    "test/registered/core/test_basic_sanity_dflash.py",
    "test/registered/kernels/ops/attention/test_flashkda_replayssm_ppu.py",
    "test/registered/kernels/benchmark/attention/bench_flashkda_replayssm_ppu.py",
)
SUITE = "stage-a-test-1-gpu-ppu"


def select_tests(repo):
    paths = {str(repo / path) for path in FILES}
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
            "all selected model, numerical and benchmark files must be registered "
            "for the PPU lane"
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
