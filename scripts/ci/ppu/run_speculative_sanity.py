# SPDX-License-Identifier: Apache-2.0
"""Run the two registered single-device PPU smoke files."""

from pathlib import Path

from sglang.test.ci.ci_register import HWBackend, collect_tests

FILES = (
    "test/registered/core/test_basic_sanity_eagle3.py",
    "test/registered/core/test_basic_sanity_dflash.py",
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
            "both speculative smoke files must be registered for the PPU lane"
        )
    return tests


def main():
    from sglang.test.ci.ci_utils import run_unittest_files

    repo = Path(__file__).resolve().parents[3]
    return run_unittest_files(select_tests(repo), timeout_per_file=1200)


if __name__ == "__main__":
    raise SystemExit(main())
