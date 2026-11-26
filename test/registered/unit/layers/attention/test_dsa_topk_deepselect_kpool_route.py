"""CPU gate suite for the kpool group top-k DeepSelect route.

The route replaces only the pooled group selection, so what has to be pinned is when it
takes over and when it declines -- and every decline must be visible, because a run that
silently never reached DeepSelect would otherwise be measured as if it had. No device is
needed here: the selection itself is stubbed and the assertions are about the gate.

The dtype case is worth spelling out: the PPU scoring entry returns float32 pooled
scores (the build measured here exposes no BF16 output option on its weighted entries),
so the route expects float32 in and stages to BF16 inside ``topk_kpool_groups``. A
non-float32 row therefore means "this is not the pooled producer we were wired to", and
the route declines rather than rewriting someone else's tensor.
"""

from __future__ import annotations

import logging
from contextlib import ExitStack, contextmanager
from unittest import mock

import pytest
import torch

from sglang.kernels.ops.attention import deep_select as wheel_gate
from sglang.kernels.ops.attention.dsa import deep_select as ds
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa import dsa_topk_backend as tb
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

GROUP_TOPK = 512  # topk 2048 at index_kpool 4: the served bucket for Flash shapes


@pytest.fixture(autouse=True)
def _fresh_log_state():
    """The once-per-reason ledger is module state; every case starts from empty."""
    tb._DEEPSELECT_LOGGED.clear()
    yield
    tb._DEEPSELECT_LOGGED.clear()


def _rows(rows: int, cols: int, dtype=torch.float32) -> torch.Tensor:
    # Values are irrelevant to the gate. Filling 4096x8192 float32 on every case would
    # cost 128 MiB and a memset each time, so the buffer is only allocated.
    return torch.zeros(rows, cols, dtype=dtype)


@contextmanager
def gate(switch=True, ppu=True, wheel=True):
    """Reproduce the state a real PPU serve has: switch on, PPU, wheel reported present.

    ``_wheel_available`` is the upstream capability probe. In the engine it turns True
    only after ``load_plugins()`` has applied the PPU hooks, so a CPU suite patches that
    same probe instead of re-implementing the capability rule.
    """
    with ExitStack() as stack:
        stack.enter_context(
            mock.patch.object(envs.SGLANG_DSA_DEEPSELECT, "get", return_value=switch)
        )
        stack.enter_context(mock.patch.object(tb, "is_sail", return_value=ppu))
        stack.enter_context(
            mock.patch.object(wheel_gate, "_wheel_available", return_value=wheel)
        )
        yield


def _call(logits, group_topk=GROUP_TOPK, lengths=None, row_starts=None):
    if lengths is None:
        lengths = torch.full((logits.shape[0],), logits.shape[1], dtype=torch.int32)
    return tb.deepselect_kpool_group_ids(
        logits, lengths, group_topk, row_starts=row_starts
    )


def _never_reached():
    def _boom(*args, **kwargs):
        raise AssertionError("the wheel must not be called on a declined route")

    return mock.patch.object(ds, "topk_kpool_groups", _boom)


def test_switch_off_short_circuits_before_any_check(caplog):
    with gate(switch=False), _never_reached(), caplog.at_level(logging.DEBUG):
        assert _call(_rows(4096, 8192)) is None
        assert "deep_select" not in caplog.text


def test_missing_wheel_declines_and_says_why(caplog):
    with gate(wheel=False), _never_reached(), caplog.at_level(logging.WARNING):
        assert _call(_rows(4096, 8192)) is None
    assert "deep_select wheel" in caplog.text


def test_non_ppu_device_is_not_routed(caplog):
    with gate(ppu=False), _never_reached(), caplog.at_level(logging.WARNING):
        assert _call(_rows(4096, 8192)) is None
    assert "not a PPU device" in caplog.text


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_gate_predicates_on_the_dtype_the_producer_ships(dtype):
    """float32 rows are the served case; other widths decline with a stated reason."""
    with gate():
        reason = tb._deepselect_kpool_skip_reason(_rows(2048, 8192, dtype), GROUP_TOPK)
        if dtype is torch.float32:
            assert reason is None
        else:
            assert "float32" in reason


def test_scores_must_be_a_matrix():
    with gate():
        reason = tb._deepselect_kpool_skip_reason(
            torch.zeros(4, 2048, 8192), GROUP_TOPK
        )
        assert "rank" in reason


@pytest.mark.parametrize(
    "rows,cols,served",
    [
        (2048, 8192, True),  # the lower-left corner of the measured band
        (2047, 8192, False),
        (2048, 8191, False),
        (4096, 8192, True),
        (4096, 16384, True),
        (1, 64, False),
        (512, 1024, False),
    ],
)
def test_served_band_edges_are_the_measured_crossover(rows, cols, served):
    with gate():
        reason = tb._deepselect_kpool_skip_reason(_rows(rows, cols), GROUP_TOPK)
        assert (reason is None) is served, reason
        if not served:
            assert "below the kpool served band" in reason


def test_served_band_constants_are_the_documented_pair():
    """The band is a measured crossover, so a silent widening has to fail here."""
    assert (ds.KPOOL_SERVE_MIN_ROWS, ds.KPOOL_SERVE_MIN_COLS) == (2048, 8192)
    assert ds.KPOOL_SUPPORTED_GROUP_TOPK == (128, 160, 192, 224, 256, 512)


def test_bucket_outside_the_fused_kernel_list_declines():
    with gate():
        reason = tb._deepselect_kpool_skip_reason(_rows(4096, 16384), 96)
        assert "not a served kpool bucket" in reason


def test_shape_predicate_and_gate_agree():
    """The host-side predicate and the routed gate must never disagree."""
    with gate():
        for rows, cols in ((64, 1024), (2048, 8192), (2048, 4096), (4096, 8192)):
            for group_topk in ds.KPOOL_SUPPORTED_GROUP_TOPK + (96,):
                logits = _rows(rows, cols)
                gate_declines = (
                    tb._deepselect_kpool_skip_reason(logits, group_topk) is not None
                )
                predicate_declines = (
                    ds.kpool_group_topk_shape_reason(rows, cols, group_topk) is not None
                )
                assert gate_declines == predicate_declines


def test_in_band_call_returns_the_selection_untouched_with_the_hit_marker(caplog):
    sentinel = torch.arange(GROUP_TOPK, dtype=torch.int32).reshape(1, GROUP_TOPK)
    logits = _rows(4096, 8192)
    lengths = torch.full((4096,), 8192, dtype=torch.int32)
    with gate(), caplog.at_level(logging.INFO):
        with mock.patch.object(ds, "topk_kpool_groups", return_value=sentinel) as stub:
            got = tb.deepselect_kpool_group_ids(logits, lengths, GROUP_TOPK)
        assert got is sentinel
        args, kwargs = stub.call_args
        assert args[0] is logits and args[1] is lengths and args[2] == GROUP_TOPK
        assert kwargs == {"row_starts": None}
    assert "DSA kpool top-k served by PPU deep_select" in caplog.text
    assert "rows=4096 cols=8192 group_topk=512" in caplog.text


def test_hit_marker_and_each_reason_are_logged_once(caplog):
    """Repeated calls must not spam the hot path, and a decline must stay visible."""
    sentinel = torch.zeros(2048, GROUP_TOPK, dtype=torch.int32)
    with gate(), caplog.at_level(logging.INFO):
        with mock.patch.object(ds, "topk_kpool_groups", return_value=sentinel):
            for _ in range(3):
                assert _call(_rows(4096, 8192)) is not None
        hits = [r for r in caplog.records if "served by PPU deep_select" in r.message]
        assert len(hits) == 1, "the hit line is one per run, not one per call"

        with _never_reached():
            for _ in range(3):
                assert _call(_rows(64, 128)) is None
        misses = [
            r for r in caplog.records if "below the kpool served band" in r.message
        ]
        assert len(misses) == 1, "a repeated decline logs one line"


def test_row_starts_are_forwarded_to_the_selection():
    starts = torch.zeros(4096, dtype=torch.int32) + 512
    lengths = torch.full((4096,), 7680, dtype=torch.int32)
    with gate():
        with mock.patch.object(ds, "topk_kpool_groups", return_value=starts) as stub:
            got = tb.deepselect_kpool_group_ids(
                _rows(4096, 8192), lengths, GROUP_TOPK, row_starts=starts
            )
        assert got is starts
        assert stub.call_args.kwargs == {"row_starts": starts}
