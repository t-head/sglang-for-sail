"""CPU-only unit test for the DSA fp8 KV cache argument guard.

What it pins
    On a PPU (sail) build with ``--kv-cache-dtype fp8_e4m3``, the post-process
    guard ``_ppu_flashmla_fp8_kv_cache_fallback`` decides whether the fp8 KV store
    is kept or silently downgraded to compute dtype ("auto"). Only two decisions
    are safe to keep as fp8:
      * both phases on ``flashmla_kv`` -- the fp8-capable KV kernel reads the pool
        directly;
      * ``flashmla_sparse`` prefill + ``flashmla_kv`` decode -- the sparse kernel
        consumes bf16 and the framework dequantizes the fp8 prefix KV in front of
        it, while the pool stays fp8 (quantize on write), so the capacity win and
        the fp8 decode path are both preserved.
    Everything else (other phase pairs, sub-SM89 capability, the dense flashmla
    decode backend, non-DSA attention) must keep falling back to auto.

Run inside a PPU task container with the branch under test installed:
    python3 test/ppu/unit/test_ppu_dsa_fp8_kv_guard_sparse_prefill.py -v
No model weights, server, device work or pytest dependency is required; the guard
is pure argument post-processing. ``is_sail``, platform capability and the backend
pair are injected per case, so the matrix is deterministic on any host that can
import sglang.

Exit code 0 = all cases behave as specified; non-zero prints the offending case.
"""

from __future__ import annotations

import argparse
import sys
import types


class _Platform:
    def __init__(self, device_sm: int) -> None:
        self.device_sm = device_sm


def _view(**kwargs):
    base = dict(
        attention_backend="dsa",
        decode_attention_backend=None,
        dsa_prefill_backend="flashmla_sparse",
        dsa_decode_backend="flashmla_kv",
        kv_cache_dtype="fp8_e4m3",
    )
    base.update(kwargs)
    return types.SimpleNamespace(**base)


def _run(guard, *, sail: bool, sm: int, view) -> dict:
    """Call the guard with is_sail / capability / backend pair injected."""
    import sglang.srt.arg_groups.overrides as overrides
    import sglang.srt.utils.common as utils_common

    saved = (
        overrides.get_platform,
        overrides.attention_backends_of,
        utils_common.is_sail,
    )
    overrides.get_platform = lambda: _Platform(sm)
    overrides.attention_backends_of = lambda v: (v.attention_backend,)
    utils_common.is_sail = lambda: sail
    try:
        return dict(guard(view))
    finally:
        (
            overrides.get_platform,
            overrides.attention_backends_of,
            utils_common.is_sail,
        ) = saved


CASES = [
    # name, kwargs, expectation
    (
        "sm89_sparse_prefill_kv_decode_keeps_fp8",
        dict(sail=True, sm=89, view=_view()),
        {},
    ),
    (
        "sm89_both_phases_flashmla_kv_keeps_fp8",
        dict(
            sail=True,
            sm=89,
            view=_view(dsa_prefill_backend="flashmla_kv"),
        ),
        {},
    ),
    (
        "sm89_sparse_prefill_sparse_decode_still_falls_back",
        dict(
            sail=True,
            sm=89,
            view=_view(dsa_decode_backend="flashmla_sparse"),
        ),
        {"kv_cache_dtype": "auto"},
    ),
    (
        "sm89_auto_prefill_still_falls_back",
        dict(
            sail=True,
            sm=89,
            view=_view(dsa_prefill_backend="flashmla_auto"),
        ),
        {"kv_cache_dtype": "auto"},
    ),
    (
        "sub_sm89_sparse_prefill_falls_back",
        dict(sail=True, sm=80, view=_view()),
        {"kv_cache_dtype": "auto"},
    ),
    (
        "dense_flashmla_decode_falls_back",
        dict(
            sail=True,
            sm=89,
            view=_view(
                decode_attention_backend="flashmla",
                attention_backend="flashmla",
            ),
        ),
        {"kv_cache_dtype": "auto"},
    ),
    (
        "compute_dtype_kv_untouched",
        dict(sail=True, sm=89, view=_view(kv_cache_dtype="auto")),
        {},
    ),
    (
        "non_sail_platform_untouched",
        dict(sail=False, sm=89, view=_view()),
        {},
    ),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    from sglang.srt.arg_groups.overrides import _ppu_flashmla_fp8_kv_cache_fallback

    failures = 0
    for name, kwargs, expected in CASES:
        got = _run(_ppu_flashmla_fp8_kv_cache_fallback, **kwargs)
        ok = got == expected
        failures += 0 if ok else 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: got={got} expected={expected}")
    print(f"RESULT: {'OK' if failures == 0 else f'{failures} FAILED'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
