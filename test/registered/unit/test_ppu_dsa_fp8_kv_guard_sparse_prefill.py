"""CPU-only unit test for the DSA fp8 KV-cache argument guard and launch validators.

What it pins
    On a PPU (sail) build with ``--kv-cache-dtype fp8_e4m3``, the post-process guard
    ``_ppu_flashmla_fp8_kv_cache_fallback`` decides whether the fp8 KV store is kept
    or silently downgraded to compute dtype ("auto"). Only these decisions keep fp8:
      * both phases on ``flashmla_kv`` (device_sm >= 89) -- the fp8-capable KV kernel
        reads the pool directly; this exemption predates the sparse-prefill one and
        its ``>= 89`` gate and mixed-chunk behaviour are deliberately unchanged;
      * ``flashmla_sparse`` prefill + ``flashmla_kv`` decode, and only when all four
        conditions hold: the resolved attention backend of *both* phases is ``dsa``,
        device_sm is exactly SM89, ``--enable-mixed-chunk`` is off and
        ``--enable-hisparse`` is off. The sparse kernel consumes bf16 and the
        framework dequantizes the fp8 prefix KV in front of it while the pool stays
        fp8 (quantize on write), so the capacity win and the fp8 decode path both
        survive on that combination.
    Each rejected condition is covered by its own row: above SM89 the k-pool tail
    route may hand the sparse prefill to another kernel; a mixed batch is not an
    EXTEND batch, so it stays on PAGED top-k where the fp8 prefix dequant does not
    run; HiSparse's backend allow-list is keyed on the KV dtype and is validated
    after this pass; and with a non-DSA phase (``--attention-backend triton`` plus a
    dotted ``dsa`` phase override) the other phase would run a kernel that asserts
    the fp8 DSA store is off (``MLATokenToKVPool.set_kv_buffer``), so it must fall
    back even though the sub-backend flags look exempt.
    Everything else (other phase pairs, sub-SM89 capability, the dense flashmla
    decode backend, non-DSA attention) keeps falling back to auto.

    It also pins the launch validators that keep the exempted combination honest:
    ``_validate_flashmla_sparse_fp8_kv_backend`` refuses
    ``flashmla_sparse`` + an fp8-stored pool on a NoPE model
    (``qk_rope_head_dim == 0``), whose packed fp8 layout the kernel side does not
    support yet, and the pre-existing ``flashmla_sparse_q8`` refusal is kept as a
    regression row. Finally the HiSparse dtype allow-list is pinned, because it is
    what makes an explicit HiSparse + fp8 + sparse request impossible before the
    guard ever runs.

Run exactly like the rest of this directory (the CPU suite runner invokes
``python3 <file> -f``, so the unittest entry point is required):
    python3 test/registered/unit/test_ppu_dsa_fp8_kv_guard_sparse_prefill.py -v
No model weights, server or device work is required: the guard is pure argument
post-processing, and ``is_sail``, platform capability, the backend pair and the
mixed-chunk / HiSparse flags are injected per case, so the matrix is deterministic
on any host that can import sglang. The validator rows import the DSA backend
module and skip if that module cannot be imported on the host.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups import overrides as overrides_module
from sglang.srt.arg_groups.overrides import (
    ResolvedView,
    _ppu_flashmla_fp8_kv_cache_fallback,
)

AUTO = {"kv_cache_dtype": "auto"}


class _Platform:
    def __init__(self, device_sm: int) -> None:
        self.device_sm = device_sm


def _view(**overrides) -> ResolvedView:
    """A minimal argument view: the guard only reads these fields."""
    fields = dict(
        attention_backend="dsa",
        prefill_attention_backend=None,
        decode_attention_backend=None,
        dsa_prefill_backend="flashmla_sparse",
        dsa_decode_backend="flashmla_kv",
        kv_cache_dtype="fp8_e4m3",
        enable_mixed_chunk=False,
        enable_hisparse=False,
    )
    fields.update(overrides)
    return ResolvedView(SimpleNamespace(**fields))


def _run_guard(*, sail: bool, sm: int, view: ResolvedView) -> dict:
    """Run the guard with is_sail / capability injected; resolution stays real."""
    with (
        patch("sglang.srt.utils.common.is_sail", return_value=sail),
        patch.object(overrides_module, "get_platform", return_value=_Platform(sm)),
    ):
        return dict(_ppu_flashmla_fp8_kv_cache_fallback(view))


GUARD_CASES = [
    # (name, sail, sm, view kwargs, expected)
    ("sm89_sparse_prefill_kv_decode_keeps_fp8", True, 89, {}, {}),
    (
        "sm89_both_phases_flashmla_kv_keeps_fp8",
        True,
        89,
        dict(dsa_prefill_backend="flashmla_kv", dsa_decode_backend="flashmla_kv"),
        {},
    ),
    (
        "above_sm89_both_phases_flashmla_kv_still_keeps_fp8",
        True,
        100,
        dict(dsa_prefill_backend="flashmla_kv", dsa_decode_backend="flashmla_kv"),
        {},
    ),
    (
        "sm89_both_phases_flashmla_kv_with_mixed_chunk_still_keeps_fp8",
        True,
        89,
        dict(
            dsa_prefill_backend="flashmla_kv",
            dsa_decode_backend="flashmla_kv",
            enable_mixed_chunk=True,
        ),
        {},
    ),
    ("sm90_sparse_prefill_kv_decode_falls_back", True, 90, {}, AUTO),
    ("sm100_sparse_prefill_kv_decode_falls_back", True, 100, {}, AUTO),
    (
        "sm89_sparse_prefill_kv_decode_with_mixed_chunk_falls_back",
        True,
        89,
        dict(enable_mixed_chunk=True),
        AUTO,
    ),
    (
        "sm89_sparse_prefill_kv_decode_with_hisparse_falls_back",
        True,
        89,
        dict(enable_hisparse=True),
        AUTO,
    ),
    (
        "sm89_sparse_prefill_kv_decode_missing_optional_fields_keeps_fp8",
        True,
        89,
        dict(strip_optional=True),
        {},
    ),
    (
        "triton_decode_with_dsa_prefill_phase_falls_back",
        True,
        89,
        dict(attention_backend="triton", prefill_attention_backend="dsa"),
        AUTO,
    ),
    (
        "triton_prefill_with_dsa_decode_phase_falls_back",
        True,
        89,
        dict(attention_backend="triton", decode_attention_backend="dsa"),
        AUTO,
    ),
    (
        "dsa_resolved_through_both_split_fields_keeps_fp8",
        True,
        89,
        dict(
            attention_backend="triton",
            prefill_attention_backend="dsa",
            decode_attention_backend="dsa",
        ),
        {},
    ),
    (
        "sm89_sparse_prefill_sparse_decode_still_falls_back",
        True,
        89,
        dict(dsa_decode_backend="flashmla_sparse"),
        AUTO,
    ),
    (
        "sm89_auto_prefill_still_falls_back",
        True,
        89,
        dict(dsa_prefill_backend="flashmla_auto"),
        AUTO,
    ),
    (
        "sm89_kv_prefill_sparse_decode_still_falls_back",
        True,
        89,
        dict(dsa_prefill_backend="flashmla_kv", dsa_decode_backend="flashmla_sparse"),
        AUTO,
    ),
    ("sub_sm89_sparse_prefill_falls_back", True, 80, {}, AUTO),
    (
        "dense_flashmla_decode_falls_back",
        True,
        89,
        dict(decode_attention_backend="flashmla"),
        AUTO,
    ),
    (
        "compute_dtype_kv_untouched",
        True,
        89,
        dict(kv_cache_dtype="auto"),
        {},
    ),
    ("non_sail_platform_untouched", False, 89, {}, {}),
]


class TestPpuDsaFp8KvCacheGuard(unittest.TestCase):
    def test_fp8_kv_guard_matrix(self):
        for name, sail, sm, kwargs, expected in GUARD_CASES:
            strip = bool(kwargs.pop("strip_optional", False))
            with self.subTest(case=name):
                if strip:
                    # Registered-style lightweight views do not carry the optional
                    # flags; the guard must read them defensively and still exempt.
                    view_fields = dict(kwargs)
                    fields = dict(
                        attention_backend="dsa",
                        prefill_attention_backend=None,
                        decode_attention_backend=None,
                        dsa_prefill_backend="flashmla_sparse",
                        dsa_decode_backend="flashmla_kv",
                        kv_cache_dtype="fp8_e4m3",
                    )
                    fields.update(view_fields)
                    view = ResolvedView(SimpleNamespace(**fields))
                else:
                    view = _view(**kwargs)
                self.assertEqual(_run_guard(sail=sail, sm=sm, view=view), expected, name)

    def test_sparse_fp8_validator_refuses_zero_rope(self):
        module = self._dsa_backend_module()
        with self.assertRaisesRegex(ValueError, "NoPE MLA models"):
            module._validate_flashmla_sparse_fp8_kv_backend(
                prefill_impl="flashmla_sparse",
                kv_cache_store_fp8=True,
                qk_rope_head_dim=0,
            )

    def test_sparse_fp8_validator_accepts_supported_rows(self):
        module = self._dsa_backend_module()
        for prefill_impl, store_fp8, rope_dim in (
            ("flashmla_sparse", True, 64),
            ("flashmla_sparse", False, 0),
            ("flashmla_kv", True, 0),
            ("tilelang", True, 0),
        ):
            with self.subTest(
                prefill_impl=prefill_impl, store_fp8=store_fp8, rope_dim=rope_dim
            ):
                module._validate_flashmla_sparse_fp8_kv_backend(
                    prefill_impl=prefill_impl,
                    kv_cache_store_fp8=store_fp8,
                    qk_rope_head_dim=rope_dim,
                )

    def test_sparse_q8_zero_rope_refusal_is_unchanged(self):
        """Regression row for the pre-existing q8 refusal next to the new one."""
        import torch

        module = self._dsa_backend_module()
        with self.assertRaisesRegex(ValueError, "NoPE MLA models"):
            module._validate_flashmla_sparse_q8_backend(
                prefill_impl="flashmla_sparse_q8",
                decode_impl="flashmla_kv",
                kv_cache_dtype=torch.float8_e4m3fn,
                kv_cache_store_fp8=True,
                qk_rope_head_dim=0,
                device_sm_major=8,
            )

    def test_hisparse_cannot_reach_the_sparse_exemption(self):
        """HiSparse never resolves an fp8 pool to a flashmla_sparse prefill.

        The dtype-aware allow-list rejects an explicit sparse request and the
        HiSparse default for fp8 is flashmla_kv (the pre-existing both-phases
        exemption), which is why the guard also refuses to exempt while HiSparse
        is enabled: the dtype it would protect is not the dtype HiSparse validates.
        """
        from sglang.srt.arg_groups import hisparse_hook

        self.assertNotIn(
            "flashmla_sparse", hisparse_hook._hisparse_allowed_backends("fp8_e4m3")
        )
        self.assertIn(
            "flashmla_kv", hisparse_hook._hisparse_allowed_backends("fp8_e4m3")
        )
        self.assertEqual(
            hisparse_hook._hisparse_default_backend("fp8_e4m3"), "flashmla_kv"
        )

    def _dsa_backend_module(self):
        try:
            import sglang.srt.layers.attention.dsa_backend as module
        except Exception as exc:  # pragma: no cover - host without DSA deps
            raise unittest.SkipTest(f"DSA backend module not importable: {exc}")
        return module


if __name__ == "__main__":
    unittest.main()
