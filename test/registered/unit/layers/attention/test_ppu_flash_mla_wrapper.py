"""The PPU flash_mla wrapper must pass the V4.1 interleaved layouts through.

The PPU flash_mla wheel (2.1.0) follows the upstream API: it has no ``use_legacy``
switch and decodes the V4.1 rows (512 B E4M3 + 16 B ue8m0 per token, i.e. the
interleaved ``V41_NEW`` arrangement) natively. ``use_legacy=False`` therefore has
to reach the wheel instead of being rejected by the wrapper.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.hardware_backend.ppu.attention import flash_mla as wrapper
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _fake_flash_mla(calls):
    def with_kvcache(*args, **kwargs):
        calls["kvcache"] = (args, kwargs)
        return torch.zeros(1), torch.zeros(1)

    def sparse_fwd(*args, **kwargs):
        calls["sparse"] = (args, kwargs)
        return torch.zeros(1), torch.zeros(1), torch.zeros(1)

    return SimpleNamespace(
        flash_mla_with_kvcache=with_kvcache, flash_mla_sparse_fwd=sparse_fwd
    )


class TestPpuFlashMlaWrapper(unittest.TestCase):
    def test_has_no_legacy_only_guard(self):
        self.assertFalse(hasattr(wrapper, "_require_legacy_kernels"))

    def test_kvcache_accepts_interleaved_layouts(self):
        calls = {}
        stub = SimpleNamespace(
            flash_mla_with_kvcache=_fake_flash_mla(calls).flash_mla_with_kvcache
        )
        with (
            mock.patch.object(wrapper, "_flashmla", stub),
            mock.patch.object(wrapper, "_import_error", None),
        ):
            out, lse = wrapper.flash_mla_with_kvcache(
                q=torch.zeros((1, 1, 2, 64), dtype=torch.float32),
                k_cache=torch.zeros((1, 1, 1, 64), dtype=torch.bfloat16),
                block_table=None,
                cache_seqlens=None,
                head_dim_v=512,
                tile_scheduler_metadata=object(),
                use_legacy=False,
            )
        self.assertIsNotNone(out)
        # The switch is wrapper-only: the wheel never sees it.
        self.assertNotIn("use_legacy", calls["kvcache"][1])

    def test_sparse_fwd_accepts_interleaved_layouts(self):
        calls = {}
        stub = SimpleNamespace(
            flash_mla_sparse_fwd=_fake_flash_mla(calls).flash_mla_sparse_fwd
        )
        with (
            mock.patch.object(wrapper, "_flashmla", stub),
            mock.patch.object(wrapper, "_import_error", None),
        ):
            wrapper.flash_mla_sparse_fwd(
                q=torch.zeros((1, 2, 64), dtype=torch.float32),
                kv=torch.zeros((1, 1, 64), dtype=torch.bfloat16),
                indices=torch.zeros((1, 1, 1), dtype=torch.int32),
                sm_scale=0.1,
                use_legacy=False,
            )
        self.assertNotIn("use_legacy", calls["sparse"][1])


class TestPpuFlashMlaKvFormatContract(unittest.TestCase):
    """``kv_format`` is a framework-side hint the PPU wheel never sees.

    The realigned indexer/FlashMLA contracts made ``dsa_backend`` pass
    ``kv_format="V32_NO_ROPE"`` for every zero-rope model, and the wrapper's
    allow-list rejected it before any delegation happened - that rejection is
    what aborted serve startup.  These checks pin all three sides of the fixed
    contract without needing PPU hardware: the hint is accepted, it is never
    forwarded to the wheel, and an unknown layout still raises before
    delegation.  Narrowing the allow-list back makes
    ``test_v32_no_rope_is_accepted`` fail, so the suite can no longer pass with
    the guard that caused the failure.
    """

    _OMIT = object()  # sentinel: call without the kv_format keyword at all

    @staticmethod
    def _call(calls, kv_format=_OMIT):
        stub = SimpleNamespace(
            flash_mla_with_kvcache=_fake_flash_mla(calls).flash_mla_with_kvcache
        )
        kwargs = dict(
            q=torch.zeros((1, 1, 2, 64), dtype=torch.float32),
            k_cache=torch.zeros((1, 1, 1, 64), dtype=torch.bfloat16),
            block_table=None,
            cache_seqlens=None,
            head_dim_v=512,
            tile_scheduler_metadata=object(),
        )
        if kv_format is not TestPpuFlashMlaKvFormatContract._OMIT:
            kwargs["kv_format"] = kv_format
        with (
            mock.patch.object(wrapper, "_flashmla", stub),
            mock.patch.object(wrapper, "_import_error", None),
        ):
            return wrapper.flash_mla_with_kvcache(**kwargs)

    def test_v32_no_rope_is_accepted(self):
        calls = {}
        out, lse = self._call(calls, kv_format="V32_NO_ROPE")
        self.assertIsNotNone(out)
        self.assertIsNotNone(lse)
        self.assertIn("kvcache", calls)

    def test_v32_and_default_stay_accepted(self):
        # The two entries the wrapper already allowed must not regress while the
        # zero-rope hint was added next to them.
        for kv_format in (None, "V32"):
            with self.subTest(kv_format=kv_format):
                calls = {}
                self._call(calls, kv_format=kv_format)
                self.assertIn("kvcache", calls)

    def test_kv_format_is_never_forwarded_to_the_wheel(self):
        # The hint is dropped before delegation: the PPU wheel's interface has
        # no kv_format parameter at all, so forwarding it would raise TypeError
        # on real hardware.
        for kv_format in (None, "V32", "V32_NO_ROPE"):
            with self.subTest(kv_format=kv_format):
                calls = {}
                self._call(calls, kv_format=kv_format)
                _args, kwargs = calls["kvcache"]
                self.assertNotIn("kv_format", kwargs)

    def test_accepted_formats_delegate_identically(self):
        # Accepting the hint must not change the call the wheel receives.
        baseline = {}
        self._call(baseline)
        base_args, base_kwargs = baseline["kvcache"]
        for kv_format in ("V32", "V32_NO_ROPE"):
            with self.subTest(kv_format=kv_format):
                probe = {}
                self._call(probe, kv_format=kv_format)
                got_args, got_kwargs = probe["kvcache"]
                self.assertEqual(len(got_args), len(base_args))
                self.assertEqual(set(got_kwargs), set(base_kwargs))
                self.assertEqual(sorted(got_kwargs), sorted(base_kwargs))

    def test_unknown_format_raises_before_delegation(self):
        for kv_format in ("V4", "V41", "v32", "", "V32_NO_ROPE_V2"):
            with self.subTest(kv_format=kv_format):
                calls = {}
                with self.assertRaises(NotImplementedError):
                    self._call(calls, kv_format=kv_format)
                # Nothing may reach the wheel once the guard fires.
                self.assertNotIn("kvcache", calls)


if __name__ == "__main__":
    unittest.main()
