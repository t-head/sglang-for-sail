"""ReplaySSM ring per-slot byte accounting and PPU native eligibility.

The memory solver charges this on top of mamba_cache_per_req so num_slots is not
over-provisioned (the ring is allocated per slot but is NOT part of the state
cache cost). Pins the arithmetic against hand-computed byte counts for the
fold window (raw v / pre-norm k / g / beta) and the external FlashKDA
correction-record layout. GDN uses per-head scalar g while KDA uses per-K vector
g (and keeps the chunked d/k rings under spec, see MambaPool). The FlashKDA
layout contract is shared directly by sizing and allocation.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.configs.mamba_utils import (
    FlashKDAReplaySSMLayout,
    KimiLinearCacheParams,
    KimiLinearStateShape,
    Mamba2CacheParams,
    Mamba2StateDType,
    Mamba2StateShape,
    ppu_kda_replayssm_native_enabled,
)
from sglang.srt.environ import envs
from sglang.srt.mem_cache.kv_cache_configurator import (  # noqa: F401
    _pp_local_per_request_bytes,
)
from sglang.srt.speculative.ragged_verify import RaggedVerifyMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

# temporal = (hv=4, v_dim=8, k_dim=8), num_k_heads_per_tp = 4, record_len = 8,
# 2 layers. conv bf16 (2B), fp32 gate/beta (4B). Ring tensors (per slot, per
# layer):
# GDN stores compact d/k plus scalar g. KDA stores raw v/k, vector g, beta,
# and its existing d/k rings.
DTYPE = Mamba2StateDType(conv=torch.bfloat16, temporal=torch.float32)
RL = 8
LAYERS = [0, 1]


def _kda_params():
    shape = KimiLinearStateShape.create(
        tp_world_size=1, num_heads=4, head_dim=8, num_k_heads=4, head_k_dim=8
    )
    return KimiLinearCacheParams(shape=shape, dtype=DTYPE, layers=LAYERS)


def _flashkda_params(temporal_dtype=torch.float32):
    shape = KimiLinearStateShape.create(
        tp_world_size=1, num_heads=2, head_dim=128, num_k_heads=2, head_k_dim=128
    )
    dtype = Mamba2StateDType(conv=torch.bfloat16, temporal=temporal_dtype)
    return KimiLinearCacheParams(shape=shape, dtype=dtype, layers=LAYERS)


def _gdn_params(temporal_dtype=torch.float32):
    # Only shape.temporal and shape.num_k_heads_per_tp are read here; the rest
    # are dummy (the accounting does not depend on them).
    shape = Mamba2StateShape(
        conv=[(4, 3)],
        temporal=(4, 8, 8),
        intermediate_size=0,
        conv_dim=0,
        ssm_state_size=0,
        num_heads=0,
        head_dim=0,
        state_size=0,
        conv_kernel=0,
        num_k_heads_per_tp=4,
    )
    dtype = Mamba2StateDType(conv=torch.bfloat16, temporal=temporal_dtype)
    return Mamba2CacheParams(shape=shape, dtype=dtype, layers=LAYERS)


class TestReplaySSMRingAccounting(CustomTestCase):
    def test_gdn_fold(self):
        # d 512 + normalized k 512 + scalar g 128 + d/k low parts 1024 = 2176
        self.assertEqual(
            _gdn_params().replayssm_ring_bytes_per_req(record_len=RL),
            2176 * len(LAYERS),
        )
        self.assertEqual(
            _gdn_params(torch.bfloat16).replayssm_ring_bytes_per_req(record_len=RL),
            2176 * len(LAYERS),
        )

    def test_kda_fold(self):
        # rawv 512 + rawk 512 + g(per-K, 4*8*8*4) 1024 + beta 128
        # + d 512 + k 512 (KDA keeps the chunked rings under spec) = 3200
        self.assertEqual(
            _kda_params().replayssm_ring_bytes_per_req(record_len=RL),
            3200 * len(LAYERS),
        )

    def test_flashkda_layout_drives_shape_and_sizing(self):
        params = _flashkda_params()
        layout = FlashKDAReplaySSMLayout.from_params(params, RL)
        self.assertEqual(layout.tensor_shape(len(LAYERS), 7), (2, 7, 2, 8, 128))
        self.assertEqual(
            params.replayssm_ring_bytes_per_req(record_len=RL, flashkda_layout=layout),
            3 * 2 * RL * 128 * 4 * len(LAYERS),
        )

    def test_flashkda_layout_rejects_mismatched_record_length(self):
        params = _flashkda_params()
        layout = FlashKDAReplaySSMLayout.from_params(params, RL)
        with self.assertRaisesRegex(ValueError, "record lengths must match"):
            params.replayssm_ring_bytes_per_req(
                record_len=RL + 1, flashkda_layout=layout
            )

    def test_zero_len_ring(self):
        self.assertEqual(_gdn_params().replayssm_ring_bytes_per_req(record_len=0), 0)
        self.assertEqual(_kda_params().replayssm_ring_bytes_per_req(record_len=0), 0)

    def test_ppu_native_selector_preserves_raw_ring_fallback(self):
        cases = (
            (torch.float32, False, False, RaggedVerifyMode.STATIC, True),
            (torch.bfloat16, False, False, RaggedVerifyMode.STATIC, True),
            (torch.float16, False, False, RaggedVerifyMode.STATIC, False),
            (torch.float32, True, False, RaggedVerifyMode.STATIC, False),
            (torch.float32, False, True, RaggedVerifyMode.STATIC, False),
            (torch.float32, False, False, RaggedVerifyMode.CAP_ACCEPT, False),
            (torch.float32, False, False, RaggedVerifyMode.COMPACT, False),
        )
        for dtype, page_major, unified, ragged_mode, expected in cases:
            with self.subTest(
                dtype=dtype,
                page_major=page_major,
                unified=unified,
                ragged_mode=ragged_mode,
            ):
                with (
                    envs.SGLANG_USE_PPU_KDA_REPLAYSSM.override("native"),
                    patch(
                        "sglang.srt.runtime_context.get_platform",
                        return_value=SimpleNamespace(is_ppu=True, device_sm=89),
                    ),
                    patch(
                        "sglang.srt.runtime_context.get_exec",
                        return_value=SimpleNamespace(
                            mamba=SimpleNamespace(
                                enable_linear_replayssm_spec=True,
                                linear_replayssm_cache_len=RL,
                            )
                        ),
                    ),
                    patch(
                        "sglang.srt.runtime_context.get_memory",
                        return_value=SimpleNamespace(
                            enable_page_major_kv_layout=page_major,
                            enable_unified_memory=unified,
                        ),
                    ),
                    patch(
                        "sglang.srt.speculative.ragged_verify.read_ragged_verify_mode",
                        return_value=ragged_mode,
                    ),
                ):
                    self.assertIs(
                        ppu_kda_replayssm_native_enabled(_flashkda_params(dtype)),
                        expected,
                    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
