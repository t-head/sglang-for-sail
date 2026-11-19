"""Protect PLA's indexed V,K layout and final-state versus checkpoint contract."""

import sys
import unittest
from types import ModuleType
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.linear.kernels import gdn_pla
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestPLAPrefillContract(CustomTestCase):
    def test_indexed_state_layout_precision_and_untouched_slots(self):
        # Rectangular states expose accidental use of PLA's legacy K,V layout.
        for dtype in (torch.float32, torch.bfloat16):
            for strided in (False, True):
                with self.subTest(dtype=dtype, strided=strided):
                    state = torch.arange(5 * 2 * 4 * 3).reshape(5, 2, 4, 3).to(dtype)
                    if strided:
                        storage = torch.empty(5, 2, 4, 3, 2, dtype=dtype)
                        storage[..., 0] = state
                        state = storage[..., 0]
                    before = state.clone()
                    slots = torch.tensor([3, 1], dtype=torch.int64)
                    cu = torch.tensor([0, 3, 5], dtype=torch.int64)
                    q = torch.randn(1, 5, 2, 6)[..., ::2]
                    g = torch.randn(1, 5, 2)
                    expected_output = torch.randn(1, 5, 2, 4)

                    def forward(**kw):
                        self.assertEqual(kw["raw_cu_seqlens"], (0, 3, 5))
                        self.assertTrue(kw["normalize_qk"])
                        self.assertFalse(kw["output_final_state"])
                        self.assertEqual(kw["final_state_dtype"], dtype)
                        self.assertEqual(kw["initial_state"].shape[-2:], (4, 3))
                        for name in ("q", "k", "v", "g", "beta", "initial_state"):
                            self.assertTrue(kw[name].is_contiguous(), name)
                        for name in ("cu_seqlens", "initial_state_indices"):
                            self.assertEqual(kw[name].dtype, torch.int32)
                        torch.testing.assert_close(kw["q"], q)
                        indices = kw["initial_state_indices"]
                        torch.testing.assert_close(
                            kw["initial_state"][indices], before[slots]
                        )
                        kw["initial_state"][indices] += 2
                        return None, None, expected_output, None, object()

                    module = ModuleType("pla.prefill.flashqla")
                    module.chunk_gated_delta_rule_fwd = forward
                    with patch.dict(
                        sys.modules,
                        {
                            "pla": ModuleType("pla"),
                            "pla.prefill": ModuleType("pla.prefill"),
                            "pla.prefill.flashqla": module,
                        },
                    ):
                        out, final, checkpoints = gdn_pla.pla_gdn_prefill(
                            q,
                            q,
                            q,
                            g,
                            g,
                            ssm_states=state,
                            cache_indices=slots,
                            query_start_loc=cu,
                            raw_cu_seqlens=(0, 3, 5),
                        )
                    self.assertIs(out, expected_output)
                    self.assertIsNone(final)
                    self.assertIsNone(checkpoints)
                    expected = before.clone()
                    expected[slots] += 2
                    torch.testing.assert_close(state, expected, atol=0, rtol=0)

    def test_host_offsets_reject_stale_or_padded_batch(self):
        cu = torch.empty(3, dtype=torch.int32)
        self.assertEqual(gdn_pla.pla_host_offsets([3, 2], cu, 5), (0, 3, 5))
        self.assertEqual(
            gdn_pla.pla_host_offsets(torch.tensor([3, 2]), cu, 5), (0, 3, 5)
        )
        for lengths, tokens in (
            ([3, 2], 8),
            ([5], 5),
            ([3, -1], 2),
            ([3, 2.0], 5),
            ([True, 4], 5),
            (None, 5),
        ):
            with self.subTest(lengths=lengths, tokens=tokens):
                self.assertIsNone(gdn_pla.pla_host_offsets(lengths, cu, tokens))


if __name__ == "__main__":
    unittest.main()
