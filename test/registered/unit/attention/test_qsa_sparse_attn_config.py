import unittest
from unittest.mock import patch

from sglang.srt.layers.attention.qsa import sparse_attn
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestQSASparseAttentionConfig(unittest.TestCase):
    def test_tile_boundaries_by_platform(self):
        for sail, sm, name, cases in (
            (True, 89, "SAIL", {512: (32, 4, 2), 513: (32, 2, 2)}),
            (True, 80, "SAIL", {513: (16, 1, 2)}),
            (False, 89, "NVIDIA L20", {513: (16, 1, 2)}),
            (False, 90, "NVIDIA H20", {1024: (32, 4, 2), 1025: (16, 1, 2)}),
        ):
            with (
                self.subTest(sail=sail, sm=sm, name=name),
                patch.object(
                    sparse_attn, "is_batch_invariant_mode_enabled", return_value=False
                ),
                patch.object(sparse_attn, "is_sail", return_value=sail),
                patch.object(sparse_attn, "get_device_sm", return_value=sm),
                patch.object(
                    sparse_attn.torch.cuda, "get_device_name", return_value=name
                ),
            ):
                for total_q, expected in cases.items():
                    self.assertEqual(sparse_attn._get_best_config(total_q), expected)

    def test_batch_invariant_config_takes_precedence(self):
        with (
            patch.object(
                sparse_attn, "is_batch_invariant_mode_enabled", return_value=True
            ),
            patch.object(sparse_attn, "is_sail", return_value=True),
            patch.object(sparse_attn, "get_device_sm", return_value=89),
        ):
            self.assertEqual(sparse_attn._get_best_config(513), (32, 4, 2))


if __name__ == "__main__":
    unittest.main()
