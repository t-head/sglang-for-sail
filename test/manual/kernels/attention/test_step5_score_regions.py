# SPDX-License-Identifier: Apache-2.0
"""本地 PPU scoring UT：物理映射、逐 head ReLU、ragged 和 graph 重放。"""

import unittest

import torch

from sglang.test.kernels.step5_scoring import (
    VARIANTS,
    compare_scores,
    make_case,
    make_runner,
    reference_scores,
)
from sglang.test.test_utils import CustomTestCase


class TestStep5ScoreRegions(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("需要可用的 PPU/CUDA")
        torch.cuda.set_device(0)

    def test_physical_mapping_ragged_causality(self):
        """随机 physical page 不能改变 logical causality，尾 span 不能越界。"""
        for heads in (2, 4):
            case = make_case(
                lengths=(7, 13, 9), prefixes=(0, 4097, 8193), heads=heads,
                width=1031, easy=True, padding=5, seed=17,
            )
            expected = reference_scores(case).cuda()
            for fp8 in (True, False):
                for variant in VARIANTS:
                    with self.subTest(heads=heads, fp8=fp8, variant=variant):
                        actual = make_runner(case, variant, fp8=fp8)["run"]()
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        self.assertTrue(torch.equal(actual.view(torch.int32), expected.view(torch.int32)))

    def test_random_scores_top512(self):
        """重排加载或 tile 不应改变原 MMA 的分数及 top-512 tie-break。"""
        case = make_case(lengths=(33, 17), prefixes=(8191, 4101), width=1037, seed=23)
        for fp8 in (True, False):
            baseline = make_runner(case, "baseline", fp8=fp8)["run"]().clone()
            for variant in VARIANTS[1:]:
                with self.subTest(fp8=fp8, variant=variant):
                    actual = make_runner(case, variant, fp8=fp8)["run"]()
                    comparison = compare_scores(actual, baseline, sample_rows=50)
                    self.assertTrue(comparison["bitwise_equal"], comparison)
                    self.assertTrue(comparison["top512_equal"], comparison)

    def test_wider_query_groups(self):
        """16/32 行复用仍必须逐 query 保留 causal mask 和 head reduction。"""
        case = make_case(lengths=(65, 3), prefixes=(8179, 4101), width=1033, easy=True, padding=7, seed=31)
        expected = reference_scores(case).cuda()
        for fp8 in (True, False):
            for group in (16, 32):
                for variant in ("tile", "compact_tile"):
                    with self.subTest(fp8=fp8, group=group, variant=variant):
                        actual = make_runner(case, variant, fp8=fp8, group_rows=group)["run"]()
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_relu_order_and_positive_zero(self):
        """ReLU 必须在 head sum 前；负权重产生的零必须归一到 +0。"""
        case = make_case(lengths=(9,), prefixes=(4104,), width=517, heads=2, easy=True)
        case["keys"].fill_(0x38)
        case["physical_keys"].fill_(1.0)
        q = torch.zeros(case["q"].shape)
        q[:, 0] = 1.0
        q[:, 1] = -1.0
        case["q"].copy_(q.to(torch.float8_e4m3fn).view(torch.uint8).cuda())
        for fp8 in (True, False):
            for variant in VARIANTS:
                runner = make_runner(case, variant, fp8=fp8)
                actual = runner["run"]()
                self.assertEqual(float(actual[0, 0]), 256.0)
                case["q"].zero_()
                case["weights"].fill_(-1.0)
                actual = runner["run"]()
                valid = torch.isfinite(actual)
                self.assertTrue(bool((actual.view(torch.int32)[valid] == 0).all()))
                case["q"].copy_(q.to(torch.float8_e4m3fn).view(torch.uint8).cuda())
                case["weights"].fill_(1.0)

    def test_graph_replay_refreshes_live_keys(self):
        """同一 graph 的 history 大、小、大变化不能消费 stale K 或 scores。"""
        case = make_case(lengths=(37,), prefixes=(8192,), width=1037, easy=True, padding=3, seed=41)
        for fp8 in (True, False):
            for variant in VARIANTS:
                with self.subTest(fp8=fp8, variant=variant):
                    runner = make_runner(case, variant, fp8=fp8)
                    for _ in range(3):
                        runner["run"]()
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        runner["run"]()
                    for prefix in (8192, 0, 8192):
                        case["positions"][:37].copy_(torch.arange(prefix, prefix + 37, dtype=torch.int32, device="cuda"))
                        case["max_positions"][1] = prefix + 36
                        graph.replay()
                        expected = reference_scores(case).cuda()
                        torch.testing.assert_close(runner["out"], expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
