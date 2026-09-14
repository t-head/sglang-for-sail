"""Byte-exact V4 gather, including int32 locations beyond 2 GiB offsets."""

import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.kvcache.dsv4_main_kv_store import GetKAndS
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestDSV4MainKVGather(CustomTestCase):
    def _check(self, page, dtype):
        size, stride = 64, 37440
        pool = SimpleNamespace(
            kv_layout=SimpleNamespace(value="v4"),
            page_size=size,
            qk_nope_head_dim=448,
            qk_rope_head_dim=64,
            quantize_block_size=64,
            scale_pad=1,
            rope_storage_dtype=torch.bfloat16,
        )
        buf = torch.zeros((page + 1, stride), dtype=torch.uint8, device="cuda")
        loc = torch.tensor(
            [page * size + 3, 0, page * size + 3], dtype=dtype, device="cuda"
        )
        rows = loc.long()
        base = rows // size * stride
        slot = rows % size
        values = (torch.arange(448, device="cuda") % 126).byte().expand(3, -1)
        rope = torch.arange(64, device="cuda").bfloat16().expand(3, -1).contiguous()
        scales = torch.full((3, 7), 127, dtype=torch.uint8, device="cuda")
        buf.flatten()[
            base[:, None] + slot[:, None] * 576 + torch.arange(448, device="cuda")
        ] = values
        buf.flatten()[
            base[:, None] + slot[:, None] * 576 + 448 + torch.arange(128, device="cuda")
        ] = rope.view(torch.uint8)
        buf.flatten()[
            base[:, None]
            + size * 576
            + slot[:, None] * 8
            + torch.arange(7, device="cuda")
        ] = scales
        got = GetKAndS.triton(pool, buf, loc)
        ref = GetKAndS.torch(pool, buf, loc)
        for attr in ("k_nope_fp8", "k_rope_bf16", "scale_k_nope_ue8m0"):
            torch.testing.assert_close(
                getattr(got, attr).view(torch.uint8),
                getattr(ref, attr).view(torch.uint8),
                rtol=0,
                atol=0,
            )
        torch.testing.assert_close(
            got.k_nope_fp8.view(torch.uint8), values, rtol=0, atol=0
        )
        torch.testing.assert_close(got.k_rope_bf16, rope, rtol=0, atol=0)
        torch.testing.assert_close(got.scale_k_nope_ue8m0, scales, rtol=0, atol=0)

    def test_page_mapping_and_duplicates(self):
        for dtype in (torch.int32, torch.int64):
            with self.subTest(dtype=dtype):
                self._check(2, dtype)

    def test_int32_location_above_two_gib(self):
        if torch.cuda.mem_get_info()[0] < 3 * 1024**3:
            self.skipTest("requires 3 GiB free for a physical offset regression")
        self._check(60000, torch.int32)


if __name__ == "__main__":
    unittest.main()
