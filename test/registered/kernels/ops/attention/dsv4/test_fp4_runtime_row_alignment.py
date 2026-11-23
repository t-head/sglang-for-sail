"""Runtime row alignment must not recompile FP4 Gather or RoPE kernels."""

import unittest
from unittest.mock import patch

import torch
import triton

from sglang.kernels.ops.attention.dsv4.fp4_rope_fake_quant import (
    _rope_tail_fake_quant_fp4_kernel,
)
from sglang.kernels.ops.kvcache.dsv4_fp4_index_k_cache import (
    _gather_fp4_index_k_kernel,
)
from sglang.kernels.ops.quantization.fp4_fake_quant import (
    FP4_AMAX_FLOOR,
    fake_quant_compressed_kv,
    fake_quant_fp4,
)
from sglang.srt.utils import get_device_sm, is_sail
from sglang.test.ci.ci_register import register_cuda_ci, register_ppu_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_ppu_ci(est_time=20, suite="stage-a-test-1-gpu-ppu")

ROWS = (128, 129, 144)
CAPACITY = max(ROWS) + 64


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA-compatible device")
class TestFp4RuntimeRowAlignment(CustomTestCase):
    def assert_one_kernel(self, launched):
        self.assertEqual(len(launched), len(ROWS))
        self.assertIsNotNone(launched[0])
        for kernel in launched[1:]:
            self.assertIs(kernel, launched[0], "runtime row alignment recompiled")

    @torch.inference_mode()
    def test_gather_alignment_and_masked_tail(self):
        page_size, pages = 64, 4
        generator = torch.Generator().manual_seed(54120)
        cache_cpu = torch.randint(
            0, 256, (pages, page_size * 68), generator=generator, dtype=torch.uint8
        )
        slots_cpu = torch.arange(CAPACITY, dtype=torch.int64) * 37 % (pages * page_size)
        slots_cpu[:8] = torch.tensor([0, 63, 64, 127, 128, 255, 64, 64])
        words_cpu = cache_cpu.view(torch.int32)
        cache = cache_cpu.cuda()
        words, slots = cache.view(torch.int32), slots_cpu.cuda()
        sentinel = 0x5A5A5A5A
        payload = torch.empty((CAPACITY, 16), device="cuda", dtype=torch.int32)
        scales = torch.empty(CAPACITY, device="cuda", dtype=torch.int32)
        launched = []
        run = _gather_fp4_index_k_kernel.run

        def record_kernel(*args, **kwargs):
            compiled = run(*args, **kwargs)
            launched.append(compiled)
            return compiled

        with patch.object(_gather_fp4_index_k_kernel, "run", side_effect=record_kernel):
            for n in ROWS:
                with self.subTest(rows=n):
                    payload.fill_(sentinel)
                    scales.fill_(sentinel)
                    _gather_fp4_index_k_kernel[(triton.cdiv(n, 64),)](
                        words,
                        words.stride(0),
                        slots,
                        payload,
                        scales,
                        n,
                        page_size,
                        BLOCK_ROWS=64,
                        num_warps=4,
                    )
                    torch.cuda.synchronize()
                    chosen = slots_cpu[:n]
                    page, offset = chosen // page_size, chosen % page_size
                    expected = words_cpu[
                        page[:, None], offset[:, None] * 16 + torch.arange(16)[None, :]
                    ]
                    expected_sf = words_cpu[page, page_size * 16 + offset]
                    torch.testing.assert_close(
                        payload[:n].cpu(), expected, rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        scales[:n].cpu(), expected_sf, rtol=0, atol=0
                    )
                    self.assertTrue(torch.all(payload[n:] == sentinel).item())
                    self.assertTrue(torch.all(scales[n:] == sentinel).item())
        self.assert_one_kernel(launched)
        torch.testing.assert_close(cache.cpu(), cache_cpu, rtol=0, atol=0)

    @torch.inference_mode()
    def test_rope_alignment_and_position_bounds(self):
        # Exact E2M1 values, with amax=6 in every block, give scale=1 in
        # both existing Torch fake-quant references. Quarter turns are exact
        # in BF16, so no numerical tolerance or near-rounding-tie oracle is needed.
        pattern = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
            dtype=torch.bfloat16,
        )
        row_ids = torch.arange(CAPACITY)
        real = (row_ids % 2 == 0).float().unsqueeze(1).expand(-1, 32).contiguous()
        imag = 1.0 - real
        table_cpu = torch.view_as_real(torch.complex(real, imag)).contiguous()
        table = table_cpu.cuda()
        positions = torch.empty(CAPACITY, device="cuda", dtype=torch.int64)
        for dim, compressed in ((128, False), (512, True)):
            x_cpu = pattern.repeat(CAPACITY, dim // 16)
            x_cpu[1::2].neg_()
            x = x_cpu.cuda()
            out = torch.empty_like(x)
            for has_positions in (False, True):
                with self.subTest(dim=dim, positions=has_positions):
                    launched = []
                    run = _rope_tail_fake_quant_fp4_kernel.run

                    def record_kernel(*args, **kwargs):
                        compiled = run(*args, **kwargs)
                        launched.append(compiled)
                        return compiled

                    with patch.object(
                        _rope_tail_fake_quant_fp4_kernel,
                        "run",
                        side_effect=record_kernel,
                    ):
                        for n in ROWS:
                            out.fill_(17)
                            pos_cpu = torch.arange(CAPACITY, dtype=torch.int64) % n
                            if has_positions:
                                pos_cpu[0] = n - 1
                                pos_cpu[n - 2] = -1
                                pos_cpu[n - 1] = n
                            positions.copy_(pos_cpu)
                            _rope_tail_fake_quant_fp4_kernel[(n,)](
                                x,
                                table,
                                positions if has_positions else table,
                                out,
                                dim,
                                dim,
                                table.stride(0),
                                1,
                                n,
                                D=dim,
                                RD=64,
                                BLK=16 if compressed else 32,
                                AMAX_FLOOR=FP4_AMAX_FLOOR,
                                INVERSE=False,
                                COMPRESSED_KV=compressed,
                                HAS_POS=has_positions,
                                EMULATE_E4M3=is_sail() and get_device_sm() < 89,
                                num_warps=4,
                            )
                            torch.cuda.synchronize()
                            selected = pos_cpu[:n] if has_positions else torch.arange(n)
                            selected = torch.where(
                                (selected >= 0) & (selected < n), selected, 0
                            )
                            expected = x_cpu[:n].clone()
                            tail = x_cpu[:n, -64:].reshape(n, 32, 2)
                            quarter_turn = (selected % 2 == 1).unsqueeze(1)
                            rotated = expected[:, -64:].reshape(n, 32, 2)
                            rotated[..., 0] = torch.where(
                                quarter_turn, -tail[..., 1], tail[..., 0]
                            )
                            rotated[..., 1] = torch.where(
                                quarter_turn, tail[..., 0], tail[..., 1]
                            )
                            expected = (
                                fake_quant_compressed_kv(expected)
                                if compressed
                                else fake_quant_fp4(expected)
                            )
                            torch.testing.assert_close(
                                out[:n].cpu(), expected, rtol=0, atol=0
                            )
                            self.assertTrue(torch.all(out[n:] == 17).item())
                    self.assert_one_kernel(launched)


if __name__ == "__main__":
    unittest.main()
