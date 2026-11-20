"""Native operator numerics for runtime-only DSPARK scalar parameters."""

import unittest

import torch
import triton

from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.test_utils import CustomTestCase

register_ppu_ci(est_time=40, suite="stage-a-test-1-gpu-ppu")


class TestDsparkScalarNumericsPPU(CustomTestCase):
    def test_expand_causal_metadata_and_padding(self):
        from sglang.kernels.ops.attention.dsv4.attn_metadata_kernels import (
            expand_prefill_causally_triton,
        )

        for bs in (1, 3, 4, 16, 17):
            with self.subTest(bs=bs):
                ext = torch.arange(bs, dtype=torch.int32) % 3 + 1
                prefix = torch.arange(bs, dtype=torch.int32) * 7
                req = torch.arange(bs, dtype=torch.int64) * 3 + 2
                causal = torch.cat(
                    [
                        torch.arange(int(p) + 1, int(p + e) + 1, dtype=torch.int32)
                        for p, e in zip(prefix, ext)
                    ]
                )
                repeated = torch.repeat_interleave(req, ext.long())
                for pad in (0, 3):
                    expected_causal = torch.cat(
                        [causal, torch.ones(pad, dtype=torch.int32)]
                    )
                    expected_req = torch.cat([repeated, req[-1:].expand(pad)])
                    output = expand_prefill_causally_triton(
                        req_pool_indices=req.cuda(),
                        seq_lens=(prefix + ext).cuda(),
                        extend_seq_lens=ext.cuda(),
                        num_tokens=causal.numel(),
                        padded_num_tokens=causal.numel() + pad,
                    )
                    self.assertTrue(
                        torch.equal(output.seq_lens_casual.cpu(), expected_causal)
                    )
                    self.assertTrue(
                        torch.equal(
                            output.req_pool_indices_repeated.cpu(), expected_req
                        )
                    )

    def test_get_last_locations(self):
        from sglang.kernels.ops.memory.common import get_last_loc_triton

        for dtype in (torch.int32, torch.int64):
            for bs in (1, 3, 16, 17, 257):
                with self.subTest(dtype=dtype, bs=bs):
                    table = torch.arange(bs * 9, dtype=torch.int64).view(bs, 9)
                    indices = torch.arange(bs - 1, -1, -1, dtype=torch.int64)
                    prefix = (torch.arange(bs) % 10).to(dtype)
                    expected = torch.where(
                        prefix > 0, table[indices, (prefix.long() - 1).clamp_min(0)], -1
                    ).to(dtype)
                    output = get_last_loc_triton(
                        table.cuda(), indices.cuda(), prefix.cuda()
                    )
                    self.assertTrue(torch.equal(output.cpu(), expected))

    def test_markov_finish_reduction(self):
        from sglang.kernels.ops.speculative.dspark.markov_sample import (
            _markov_sample_finish_kernel,
        )

        world, parts = 2, 4
        for bs in (1, 3, 16, 17):
            with self.subTest(bs=bs):
                values = (
                    torch.arange(world * bs * parts).view(world, bs, parts) % 5
                ).float()
                ids = torch.arange(world * bs * parts, dtype=torch.int32).view(
                    world, bs, parts
                )
                values[0, 0, 1] = values[1, 0, 2] = float("nan")
                packed = torch.empty((world, bs, parts, 2), dtype=torch.float32)
                packed[..., 0] = values
                packed.view(torch.int32)[..., 1] = ids
                expected = []
                for row in range(bs):
                    row_values, row_ids = (
                        values[:, row].flatten(),
                        ids[:, row].flatten(),
                    )
                    wins = (
                        torch.isnan(row_values)
                        if torch.isnan(row_values).any()
                        else row_values == row_values.max()
                    )
                    expected.append(int(row_ids[wins].min()))
                output = torch.empty(bs, dtype=torch.int64, device="cuda")
                _markov_sample_finish_kernel[(bs,)](
                    packed.cuda(),
                    output,
                    bs,
                    PARTS=parts,
                    WORLD=world,
                    BLOCK=triton.next_power_of_2(parts * world),
                )
                self.assertEqual(output.cpu().tolist(), expected)

    def test_finalize_accept_lengths(self):
        from sglang.kernels.ops.speculative.dspark.dspark_accept import (
            finalize_accept_lens_triton,
        )

        for dtype in (torch.int32, torch.int64):
            for bs in (1, 3, 16, 17, 257):
                with self.subTest(dtype=dtype, bs=bs):
                    correct = torch.arange(bs, dtype=torch.int32) % 6
                    trim = torch.arange(bs, dtype=torch.int32) % 4
                    prefix = torch.arange(bs, dtype=dtype) * 11
                    result = finalize_accept_lens_triton(
                        correct_len=correct.cuda(),
                        cap_trim_lens=trim.cuda(),
                        prefix_lens=prefix.cuda(),
                    )
                    self.assertTrue(torch.equal(result.commit_lens.cpu(), correct + 1))
                    self.assertTrue(
                        torch.equal(result.new_seq_lens.cpu(), prefix + correct + 1)
                    )
                    self.assertTrue(torch.equal(result.cap_trim_lens.cpu(), trim))

    def test_committed_swa_locations_and_tail_masks(self):
        from sglang.kernels.ops.speculative.dspark.commit_swa import (
            committed_swa_locations,
        )

        mapping = torch.arange(2048, dtype=torch.int64) * 3 + 7
        for width in (4, 6):
            for bs in (0, 1, 3, 43, 65):
                with self.subTest(width=width, bs=bs):
                    lens = torch.arange(bs, dtype=torch.int32) % (width + 1)
                    locations = (
                        torch.arange(bs * width, dtype=torch.int64).view(bs, width)
                        % mapping.numel()
                    )
                    locations = torch.where(
                        locations % 3 == 0, -locations - 1, locations
                    )
                    active = torch.arange(width)[None, :] < lens[:, None]
                    locations[~active] = 999999
                    expected = torch.full((bs, width), -1, dtype=torch.int32)
                    expected[active] = mapping[locations[active]].to(torch.int32)
                    output = committed_swa_locations(
                        locations.flatten().cuda(), mapping.cuda(), lens.cuda(), width
                    )
                    self.assertTrue(torch.equal(output.cpu().view(bs, width), expected))


if __name__ == "__main__":
    unittest.main()
