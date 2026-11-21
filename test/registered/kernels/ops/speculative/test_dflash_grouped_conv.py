"""Grouped DFlash convolution against an independent FP32 reference."""

import sys

import pytest
import torch

from sglang.kernels.ops.speculative.dflash import dflash_grouped_conv
from sglang.srt.models.dflash import _grouped_conv
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-small")


@pytest.mark.parametrize("num_tokens", [0, 1, 2304])
@pytest.mark.parametrize("block_size,taps", [(8, 2), (5, 3)])
def test_matches_reference(num_tokens, block_size, taps):
    hidden_size, group_size = 7168, 16
    num_groups = hidden_size // group_size
    gen = torch.Generator(device="cuda").manual_seed(num_tokens * 10 + taps)
    hidden = torch.randn(
        num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16, generator=gen
    )
    delta = torch.randn(
        num_tokens, taps, num_groups, device="cuda", dtype=torch.bfloat16, generator=gen
    )
    base = torch.randn(
        taps, hidden_size, device="cuda", dtype=torch.bfloat16, generator=gen
    )

    out = dflash_grouped_conv(hidden, delta, base, block_size, group_size, taps)
    if num_tokens == 0:
        assert out.shape == hidden.shape
        return
    ref = _grouped_conv(hidden, delta, base, block_size, num_groups, group_size, taps)
    # The reference returns extra rows for a batch shorter than `taps - 1`.
    ref = ref[:num_tokens]
    if taps == 2:
        # The shipped DFlash2 drafts (conv_kernel_size 2) match bit for bit.
        torch.testing.assert_close(out, ref, rtol=0, atol=0)
    else:
        torch.testing.assert_close(out, ref)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "block_size,num_tokens,strided",
    [(5, 10, False), (8, 19, True), (5, 1, False), (8, 0, False)],
)
def test_grouped_conv_preserves_fp32_accumulation(
    dtype, block_size, num_tokens, strided
):
    torch.manual_seed(11)
    groups, group_size, taps = 3, 32, 3
    row_step = 2 if strided else 1
    hidden = torch.randn(
        num_tokens * row_step, groups * group_size, device="cuda", dtype=dtype
    )[::row_step]
    delta = torch.randn(
        num_tokens * row_step, taps, groups, device="cuda", dtype=dtype
    )[::row_step]
    base = torch.randn(taps, groups * group_size, device="cuda", dtype=dtype)
    actual = dflash_grouped_conv(hidden, delta, base, block_size, group_size, taps)

    expected = torch.zeros(
        num_tokens, groups, group_size, device="cuda", dtype=torch.float32
    )
    base_fp32 = base.float().view(taps, groups, group_size)
    hidden_fp32 = hidden.float().view(num_tokens, groups, group_size)
    for row in range(num_tokens):
        for tap in range(min(taps, row % block_size + 1)):
            coefficient = base_fp32[tap] + delta[row, tap, :, None].float()
            expected[row] += coefficient * hidden_fp32[row - tap]
    tolerance = {torch.float32: 1e-5, torch.float16: 2e-3, torch.bfloat16: 1e-2}[dtype]
    torch.testing.assert_close(
        actual,
        expected.flatten(1).to(dtype),
        rtol=tolerance,
        atol=tolerance,
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
