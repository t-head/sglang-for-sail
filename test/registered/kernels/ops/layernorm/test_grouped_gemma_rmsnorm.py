from unittest.mock import patch

import pytest
import torch

from sglang.kernels.ops.layernorm.grouped_gemma_rmsnorm import grouped_gemma_rmsnorm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _reference_grouped_gemma_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    group_size: int,
    eps: float,
    compute_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Mirrors GroupedGemmaRMSNorm.forward (hyperconnection.py); keep them in sync."""
    x_float = x.to(compute_dtype)
    hidden = x_float.shape[-1]
    x_grouped = x_float.reshape(*x_float.shape[:-1], hidden // group_size, group_size)
    variance = x_grouped.pow(2).mean(dim=-1, keepdim=True)
    x_norm = (x_grouped * torch.rsqrt(variance + eps)).flatten(-2)
    return x_norm * (1.0 + weight.to(compute_dtype))


# Preserve output-dtype tolerances while keeping the oracle in FP64. Rounding
# the reference to BF16 can add a full-ULP discrepancy at a rounding boundary;
# the kernel itself computes in FP32, like the eager implementation.
_TOLERANCES = {
    torch.bfloat16: dict(rtol=5e-3, atol=5e-3),
    torch.float16: dict(rtol=1e-3, atol=1e-3),
}


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "num_tokens,hidden_size,group_size",
    [
        (0, 10240, 2560),  # idle DP rank in the PLE forward path
        (1, 10240, 2560),  # production shape (HC 4 x 2560)
        (7, 10240, 2560),
        (128, 10240, 2560),
        (33, 1024, 512),
        (5, 512, 512),  # single group == plain gemma rmsnorm
        (1024, 2048, 1024),
    ],
)
@pytest.mark.parametrize("eps", [1e-6, 1e-5])
def test_grouped_gemma_rmsnorm_correctness(
    dtype, num_tokens, hidden_size, group_size, eps
):
    torch.manual_seed(0)
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device="cuda")
    weight = torch.randn(hidden_size, dtype=dtype, device="cuda") * 0.2

    out = grouped_gemma_rmsnorm(x, weight, group_size, eps)
    expected = _reference_grouped_gemma_rmsnorm(
        x, weight, group_size, eps, compute_dtype=torch.float64
    )

    assert out.dtype == dtype
    torch.testing.assert_close(out.to(torch.float64), expected, **_TOLERANCES[dtype])


@pytest.mark.parametrize("num_tokens", [0, 64])
def test_grouped_gemma_rmsnorm_out_param(num_tokens):
    x = torch.randn(num_tokens, 10240, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(10240, dtype=torch.bfloat16, device="cuda") * 0.2
    out = torch.empty_like(x)

    result = grouped_gemma_rmsnorm(x, weight, 2560, 1e-6, out=out)
    expected = _reference_grouped_gemma_rmsnorm(
        x, weight, 2560, 1e-6, compute_dtype=torch.float64
    )

    assert result.dtype == x.dtype
    assert result.data_ptr() == out.data_ptr()
    torch.testing.assert_close(
        result.to(torch.float64), expected, **_TOLERANCES[x.dtype]
    )


def test_grouped_gemma_rmsnorm_bad_group_size():
    x = torch.randn(4, 10240, dtype=torch.bfloat16, device="cuda")
    weight = torch.zeros(10240, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="group_size"):
        grouped_gemma_rmsnorm(x, weight, 1000, 1e-6)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("use_out", [False, True])
def test_grouped_gemma_rmsnorm_empty_without_jit(dtype, use_out):
    """Idle input must work even when the JIT compiler is unavailable."""
    x = torch.empty((0, 10240), dtype=dtype, device="cuda")
    weight = torch.empty((10240,), dtype=dtype, device="cuda")
    backing = torch.full((1, 10240), 7, dtype=dtype, device="cuda")
    out = backing[:0] if use_out else None
    with patch(
        "sglang.kernels.ops.layernorm.grouped_gemma_rmsnorm."
        "_jit_grouped_gemma_rmsnorm_module",
        side_effect=AssertionError("Empty input must not request a JIT module"),
    ):
        result = grouped_gemma_rmsnorm(x, weight, 2560, out=out)
    assert result.shape == x.shape
    assert result.dtype == dtype
    assert result.device == x.device
    if use_out:
        assert result.is_set_to(out)
        torch.testing.assert_close(backing, torch.full_like(backing, 7))


@pytest.mark.parametrize(
    "dtype,group_size,error",
    [
        (torch.float32, 2560, "Unsupported dtype"),
        (torch.bfloat16, 0, "Unsupported group_size"),
        (torch.bfloat16, -512, "Unsupported group_size"),
        (torch.bfloat16, 1000, "Unsupported group_size"),
    ],
)
def test_grouped_gemma_rmsnorm_empty_validates_args(dtype, group_size, error):
    """Skipping compilation must preserve the operator's argument errors."""
    x = torch.empty((0, 10240), dtype=dtype, device="cuda")
    weight = torch.empty((10240,), dtype=dtype, device="cuda")
    with (
        patch(
            "sglang.kernels.ops.layernorm.grouped_gemma_rmsnorm."
            "_jit_grouped_gemma_rmsnorm_module",
            side_effect=AssertionError("Empty input must not request a JIT module"),
        ),
        pytest.raises(RuntimeError, match=error),
    ):
        grouped_gemma_rmsnorm(x, weight, group_size)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
