from __future__ import annotations

import torch

from sglang.kernels.arch import current_arch
from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.kernels.jit.utils.common import is_sail_runtime
from sglang.srt.utils.custom_op import register_custom_op


@cache_once
def _jit_per_token_quant_fp8_module(dtype):
    if dtype != torch.bfloat16:
        raise ValueError("fused RMSNorm per-token quant requires BF16 input")
    if not is_sail_runtime() and current_arch().cuda is None:
        raise NotImplementedError("fused RMSNorm per-token quant requires CUDA or SAIL")
    # Match the standalone SAIL quantizer/AOT math mode so a fused producer
    # emits the exact same FP8 codes/scales consumed by apply_fp8_linear.
    flags = (
        ["-DSGL_PER_TOKEN_QUANT_SAIL", "--use_fast_math"] if is_sail_runtime() else []
    )
    args = make_cpp_args(dtype)
    return load_jit(
        "rmsnorm_per_token_quant",
        *args,
        cuda_files=["gemm/per_token_quant_fp8.cuh"],
        cuda_wrappers=[
            (name, f"{name}<{args}>")
            for name in ("rmsnorm_per_token_quant_fp8", "rmsnorm_per_token_quant_int8")
        ],
        extra_cuda_cflags=flags,
    )


@register_custom_op(
    op_name="rmsnorm_per_token_quant_fp8",
    mutates_args=["output_y", "output_q", "output_s"],
)
def rmsnorm_per_token_quant_fp8(
    input: torch.Tensor,
    weight: torch.Tensor,
    output_y: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    eps: float,
) -> None:
    """RMSNorm each row and quantize the normalized values to FP8 E4M3."""
    module = _jit_per_token_quant_fp8_module(input.dtype)
    module.rmsnorm_per_token_quant_fp8(
        input,
        weight,
        output_y,
        output_q,
        output_s.view(output_s.shape[0], 1),
        eps,
    )


@register_custom_op(
    op_name="rmsnorm_per_token_quant_int8",
    mutates_args=["output_y", "output_q", "output_s"],
)
def rmsnorm_per_token_quant_int8(
    input: torch.Tensor,
    weight: torch.Tensor,
    output_y: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    eps: float,
) -> None:
    """RMSNorm each row and quantize it to int8 with max(max|y|, 1e-10)/127."""
    module = _jit_per_token_quant_fp8_module(input.dtype)
    module.rmsnorm_per_token_quant_int8(
        input,
        weight,
        output_y,
        output_q,
        output_s.view(output_s.shape[0], 1),
        eps,
    )


def fused_rmsnorm_per_token_quant_fp8(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """RMSNorm + per-token FP8 quantize in one launch.

    Returns the normalized row (input dtype, contiguous), the FP8 payload and
    its ``[M, 1]`` fp32 per-token scale.
    """
    output_y = torch.empty(
        input.shape,
        dtype=input.dtype,
        device=input.device,
        memory_format=torch.contiguous_format,
    )
    output_q = torch.empty(
        input.shape,
        dtype=torch.float8_e4m3fn,
        device=input.device,
        memory_format=torch.contiguous_format,
    )
    output_s = torch.empty(
        (input.shape[0], 1), dtype=torch.float32, device=input.device
    )
    rmsnorm_per_token_quant_fp8(input, weight, output_y, output_q, output_s, eps)
    return output_y, output_q, output_s


def fused_rmsnorm_per_token_quant_int8(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """RMSNorm + per-token int8 quantize in one launch.

    Returns the normalized row (input dtype, contiguous), the int8 payload and
    its ``[M, 1]`` fp32 per-token scale, matching per_token_quant_int8's
    ``max(max|y|, 1e-10) / 127`` contract.
    """
    output_y = torch.empty(
        input.shape,
        dtype=input.dtype,
        device=input.device,
        memory_format=torch.contiguous_format,
    )
    output_q = torch.empty(
        input.shape,
        dtype=torch.int8,
        device=input.device,
        memory_format=torch.contiguous_format,
    )
    output_s = torch.empty(
        (input.shape[0], 1), dtype=torch.float32, device=input.device
    )
    rmsnorm_per_token_quant_int8(input, weight, output_y, output_q, output_s, eps)
    return output_y, output_q, output_s
