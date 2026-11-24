from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from sglang.kernels.arch import current_arch
from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.kernels.jit.utils.common import is_sail_runtime
from sglang.kernels.jit.utils.compile.toolchain import cuda_home
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_per_token_quant_fp8_module(dtype: torch.dtype) -> Module:
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise RuntimeError(
            f"Unsupported dtype {dtype}. Supported: float16, bfloat16, float32"
        )
    use_fast_math = current_arch().is_sm(9, 0)
    math_mode = "fast_math" if use_fast_math else "precise_math"
    cuda_cflags = ["--use_fast_math"] if use_fast_math else []
    include_paths = []
    if is_sail_runtime():
        cuda_cflags.append("-DSGL_PER_TOKEN_QUANT_SAIL")
        # Newer SAIL SDKs ship the bulk-copy PTX wrappers under include/cccl.
        if torch.cuda.get_device_capability()[0] >= 9:
            cccl = Path(cuda_home()) / "include" / "cccl"
            if (cccl / "cuda" / "ptx").is_file():
                cuda_cflags.append("-DSGL_PER_TOKEN_QUANT_SAIL_BULK")
                include_paths.append(str(cccl))
    args = make_cpp_args(dtype)
    return load_jit(
        "per_token_quant_fp8",
        math_mode,
        *args,
        cuda_files=["gemm/per_token_quant_fp8.cuh"],
        cuda_wrappers=[("per_token_quant_fp8", f"per_token_quant_fp8<{args}>")],
        extra_cuda_cflags=cuda_cflags,
        extra_include_paths=include_paths,
    )


@register_custom_op(
    op_name="per_token_quant_fp8",
    mutates_args=["output_q", "output_s"],
)
def per_token_quant_fp8(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
) -> None:
    """Dynamically quantize each row to FP8 E4M3."""
    module = _jit_per_token_quant_fp8_module(input.dtype)
    module.per_token_quant_fp8(input, output_q, output_s.view(output_s.shape[0], 1))
