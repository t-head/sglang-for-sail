from __future__ import annotations

import torch

from sglang.kernels.jit.utils import cache_once, load_jit


@cache_once
def _jit_dequantize_k_cache_module():
    return load_jit(
        "dpsk_v4_dequantize_k_cache",
        cuda_files=["deepseek_v4/dequantize_k_cache.cuh"],
        cuda_wrappers=[
            ("dequantize_k_cache_paged", "DequantizeKCacheKernel::run"),
        ],
    )


def dequantize_k_cache_paged_cuda(
    out: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    page_size: int,
    use_fp8_native: bool = True,
) -> None:
    """CUDA equivalent of the Triton `_dequantize_k_cache_paged_kernel`.

    Dequantizes FP8 nope + copies BF16 rope from a DSv4 paged K cache for
    a flat list of token IDs given by ``page_table``.

    Args:
        out: [num_tokens, 1, 512] bf16 output tensor.
        k_cache: [num_pages, bytes_per_page] uint8 raw cache buffer.
        page_table: [num_tokens] int32, flat token IDs into the cache.
        page_size: tokens per cache page.
        use_fp8_native: When True, use the CUDA FP8 intrinsic for
            dequantization. When False, use a fused IEEE754 bit-construction
            path for GPUs without native FP8 support (e.g. SM80/A100).
    """
    module = _jit_dequantize_k_cache_module()
    module.dequantize_k_cache_paged(
        out,
        k_cache,
        page_table,
        page_size,
        int(use_fp8_native),
    )
