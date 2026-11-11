import logging
import time

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


def warmup_bf16_gemms(*, model, device):
    """Run every unquantized bf16 linear shape once over a spread of M.

    cuBLASLt loads its kernel libraries on first use, which
    CUDA_MODULE_LOADING=EAGER does not cover. A first load in the middle
    of a PP prefill, while the neighbour stage spins in an NCCL send to
    this rank, never returned from cuLibraryLoadData; load them here,
    before any traffic.
    """
    max_m = envs.SGLANG_WARMUP_BF16_GEMM_MAX_M.get()
    if max_m <= 0 or device != "cuda":
        return
    from sglang.srt.layers.linear import LinearBase
    from sglang.srt.layers.quantization.unquant import (
        UnquantizedLinearMethod,
        _bf16_gemm_dispatch_impl,
    )

    weights = {}
    for module in model.modules():
        if not isinstance(module, LinearBase) or not isinstance(
            getattr(module, "quant_method", None), UnquantizedLinearMethod
        ):
            continue
        w = getattr(module, "weight", None)
        if isinstance(w, torch.Tensor) and w.dim() == 2 and w.dtype == torch.bfloat16:
            weights.setdefault((tuple(w.shape), w.stride()), w)
    ms = set()
    p = 1
    while p <= max_m:
        # cuBLASLt picks kernels per M, including by M alignment.
        ms.update((p, p + 1, p * 3 // 2, p * 3 // 2 + 7))
        p *= 2
    ms = sorted(m for m in ms if m <= max_m)
    tic = time.perf_counter()
    with torch.inference_mode():
        for w in weights.values():
            n, k = w.shape
            x = torch.zeros((max_m, k), dtype=torch.bfloat16, device=w.device)
            for m in ms:
                _bf16_gemm_dispatch_impl(x[:m], w, None)
                _bf16_gemm_dispatch_impl(
                    x[:m],
                    w,
                    None,
                    addend=torch.zeros((m, n), dtype=w.dtype, device=w.device),
                )
            del x
        torch.cuda.synchronize()
    torch.cuda.empty_cache()
    logger.info(
        f"bf16 GEMM warmup: {len(weights)} weight shapes x {len(ms)} M values "
        f"(max {max_m}) in {time.perf_counter() - tic:.1f}s"
    )
