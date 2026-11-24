from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_finalize_module(
    weight_dtype: torch.dtype,
    top_k: int,
    hidden: int,
    has_shared: bool,
    use_pdl: bool,
    output_dtype: torch.dtype = torch.bfloat16,
    round_routed: bool = False,
) -> Module:
    # FP32 输出的向量同样限制在 16 字节，兼容 pre-Blackwell。
    widths = (8, 4, 2) if output_dtype == torch.bfloat16 else (4, 2)
    vec = next(v for v in widths if hidden % v == 0)
    args = make_cpp_args(
        weight_dtype, top_k, vec, has_shared, use_pdl, output_dtype, round_routed
    )
    return load_jit(
        "moe_finalize_fuse_shared",
        *args,
        cuda_files=["moe/moe_finalize_fuse_shared.cuh"],
        cuda_wrappers=[("run", f"MoeFinalizeKernel<{args}>::run")],
    )


def moe_finalize_fuse_shared(
    gemm2_out: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    expert_weights: torch.Tensor,
    shared_output: Optional[torch.Tensor],
    top_k: int,
    enable_pdl: bool = False,
    out: Optional[torch.Tensor] = None,
    routing_ready: bool = True,
    *,
    output_dtype: torch.dtype = torch.bfloat16,
    round_routed: bool = False,
) -> torch.Tensor:
    """以 FP32 累加路由专家结果，跳过 index=-1 的行，最后加 shared_output。

    gemm2_out 为 [num_permuted_rows, padded_hidden] BF16；路由 index 为
    [num_tokens * top_k] INT32，权重为 [num_tokens, top_k] FP32 或 BF16。
    shared_output 为可选的 [num_tokens, hidden] BF16，hidden <= padded_hidden。
    out 为可选的单位列步长输出缓冲区，允许行步长和截取 padded hidden；
    output_dtype 控制最终存储，round_routed 在加 shared 前先将路由结果截为 BF16。
    默认保持原 FP32 累加后一次 BF16 存储的行为。

    启用 PDL 时，routing_ready=True 表示路由在本 kernel launch 前已经完成，
    可在 PDL wait 前读取；shared_output 始终在 wait 后读取。若生产者在等待
    路由之前就触发 dependents，必须传 False。
    """
    assert output_dtype in (torch.bfloat16, torch.float32)
    assert gemm2_out.dtype == torch.bfloat16 and gemm2_out.dim() == 2
    assert expert_weights.dtype in (torch.float32, torch.bfloat16)
    assert expert_weights.dim() == 2
    assert expanded_idx_to_permuted_idx.dtype == torch.int32
    num_tokens, top_k_check = expert_weights.shape
    assert top_k_check == top_k
    hidden_dim = (
        out.shape[1]
        if out is not None
        else shared_output.shape[1]
        if shared_output is not None
        else gemm2_out.shape[1]
    )
    if shared_output is not None:
        assert shared_output.dtype == torch.bfloat16 and shared_output.dim() == 2
        assert shared_output.shape == (num_tokens, hidden_dim)
    assert 0 < hidden_dim <= gemm2_out.shape[1] and hidden_dim % 2 == 0
    if out is None:
        out = torch.empty(
            num_tokens, hidden_dim, dtype=output_dtype, device=gemm2_out.device
        )
    else:
        assert out.dtype == output_dtype and out.shape == (num_tokens, hidden_dim)
    module = _jit_finalize_module(
        expert_weights.dtype,
        top_k,
        hidden_dim,
        shared_output is not None,
        enable_pdl,
        output_dtype,
        round_routed,
    )
    module.run(
        out,
        gemm2_out,
        expanded_idx_to_permuted_idx,
        expert_weights,
        shared_output,
        routing_ready,
    )
    return out
