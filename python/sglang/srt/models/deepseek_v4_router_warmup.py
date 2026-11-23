"""Prepare the complete finite CP Router row domain on the serving stream."""


def router_prefill_rows(max_rows, cp_size):
    """All legal positive physical CP row counts, not selected prompt lengths."""
    if type(max_rows) is not int or not 0 <= max_rows <= 1 << 20:
        raise ValueError("invalid Router row cap")
    if type(cp_size) is not int or cp_size < 2 or max_rows % cp_size:
        raise ValueError("invalid CP row alignment")
    return tuple(range(cp_size, max_rows + 1, cp_size))


ROUTER_ROW_SELECTOR = router_prefill_rows


def plan_router_prefill_warmup(
    server_args, *, model_type, is_sail, cp_size, row_selector=None, runtime=None
):
    if type(cp_size) is int and cp_size == 1:
        from sglang.srt.models.deepseek_v4_prefill_context import plan_dp_prefill_rows

        return plan_dp_prefill_rows(
            server_args, model_type=model_type, is_sail=is_sail, runtime=runtime
        )
    if (
        not is_sail
        or model_type != "deepseek_v41"
        or not server_args.enable_prefill_cp
        or server_args.cp_strategy != "interleave"
        or "prefill_shapes" not in (server_args.warmups or ())
        or type(cp_size) is not int
        or cp_size < 2
    ):
        return ()
    chunk = (
        runtime.chunked_prefill_size
        if runtime is not None
        else server_args.chunked_prefill_size
    )
    if type(chunk) is not int or chunk <= 0:
        return ()
    from sglang.srt.models.deepseek_v4_warmup import max_cp_physical_rows

    limit = max_cp_physical_rows(chunk, cp_size, cp_size)
    # With no A2A backend, CP gathers the physical rows before the gate.
    gather_factor = (
        cp_size if runtime is not None and runtime.moe_a2a_backend == "none" else 1
    )
    if limit * gather_factor > 1 << 20:
        raise ValueError("Router rows after CP transport exceed the supported range")
    rows = tuple((row_selector or router_prefill_rows)(limit, cp_size))
    if any(type(n) is not int or not 0 < n <= limit or n % cp_size for n in rows):
        raise ValueError("row policy exceeded actual CP shape contract")
    return tuple(n * gather_factor for n in sorted(set(rows)))


def select_nonhash_router_gate(layers):
    """Mirror the real MoEGate dispatch, without running its forward method."""
    from sglang.srt.models import deepseek_v2

    # Mirror the actual MoEGate.forward dispatch before calling its underlying
    # GEMM directly. Never call forward/TopK/communication to prepare weights.
    if not deepseek_v2._is_sail or not deepseek_v2._is_cuda or deepseek_v2._use_aiter:
        return
    try:
        if deepseek_v2.get_exec().deterministic.enable_deterministic_inference:
            return
    except ValueError:
        return  # Runtime configuration not published: do not guess a branch.

    import torch

    from sglang.kernels.ops.gemm import bf16_fp32

    if bf16_fp32._linear_bf16_fp32_algo != "cublas":
        return
    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is None or getattr(mlp, "is_hash", True):
            continue
        item = getattr(mlp, "gate", None)
        if type(item) is not deepseek_v2.MoEGate:
            continue
        weight = item.weight
        if (
            weight.device.type != "cuda"
            or weight.dtype != torch.bfloat16
            or tuple(weight.shape) != (384, 5120)
            or not weight.is_contiguous()
            or deepseek_v2.use_intel_amx_backend(item)
        ):
            continue
        return item
    return None


def prepare_nonhash_router_startup(layers, *, local_rows, stream):
    """Select the actual gate/dispatch, then prepare only its GEMM row domain."""
    rows = tuple(local_rows)
    if any(type(n) is not int or not 0 < n <= 1 << 20 for n in rows):
        raise ValueError("invalid Router rows")
    if not rows:
        return
    gate = select_nonhash_router_gate(layers)
    if gate is None:
        return
    rows = tuple(sorted({n for n in rows if n > gate.tiny_router_gemm_max_tokens}))
    if not rows:
        return
    prepare_router_bf16_weight(gate.weight, local_rows=rows, stream=stream)


def prepare_router_bf16_weight(weight, *, local_rows, stream):
    """Prepare real BF16 GEMMs with one private zero buffer; no runtime context.

    This pure execution entry is also used unchanged by the operator probe.
    It never writes weight, consumes RNG or calls routing/communication.
    """
    rows = tuple(sorted(set(local_rows)))
    if any(type(n) is not int or not 0 < n <= 1 << 20 for n in rows):
        raise ValueError("invalid Router rows")
    if not rows:
        return
    import torch

    from sglang.kernels.ops.gemm import bf16_fp32

    if (
        weight.device.type != "cuda"
        or weight.dtype != torch.bfloat16
        or tuple(weight.shape) != (384, 5120)
        or not weight.is_contiguous()
        or bf16_fp32._linear_bf16_fp32_algo != "cublas"
    ):
        raise ValueError("weight/backend differs from BF16 cublas Router contract")
    device = weight.device
    if stream is None or stream.device != device:
        raise ValueError("pass the selected serving stream on the weight device")
    module = torch.get_device_module(device.type)
    with module.device(device):
        previous = module.current_stream(device=device)
        if module.is_current_stream_capturing():
            raise RuntimeError("Router preparation must precede graph capture")
        with module.stream(stream), torch.inference_mode():
            if module.is_current_stream_capturing():
                raise RuntimeError("selected serving stream is capturing")
            stream.wait_stream(previous)
            private_input = torch.zeros(
                (max(rows), 5120), device=device, dtype=torch.bfloat16
            )
            for n in rows:
                x = private_input[:n]
                if not x.is_contiguous() or tuple(x.stride()) != (5120, 1):
                    raise RuntimeError("private prefix changed GEMM input layout")
                output = bf16_fp32.linear_bf16_fp32(x, weight)
                if tuple(output.shape) != (n, 384) or output.dtype != torch.float32:
                    raise RuntimeError("Router GEMM output contract mismatch")
                del output, x
            stream.synchronize()
