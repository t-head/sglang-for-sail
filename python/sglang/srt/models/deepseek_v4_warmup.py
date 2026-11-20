"""Prepare eligible BF16 MegaMHC prefill rows before incoming requests."""


def plan_prefill_warmup(
    server_args, *, model_type, is_sail, cp_size, mega_mhc_enabled, runtime=None
):
    """Bound startup preparation by actual CP chunking and existing opt-ins."""
    if not mega_mhc_enabled:
        return ()
    if type(cp_size) is int and cp_size == 1:
        from sglang.srt.models.deepseek_v4_prefill_context import plan_dp_prefill_rows

        return plan_dp_prefill_rows(
            server_args, model_type=model_type, is_sail=is_sail, runtime=runtime
        )
    if (
        not is_sail
        or model_type != "deepseek_v41"
        or not mega_mhc_enabled
        or not server_args.enable_prefill_cp
        or server_args.cp_strategy != "interleave"
        or "prefill_shapes" not in (server_args.warmups or ())
        or type(cp_size) is not int
        or cp_size < 1
    ):
        return ()
    chunk = (
        runtime.chunked_prefill_size
        if runtime is not None
        else server_args.chunked_prefill_size
    )
    if type(chunk) is not int or chunk <= 0:
        return ()
    limit = max_cp_physical_rows(chunk, cp_size, cp_size)
    # Boundary callers may retain logical rows before physical CP padding.
    # Keep the physical upper bound and include all logical tail rows below it.
    return tuple(range(1, limit + 1))


def prefill_warmup_rows(max_rows, alignment):
    """Representative aligned tails of row-count buckets, not prompt lengths."""
    if type(max_rows) is not int or not 0 <= max_rows <= 1 << 20:
        raise ValueError("max_rows must be an integer in the supported row range")
    if type(alignment) is not int or alignment < 1:
        raise ValueError("alignment must be a positive integer")
    if max_rows == 0:
        return ()
    limit = (max_rows + alignment - 1) // alignment * alignment
    tails = set()
    if limit >= 128 and limit % 32:
        tails.add(limit)
    bucket = 256
    while bucket < limit * 2:
        n = min(bucket, limit) - alignment
        n = n // alignment * alignment
        if n >= 128 and n % 32:
            tails.add(n)
        if bucket >= limit:
            break
        bucket *= 2
    return tuple(sorted(tails))


def max_cp_physical_rows(max_tokens, cp_size, alignment):
    """Match maximum logical rank rows followed by physical CP padding."""
    if type(max_tokens) is not int or max_tokens < 0:
        raise ValueError("max_tokens must be a nonnegative integer")
    if type(cp_size) is not int or cp_size < 1:
        raise ValueError("cp_size must be a positive integer")
    if type(alignment) is not int or alignment < 1:
        raise ValueError("alignment must be a positive integer")
    logical = (max_tokens + cp_size - 1) // cp_size
    return (logical + alignment - 1) // alignment * alignment


def normalize_local_rows(local_rows):
    counts = tuple(local_rows)
    if any(type(n) is not int or not 0 <= n <= 1 << 20 for n in counts):
        raise ValueError("local rows must be integers in the wrapper's supported range")
    return tuple(sorted({n for n in counts if n}))


def prepare_mega_mhc_bf16_startup(hc, *, local_rows, stream):
    """Use actual boundary weights with private synthetic BF16 input tensors."""
    rows = normalize_local_rows(local_rows)
    cfg = hc.cfg
    if cfg.mult != 4 or cfg.hidden != 5120 or not cfg.pre_from_prev:
        raise ValueError("need a real shifted 4-stream, hidden=5120 boundary")
    if type(cfg.sinkhorn_iters) is not int or cfg.sinkhorn_iters < 1:
        raise ValueError("invalid Sinkhorn iteration count")
    if stream is None:
        raise ValueError(
            "pass the actual eager execution stream, not a temporary stream"
        )
    if not rows:
        return

    import torch

    from sglang.kernels.ops.layernorm.mhc_mega import try_mhc_mega_boundary

    weight = hc.norm.weight
    device = weight.device
    tensors = (
        (weight, (5120,), torch.bfloat16),
        (hc.fn, (24, 4 * 5120), torch.float32),
        (hc.scale, (3,), torch.float32),
        (hc.base, (24,), torch.float32),
    )
    if device.type != "cuda" or any(
        t.device != device
        or t.dtype != dtype
        or tuple(t.shape) != shape
        or not t.is_contiguous()
        for t, shape, dtype in tensors
    ):
        raise ValueError(
            "boundary weights do not match the actual BF16 wrapper contract"
        )
    if stream.device != device:
        raise ValueError("execution stream and weights must be on the same device")
    module = torch.get_device_module(device.type)
    with module.device(device):
        previous_stream = module.current_stream(device=device)
        if module.is_current_stream_capturing():
            raise RuntimeError("startup preparation cannot run inside a graph capture")
        with module.stream(stream), torch.inference_mode():
            if module.is_current_stream_capturing():
                raise RuntimeError("target stream is capturing; prepare before capture")
            stream.wait_stream(previous_stream)
            for n in rows:
                x = torch.zeros((n, 5120), device=device, dtype=torch.bfloat16)
                residual = torch.zeros(
                    (n, 4, 5120), device=device, dtype=torch.bfloat16
                )
                pre = torch.ones((n, 4), device=device, dtype=torch.float32)
                post = torch.zeros_like(pre)
                comb = torch.zeros((n, 4, 4), device=device, dtype=torch.float32)
                result = try_mhc_mega_boundary(
                    x,
                    residual,
                    pre,
                    post,
                    comb,
                    hc.fn,
                    hc.scale,
                    hc.base,
                    weight,
                    cfg.rms_eps,
                    cfg.eps,
                    hc.norm.variance_epsilon,
                    cfg.sinkhorn_iters,
                )
                if result is None:
                    raise RuntimeError(
                        "production wrapper declined the boundary; not prepared"
                    )
                del result, x, residual, pre, post, comb
            stream.synchronize()
