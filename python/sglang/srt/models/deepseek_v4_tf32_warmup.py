"""Precompile standalone PPU mHC prenorm without changing kernel arithmetic."""


def plan_tf32_prenorm_warmup(
    server_args, *, model_type, is_sail, cp_size, deep_gemm_enabled, runtime=None
):
    """Reuse audited rank-local bounds, independently of MegaMHC dispatch."""
    if not deep_gemm_enabled:
        return ()
    from sglang.srt.models.deepseek_v4_warmup import plan_prefill_warmup

    # The shared planner describes CP/DP row bounds. TF32 eligibility is not
    # conditional on the separate MegaMHC feature being enabled.
    return plan_prefill_warmup(
        server_args,
        model_type=model_type,
        is_sail=is_sail,
        cp_size=cp_size,
        mega_mhc_enabled=True,
        runtime=runtime,
    )


def prepare_tf32_prenorm_startup(fn, *, max_rows, stream, num_splits):
    """Visit the legal row domain; native full-signature caching deduplicates it.

    DeepGEMM compile-only mode builds and loads the runtime but skips the GEMM
    launch. This avoids copying a version-specific native tuning table into
    Python, or assuming that the outer n_splits bucket identifies the kernel.
    Call only during startup, before this worker can accept inference work.
    """
    if type(max_rows) is not int or not 0 <= max_rows <= 1 << 20:
        raise ValueError("max_rows must be an integer in the supported row range")
    # The serving path asserts split-k == 1 on SAIL and computes split-k
    # elsewhere; precompile the exact signature the worker will later launch.
    if num_splits is not None and (type(num_splits) is not int or num_splits < 1):
        raise ValueError("num_splits must be None or a positive integer")
    if max_rows == 0:
        return False

    import torch

    device = fn.device
    shape = tuple(fn.shape)
    if (
        device.type != "cuda"
        or fn.dtype != torch.float32
        or len(shape) != 2
        or any(type(n) is not int or n <= 0 for n in shape)
        or not fn.is_contiguous()
    ):
        raise ValueError("need the actual contiguous FP32 prenorm weight matrix")
    if stream is None or stream.device != device:
        raise ValueError("pass the actual execution stream on the weight device")

    module = torch.get_device_module(device.type)
    with module.device(device):
        previous_stream = module.current_stream(device=device)
        if module.is_current_stream_capturing():
            raise RuntimeError("startup preparation cannot run inside graph capture")
        with module.stream(stream):
            if module.is_current_stream_capturing():
                raise RuntimeError("target stream is capturing; prepare before capture")
            stream.wait_stream(previous_stream)
            device_index = device.index
            if device_index is None:
                device_index = module.current_device()
            with torch.random.fork_rng(devices=[device_index]), torch.inference_mode():
                import deep_gemm

                if not all(
                    callable(getattr(deep_gemm, name, None))
                    for name in (
                        "get_compile_mode",
                        "set_compile_mode",
                        "tf32_hc_prenorm_gemm",
                    )
                ):
                    raise RuntimeError("PPU DeepGEMM compile-only API is required")
                previous_mode = deep_gemm.get_compile_mode()
                x = out = sqrsum = None
                try:
                    deep_gemm.set_compile_mode(1)
                    if deep_gemm.get_compile_mode() != 1:
                        raise RuntimeError("DeepGEMM did not enter compile-only mode")
                    n, k = shape
                    # No values are read in compile-only mode. Allocate once,
                    # and keep the serving dtype, contiguity and prefix strides.
                    x = torch.empty((max_rows, k), dtype=torch.bfloat16, device=device)
                    out = torch.empty(
                        (max_rows, n), dtype=torch.float32, device=device
                    )
                    sqrsum = torch.empty(
                        (max_rows,), dtype=torch.float32, device=device
                    )
                    # Descending order prepares the largest scratch capacity
                    # first and avoids repeated growth for common configurations.
                    for rows in range(max_rows, 0, -1):
                        deep_gemm.tf32_hc_prenorm_gemm(
                            x[:rows],
                            fn,
                            out[:rows],
                            sqrsum[:rows],
                            num_splits=num_splits,
                        )
                    stream.synchronize()
                finally:
                    deep_gemm.set_compile_mode(previous_mode)
                    del x, out, sqrsum
    return True
