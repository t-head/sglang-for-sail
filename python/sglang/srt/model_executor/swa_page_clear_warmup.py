"""Private SWA page-clear preparation; never release real allocator pages."""


def plan_swa_page_clear_warmup(
    *,
    enabled,
    is_ppu,
    is_swa_allocator,
    swa_req_ring,
    page_size,
    mapping_dtype,
    representative_dtype,
    debug_mode,
):
    flags = (enabled, is_ppu, is_swa_allocator, swa_req_ring, debug_mode)
    if any(type(value) is not bool for value in flags):
        raise ValueError("eligibility/ring/debug fields must be actual booleans")
    if not enabled or not is_ppu or not is_swa_allocator or swa_req_ring:
        return ()
    if type(page_size) is not int or page_size < 1:
        raise ValueError("page_size must be a positive resolved integer")
    if type(mapping_dtype) is not str or type(representative_dtype) is not str:
        raise ValueError("dtype names must come from actual plain metadata strings")
    if mapping_dtype != "torch.int64" or representative_dtype not in (
        "torch.int32",
        "torch.int64",
    ):
        raise ValueError("SWA mapping must be int64 and representatives int32/int64")
    block_offsets = 1 << (page_size - 1).bit_length()
    block_pages = max(1, 256 // block_offsets)
    return tuple(
        {
            "page_size": page_size,
            "index_stride": stride,
            "num_pages": block_pages + 1,
            "block_pages": block_pages,
            "block_offsets": block_offsets,
            "check_page_mappings": page_size > 1 and debug_mode,
            "representative_dtype": representative_dtype,
        }
        for stride in dict.fromkeys((1, page_size))
    )


def _check_schedule_stream(torch, device, schedule_stream):
    device = torch.device(device)
    if device.type != "cuda" or device.index is None:
        raise ValueError("SWA preparation requires the explicit scheduler CUDA device")
    if torch.cuda.current_device() != device.index:
        raise ValueError("scheduler device is not the current device")
    if schedule_stream is None or torch.device(schedule_stream.device) != device:
        raise ValueError("schedule_stream belongs to a different device")
    if torch.cuda.current_stream(device).cuda_stream != schedule_stream.cuda_stream:
        raise ValueError(
            "enter the actual Scheduler.schedule_stream before preparing SWA"
        )
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("SWA preparation must be outside graph capture")
    return device


def _private_swa_inputs(
    torch,
    *,
    device,
    page_size,
    num_pages,
    index_stride,
    representative_dtype="torch.int64",
):
    """Valid distinct FULL pages and positive distinct SWA pages, private only."""
    if index_stride not in (1, page_size) or num_pages < 1:
        raise ValueError("unsupported private representative layout")
    if representative_dtype not in ("torch.int32", "torch.int64"):
        raise ValueError("SWA representatives must be int32/int64")
    index_dtype = getattr(torch, representative_dtype.removeprefix("torch."))
    mapping = torch.full(
        ((num_pages + 4) * page_size + 1,), -7, dtype=torch.int64, device=device
    )
    mapping[:page_size] = 0  # Reserved FULL page0, never selected.
    mapping[-1] = -1  # Same trailing last_loc=-1 sentinel as the allocator.
    mapping[page_size : (num_pages + 1) * page_size] = torch.arange(
        (num_pages + 2) * page_size,
        (2 * num_pages + 2) * page_size,
        dtype=torch.int64,
        device=device,
    )
    token_rows = torch.arange(
        page_size, (num_pages + 1) * page_size, dtype=index_dtype, device=device
    )
    representatives = token_rows[::page_size]
    if index_stride == 1:
        contiguous = torch.empty((num_pages,), dtype=index_dtype, device=device)
        contiguous.copy_(representatives)
        representatives = contiguous
    return representatives, mapping


def prepare_swa_page_clear_startup(
    *,
    device,
    schedule_stream,
    enabled,
    is_ppu,
    is_swa_allocator,
    swa_req_ring,
    page_size,
    mapping_dtype,
    representative_dtype,
    debug_mode,
):
    plan = plan_swa_page_clear_warmup(
        enabled=enabled,
        is_ppu=is_ppu,
        is_swa_allocator=is_swa_allocator,
        swa_req_ring=swa_req_ring,
        page_size=page_size,
        mapping_dtype=mapping_dtype,
        representative_dtype=representative_dtype,
        debug_mode=debug_mode,
    )
    if not plan:
        return
    import torch

    device = _check_schedule_stream(torch, device, schedule_stream)
    from sglang.kernels.ops.memory.allocator import get_and_clear_swa_pages

    with torch.inference_mode():
        for case in plan:
            representatives, mapping = _private_swa_inputs(
                torch,
                device=device,
                page_size=page_size,
                num_pages=case["num_pages"],
                index_stride=case["index_stride"],
                representative_dtype=case["representative_dtype"],
            )
            result = get_and_clear_swa_pages(
                representatives,
                mapping,
                page_size,
                check_page_mappings=case["check_page_mappings"],
            )
            del result, representatives, mapping
        schedule_stream.synchronize()


def prepare_swa_allocator_startup(
    allocator, *, device, schedule_stream, representative_dtypes
):
    """Read allocator metadata only; never pass its mapping to the kernel."""
    from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator

    if not isinstance(allocator, SWATokenToKVPoolAllocator) or allocator.swa_req_ring:
        return
    dtypes = tuple(dict.fromkeys(representative_dtypes))
    if not dtypes:
        raise ValueError("pass the actual request and cache index dtypes")
    metadata = dict(
        enabled=True,
        is_ppu=True,
        is_swa_allocator=True,
        swa_req_ring=False,
        page_size=allocator.page_size,
        mapping_dtype=str(allocator.full_to_swa_index_mapping.dtype),
        debug_mode=allocator.page_size > 1 and allocator.swa_attn_allocator.debug_mode,
    )
    # Reject unsupported metadata before preparing even the first dtype.
    for dtype in dtypes:
        plan_swa_page_clear_warmup(**metadata, representative_dtype=dtype)
    for dtype in dtypes:
        prepare_swa_page_clear_startup(
            device=device,
            schedule_stream=schedule_stream,
            **metadata,
            representative_dtype=dtype,
        )
