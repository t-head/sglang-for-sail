# SPDX-License-Identifier: Apache-2.0
"""Step5 region selection over FlashInfer top-k or the SAIL AOT radix selector."""

import torch

from sglang.kernels.ops.attention.step5.selection_ppu import (
    can_use_step5_selection_ppu,
)


def can_use_step5_selection_cute(scores: torch.Tensor) -> bool:
    return (
        scores.device.type == "cuda"
        and torch.version.hip is None
        and torch.cuda.get_device_capability(scores.device)[0] == 10
    )


def step5_select_regions(
    scores: torch.Tensor,
    *,
    positions: torch.Tensor,
    row_to_req: torch.Tensor,
    region_ids: torch.Tensor,
    counts: torch.Tensor,
    use_cute: bool | None = None,
    ppu_workspace=None,
    req_to_token: torch.Tensor | None = None,
    packed: torch.Tensor | None = None,
    causal_lengths: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select <=512 historical regions, sort IDs, then append current exactly once.

    Inputs follow step5_paged_scores, which owns the +0/-0 fold: the radix top-k
    orders by bit pattern, so a -0.0 would rank below a +0.0 and beat the smaller
    region id. Only the first positions//8 columns are ranked, and scores are read
    as given, so the future tail may hold stale values. Width covers positions//8.
    Slot zero is padding. Output IDs are logical, int32 [T,513], padded with
    -1; counts are int32 [T]. The FlashInfer path returns them ascending; the
    PPU radix path returns the same set in an unspecified slot order, because
    its staging pass hands out output slots with atomics. The append and the
    address fold below do not rely on that order. Both fast paths only run once
    the width exceeds the 512 history slots; narrower tiles keep every history
    region, so ranking them would be pure overhead. Caller must keep outputs
    disjoint from inputs and each other. Warm FlashInfer outside capture and
    use its shared device scratch on the sequential forward stream.

    ``packed``/``causal_lengths`` opt into the sparse-attention ABI in the same
    launch: the logical-to-physical fold that ``step5_pack_decode_regions`` used
    to perform reads the IDs this kernel has just written, so folding them here
    saves a whole launch per layer. Both must be supplied together with
    ``req_to_token``.
    """
    if scores.device.type != "cuda" or torch.version.hip is not None:
        raise ValueError("Step5 selection requires NVIDIA CUDA")
    if (
        scores.ndim != 2
        or scores.dtype != torch.float32
        or not scores.is_contiguous()
        or not 1 <= scores.shape[1] < 2**31
    ):
        raise ValueError(
            "Step5 selection requires contiguous FP32 scores [T,R], 1<=R<2**31"
        )
    rows, width = scores.shape
    for name, value, shape in (
        ("positions", positions, (rows,)),
        ("row_to_req", row_to_req, (rows,)),
        ("region_ids", region_ids, (rows, 513)),
        ("counts", counts, (rows,)),
    ):
        dtype_ok = (
            value.dtype in (torch.int32, torch.int64)
            if name in ("positions", "row_to_req")
            else value.dtype == torch.int32
        )
        if (
            value.device != scores.device
            or value.layout != torch.strided
            or tuple(value.shape) != shape
            or not value.is_contiguous()
            or not dtype_ok
        ):
            raise ValueError(f"Step5 selection invalid {name} metadata")
    if (packed is None) != (causal_lengths is None):
        raise ValueError(
            "Step5 selection packed output needs both packed and causal_lengths"
        )
    if packed is not None:
        if req_to_token is None:
            raise ValueError("Step5 selection packed output needs req_to_token")
        if (
            packed.device != scores.device
            or packed.dtype != torch.int64
            or packed.ndim != 2
            or packed.shape[0] != rows
            or packed.shape[1] not in (513, 514, 516)
            or not packed.is_contiguous()
            or causal_lengths.device != scores.device
            or causal_lengths.dtype != torch.int32
            or tuple(causal_lengths.shape) != (rows,)
            or not causal_lengths.is_contiguous()
        ):
            raise ValueError(
                "Step5 selection packed output requires int64 [T,513/514/516] "
                "and contiguous int32 [T] causal lengths"
            )
        if (
            req_to_token.device != scores.device
            or req_to_token.ndim != 2
            or req_to_token.stride(1) != 1
        ):
            raise ValueError("Step5 selection req_to_token requires CUDA integer [R,C]")
    if rows == 0:
        return region_ids, counts
    for out, others in (
        (region_ids, (scores, positions, row_to_req, counts)),
        (counts, (scores, positions, row_to_req)),
    ):
        a = out.data_ptr()
        b = a + out.numel() * out.element_size()
        for value in others:
            lo = value.data_ptr()
            if max(a, lo) < min(b, lo + value.numel() * value.element_size()):
                raise ValueError(
                    "Step5 selection outputs must not alias inputs or each other"
                )
    if use_cute is None:
        use_cute = can_use_step5_selection_cute(scores)
    if use_cute:
        from sglang.kernels.ops.attention.step5.selection_cute import (
            step5_select_regions_cute,
        )

        return step5_select_regions_cute(
            scores,
            positions,
            row_to_req,
            region_ids,
            counts,
            req_to_token=req_to_token,
            causal_lengths=causal_lengths,
        )
    selected_k = 0
    indices = region_ids
    # A tile that fits in the 512 history slots keeps all of them, so ranking it
    # would only add work; the FlashInfer branch below uses the same threshold.
    ppu_selected = (
        ppu_workspace is not None
        and width > 512
        and can_use_step5_selection_ppu(scores)
    )
    if ppu_selected:
        from sglang.kernels.ops.attention.step5.selection_ppu import (
            step5_select_regions_ppu,
        )

        indices = step5_select_regions_ppu(
            scores, positions=positions, workspace=ppu_workspace
        )
        selected_k = 513
    elif width > 512:
        import flashinfer

        from sglang.kernels.ops.attention.step5.selection_triton import (
            scan_lengths,
        )

        # The plain entry always scans the full width; the ragged one takes a
        # per-row length, and only live regions can be selected. Deterministic
        # mode also returns the IDs ascending, which _pack_regions relies on.
        meta = torch.empty(2 * rows, dtype=torch.int32, device=scores.device)
        offsets, lengths = scan_lengths(positions, row_to_req, meta, width)
        indices = flashinfer.top_k_ragged_transform(
            scores,
            offsets,
            lengths,
            512,
            deterministic=True,
            tie_break=1,
        )
        selected_k = 512
    from sglang.kernels.ops.attention.step5.selection_triton import launch

    launch(
        indices,
        positions,
        row_to_req,
        region_ids,
        counts,
        selected_k=selected_k,
        req_to_token=req_to_token,
        packed=packed,
        causal_lengths=causal_lengths,
    )
    return region_ids, counts
