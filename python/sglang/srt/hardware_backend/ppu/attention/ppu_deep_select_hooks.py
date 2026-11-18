"""PPU DeepSelect capability and sparse-indexer adapters.

The PPU backend requires SM89 or newer, BF16 index logits and the precompiled
wheel. The BF16 dtype resolver honors SGLANG_OPT_DSV41_BF16_INDEX_LOGITS,
including its platform default when unset.
"""

from sglang.srt.plugins.hook_registry import HookType, plugin_hook


@plugin_hook(
    "sglang.kernels.ops.attention.deep_select.is_deep_select_supported",
    type=HookType.REPLACE,
)
def _ppu_is_deep_select_supported() -> bool:
    """Use the wheel's requirements instead of the CUDA JIT architecture list."""
    import torch

    from sglang.kernels.ops.attention.deep_select import _wheel_available
    from sglang.kernels.ops.attention.dsv4.index_logits import index_logits_dtype
    from sglang.srt.utils.common import get_device_sm

    return (
        get_device_sm() >= 89
        and index_logits_dtype() == torch.bfloat16
        and _wheel_available()
    )


@plugin_hook(
    "sglang.kernels.ops.attention.dsv4.topk.topk_transform_sparse",
    type=HookType.AROUND,
)
def _ppu_topk_transform_sparse(original_fn, logits, valid_lens, blocks, out_indices):
    """Rank compact sparse rows with the wheel, then restore block positions."""
    import torch

    if logits.dtype != torch.bfloat16 or not _ppu_is_deep_select_supported():
        return original_fn(logits, valid_lens, blocks, out_indices)

    rows, topk = out_indices.shape
    if rows == 0 or topk == 0:
        return

    from sglang.kernels.ops.attention.deep_select import _wheel_topk
    from sglang.kernels.ops.attention.dsv4.candidate_table import CANDIDATE_BLOCK_SIZE

    _, compact = _wheel_topk(
        logits,
        topk,
        sorted=False,
        end=valid_lens,
        sorted_index=False,
        output_idx=None,
        output_idx_offset=None,
        idx_oob_fill_value=-1,
        return_value=False,
    )
    valid = compact >= 0
    columns = compact.clamp_min(0)
    block_ids = blocks.gather(1, (columns // CANDIDATE_BLOCK_SIZE).to(torch.int64))
    out_indices.copy_(
        torch.where(
            valid,
            block_ids * CANDIDATE_BLOCK_SIZE + columns % CANDIDATE_BLOCK_SIZE,
            -1,
        ).to(torch.int32)
    )
