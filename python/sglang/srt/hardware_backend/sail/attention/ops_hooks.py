"""Route shared FlashMLA consumers, including DSV4, through SAIL ops.

The SAIL MLA/DSA subclasses already own their backend-specific behavior.
These three hooks cover consumers of the shared ops module outside those
subclasses. Plugins apply them before backend construction and partial binding.
"""

from sglang.srt.plugins.hook_registry import HookType, plugin_hook


@plugin_hook(
    "sglang.kernels.ops.attention.flash_mla_interface.get_mla_metadata",
    type=HookType.REPLACE,
)
def _sail_get_mla_metadata(*args, **kwargs):
    from sglang.srt.hardware_backend.sail.attention.flash_mla import get_mla_metadata

    return get_mla_metadata(*args, **kwargs)


@plugin_hook(
    "sglang.kernels.ops.attention.flash_mla_interface.flash_mla_with_kvcache",
    type=HookType.REPLACE,
)
def _sail_flash_mla_with_kvcache(*args, **kwargs):
    from sglang.srt.hardware_backend.sail.attention.flash_mla import (
        flash_mla_with_kvcache,
    )

    return flash_mla_with_kvcache(*args, **kwargs)


@plugin_hook(
    "sglang.kernels.ops.attention.flash_mla_interface.flash_mla_sparse_fwd",
    type=HookType.REPLACE,
)
def _sail_flash_mla_sparse_fwd(*args, **kwargs):
    from sglang.srt.hardware_backend.sail.attention.flash_mla import (
        flash_mla_sparse_fwd,
    )

    return flash_mla_sparse_fwd(*args, **kwargs)
