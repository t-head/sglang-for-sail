"""SAIL device operations for the SRT platform layer.

T-HEAD SAIL exposes a CUDA-compatible API through its torch backend, so
``SailDeviceMixin`` inherits all device ops from ``CudaDeviceMixin`` and
only overrides identity (``_enum``, ``device_name``).
"""

from functools import lru_cache

import torch

from sglang.srt.platforms.cuda import CudaDeviceMixin
from sglang.srt.platforms.device_mixin import PlatformEnum
from sglang.srt.platforms.interface import PlatformCapabilities, SRTPlatform

_ZW_TAG = "ZW"
_ZW810E_NAME = "ZW810E"


@lru_cache(maxsize=1)
def is_sail_available() -> bool:
    """Return True when the local device is a T-HEAD SAIL (device name holds "ZW")."""
    return bool(torch.cuda.is_available() and _ZW_TAG in torch.cuda.get_device_name())


class SailDeviceMixin(CudaDeviceMixin):
    """SAIL device ops — identical surface to CUDA via T-HEAD's torch backend."""

    _enum: PlatformEnum = PlatformEnum.SAIL
    device_name: str = "sail"
    # device_type stays "cuda" — torch.device("cuda") is the only valid
    # device-type string for SAIL devices in PyTorch.


class SailSRTPlatform(SailDeviceMixin, SRTPlatform):
    """Default in-tree T-HEAD SAIL SRT platform."""

    capabilities = PlatformCapabilities(
        supports_triton=True,
        graph_capture=True,
        piecewise_graph=True,
        supports_pynccl=True,
    )

    def is_zw810e(self) -> bool:
        return _ZW810E_NAME in self.get_device_name()

    def get_device_num_tensorcores(self, device_id: int = 0) -> int:
        if _ZW810E_NAME in self.get_device_name(device_id):
            return 20
        return super().get_device_num_tensorcores(device_id)


if is_sail_available():
    import sglang.srt.hardware_backend.sail.attention.ops_hooks  # noqa: F401
    import sglang.srt.hardware_backend.sail.attention.ppu_mhc_prenorm_hooks  # noqa: F401
    import sglang.srt.hardware_backend.sail.attention.ppu_msa_hook  # noqa: F401
    import sglang.srt.hardware_backend.sail.moe.deepep_hooks  # noqa: F401
    import sglang.srt.hardware_backend.sail.ppu_cuda_graph_hooks  # noqa: F401
