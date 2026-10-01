# SPDX-License-Identifier: Apache-2.0
"""T-Head PPU device identity through the SAIL CUDA-compatible runtime."""

from sglang.srt.platforms.cuda import CudaDeviceMixin
from sglang.srt.platforms.device_mixin import PlatformEnum
from sglang.srt.platforms.interface import SRTPlatform


class PPUDeviceMixin(CudaDeviceMixin):
    """Reuse SAIL's torch.cuda API while exposing a distinct PPU identity."""

    _enum: PlatformEnum = PlatformEnum.PPU
    device_name: str = "ppu"
    # PPU is the platform identity; PyTorch's device type remains "cuda".
    device_type: str = "cuda"

    def is_cuda(self) -> bool:
        return True


class PPUSRTPlatform(PPUDeviceMixin, SRTPlatform):
    """PPU SRT platform with conservative inherited capability defaults.

    Kernel builds, attention implementations, quantization, and graph support
    require separate PPU integration and device validation.
    """
