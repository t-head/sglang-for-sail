# SPDX-License-Identifier: Apache-2.0
"""PPU detection and selection tests that run without accelerator hardware."""

import os
import unittest
from unittest.mock import MagicMock, patch

import torch

import sglang.srt.platforms as platforms
from sglang.srt.platforms.cpu import CpuSRTPlatform
from sglang.srt.platforms.cuda import CudaSRTPlatform
from sglang.srt.platforms.device_mixin import PlatformEnum
from sglang.srt.platforms.interface import SRTPlatform
from sglang.srt.platforms.ppu import PPUSRTPlatform
from sglang.srt.platforms.rocm import RocmSRTPlatform
from sglang.srt.utils import common
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestPPUPlatform(unittest.TestCase):
    def test_identity_preserves_cuda_device_type(self):
        platform = PPUSRTPlatform()
        self.assertEqual(platform._enum, PlatformEnum.PPU)
        self.assertEqual(platform.device_name, "ppu")
        self.assertTrue(platform.is_ppu())
        self.assertTrue(platform.is_cuda())
        self.assertTrue(platform.is_cuda_alike())
        self.assertFalse(platform.is_out_of_tree())
        self.assertFalse(platform.is_rocm())
        self.assertEqual(platform.get_device(2), torch.device("cuda", 2))

    @patch("torch.cuda.get_device_properties")
    def test_memory_queries_use_sail_cuda_interface(self, properties):
        properties.return_value.total_memory = 80 * 1024**3
        self.assertEqual(PPUSRTPlatform().get_device_total_memory(1), 80 * 1024**3)
        properties.assert_called_once_with(1)

    def test_existing_platforms_do_not_claim_ppu(self):
        for platform in (
            SRTPlatform(),
            CpuSRTPlatform(),
            CudaSRTPlatform(),
            RocmSRTPlatform(),
        ):
            with self.subTest(platform=platform):
                self.assertFalse(platform.is_ppu())


class TestPPUDiscovery(unittest.TestCase):
    def setUp(self):
        self.enterContext(
            patch.dict(os.environ, {"PPU_SDK": "v2.1.0", "SGLANG_USE_CPU_ENGINE": "0"})
        )
        self.plugins = self.enterContext(
            patch.object(platforms, "load_plugins_by_group", return_value={})
        )
        self.enterContext(
            patch.object(platforms.envs.SGLANG_PLATFORM, "get", return_value="")
        )
        self.available = self.enterContext(
            patch("torch.cuda.is_available", return_value=True)
        )
        self.enterContext(patch("torch.version.hip", None))
        self.enterContext(
            patch.object(platforms, "_is_npu_available", return_value=False)
        )
        self.enterContext(
            patch.object(platforms, "_is_xpu_available", return_value=False)
        )

    def test_sail_runtime_selects_ppu_before_cuda(self):
        self.assertIsInstance(platforms._resolve_platform(), PPUSRTPlatform)

    def test_missing_or_empty_marker_keeps_cuda_platform(self):
        for value in (None, ""):
            with self.subTest(marker=value):
                with patch.dict(os.environ):
                    if value is None:
                        os.environ.pop("PPU_SDK", None)
                    else:
                        os.environ["PPU_SDK"] = value
                    self.assertIsInstance(
                        platforms._resolve_platform(), CudaSRTPlatform
                    )

    def test_marker_without_visible_device_does_not_select_ppu(self):
        self.available.return_value = False
        self.assertFalse(platforms._is_ppu_available())
        self.assertEqual(type(platforms._resolve_platform()), SRTPlatform)

    def test_marker_does_not_override_rocm(self):
        with patch("torch.version.hip", "6.0"):
            self.assertFalse(platforms._is_ppu_available())
            self.assertIsInstance(platforms._resolve_platform(), RocmSRTPlatform)

    def test_cpu_opt_in_wins_over_ppu(self):
        with patch.dict(os.environ, {"SGLANG_USE_CPU_ENGINE": "1"}):
            self.assertIsInstance(platforms._resolve_platform(), CpuSRTPlatform)

    def test_activated_plugin_wins_over_ppu(self):
        plugin = MagicMock(return_value="external.platform:CustomPlatform")
        self.plugins.return_value = {"custom": (plugin, "custom-dist")}
        instance = SRTPlatform()
        with (
            patch.object(
                platforms,
                "_load_platform_class",
                return_value=MagicMock(return_value=instance),
            ),
            patch.object(
                platforms,
                "_is_ppu_available",
                side_effect=AssertionError(
                    "Plugin selection must precede PPU detection"
                ),
            ),
        ):
            self.assertIs(platforms._resolve_platform(), instance)


class TestPPURuntimeHelpers(unittest.TestCase):
    def setUp(self):
        self.clear_caches()
        self.addCleanup(self.clear_caches)

    @staticmethod
    def clear_caches():
        common.is_ppu.cache_clear()
        common.is_cuda.cache_clear()
        common.is_cuda_alike.cache_clear()
        common.is_hip.cache_clear()

    def test_ppu_remains_cuda_compatible_without_nvidia_version_metadata(self):
        with (
            patch.object(common, "current_platform", PPUSRTPlatform()),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.version.cuda", None),
        ):
            self.assertTrue(common.is_ppu())
            self.assertTrue(common.is_cuda())
            self.assertTrue(common.is_cuda_alike())

    def test_nvidia_runtime_helpers_keep_existing_behavior(self):
        with (
            patch.object(common, "current_platform", CudaSRTPlatform()),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.version.cuda", "13.0"),
        ):
            self.assertFalse(common.is_ppu())
            self.assertTrue(common.is_cuda())
            self.assertTrue(common.is_cuda_alike())

    def test_cpu_runtime_helpers_keep_existing_behavior(self):
        with (
            patch.object(common, "current_platform", CpuSRTPlatform()),
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.version.hip", None),
        ):
            self.assertFalse(common.is_ppu())
            self.assertFalse(common.is_cuda())
            self.assertFalse(common.is_cuda_alike())


if __name__ == "__main__":
    unittest.main()
