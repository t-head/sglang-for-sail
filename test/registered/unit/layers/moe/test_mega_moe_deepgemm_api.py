"""Unit tests for the DeepGEMM MegaMoE interface."""

import sys
import unittest
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, call, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.moe import MoeA2ABackend, MoeRunnerBackend, mega_moe
from sglang.srt.layers.moe import topk as topk_module
from sglang.srt.layers.moe.fused_moe_triton import layer as fused_moe_layer_module
from sglang.srt.layers.moe.topk import TopKConfig
from sglang.srt.layers.moe.utils import draft_model_build_scope
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from sglang.srt.runtime_context import get_context, get_exec, get_flags, get_parallel

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class TestDeepGemmMegaMoeApi(CustomTestCase):
    def setUp(self):
        super().setUp()
        mega_moe._MEGA_MOE_SYMM_BUFFER.clear()
        self.original_max_num_sms = mega_moe._mega_moe_max_num_sms
        self.deep_gemm = ModuleType("deep_gemm")
        self.deep_gemm.num_sms = 132
        self.deep_gemm.get_num_sms = MagicMock(
            side_effect=lambda: self.deep_gemm.num_sms
        )
        self.deep_gemm.set_num_sms = MagicMock(
            side_effect=lambda num_sms: setattr(self.deep_gemm, "num_sms", num_sms)
        )
        max_num_sms = patch.object(mega_moe, "_mega_moe_max_num_sms", return_value=130)
        self.max_num_sms = max_num_sms.start()
        self.addCleanup(max_num_sms.stop)

    def tearDown(self):
        mega_moe._MEGA_MOE_SYMM_BUFFER.clear()
        super().tearDown()

    def test_mma_type_follows_layer_w4a4_mode(self):
        experts = SimpleNamespace(_mega_moe_nvfp4=False, _mega_moe_w4a4=False)
        with patch.object(mega_moe, "_is_sail", False):
            self.assertEqual(mega_moe._mega_moe_mma_type(experts), "fp8xfp4")
            experts._mega_moe_w4a4 = True
            self.assertEqual(mega_moe._mega_moe_mma_type(experts), "mxf4xmxf4")

    def test_ppu_mma_type_follows_transformed_weights(self):
        experts = SimpleNamespace(
            _mega_moe_nvfp4=False,
            _mega_moe_w4a4=False,
            _mega_moe_mma_type="mxf4xmxf4",
        )
        with patch.object(mega_moe, "_is_sail", True):
            self.assertEqual(mega_moe._mega_moe_mma_type(experts), "mxf4xmxf4")
            experts._mega_moe_mma_type = "fp8xmxfp4"
            self.assertEqual(mega_moe._mega_moe_mma_type(experts), "fp8xmxfp4")
        with patch.object(mega_moe, "_is_sail", False):
            self.assertEqual(mega_moe._mega_moe_mma_type(experts), "fp8xfp4")

    def test_mxf4_buffer_uses_typed_api(self):
        deep_gemm = self.deep_gemm
        expected_buffer = object()
        deep_gemm.get_symm_buffer_for_mega_moe = MagicMock(return_value=expected_buffer)
        group = object()

        with patch.dict(sys.modules, {"deep_gemm": deep_gemm}):
            actual_buffer = mega_moe._get_mega_moe_symm_buffer(
                group,
                num_experts=8,
                num_max_tokens_per_rank=64,
                num_topk=2,
                hidden=128,
                intermediate_hidden=256,
                mma_type="mxf4xmxf4",
            )

        self.assertIs(actual_buffer, expected_buffer)
        call = deep_gemm.get_symm_buffer_for_mega_moe.call_args
        self.assertEqual(call.kwargs.get("mma_type"), "mxf4xmxf4")
        self.assertNotIn("use_fp8_dispatch", call.kwargs)

    def test_ppu_buffer_uses_mma_type_without_legacy_dispatch_flag(self):
        deep_gemm = self.deep_gemm
        deep_gemm.get_symm_buffer_for_mega_moe = MagicMock(
            side_effect=(object(), object(), object())
        )
        group = object()

        with (
            patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
            patch.object(mega_moe, "_is_sail", True),
        ):
            buffers = [
                mega_moe._get_mega_moe_symm_buffer(
                    group, 8, 64, 2, 128, 256, mma_type=mma_type, activation="situ"
                )
                for mma_type in ("fp8xmxfp4", "mxf4xmxf4", "fp8xmxfp4", "bf16xmxfp4")
            ]

        self.assertIs(buffers[0], buffers[2])
        self.assertIsNot(buffers[0], buffers[1])
        self.assertIsNot(buffers[0], buffers[3])
        self.assertEqual(
            deep_gemm.get_symm_buffer_for_mega_moe.call_args_list,
            [
                call(
                    group,
                    8,
                    64,
                    2,
                    128,
                    256,
                    num_shared_experts=0,
                    mma_type=mma_type,
                    activation="situ",
                )
                for mma_type in ("fp8xmxfp4", "mxf4xmxf4", "bf16xmxfp4")
            ],
        )
        deep_gemm.get_num_sms.assert_called()
        self.assertEqual(deep_gemm.num_sms, 132)

    def test_ppu_buffer_cache_separates_situ_activation(self):
        deep_gemm = self.deep_gemm
        deep_gemm.get_symm_buffer_for_mega_moe = MagicMock(
            side_effect=(object(), object())
        )
        group = object()
        with (
            patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
            patch.object(mega_moe, "_is_sail", True),
        ):
            buffers = [
                mega_moe._get_mega_moe_symm_buffer(
                    group,
                    8,
                    64,
                    2,
                    128,
                    256,
                    mma_type="mxf4xmxf4",
                    activation=activation,
                )
                for activation in ("swiglu", "situ", "swiglu")
            ]
        self.assertIs(buffers[0], buffers[2])
        self.assertIsNot(buffers[0], buffers[1])
        self.assertEqual(
            [
                call.kwargs["activation"]
                for call in deep_gemm.get_symm_buffer_for_mega_moe.call_args_list
            ],
            ["swiglu", "situ"],
        )

    def test_ppu_sm89_and_newer_use_physical_sm_budget(self):
        with (
            patch.object(mega_moe, "_is_sail", True),
            patch.object(
                torch.cuda,
                "get_device_properties",
                return_value=SimpleNamespace(multi_processor_count=140),
            ),
            patch.object(
                mega_moe.envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_RESERVED_SMS,
                "get",
                return_value=8,
            ),
        ):
            for sm in (89, 90):
                with patch.object(mega_moe, "_device_sm", sm):
                    self.assertEqual(self.original_max_num_sms.__wrapped__(), 132)
        with (
            patch.object(mega_moe, "_is_sail", False),
            patch.object(mega_moe, "_device_sm", 89),
        ):
            self.assertIsNone(self.original_max_num_sms.__wrapped__())

    def test_buffer_cache_separates_mma_types(self):
        deep_gemm = self.deep_gemm
        expected_buffers = (object(), object())
        deep_gemm.get_symm_buffer_for_mega_moe = MagicMock(side_effect=expected_buffers)
        group = object()

        with patch.dict(sys.modules, {"deep_gemm": deep_gemm}):
            actual_buffers = tuple(
                mega_moe._get_mega_moe_symm_buffer(
                    group,
                    num_experts=8,
                    num_max_tokens_per_rank=64,
                    num_topk=2,
                    hidden=128,
                    intermediate_hidden=256,
                    mma_type=mma_type,
                )
                for mma_type in ("fp8xfp4", "mxf4xmxf4")
            )

        self.assertEqual(actual_buffers, expected_buffers)
        self.assertEqual(deep_gemm.get_symm_buffer_for_mega_moe.call_count, 2)
        self.assertEqual(
            [
                call.kwargs["mma_type"]
                for call in deep_gemm.get_symm_buffer_for_mega_moe.call_args_list
            ],
            ["fp8xfp4", "mxf4xmxf4"],
        )

    def test_buffer_cache_uses_effective_sm_budget(self):
        deep_gemm = self.deep_gemm
        deep_gemm.get_symm_buffer_for_mega_moe = MagicMock(
            side_effect=lambda *_args, **_kwargs: SimpleNamespace(
                num_sms=deep_gemm.get_num_sms()
            )
        )
        group = object()
        buffers = []

        for current_num_sms, expected_num_sms in (
            (132, 130),
            (131, 130),
            (96, 96),
            (97, 96),
            (132, 130),
        ):
            with self.subTest(current_num_sms=current_num_sms):
                deep_gemm.set_num_sms(current_num_sms)
                buf = self._get_test_buffer(group)
                buffers.append(buf)
                self.assertEqual(buf.num_sms, expected_num_sms)
                self.assertEqual(deep_gemm.get_num_sms(), current_num_sms)
                with mega_moe._configure_mega_moe_deep_gemm_num_sms(deep_gemm):
                    self.assertEqual(buf.num_sms, deep_gemm.get_num_sms())
                self.assertEqual(deep_gemm.get_num_sms(), current_num_sms)

        self.assertIs(buffers[0], buffers[1])
        self.assertIs(buffers[0], buffers[4])
        self.assertIs(buffers[2], buffers[3])
        self.assertIsNot(buffers[0], buffers[2])
        self.assertEqual(deep_gemm.get_symm_buffer_for_mega_moe.call_count, 2)

    def test_buffer_allocation_without_sm_override(self):
        deep_gemm = self.deep_gemm
        self.max_num_sms.return_value = None
        deep_gemm.get_symm_buffer_for_mega_moe = MagicMock(
            side_effect=lambda *_args, **_kwargs: deep_gemm.get_num_sms()
        )

        buf = self._get_test_buffer(object())
        self.assertEqual(buf, 132)
        deep_gemm.set_num_sms.assert_not_called()

    def test_buffer_allocation_failure_restores_sm_budget(self):
        deep_gemm = self.deep_gemm
        deep_gemm.get_symm_buffer_for_mega_moe = MagicMock(
            side_effect=RuntimeError("allocation failed")
        )

        with self.assertRaisesRegex(RuntimeError, "allocation failed"):
            self._get_test_buffer(object())

        self.assertEqual(deep_gemm.get_num_sms(), 132)
        self.assertEqual(deep_gemm.set_num_sms.call_args_list, [call(130), call(132)])
        self.assertEqual(mega_moe._MEGA_MOE_SYMM_BUFFER, {})

    def test_mxf4_weight_transform_uses_matching_mma_type(self):
        from sglang.srt.layers.quantization.mxfp4 import Mxfp4MoEMethod

        deep_gemm = ModuleType("deep_gemm")
        deep_gemm.transform_sf_into_required_layout = MagicMock(
            side_effect=lambda _sf, mn, k, recipe, num_groups, disable_ue8m0_cast: (
                torch.zeros((num_groups, mn, max(1, k // 32)), dtype=torch.int32)
            )
        )
        deep_gemm.transform_weights_for_mega_moe = MagicMock(
            side_effect=lambda l1, l2, **_kwargs: (l1, l2)
        )
        method = object.__new__(Mxfp4MoEMethod)
        method.use_marlin = False
        method.use_deep_gemm = False
        method.use_mega_moe = True
        layer = SimpleNamespace(
            w13_weight=torch.nn.Parameter(
                torch.zeros((1, 32, 16), dtype=torch.uint8), requires_grad=False
            ),
            w13_weight_scale=torch.nn.Parameter(
                torch.zeros((1, 32, 1), dtype=torch.uint8), requires_grad=False
            ),
            w2_weight=torch.nn.Parameter(
                torch.zeros((1, 32, 16), dtype=torch.uint8), requires_grad=False
            ),
            w2_weight_scale=torch.nn.Parameter(
                torch.zeros((1, 32, 1), dtype=torch.uint8), requires_grad=False
            ),
        )

        with (
            patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
            patch("sglang.srt.layers.quantization.mxfp4._is_sail", False),
            patch.object(mega_moe, "_mega_moe_mma_type", return_value="mxf4xmxf4"),
            # Toy shapes.
            patch.object(mega_moe, "check_mega_moe_shapes"),
        ):
            method.process_weights_after_loading(layer)

        call = deep_gemm.transform_weights_for_mega_moe.call_args
        self.assertEqual(call.kwargs.get("mma_type"), "mxf4xmxf4")

    def test_ppu_mxf4_transform_releases_legacy_layout(self):
        from sglang.srt.layers.quantization import mxfp4

        deep_gemm = ModuleType("deep_gemm")

        def transform(l1, l2):
            return (
                (l1[0].clone(), l1[1].to(torch.uint16)),
                (l2[0], l2[1].to(torch.uint16)),
            )

        deep_gemm.transform_weights_for_mega_moe = MagicMock(side_effect=transform)
        method = object.__new__(mxfp4.Mxfp4MoEMethod)
        method._mega_moe_mma_type = "mxf4xmxf4"
        method.use_marlin = False
        method.use_deep_gemm = False
        method.use_mega_moe = True
        method.use_deepgemm_mxfp4_w4a16_valu = False
        method.use_deepgemm_mxfp4_w4a16_mma = False
        method.with_bias = False
        layer = SimpleNamespace(
            w13_weight=torch.nn.Parameter(
                torch.zeros((1, 32, 16), dtype=torch.uint8), requires_grad=False
            ),
            w13_weight_scale=torch.nn.Parameter(
                torch.full((1, 32, 1), 127, dtype=torch.uint8), requires_grad=False
            ),
            w2_weight=torch.nn.Parameter(
                torch.zeros((1, 32, 16), dtype=torch.uint8), requires_grad=False
            ),
            w2_weight_scale=torch.nn.Parameter(
                torch.full((1, 32, 1), 126, dtype=torch.uint8), requires_grad=False
            ),
        )
        original_weight_ptr = layer.w13_weight.data_ptr()

        with (
            patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
            patch.object(mxfp4, "_is_sail", True),
            patch.object(mxfp4, "preprocess_mxfp4_scales") as preprocess,
            patch.object(torch.cuda, "empty_cache"),
        ):
            method.process_weights_after_loading(layer)
            method.process_weights_after_loading(layer)

        self.assertEqual(deep_gemm.transform_weights_for_mega_moe.call_count, 1)
        l1, l2 = deep_gemm.transform_weights_for_mega_moe.call_args.args
        self.assertEqual(l1[1].dtype, torch.uint8)
        self.assertEqual(l2[1].dtype, torch.uint8)
        self.assertTrue(torch.all(l1[1] == 127))
        self.assertTrue(torch.all(l2[1] == 126))
        self.assertNotEqual(layer.w13_weight.data_ptr(), original_weight_ptr)
        self.assertEqual(
            layer.w13_weight.data_ptr(), layer.mega_l1_weights[0].data_ptr()
        )
        self.assertEqual(
            layer.w13_weight_scale.data_ptr(), layer.mega_l1_weights[1].data_ptr()
        )
        self.assertEqual(
            layer.w2_weight.data_ptr(), layer.mega_l2_weights[0].data_ptr()
        )
        self.assertEqual(
            layer.w2_weight_scale.data_ptr(), layer.mega_l2_weights[1].data_ptr()
        )
        self.assertEqual(layer.w13_weight_scale.dtype, torch.uint16)
        preprocess.assert_not_called()
        self.assertEqual(layer._mxfp4_backend, "deep_gemm")
        self.assertEqual(layer._mega_moe_mma_type, "mxf4xmxf4")
        self.assertTrue(layer._mega_moe_weights_built)

    def test_ppu_mega_moe_precision_is_available_before_weight_postprocessing(self):
        """Kimi preallocates its buffer inside load_weights, before quant repacking."""
        from sglang.srt.layers.quantization import mxfp4

        for precision, expected in (
            ("w4a4", "mxf4xmxf4"),
            ("w4a8", "fp8xmxfp4"),
            ("w4a16", "bf16xmxfp4"),
            ("w4a16_mma", "bf16xmxfp4"),
        ):
            with (
                self.subTest(precision=precision),
                patch.object(mxfp4, "_is_sail", True),
                patch.object(mega_moe, "_is_sail", True),
                patch.object(mxfp4, "get_device_sm", return_value=89),
                patch.object(
                    mxfp4, "get_moe_runner_backend", return_value=MoeRunnerBackend.AUTO
                ),
                patch.object(
                    mxfp4, "get_moe_a2a_backend", return_value=MoeA2ABackend.MEGAMOE
                ),
                patch.object(
                    mxfp4,
                    "get_exec",
                    return_value=SimpleNamespace(
                        moe=SimpleNamespace(flashinfer_mxfp4_moe_precision="default")
                    ),
                ),
                patch.object(mxfp4.deep_gemm_wrapper, "ENABLE_JIT_DEEPGEMM", True),
                patch.object(
                    mxfp4.envs.SGLANG_SAIL_DEEPGEMM_MXFP4_W4A8,
                    "get",
                    return_value=precision == "w4a8",
                ),
                patch.object(
                    mxfp4.envs.SGLANG_SAIL_DEEPGEMM_MXFP4_W4A16,
                    "get",
                    return_value=precision == "w4a16",
                ),
                patch.object(
                    mxfp4.envs.SGLANG_SAIL_DEEPGEMM_MXFP4_W4A16_MMA,
                    "get",
                    return_value=precision == "w4a16_mma",
                ),
            ):
                method = mxfp4.Mxfp4MoEMethod("model.layers.0.experts")
                layer = torch.nn.Module()
                layer.num_local_experts = 1
                layer.hidden_size = 1024
                layer.intermediate_size_per_partition = 128
                layer.moe_runner_config = SimpleNamespace(activation="situ")
                layer._mega_moe_nvfp4 = False
                layer._mega_moe_w4a4 = False
                layer._mega_moe_weights_built = False
                method.create_weights(layer, 1, 1024, 128, torch.bfloat16)

                self.assertEqual(mega_moe._mega_moe_mma_type(layer), expected)
                self.assertFalse(layer._mega_moe_weights_built)
                self.assertEqual(layer.w13_weight_scale.dtype, torch.uint8)

    def test_ppu_mega_moe_preserves_precision_specific_weight_layouts(self):
        """MegaMoE must see raw scales and retain both W4A8 inverse scales."""
        from sglang.srt.layers.quantization import mxfp4

        for precision in ("w4a8", "w4a16", "w4a16_mma"):
            with self.subTest(precision=precision):
                layer = SimpleNamespace(
                    w13_weight=torch.nn.Parameter(
                        torch.zeros((1, 32, 16), dtype=torch.uint8), requires_grad=False
                    ),
                    w13_weight_scale=torch.nn.Parameter(
                        torch.full((1, 32, 1), 127, dtype=torch.uint8),
                        requires_grad=False,
                    ),
                    w2_weight=torch.nn.Parameter(
                        torch.zeros((1, 32, 16), dtype=torch.uint8), requires_grad=False
                    ),
                    w2_weight_scale=torch.nn.Parameter(
                        torch.full((1, 32, 1), 126, dtype=torch.uint8),
                        requires_grad=False,
                    ),
                )

                def transform(l1, l2):
                    self.assertTrue(torch.all(l1[1] == 127))
                    self.assertTrue(torch.all(l2[1] == 126))
                    if precision == "w4a8":
                        return (*l1, 2.0), (*l2, 4.0)
                    return (l1[0], l1[1].to(torch.uint16)), (
                        l2[0],
                        l2[1].to(torch.uint16),
                    )

                deep_gemm = ModuleType("deep_gemm")
                deep_gemm.transform_weights_for_mega_moe = MagicMock(
                    side_effect=AssertionError("W4A4 transform changes the precision")
                )
                deep_gemm.transform_weights_for_w4a8_mega_moe = transform
                deep_gemm.transform_weights_for_w4a16_mega_moe = transform
                with (
                    patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
                    patch.object(mxfp4, "_is_sail", True),
                    patch.object(
                        mxfp4,
                        "get_moe_runner_backend",
                        return_value=MoeRunnerBackend.AUTO,
                    ),
                    patch.object(
                        mxfp4, "get_moe_a2a_backend", return_value=MoeA2ABackend.MEGAMOE
                    ),
                    patch.object(
                        mxfp4,
                        "get_exec",
                        return_value=SimpleNamespace(
                            moe=SimpleNamespace(
                                flashinfer_mxfp4_moe_precision="default"
                            )
                        ),
                    ),
                    patch.object(
                        mxfp4.envs.SGLANG_SAIL_DEEPGEMM_MXFP4_W4A8,
                        "get",
                        return_value=precision == "w4a8",
                    ),
                    patch.object(
                        mxfp4.envs.SGLANG_SAIL_DEEPGEMM_MXFP4_W4A16,
                        "get",
                        return_value=precision == "w4a16",
                    ),
                    patch.object(
                        mxfp4.envs.SGLANG_SAIL_DEEPGEMM_MXFP4_W4A16_MMA,
                        "get",
                        return_value=precision == "w4a16_mma",
                    ),
                    patch.object(torch.cuda, "empty_cache"),
                ):
                    method = mxfp4.Mxfp4MoEMethod("model.layers.0.experts")
                    method.process_weights_after_loading(layer)
                    method.process_weights_after_loading(layer)

                self.assertFalse(method.use_deepgemm_mxfp4_w4a8)
                self.assertFalse(method.use_deepgemm_mxfp4_w4a16_valu)
                self.assertFalse(method.use_deepgemm_mxfp4_w4a16_mma)
                self.assertTrue(layer._mega_moe_weights_built)
                if precision == "w4a8":
                    self.assertEqual(layer._mega_moe_mma_type, "fp8xmxfp4")
                    self.assertEqual(layer.mega_l1_weights[2], 2.0)
                    self.assertEqual(layer.mega_l2_weights[2], 4.0)
                else:
                    self.assertEqual(layer._mega_moe_mma_type, "bf16xmxfp4")
                    self.assertEqual(layer.mega_l1_weights[1].dtype, torch.uint16)
                self.assertEqual(
                    layer.w13_weight.data_ptr(), layer.mega_l1_weights[0].data_ptr()
                )
                self.assertEqual(
                    layer.w2_weight_scale.data_ptr(),
                    layer.mega_l2_weights[1].data_ptr(),
                )

    def test_ppu_mxf4_rejects_expert_bias_before_conversion(self):
        from sglang.srt.layers.quantization import mxfp4

        method = object.__new__(mxfp4.Mxfp4MoEMethod)
        method.use_marlin = False
        method.use_deep_gemm = False
        method.use_mega_moe = True
        method.use_deepgemm_mxfp4_w4a16_valu = False
        method.use_deepgemm_mxfp4_w4a16_mma = False
        method.with_bias = True
        with patch.object(mxfp4, "_is_sail", True):
            with self.assertRaisesRegex(ValueError, "expert bias"):
                method.process_weights_after_loading(SimpleNamespace())

    def test_mxf4_pre_dispatch_uses_typed_api(self):
        deep_gemm = ModuleType("deep_gemm")
        deep_gemm.mega_moe_pre_dispatch = MagicMock()
        deep_gemm.fp8_fp4_mega_moe = MagicMock()
        buffer = SimpleNamespace(
            x=object(),
            x_sf=object(),
            topk_idx=object(),
            topk_weights=object(),
        )
        experts = SimpleNamespace(
            num_experts=8,
            mega_l1_weights=object(),
            mega_l2_weights=object(),
            should_fuse_routed_scaling_factor_in_topk=True,
            moe_runner_config=SimpleNamespace(swiglu_limit=None),
            _mega_moe_weights_built=True,
            _mega_moe_nvfp4=False,
        )
        topk_output = SimpleNamespace(
            topk_ids=torch.tensor([[0, 1]]),
            topk_weights=torch.tensor([[0.6, 0.4]]),
        )
        moe = SimpleNamespace(
            config=SimpleNamespace(
                hidden_size=4,
                num_experts_per_tok=2,
                moe_intermediate_size=8,
                swiglu_limit=None,
            ),
            experts=experts,
            gate=MagicMock(
                return_value=torch.empty((1, 8)), e_score_correction_bias_vl=None
            ),
            topk=MagicMock(return_value=topk_output),
            is_hash=False,
            num_fused_shared_experts=0,
            layer_id=0,
            routed_scaling_factor=1.0,
            mega_shared_l1_weights=None,
            mega_shared_l2_weights=None,
        )

        with (
            patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
            patch.object(mega_moe, "_is_sail", False),
            patch.object(mega_moe, "_device_sm", 100),
            patch.object(mega_moe, "_mega_moe_mma_type", return_value="mxf4xmxf4"),
            patch.object(
                mega_moe,
                "_get_mega_moe_symm_buffer",
                return_value=buffer,
            ),
            patch.object(
                mega_moe,
                "_configure_mega_moe_deep_gemm_num_sms",
                return_value=nullcontext(),
            ),
            # Toy shapes.
            patch.object(mega_moe, "check_mega_moe_shapes"),
            patch.object(
                mega_moe.ExpertLocationDispatchInfo,
                "init_new",
                return_value=object(),
            ),
            get_parallel().override(
                moe_ep_group=SimpleNamespace(device_group=object())
            ),
        ):
            mega_moe._run_mega_routed(
                moe,
                torch.zeros((1, 4)),
                forward_batch=None,
                input_ids_global=None,
                num_tokens=1,
            )

        self.assertTrue(deep_gemm.mega_moe_pre_dispatch.called)
        call = deep_gemm.mega_moe_pre_dispatch.call_args
        self.assertEqual(call.kwargs.get("mma_type"), "mxf4xmxf4")
        self.assertNotIn("use_fp4_acts", call.kwargs)

    def test_run_mega_routed_experts_generic_entry(self):
        deep_gemm = ModuleType("deep_gemm")
        deep_gemm.fp8_fp4_mega_moe = MagicMock()
        buffer = SimpleNamespace(
            x=object(),
            x_sf=object(),
            topk_idx=object(),
            topk_weights=object(),
        )
        experts = SimpleNamespace(
            num_experts=8,
            mega_l1_weights=object(),
            mega_l2_weights=object(),
            _mega_moe_weights_built=True,
            _mega_moe_nvfp4=False,
        )
        hidden_states = torch.zeros((3, 4), dtype=torch.bfloat16)
        topk_ids = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.int64)
        topk_weights = torch.full((3, 2), 0.5, dtype=torch.bfloat16)

        with (
            patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
            patch.object(mega_moe, "_device_sm", 100),
            patch.object(mega_moe, "_mega_moe_mma_type", return_value="fp8xfp4"),
            patch.object(mega_moe, "mega_moe_pre_dispatch") as pre_dispatch,
            patch.object(
                mega_moe, "_get_mega_moe_symm_buffer", return_value=buffer
            ) as get_buffer,
            patch.object(
                mega_moe,
                "_configure_mega_moe_deep_gemm_num_sms",
                return_value=nullcontext(),
            ),
            # Toy shapes.
            patch.object(mega_moe, "check_mega_moe_shapes"),
            patch(
                "sglang.srt.runtime_context.get_parallel",
                return_value=SimpleNamespace(
                    moe_ep_group=SimpleNamespace(device_group=object())
                ),
            ),
        ):
            out = mega_moe.run_mega_routed_experts(
                experts,
                hidden_states,
                topk_ids,
                topk_weights,
                hidden_size=4,
                intermediate_size=8,
                top_k=2,
                num_tokens=3,
                activation_clamp=7.0,
                routed_scaling_factor=1.0,
            )

        self.assertEqual(out.shape, (3, 4))
        self.assertEqual(out.dtype, torch.bfloat16)
        buf_call = get_buffer.call_args
        self.assertEqual(buf_call.kwargs.get("num_topk"), 2)
        self.assertEqual(buf_call.kwargs.get("hidden"), 4)
        self.assertEqual(buf_call.kwargs.get("intermediate_hidden"), 8)
        # The kernel wants int32 ids and fp32 weights regardless of the router dtype.
        ids_arg, weights_arg = pre_dispatch.call_args.args[1:3]
        self.assertEqual(ids_arg.dtype, torch.int32)
        self.assertEqual(weights_arg.dtype, torch.float32)
        mega_call = deep_gemm.fp8_fp4_mega_moe.call_args
        self.assertIs(mega_call.args[1], experts.mega_l1_weights)
        self.assertIs(mega_call.args[2], experts.mega_l2_weights)
        self.assertEqual(mega_call.kwargs.get("activation_clamp"), 7.0)

    def test_v41_megamoe_selects_image_bias_on_local_rows(self):
        """Image rows must retain their router bias after attention-TP sharding.

        Equal logits make the two biases select disjoint experts. A numerical
        expert boundary exposes accidentally routing image rows as text, and
        padded rows must contribute nothing.
        """
        for fused_shared in (0, 1):
            for scale_in_topk in (False, True):
                with self.subTest(
                    fused_shared=fused_shared, scale_in_topk=scale_in_topk
                ):
                    moe = self._v41_moe(fused_shared, scale_in_topk)
                    ids = torch.tensor([7, 99, 99])
                    hidden = torch.ones((3, 4))
                    batch = SimpleNamespace(
                        moe_num_token_non_padded=lambda: torch.tensor(
                            2, dtype=torch.int32
                        )
                    )

                    def run_experts(_experts, x, topk_ids, weights, **kwargs):
                        # Routed experts multiply x by their physical ID + 1;
                        # the rank-1 shared slot (ID 5) is an identity expert.
                        factors = (topk_ids + 1).float()
                        factors[topk_ids == 5] = 1.0
                        self.assertTrue((topk_ids[2] == -1).all())
                        return x * (
                            (factors * weights).sum(-1, keepdim=True)
                            * kwargs["routed_scaling_factor"]
                        )

                    with (
                        get_flags().moe.override(a2a_backend=MoeA2ABackend.MEGAMOE),
                        get_parallel().override(moe_ep_size=2, moe_ep_rank=1),
                        patch.object(
                            mega_moe.ExpertLocationDispatchInfo,
                            "init_new",
                            return_value=None,
                        ),
                        patch.object(mega_moe, "run_mega_routed_experts", run_experts),
                        # The fused padded-row fill requires Triton/GPU.
                        patch(
                            "sglang.srt.multimodal.dsv41.vl_routing.is_cuda",
                            return_value=False,
                        ),
                        patch.object(topk_module, "_is_cuda", False),
                        patch.object(
                            topk_module, "_can_fuse_padded_region", return_value=False
                        ),
                    ):
                        out = mega_moe._run_mega_routed(
                            moe, hidden, batch, ids, num_tokens=3
                        )
                        text_out = mega_moe._run_mega_routed(
                            moe, hidden, batch, None, num_tokens=3
                        )
                    # Text selects logical 0/1; image selects logical 2/3. With
                    # a shared slot, rank-1 routed IDs shift by one (2/3 -> 3/4).
                    expected = torch.tensor(
                        [3.0 + fused_shared, 7.0 + 3.0 * fused_shared, 0.0]
                    )[:, None].expand_as(hidden)
                    torch.testing.assert_close(out, expected)
                    text_expected = torch.tensor(
                        [3.0 + fused_shared, 3.0 + fused_shared, 0.0]
                    )[:, None].expand_as(hidden)
                    torch.testing.assert_close(text_out, text_expected)

    def test_v41_megamoe_empty_rank_still_participates(self):
        """An idle DP rank must reach A2A combine without running its router."""
        moe = self._v41_moe(0, False)
        hidden = torch.empty((0, 4))

        def run_experts(_experts, x, ids, weights, **kwargs):
            self.assertIsNone(ids)
            self.assertIsNone(weights)
            # The collective boundary owns the result even on an idle rank.
            return x.new_empty((0, kwargs["hidden_size"]))

        with patch.object(mega_moe, "run_mega_routed_experts", run_experts):
            out = mega_moe._run_mega_routed(
                moe, hidden, None, torch.empty(0, dtype=torch.long), num_tokens=0
            )
        self.assertEqual(out.shape, (0, 4))
        moe.gate.assert_not_called()

    @staticmethod
    def _v41_moe(fused_shared, scale_in_topk):
        return SimpleNamespace(
            config=SimpleNamespace(
                hidden_size=4,
                num_experts_per_tok=2,
                moe_intermediate_size=8,
                image_token_id=99,
            ),
            experts=SimpleNamespace(
                should_fuse_routed_scaling_factor_in_topk=scale_in_topk,
                moe_runner_config=SimpleNamespace(swiglu_limit=None),
            ),
            gate=MagicMock(
                return_value=torch.zeros((3, 4)),
                e_score_correction_bias=torch.tensor([4.0, 3.0, 0.0, 0.0]),
                e_score_correction_bias_vl=torch.tensor([0.0, 0.0, 3.0, 4.0]),
            ),
            topk=SimpleNamespace(
                topk_config=TopKConfig(
                    top_k=2 + fused_shared,
                    num_fused_shared_experts=fused_shared,
                    renormalize=True,
                    routed_scaling_factor=2.0,
                    apply_routed_scaling_factor_on_output=scale_in_topk,
                )
            ),
            is_hash=False,
            num_fused_shared_experts=fused_shared,
            layer_id=0,
            routed_scaling_factor=2.0,
            mega_shared_l1_weights=None,
            mega_shared_l2_weights=None,
        )

    def test_shape_check_uses_pinned_fp8_fp4_alignment(self):
        # Packed SF groups need 128 elements, not a 16-byte token SF row.
        mega_moe.check_mega_moe_shapes(2048, 768, "fp8xfp4")
        mega_moe.check_mega_moe_shapes(5120, 2304, "fp8xfp4")
        mega_moe.check_mega_moe_shapes(512, 128, "fp8xfp4")
        with self.assertRaisesRegex(ValueError, "intermediate_size.*multiple of 128"):
            mega_moe.check_mega_moe_shapes(2048, 736, "fp8xfp4")
        with self.assertRaisesRegex(ValueError, "hidden_size.*multiple of 512"):
            mega_moe.check_mega_moe_shapes(2016, 768, "fp8xfp4")
        with self.assertRaisesRegex(ValueError, "hidden_size.*multiple of 512"):
            mega_moe.check_mega_moe_shapes(256, 128, "fp8xfp4")
        mega_moe.check_mega_moe_shapes(4096, 1536, "fp8xfp4")
        mega_moe.check_mega_moe_shapes(4096, 1024, "mxf4xmxf4")
        # 768 is a multiple of 256, so the NVFP4 (g16) rule accepts it.
        mega_moe.check_mega_moe_shapes(2048, 768, "nvfp4xnvfp4")
        with self.assertRaisesRegex(ValueError, "intermediate_size.*multiple of 256"):
            mega_moe.check_mega_moe_shapes(2048, 384, "nvfp4xnvfp4")

    def test_ppu_mxf4_uses_typed_pre_dispatch_and_fp4_compute(self):
        deep_gemm = ModuleType("deep_gemm")
        deep_gemm.mega_moe_pre_dispatch = MagicMock()
        deep_gemm.fp4_fp4_mega_moe = MagicMock()
        deep_gemm.fp8_fp4_mega_moe = MagicMock()
        deep_gemm.bf16_fp4_mega_moe = MagicMock()
        buffer = SimpleNamespace(
            x=object(), x_sf=object(), topk_idx=object(), topk_weights=object()
        )
        experts = SimpleNamespace(
            num_experts=8,
            mega_l1_weights=object(),
            mega_l2_weights=object(),
            _mega_moe_nvfp4=False,
        )
        with (
            patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
            patch.object(mega_moe, "_is_sail", True),
            patch.object(mega_moe, "_device_sm", 89),
            patch.object(mega_moe, "_mega_moe_mma_type", return_value="mxf4xmxf4"),
            patch.object(mega_moe, "_get_mega_moe_symm_buffer", return_value=buffer),
            patch.object(
                mega_moe,
                "_configure_mega_moe_deep_gemm_num_sms",
                return_value=nullcontext(),
            ),
            patch.object(mega_moe, "check_mega_moe_shapes"),
            patch(
                "sglang.srt.runtime_context.get_parallel",
                return_value=SimpleNamespace(
                    moe_ep_group=SimpleNamespace(device_group=object())
                ),
            ),
        ):
            out = mega_moe.run_mega_routed_experts(
                experts,
                torch.zeros((1, 8), dtype=torch.bfloat16)[:, ::2],
                torch.tensor([[0, 2, 1, 3]])[:, ::2],
                torch.tensor([[0.5, 1.0, 0.5, 1.0]])[:, ::2],
                hidden_size=4,
                intermediate_size=8,
                top_k=2,
                num_tokens=1,
            )

        self.assertEqual(out.shape, (1, 4))
        for actual, expected in zip(
            deep_gemm.mega_moe_pre_dispatch.call_args.args[3:],
            (buffer.x, buffer.x_sf, buffer.topk_idx, buffer.topk_weights),
            strict=True,
        ):
            self.assertIs(actual, expected)
        staged_input, staged_ids, staged_weights = (
            deep_gemm.mega_moe_pre_dispatch.call_args.args[:3]
        )
        self.assertEqual(staged_input.dtype, torch.bfloat16)
        self.assertEqual(staged_ids.dtype, torch.int64)
        self.assertEqual(staged_weights.dtype, torch.float32)
        for tensor in (staged_input, staged_ids, staged_weights):
            self.assertTrue(tensor.is_contiguous())
        torch.testing.assert_close(staged_ids, torch.tensor([[0, 1]]))
        self.assertEqual(
            deep_gemm.mega_moe_pre_dispatch.call_args.kwargs,
            {"num_tokens": 1, "group_size": 32, "mma_type": "mxf4xmxf4"},
        )
        self.assertNotIn(
            "shared_l1_weights", deep_gemm.fp4_fp4_mega_moe.call_args.kwargs
        )
        deep_gemm.fp4_fp4_mega_moe.assert_called_once()
        deep_gemm.fp8_fp4_mega_moe.assert_not_called()

    def test_ppu_experts_ready_from_sm89(self):
        with patch.object(mega_moe, "_is_sail", True):
            for sm in (88, 89, 90):
                with patch.object(mega_moe, "_device_sm", sm):
                    self.assertEqual(
                        mega_moe.is_mega_moe_experts_ready(
                            SimpleNamespace(_mega_moe_weights_built=True)
                        ),
                        sm >= 89,
                    )
                    self.assertFalse(
                        mega_moe.is_mega_moe_experts_ready(
                            SimpleNamespace(_mega_moe_weights_built=False)
                        )
                    )

    def test_kimi_k3_situ_selects_ppu_mma_api_without_changing_cuda_path(self):
        from sglang.srt.models.kimi_k3 import KimiK3MoE

        buffer = SimpleNamespace(
            x=object(),
            x_sf=object(),
            topk_idx=object(),
            topk_weights=object(),
        )
        experts = SimpleNamespace(
            num_experts=8,
            mega_l1_weights=object(),
            mega_l2_weights=object(),
            moe_runner_config=SimpleNamespace(
                gemm1_alpha=3.0,
                gemm1_clamp_limit=17.0,
            ),
            should_fuse_routed_scaling_factor_in_topk=True,
        )
        owner = SimpleNamespace(
            experts=experts,
            _mega_top_k=2,
            moe_hidden_size=4,
            _mega_intermediate_size=8,
        )
        owner._mega_moe_buffer = lambda: KimiK3MoE._mega_moe_buffer(owner)
        topk_output = SimpleNamespace(
            topk_ids=torch.tensor([[0, 1]], dtype=torch.int32),
            topk_weights=torch.tensor([[0.6, 0.4]]),
        )

        for ppu, mma_type, num_tokens in (
            (True, "mxf4xmxf4", 1),
            (True, "fp8xmxfp4", 1),
            (True, "bf16xmxfp4", 1),
            (True, "mxf4xmxf4", 0),
            (True, "fp8xmxfp4", 0),
            (True, "bf16xmxfp4", 0),
            (False, "mxf4xmxf4", 1),
            (False, "fp8xfp4", 1),
            (False, "fp8xfp4", 0),
        ):
            with self.subTest(ppu=ppu, mma_type=mma_type, num_tokens=num_tokens):
                deep_gemm = ModuleType("deep_gemm")
                deep_gemm.mega_moe_pre_dispatch = MagicMock()
                deep_gemm.fp4_fp4_mega_moe = MagicMock()
                deep_gemm.fp8_fp4_mega_moe = MagicMock()
                deep_gemm.bf16_fp4_mega_moe = MagicMock()
                with (
                    patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
                    patch("sglang.srt.models.kimi_k3.is_sail", return_value=ppu),
                    patch.object(
                        mega_moe, "_mega_moe_mma_type", return_value=mma_type
                    ) as select_type,
                    patch.object(
                        mega_moe, "_get_mega_moe_symm_buffer", return_value=buffer
                    ) as get_buffer,
                    patch.object(
                        mega_moe,
                        "_configure_mega_moe_deep_gemm_num_sms",
                        return_value=nullcontext(),
                    ),
                    patch(
                        "sglang.srt.models.kimi_k3.get_parallel",
                        return_value=SimpleNamespace(
                            moe_ep_group=SimpleNamespace(device_group=object())
                        ),
                    ),
                    patch(
                        "sglang.kernels.ops.moe.dsv4.mega_moe_pre_dispatch"
                    ) as triton_dispatch,
                ):
                    result = KimiK3MoE._forward_mega_experts(
                        owner,
                        torch.zeros((num_tokens, 4), dtype=torch.bfloat16),
                        topk_output,
                    )

                self.assertEqual(result.shape, (num_tokens, 4))
                self.assertEqual(select_type.call_args_list, [call(experts)] * 2)
                self.assertEqual(get_buffer.call_args.kwargs["mma_type"], mma_type)
                if ppu:
                    self.assertEqual(get_buffer.call_args.kwargs["activation"], "situ")
                else:
                    self.assertNotIn("activation", get_buffer.call_args.kwargs)
                if ppu:
                    dispatch_args = deep_gemm.mega_moe_pre_dispatch.call_args.args
                    self.assertEqual(dispatch_args[1].dtype, torch.int64)
                    for tensor in dispatch_args[:3]:
                        self.assertTrue(tensor.is_contiguous())
                    for actual, expected in zip(
                        dispatch_args[3:],
                        (buffer.x, buffer.x_sf, buffer.topk_idx, buffer.topk_weights),
                        strict=True,
                    ):
                        self.assertIs(actual, expected)
                    self.assertEqual(
                        deep_gemm.mega_moe_pre_dispatch.call_args.kwargs,
                        {
                            "num_tokens": num_tokens,
                            "group_size": 32,
                            "mma_type": mma_type,
                        },
                    )
                    triton_dispatch.assert_not_called()
                    kernel = {
                        "mxf4xmxf4": deep_gemm.fp4_fp4_mega_moe,
                        "fp8xmxfp4": deep_gemm.fp8_fp4_mega_moe,
                        "bf16xmxfp4": deep_gemm.bf16_fp4_mega_moe,
                    }[mma_type]
                    kernel.assert_called_once()
                    self.assertEqual(kernel.call_args.args[0].shape, (num_tokens, 4))
                    if mma_type != "mxf4xmxf4":
                        self.assertNotIn("recipe", kernel.call_args.kwargs)
                    self.assertEqual(kernel.call_args.kwargs["situ_beta"], 3.0)
                    self.assertEqual(
                        kernel.call_args.kwargs["situ_linear_beta"],
                        17.0,
                    )
                else:
                    if mma_type == "mxf4xmxf4":
                        deep_gemm.mega_moe_pre_dispatch.assert_called_once()
                        self.assertEqual(
                            deep_gemm.mega_moe_pre_dispatch.call_args.args[1].dtype,
                            torch.int32,
                        )
                        triton_dispatch.assert_not_called()
                    else:
                        triton_dispatch.assert_called_once()
                        self.assertEqual(
                            triton_dispatch.call_args.args[1].dtype, torch.int32
                        )
                        self.assertEqual(
                            triton_dispatch.call_args.kwargs["quant_group_size"], 32
                        )
                    deep_gemm.fp8_fp4_mega_moe.assert_called_once()
                    deep_gemm.fp4_fp4_mega_moe.assert_not_called()

    def test_kimi_ppu_precision_masks_padded_rows_and_scales_routes_once(self):
        from sglang.srt.models.kimi_k3 import KimiK3MoE

        routed_input = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0], [float("nan")] * 4],
            dtype=torch.bfloat16,
        )
        valid_token_count = torch.tensor(1)
        buffer = SimpleNamespace(
            x=object(), x_sf=object(), topk_idx=object(), topk_weights=object()
        )
        staged = {}

        def pre_dispatch(
            x,
            ids,
            weights,
            buf_x,
            buf_x_sf,
            buf_topk_idx,
            buf_topk_weights,
            num_tokens,
            group_size=32,
            mma_type="mxf4xmxf4",
        ):
            self.assertIs(buf_x, buffer.x)
            self.assertIs(buf_x_sf, buffer.x_sf)
            self.assertIs(buf_topk_idx, buffer.topk_idx)
            self.assertIs(buf_topk_weights, buffer.topk_weights)
            self.assertEqual(num_tokens, 3)
            self.assertEqual(group_size, 32)
            self.assertEqual(mma_type, expected_mma_type)
            self.assertEqual(ids.dtype, torch.int64)
            self.assertTrue(torch.isfinite(x).all())
            valid = int(valid_token_count)
            self.assertTrue(torch.all(ids[valid:] == -1))
            self.assertTrue(torch.all(weights[valid:] == 0))
            staged.update(x=x, weights=weights)

        def kernel(y, l1, l2, buf, **kwargs):
            y.copy_(staged["x"] * staged["weights"].sum(-1, keepdim=True))
            y[int(valid_token_count) :].fill_(float("nan"))

        for mma_type in ("mxf4xmxf4", "fp8xmxfp4", "bf16xmxfp4"):
            expected_mma_type = mma_type
            for scale_in_topk in (False, True):
                with self.subTest(mma_type=mma_type, scale_in_topk=scale_in_topk):
                    experts = SimpleNamespace(
                        mega_l1_weights=object(),
                        mega_l2_weights=object(),
                        moe_runner_config=SimpleNamespace(
                            gemm1_alpha=4.0, gemm1_clamp_limit=25.0
                        ),
                        should_fuse_routed_scaling_factor_in_topk=scale_in_topk,
                    )
                    owner = SimpleNamespace(
                        experts=experts,
                        _mega_top_k=2,
                        moe_hidden_size=4,
                        _mega_moe_buffer=lambda: buffer,
                        routed_scaling_factor=2.0,
                    )
                    topk = SimpleNamespace(
                        topk_ids=torch.tensor([[0, 1]] * 3, dtype=torch.int32),
                        topk_weights=torch.tensor([[0.25, 0.75]] * 3)
                        * (2.0 if scale_in_topk else 1.0),
                    )
                    deep_gemm = ModuleType("deep_gemm")
                    deep_gemm.mega_moe_pre_dispatch = pre_dispatch
                    deep_gemm.fp4_fp4_mega_moe = kernel
                    deep_gemm.fp8_fp4_mega_moe = kernel
                    deep_gemm.bf16_fp4_mega_moe = kernel
                    with (
                        patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
                        patch("sglang.srt.models.kimi_k3.is_sail", return_value=True),
                        patch.object(
                            mega_moe, "_mega_moe_mma_type", return_value=mma_type
                        ),
                        patch.object(
                            mega_moe,
                            "_configure_mega_moe_deep_gemm_num_sms",
                            return_value=nullcontext(),
                        ),
                    ):
                        for valid in (1, 0, 2):
                            valid_token_count.fill_(valid)
                            actual = KimiK3MoE._forward_mega_experts(
                                owner,
                                routed_input,
                                topk,
                                valid_token_count=valid_token_count,
                            )
                            expected = torch.zeros_like(routed_input)
                            expected[:valid] = routed_input[:valid] * 2
                            torch.testing.assert_close(actual, expected)

    def test_mxf4_l1_uses_packed_gate_up_interleave(self):
        source = torch.arange(32).reshape(1, 32)
        expected = torch.tensor(
            [
                0,
                2,
                4,
                6,
                8,
                10,
                12,
                14,
                16,
                18,
                20,
                22,
                24,
                26,
                28,
                30,
                1,
                3,
                5,
                7,
                9,
                11,
                13,
                15,
                17,
                19,
                21,
                23,
                25,
                27,
                29,
                31,
            ]
        ).reshape(1, 32)

        actual = mega_moe._interleave_mega_moe_gate_up(source, gran=16)

        torch.testing.assert_close(actual, expected)

    def test_draft_layers_keep_their_own_w4a4_choice(self):
        """Draft MegaMoE layers must not follow the target's MMA type once built.

        The draft forward runs outside draft_model_build_scope, so a layer that
        re-read the global flag would pair draft weights with the wrong kernel.
        """
        for target_flag, draft_flag, expected_target, expected_draft in (
            (True, False, "mxf4xmxf4", "fp8xfp4"),
            (True, None, "mxf4xmxf4", "mxf4xmxf4"),
            (False, True, "fp8xfp4", "mxf4xmxf4"),
            (False, None, "fp8xfp4", "fp8xfp4"),
        ):
            with self.subTest(
                enable_w4a4_mxfp4_megamoe=target_flag,
                speculative_enable_w4a4_mxfp4_megamoe=draft_flag,
            ):
                with get_context().override_server_args(
                    model_path="dummy",
                    enable_w4a4_mxfp4_megamoe=target_flag,
                    speculative_enable_w4a4_mxfp4_megamoe=draft_flag,
                ):
                    target = self._build_fused_moe()
                    with draft_model_build_scope():
                        draft = self._build_fused_moe()

                    self.assertEqual(
                        get_exec().moe.enable_w4a4_mxfp4_megamoe, target_flag
                    )
                    self.assertEqual(
                        mega_moe._mega_moe_mma_type(target), expected_target
                    )
                    self.assertEqual(mega_moe._mega_moe_mma_type(draft), expected_draft)

    def test_draft_build_scope_restores_w4a4_on_exception(self):
        """A failed draft build must not leave the target on the draft's MMA type."""
        with get_context().override_server_args(
            model_path="dummy",
            enable_w4a4_mxfp4_megamoe=False,
            speculative_enable_w4a4_mxfp4_megamoe=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "draft build failed"):
                with draft_model_build_scope():
                    self.assertTrue(get_exec().moe.enable_w4a4_mxfp4_megamoe)
                    raise RuntimeError("draft build failed")

            self.assertFalse(get_exec().moe.enable_w4a4_mxfp4_megamoe)
            target = self._build_fused_moe()
            self.assertEqual(mega_moe._mega_moe_mma_type(target), "fp8xfp4")

    def _build_fused_moe(self):
        method = UnquantizedFusedMoEMethod()
        with (
            patch.object(method, "create_weights"),
            patch.object(method, "create_moe_runner"),
            patch.object(
                fused_moe_layer_module,
                "create_moe_dispatcher",
                return_value=SimpleNamespace(),
            ),
            get_flags().moe.override(
                runner_backend=MoeRunnerBackend.AUTO,
                a2a_backend=MoeA2ABackend.MEGAMOE,
            ),
            get_parallel().override(
                moe_ep_size=1,
                moe_ep_rank=0,
                moe_tp_size=1,
                moe_tp_rank=0,
                tp_size=1,
                tp_rank=0,
            ),
        ):
            return fused_moe_layer_module.FusedMoE(
                num_experts=2,
                hidden_size=4,
                intermediate_size=8,
                layer_id=0,
                quant_method=method,
            )

    def _get_test_buffer(self, group):
        with patch.dict(sys.modules, {"deep_gemm": self.deep_gemm}):
            return mega_moe._get_mega_moe_symm_buffer(
                group,
                num_experts=8,
                num_max_tokens_per_rank=64,
                num_topk=2,
                hidden=128,
                intermediate_hidden=256,
                mma_type="fp8xfp4",
            )


if __name__ == "__main__":
    unittest.main()
