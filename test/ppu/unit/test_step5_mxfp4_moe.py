"""Step 5 MXFP4 routing, scale layout, and post-SiLU clamp on SAIL."""

import unittest

import torch

from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.layers.quantization.mxfp4 import Mxfp4Config
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.runtime_context import get_context, get_flags, get_platform
from sglang.test.layer_ut_utils import init_single_process_dist
from sglang.test.test_utils import CustomTestCase


@unittest.skipUnless(
    torch.cuda.is_available() and get_platform().is_sail, "requires SAIL"
)
class TestStep5Mxfp4Moe(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        init_single_process_dist()

    @classmethod
    def tearDownClass(cls):
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    def test_checkpoint_layout_and_post_silu_clamp(self):
        with (
            get_context().override_server_args(model_path="dummy"),
            get_flags().moe.override(runner_backend=MoeRunnerBackend.DEEP_GEMM),
            torch.device("cuda"),
            torch.inference_mode(),
        ):
            model = torch.nn.Module()
            model.experts = FusedMoE(
                num_experts=2,
                hidden_size=128,
                intermediate_size=128,
                layer_id=3,
                top_k=2,
                params_dtype=torch.bfloat16,
                quant_config=Mxfp4Config(is_checkpoint_mxfp4_serialized=True),
                prefix="model.layers.3.moe.experts",
                inplace=False,
                gate_up_interleaved=False,
                gemm1_clamp_limit=7.0,
                silu_clamp_mode="post",
                uses_expert_location=False,
            )
            experts = model.experts
            experts.w13_weight.fill_(0x22)  # Two packed 1.0 values.
            experts.w2_weight[0].fill_(0x22)
            experts.w2_weight[1].fill_(0x44)  # Two packed 2.0 values.
            experts.w13_weight_scale.fill_(127)
            experts.w2_weight_scale.fill_(127)
            DefaultModelLoader.postprocess_weights(model, torch.device("cuda"))
            self.assertEqual(experts.w13_weight.dtype, torch.uint8)
            self.assertEqual(experts.w13_weight_scale.dtype, torch.uint16)
            self.assertEqual(experts.w13_weight_scale.stride(1), 1)

            topk = StandardTopKOutput(
                topk_weights=torch.tensor([[0.25, 0.75], [0.5, 0.5], [0.5, 0.5]]),
                topk_ids=torch.tensor([[0, 1], [-1, 1], [-1, -1]], dtype=torch.int32),
                router_logits=torch.empty((3, 2)),
            )
            inputs = torch.ones((3, 128), dtype=torch.bfloat16)
            # Each product clamps to 7 * 7 and rounds to 48 in MXFP4.
            # The two expert down projections therefore produce 6144 and 12288.
            expected = torch.tensor([10752, 6144, 0], dtype=torch.bfloat16)
            torch.testing.assert_close(
                experts(inputs, topk),
                expected[:, None].expand_as(inputs),
                rtol=0,
                atol=0,
            )


if __name__ == "__main__":
    unittest.main()
