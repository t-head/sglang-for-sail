"""DeepEP vision routing regressions without GPU imports from the model."""

import __future__
import ast
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs
from sglang.srt.layers.moe import MoeA2ABackend
from sglang.srt.layers.moe import topk as topk_module
from sglang.srt.layers.moe.topk import TopKConfig
from sglang.srt.multimodal.dsv41 import vl_routing
from sglang.srt.runtime_context import get_flags, get_parallel

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestDeepEPVisionRouting(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        root = next(
            p
            for p in Path(__file__).resolve().parents
            if (p / "python/sglang/srt/models/deepseek_v2.py").exists()
        )
        cls.source = root / "python/sglang/srt/models/deepseek_v2.py"
        tree = ast.parse(cls.source.read_text())
        model = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "DeepseekV2MoE"
        )
        cls.methods = {n.name: n for n in model.body if isinstance(n, ast.FunctionDef)}
        v4_source = root / "python/sglang/srt/models/deepseek_v4.py"
        v4_tree = ast.parse(v4_source.read_text())
        v4_model = next(
            n
            for n in v4_tree.body
            if isinstance(n, ast.ClassDef) and n.name == "DeepseekV4ForCausalLM"
        )
        init = next(
            n
            for n in v4_model.body
            if isinstance(n, ast.FunctionDef) and n.name == "__init__"
        )
        vision = next(
            n
            for n in init.body
            if isinstance(n, ast.If) and "config.vision_n_layers" in ast.unparse(n.test)
        )
        cls.vision_guard = compile(
            ast.Module(body=[vision.body[0]], type_ignores=[]), str(v4_source), "exec"
        )

    def setUp(self):
        super().setUp()
        self.recorder = Mock()
        self.recorder.with_current_layer.return_value = nullcontext()
        self.mapping = Mock(return_value=None)
        self.env = {
            "envs": envs,
            "vision_topk": vl_routing.vision_topk,
            "ExpertLocationDispatchInfo": SimpleNamespace(init_new=self.mapping),
            "get_global_expert_distribution_recorder": lambda: self.recorder,
        }
        # Execute the actual model methods without importing the GPU-heavy model.
        # Routing, shared-slot remapping and padding use the real implementations.
        for name in ("_select_deepep_topk", "op_select_experts"):
            module = ast.Module(body=[self.methods[name]], type_ignores=[])
            exec(
                compile(
                    module,
                    str(self.source),
                    "exec",
                    flags=__future__.annotations.compiler_flag,
                ),
                self.env,
            )
        for cm in (
            get_flags().moe.override(a2a_backend=MoeA2ABackend.DEEPEP),
            get_parallel().override(moe_ep_size=2, moe_ep_rank=1),
            patch.object(vl_routing, "is_cuda", return_value=False),
            patch.object(vl_routing, "is_hip", return_value=False),
            envs.SGLANG_SIMULATE_UNIFORM_EXPERTS.override(False),
            envs.SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS.override(False),
            patch.object(topk_module, "_is_cuda", False),
            patch.object(topk_module, "_is_hip", False),
            patch.object(topk_module, "_use_aiter", False),
            patch.object(topk_module, "_can_fuse_padded_region", return_value=False),
            patch.object(
                vl_routing,
                "get_global_expert_distribution_recorder",
                return_value=self.recorder,
            ),
        ):
            cm.__enter__()
            self.addCleanup(cm.__exit__, None, None, None)

    def _moe(self, fused_shared=0, scale_in_topk=False):
        moe = SimpleNamespace(
            config=SimpleNamespace(image_token_id=99),
            gate=SimpleNamespace(
                e_score_correction_bias=torch.tensor([2.0, 2.0, 0.0, 0.0]),
                e_score_correction_bias_vl=torch.tensor([0.0, 0.0, 2.0, 2.0]),
            ),
            is_nextn=False,
            is_hash=False,
            layer_id=7,
            topk=Mock(
                return_value="normal_output",
                topk_config=TopKConfig(
                    top_k=2 + fused_shared,
                    num_fused_shared_experts=fused_shared,
                    renormalize=True,
                    routed_scaling_factor=2.0,
                    apply_routed_scaling_factor_on_output=scale_in_topk,
                    allow_routed_experts_capture=False,
                ),
            ),
        )
        moe._select_deepep_topk = lambda *args: self.env["_select_deepep_topk"](
            moe, *args
        )
        return moe

    def test_mixed_modalities_shared_slots_scaling_and_padding(self):
        """Image bias must change ranking, not weights, even with fused sharing.

        DeepEP used to reject fusion before dispatch. Removing only that guard
        would leave shared IDs, shared contribution or padded rows incorrect.
        """
        logits = torch.tensor([[0.8, 0.7, 0.6, 0.5]]).repeat(3, 1)
        ids = torch.tensor([7, 99, 99])
        scores = F.softplus(logits[:2]).sqrt()
        expected_weights = torch.stack((scores[0, :2], scores[1, 2:]))
        expected_weights /= (
            expected_weights.sum(-1, keepdim=True)
            + topk_module._RENORMALIZE_SUM_EPSILON
        )
        for fused_shared in (0, 1):
            for scale_in_topk in (False, True):
                for overlap in (False, True):
                    with self.subTest(
                        fused_shared=fused_shared,
                        scale_in_topk=scale_in_topk,
                        overlap=overlap,
                    ):
                        moe = self._moe(fused_shared, scale_in_topk)
                        count = torch.tensor(2, dtype=torch.int32)
                        if overlap:
                            state = SimpleNamespace(
                                pop=lambda key: logits,
                                hidden_states_mlp_input=torch.ones((3, 4)),
                                forward_batch=SimpleNamespace(
                                    input_ids=ids,
                                    moe_num_token_non_padded=lambda: count,
                                ),
                            )
                            self.env["op_select_experts"](moe, state)
                            out = state.topk_output
                        else:
                            out = moe._select_deepep_topk(
                                torch.ones((3, 4)), logits, ids, count
                            )
                        expected_ids = torch.tensor(
                            [[0, 1], [2 + fused_shared, 3 + fused_shared]],
                            dtype=torch.int32,
                        )
                        torch.testing.assert_close(out.topk_ids[:2, :2], expected_ids)
                        scale = 2.0 if scale_in_topk else 1.0
                        torch.testing.assert_close(
                            out.topk_weights[:2, :2], expected_weights * scale
                        )
                        if fused_shared:
                            # EP=2, rank=1: [0,1,shared=2,3,4,shared=5].
                            self.assertTrue((out.topk_ids[:2, -1] == 5).all())
                            post_scale = 1.0 if scale_in_topk else 2.0
                            torch.testing.assert_close(
                                out.topk_weights[:2, -1] * post_scale, torch.ones(2)
                            )
                        self.assertTrue((out.topk_ids[2] == -1).all())
                        self.assertTrue((out.topk_weights[2] == 0).all())
                        moe.topk.assert_not_called()

    def test_cuda_masked_rows_stay_empty_after_shared_remap(self):
        """A router-masked row must not regain a shared expert in postprocessing."""
        moe = self._moe(fused_shared=1)
        weights, indices = vl_routing._finish_vision_topk(
            moe,
            torch.zeros((2, 4)),
            torch.tensor([[0.5, 0.5, 0.5], [0.0, 0.0, 0.0]]),
            torch.tensor([[0, 1, 4], [-1, -1, -1]], dtype=torch.int32),
            torch.tensor(1, dtype=torch.int32),
            None,
            padded_rows_masked=True,
        )
        torch.testing.assert_close(
            indices[0], torch.tensor([0, 1, 5], dtype=torch.int32)
        )
        self.assertTrue((indices[1] == -1).all())
        self.assertTrue((weights[1] == 0).all())

    def test_dispatch_info_and_recording_are_forwarded_once(self):
        """Dropping dispatch info would route redundant shared slots incorrectly."""
        moe = self._moe(fused_shared=1)
        self.mapping.return_value = SimpleNamespace(
            num_physical_experts=8, ep_dispatch_algorithm="static"
        )
        out = moe._select_deepep_topk(
            torch.ones((1, 4)), torch.zeros((1, 4)), torch.tensor([99]), None
        )
        # With eight physical routed experts, rank 1's shared slot is 9, not 5.
        self.assertEqual(out.topk_ids[0, -1].item(), 9)
        self.recorder.on_select_experts.assert_called_once()

    def test_nextn_vision_routing_does_not_request_eplb_mapping(self):
        moe = self._moe()
        moe.is_nextn = True
        out = moe._select_deepep_topk(
            torch.ones((1, 4)), torch.zeros((1, 4)), torch.tensor([99]), None
        )
        self.assertEqual(set(out.topk_ids[0].tolist()), {2, 3})
        self.mapping.assert_not_called()

    def test_normal_and_hash_router_keep_dispatch_padding_and_token_ids(self):
        for is_hash in (False, True):
            with self.subTest(is_hash=is_hash):
                moe = self._moe()
                moe.gate.e_score_correction_bias_vl = None
                moe.is_hash = is_hash
                out = moe._select_deepep_topk("hidden", "logits", "ids", "count")
                self.assertEqual(out, "normal_output")
                kwargs = dict(
                    hidden_states="hidden",
                    router_logits="logits",
                    num_token_non_padded="count",
                    expert_location_dispatch_info=None,
                )
                if is_hash:
                    kwargs["input_ids"] = "ids"
                moe.topk.assert_called_once_with(**kwargs)

    def test_overlap_empty_input_uses_empty_topk(self):
        moe = self._moe()
        state = SimpleNamespace(
            pop=lambda key: None,
            hidden_states_mlp_input=SimpleNamespace(device="cpu"),
        )
        self.env["op_select_experts"](moe, state)
        self.assertIs(state.topk_output, moe.topk.empty_topk_output.return_value)
        moe.topk.empty_topk_output.assert_called_once_with("cpu", layer_id=7)

    def test_hip_padding_keeps_safe_ids_and_zero_weights(self):
        for fused_shared in (0, 1):
            with self.subTest(fused_shared=fused_shared), patch.object(
                vl_routing, "is_hip", return_value=True
            ), patch.object(topk_module, "_is_hip", True), patch.object(
                topk_module, "_eplb_remap_enabled", return_value=False
            ), patch.object(topk_module, "_skip_hip_pad_mask", False):
                moe = self._moe(fused_shared)
                out = moe._select_deepep_topk(
                    torch.ones((2, 4)),
                    torch.zeros((2, 4)),
                    torch.tensor([99, 99]),
                    torch.tensor(1, dtype=torch.int32),
                )
                self.assertTrue((out.topk_ids[1] == 0).all())
                self.assertTrue((out.topk_weights[1] == 0).all())
                self.assertTrue((out.topk_weights[0] > 0).all())

    def test_synthetic_routing_uses_topk_controls(self):
        # Only the GPU simulation kernel is replaced; input transforms,
        # override selection and dispatch postprocessing are real.
        def cpu_route(hidden_states, gating_output, topk, renormalize):
            weights, ids = gating_output.softmax(-1).topk(topk, dim=-1)
            return weights / weights.sum(-1, keepdim=True), ids.int()

        def simulate(ids, weights, num_experts, **kwargs):
            ids[:] = torch.tensor([0, 2], dtype=ids.dtype)
            weights.fill_(0.5)

        for mode in ("fake", "uniform", "round_robin", "both"):
            with self.subTest(mode=mode), envs.SGLANG_SIMULATE_UNIFORM_EXPERTS.override(
                mode in ("uniform", "both")
            ), envs.SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS.override(
                mode in ("round_robin", "both")
            ), patch.object(
                topk_module, "_simulate_balanced_routing", side_effect=simulate
            ), patch.object(
                topk_module,
                "get_global_expert_distribution_recorder",
                return_value=self.recorder,
            ), patch.object(
                topk_module, "is_moe_input_scattered_across_dp_ranks", return_value=False
            ):
                moe = self._moe()
                config = moe.topk.topk_config
                config.custom_routing_function = cpu_route
                if mode == "fake":
                    self.mapping.return_value = SimpleNamespace(ep_dispatch_algorithm="fake")
                else:
                    self.mapping.return_value = None
                moe.topk = lambda **kwargs: topk_module.select_experts(
                    topk_config=config, layer_id=moe.layer_id, **kwargs
                )
                logits = torch.tensor([[0.8, 0.7, 0.6, 0.5]])
                if mode == "both":
                    with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                        moe._select_deepep_topk(
                            torch.ones((1, 4)), logits, torch.tensor([99]), None
                        )
                    continue
                out = moe._select_deepep_topk(
                    torch.ones((1, 4)), logits, torch.tensor([99]), None
                )
                if mode == "fake":
                    self.assertTrue(((logits >= 5) & (logits <= 10)).all())
                    expected_weights, expected_ids = cpu_route(None, logits, 2, True)
                    torch.testing.assert_close(out.topk_ids, expected_ids)
                    torch.testing.assert_close(out.topk_weights, expected_weights)
                else:
                    torch.testing.assert_close(
                        out.topk_ids, torch.tensor([[0, 2]], dtype=torch.int32)
                    )
                    torch.testing.assert_close(
                        out.topk_weights, torch.full((1, 2), 0.5)
                    )

    def test_vision_backend_guard_preserves_parallel_restrictions(self):
        for backend in MoeA2ABackend:
            for cuda in (False, True):
                for cp, pp in ((1, 1), (2, 1), (1, 2)):
                    with self.subTest(backend=backend, cuda=cuda, cp=cp, pp=pp):
                        allowed = cp == pp == 1 and (
                            backend.is_none()
                            or (
                                cuda
                                and backend in (
                                    MoeA2ABackend.MEGAMOE,
                                    MoeA2ABackend.DEEPEP,
                                    MoeA2ABackend.DEEPEP_V2,
                                )
                            )
                        )
                        scope = {
                            "_is_cuda": cuda,
                            "get_moe_a2a_backend": lambda: backend,
                            "get_parallel": lambda: SimpleNamespace(
                                attn_cp_size=cp,
                                pp_group=SimpleNamespace(world_size=pp),
                            ),
                        }
                        if allowed:
                            exec(self.vision_guard, scope)
                        else:
                            with self.assertRaisesRegex(ValueError, "V4.1 vision"):
                                exec(self.vision_guard, scope)


if __name__ == "__main__":
    unittest.main()
