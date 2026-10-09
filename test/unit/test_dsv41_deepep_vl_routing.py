"""CPU-only routing regressions; execute model methods without GPU imports."""

import __future__

import ast
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


class TestDeepEPVisionRouting(unittest.TestCase):
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

    def setUp(self):
        self.recorder = Mock()
        self.recorder.with_current_layer.return_value = nullcontext()
        self.mapping = Mock()
        self.env = {
            "vision_topk": Mock(
                return_value=SimpleNamespace(topk_ids="logical", topk_weights="weights")
            ),
            "_post_process_topk_ids": Mock(
                return_value=("physical", "masked_weights", "recorder_ids")
            ),
            "ExpertLocationDispatchInfo": SimpleNamespace(init_new=self.mapping),
            "get_global_expert_distribution_recorder": lambda: self.recorder,
            "StandardTopKOutput": lambda w, i, r: SimpleNamespace(
                topk_weights=w, topk_ids=i, router_logits=r
            ),
        }
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
        self.topk = Mock(return_value="normal_output", topk_config="config")
        self.moe = SimpleNamespace(
            gate=SimpleNamespace(e_score_correction_bias_vl="vl_bias"),
            is_nextn=False,
            is_hash=False,
            layer_id=7,
            topk=self.topk,
        )
        self.moe._select_deepep_topk = lambda *args: self.env["_select_deepep_topk"](
            self.moe, *args
        )

    def test_visual_routing_maps_experts_with_same_padding_bound(self):
        out = self.moe._select_deepep_topk(
            "hidden", "logits", "mixed_image_ids", "count"
        )
        self.env["vision_topk"].assert_called_once_with(
            self.moe, "logits", "mixed_image_ids", num_token_non_padded="count"
        )
        post = self.env["_post_process_topk_ids"].call_args.kwargs
        self.assertEqual(post["topk_ids"], "logical")
        self.assertEqual(post["num_token_non_padded"], "count")
        self.assertIs(post["expert_location_dispatch_info"], self.mapping.return_value)
        self.assertEqual(
            (out.topk_ids, out.topk_weights), ("physical", "masked_weights")
        )
        self.recorder.on_select_experts.assert_called_once_with(topk_ids="recorder_ids")
        self.topk.assert_not_called()

    def test_normal_router_keeps_dispatch_and_padding(self):
        self.moe.gate.e_score_correction_bias_vl = None
        self.assertEqual(
            self.moe._select_deepep_topk("hidden", "logits", "ids", "count"),
            "normal_output",
        )
        self.topk.assert_called_once_with(
            hidden_states="hidden",
            router_logits="logits",
            num_token_non_padded="count",
            expert_location_dispatch_info=self.mapping.return_value,
        )
        self.env["vision_topk"].assert_not_called()

    def test_hash_router_keeps_token_ids(self):
        self.moe.gate.e_score_correction_bias_vl = None
        self.moe.is_hash = True
        self.moe._select_deepep_topk("hidden", "logits", "hash_ids", None)
        self.assertEqual(self.topk.call_args.kwargs["input_ids"], "hash_ids")

    def test_nextn_and_absent_recorder_ids(self):
        self.moe.is_nextn = True
        self.env["_post_process_topk_ids"].return_value = ("ids", "weights", None)
        self.moe._select_deepep_topk("hidden", "logits", "ids", None)
        self.mapping.assert_not_called()
        self.assertIsNone(
            self.env["_post_process_topk_ids"].call_args.kwargs[
                "expert_location_dispatch_info"
            ]
        )
        self.recorder.on_select_experts.assert_not_called()

    def test_overlap_uses_local_ids_and_moe_padding_getter(self):
        fb = SimpleNamespace(
            input_ids="local_image_ids",
            moe_num_token_non_padded=Mock(return_value="local_count"),
        )
        state = SimpleNamespace(
            pop=lambda key: "logits", hidden_states_mlp_input="hidden", forward_batch=fb
        )
        self.env["op_select_experts"](self.moe, state)
        self.env["vision_topk"].assert_called_once_with(
            self.moe, "logits", "local_image_ids", num_token_non_padded="local_count"
        )
        fb.moe_num_token_non_padded.assert_called_once_with()
        self.assertEqual(state.topk_output.topk_ids, "physical")

    def test_overlap_empty_input_uses_empty_topk(self):
        hidden = SimpleNamespace(device="device")
        state = SimpleNamespace(pop=lambda key: None, hidden_states_mlp_input=hidden)
        self.env["op_select_experts"](self.moe, state)
        self.topk.empty_topk_output.assert_called_once_with("device", layer_id=7)
        self.env["vision_topk"].assert_not_called()


if __name__ == "__main__":
    unittest.main()
