"""CPU-only contract tests for ``--dcp-kv-layout`` resolution."""

import argparse
import os
import tempfile
import unittest
from unittest.mock import patch

from transformers import LlamaConfig

from sglang.srt.arg_groups.overrides import declare_resolution
from sglang.srt.arg_groups.pd_disaggregation_hook import validate_dcp_kv_layout
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _page_args(**overrides) -> ServerArgs:
    values = {
        "model_path": "dummy",
        "dcp_kv_layout": "page",
        "disaggregation_mode": "decode",
        "disaggregation_transfer_backend": "mooncake",
        "dcp_size": 2,
        "decode_attention_backend": "cutedsl_mla",
    }
    values.update(overrides)
    return ServerArgs(**values)


class TestDcpKvLayout(CustomTestCase):
    def test_defaults_to_token_and_is_exposed_by_cli(self):
        self.assertEqual(ServerArgs(model_path="dummy").dcp_kv_layout, "token")

        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        parsed = parser.parse_args(["--model-path", "dummy", "--dcp-kv-layout", "page"])
        self.assertEqual(parsed.dcp_kv_layout, "page")

    def test_token_keeps_existing_config_combinations(self):
        for overrides in (
            {},
            {"disaggregation_mode": "null", "enable_unified_memory": True},
            {"disaggregation_mode": "prefill", "dcp_size": 1},
            {"disaggregation_mode": "prefill", "dcp_size": 2},
        ):
            with self.subTest(overrides=overrides):
                validate_dcp_kv_layout(ServerArgs(model_path="dummy", **overrides))

    def test_token_publishes_through_parallel_context(self):
        with get_context().override_server_args():
            self.assertEqual(get_parallel().dcp_kv_layout, "token")

    def test_page_accepts_final_supported_contract(self):
        validate_dcp_kv_layout(_page_args())

    def test_page_reads_the_resolved_decode_backend(self):
        server_args = _page_args(decode_attention_backend="aiter")
        declare_resolution(server_args, "test", decode_attention_backend="cutedsl_mla")

        validate_dcp_kv_layout(server_args)

    def test_page_rejects_non_decode_execution_paths(self):
        """The page option is decode-only, including when prefill uses DCP1."""
        for overrides, message in (
            (
                {"disaggregation_mode": "null", "enable_unified_memory": True},
                "PD decode",
            ),
            ({"disaggregation_mode": "prefill", "dcp_size": 1}, "dcp-size > 1"),
            ({"disaggregation_mode": "prefill", "dcp_size": 2}, "PD decode"),
        ):
            with (
                self.subTest(overrides=overrides),
                self.assertRaisesRegex(ValueError, message),
            ):
                validate_dcp_kv_layout(_page_args(**overrides))

    def test_resolution_rejects_page_on_prefill(self):
        """Resolution accepts token layout but rejects page layout on DCP1 prefill."""
        with tempfile.TemporaryDirectory() as model_path:
            LlamaConfig(
                architectures=["LlamaForCausalLM"],
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=2,
                vocab_size=128,
                max_position_embeddings=2048,
            ).save_pretrained(model_path)
            for layout in ("token", "page"):
                with (
                    self.subTest(layout=layout),
                    patch.dict(os.environ, os.environ.copy()),
                ):
                    args = ServerArgs(
                        model_path=model_path,
                        device="cuda",
                        disaggregation_mode="prefill",
                        disaggregation_transfer_backend="mooncake",
                        dcp_size=1,
                        dcp_kv_layout=layout,
                        random_seed=42,
                    )
                    if layout == "page":
                        with self.assertRaisesRegex(ValueError, "dcp-size > 1"):
                            args.resolve_once()
                    else:
                        args.resolve_once()

    def test_page_rejects_unsupported_static_combinations(self):
        cases = (
            ({"dcp_size": 1}, "dcp-size > 1"),
            ({"disaggregation_transfer_backend": "nixl"}, "backend mooncake"),
            ({"speculative_algorithm": "EAGLE"}, "speculative decoding"),
            ({"decode_attention_backend": "aiter"}, "'cutedsl_mla'"),
        )
        for overrides, message in cases:
            with (
                self.subTest(overrides=overrides),
                self.assertRaisesRegex(ValueError, message),
            ):
                validate_dcp_kv_layout(_page_args(**overrides))


if __name__ == "__main__":
    unittest.main()
