# SPDX-License-Identifier: Apache-2.0

import argparse
import contextlib
import copy
import io
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import yaml
from ppu_recipe import (
    DEFAULT_AUTHORING_DOC,
    DEFAULT_MATRIX_DOC,
    DEFAULT_POLICY_ROOT,
    REPO_ROOT,
    ValidationError,
    _load_yaml,
    _scan_text,
    iter_recipe_catalog_entries,
)
from ppu_recipe import main as ppu_recipe_main
from ppu_recipe import (
    render_ci_matrix_markdown,
    resolve_recipe,
    substitute_validation_placeholders,
    validate_flags_with_parser,
    validate_paths,
    validate_policy_structure,
    validate_serving_recipe_structure,
)
from recipe_bundle import sha256, wheel_inventory
from recipe_cluster_config import BACKEND_SHA, CONFIG_NAME, load_settings
from recipe_cluster_diagnostics import diagnostic_codes, recent_failures
from recipe_publish import publish, validate
from recipe_report import legacy_result
from recipe_report import main as report_recipe
from recipe_retention import retain
from recipe_runner import (
    cached_model_path,
    metrics_zero,
    probes,
    read_eval_reviews,
    run,
)
from recipe_select import cluster_width, select


def fast_result():
    return {
        "schema": "ppu-sanity/v1",
        "model": "glm-5.3",
        "tier": "fast_sanity",
        "source_sha": "a" * 40,
        "recipe_tree_sha256": "0" * 64,
        "config_digest": "c" * 64,
        "outcome": "pass",
        "runtime_image": "@sha256:" + "d" * 64,
        "route": "native-build + sglang-release-runtime",
        "parser_checked": True,
        "sse_verified": True,
        "private_retention": True,
        "queue_s": 2,
        "cold_start_s": 3.0,
        "layer_override": None,
    }


class RecipeContractTest(unittest.TestCase):
    def test_fast_probe_accepts_reasoning_only_completion_with_complete_sse(self):
        sampling = {"temperature": 1.0, "top_p": 1.0, "top_k": -1}
        settings = {"prompt": "Compute 6 * 7.", "max_tokens": 16, **sampling}
        answer = {
            "message": {"content": None, "reasoning_content": "reasoning text"},
            "finish_reason": "length",
        }
        stream = io.BytesIO(
            b'data: {"choices":[{"delta":{"reasoning_content":"R"},"finish_reason":null}]}\n\n'
            b'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\n'
            b"data: [DONE]\n\n"
        )
        stream.headers = Mock()
        stream.headers.get_content_type.return_value = "text/event-stream"
        with tempfile.TemporaryDirectory() as temp:
            results = Path(temp) / "results"
            results.mkdir()
            with patch("recipe_runner.RESULTS", results):
                response = io.BytesIO(json.dumps({"choices": [answer]}).encode())
                with patch(
                    "recipe_runner.request", side_effect=(response, stream)
                ) as request:
                    probes(settings, fast=True)
                for call in request.call_args_list:
                    payload = json.loads(call.args[1])
                    self.assertEqual({key: payload[key] for key in sampling}, sampling)
                self.assertEqual(
                    json.loads((results / "chat-probe-response.json").read_text()),
                    answer,
                )
                for fast, choice in (
                    (True, dict(answer, finish_reason=None)),
                    (True, dict(answer, message={"content": " "})),
                    (False, answer),
                ):
                    with self.subTest(fast=fast, choice=choice):
                        response = io.BytesIO(
                            json.dumps({"choices": [choice]}).encode()
                        )
                        with patch("recipe_runner.request", return_value=response):
                            with self.assertRaisesRegex(
                                ValueError, "non-streaming response"
                            ):
                                probes(settings, fast=fast)

    def test_default_fast_selects_all_models(self):
        self.assertEqual(
            select("fast_sanity", ""),
            {
                "include": [
                    {"model": "deepseek-v4-flash-tp8", "ppu_count": 8},
                    {"model": "deepseek-v4-flash", "ppu_count": 8},
                    {"model": "deepseek-v4-pro", "ppu_count": 8},
                    {"model": "deepseek-v4.1-flash", "ppu_count": 8},
                    {"model": "glm-5.3-flash", "ppu_count": 2},
                    {"model": "glm-5.3-tp4", "ppu_count": 4},
                    {"model": "glm-5.3", "ppu_count": 8},
                    {"model": "kimi-k2.6", "ppu_count": 8},
                    {"model": "minimax-m3-fp8", "ppu_count": 8},
                    {"model": "minimax-m3", "ppu_count": 8},
                    {"model": "qwen3.8-27b", "ppu_count": 2},
                    {"model": "qwen3.8-flash-next-tp4", "ppu_count": 4},
                    {"model": "qwen3.8-flash-next", "ppu_count": 8},
                ]
            },
        )

    def test_nightly_exact_filter(self):
        self.assertEqual(
            select("model_sanity", "glm-5.3"),
            {"include": [{"model": "glm-5.3", "ppu_count": 8}]},
        )
        self.assertEqual(
            select("model_sanity", "deepseek-v4-flash-tp8"),
            {"include": [{"model": "deepseek-v4-flash-tp8", "ppu_count": 8}]},
        )
        with self.assertRaises(ValueError):
            select("model_sanity", "glm")
        with self.assertRaises(ValueError):
            select("model_sanity", "glm-5.3,glm-5.3")
        with self.assertRaises(ValueError):
            select("model_sanity", "qwen3.8-27b")

    def test_plan_rejects_width_mismatch_and_multiple_nodes(self):
        policy = DEFAULT_POLICY_ROOT / "models/qwen3.8-27b.yaml"
        for topology in (
            {"kind": "single_node", "nodes": 1, "devices_per_node": 8},
            {"kind": "multi_node_ep", "nodes": 2, "devices_per_node": 2},
        ):
            with (
                self.subTest(topology=topology),
                patch(
                    "recipe_select._load_yaml",
                    side_effect=[_load_yaml(policy), {"ppu": {"topology": topology}}],
                ),
            ):
                with self.assertRaisesRegex(ValueError, "unsupported cluster topology"):
                    cluster_width(policy, "fast_sanity")

    def test_glm_tp4_preserves_the_pinned_model_identity(self):
        recipes = REPO_ROOT / "test/ppu/recipes/zai-org"
        tp8 = resolve_recipe(recipes / "GLM-5.3.yaml")
        tp4 = resolve_recipe(recipes / "GLM-5.3-TP4.yaml")
        self.assertEqual(tp4["model_id"], tp8["model_id"])
        self.assertEqual(tp4["hf_model_id"], tp8["hf_model_id"])
        self.assertEqual(tp4["server_env"], tp8["server_env"])
        self.assertEqual((tp4["ppu_count"], tp8["ppu_count"]), (4, 8))

    def test_deepseek_v41_fast_uses_pure_tp8(self):
        model = "deepseek-v4.1-flash"
        policy = DEFAULT_POLICY_ROOT / f"models/{model}.yaml"
        recipe = REPO_ROOT / _load_yaml(policy)["recipe"]
        resolved = resolve_recipe(recipe, policy_path=policy, tier="fast_sanity")
        flags = resolved["flags"]
        self.assertEqual(resolved["ppu_count"], 8)
        self.assertEqual(flags[flags.index("--tp-size") + 1], "8")
        self.assertIsNone(resolved["layer_override"])
        for flag in (
            "--attn-dp-size",
            "--dp-size",
            "--enable-dp-lm-head",
            "--moe-a2a-backend",
            "--deepep-mode",
        ):
            self.assertNotIn(flag, flags)
        with self.assertRaises(ValueError):
            select("model_sanity", model)

    def test_minimax_fp8_selects_the_fp8_checkpoint(self):
        model = "minimax-m3-fp8"
        policy_path = DEFAULT_POLICY_ROOT / f"models/{model}.yaml"
        recipe = REPO_ROOT / _load_yaml(policy_path)["recipe"]
        config = resolve_recipe(recipe, policy_path=policy_path, tier="fast_sanity")
        self.assertEqual(
            config["hf_model_id"],
            "T-HEAD/MiniMax-M3-Quant-W-FP8-PerChannel-A-FP8-PerToken",
        )
        self.assertEqual(config["ppu_count"], 8)
        self.assertTrue(config["dummy_weights"])
        self.assertIsNone(config["layer_override"])

    def test_minimax_nightly_uses_published_quantized_reference(self):
        self.assert_nightly_reference("minimax-m3", width=8, correct=1284)
        self.assert_nightly_reference("minimax-m3-fp8", width=8, correct=1284)

    def test_deepseek_pro_can_run_full_gsm8k_against_published_reference(self):
        self.assert_nightly_reference("deepseek-v4-pro", width=8, correct=1285)

    def test_kimi_nightly_uses_published_tp8_reference(self):
        self.assert_nightly_reference("kimi-k2.6", width=8, correct=1291)

    def test_glm_flash_nightly_uses_tp2_and_full_gsm8k_reference(self):
        self.assert_nightly_reference("glm-5.3-flash", width=2, correct=1279)

    def test_glm_nightly_has_separate_tp4_reference(self):
        self.assert_nightly_reference("glm-5.3-tp4", width=4, correct=1282)

    def test_fast_sanity_policy_rejects_an_unknown_cache_value(self):
        policy = _load_yaml(DEFAULT_POLICY_ROOT / "models/qwen3.8-flash-next.yaml")
        policy["tiers"]["fast_sanity"]["cache"] = "auto"
        with self.assertRaisesRegex(ValidationError, r"fast_sanity\.cache"):
            validate_policy_structure(policy, context="test")

    def test_qwen_flash_next_fast_uses_the_eight_device_deepep_profile(self):
        policy = DEFAULT_POLICY_ROOT / "models/qwen3.8-flash-next.yaml"
        recipe = REPO_ROOT / _load_yaml(policy)["recipe"]
        resolved = resolve_recipe(recipe, policy_path=policy, tier="fast_sanity")
        self.assertEqual(resolved["ppu_count"], 8)
        self.assertIsNone(resolved["layer_override"])
        flags = resolved["flags"]
        for name, value in (
            ("--tp-size", "8"),
            ("--attn-dp-size", "8"),
            ("--attention-backend", "fa3"),
            ("--moe-a2a-backend", "deepep"),
            ("--speculative-algorithm", "NEXTN"),
            ("--speculative-num-draft-tokens", "3"),
            ("--dist-timeout", "3600"),
            ("--watchdog-timeout", "3600"),
            ("--load-format", "dummy"),
        ):
            self.assertEqual(flags[flags.index(name) + 1], value)
        self.assertEqual(
            resolved["server_env"],
            {
                "NCCL_CUMEM_ENABLE": "1",
                "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "512",
                "SGLANG_SAIL_DEEPEP_RECV_HOOK": "0",
            },
        )
        self.assertIn("--disable-radix-cache", flags)
        self.assertIn("--ple-offload-embedding", flags)
        self.assertNotIn("--dp-size", flags)
        self.assertNotIn("--page-size", flags)
        self.assertNotIn("--mamba-radix-cache-strategy", flags)

    def test_qwen_flash_next_dp8_nightly_keeps_nextn_and_enables_paged_cache(self):
        model = "qwen3.8-flash-next"
        self.assertEqual(
            select("model_sanity", model),
            {"include": [{"model": model, "ppu_count": 8}]},
        )
        policy = DEFAULT_POLICY_ROOT / f"models/{model}.yaml"
        config = resolve_recipe(
            REPO_ROOT / _load_yaml(policy)["recipe"],
            policy_path=policy,
            tier="model_sanity",
        )
        self.assertEqual(config["golden_model_key"], "qwen3.8-flash-next-dp8-ep")
        self.assertFalse(config["dummy_weights"])
        flags = config["flags"]
        for name, value in (
            ("--tp-size", "8"),
            ("--attn-dp-size", "8"),
            ("--moe-a2a-backend", "deepep"),
            ("--attention-backend", "fa3"),
            ("--linear-attn-prefill-backend", "triton"),
            ("--linear-attn-decode-backend", "triton"),
            ("--linear-attn-verify-backend", "triton"),
            ("--speculative-algorithm", "NEXTN"),
            ("--speculative-num-steps", "2"),
            ("--speculative-eagle-topk", "1"),
            ("--speculative-num-draft-tokens", "3"),
            ("--speculative-draft-model-quantization", "unquant"),
            ("--dist-timeout", "3600"),
            ("--watchdog-timeout", "3600"),
            ("--page-size", "64"),
            ("--mamba-radix-cache-strategy", "extra_buffer"),
        ):
            self.assertEqual(flags[flags.index(name) + 1], value)
        self.assertEqual(
            config["server_env"],
            {
                "NCCL_CUMEM_ENABLE": "1",
                "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "512",
                "SGLANG_SAIL_DEEPEP_RECV_HOOK": "0",
            },
        )
        self.assertIn("--enable-dp-lm-head", flags)
        for name in (
            "--disable-radix-cache",
            "--load-format",
            "--disable-overlap-schedule",
        ):
            self.assertNotIn(name, flags)
        stages = _load_yaml(policy)["tiers"]["model_sanity"]["stages"]
        stage = next(stage for stage in stages if stage["kind"] == "evalscope")
        self.assertIsNone(stage["limit"])
        self.assertEqual(stage["seed"], 42)
        self.assertEqual(stage["eval_batch_size"], 128)
        golden_path = (
            REPO_ROOT
            / f"test/ppu/golden/accuracy_{config['golden_model_key']}_890p.json"
        )
        golden = json.loads(golden_path.read_text())
        self.assertEqual(golden["model"], config["golden_model_key"])
        self.assertEqual(golden["checkpoint"], config["hf_model_id"])
        self.assertEqual(golden["serving"]["ppu_count"], config["ppu_count"])
        self.assertEqual(golden["serving"]["data_parallel_size"], 8)
        self.assertTrue(golden["serving"]["enable_expert_parallel"])
        self.assertEqual(golden["generation"], stage["generation"])
        self.assertEqual(golden["dataset"]["dataset_id"], stage["dataset"])
        self.assertEqual(golden["dataset"]["num_expected"], 1319)
        self.assertEqual(golden["score_num"], 1287)
        self.assertEqual(golden["score"], 0.9757)

    def test_qwen_flash_next_tp4_uses_pure_tensor_parallelism(self):
        model = "qwen3.8-flash-next-tp4"
        self.assertEqual(
            select("fast_sanity", model),
            {"include": [{"model": model, "ppu_count": 4}]},
        )
        policy = DEFAULT_POLICY_ROOT / f"models/{model}.yaml"
        recipe = REPO_ROOT / _load_yaml(policy)["recipe"]
        config = resolve_recipe(recipe, policy_path=policy, tier="fast_sanity")
        dp8 = resolve_recipe(recipe.with_name("Qwen3.8-Flash-Next.yaml"))
        self.assertEqual(config["hf_model_id"], dp8["hf_model_id"])
        self.assertEqual(config["resolved_tokenizer"], dp8["resolved_tokenizer"])
        self.assertIsNone(config["layer_override"])
        flags = config["flags"]
        for name, value in (
            ("--tp-size", "4"),
            ("--context-length", "16384"),
            ("--attention-backend", "triton"),
            ("--speculative-algorithm", "NEXTN"),
            ("--load-format", "dummy"),
        ):
            self.assertEqual(flags[flags.index(name) + 1], value)
        for name in (
            "--dp-size",
            "--attn-dp-size",
            "--enable-dp-lm-head",
            "--moe-a2a-backend",
            "--moe-dense-tp-size",
        ):
            self.assertNotIn(name, flags)
        self.assertIn("--disable-radix-cache", flags)

    def test_qwen_flash_next_tp4_nightly_uses_real_weights_and_paged_mamba_cache(self):
        model = "qwen3.8-flash-next-tp4"
        self.assert_nightly_reference(model, width=4, correct=1285)
        policy = DEFAULT_POLICY_ROOT / f"models/{model}.yaml"
        recipe = REPO_ROOT / _load_yaml(policy)["recipe"]
        config = resolve_recipe(recipe, policy_path=policy, tier="model_sanity")
        flags = config["flags"]
        self.assertEqual(flags[flags.index("--attention-backend") + 1], "triton")
        self.assertEqual(flags[flags.index("--page-size") + 1], "64")
        self.assertEqual(
            flags[flags.index("--mamba-radix-cache-strategy") + 1], "extra_buffer"
        )
        for flag in (
            "--disable-radix-cache",
            "--speculative-algorithm",
            "--load-format",
        ):
            self.assertNotIn(flag, flags)

    def assert_nightly_reference(self, model, *, width, correct):
        self.assertEqual(
            select("model_sanity", model),
            {"include": [{"model": model, "ppu_count": width}]},
        )
        policy_path = DEFAULT_POLICY_ROOT / f"models/{model}.yaml"
        policy = _load_yaml(policy_path)
        config = resolve_recipe(
            REPO_ROOT / policy["recipe"], policy_path=policy_path, tier="model_sanity"
        )
        reference_path = (
            REPO_ROOT
            / "test/ppu/golden"
            / f"accuracy_{config['golden_model_key']}_890p.json"
        )
        reference = json.loads(reference_path.read_text())
        stage = next(
            stage
            for stage in policy["tiers"]["model_sanity"]["stages"]
            if stage["kind"] == "evalscope"
        )
        self.assertEqual(config["hf_model_id"], reference["checkpoint"])
        self.assertEqual(config["ppu_count"], reference["serving"]["ppu_count"])
        self.assertEqual(stage["dataset"], reference["dataset"]["dataset_id"])
        self.assertEqual(stage["generation"], reference["generation"])
        self.assertEqual(reference["tolerance"], {"mode": "band", "abs": 0.02})
        self.assertEqual(reference["dataset"]["num_expected"], 1319)
        self.assertEqual(reference["score_num"], correct)
        self.assertIsNone(stage["limit"])
        self.assertFalse(config["dummy_weights"])

    def test_policy_switches(self):
        policy = DEFAULT_POLICY_ROOT / "models/glm-5.3.yaml"
        recipe = REPO_ROOT / _load_yaml(policy)["recipe"]
        resolved = resolve_recipe(recipe, policy_path=policy, tier="fast_sanity")
        self.assertEqual(
            resolved["argv"][:5], ["python3", "-I", "-B", "-m", "sglang.launch_server"]
        )
        self.assertIsNone(resolved["layer_override"])
        self.assertNotIn("--json-model-override-args", resolved["flags"])
        self.assertIn("dummy", resolved["flags"])
        self.assertNotIn("--disable-radix-cache", resolved["flags"])
        explicit_override = resolve_recipe(
            recipe, policy_path=policy, tier="fast_sanity", layer_override=2
        )
        self.assertIn('{"num_hidden_layers":2}', explicit_override["flags"])
        self.assertNotEqual(
            explicit_override["config_digest"], resolved["config_digest"]
        )
        perf = resolve_recipe(recipe, cache="disabled")
        self.assertIn("--disable-radix-cache", perf["flags"])
        self.assertNotIn("--load-format", perf["flags"])

    def test_fast_runs_each_model_directly_with_full_layers(self):
        @contextlib.contextmanager
        def ready(config, label, timings, timeout):
            timings["cold_start_s"] = 3.0
            yield

        for model, ready_timeout, width in (
            ("glm-5.3", 1350, 8),
            ("glm-5.3-tp4", 1350, 4),
            ("glm-5.3-flash", 1000, 2),
            ("deepseek-v4-flash", 1000, 8),
            ("deepseek-v4-flash-tp8", 1000, 8),
            ("deepseek-v4-pro", 1000, 8),
            ("kimi-k2.6", 1000, 8),
            ("minimax-m3", 1000, 8),
            ("minimax-m3-fp8", 1000, 8),
            ("qwen3.8-27b", 1000, 2),
            ("qwen3.8-flash-next", 1350, 8),
            ("qwen3.8-flash-next-tp4", 1350, 4),
        ):
            with (
                self.subTest(model=model),
                tempfile.TemporaryDirectory() as temp,
                patch.dict(
                    os.environ,
                    {
                        "PPU_RECIPE_CONTAINER_MODEL": temp,
                        "RECIPE_PPU_COUNT": str(width),
                    },
                ),
                patch("recipe_runner.RESULTS", Path(temp) / "results"),
                patch("recipe_runner.server", side_effect=ready) as server,
                patch("recipe_runner.probes") as probes,
            ):
                result = run(
                    DEFAULT_POLICY_ROOT / f"models/{model}.yaml", "fast_sanity", 0
                )
                self.assertEqual(server.call_count, 1)
                config, label, _, timeout = server.call_args.args
                self.assertEqual((label, timeout), ("fast_full", ready_timeout))
                self.assertEqual(config["ppu_count"], width)
                self.assertIn("dummy", config["flags"])
                self.assertNotIn("--json-model-override-args", config["flags"])
                probes.assert_called_once()
                self.assertIsNone(result["layer_override"])
                self.assertNotIn("fallback_reason", result)

    def test_model_sanity_startup_budget_accounts_for_slow_weight_loads(self):
        for model, width, expected_timeout in (
            ("glm-5.3-flash", 2, 6600),
            ("minimax-m3", 8, 6600),
            ("minimax-m3-fp8", 8, 6600),
            ("deepseek-v4-pro", 8, 6000),
            ("kimi-k2.6", 8, 6000),
            ("glm-5.3", 8, 6600),
            ("glm-5.3-tp4", 4, 6600),
            ("qwen3.8-flash-next", 8, 5100),
            ("qwen3.8-flash-next-tp4", 4, 5100),
            ("deepseek-v4-flash", 8, 2400),
        ):
            with (
                self.subTest(model=model),
                tempfile.TemporaryDirectory() as temp,
                patch.dict(
                    os.environ,
                    {
                        "PPU_RECIPE_CONTAINER_MODEL": temp,
                        "RECIPE_PPU_COUNT": str(width),
                    },
                ),
                patch("recipe_runner.RESULTS", Path(temp) / "results"),
                patch(
                    "recipe_runner.server", side_effect=RuntimeError("stop at launch")
                ) as launch,
            ):
                with self.assertRaisesRegex(RuntimeError, "stop at launch"):
                    run(
                        DEFAULT_POLICY_ROOT / f"models/{model}.yaml",
                        "model_sanity",
                        0,
                    )
                config, tier, _, timeout = launch.call_args.args
                self.assertEqual((tier, timeout), ("model_sanity", expected_timeout))
                self.assertNotIn("dummy", config["flags"])

    def test_worker_rejects_recipe_with_wrong_allocation(self):
        with (
            tempfile.TemporaryDirectory() as temp,
            patch.dict(
                os.environ,
                {"PPU_RECIPE_CONTAINER_MODEL": "unused", "RECIPE_PPU_COUNT": "8"},
            ),
            patch("recipe_runner.RESULTS", Path(temp) / "results"),
            patch("recipe_runner.server") as server,
        ):
            with self.assertRaisesRegex(
                ValueError, "incorrect recipe identity or device width"
            ):
                run(DEFAULT_POLICY_ROOT / "models/qwen3.8-27b.yaml", "fast_sanity", 0)
            server.assert_not_called()

    def test_cached_model_requires_the_exact_snapshot_offline(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            snapshot = root / ("a" * 40)
            snapshot.symlink_to(checkpoint, target_is_directory=True)
            for name in ("config.json", "tokenizer.json"):
                (checkpoint / name).write_text("{}")
            (checkpoint / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"weight": "weight.safetensors"}})
            )
            shard = checkpoint / "weight.safetensors"
            shard.write_bytes(b"fixture")
            repo_id = "example/quantized-model"
            revision = "a" * 40
            download = Mock(return_value=str(snapshot))
            with (
                patch.dict(os.environ, {"HF_HUB_CACHE": str(root)}),
                patch.dict(
                    sys.modules,
                    {"huggingface_hub": SimpleNamespace(snapshot_download=download)},
                ),
            ):
                with self.assertRaisesRegex(ValueError, "pinned model snapshot"):
                    cached_model_path(repo_id, "main")
                download.assert_not_called()
                self.assertEqual(
                    cached_model_path(repo_id, revision),
                    (str(checkpoint.resolve()), revision),
                )
                download.assert_called_once_with(
                    repo_id=repo_id,
                    revision=revision,
                    cache_dir=root,
                    local_files_only=True,
                )
                shard.unlink()
                with self.assertRaisesRegex(ValueError, "missing shards"):
                    cached_model_path(repo_id, revision)

    def test_kimi_snapshot_accepts_only_complete_tiktoken_metadata(self):
        repo_id = "T-HEAD/Kimi-K2.6-MXFP4-FP8"
        revision = "d" * 40
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            snapshot = root / revision
            snapshot.mkdir()
            config = snapshot / "config.json"
            config.write_text(json.dumps({"model_type": "kimi_k25"}))
            for name in (
                "tokenizer_config.json",
                "tokenization_kimi.py",
                "tool_declaration_ts.py",
            ):
                (snapshot / name).write_text("fixture")
            (snapshot / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"weight": "weight.safetensors"}})
            )
            (snapshot / "weight.safetensors").write_bytes(b"fixture")
            download = Mock(return_value=str(snapshot))
            with (
                patch.dict(os.environ, {"HF_HUB_CACHE": str(root)}),
                patch.dict(
                    sys.modules,
                    {"huggingface_hub": SimpleNamespace(snapshot_download=download)},
                ),
            ):
                with self.assertRaisesRegex(ValueError, "lacks tokenizer metadata"):
                    cached_model_path(repo_id, revision)
                (snapshot / "tiktoken.model").write_bytes(b"fixture")
                self.assertEqual(
                    cached_model_path(repo_id, revision),
                    (str(snapshot.resolve()), revision),
                )
                config.write_text(json.dumps({"model_type": "qwen3"}))
                with self.assertRaisesRegex(ValueError, "lacks tokenizer metadata"):
                    cached_model_path(repo_id, revision)

    def test_new_model_ref_is_resolved_to_a_concrete_revision(self):
        repo_id = "example/quantized-model"
        revision = "b" * 40
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = root / "models--example--quantized-model"
            ref = repo / "refs/main"
            ref.parent.mkdir(parents=True)
            ref.write_text(revision)
            snapshot = repo / "snapshots" / revision
            snapshot.mkdir(parents=True)
            for name in ("config.json", "tokenizer.json"):
                (snapshot / name).write_text("{}")
            (snapshot / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"weight": "weight.safetensors"}})
            )
            (snapshot / "weight.safetensors").write_bytes(b"fixture")
            download = Mock(return_value=str(snapshot))
            with (
                patch.dict(os.environ, {"HF_HUB_CACHE": str(root)}),
                patch.dict(
                    sys.modules,
                    {
                        "huggingface_hub": SimpleNamespace(snapshot_download=download),
                        "huggingface_hub.file_download": SimpleNamespace(
                            repo_folder_name=Mock(return_value=repo.name)
                        ),
                        "huggingface_hub.utils": SimpleNamespace(
                            validate_repo_id=Mock()
                        ),
                    },
                ),
            ):
                self.assertEqual(
                    cached_model_path(repo_id, None),
                    (str(snapshot.resolve()), revision),
                )
                download.assert_called_once_with(
                    repo_id=repo_id,
                    revision=revision,
                    cache_dir=root,
                    local_files_only=True,
                )
                ref.write_text("not-a-revision")
                with self.assertRaisesRegex(ValueError, "pinned model snapshot"):
                    cached_model_path(repo_id, None)
                self.assertEqual(download.call_count, 1)

    def test_deepseek_attention_parallelism_uses_eight_devices(self):
        policy = DEFAULT_POLICY_ROOT / "models/deepseek-v4-flash.yaml"
        recipe = REPO_ROOT / _load_yaml(policy)["recipe"]
        for tier in ("fast_sanity", "model_sanity"):
            with self.subTest(tier=tier):
                resolved = resolve_recipe(recipe, policy_path=policy, tier=tier)
                self.assertEqual(resolved["ppu_count"], 8)
                flags = resolved["flags"]
                for name, value in (
                    ("--attn-dp-size", "8"),
                    ("--moe-a2a-backend", "deepep"),
                    ("--deepep-mode", "auto"),
                    ("--moe-dense-tp-size", "1"),
                    ("--dist-timeout", "3600"),
                    ("--watchdog-timeout", "3600"),
                ):
                    self.assertEqual(flags[flags.index(name) + 1], value)
                self.assertEqual(
                    resolved["server_env"],
                    {
                        "DG_USE_MOE_DYNAMIC_TILE": "1",
                        "HGGC_EMBEDDED_COPY_THRESHOLD": "0x10000",
                        "NCCL_CUMEM_ENABLE": "1",
                        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "512",
                        "SGLANG_SAIL_DEEPEP_RECV_HOOK": "0",
                    },
                )
                page_size = int(flags[flags.index("--page-size") + 1])
                chunk_size = int(flags[flags.index("--chunked-prefill-size") + 1])
                self.assertEqual(chunk_size % (8 * page_size), 0)

    def test_deepseek_tp8_uses_same_checkpoint_and_paged_short_prefill(self):
        policy = DEFAULT_POLICY_ROOT / "models/deepseek-v4-flash-tp8.yaml"
        recipe = REPO_ROOT / _load_yaml(policy)["recipe"]
        default_policy = DEFAULT_POLICY_ROOT / "models/deepseek-v4-flash.yaml"
        for tier in ("fast_sanity", "model_sanity"):
            with self.subTest(tier=tier):
                resolved = resolve_recipe(recipe, policy_path=policy, tier=tier)
                default = resolve_recipe(recipe, policy_path=default_policy, tier=tier)
                self.assertEqual(resolved["preset"], "tp8")
                self.assertEqual(resolved["ppu_count"], 8)
                self.assertEqual(resolved["hf_model_id"], default["hf_model_id"])
                self.assertNotEqual(resolved["config_digest"], default["config_digest"])
                sparse_prefill = "SGLANG_OPT_FLASHMLA_SPARSE_PREFILL"
                self.assertEqual(
                    resolved["server_env"],
                    {
                        "DG_USE_MOE_DYNAMIC_TILE": "1",
                        "HGGC_EMBEDDED_COPY_THRESHOLD": "0x10000",
                        sparse_prefill: "0",
                    },
                )
                self.assertNotIn(sparse_prefill, default["server_env"])
                flags = resolved["flags"]
                self.assertEqual(flags[flags.index("--tp-size") + 1], "8")
                for flag in (
                    "--attn-dp-size",
                    "--enable-dp-lm-head",
                    "--moe-a2a-backend",
                    "--deepep-mode",
                    "--moe-dense-tp-size",
                    "--dist-timeout",
                    "--watchdog-timeout",
                ):
                    self.assertNotIn(flag, flags)

    def test_required_parser_fails_if_runtime_unavailable(self):
        with patch("ppu_recipe._build_sglang_serve_parser", side_effect=ImportError):
            with self.assertRaises(ValidationError):
                validate_paths(
                    [REPO_ROOT / "test/ppu/recipes/zai-org/GLM-5.3.yaml"],
                    sglang_parser_mode="required",
                )

    def test_resolve_recipe_reports_enabled_features(self):
        dsv4_recipe = REPO_ROOT / "test/ppu/recipes/deepseek-ai/DeepSeek-V4-Flash.yaml"
        dsv4_policy = DEFAULT_POLICY_ROOT / "models/deepseek-v4-flash.yaml"
        dsv4_tp8_policy = DEFAULT_POLICY_ROOT / "models/deepseek-v4-flash-tp8.yaml"
        glm_recipe = REPO_ROOT / "test/ppu/recipes/zai-org/GLM-5.3.yaml"
        glm_policy = DEFAULT_POLICY_ROOT / "models/glm-5.3.yaml"

        self.assertEqual(
            resolve_recipe(dsv4_recipe, policy_path=dsv4_policy, tier="fast_sanity")[
                "enabled_features"
            ],
            ["deepep"],
        )
        self.assertEqual(
            resolve_recipe(
                dsv4_recipe, policy_path=dsv4_tp8_policy, tier="model_sanity"
            )["enabled_features"],
            [],
        )
        self.assertEqual(
            resolve_recipe(glm_recipe, policy_path=glm_policy, tier="fast_sanity")[
                "enabled_features"
            ],
            [],
        )

    def test_checked_in_recipe_matrix_doc_is_up_to_date(self):
        expected = render_ci_matrix_markdown()
        self.assertTrue(DEFAULT_MATRIX_DOC.is_file())
        actual = DEFAULT_MATRIX_DOC.read_text(encoding="utf-8")
        self.assertEqual(
            actual,
            expected,
            "test/ppu/README.md is out of date; regenerate with: "
            "python3 -B scripts/ci/ppu/ppu_recipe.py render-matrix --output test/ppu/README.md",
        )
        self.assertEqual(_scan_text(actual, context="test/ppu/README.md"), [])
        self.assertIn("## PR CI (`fast_sanity`)", actual)
        self.assertIn("## Nightly CI (`model_sanity`)", actual)
        self.assertIn("`default` (none)", actual)
        self.assertIn("`deepep`", actual)

        fast_section, nightly_section = actual.split(
            "## Nightly CI (`model_sanity`)", 1
        )
        for entry in select("fast_sanity", "")["include"]:
            self.assertIn(f"[`{entry['model']}`]", fast_section)
        for entry in select("model_sanity", "")["include"]:
            self.assertIn(f"[`{entry['model']}`]", nightly_section)

    def test_render_matrix_uses_enforced_accuracy_band(self):
        glm_policy = DEFAULT_POLICY_ROOT / "models/glm-5.3.yaml"

        def load_policy_with_different_band(path):
            doc = _load_yaml(path)
            if path == glm_policy:
                doc["accuracy_gate"]["golden_band_abs"] = 0.03
            return doc

        with patch(
            "ppu_recipe._load_yaml", side_effect=load_policy_with_different_band
        ):
            actual = render_ci_matrix_markdown()

        self.assertEqual(actual, DEFAULT_MATRIX_DOC.read_text(encoding="utf-8"))

    def test_render_matrix_cli_check_passes_and_detects_drift(self):
        self.assertEqual(ppu_recipe_main(["render-matrix", "--check"]), 0)

        with tempfile.TemporaryDirectory() as root:
            tmp_dir = Path(root)
            missing = tmp_dir / "missing.md"
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                self.assertEqual(
                    ppu_recipe_main(
                        ["render-matrix", "--output", str(missing), "--check"]
                    ),
                    1,
                )
            self.assertIn("matrix doc missing", stderr.getvalue())

            target = tmp_dir / "README.md"
            target.write_text("# stale\n", encoding="utf-8")
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                self.assertEqual(
                    ppu_recipe_main(
                        ["render-matrix", "--output", str(target), "--check"]
                    ),
                    1,
                )
            self.assertIn("matrix doc out of date", stderr.getvalue())

            self.assertEqual(
                ppu_recipe_main(["render-matrix", "--output", str(target)]), 0
            )
            self.assertEqual(
                ppu_recipe_main(["render-matrix", "--output", str(target), "--check"]),
                0,
            )
            self.assertEqual(
                target.read_text(encoding="utf-8"), render_ci_matrix_markdown()
            )

    def test_bundle_rejects_modified_wheel_and_source(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            source = "a" * 40
            files = (
                "source-sha.txt",
                "sglang-0.5-py3-none-any.whl",
                "sglang_kernel-0.1-cp312-cp312-linux_x86_64.whl",
            )
            for name in files:
                (directory / name).write_bytes(b"fixture")
            (directory / files[0]).write_text(source + "\n")
            (directory / "SHA256SUMS").write_text(
                "".join(f"{sha256(directory / name)}  {name}\n" for name in files)
            )
            wheel_inventory(directory, source)
            with self.assertRaises(ValueError):
                wheel_inventory(directory, "b" * 40)
            (directory / files[1]).write_bytes(b"changed")
            with self.assertRaises(ValueError):
                wheel_inventory(directory, source)


class RunEvidenceTest(unittest.TestCase):
    def test_metrics_require_present_and_zero_for_all_series(self):
        names = ("sglang:num_running_reqs", "sglang:num_queue_reqs")
        first, second = names
        idle = f'{first}{{rank="0"}} 0.0\n{first}{{rank="1"}} 0.0\n{second} 0\n'
        self.assertTrue(metrics_zero(idle, names))
        self.assertFalse(metrics_zero(f"{first} 0.0", names))
        self.assertFalse(metrics_zero(idle + f'{first}{{rank="2"}} 1.0', names))
        self.assertFalse(
            metrics_zero(idle.replace(f"{second} 0", f"{second} NaN"), names)
        )

    def test_eval_rows_count_score_and_uniqueness(self):
        with tempfile.TemporaryDirectory() as root:
            summary = Path(root) / "reviews/model/gsm8k_main.jsonl"
            summary.parent.mkdir(parents=True)
            rows = [
                {
                    "sample_score": {
                        "sample_id": number,
                        "score": {"status": "success", "value": int(number < 1278)},
                    }
                }
                for number in range(1319)
            ]
            summary.write_text("".join(json.dumps(row) + "\n" for row in rows))
            self.assertEqual(read_eval_reviews(Path(root), 1319, 0.9689, 1319)[1], 1278)
            with self.assertRaises(ValueError):
                read_eval_reviews(Path(root), 1319, 0.96, 1319)
            rows[-1] = rows[0]
            summary.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                read_eval_reviews(Path(root), 1319, 0.9689, 1319)

    def test_report_publishes_failure_when_worker_outputs_are_unreadable(self):
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "public.json"
            for outcome in ("failure", "success"):
                with (
                    self.subTest(outcome=outcome),
                    patch.dict(
                        os.environ,
                        {
                            "PPU_CI_ARTIFACT_DIR": temp,
                            "RECIPE_OUTCOME": outcome,
                            "TIER": "fast_sanity",
                            "MODEL_KEY": "glm-5.3",
                            "PPU_CI_SOURCE_SHA": "a" * 40,
                        },
                    ),
                    patch.object(sys, "argv", ["recipe_report", str(destination)]),
                    patch("recipe_report.retain", side_effect=PermissionError),
                    patch.object(Path, "is_file", side_effect=PermissionError),
                ):
                    self.assertEqual(report_recipe(), 1)
                    result = json.loads(destination.read_text())
                    self.assertEqual(result["outcome"], "fail")
                    self.assertEqual(result["failure_phase"], "private_retention")
                    self.assertFalse(result["private_retention"])
                validate(result)

    def test_fast_evidence_accepts_direct_full_and_previous_fallback(self):
        result = fast_result()
        validate(result)
        validate({**result, "fallback_reason": "server_exit"})
        validate({**result, "layer_override": 2})
        with self.assertRaisesRegex(ValueError, "retained a layer override"):
            validate({**result, "layer_override": 2, "fallback_reason": "server_exit"})

    def test_pass_requires_complete_nightly_evidence(self):
        result = fast_result()
        result["tier"] = "model_sanity"
        result.pop("layer_override")
        result.update(
            eval_s=16,
            bench_s=1,
            gsm8k_count=1319,
            gsm8k_correct=1278,
            gsm8k_score=0.9689,
            golden_score=0.9689,
            accuracy_within_band=True,
            baseline_established=True,
            kv_gauge_zero_verified=True,
            c1_passed=True,
            c16_passed=True,
            cache_mode="sanity_throughput_guard",
        )
        validate(result)
        for key in ("baseline_established", "kv_gauge_zero_verified", "c16_passed"):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    validate({**result, key: False})

    def test_public_positive_control(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            artifact = root / "artifact.json"
            summary = root / "summary"
            console = io.StringIO()
            publish(fast_result(), artifact, summary, console)
            self.assertEqual(json.loads(console.getvalue()), fast_result())
            self.assertEqual(artifact.read_text(), console.getvalue())
            self.assertIn(console.getvalue(), summary.read_text())

    def test_three_channels_fail_closed_on_injected_path_and_ip(self):
        simulated = (
            "/" + "nas" + "_aisw/example-secret " + ".".join(("10", "1", "2", "3"))
        )
        for key in (
            "diagnostic",
            "failure_phase",
            "runtime_image",
            "model",
            "source_sha",
        ):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as root:
                root = Path(root)
                candidate, artifact, summary = (
                    root / name
                    for name in ("candidate.json", "artifact.json", "summary")
                )
                candidate.write_text(
                    json.dumps({**fast_result(), "outcome": "fail", key: simulated})
                )
                process = subprocess.run(
                    [
                        sys.executable,
                        "-B",
                        str(REPO_ROOT / "scripts/ci/ppu/recipe_publish.py"),
                        str(candidate),
                        str(artifact),
                        str(summary),
                    ],
                    capture_output=True,
                    text=True,
                    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                )
                self.assertEqual(process.returncode, 1)
                self.assertEqual(process.stdout, "")
                for secret in simulated.split():
                    self.assertNotIn(secret, process.stderr)
                self.assertFalse(artifact.exists())
                self.assertFalse(summary.exists())

    def test_single_device_evidence_excludes_private_diagnostics(self):
        injected = (
            "/" + "nas" + "_aisw/example-secret " + ".".join(("10", "1", "2", "3"))
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics = root / "suite-metrics.jsonl"
            metrics.write_text(
                json.dumps(
                    {
                        "kind": "file",
                        "test_file": "test_basic_sanity_dflash.py",
                        "status": "fail",
                        "duration": 230.2,
                        "error": injected,
                    }
                )
                + "\n"
            )
            result = legacy_result(
                {
                    "PPU_CI_SOURCE_SHA": "a" * 40,
                    "PPU_CI_BOARD": "ZW-M890P",
                    "PPU_LEGACY_RETAINED": "true",
                    "RECIPE_OUTCOME": "failure",
                },
                root,
            )
            self.assertEqual(
                result["cases"],
                {
                    "dflash": {"status": "fail", "duration_s": 230.2},
                    "eagle3": {"status": "not_run", "duration_s": None},
                },
            )
            with self.assertRaises(ValueError):
                validate({**result, "outcome": "pass"})
            console = io.StringIO()
            artifact, summary = root / "artifact.json", root / "summary"
            publish(result, artifact, summary, console)
            self.assertEqual(artifact.read_text(), console.getvalue())
            self.assertIn(console.getvalue(), summary.read_text())
            for output in (
                artifact.read_text(),
                summary.read_text(),
                console.getvalue(),
            ):
                for secret in injected.split():
                    self.assertNotIn(secret, output)
            with self.assertRaises(ValueError):
                publish(
                    {**result, "board": injected},
                    root / "rejected.json",
                    summary,
                    console,
                )
            self.assertFalse((root / "rejected.json").exists())
            self.assertEqual(artifact.read_text(), console.getvalue())


class ClusterConfigBootstrapTest(unittest.TestCase):
    def test_active_runs_keep_their_pinned_private_config(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            key = "PPU_CI_GLM53_MODEL_PATH"
            old_settings = {key: "/old-fixture"}
            new_settings = {key: "/new-fixture"}
            original = root / "config.json"
            original.write_text(json.dumps({"settings": old_settings}))
            original.chmod(0o600)
            old_digest = sha256(original)

            staged = root / "staged.json"
            staged.write_text(json.dumps({"settings": new_settings}))
            staged.chmod(0o600)
            new_digest = sha256(staged)
            versioned = staged.rename(root / f"config.{new_digest}.json")

            self.assertEqual(load_settings(root, old_digest), old_settings)
            self.assertEqual(load_settings(root, new_digest), new_settings)
            versioned.write_text(original.read_text())
            with self.assertRaisesRegex(ValueError, "configuration digest mismatch"):
                load_settings(root, new_digest)
            self.assertEqual(load_settings(root, old_digest), old_settings)

    def test_prior_cluster_diagnostics_produce_fixed_codes(self):
        value = "ImagePullBackOff Permission denied " + "/" + "nas" + "_aisw/example"
        self.assertEqual(diagnostic_codes(value), ["image_pull", "permission_denied"])

    def test_pinned_config_bootstrap_takes_precedence_over_inherited_root(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            scripts = root / "scripts/ci/ppu"
            scripts.mkdir(parents=True)
            (scripts / "recipe_cluster_config.py").write_text("raise SystemExit(29)\n")
            process = subprocess.run(
                ["bash", str(REPO_ROOT / "scripts/ci/ppu/run_recipe_cluster.sh")],
                cwd=root,
                env={
                    **os.environ,
                    "PATH": os.pathsep.join(
                        (str(Path(sys.executable).parent), os.environ["PATH"])
                    ),
                    "GITHUB_OUTPUT": str(root / "output"),
                    "TIER": "fast_sanity",
                    "PPU_CI_PRIVATE_RESULTS_ROOT": str(root / "inherited"),
                    "PPU_CI_K8S_CONFIG_SHA256": "a" * 64,
                    "PPU_CI_K8S_LOADED_CONFIG_SHA256": "",
                },
                capture_output=True,
                timeout=10,
            )
            self.assertEqual(process.returncode, 29)

    def test_cluster_config_resolves_retention_root_and_decouples_scratch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            workspace = root / "workspace"
            runner_temp = root / "runner-temp"
            staging_mount = root / "stage-mount"
            tools = root / "bin"
            backend = workspace / ".ppu-infra/cluster-run"
            scripts = workspace / "scripts/ci/ppu"
            config_dir = staging_mount / CONFIG_NAME
            retained = config_dir / "output/retained"
            env_archive = root / "env-archive"
            cfg_archive = root / "cfg-archive"
            for folder in (
                workspace,
                runner_temp,
                backend,
                tools,
                config_dir,
                retained,
                env_archive,
                cfg_archive,
            ):
                folder.mkdir(parents=True, exist_ok=True)
            config_dir.chmod(0o700)
            retained.chmod(0o700)
            env_archive.chmod(0o755)
            cfg_archive.chmod(0o755)
            shutil.copytree(REPO_ROOT / "scripts/ci/ppu", scripts)
            shutil.copytree(DEFAULT_POLICY_ROOT, workspace / "test/ppu/policy")
            (backend / "action.yml").write_text(
                f'STAGE_TARBALL="{staging_mount}/${{GITHUB_RUN_ID}}-bundle.tar.gz"\n',
                encoding="utf-8",
            )
            (tools / "git").write_text(
                f"#!/bin/bash\necho {BACKEND_SHA}\n", encoding="utf-8"
            )
            (tools / "git").chmod(0o700)
            captured_path = root / "captured-env.json"
            (scripts / "run_recipe_cluster.sh").write_text(
                "#!/usr/bin/env bash\n"
                f"{sys.executable} -c 'import json, os, sys; "
                'open(sys.argv[1], "w").write(json.dumps({k: os.environ.get(k) for k in '
                '("PPU_CI_K8S_HOST_SCRATCH_ROOT", "PPU_CI_PRIVATE_RESULTS_ROOT", '
                '"PPU_CI_K8S_PRIVATE_WORKER_ROOT", "PPU_CI_K8S_LOADED_CONFIG_SHA256")}))\' '
                f'"{captured_path}"\n',
                encoding="utf-8",
            )
            (scripts / "run_recipe_cluster.sh").chmod(0o700)

            def write_config(settings):
                cfg = config_dir / "config.json"
                cfg.write_text(json.dumps({"settings": settings}), encoding="utf-8")
                cfg.chmod(0o600)
                return sha256(cfg)

            base_env = {
                **os.environ,
                "PATH": os.pathsep.join(
                    (
                        str(tools),
                        str(Path(sys.executable).parent),
                        os.environ["PATH"],
                    )
                ),
                "GITHUB_WORKSPACE": str(workspace),
                "RUNNER_TEMP": str(runner_temp),
                "GITHUB_RUN_ID": "777",
            }
            base_env.pop("PPU_CI_PRIVATE_RESULTS_ROOT", None)

            # 1. External env overrides config.json and keeps scratch on output/retained.
            digest_with_cfg = write_config(
                {
                    "PPU_CI_K8S_PRIVATE_WORKER_ROOT": "/worker/retained",
                    "PPU_CI_PRIVATE_RESULTS_ROOT": str(cfg_archive),
                }
            )
            env_case = {
                **base_env,
                "PPU_CI_K8S_CONFIG_SHA256": digest_with_cfg,
                "PPU_CI_PRIVATE_RESULTS_ROOT": str(env_archive),
            }
            check_env = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(scripts / "recipe_cluster_config.py"),
                    "--check",
                ],
                cwd=workspace,
                env=env_case,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(
                check_env.returncode, 0, check_env.stdout + check_env.stderr
            )
            self.assertIn(
                "PPU cluster private configuration verified", check_env.stdout
            )
            status_line = [
                json.loads(line)["private_retention_check"]
                for line in check_env.stdout.splitlines()
                if line.startswith("{")
            ][0]
            self.assertEqual(
                status_line,
                {
                    "external_retention_root": True,
                    "retention_root_valid": True,
                    "retention_source": "env",
                    "retention_writable": True,
                },
            )
            for secret in (str(env_archive), str(cfg_archive), str(staging_mount)):
                self.assertNotIn(secret, check_env.stdout + check_env.stderr)
            probe_dir = env_archive / "sglang/logs/sglang-for-sail"
            self.assertTrue(probe_dir.is_dir())
            self.assertEqual(list(probe_dir.glob(".probe-*")), [])

            exec_env = subprocess.run(
                [sys.executable, "-B", str(scripts / "recipe_cluster_config.py")],
                cwd=workspace,
                env=env_case,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(exec_env.returncode, 0, exec_env.stdout + exec_env.stderr)
            captured = json.loads(captured_path.read_text(encoding="utf-8"))
            self.assertEqual(captured["PPU_CI_K8S_HOST_SCRATCH_ROOT"], str(retained))
            self.assertEqual(captured["PPU_CI_PRIVATE_RESULTS_ROOT"], str(env_archive))
            self.assertEqual(
                captured["PPU_CI_K8S_PRIVATE_WORKER_ROOT"], "/worker/retained"
            )

            # 2. Config value is used when env is unset or empty.
            cfg_case = {
                **base_env,
                "PPU_CI_K8S_CONFIG_SHA256": digest_with_cfg,
                "PPU_CI_PRIVATE_RESULTS_ROOT": "",
            }
            check_cfg = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(scripts / "recipe_cluster_config.py"),
                    "--check",
                ],
                cwd=workspace,
                env=cfg_case,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(
                check_cfg.returncode, 0, check_cfg.stdout + check_cfg.stderr
            )
            cfg_status = [
                json.loads(line)["private_retention_check"]
                for line in check_cfg.stdout.splitlines()
                if line.startswith("{")
            ][0]
            self.assertEqual(cfg_status["retention_source"], "config")
            self.assertTrue(cfg_status["external_retention_root"])
            self.assertNotIn(str(cfg_archive), check_cfg.stdout + check_cfg.stderr)

            exec_cfg = subprocess.run(
                [sys.executable, "-B", str(scripts / "recipe_cluster_config.py")],
                cwd=workspace,
                env=cfg_case,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(exec_cfg.returncode, 0, exec_cfg.stdout + exec_cfg.stderr)
            captured = json.loads(captured_path.read_text(encoding="utf-8"))
            self.assertEqual(captured["PPU_CI_K8S_HOST_SCRATCH_ROOT"], str(retained))
            self.assertEqual(captured["PPU_CI_PRIVATE_RESULTS_ROOT"], str(cfg_archive))

            # 3. Fallback to output/retained when neither env nor config sets it.
            digest_fallback = write_config(
                {"PPU_CI_K8S_PRIVATE_WORKER_ROOT": "/worker/retained"}
            )
            fallback_case = {
                **base_env,
                "PPU_CI_K8S_CONFIG_SHA256": digest_fallback,
            }
            check_fallback = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(scripts / "recipe_cluster_config.py"),
                    "--check",
                ],
                cwd=workspace,
                env=fallback_case,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(
                check_fallback.returncode,
                0,
                check_fallback.stdout + check_fallback.stderr,
            )
            fb_status = [
                json.loads(line)["private_retention_check"]
                for line in check_fallback.stdout.splitlines()
                if line.startswith("{")
            ][0]
            self.assertEqual(
                fb_status,
                {
                    "external_retention_root": False,
                    "retention_root_valid": True,
                    "retention_source": "fallback",
                    "retention_writable": True,
                },
            )
            self.assertNotIn(
                str(retained), check_fallback.stdout + check_fallback.stderr
            )

            exec_fallback = subprocess.run(
                [sys.executable, "-B", str(scripts / "recipe_cluster_config.py")],
                cwd=workspace,
                env=fallback_case,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(
                exec_fallback.returncode, 0, exec_fallback.stdout + exec_fallback.stderr
            )
            captured = json.loads(captured_path.read_text(encoding="utf-8"))
            self.assertEqual(captured["PPU_CI_K8S_HOST_SCRATCH_ROOT"], str(retained))
            self.assertEqual(captured["PPU_CI_PRIVATE_RESULTS_ROOT"], str(retained))

            # 4. Invalid external roots fail closed without leaking paths.
            bad_symlink = root / "bad-link"
            bad_symlink.symlink_to(env_archive)
            bad_mode = root / "bad-mode"
            bad_mode.mkdir()
            bad_mode.chmod(0o777)
            bad_ephemeral = runner_temp / "nested"
            bad_ephemeral.mkdir(mode=0o700)
            for bad_root in (bad_symlink, bad_mode, bad_ephemeral):
                with self.subTest(bad_root=bad_root.name):
                    bad_proc = subprocess.run(
                        [
                            sys.executable,
                            "-B",
                            str(scripts / "recipe_cluster_config.py"),
                            "--check",
                        ],
                        cwd=workspace,
                        env={
                            **fallback_case,
                            "PPU_CI_PRIVATE_RESULTS_ROOT": str(bad_root),
                        },
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    self.assertEqual(bad_proc.returncode, 2)
                    self.assertNotIn(str(bad_root), bad_proc.stdout + bad_proc.stderr)
                    self.assertNotIn(str(root), bad_proc.stdout + bad_proc.stderr)
                    bad_status = [
                        json.loads(line)["private_retention_check"]
                        for line in bad_proc.stdout.splitlines()
                        if line.startswith("{")
                    ][0]
                    self.assertTrue(bad_status["external_retention_root"])
                    self.assertFalse(bad_status["retention_root_valid"])
                    self.assertFalse(bad_status["retention_writable"])

            # 5. Normal Recipe exec path fails fast on unwritable external archive,
            #    while legacy tier bypasses archive probing.
            ro_archive = root / "ro-archive"
            ro_base = ro_archive / "sglang/logs/sglang-for-sail"
            ro_base.mkdir(parents=True)
            ro_archive.chmod(0o755)
            (ro_archive / "sglang").chmod(0o755)
            (ro_archive / "sglang/logs").chmod(0o755)
            ro_base.chmod(0o500)
            try:
                for tier in ("fast_sanity", "model_sanity"):
                    with self.subTest(fail_fast_tier=tier):
                        if captured_path.exists():
                            captured_path.unlink()
                        blocked = subprocess.run(
                            [
                                sys.executable,
                                "-B",
                                str(scripts / "recipe_cluster_config.py"),
                            ],
                            cwd=workspace,
                            env={
                                **fallback_case,
                                "TIER": tier,
                                "PPU_CI_PRIVATE_RESULTS_ROOT": str(ro_archive),
                            },
                            capture_output=True,
                            text=True,
                            timeout=10,
                        )
                        self.assertEqual(blocked.returncode, 2)
                        self.assertFalse(captured_path.exists())
                        self.assertNotIn(
                            str(ro_archive), blocked.stdout + blocked.stderr
                        )
                        blocked_status = [
                            json.loads(line)["private_retention_check"]
                            for line in blocked.stdout.splitlines()
                            if line.startswith("{")
                        ][0]
                        self.assertTrue(blocked_status["external_retention_root"])
                        self.assertFalse(blocked_status["retention_writable"])

                if captured_path.exists():
                    captured_path.unlink()
                legacy_exec = subprocess.run(
                    [sys.executable, "-B", str(scripts / "recipe_cluster_config.py")],
                    cwd=workspace,
                    env={
                        **fallback_case,
                        "TIER": "legacy",
                        "PPU_CI_PRIVATE_RESULTS_ROOT": str(ro_archive),
                    },
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self.assertEqual(
                    legacy_exec.returncode, 0, legacy_exec.stdout + legacy_exec.stderr
                )
                self.assertTrue(captured_path.exists())
            finally:
                ro_base.chmod(0o700)

            # 6. Pre-migration diagnostics under retained/sglang-for-sail remain
            #    visible via --check when switching to a fresh external archive root.
            fresh_archive = root / "fresh-archive"
            fresh_archive.mkdir(mode=0o755)
            fresh_archive.chmod(0o755)
            (fresh_archive / "sglang/logs/sglang-for-sail/99999-1/results").mkdir(
                parents=True
            )
            old_run = retained / "sglang-for-sail/770-1/fast_sanity/qwen3.8-27b"
            old_run.mkdir(parents=True)
            (old_run / "index.json").write_text("{}", encoding="utf-8")
            (old_run / "launcher.log").write_text(
                "ImagePullBackOff failed to pull image\n", encoding="utf-8"
            )
            migrated_check = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(scripts / "recipe_cluster_config.py"),
                    "--check",
                ],
                cwd=workspace,
                env={
                    **fallback_case,
                    "PPU_CI_PRIVATE_RESULTS_ROOT": str(fresh_archive),
                },
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(
                migrated_check.returncode,
                0,
                migrated_check.stdout + migrated_check.stderr,
            )
            diag_lines = [
                json.loads(line)["prior_cluster_diagnostics"]
                for line in migrated_check.stdout.splitlines()
                if line.startswith('{"prior_cluster_diagnostics"')
            ]
            self.assertEqual(
                diag_lines,
                [
                    [
                        {
                            "codes": ["image_pull"],
                            "model": "qwen3.8-27b",
                            "run_id": 770,
                            "tier": "fast_sanity",
                        }
                    ]
                ],
            )
            self.assertNotIn(
                str(fresh_archive), migrated_check.stdout + migrated_check.stderr
            )
            self.assertNotIn(
                str(retained), migrated_check.stdout + migrated_check.stderr
            )

    def test_same_parent_run_retains_both_source_versions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            archive = root / "archive"
            archive.mkdir(mode=0o700)
            source = root / "scratch"
            (source / "results").mkdir(parents=True)
            (source / "results/summary.json").write_text('{"outcome": "pass"}')
            (source / "launcher.log").write_text("ok\n")
            (source / "provenance.json").write_text(
                json.dumps(
                    {
                        "runtime_image": fast_result()["runtime_image"],
                        "recipe_tree_sha256": "c" * 64,
                    }
                )
            )
            env = {
                "TIER": "model_sanity",
                "MODEL_KEY": "glm-5.3-flash",
                "GITHUB_RUN_ID": "12345",
                "GITHUB_RUN_ATTEMPT": "1",
                "PPU_CI_PRIVATE_RESULTS_ROOT": str(archive),
                "RUNNER_TEMP": str(root / "runner-temp"),
                "GITHUB_WORKSPACE": str(root / "workspace"),
                "PPU_CI_ARTIFACT_DIR": str(source),
                "RECIPE_OUTCOME": "success",
            }
            for attempt, use_report in (("1", False), ("2", True)):
                run = archive / "sglang/logs/sglang-for-sail" / f"12345-{attempt}"
                env["GITHUB_RUN_ATTEMPT"] = attempt
                for checkout_sha in ("a" * 40, "b" * 40):
                    env["PPU_CI_SOURCE_SHA"] = checkout_sha
                    with self.subTest(attempt=attempt, sha=checkout_sha):
                        if use_report:
                            report = root / f"report-{checkout_sha[:12]}.json"
                            with (
                                patch.dict(os.environ, env),
                                patch.object(
                                    sys, "argv", ["recipe_report", str(report)]
                                ),
                            ):
                                self.assertEqual(report_recipe(), 0)
                            result = json.loads(report.read_text())
                            self.assertEqual(result["outcome"], "pass")
                            self.assertIs(result["private_retention"], True)
                            self.assertEqual(result["source_sha"], checkout_sha)
                        else:
                            retain(env)
                        scope = run / checkout_sha[:12]
                        tier = scope / env["TIER"]
                        dest = tier / env["MODEL_KEY"]
                        index_path = dest / "index.json"
                        self.assertEqual(
                            json.loads(index_path.read_text()),
                            {
                                "checkout_sha": checkout_sha,
                                "run_id": env["GITHUB_RUN_ID"],
                                "attempt": attempt,
                                "tier": env["TIER"],
                                "model_key": env["MODEL_KEY"],
                                "files": {
                                    name: sha256(dest / name)
                                    for name in (
                                        "launcher.log",
                                        "summary.json",
                                        "provenance.json",
                                    )
                                },
                            },
                        )
                        for directory in (run, scope, tier, dest):
                            self.assertEqual(
                                stat.S_IMODE(directory.lstat().st_mode), 0o700
                            )
                        self.assertEqual(
                            stat.S_IMODE(index_path.lstat().st_mode), 0o600
                        )
                        with self.assertRaisesRegex(
                            FileExistsError, "retention destination already exists"
                        ):
                            retain(env)

    def test_recent_failures_combines_source_scopes_and_legacy_layout(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / "sglang/logs/sglang-for-sail/12345-2"
            for scope, log in (
                ("a" * 12, "ImagePullBackOff"),
                ("b" * 12, "ImagePullBackOff\nInsufficient cpu"),
                ("", "permission denied"),
            ):
                model = run / scope / "model_sanity/glm-5.3-flash"
                model.mkdir(parents=True)
                (model / "index.json").write_text("{}")
                (model / "launcher.log").write_text(log)

            self.assertEqual(
                recent_failures(root, 12346),
                [
                    {
                        "run_id": 12345,
                        "tier": "model_sanity",
                        "model": "glm-5.3-flash",
                        "codes": ["capacity_cpu", "image_pull", "permission_denied"],
                    }
                ],
            )

    def test_retain_accepts_0755_prefix_dirs_and_rejects_bad_run_dir_or_symlink(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            archive = root / "archive"
            workspace = root / "workspace"
            runner_temp = root / "runner-temp"
            source_dir = root / "scratch"
            for folder in (archive, workspace, runner_temp, source_dir / "results"):
                folder.mkdir(parents=True, exist_ok=True)
            archive.chmod(0o755)
            (archive / "sglang").mkdir(mode=0o755)
            (archive / "sglang").chmod(0o755)
            (archive / "sglang/logs").mkdir(mode=0o755)
            (archive / "sglang/logs").chmod(0o755)
            (source_dir / "launcher.log").write_text("ok\n", encoding="utf-8")
            (source_dir / "results/summary.json").write_text("{}", encoding="utf-8")
            (source_dir / "provenance.json").write_text("{}", encoding="utf-8")

            env = {
                "PPU_CI_SOURCE_SHA": "a" * 40,
                "TIER": "fast_sanity",
                "MODEL_KEY": "qwen3.8-27b",
                "GITHUB_RUN_ID": "12345",
                "GITHUB_RUN_ATTEMPT": "2",
                "PPU_CI_PRIVATE_RESULTS_ROOT": str(archive),
                "RUNNER_TEMP": str(runner_temp),
                "GITHUB_WORKSPACE": str(workspace),
                "PPU_CI_ARTIFACT_DIR": str(source_dir),
                "RECIPE_OUTCOME": "success",
            }
            retain(env)

            dest = (
                archive
                / "sglang/logs/sglang-for-sail/12345-2/aaaaaaaaaaaa/fast_sanity/qwen3.8-27b"
            )
            self.assertTrue((dest / "index.json").is_file())
            self.assertTrue((dest / "launcher.log").is_file())
            self.assertEqual(
                stat.S_IMODE((archive / "sglang/logs").lstat().st_mode), 0o755
            )
            self.assertEqual(
                stat.S_IMODE(
                    (archive / "sglang/logs/sglang-for-sail/12345-2").lstat().st_mode
                ),
                0o700,
            )
            self.assertEqual(
                stat.S_IMODE(
                    (
                        archive
                        / "sglang/logs/sglang-for-sail/12345-2/aaaaaaaaaaaa/fast_sanity"
                    )
                    .lstat()
                    .st_mode
                ),
                0o700,
            )

            # Single-run directory must remain strict 0o700.
            bad_run = archive / "sglang/logs/sglang-for-sail/12346-1"
            bad_run.mkdir(mode=0o755)
            bad_run.chmod(0o755)
            with self.assertRaisesRegex(
                ValueError, "unexpected retention directory type or mode"
            ):
                retain({**env, "GITHUB_RUN_ID": "12346", "GITHUB_RUN_ATTEMPT": "1"})

            bad_scope = archive / "sglang/logs/sglang-for-sail/12345-2" / ("b" * 12)
            bad_scope.mkdir(mode=0o755)
            bad_scope.chmod(0o755)
            bad_scope_env = {**env, "PPU_CI_SOURCE_SHA": "b" * 40}
            with self.assertRaisesRegex(
                ValueError, "unexpected retention directory type or mode"
            ):
                retain(bad_scope_env)
            bad_scope.rmdir()
            bad_scope.symlink_to(bad_scope.with_name("a" * 12))
            with self.assertRaisesRegex(
                ValueError, "unexpected retention directory type or mode"
            ):
                retain(bad_scope_env)

            # Symlink prefix component must be rejected.
            sym_archive = root / "sym-archive"
            sym_archive.mkdir(mode=0o755)
            sym_archive.chmod(0o755)
            (sym_archive / "sglang").symlink_to(archive / "sglang")
            with self.assertRaisesRegex(
                ValueError, "unexpected retention directory type or mode"
            ):
                retain({**env, "PPU_CI_PRIVATE_RESULTS_ROOT": str(sym_archive)})

            # Effective archive base overlapping workspace must be rejected, and
            # missing RUNNER_TEMP must raise ValueError instead of KeyError.
            overlap_ws = archive / "sglang"
            with self.assertRaisesRegex(ValueError, "must not be job-local"):
                retain({**env, "GITHUB_WORKSPACE": str(overlap_ws)})
            missing_temp_env = {k: v for k, v in env.items() if k != "RUNNER_TEMP"}
            with self.assertRaisesRegex(ValueError, "missing runner workspace path"):
                retain(missing_temp_env)


@unittest.skipUnless(
    all(shutil.which(tool) for tool in ("jq", "sha256sum")),
    "runner tools are required",
)
class ClusterTransportTest(unittest.TestCase):
    def test_scheduler_streams_stay_private(self):
        injected = "/" + "nas" + "_aisw/fixture " + ".".join(("10", "4", "5", "6"))
        for model, width, tampered in (
            ("glm-5.3", 8, False),
            ("glm-5.3-tp4", 4, False),
            ("deepseek-v4-flash", 8, False),
            ("deepseek-v4-flash-tp8", 8, False),
            ("deepseek-v4.1-flash", 8, False),
            ("qwen3.8-27b", 2, False),
            ("qwen3.8-flash-next-tp4", 4, False),
            ("glm-5.3", 8, True),
        ):
            with (
                self.subTest(model=model, tampered=tampered),
                tempfile.TemporaryDirectory() as temp,
            ):
                result = dict(fast_result(), model=model)
                root = Path(temp)
                workspace = root / "checkout"
                staging = root / "staging"
                retained = root / "retained"
                tools = root / "bin"
                backend = workspace / ".ppu-infra/cluster-run/scripts"
                payloads = workspace / "wheels"
                for folder in (staging, retained, tools, backend, payloads):
                    folder.mkdir(parents=True)
                scripts = workspace / "scripts/ci/ppu"
                shutil.copytree(REPO_ROOT / "scripts/ci/ppu", scripts)
                shutil.copytree(DEFAULT_POLICY_ROOT, workspace / "test/ppu/policy")
                (scripts / "recipe_bundle.py").write_text("# Verified separately\n")
                (payloads / "recipe-manifest.json").write_text(
                    json.dumps({"recipe_tree_sha256": result["recipe_tree_sha256"]})
                )
                summary = {
                    "outcome": "pass",
                    "parser_checked": True,
                    "sse_verified": True,
                    "cold_start_s": 3.0,
                    "queue_s": 0,
                    "layer_override": None,
                    "config_digest": result["config_digest"],
                }
                if tampered:
                    summary["diagnostic"] = injected
                (workspace / "fixture.json").write_text(json.dumps(summary))
                (tools / "git").write_text(
                    '#!/bin/bash\ncase "$1" in -C) echo 96e25147971f5d2a04d828f8ad508cae38a4fa8a;; '
                    f"*) echo {result['source_sha']};; esac\n"
                )
                kube_content = json.dumps(
                    {
                        "status": {
                            "containerStatuses": [
                                {
                                    "name": "worker",
                                    "imageID": "docker-pullable://registry.internal/team/sglang"
                                    + result["runtime_image"],
                                }
                            ]
                        }
                    }
                )
                (tools / "kubectl").write_text(
                    f"#!/bin/bash\nif [[ \"$*\" == *'get pod '* ]]; then echo '{kube_content}'; fi\n"
                )
                (tools / "timeout").write_text('#!/bin/bash\nshift\nexec "$@"\n')
                for tool in tools.iterdir():
                    tool.chmod(0o700)
                (backend / "submit.sh").write_text(
                    f'test "$INPUT_NNODES/$INPUT_NPROC_PER_NODE/$INPUT_NODE_SELECTOR" = "0/{width}/board-type=ZW-M890P" || exit 1\n'
                    'test -f "$SOURCE_STAGE_DIR" || exit 1\n'
                    'jq -e --arg cache "$PPU_CI_HF_CACHE_ROOT" --arg model "$MODEL_KEY" '
                    '--arg width "$INPUT_NPROC_PER_NODE" --arg revision "$EXPECTED_REVISION" '
                    "'.PPU_CI_HF_CACHE_ROOT == $cache and .MODEL_KEY == $model and "
                    '.RECIPE_PPU_COUNT == $width and (.PPU_CI_HF_MODEL_REVISION // "") == $revision\' '
                    '<<< "$USER_ENV_JSON" >/dev/null || exit 1\n'
                    'printf "%s\\n" "$INJECTED_FIXTURE"\n'
                    'printf "job_status=scheduled\\nstart_time=1\\ndeadline=2\\nqueue_duration_seconds=7\\n" >> "$GITHUB_OUTPUT"\n'
                )
                (backend / "monitor.sh").write_text(
                    'printf "%s\\n" "$INJECTED_FIXTURE" >> "$GITHUB_STEP_SUMMARY"\n'
                    'mkdir -p "$PPU_CI_WORKER_ARTIFACT_DIR/results"\n'
                    'cp fixture.json "$PPU_CI_WORKER_ARTIFACT_DIR/results/summary.json"\n'
                    'echo job_status=succeeded >> "$GITHUB_OUTPUT"\n'
                )
                (backend / "cleanup.sh").write_text(
                    'printf "%s\\n" "$INJECTED_FIXTURE"\n'
                    'unlink "$SOURCE_STAGE_DIR"\n'
                    'rmdir "$(dirname "$SOURCE_STAGE_DIR")"\n'
                )
                export_env = {
                    **os.environ,
                    "PATH": os.pathsep.join(
                        (
                            str(tools),
                            str(Path(sys.executable).parent),
                            os.environ["PATH"],
                        )
                    ),
                    "TIER": "fast_sanity",
                    "MODEL_KEY": model,
                    "RECIPE_PPU_COUNT": str(width),
                    "GITHUB_OUTPUT": str(root / "gha-output"),
                    "PPU_PUBLIC_SUMMARY": str(root / "gha-summary"),
                    "GITHUB_WORKSPACE": str(workspace),
                    "RUNNER_TEMP": str(root / "runner-temp"),
                    "GITHUB_RUN_ID": "12",
                    "GITHUB_RUN_ATTEMPT": "1",
                    "PPU_CI_K8S_CONFIG_SHA256": "",
                    "PPU_CI_K8S_LOADED_CONFIG_SHA256": "",
                    "PPU_CI_PRIVATE_RESULTS_ROOT": str(retained),
                    "PPU_CI_K8S_PRIVATE_WORKER_ROOT": str(retained),
                    "PPU_CI_K8S_STAGE_ROOT": str(staging),
                    "PPU_CI_SOURCE_SHA": result["source_sha"],
                    "PPU_CI_JIT_ROOT": str(root / "jit"),
                    "PPU_CI_HF_CACHE_ROOT": str(root / "cache"),
                    "PPU_CI_GLM53_MODEL_REVISION": "c" * 40,
                    "PPU_CI_DSV4_FLASH_MODEL_REVISION": "d" * 40,
                    "EXPECTED_REVISION": {
                        "glm-5.3": "c" * 40,
                        "glm-5.3-tp4": "c" * 40,
                        "deepseek-v4-flash": "d" * 40,
                        "deepseek-v4-flash-tp8": "d" * 40,
                    }.get(model, ""),
                    "PPU_CI_K8S_RUNTIME_DIGEST": "registry.internal/team/sglang"
                    + result["runtime_image"],
                    "INJECTED_FIXTURE": injected,
                }
                process = subprocess.run(
                    ["bash", str(scripts / "run_recipe_cluster.sh")],
                    cwd=workspace,
                    env=export_env,
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                artifact = workspace / "ppu-public/recipe.json"
                job_summary = root / "gha-summary"
                self.assertEqual(
                    process.returncode, int(tampered), process.stdout + process.stderr
                )
                self.assertNotIn(injected, process.stdout + process.stderr)
                self.assertEqual(list(staging.iterdir()), [])
                if tampered:
                    self.assertFalse(artifact.exists())
                    self.assertFalse(job_summary.exists())
                    self.assertFalse((root / "gha-output").exists())
                else:
                    public_record = json.loads(artifact.read_text())
                    self.assertEqual(public_record, {**result, "queue_s": 7})
                    self.assertIn(artifact.read_text(), job_summary.read_text())
                    self.assertIn(artifact.read_text(), process.stdout)
                    self.assertFalse((root / "gha-output").exists())
                    self.assertNotIn(
                        injected, artifact.read_text() + job_summary.read_text()
                    )
                    self.assertNotIn(
                        "registry.internal",
                        artifact.read_text()
                        + job_summary.read_text()
                        + process.stdout
                        + process.stderr,
                    )
                (original_log,) = retained.glob("cluster.*/launcher.log")
                self.assertIn(injected, original_log.read_text())
                if not tampered:
                    archived_index = (
                        retained
                        / "sglang/logs/sglang-for-sail/12-1/aaaaaaaaaaaa/fast_sanity"
                        / model
                        / "index.json"
                    )
                    self.assertTrue(archived_index.is_file())

    def test_cluster_transport_separates_scratch_from_external_archive_root(self):
        with tempfile.TemporaryDirectory() as temp:
            result = dict(fast_result(), model="qwen3.8-27b")
            root = Path(temp).resolve()
            workspace = root / "checkout"
            staging = root / "staging"
            scratch = root / "scratch"
            archive = root / "external-archive"
            tools = root / "bin"
            backend = workspace / ".ppu-infra/cluster-run/scripts"
            payloads = workspace / "wheels"
            for folder in (staging, scratch, archive, tools, backend, payloads):
                folder.mkdir(parents=True)
            scratch.chmod(0o700)
            archive.chmod(0o755)
            (archive / "sglang").mkdir(mode=0o755)
            (archive / "sglang").chmod(0o755)
            (archive / "sglang/logs").mkdir(mode=0o755)
            (archive / "sglang/logs").chmod(0o755)
            scripts = workspace / "scripts/ci/ppu"
            shutil.copytree(REPO_ROOT / "scripts/ci/ppu", scripts)
            shutil.copytree(DEFAULT_POLICY_ROOT, workspace / "test/ppu/policy")
            (scripts / "recipe_bundle.py").write_text("# Verified separately\n")
            (payloads / "recipe-manifest.json").write_text(
                json.dumps({"recipe_tree_sha256": result["recipe_tree_sha256"]})
            )
            summary = {
                "outcome": "pass",
                "parser_checked": True,
                "sse_verified": True,
                "cold_start_s": 3.0,
                "queue_s": 0,
                "layer_override": None,
                "config_digest": result["config_digest"],
            }
            (workspace / "fixture.json").write_text(json.dumps(summary))
            (tools / "git").write_text(
                f'#!/bin/bash\ncase "$1" in -C) echo {BACKEND_SHA};; '
                f"*) echo {result['source_sha']};; esac\n"
            )
            kube_content = json.dumps(
                {
                    "status": {
                        "containerStatuses": [
                            {
                                "name": "worker",
                                "imageID": "docker-pullable://registry.internal/team/sglang"
                                + result["runtime_image"],
                            }
                        ]
                    }
                }
            )
            (tools / "kubectl").write_text(
                f"#!/bin/bash\nif [[ \"$*\" == *'get pod '* ]]; then echo '{kube_content}'; fi\n"
            )
            (tools / "timeout").write_text('#!/bin/bash\nshift\nexec "$@"\n')
            for tool in tools.iterdir():
                tool.chmod(0o700)
            (backend / "submit.sh").write_text(
                'printf "job_status=scheduled\\nstart_time=1\\ndeadline=2\\nqueue_duration_seconds=3\\n" >> "$GITHUB_OUTPUT"\n'
            )
            (backend / "monitor.sh").write_text(
                'mkdir -p "$PPU_CI_WORKER_ARTIFACT_DIR/results"\n'
                'cp fixture.json "$PPU_CI_WORKER_ARTIFACT_DIR/results/summary.json"\n'
                'echo job_status=succeeded >> "$GITHUB_OUTPUT"\n'
            )
            (backend / "cleanup.sh").write_text(
                'unlink "$SOURCE_STAGE_DIR"\nrmdir "$(dirname "$SOURCE_STAGE_DIR")"\n'
            )
            export_env = {
                **os.environ,
                "PATH": os.pathsep.join(
                    (
                        str(tools),
                        str(Path(sys.executable).parent),
                        os.environ["PATH"],
                    )
                ),
                "TIER": "fast_sanity",
                "MODEL_KEY": "qwen3.8-27b",
                "RECIPE_PPU_COUNT": "2",
                "GITHUB_OUTPUT": str(root / "gha-output"),
                "PPU_PUBLIC_SUMMARY": str(root / "gha-summary"),
                "GITHUB_WORKSPACE": str(workspace),
                "RUNNER_TEMP": str(root / "runner-temp"),
                "GITHUB_RUN_ID": "42",
                "GITHUB_RUN_ATTEMPT": "1",
                "PPU_CI_K8S_CONFIG_SHA256": "",
                "PPU_CI_K8S_LOADED_CONFIG_SHA256": "",
                "PPU_CI_K8S_HOST_SCRATCH_ROOT": str(scratch),
                "PPU_CI_K8S_PRIVATE_WORKER_ROOT": str(scratch),
                "PPU_CI_PRIVATE_RESULTS_ROOT": str(archive),
                "PPU_CI_K8S_STAGE_ROOT": str(staging),
                "PPU_CI_SOURCE_SHA": result["source_sha"],
                "PPU_CI_JIT_ROOT": str(root / "jit"),
                "PPU_CI_HF_CACHE_ROOT": str(root / "cache"),
                "PPU_CI_K8S_RUNTIME_DIGEST": "registry.internal/team/sglang"
                + result["runtime_image"],
            }
            process = subprocess.run(
                ["bash", str(scripts / "run_recipe_cluster.sh")],
                cwd=workspace,
                env=export_env,
                capture_output=True,
                text=True,
                timeout=20,
            )
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            self.assertEqual(len(list(scratch.glob("cluster.*/launcher.log"))), 1)
            self.assertEqual(list(archive.glob("cluster.*")), [])
            archived_dir = (
                archive
                / "sglang/logs/sglang-for-sail/42-1/aaaaaaaaaaaa/fast_sanity/qwen3.8-27b"
            )
            self.assertTrue((archived_dir / "index.json").is_file())
            self.assertTrue((archived_dir / "launcher.log").is_file())
            public_text = (
                (workspace / "ppu-public/recipe.json").read_text()
                + (root / "gha-summary").read_text()
                + process.stdout
                + process.stderr
            )
            self.assertNotIn(str(archive), public_text)
            self.assertNotIn(str(scratch), public_text)

    def test_cluster_diagnostics_classifies_server_logs_and_startup_phases(self):
        import recipe_cluster_diagnostics
        import recipe_diagnostics
        import recipe_runner

        self.assertEqual(recipe_runner.bench_module_name(), "sglang.benchmark.serving")
        for phase in ("server_exit", "readiness_timeout"):
            self.assertIn(phase, recipe_diagnostics.PHASES)
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            fast_dir = (
                root
                / "sglang/logs/sglang-for-sail/45014946019-1/fast_sanity/qwen3.8-27b"
            )
            fast_dir.mkdir(parents=True)
            (fast_dir / "index.json").write_text("{}")
            (fast_dir / "sglang_serve_fast_full.log").write_text(
                "ModuleNotFoundError: No module named 'blake3'\n"
            )
            nightly_dir = (
                root / "sglang/logs/sglang-for-sail/45055310116-1/model_sanity/glm-5.3"
            )
            nightly_dir.mkdir(parents=True)
            (nightly_dir / "index.json").write_text("{}")
            (nightly_dir / "bench-warmup-1.log").write_text(
                "/usr/bin/python3: No module named sglang.bench_serving\n"
            )
            (nightly_dir / "sglang_serve_model_sanity.log").write_text(
                "server_args={'moe_a2a_backend': 'deepep', 'watchdog_timeout': 3600.0}\n"
                "ModuleNotFoundError: No module named 'triton_kernels'\n"
            )
            records = recipe_cluster_diagnostics.recent_failures(root, 45055310117)
            self.assertEqual(
                records,
                [
                    {
                        "run_id": 45055310116,
                        "tier": "model_sanity",
                        "model": "glm-5.3",
                        "codes": ["missing_triton_kernels", "python_import"],
                    }
                ],
            )

        with tempfile.TemporaryDirectory() as legacy_root:
            root = Path(legacy_root)
            (root / "sglang/logs/sglang-for-sail").mkdir(parents=True)
            legacy_dir = root / "sglang-for-sail/45014946019-1/fast_sanity/qwen3.8-27b"
            legacy_dir.mkdir(parents=True)
            (legacy_dir / "index.json").write_text("{}")
            (legacy_dir / "launcher.log").write_text(
                "ImagePullBackOff failed to pull image\n"
            )
            records = recipe_cluster_diagnostics.recent_failures(root, 45055310117)
            self.assertEqual(
                records,
                [
                    {
                        "run_id": 45014946019,
                        "tier": "fast_sanity",
                        "model": "qwen3.8-27b",
                        "codes": ["image_pull"],
                    }
                ],
            )

        for layout in ("sglang/logs/sglang-for-sail", "sglang-for-sail"):
            for link_level in ("tier", "model", "index", "log"):
                with (
                    self.subTest(layout=layout, link_level=link_level),
                    tempfile.TemporaryDirectory() as outer_tmp,
                ):
                    outer = Path(outer_tmp)
                    root = outer / "archive"
                    outside = outer / "outside"
                    outside_model = outside / "qwen3.8-27b"
                    outside_model.mkdir(parents=True)
                    (outside_model / "index.json").write_text("{}")
                    (outside_model / "launcher.log").write_text(
                        "ImagePullBackOff failed to pull image\n"
                    )
                    run_dir = root / layout / "775-1"
                    if link_level == "tier":
                        run_dir.mkdir(parents=True)
                        (run_dir / "fast_sanity").symlink_to(outside)
                    elif link_level == "model":
                        (run_dir / "fast_sanity").mkdir(parents=True)
                        (run_dir / "fast_sanity/qwen3.8-27b").symlink_to(outside_model)
                    elif link_level == "index":
                        model_dir = run_dir / "fast_sanity/qwen3.8-27b"
                        model_dir.mkdir(parents=True)
                        (model_dir / "index.json").symlink_to(
                            outside_model / "index.json"
                        )
                        (model_dir / "launcher.log").write_text(
                            "ImagePullBackOff failed to pull image\n"
                        )
                    else:
                        model_dir = run_dir / "fast_sanity/qwen3.8-27b"
                        model_dir.mkdir(parents=True)
                        (model_dir / "index.json").write_text("{}")
                        (model_dir / "launcher.log").symlink_to(
                            outside_model / "launcher.log"
                        )
                    self.assertEqual(
                        recipe_cluster_diagnostics.recent_failures(root, 776), []
                    )

        for layout in ("sglang/logs/sglang-for-sail", "sglang-for-sail"):
            for non_regular in ("dir", "fifo"):
                if non_regular == "fifo" and not hasattr(os, "mkfifo"):
                    continue
                with (
                    self.subTest(layout=layout, non_regular=non_regular),
                    tempfile.TemporaryDirectory() as outer_tmp,
                ):
                    root = Path(outer_tmp) / "archive"
                    model_dir = root / layout / "775-1/fast_sanity/qwen3.8-27b"
                    model_dir.mkdir(parents=True)
                    (model_dir / "index.json").write_text("{}")
                    log_path = model_dir / "launcher.log"
                    if non_regular == "dir":
                        log_path.mkdir()
                    else:
                        os.mkfifo(log_path)
                    self.assertEqual(
                        recipe_cluster_diagnostics.recent_failures(root, 776), []
                    )

        # Transient OSError while listing a candidate parent falls back cleanly.
        with tempfile.TemporaryDirectory() as outer_tmp:
            outer = Path(outer_tmp)
            primary = outer / "external"
            fallback = outer / "retained"
            blocked_parent = primary / "sglang/logs/sglang-for-sail"
            blocked_parent.mkdir(parents=True)
            legacy_dir = fallback / "sglang-for-sail/774-1/fast_sanity/qwen3.8-27b"
            legacy_dir.mkdir(parents=True)
            (legacy_dir / "index.json").write_text("{}")
            (legacy_dir / "launcher.log").write_text("ImagePullBackOff failed\n")
            real_iterdir = Path.iterdir

            def flaky_iterdir(self_path):
                if self_path == blocked_parent:
                    raise PermissionError("transient nfs error")
                return real_iterdir(self_path)

            with patch.object(
                Path, "iterdir", autospec=True, side_effect=flaky_iterdir
            ):
                records = recipe_cluster_diagnostics.recent_failures(
                    primary, 776, fallback_root=fallback
                )
            self.assertEqual(
                records,
                [
                    {
                        "run_id": 774,
                        "tier": "fast_sanity",
                        "model": "qwen3.8-27b",
                        "codes": ["image_pull"],
                    }
                ],
            )


class RuntimeImageAndPrebuiltModeTest(unittest.TestCase):
    def test_rejects_illegal_tags_digests_and_cross_repo_injection(self):
        import recipe_image

        env = {
            "PPU_CI_K8S_RUNTIME_DIGEST": "registry.internal/team/sglang@sha256:"
            + "a" * 64,
            "PPU_CI_RECIPE_ALLOWED_IMAGE_REPOS": "registry.internal/team/",
        }
        for bad_tag in (
            "evil.example/sglang:nightly",
            "team/sglang:nightly",
            "nightly@sha256:" + "a" * 64,
            "../escape",
            "-flag",
            ".hidden",
            "has space",
            "t" * 129,
        ):
            with self.subTest(bad_tag=bad_tag):
                with self.assertRaises(ValueError):
                    recipe_image.resolve_target(env, image_tag=bad_tag)
        for bad_digest in (
            "sha256:short",
            "sha256:" + "A" * 64,
            "evil.example/sglang@sha256:" + "a" * 64,
        ):
            with self.subTest(bad_digest=bad_digest):
                with self.assertRaises(ValueError):
                    recipe_image.resolve_target(env, image_digest=bad_digest)
        with self.assertRaises(ValueError):
            recipe_image.resolve_target(
                {**env, "PPU_CI_RECIPE_IMAGE_REPO": "evil.example/sglang"},
                image_tag="nightly-20261122",
            )
        with self.assertRaises(ValueError):
            recipe_image.verify_pullable_image_id(
                "docker-pullable://evil.example/sglang@sha256:" + "b" * 64,
                env,
                expected_repo="registry.internal/team/sglang",
            )
        with self.assertRaises(ValueError):
            recipe_image.verify_pullable_image_id(
                "docker-pullable://registry.internal/team/sglang@sha256:" + "b" * 64,
                env,
                expected_repo="registry.internal/team/sglang",
                expected_digest="sha256:" + "c" * 64,
            )

    def test_resolves_tag_and_digest_and_verifies_docker_pull(self):
        import recipe_image

        digest = "sha256:" + "e" * 64
        image_id = "sha256:" + "f" * 64
        env = {
            "PPU_CI_K8S_RUNTIME_DIGEST": "registry.internal/team/sglang@sha256:"
            + "a" * 64,
            "PPU_CI_RECIPE_IMAGE_REPO": "registry.internal/team/sglang-nightly",
            "PPU_CI_RECIPE_ALLOWED_IMAGE_REPOS": "registry.internal/team/",
        }
        tag_res = recipe_image.resolve_target(env, image_tag="nightly-20261122")
        self.assertEqual(
            tag_res["pull_ref"],
            "registry.internal/team/sglang-nightly:nightly-20261122",
        )
        self.assertEqual(
            tag_res["expected_repo"], "registry.internal/team/sglang-nightly"
        )
        self.assertEqual(tag_res["expected_digest"], "")
        self.assertEqual(len(tag_res["image_key"]), 64)

        dig_res = recipe_image.resolve_target(
            env, image_tag="nightly-20261122", image_digest=digest
        )
        self.assertEqual(
            dig_res["pull_ref"],
            f"registry.internal/team/sglang-nightly@{digest}",
        )
        self.assertEqual(dig_res["expected_digest"], digest)

        pub = recipe_image.verify_pullable_image_id(
            f"docker-pullable://registry.internal/team/sglang-nightly@{digest}",
            env,
            expected_repo="registry.internal/team/sglang-nightly",
            expected_digest=digest,
        )
        self.assertEqual(pub, f"@{digest}")
        self.assertNotIn("registry.internal", pub)

        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            calls_log = root / "docker-calls.log"
            docker = bin_dir / "docker"
            docker.write_text(
                f"""#!/bin/bash
echo "$*" >> "{calls_log}"
if [[ "$1" == "pull" ]]; then
  echo "Digest: {digest}"
  exit 0
fi
if [[ "$1" == "image" && "$2" == "inspect" ]]; then
  if [[ "$*" == *"{{.Id}}"* ]]; then
    echo "{image_id}"
    exit 0
  fi
  if [[ "$*" == *"RepoDigests"* ]]; then
    echo "registry.internal/team/sglang-nightly@{digest}"
    exit 0
  fi
fi
exit 1
""",
                encoding="utf-8",
            )
            docker.chmod(0o755)
            old_path = os.environ["PATH"]
            try:
                os.environ["PATH"] = f"{bin_dir}:{old_path}"
                pulled = recipe_image.docker_pull_and_verify(
                    env,
                    root / "pull.log",
                    image_tag="nightly-20261122",
                    image_digest=digest,
                )
            finally:
                os.environ["PATH"] = old_path
            self.assertEqual(pulled["image_id"], image_id)
            self.assertEqual(pulled["public_digest"], f"@{digest}")
            self.assertIn(
                f"pull registry.internal/team/sglang-nightly@{digest}",
                calls_log.read_text(),
            )

    def test_prebuilt_image_evidence_requires_redacted_digest_and_sglang_version(self):
        base = {
            **fast_result(),
            "route": "prebuilt-image",
            "runtime_versions": {
                "sglang": "0.5.23+sail",
                "sgl_kernel": "0.3.1",
                "torch": "2.13.0",
            },
        }
        validate(base)
        with self.assertRaises(ValueError):
            validate(
                {
                    **base,
                    "runtime_image": "private.internal/team/sglang@sha256:" + "d" * 64,
                }
            )
        without_versions = {k: v for k, v in base.items() if k != "runtime_versions"}
        with self.assertRaises(ValueError):
            validate(without_versions)
        with self.assertRaises(ValueError):
            validate({**base, "runtime_versions": {"torch": "2.13.0"}})
        with self.assertRaises(ValueError):
            validate({**base, "runtime_versions": {"sglang": "/private/path"}})
        # PEP 440 local suffixes containing domains or IPv4 addresses must fail closed.
        for bad_ver in (
            "0.5.23+registry.internal",
            "0.5.23+10.1.2.3",
            "0.5.23+corp.example.com",
        ):
            with self.subTest(bad_ver=bad_ver):
                with self.assertRaises(ValueError):
                    validate({**base, "runtime_versions": {"sglang": bad_ver}})

    def test_recipe_image_does_not_fallback_to_legacy_runtime_image(self):
        import recipe_image

        legacy_only = {
            "PPU_CI_RUNTIME_IMAGE": "registry.internal/legacy/single-card:v1",
        }
        with self.assertRaises(ValueError):
            recipe_image.resolve_target(legacy_only)
        with self.assertRaises(ValueError):
            recipe_image.resolve_target(legacy_only, image_tag="nightly-20261122")

    def test_docker_pull_handles_multiple_repo_digests(self):
        import recipe_image

        older_digest = "sha256:" + "3" * 64
        pulled_digest = "sha256:" + "2" * 64
        image_id = "sha256:" + "f" * 64
        env = {
            "PPU_CI_K8S_RUNTIME_DIGEST": "registry.internal/team/sglang@sha256:"
            + "a" * 64,
            "PPU_CI_RECIPE_ALLOWED_IMAGE_REPOS": "registry.internal/team/",
        }
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            docker = bin_dir / "docker"
            docker.write_text(
                f"""#!/bin/bash
if [[ "$1" == "pull" ]]; then
  if [[ "${{OMIT_PULL_DIGEST:-0}}" != "1" ]]; then
    echo "Digest: {pulled_digest}"
  fi
  exit 0
fi
if [[ "$1" == "image" && "$2" == "inspect" ]]; then
  if [[ "$*" == *"{{.Id}}"* ]]; then
    echo "{image_id}"
    exit 0
  fi
  if [[ "$*" == *"RepoDigests"* ]]; then
    printf '%s\\n%s\\n' \\
      "registry.internal/team/sglang@{older_digest}" \\
      "registry.internal/team/sglang@{pulled_digest}"
    exit 0
  fi
fi
exit 1
""",
                encoding="utf-8",
            )
            docker.chmod(0o755)
            old_path = os.environ["PATH"]
            try:
                os.environ["PATH"] = f"{bin_dir}:{old_path}"
                # Tag override must pick freshly pulled digest, not stale first entry.
                pulled_tag = recipe_image.docker_pull_and_verify(
                    env, root / "pull-tag.log", image_tag="nightly-20261122"
                )
                self.assertEqual(pulled_tag["public_digest"], f"@{pulled_digest}")
                # Explicit digest matching second RepoDigests entry must succeed.
                pulled_pin = recipe_image.docker_pull_and_verify(
                    env,
                    root / "pull-pin.log",
                    image_tag="nightly-20261122",
                    image_digest=pulled_digest,
                )
                self.assertEqual(pulled_pin["public_digest"], f"@{pulled_digest}")
                # Ambiguous multi-RepoDigests without pull Digest line must fail closed.
                os.environ["OMIT_PULL_DIGEST"] = "1"
                with self.assertRaises(ValueError):
                    recipe_image.docker_pull_and_verify(
                        env, root / "pull-ambig.log", image_tag="nightly-20261122"
                    )
            finally:
                os.environ.pop("OMIT_PULL_DIGEST", None)
                os.environ["PATH"] = old_path

    def test_checkout_digest_rejects_mutated_or_untracked_recipe_files(self):
        with tempfile.TemporaryDirectory() as raw_root:
            repo = Path(raw_root) / "repo"
            repo.mkdir()
            policy = repo / "test/ppu/policy/models/demo.yaml"
            policy.parent.mkdir(parents=True)
            policy.write_text("max_tokens: 16\n", encoding="utf-8")
            git_env = {
                **os.environ,
                "GIT_AUTHOR_NAME": "Test",
                "GIT_AUTHOR_EMAIL": "test@example.com",
                "GIT_COMMITTER_NAME": "Test",
                "GIT_COMMITTER_EMAIL": "test@example.com",
            }
            subprocess.run(["git", "init", "-q"], cwd=repo, env=git_env, check=True)
            subprocess.run(["git", "add", "."], cwd=repo, env=git_env, check=True)
            subprocess.run(
                ["git", "commit", "-q", "-m", "init"], cwd=repo, env=git_env, check=True
            )
            sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo, text=True
            ).strip()
            bundle = REPO_ROOT / "scripts/ci/ppu/recipe_bundle.py"

            # Clean checkout matches committed git blobs.
            clean = subprocess.run(
                [sys.executable, "-B", str(bundle), "checkout-digest", sha],
                cwd=repo,
                capture_output=True,
                text=True,
            )
            self.assertEqual(clean.returncode, 0, clean.stderr)
            self.assertEqual(len(clean.stdout.strip()), 64)

            # Mutating a tracked policy file on disk must fail closed.
            policy.write_text("max_tokens: 32\n", encoding="utf-8")
            mutated = subprocess.run(
                [sys.executable, "-B", str(bundle), "checkout-digest", sha],
                cwd=repo,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(mutated.returncode, 0)

            # Restoring tracked file but adding an untracked file under test/ppu/ must fail closed.
            policy.write_text("max_tokens: 16\n", encoding="utf-8")
            (repo / "test/ppu/extra.yaml").write_text("x: 1\n", encoding="utf-8")
            untracked = subprocess.run(
                [sys.executable, "-B", str(bundle), "checkout-digest", sha],
                cwd=repo,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(untracked.returncode, 0)

    @unittest.skipUnless(
        all(shutil.which(tool) for tool in ("jq", "sha256sum")),
        "runner tools are required",
    )
    def test_cluster_transport_prebuilt_image_mode_skips_wheels_and_exports_fingerprint(
        self,
    ):
        digest = "sha256:" + "e" * 64
        tree_sha = "1" * 64
        source_sha = "a" * 40
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = root / "checkout"
            staging = root / "staging"
            retained = root / "retained"
            tools = root / "bin"
            backend = workspace / ".ppu-infra/cluster-run/scripts"
            for folder in (staging, retained, tools, backend):
                folder.mkdir(parents=True)
            scripts = workspace / "scripts/ci/ppu"
            shutil.copytree(REPO_ROOT / "scripts/ci/ppu", scripts)
            shutil.copytree(DEFAULT_POLICY_ROOT, workspace / "test/ppu/policy")
            (scripts / "recipe_bundle.py").write_text(
                f"import sys\nif sys.argv[1] == 'checkout-digest': print('{tree_sha}')\n"
            )
            summary = {
                "outcome": "pass",
                "parser_checked": True,
                "sse_verified": True,
                "cold_start_s": 3.0,
                "queue_s": 0,
                "layer_override": None,
                "config_digest": "c" * 64,
            }
            (workspace / "fixture.json").write_text(json.dumps(summary))
            (workspace / "versions.json").write_text(
                json.dumps({"sglang": "0.5.23+sail", "torch": "2.13.0"})
            )
            (tools / "git").write_text(
                '#!/bin/bash\ncase "$1" in -C) echo 96e25147971f5d2a04d828f8ad508cae38a4fa8a;; '
                f"*) echo {source_sha};; esac\n"
            )
            kube_content = json.dumps(
                {
                    "status": {
                        "containerStatuses": [
                            {
                                "name": "worker",
                                "imageID": f"docker-pullable://registry.internal/team/sglang@{digest}",
                            }
                        ]
                    }
                }
            )
            (tools / "kubectl").write_text(
                f"#!/bin/bash\nif [[ \"$*\" == *'get pod '* ]]; then echo '{kube_content}'; fi\n"
            )
            (tools / "timeout").write_text('#!/bin/bash\nshift\nexec "$@"\n')
            for tool in tools.iterdir():
                tool.chmod(0o700)
            (backend / "submit.sh").write_text(
                f'test "$INPUT_IMAGE" = "registry.internal/team/sglang@{digest}" || exit 1\n'
                'jq -e \'.PPU_CI_SKIP_WHEEL_INSTALL == "true" and .PPU_CI_RECIPE_TREE_SHA256 == "'
                + tree_sha
                + '"\' <<< "$USER_ENV_JSON" >/dev/null || exit 1\n'
                'printf "job_status=scheduled\\nstart_time=1\\ndeadline=2\\nqueue_duration_seconds=5\\n" >> "$GITHUB_OUTPUT"\n'
            )
            (backend / "monitor.sh").write_text(
                'mkdir -p "$PPU_CI_WORKER_ARTIFACT_DIR/results"\n'
                'cp fixture.json "$PPU_CI_WORKER_ARTIFACT_DIR/results/summary.json"\n'
                'cp versions.json "$PPU_CI_WORKER_ARTIFACT_DIR/runtime-versions.json"\n'
                'echo job_status=succeeded >> "$GITHUB_OUTPUT"\n'
            )
            (backend / "cleanup.sh").write_text(
                'unlink "$SOURCE_STAGE_DIR"\nrmdir "$(dirname "$SOURCE_STAGE_DIR")"\n'
            )
            export_env = {
                **os.environ,
                "PATH": os.pathsep.join(
                    (
                        str(tools),
                        str(Path(sys.executable).parent),
                        os.environ["PATH"],
                    )
                ),
                "TIER": "fast_sanity",
                "MODEL_KEY": "qwen3.8-27b",
                "RECIPE_PPU_COUNT": "2",
                "GITHUB_OUTPUT": str(root / "gha-output"),
                "PPU_PUBLIC_SUMMARY": str(root / "gha-summary"),
                "GITHUB_WORKSPACE": str(workspace),
                "RUNNER_TEMP": str(root / "runner-temp"),
                "GITHUB_RUN_ID": "99",
                "GITHUB_RUN_ATTEMPT": "1",
                "PPU_CI_K8S_CONFIG_SHA256": "",
                "PPU_CI_K8S_LOADED_CONFIG_SHA256": "",
                "PPU_CI_PRIVATE_RESULTS_ROOT": str(retained),
                "PPU_CI_K8S_PRIVATE_WORKER_ROOT": str(retained),
                "PPU_CI_K8S_STAGE_ROOT": str(staging),
                "PPU_CI_SOURCE_SHA": source_sha,
                "PPU_CI_JIT_ROOT": str(root / "jit"),
                "PPU_CI_HF_CACHE_ROOT": str(root / "cache"),
                "PPU_CI_K8S_RUNTIME_DIGEST": "registry.internal/team/sglang@sha256:"
                + "a" * 64,
                "PPU_CI_RUNTIME_IMAGE_TAG": "nightly-20261122",
                "PPU_CI_RUNTIME_IMAGE_DIGEST": digest,
                "PPU_CI_SKIP_WHEEL_INSTALL": "true",
            }
            process = subprocess.run(
                ["bash", str(scripts / "run_recipe_cluster.sh")],
                cwd=workspace,
                env=export_env,
                capture_output=True,
                text=True,
                timeout=20,
            )
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            public_record = json.loads(
                (workspace / "ppu-public/recipe.json").read_text()
            )
            self.assertEqual(public_record["route"], "prebuilt-image")
            self.assertEqual(public_record["runtime_image"], f"@{digest}")
            self.assertEqual(
                public_record["runtime_versions"],
                {"sglang": "0.5.23+sail", "torch": "2.13.0"},
            )
            self.assertNotIn(
                "registry.internal",
                (workspace / "ppu-public/recipe.json").read_text()
                + (root / "gha-summary").read_text()
                + process.stdout
                + process.stderr,
            )

    def test_nightly_workflow_has_no_plaintext_runtime_image_and_exposes_safe_inputs(
        self,
    ):
        import yaml

        path = REPO_ROOT / ".github/workflows/nightly-test-ppu.yml"
        raw = path.read_text(encoding="utf-8")
        self.assertNotIn("docker_" + "test/sglang", raw)
        self.assertNotIn("runtime_image:", raw)
        workflow = yaml.load(raw, Loader=yaml.BaseLoader)
        dispatch_inputs = workflow["on"]["workflow_dispatch"]["inputs"]
        for key in (
            "target_ref",
            "model_filter",
            "image_tag",
            "image_digest",
            "skip_wheel_build",
        ):
            self.assertIn(key, dispatch_inputs)
            default = str(dispatch_inputs[key].get("default", ""))
            self.assertNotIn(":", default)
            self.assertNotIn("docker_", default)
        call_inputs = workflow["on"]["workflow_call"]["inputs"]
        for key in (
            "ref",
            "target_ref",
            "model_filter",
            "image_tag",
            "image_digest",
            "skip_wheel_build",
        ):
            self.assertIn(key, call_inputs)
            default = str(call_inputs[key].get("default", ""))
            self.assertNotIn(":", default)
            self.assertNotIn("docker_", default)
        expected_secrets = (
            "PPU_CI_K8S_RUNTIME_DIGEST",
            "PPU_CI_RECIPE_IMAGE_REPO",
            "PPU_CI_RECIPE_ALLOWED_IMAGE_REPOS",
        )
        call_secrets = workflow["on"]["workflow_call"]["secrets"]
        recipe_env = workflow["jobs"]["recipe-sanity"]["env"]
        pr_workflow = yaml.load(
            (REPO_ROOT / ".github/workflows/pr-test-ppu.yml").read_text(
                encoding="utf-8"
            ),
            Loader=yaml.BaseLoader,
        )
        pr_fast_secrets = pr_workflow["jobs"]["ppu-fast-sanity"]["secrets"]
        for secret_key in expected_secrets:
            self.assertEqual(
                str(call_secrets[secret_key].get("required", "")).lower(), "false"
            )
            expected_expr = f"${{{{ secrets.{secret_key} }}}}"
            self.assertEqual(recipe_env[secret_key], expected_expr)
            self.assertEqual(pr_fast_secrets[secret_key], expected_expr)

    def test_nightly_model_sanity_throttles_concurrency(self):
        import yaml

        workflow = yaml.load(
            (REPO_ROOT / ".github/workflows/nightly-test-ppu.yml").read_text(
                encoding="utf-8"
            ),
            Loader=yaml.BaseLoader,
        )
        strategy = workflow["jobs"]["recipe-sanity"]["strategy"]
        self.assertEqual(
            strategy["max-parallel"],
            "${{ inputs.tier == 'fast_sanity' && 32 || 4 }}",
        )


def _sample_sglang_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sglang", allow_abbrev=False)
    for flag in (
        "--model-path",
        "--tokenizer-path",
        "--served-model-name",
        "--host",
        "--port",
        "--tp-size",
        "--tensor-parallel-size",
        "--dp-size",
        "--data-parallel-size",
        "--pp-size",
        "--pipeline-parallel-size",
        "--ep-size",
        "--expert-parallel-size",
        "--ep",
        "--attn-dp-size",
        "--attention-data-parallel-size",
        "--attn-cp-size",
        "--attention-context-parallel-size",
        "--moe-dense-tp-size",
        "--page-size",
        "--mem-fraction-static",
        "--chunked-prefill-size",
        "--max-prefill-tokens",
        "--max-running-requests",
        "--context-length",
        "--cuda-graph-max-bs-decode",
        "--dtype",
        "--kv-cache-dtype",
        "--attention-backend",
        "--prefill-attention-backend",
        "--decode-attention-backend",
        "--dsa-prefill-backend",
        "--dsa-decode-backend",
        "--dsv4-prefill-backend",
        "--moe-a2a-backend",
        "--deepep-mode",
        "--moe-runner-backend",
        "--tool-call-parser",
        "--reasoning-parser",
        "--speculative-algorithm",
        "--speculative-num-steps",
        "--speculative-eagle-topk",
        "--speculative-num-draft-tokens",
        "--speculative-draft-model-quantization",
        "--linear-attn-prefill-backend",
        "--linear-attn-decode-backend",
        "--linear-attn-verify-backend",
        "--mamba-ssm-dtype",
        "--mamba-radix-cache-strategy",
        "--nnodes",
        "--node-rank",
        "--dist-init-addr",
        "--nccl-init-addr",
        "--dist-timeout",
        "--watchdog-timeout",
        "--disaggregation-mode",
        "--disaggregation-transfer-backend",
        "--disaggregation-ib-device",
        "--disaggregation-bootstrap-port",
        "--load-format",
        "--json-model-override-args",
        "--cp-strategy",
        "--schedule-policy",
        "--load-balance-method",
    ):
        parser.add_argument(flag)
    for bflag in (
        "--trust-remote-code",
        "--no-trust-remote-code",
        "--disable-custom-all-reduce",
        "--disable-shared-experts-fusion",
        "--disable-prefill-cuda-graph",
        "--disable-overlap-schedule",
        "--disable-radix-cache",
        "--enable-metrics",
        "--enable-dp-lm-head",
        "--ple-offload-embedding",
        "--enable-prefill-cp",
        "--enable-deepseek-v4-fp4-indexer",
    ):
        parser.add_argument(bflag, action="store_true")
    return parser


class PresetCatalogContractTest(unittest.TestCase):
    def test_authoring_guide_and_catalog_render_stay_in_sync(self):
        self.assertTrue(DEFAULT_AUTHORING_DOC.is_file())
        authoring = DEFAULT_AUTHORING_DOC.read_text(encoding="utf-8")
        self.assertEqual(_scan_text(authoring, context="test/ppu/AUTHORING.md"), [])
        for required_token in (
            "[README.md](README.md)",
            "`zhenwu_m890`",
            "`zhenwu_810e`",
            "`prefill`",
            "`decode`",
            "`decode_ep`",
            "`pd_prefill` + `pd_decode`",
            "`{master_addr}`",
            "`{master_port}`",
            "`{node_rank}`",
            "`{kv_ip}`",
            "`{kv_port}`",
            "`{ib_devices}`",
            "`SGLANG_HOST_IP`",
        ):
            self.assertIn(required_token, authoring)

        matrix = render_ci_matrix_markdown()
        self.assertIn("[AUTHORING.md](AUTHORING.md)", matrix)
        self.assertIn(
            "## Registered Serving Presets (Benchmark & Disagg Catalog)", matrix
        )
        _, catalog_section = matrix.split(
            "## Registered Serving Presets (Benchmark & Disagg Catalog)", 1
        )
        self.assertIn("`Qwen3.8-27B.yaml`", catalog_section)
        self.assertIn("`zhenwu_810e` / 2 / `single_node`", catalog_section)
        self.assertIn("`fp8_channel` (`fp8`)", catalog_section)
        self.assertIn("`w8a8_int8` (`w8a8_int8`)", catalog_section)
        self.assertIn("`multi_node_ep` (4x8)", catalog_section)
        self.assertIn("`pd_disagg` (prefill)", catalog_section)
        self.assertIn("`pd_disagg` (decode)", catalog_section)
        self.assertEqual(ppu_recipe_main(["render-matrix", "--check"]), 0)

    def test_glm53_reference_exemplar_expands_all_variants_and_four_scenarios(self):
        glm_path = REPO_ROOT / "test/ppu/recipes/zai-org/GLM-5.3.yaml"
        entries = iter_recipe_catalog_entries(glm_path)
        self.assertEqual(
            [(e["variant"], e["preset"], e["hardware"]) for e in entries],
            [
                ("default", None, "zhenwu_m890"),
                ("default", "prefill", "zhenwu_m890"),
                ("default", "decode", "zhenwu_m890"),
                ("default", "decode_ep", "zhenwu_m890"),
                ("default", "pd_prefill", "zhenwu_m890"),
                ("default", "pd_decode", "zhenwu_m890"),
                ("fp8_channel", None, "zhenwu_m890"),
                ("fp8_channel", "prefill", "zhenwu_m890"),
                ("fp8_channel", "decode", "zhenwu_m890"),
                ("fp8_channel", "decode_ep", "zhenwu_m890"),
                ("fp8_channel", "pd_prefill", "zhenwu_m890"),
                ("fp8_channel", "pd_decode", "zhenwu_m890"),
                ("w8a8_int8", None, "zhenwu_810e"),
                ("w8a8_int8", "prefill", "zhenwu_810e"),
            ],
        )
        report = validate_paths(
            [glm_path],
            sglang_parser_mode="required",
            parser=_sample_sglang_parser(),
        )
        self.assertTrue(report["parser_checked"])
        self.assertEqual(report["checked_combinations"], 14)

        fp8 = resolve_recipe(glm_path, variant="fp8_channel", preset="decode")
        self.assertEqual(fp8["hardware"], "zhenwu_m890")
        self.assertEqual(fp8["ppu_count"], 8)
        self.assertEqual(
            fp8["hf_model_id"],
            "T-HEAD/GLM-5.3-Quant-W-FP8-PerChannel-A-FP8-PerChannel",
        )
        self.assertEqual(fp8["golden_model_key"], "glm_5_3_fp8_channel")
        self.assertEqual(fp8["served_name"], "glm-5.3-fp8-channel")
        self.assertEqual(
            fp8["enabled_features"], ["tool_calling", "reasoning", "spec_decoding"]
        )

        int8 = resolve_recipe(glm_path, variant="w8a8_int8", preset="prefill")
        self.assertEqual(int8["hardware"], "zhenwu_810e")
        self.assertEqual(int8["ppu_count"], 16)
        self.assertEqual(
            int8["hf_model_id"],
            "T-HEAD/GLM-5.3-Quant-W-INT8-PerChannel-A-INT8-PerToken",
        )
        self.assertEqual(int8["golden_model_key"], "glm_5_3_w8a8_int8")
        self.assertEqual(
            int8["flags"][int8["flags"].index("--mem-fraction-static") + 1], "0.90"
        )
        self.assertNotIn("--speculative-algorithm", int8["flags"])

        # Missing or empty variants on sglang recipes must fail closed.
        glm_doc = _load_yaml(glm_path)
        for bad_variants in ({}, None):
            bad = copy.deepcopy(glm_doc)
            bad["presets"] = {}
            if bad_variants is None:
                bad.pop("variants", None)
            else:
                bad["variants"] = bad_variants
            with self.assertRaisesRegex(
                ValidationError, "variants must be a non-empty mapping"
            ):
                validate_serving_recipe_structure(bad, context="bad_variants")

    def test_hardware_precision_binding_and_portable_bf16(self):
        qwen_path = REPO_ROOT / "test/ppu/recipes/Qwen/Qwen3.8-27B.yaml"
        qwen_doc = _load_yaml(qwen_path)
        qwen_entries = iter_recipe_catalog_entries(qwen_path, qwen_doc)
        self.assertEqual(
            [(e["variant"], e["preset"], e["hardware"]) for e in qwen_entries],
            [
                ("default", None, "zhenwu_m890"),
                ("default", None, "zhenwu_810e"),
            ],
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copytree(REPO_ROOT / "test/ppu", root / "test/ppu")
            small_bf16 = copy.deepcopy(qwen_doc)
            small_bf16["hardware_overrides"] = {
                "zhenwu_810e": {"extra_args": ["--max-running-requests", "64"]}
            }
            small_bf16["presets"] = {
                "prefill": {
                    "hardware": "zhenwu_m890",
                    "args": ["--chunked-prefill-size", "8192"],
                },
                "decode": {
                    "hardware": "zhenwu_810e",
                    "args": ["--chunked-prefill-size", "2048"],
                },
            }
            rec_path = root / "test/ppu/recipes/Qwen/Qwen3.8-27B.yaml"
            rec_path.write_text(yaml.safe_dump(small_bf16, sort_keys=False))

            with (
                patch("ppu_recipe.REPO_ROOT", root),
                patch("recipe_select.REPO_ROOT", root),
            ):
                report = validate_paths(
                    [rec_path],
                    sglang_parser_mode="required",
                    parser=_sample_sglang_parser(),
                )
                self.assertEqual(report["checked_combinations"], 4)
                decode_810e = resolve_recipe(rec_path, preset="decode")
                self.assertEqual(decode_810e["hardware"], "zhenwu_810e")
                self.assertIn("--max-running-requests", decode_810e["flags"])

                pol_810e = _load_yaml(root / "test/ppu/policy/models/qwen3.8-27b.yaml")
                pol_810e["hardware"] = "zhenwu_810e"
                pol_810e["tiers"]["fast_sanity"]["preset"] = "decode"
                (root / "test/ppu/policy/models/qwen3.8-27b-810e.yaml").write_text(
                    yaml.safe_dump(pol_810e, sort_keys=False)
                )
                validate_policy_structure(pol_810e, context="pol_810e", base_dir=root)
                selected = {m["model"] for m in select("fast_sanity", "")["include"]}
                self.assertIn("qwen3.8-27b", selected)
                self.assertNotIn("qwen3.8-27b-810e", selected)
                with self.assertRaisesRegex(
                    ValueError, "targets hardware 'zhenwu_810e'"
                ):
                    select("fast_sanity", "qwen3.8-27b-810e")

            # Cross-binding M890 with int8 or 810E with mxfp4/fp8 must fail closed.
            for hw, bad_prec in (
                ("zhenwu_m890", "w8a8_int8"),
                ("zhenwu_810e", "mxfp4"),
                ("zhenwu_810e", "fp8"),
            ):
                bad_var = copy.deepcopy(qwen_doc)
                bad_var["variants"]["default"] = {
                    "precision": bad_prec,
                    "hardware": hw,
                }
                with self.assertRaisesRegex(ValidationError, "only allows bf16 or"):
                    validate_serving_recipe_structure(bad_var, context="cross_bind")

            # Explicitly pinned BF16 variant rejects conflicting preset/policy hardware.
            pinned_m890 = copy.deepcopy(small_bf16)
            pinned_m890["variants"]["default"]["hardware"] = "zhenwu_m890"
            with self.assertRaisesRegex(
                ValidationError, "matches no compatible variant"
            ):
                validate_serving_recipe_structure(pinned_m890, context="orphan_810e")

    def test_policy_model_id_spoofing_variant_selection_and_shared_stage_presets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copytree(REPO_ROOT / "test/ppu", root / "test/ppu")
            with (
                patch("ppu_recipe.REPO_ROOT", root),
                patch("recipe_select.REPO_ROOT", root),
            ):
                spoof = _load_yaml(root / "test/ppu/policy/models/qwen3.8-27b.yaml")
                spoof["model_id"] = "zai-org/GLM-5.3"
                spoof_path = root / "test/ppu/policy/models/spoof-qwen.yaml"
                spoof_path.write_text(yaml.safe_dump(spoof, sort_keys=False))
                with self.assertRaisesRegex(
                    ValidationError, "policy model_id 'zai-org/GLM-5.3' does not match"
                ):
                    validate_policy_structure(spoof, context="spoof", base_dir=root)
                with self.assertRaisesRegex(
                    ValidationError, "policy model_id 'zai-org/GLM-5.3' does not match"
                ):
                    select("fast_sanity", "")
                spoof_path.unlink()

                # Positive control: policy selecting fp8_channel variant on GLM-5.3.yaml.
                fp8_pol = _load_yaml(root / "test/ppu/policy/models/glm-5.3.yaml")
                fp8_pol["variant"] = "fp8_channel"
                fp8_pol["tiers"] = {"fast_sanity": fp8_pol["tiers"]["fast_sanity"]}
                fp8_pol_path = root / "test/ppu/policy/models/glm-5.3-fp8.yaml"
                fp8_pol_path.write_text(yaml.safe_dump(fp8_pol, sort_keys=False))
                validate_policy_structure(fp8_pol, context="fp8_pol", base_dir=root)
                fp8_res = resolve_recipe(
                    root / "test/ppu/recipes/zai-org/GLM-5.3.yaml",
                    policy_path=fp8_pol_path,
                    tier="fast_sanity",
                )
                self.assertEqual(fp8_res["model_id"], "zai-org/GLM-5.3")
                self.assertEqual(
                    fp8_res["hf_model_id"],
                    "T-HEAD/GLM-5.3-Quant-W-FP8-PerChannel-A-FP8-PerChannel",
                )

                # Shared-server model_sanity stages must agree on preset.
                bad_ms = _load_yaml(
                    root / "test/ppu/policy/models/deepseek-v4-flash-tp8.yaml"
                )
                bad_ms["tiers"]["model_sanity"]["stages"][1].pop("preset", None)
                with self.assertRaisesRegex(
                    ValidationError, "all shared-server stages must agree on preset"
                ):
                    validate_policy_structure(bad_ms, context="bad_ms", base_dir=root)

    def test_recipe_boundary_and_redaction_counterexamples(self):
        base = _load_yaml(REPO_ROOT / "test/ppu/recipes/Qwen/Qwen3.8-27B.yaml")

        for bad_args, pattern in (
            (["--disable-radix-cache"], "framework runner owns --disable-radix-cache"),
            (["--enable-metrics"], "framework runner owns --enable-metrics"),
            (["--load-format", "dummy"], "framework runner owns --load-format"),
            (
                ["--model-path", "Qwen/Qwen3.8-27B"],
                "framework runner owns --model-path",
            ),
            (["--tp_size", "2"], "canonical hyphenated spelling"),
            (["--enable-dp-attention"], "deprecated flag --enable-dp-attention"),
            (["--mem-fraction-static", "0"], "--mem-fraction-static must be in"),
            (["--mem-fraction-static", "1.2"], "--mem-fraction-static must be in"),
            (["--mem-fraction-static", "nan"], "--mem-fraction-static must be in"),
            (
                ["--json-model-override-args", '{"num_hidden_layers": 2}'],
                "layer override must come from the test policy",
            ),
            (
                ["--disaggregation-ib-device", "mlx5_" + "bond_0"],
                "matched nic_device",
            ),
        ):
            with self.subTest(bad_args=bad_args):
                doc = copy.deepcopy(base)
                doc["model"]["base_args"] = [*doc["model"]["base_args"], *bad_args]
                with self.assertRaisesRegex(ValidationError, pattern):
                    validate_serving_recipe_structure(doc, context="bad_flag")

        for bad_env, pattern in (
            ({"SGLANG_USE_MODELSCOPE": "1"}, "must not be set in recipe"),
            ({"SGLANG_WARMUP_TIMEOUT": "3600"}, "must not be set in recipe"),
            ({"DG_USE_MOE_DYNAMIC_TILE": "1"}, "must be listed in ppu.external_env"),
            ({"SGLANG_MODEL_DIR": "/" + "cpfs/models/demo"}, "matched local_path"),
        ):
            with self.subTest(bad_env=bad_env):
                doc = copy.deepcopy(base)
                doc["model"]["base_env"] = bad_env
                with self.assertRaisesRegex(ValidationError, pattern):
                    validate_serving_recipe_structure(doc, context="bad_env")

    def test_sglang_multinode_ep_and_parallelism_invariants(self):
        glm_path = REPO_ROOT / "test/ppu/recipes/zai-org/GLM-5.3.yaml"
        glm_doc = _load_yaml(glm_path)

        test_ip_a = ".".join(("198", "51", "100", "10"))
        ep_res = resolve_recipe(glm_path, preset="decode_ep")
        self.assertEqual(ep_res["ppu_count"], 32)
        sub_flags, _ = substitute_validation_placeholders(
            ep_res["flags"],
            ep_res["server_env"],
            values={"master_addr": test_ip_a, "master_port": 29600, "node_rank": 3},
        )
        self.assertEqual(
            sub_flags[sub_flags.index("--dist-init-addr") + 1], f"{test_ip_a}:29600"
        )
        self.assertEqual(sub_flags[sub_flags.index("--node-rank") + 1], "3")
        validate_flags_with_parser(
            _sample_sglang_parser(), sub_flags, context="decode_ep_nondefault"
        )

        ipv6_doc = copy.deepcopy(glm_doc)
        ep_args = ipv6_doc["presets"]["decode_ep"]["args"]
        ep_args[ep_args.index("--dist-init-addr") + 1] = "[{master_addr}]:{master_port}"
        validate_serving_recipe_structure(ipv6_doc, context="ipv6_dist_init")

        # Counterexample 1: single_node smuggling multi-node or disagg flags.
        for bad_single_args, pat in (
            (["--nnodes", "4", "--node-rank", "0"], "must not set multi-node flag"),
            (["--disaggregation-mode", "prefill"], "must not set disaggregation flag"),
            (["--kv-transfer-config", "{}"], "must not set disaggregation flag"),
        ):
            bad = copy.deepcopy(glm_doc)
            bad["presets"]["decode"]["args"] = bad_single_args
            with self.assertRaisesRegex(ValidationError, pat):
                validate_serving_recipe_structure(bad, context="bad_single")

        # Counterexample 2: omitting --tp-size 32 on 4x8 decode_ep yields tp=8 != 32.
        no_tp32 = copy.deepcopy(glm_doc)
        no_tp32["presets"]["decode_ep"].pop("tp", None)
        with self.assertRaisesRegex(ValidationError, "must be divisible|must match"):
            validate_serving_recipe_structure(no_tp32, context="no_tp32")

        # Counterexample 3: multi_node_ep without expert parallelism fails closed.
        no_ep = copy.deepcopy(glm_doc)
        ep_a = no_ep["presets"]["decode_ep"]["args"]
        idx_a2a = ep_a.index("--moe-a2a-backend")
        ep_a[idx_a2a + 1] = "none"
        with self.assertRaisesRegex(
            ValidationError, "requires effective expert parallelism"
        ):
            validate_serving_recipe_structure(no_ep, context="no_ep")

        # Counterexample 4: malformed --dist-init-addr (bare host, scheme, path).
        for bad_addr in (
            "{master_addr}",
            "http://{master_addr}:{master_port}",
            "{master_addr}:{master_port}/init",
        ):
            bad = copy.deepcopy(glm_doc)
            args = bad["presets"]["decode_ep"]["args"]
            args[args.index("--dist-init-addr") + 1] = bad_addr
            with self.assertRaisesRegex(
                ValidationError, "missing required placeholder|--dist-init-addr"
            ):
                validate_serving_recipe_structure(bad, context="bad_dist_addr")

        # Counterexample 5: dp_size > 1 with attn_dp_size > 1 or multi-node.
        bad_dp = copy.deepcopy(glm_doc)
        bad_dp["presets"]["decode"]["args"] = [
            "--dp-size",
            "2",
            "--attn-dp-size",
            "2",
        ]
        bad_dp["presets"]["decode"]["topology"]["devices_per_node"] = 16
        with self.assertRaisesRegex(
            ValidationError, "--dp-size .* and --attn-dp-size .* cannot both"
        ):
            validate_serving_recipe_structure(bad_dp, context="bad_dp_attn_dp")

    def test_sglang_mooncake_pd_disagg_and_pr397_1p1d_shape(self):
        glm_path = REPO_ROOT / "test/ppu/recipes/zai-org/GLM-5.3.yaml"
        glm_doc = _load_yaml(glm_path)

        # 1. Non-default Mooncake endpoint substitution on GLM-5.3 pd_prefill / pd_decode.
        kv_ip_a = ".".join(("198", "51", "100", "10"))
        kv_ip_b = ".".join(("198", "51", "100", "20"))
        prefill = resolve_recipe(glm_path, preset="pd_prefill")
        p_flags, p_env = substitute_validation_placeholders(
            prefill["flags"],
            prefill["server_env"],
            values={
                "kv_ip": kv_ip_a,
                "kv_port": 18998,
                "ib_devices": "rdma0,rdma1,rdma2,rdma3",
            },
        )
        self.assertEqual(p_env["SGLANG_HOST_IP"], kv_ip_a)
        self.assertEqual(p_flags[p_flags.index("--disaggregation-mode") + 1], "prefill")
        self.assertEqual(
            p_flags[p_flags.index("--disaggregation-ib-device") + 1],
            "rdma0,rdma1,rdma2,rdma3",
        )
        self.assertEqual(
            int(p_flags[p_flags.index("--disaggregation-bootstrap-port") + 1]), 18998
        )

        decode = resolve_recipe(glm_path, preset="pd_decode")
        d_flags, d_env = substitute_validation_placeholders(
            decode["flags"],
            decode["server_env"],
            values={"kv_ip": kv_ip_b, "ib_devices": "rdma4,rdma5"},
        )
        self.assertEqual(d_env["SGLANG_HOST_IP"], kv_ip_b)
        self.assertEqual(d_flags[d_flags.index("--disaggregation-mode") + 1], "decode")
        self.assertNotIn("--disaggregation-bootstrap-port", d_flags)

        # 2. Positive control for PR #397 migration shape: DeepSeek-V4.1-Flash 4P + 4D 1P1D.
        dsv41_doc = _load_yaml(
            REPO_ROOT / "test/ppu/recipes/deepseek-ai/DeepSeek-V4.1-Flash.yaml"
        )
        dsv41_1p1d = copy.deepcopy(dsv41_doc)
        dsv41_1p1d["presets"] = {
            "pd_prefill": {
                "hardware": "zhenwu_m890",
                "topology": {
                    "kind": "pd_disagg",
                    "nodes": 1,
                    "devices_per_node": 4,
                    "role": "prefill",
                },
                "tp": 4,
                "args": [
                    "--enable-prefill-cp",
                    "--cp-strategy",
                    "interleave",
                    "--attn-cp-size",
                    "4",
                    "--moe-a2a-backend",
                    "megamoe",
                    "--moe-dense-tp-size",
                    "1",
                    "--disaggregation-mode",
                    "prefill",
                    "--disaggregation-transfer-backend",
                    "mooncake",
                    "--disaggregation-ib-device",
                    "{ib_devices}",
                    "--disaggregation-bootstrap-port",
                    "{kv_port}",
                    "--json-model-override-args",
                    '{"vision_n_layers":0}',
                ],
                "env": {
                    "MC_NUM_QP_PER_EP": "4",
                    "MOONCAKE_PROTOCOL": "rdma",
                    "SGLANG_HOST_IP": "{kv_ip}",
                },
            },
            "pd_decode": {
                "hardware": "zhenwu_m890",
                "topology": {
                    "kind": "pd_disagg",
                    "nodes": 1,
                    "devices_per_node": 4,
                    "role": "decode",
                },
                "tp": 4,
                "args": [
                    "--attn-dp-size",
                    "4",
                    "--enable-dp-lm-head",
                    "--moe-a2a-backend",
                    "megamoe",
                    "--moe-dense-tp-size",
                    "1",
                    "--disaggregation-mode",
                    "decode",
                    "--disaggregation-transfer-backend",
                    "mooncake",
                    "--disaggregation-ib-device",
                    "{ib_devices}",
                    "--json-model-override-args",
                    '{"vision_n_layers":0}',
                ],
                "env": {
                    "MC_NUM_QP_PER_EP": "4",
                    "MOONCAKE_PROTOCOL": "rdma",
                    "SGLANG_HOST_IP": "{kv_ip}",
                },
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            rec_file = Path(tmp) / "DeepSeek-V4.1-Flash-1P1D.yaml"
            rec_file.write_text(yaml.safe_dump(dsv41_1p1d, sort_keys=False))
            rep = validate_paths(
                [rec_file],
                sglang_parser_mode="required",
                parser=_sample_sglang_parser(),
            )
            self.assertEqual(rep["checked_combinations"], 3)
            self.assertEqual(
                resolve_recipe(rec_file, preset="pd_prefill")["ppu_count"], 4
            )
            self.assertEqual(
                resolve_recipe(rec_file, preset="pd_decode")["ppu_count"], 4
            )

        # 3. PD counterexamples: unpaired slot, vLLM dead wiring, decode bootstrap port, role mismatch.
        unpaired = copy.deepcopy(glm_doc)
        unpaired["presets"].pop("pd_decode")
        with self.assertRaisesRegex(ValidationError, "must be declared as a pair"):
            validate_serving_recipe_structure(unpaired, context="unpaired")

        vllm_env = copy.deepcopy(glm_doc)
        vllm_env["presets"]["pd_decode"]["env"]["VLLM_HOST_IP"] = "{kv_ip}"
        with self.assertRaisesRegex(ValidationError, "does not read VLLM_HOST_IP"):
            validate_serving_recipe_structure(vllm_env, context="vllm_env")

        decode_port = copy.deepcopy(glm_doc)
        decode_port["presets"]["pd_decode"]["args"].extend(
            ["--disaggregation-bootstrap-port", "{kv_port}"]
        )
        with self.assertRaisesRegex(
            ValidationError, "pd_decode must not set --disaggregation-bootstrap-port"
        ):
            validate_serving_recipe_structure(decode_port, context="decode_port")

        wrong_role = copy.deepcopy(glm_doc)
        d_args = wrong_role["presets"]["pd_decode"]["args"]
        d_args[d_args.index("--disaggregation-mode") + 1] = "prefill"
        with self.assertRaisesRegex(
            ValidationError, "requires --disaggregation-mode='decode'"
        ):
            validate_serving_recipe_structure(wrong_role, context="wrong_role")

    def test_single_node_selector_skips_multinode_pd_810e_and_oversized_policies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copytree(REPO_ROOT / "test/ppu", root / "test/ppu")
            base_pol = _load_yaml(root / "test/ppu/policy/models/glm-5.3.yaml")
            base_pol["tiers"] = {"fast_sanity": base_pol["tiers"]["fast_sanity"]}

            # Add 16-card M890 preset to GLM-5.3.yaml for runner-cap testing.
            glm_rec_path = root / "test/ppu/recipes/zai-org/GLM-5.3.yaml"
            glm_doc = _load_yaml(glm_rec_path)
            glm_doc["presets"]["tp8"] = {
                "hardware": "zhenwu_m890",
                "topology": {"kind": "single_node", "nodes": 1, "devices_per_node": 16},
                "tp": 16,
            }
            glm_rec_path.write_text(yaml.safe_dump(glm_doc, sort_keys=False))

            extra_policies = {
                "glm-5.3-decode-ep": {"preset": "decode_ep"},
                "glm-5.3-pd-prefill": {"preset": "pd_prefill"},
                "glm-5.3-810e": {
                    "hardware": "zhenwu_810e",
                    "variant": "w8a8_int8",
                    "preset": "prefill",
                },
                "glm-5.3-tp16": {"preset": "tp8"},
            }
            for slug, overrides in extra_policies.items():
                pol = copy.deepcopy(base_pol)
                if "hardware" in overrides:
                    pol["hardware"] = overrides["hardware"]
                if "variant" in overrides:
                    pol["variant"] = overrides["variant"]
                pol["tiers"]["fast_sanity"]["preset"] = overrides["preset"]
                (root / f"test/ppu/policy/models/{slug}.yaml").write_text(
                    yaml.safe_dump(pol, sort_keys=False)
                )

            with (
                patch("ppu_recipe.REPO_ROOT", root),
                patch("recipe_select.REPO_ROOT", root),
            ):
                auto_selected = select("fast_sanity", "")["include"]
                self.assertEqual(len(auto_selected), 13)
                auto_keys = {item["model"] for item in auto_selected}
                for slug in extra_policies:
                    self.assertNotIn(slug, auto_keys)

                with self.assertRaisesRegex(
                    ValueError, "requires topology 'multi_node_ep'"
                ):
                    select("fast_sanity", "glm-5.3-decode-ep")
                with self.assertRaisesRegex(
                    ValueError, "requires topology 'pd_disagg'"
                ):
                    select("fast_sanity", "glm-5.3-pd-prefill")
                with self.assertRaisesRegex(
                    ValueError, "targets hardware 'zhenwu_810e'"
                ):
                    select("fast_sanity", "glm-5.3-810e")
                with self.assertRaisesRegex(
                    ValueError, "requires 16 devices, exceeding single-node"
                ):
                    select("fast_sanity", "glm-5.3-tp16")

                admitted = select("fast_sanity", "glm-5.3-tp16", max_devices=16)[
                    "include"
                ]
                self.assertEqual(admitted, [{"model": "glm-5.3-tp16", "ppu_count": 16}])

    def test_runner_binds_effective_variant_checkpoint_assets_and_revision(self):
        rev_mxfp4 = "a" * 40
        rev_fp8 = "b" * 40
        base_glm = _load_yaml(REPO_ROOT / "test/ppu/recipes/zai-org/GLM-5.3.yaml")
        mxfp4_id = base_glm["ppu"]["hf_model_id"]
        fp8_id = base_glm["variants"]["fp8_channel"]["hf_model_id"]
        mxfp4_asset = base_glm["ppu"]["assets"]["model"]
        fp8_asset = base_glm["variants"]["fp8_channel"]["assets"]["model"]
        fp8_golden = base_glm["variants"]["fp8_channel"]["golden_model_key"]

        def _make_snapshot(cache_root: Path, repo_id: str, revision: str) -> Path:
            folder = f"models--{repo_id.replace('/', '--')}"
            repo_dir = cache_root / folder
            (repo_dir / "refs").mkdir(parents=True, exist_ok=True)
            (repo_dir / "refs/main").write_text(revision)
            snap = repo_dir / "snapshots" / revision
            snap.mkdir(parents=True, exist_ok=True)
            for name in ("config.json", "tokenizer.json"):
                (snap / name).write_text("{}")
            (snap / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"w": "weight.safetensors"}})
            )
            (snap / "weight.safetensors").write_bytes(b"fixture")
            return snap.resolve()

        for inherit_assets in (False, True):
            with (
                self.subTest(inherit_assets=inherit_assets),
                tempfile.TemporaryDirectory() as tmp,
            ):
                root = Path(tmp)
                shutil.copytree(REPO_ROOT / "test/ppu", root / "test/ppu")
                cache_root = root / "hf_cache"
                snap_mxfp4 = _make_snapshot(cache_root, mxfp4_id, rev_mxfp4)
                snap_fp8 = _make_snapshot(cache_root, fp8_id, rev_fp8)
                snapshots = {
                    (mxfp4_id, rev_mxfp4): str(snap_mxfp4),
                    (fp8_id, rev_fp8): str(snap_fp8),
                }

                glm_rec_path = root / "test/ppu/recipes/zai-org/GLM-5.3.yaml"
                if inherit_assets:
                    glm_doc = _load_yaml(glm_rec_path)
                    glm_doc["variants"]["fp8_channel"].pop("assets", None)
                    glm_rec_path.write_text(yaml.safe_dump(glm_doc, sort_keys=False))

                pol = _load_yaml(root / "test/ppu/policy/models/glm-5.3.yaml")
                pol["variant"] = "fp8_channel"
                pol["tiers"] = {"fast_sanity": pol["tiers"]["fast_sanity"]}
                pol_path = root / "test/ppu/policy/models/glm-5.3-fp8.yaml"
                pol_path.write_text(yaml.safe_dump(pol, sort_keys=False))

                results_dir = root / "artifacts/results"
                results_dir.mkdir(parents=True)
                captured_configs = []

                @contextlib.contextmanager
                def fake_server(config, label, result_dict, timeout):
                    captured_configs.append((config, label, timeout))
                    result_dict["cold_start_s"] = 0.5
                    yield

                def fake_download(*, repo_id, revision, cache_dir, local_files_only):
                    self.assertTrue(local_files_only)
                    self.assertEqual(Path(cache_dir), cache_root)
                    return snapshots[(repo_id, revision)]

                fake_hf_modules = {
                    "huggingface_hub": SimpleNamespace(snapshot_download=fake_download),
                    "huggingface_hub.file_download": SimpleNamespace(
                        repo_folder_name=lambda *, repo_id, repo_type: (
                            f"models--{repo_id.replace('/', '--')}"
                        )
                    ),
                    "huggingface_hub.utils": SimpleNamespace(
                        validate_repo_id=lambda repo_id: None
                    ),
                }
                clean_env = {
                    k: v
                    for k, v in os.environ.items()
                    if k
                    not in {
                        "PPU_CI_HF_MODEL_REVISION",
                        "PPU_RECIPE_CONTAINER_MODEL",
                    }
                }
                clean_env.update(
                    {
                        "PPU_CI_HF_CACHE_ROOT": str(cache_root),
                        "HF_HUB_CACHE": str(cache_root),
                        "RECIPE_PPU_COUNT": "8",
                    }
                )
                with (
                    patch.dict(os.environ, clean_env, clear=True),
                    patch.dict(sys.modules, fake_hf_modules),
                    patch("ppu_recipe.REPO_ROOT", root),
                    patch("recipe_select.REPO_ROOT", root),
                    patch("recipe_runner.REPO_ROOT", root),
                    patch("recipe_runner.RESULTS", results_dir),
                    patch("recipe_runner.phase"),
                    patch("recipe_runner.server", side_effect=fake_server),
                    patch("recipe_runner.probes") as mock_probes,
                ):
                    admitted = select("fast_sanity", "glm-5.3-fp8")["include"]
                    self.assertEqual(
                        admitted, [{"model": "glm-5.3-fp8", "ppu_count": 8}]
                    )
                    summary = run(pol_path, "fast_sanity", 1.0)
                    mock_probes.assert_called_once()
                    self.assertEqual(len(captured_configs), 1)
                    cfg, _, _ = captured_configs[0]
                    self.assertEqual(cfg["variant"], "fp8_channel")
                    self.assertEqual(cfg["hf_model_id"], fp8_id)
                    self.assertEqual(cfg["golden_model_key"], fp8_golden)
                    self.assertEqual(cfg["resolved_model"], str(snap_fp8))
                    self.assertEqual(cfg["resolved_tokenizer"], str(snap_fp8))
                    self.assertNotEqual(cfg["resolved_model"], str(snap_mxfp4))
                    argv = cfg["argv"]
                    self.assertEqual(
                        argv[argv.index("--model-path") + 1], str(snap_fp8)
                    )
                    self.assertEqual(
                        argv[argv.index("--tokenizer-path") + 1], str(snap_fp8)
                    )
                    assets_written = json.loads(
                        (results_dir.parent / "assets.json").read_text()
                    )
                    expected_asset_key = mxfp4_asset if inherit_assets else fp8_asset
                    self.assertEqual(
                        assets_written, {expected_asset_key: str(snap_fp8)}
                    )
                    expected_fp8_digest = resolve_recipe(
                        glm_rec_path,
                        policy_path=pol_path,
                        tier="fast_sanity",
                        assets_map_path=results_dir.parent / "assets.json",
                        served_model_name="recipe-model",
                        cache_revision=rev_fp8,
                    )["config_digest"]
                    wrong_mxfp4_digest = resolve_recipe(
                        glm_rec_path,
                        policy_path=pol_path,
                        tier="fast_sanity",
                        assets_map_path=results_dir.parent / "assets.json",
                        served_model_name="recipe-model",
                        cache_revision=rev_mxfp4,
                    )["config_digest"]
                    self.assertEqual(summary["config_digest"], expected_fp8_digest)
                    self.assertNotEqual(summary["config_digest"], wrong_mxfp4_digest)

    def test_exact_topology_placeholder_identity_and_per_gpu_ib_map(self):
        glm_doc = _load_yaml(REPO_ROOT / "test/ppu/recipes/zai-org/GLM-5.3.yaml")

        # 1. --node-rank "{node_rank}0" or prefixed rank must fail closed.
        for bad_rank in ("{node_rank}0", "0{node_rank}", "rank-{node_rank}"):
            bad = copy.deepcopy(glm_doc)
            args = bad["presets"]["decode_ep"]["args"]
            args[args.index("--node-rank") + 1] = bad_rank
            with self.assertRaisesRegex(
                ValidationError,
                "numeric topology placeholder|--node-rank to be exact",
            ):
                validate_serving_recipe_structure(bad, context="bad_rank")

        # 2. Multi-node rank substitution preserves exact identity for all ranks 0..3.
        ep_resolved = resolve_recipe(
            REPO_ROOT / "test/ppu/recipes/zai-org/GLM-5.3.yaml", preset="decode_ep"
        )
        for expected_rank in range(4):
            sub_flags, _ = substitute_validation_placeholders(
                ep_resolved["flags"],
                ep_resolved["server_env"],
                values={"node_rank": expected_rank},
            )
            self.assertEqual(
                sub_flags[sub_flags.index("--node-rank") + 1], str(expected_rank)
            )

        # 3. --dist-init-addr digit concatenation or prefix must fail; IPv6 bracket form passes.
        for bad_addr in (
            "{master_addr}:{master_port}0",
            "prefix-{master_addr}:{master_port}",
        ):
            bad = copy.deepcopy(glm_doc)
            args = bad["presets"]["decode_ep"]["args"]
            args[args.index("--dist-init-addr") + 1] = bad_addr
            with self.assertRaisesRegex(
                ValidationError,
                "numeric topology placeholder|--dist-init-addr to be exact",
            ):
                validate_serving_recipe_structure(bad, context="bad_dist")

        ipv6_ok = copy.deepcopy(glm_doc)
        ep_args = ipv6_ok["presets"]["decode_ep"]["args"]
        ep_args[ep_args.index("--dist-init-addr") + 1] = "[{master_addr}]:{master_port}"
        validate_serving_recipe_structure(ipv6_ok, context="ipv6_dist")

        # 4. PD placeholder exactness: SGLANG_HOST_IP, bootstrap port, and IB device map.
        bad_ip = copy.deepcopy(glm_doc)
        bad_ip["presets"]["pd_prefill"]["env"]["SGLANG_HOST_IP"] = "host-{kv_ip}"
        with self.assertRaisesRegex(
            ValidationError, "SGLANG_HOST_IP to be exact '\\{kv_ip\\}'"
        ):
            validate_serving_recipe_structure(bad_ip, context="bad_kv_ip")

        bad_port = copy.deepcopy(glm_doc)
        p_args = bad_port["presets"]["pd_prefill"]["args"]
        p_args[p_args.index("--disaggregation-bootstrap-port") + 1] = "{kv_port}0"
        with self.assertRaisesRegex(
            ValidationError,
            "numeric topology placeholder|--disaggregation-bootstrap-port to be exact",
        ):
            validate_serving_recipe_structure(bad_port, context="bad_kv_port")

        for bad_ib in ("nic_{ib_devices}", '{"0":"rdma0","1":"{ib_devices}"}'):
            bad_ib_doc = copy.deepcopy(glm_doc)
            ib_args = bad_ib_doc["presets"]["pd_prefill"]["args"]
            ib_args[ib_args.index("--disaggregation-ib-device") + 1] = bad_ib
            with self.assertRaisesRegex(
                ValidationError,
                "--disaggregation-ib-device|missing required placeholder",
            ):
                validate_serving_recipe_structure(bad_ib_doc, context="bad_ib")

        per_gpu_ib = copy.deepcopy(glm_doc)
        ib_args = per_gpu_ib["presets"]["pd_prefill"]["args"]
        ib_args[ib_args.index("--disaggregation-ib-device") + 1] = (
            '{"0":"{ib_devices}","1":"{ib_devices}"}'
        )
        validate_serving_recipe_structure(per_gpu_ib, context="per_gpu_ib")

    def test_cluster_revision_pin_applies_only_to_default_mxfp4_variant(self):
        from recipe_cluster_config import resolve_pinned_model_revision_key

        rev_mxfp4 = "c" * 40
        rev_fp8 = "b" * 40
        base_glm = _load_yaml(REPO_ROOT / "test/ppu/recipes/zai-org/GLM-5.3.yaml")
        mxfp4_id = base_glm["ppu"]["hf_model_id"]
        fp8_id = base_glm["variants"]["fp8_channel"]["hf_model_id"]

        for disable_pyyaml in (False, True):
            with (
                self.subTest(disable_pyyaml=disable_pyyaml),
                tempfile.TemporaryDirectory() as tmp,
            ):
                root = Path(tmp)
                shutil.copytree(REPO_ROOT / "test/ppu", root / "test/ppu")
                ctx = (
                    patch.dict(sys.modules, {"yaml": None})
                    if disable_pyyaml
                    else contextlib.nullcontext()
                )
                with ctx:
                    self.assertEqual(
                        resolve_pinned_model_revision_key("glm-5.3", root),
                        "PPU_CI_GLM53_MODEL_REVISION",
                    )
                    self.assertEqual(
                        resolve_pinned_model_revision_key("deepseek-v4-flash", root),
                        "PPU_CI_DSV4_FLASH_MODEL_REVISION",
                    )
                    self.assertIsNone(
                        resolve_pinned_model_revision_key("qwen3.8-27b", root)
                    )

                glm_pol_path = root / "test/ppu/policy/models/glm-5.3.yaml"
                pol_doc = _load_yaml(glm_pol_path)
                pol_doc["variant"] = "fp8_channel"
                glm_pol_path.write_text(yaml.safe_dump(pol_doc, sort_keys=False))
                with ctx:
                    self.assertIsNone(
                        resolve_pinned_model_revision_key("glm-5.3", root)
                    )

                # Flow-style YAML mapping with non-default variant must also omit MXFP4 pin.
                glm_pol_path.write_text(
                    "{schema_version: ppu-test-policy/v1, variant: fp8_channel, "
                    "recipe: test/ppu/recipes/zai-org/GLM-5.3.yaml}\n"
                )
                with ctx:
                    self.assertIsNone(
                        resolve_pinned_model_revision_key("glm-5.3", root)
                    )

                # Default variant whose recipe overrides hf_model_id away from MXFP4 must omit pin.
                pol_doc.pop("variant", None)
                glm_pol_path.write_text(yaml.safe_dump(pol_doc, sort_keys=False))
                rec_path = root / "test/ppu/recipes/zai-org/GLM-5.3.yaml"
                rec_doc = _load_yaml(rec_path)
                rec_doc["variants"]["default"]["hf_model_id"] = fp8_id
                rec_path.write_text(yaml.safe_dump(rec_doc, sort_keys=False))
                with ctx:
                    self.assertIsNone(
                        resolve_pinned_model_revision_key("glm-5.3", root)
                    )

        if not all(shutil.which(tool) for tool in ("jq", "sha256sum")):
            return

        # End-to-end launcher -> USER_ENV_JSON -> worker run() on legacy key glm-5.3 with variant=fp8_channel.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            workspace = root / "checkout"
            staging = root / "staging"
            retained = root / "retained"
            tools = root / "bin"
            cache_root = root / "cache"
            backend = workspace / ".ppu-infra/cluster-run/scripts"
            payloads = workspace / "wheels"
            for folder in (staging, retained, tools, cache_root, backend, payloads):
                folder.mkdir(parents=True)
            retained.chmod(0o700)
            scripts = workspace / "scripts/ci/ppu"
            shutil.copytree(REPO_ROOT / "scripts/ci/ppu", scripts)
            shutil.copytree(REPO_ROOT / "test/ppu", workspace / "test/ppu")
            pol_path = workspace / "test/ppu/policy/models/glm-5.3.yaml"
            pol_doc = _load_yaml(pol_path)
            pol_doc["variant"] = "fp8_channel"
            pol_doc["tiers"] = {"fast_sanity": pol_doc["tiers"]["fast_sanity"]}
            pol_path.write_text(yaml.safe_dump(pol_doc, sort_keys=False))

            result_fixture = dict(fast_result(), model="glm-5.3")
            (scripts / "recipe_bundle.py").write_text("# Verified separately\n")
            (payloads / "recipe-manifest.json").write_text(
                json.dumps({"recipe_tree_sha256": result_fixture["recipe_tree_sha256"]})
            )
            (workspace / "fixture.json").write_text(
                json.dumps(
                    {
                        "outcome": "pass",
                        "parser_checked": True,
                        "sse_verified": True,
                        "cold_start_s": 3.0,
                        "queue_s": 0,
                        "layer_override": None,
                        "config_digest": result_fixture["config_digest"],
                    }
                )
            )
            (tools / "git").write_text(
                f'#!/bin/bash\ncase "$1" in -C) echo {BACKEND_SHA};; '
                f"*) echo {result_fixture['source_sha']};; esac\n"
            )
            kube_content = json.dumps(
                {
                    "status": {
                        "containerStatuses": [
                            {
                                "name": "worker",
                                "imageID": "docker-pullable://registry.internal/team/sglang"
                                + result_fixture["runtime_image"],
                            }
                        ]
                    }
                }
            )
            (tools / "kubectl").write_text(
                f"#!/bin/bash\nif [[ \"$*\" == *'get pod '* ]]; then echo '{kube_content}'; fi\n"
            )
            (tools / "timeout").write_text('#!/bin/bash\nshift\nexec "$@"\n')
            for tool in tools.iterdir():
                tool.chmod(0o700)
            captured_env_file = root / "captured-user-env.json"
            (backend / "submit.sh").write_text(
                f'printf "%s" "$USER_ENV_JSON" > "{captured_env_file}"\n'
                'printf "job_status=scheduled\\nstart_time=1\\ndeadline=2\\nqueue_duration_seconds=4\\n" >> "$GITHUB_OUTPUT"\n'
            )
            (backend / "monitor.sh").write_text(
                'mkdir -p "$PPU_CI_WORKER_ARTIFACT_DIR/results"\n'
                'cp fixture.json "$PPU_CI_WORKER_ARTIFACT_DIR/results/summary.json"\n'
                'echo job_status=succeeded >> "$GITHUB_OUTPUT"\n'
            )
            (backend / "cleanup.sh").write_text(
                'unlink "$SOURCE_STAGE_DIR"\nrmdir "$(dirname "$SOURCE_STAGE_DIR")"\n'
            )
            export_env = {
                **os.environ,
                "PATH": os.pathsep.join(
                    (
                        str(tools),
                        str(Path(sys.executable).parent),
                        os.environ["PATH"],
                    )
                ),
                "TIER": "fast_sanity",
                "MODEL_KEY": "glm-5.3",
                "RECIPE_PPU_COUNT": "8",
                "GITHUB_OUTPUT": str(root / "gha-output"),
                "PPU_PUBLIC_SUMMARY": str(root / "gha-summary"),
                "GITHUB_WORKSPACE": str(workspace),
                "RUNNER_TEMP": str(root / "runner-temp"),
                "GITHUB_RUN_ID": "88",
                "GITHUB_RUN_ATTEMPT": "1",
                "PPU_CI_K8S_CONFIG_SHA256": "",
                "PPU_CI_K8S_LOADED_CONFIG_SHA256": "",
                "PPU_CI_PRIVATE_RESULTS_ROOT": str(retained),
                "PPU_CI_K8S_PRIVATE_WORKER_ROOT": str(retained),
                "PPU_CI_K8S_STAGE_ROOT": str(staging),
                "PPU_CI_SOURCE_SHA": result_fixture["source_sha"],
                "PPU_CI_JIT_ROOT": str(root / "jit"),
                "PPU_CI_HF_CACHE_ROOT": str(cache_root),
                "PPU_CI_GLM53_MODEL_REVISION": rev_mxfp4,
                "PPU_CI_K8S_RUNTIME_DIGEST": "registry.internal/team/sglang"
                + result_fixture["runtime_image"],
            }
            proc = subprocess.run(
                ["bash", str(scripts / "run_recipe_cluster.sh")],
                cwd=workspace,
                env=export_env,
                capture_output=True,
                text=True,
                timeout=20,
            )
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            user_env = json.loads(captured_env_file.read_text())
            self.assertNotIn("PPU_CI_HF_MODEL_REVISION", user_env)

            # Feed the exact launcher USER_ENV_JSON into recipe_runner.run().
            for repo_id, rev in ((mxfp4_id, rev_mxfp4), (fp8_id, rev_fp8)):
                repo_dir = cache_root / f"models--{repo_id.replace('/', '--')}"
                (repo_dir / "refs").mkdir(parents=True, exist_ok=True)
                (repo_dir / "refs/main").write_text(rev)
                snap = repo_dir / "snapshots" / rev
                snap.mkdir(parents=True, exist_ok=True)
                for name in ("config.json", "tokenizer.json"):
                    (snap / name).write_text("{}")
                (snap / "model.safetensors.index.json").write_text(
                    json.dumps({"weight_map": {"w": "weight.safetensors"}})
                )
                (snap / "weight.safetensors").write_bytes(b"fixture")

            expected_fp8_snap = str(
                (
                    cache_root
                    / f"models--{fp8_id.replace('/', '--')}"
                    / "snapshots"
                    / rev_fp8
                ).resolve()
            )
            worker_results = root / "worker-artifacts/results"
            worker_results.mkdir(parents=True)
            captured_cfgs = []

            @contextlib.contextmanager
            def fake_server(config, label, result_dict, timeout):
                captured_cfgs.append(config)
                result_dict["cold_start_s"] = 0.25
                yield

            fake_hf = {
                "huggingface_hub": SimpleNamespace(
                    snapshot_download=lambda *, repo_id, revision, cache_dir, local_files_only: (
                        str(
                            (
                                Path(cache_dir)
                                / f"models--{repo_id.replace('/', '--')}"
                                / "snapshots"
                                / revision
                            ).resolve(strict=True)
                        )
                    )
                ),
                "huggingface_hub.file_download": SimpleNamespace(
                    repo_folder_name=lambda *, repo_id, repo_type: (
                        f"models--{repo_id.replace('/', '--')}"
                    )
                ),
                "huggingface_hub.utils": SimpleNamespace(
                    validate_repo_id=lambda repo_id: None
                ),
            }
            worker_env = {**user_env, "HF_HUB_CACHE": str(cache_root)}
            with (
                patch.dict(os.environ, worker_env, clear=True),
                patch.dict(sys.modules, fake_hf),
                patch("ppu_recipe.REPO_ROOT", workspace),
                patch("recipe_runner.REPO_ROOT", workspace),
                patch("recipe_runner.RESULTS", worker_results),
                patch("recipe_runner.phase"),
                patch("recipe_runner.server", side_effect=fake_server),
                patch("recipe_runner.probes"),
            ):
                worker_summary = run(pol_path, "fast_sanity", 0.0)
                self.assertEqual(len(captured_cfgs), 1)
                self.assertEqual(captured_cfgs[0]["hf_model_id"], fp8_id)
                self.assertEqual(captured_cfgs[0]["resolved_model"], expected_fp8_snap)
                self.assertEqual(worker_summary["outcome"], "pass")


if __name__ == "__main__":
    unittest.main()
