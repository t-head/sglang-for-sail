# SPDX-License-Identifier: Apache-2.0

import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ppu_recipe import (
    DEFAULT_POLICY_ROOT,
    REPO_ROOT,
    ValidationError,
    _load_yaml,
    resolve_recipe,
    validate_paths,
    validate_policy_structure,
)
from recipe_bundle import sha256, wheel_inventory
from recipe_cluster_config import load_settings
from recipe_cluster_diagnostics import diagnostic_codes
from recipe_publish import publish, validate
from recipe_report import legacy_result
from recipe_report import main as report_recipe
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
                    {"model": "deepseek-v4.1-flash-1017", "ppu_count": 8},
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

    def test_deepseek_v41_1017_uses_engram_snapshot_for_both_tiers(self):
        policy_path = DEFAULT_POLICY_ROOT / "models/deepseek-v4.1-flash-1017.yaml"
        policy = _load_yaml(policy_path)
        recipe = REPO_ROOT / policy["recipe"]
        fast = resolve_recipe(recipe, policy_path=policy_path, tier="fast_sanity")
        nightly = resolve_recipe(recipe, policy_path=policy_path, tier="model_sanity")
        legacy = resolve_recipe(recipe.with_name("DeepSeek-V4.1-Flash.yaml"))
        for config in (fast, nightly):
            self.assertEqual(config["ppu_count"], 8)
            self.assertIsNone(config["layer_override"])
            self.assertNotIn(
                "SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE", config["server_env"]
            )
            self.assertNotEqual(config["hf_model_id"], legacy["hf_model_id"])
        self.assertTrue(fast["dummy_weights"])
        self.assertFalse(nightly["dummy_weights"])
        golden_path = (
            REPO_ROOT
            / "test/ppu/golden"
            / f"accuracy_{nightly['golden_model_key']}_890p.json"
        )
        golden = json.loads(golden_path.read_text())
        stage = next(
            stage
            for stage in policy["tiers"]["model_sanity"]["stages"]
            if stage["kind"] == "evalscope"
        )
        self.assertEqual(fast["hf_model_id"], golden["checkpoint"])
        self.assertEqual(nightly["hf_model_id"], golden["checkpoint"])
        self.assertEqual(stage["generation"], golden["generation"])
        self.assertEqual(golden["dataset"]["num_expected"], 1319)
        self.assertEqual(golden["score_num"], 1281)

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
            ("glm-5.3", 8, 4500),
            ("glm-5.3-tp4", 4, 4500),
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

    def test_cluster_diagnostics_classifies_server_logs_and_startup_phases(self):
        import recipe_cluster_diagnostics
        import recipe_diagnostics
        import recipe_runner

        self.assertEqual(recipe_runner.bench_module_name(), "sglang.benchmark.serving")
        for phase in ("server_exit", "readiness_timeout"):
            self.assertIn(phase, recipe_diagnostics.PHASES)
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            fast_dir = root / "sglang-for-sail/45014946019-1/fast_sanity/qwen3.8-27b"
            fast_dir.mkdir(parents=True)
            (fast_dir / "index.json").write_text("{}")
            (fast_dir / "sglang_serve_fast_full.log").write_text(
                "ModuleNotFoundError: No module named 'blake3'\n"
            )
            nightly_dir = root / "sglang-for-sail/45055310116-1/model_sanity/glm-5.3"
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


if __name__ == "__main__":
    unittest.main()
