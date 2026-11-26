# SPDX-License-Identifier: Apache-2.0

import contextlib
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

from ppu_recipe import (
    DEFAULT_MATRIX_DOC,
    DEFAULT_POLICY_ROOT,
    REPO_ROOT,
    ValidationError,
    _load_yaml,
    _scan_text,
)
from ppu_recipe import main as ppu_recipe_main
from ppu_recipe import (
    render_ci_matrix_markdown,
    resolve_recipe,
    validate_paths,
    validate_policy_structure,
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
            launcher = scripts / "run_recipe_cluster.sh"
            launcher.write_bytes(
                (REPO_ROOT / "scripts/ci/ppu/run_recipe_cluster.sh").read_bytes()
            )
            launcher.chmod(0o755)
            (scripts / "recipe_cluster_config.py").write_text("raise SystemExit(29)\n")
            process = subprocess.run(
                ["bash", str(launcher)],
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


class MultiNodeRecipeTest(unittest.TestCase):
    def test_qwen38_24t_mxfp4_recipe_resolves_tp8_pp2_and_is_isolated_from_default_fast(
        self,
    ):
        from recipe_select import cluster_topology

        model = "qwen3.8-2.4t-mxfp4"
        policy = DEFAULT_POLICY_ROOT / f"models/{model}.yaml"
        recipe = REPO_ROOT / _load_yaml(policy)["recipe"]
        resolved = resolve_recipe(recipe, policy_path=policy, tier="fast_sanity")
        self.assertEqual(resolved["model_id"], "Qwen/Qwen3.8-2.4T-A95B")
        self.assertEqual(
            resolved["hf_model_id"],
            "T-HEAD/Qwen3.8-2.4T-A95B-FP8-mxfp4-fp8-030",
        )
        self.assertEqual(resolved["nodes"], 2)
        self.assertEqual(resolved["devices_per_node"], 8)
        self.assertEqual(resolved["ppu_count"], 16)
        self.assertEqual(cluster_topology(policy, "fast_sanity"), (2, 8))
        self.assertTrue(resolved["dummy_weights"])
        self.assertIsNone(resolved["layer_override"])
        flags = resolved["flags"]
        for name, value in (
            ("--tp-size", "8"),
            ("--pp-size", "2"),
            ("--attention-backend", "fa3"),
            ("--mem-fraction-static", "0.8"),
            ("--linear-attn-prefill-backend", "triton"),
            ("--linear-attn-decode-backend", "triton"),
            ("--mamba-ssm-dtype", "bfloat16"),
            ("--reasoning-parser", "qwen3-thinking"),
            ("--dist-timeout", "14400"),
            ("--watchdog-timeout", "7200"),
            ("--load-format", "dummy"),
        ):
            self.assertEqual(flags[flags.index(name) + 1], value)
        self.assertIn("--disable-radix-cache", flags)
        self.assertIn("--trust-remote-code", flags)

        model_resolved = resolve_recipe(recipe, policy_path=policy, tier="model_sanity")
        mflags = model_resolved["flags"]
        self.assertEqual(mflags[mflags.index("--page-size") + 1], "64")
        self.assertEqual(
            mflags[mflags.index("--mamba-radix-cache-strategy") + 1], "extra_buffer"
        )

        # Capacity isolation: absent from default fast_sanity, selectable explicitly.
        default_fast = select("fast_sanity", "")
        self.assertNotIn(model, [item["model"] for item in default_fast["include"]])
        self.assertEqual(
            select("fast_sanity", "", topology="multi_node"), {"include": []}
        )
        self.assertEqual(
            select("fast_sanity", model, topology="single_node"), {"include": []}
        )
        self.assertEqual(
            select("fast_sanity", model, topology="multi_node"),
            {"include": [{"model": model, "ppu_count": 8, "nnodes": 2}]},
        )
        # Model sanity capacity isolation: excluded from single_node slot, selected in multi_node slot.
        self.assertNotIn(
            model,
            [
                item["model"]
                for item in select("model_sanity", "", topology="single_node")[
                    "include"
                ]
            ],
        )
        self.assertEqual(
            select("model_sanity", model, topology="multi_node"),
            {"include": [{"model": model, "ppu_count": 8, "nnodes": 2}]},
        )
        model_resolved = resolve_recipe(recipe, policy_path=policy, tier="model_sanity")
        self.assertFalse(model_resolved["dummy_weights"])
        self.assertEqual(model_resolved["ppu_count"], 16)
        self.assertEqual(
            model_resolved["flags"][model_resolved["flags"].index("--page-size") + 1],
            "64",
        )
        self.assertEqual(
            model_resolved["flags"][
                model_resolved["flags"].index("--mamba-radix-cache-strategy") + 1
            ],
            "extra_buffer",
        )

        # Multi-node runner-owned flags must be rejected in public recipe YAML.
        bad_recipe = _load_yaml(recipe)
        bad_recipe["model"]["base_args"] = [
            *bad_recipe["model"]["base_args"],
            "--dist-init-addr",
            "127.0.0.1:29500",
        ]
        from ppu_recipe import validate_serving_recipe_structure

        with self.assertRaisesRegex(ValidationError, "framework runner owns"):
            validate_serving_recipe_structure(bad_recipe, context="bad")

    def test_multinode_rendezvous_publishes_joins_and_aborts_on_peer_failure(self):
        script = REPO_ROOT / "scripts/ci/ppu/recipe_multinode_rendezvous.sh"
        with tempfile.TemporaryDirectory() as raw_root:
            ranks = Path(raw_root) / "ranks"
            rank0 = subprocess.run(
                ["bash", str(script), str(ranks)],
                env={
                    **os.environ,
                    "PPU_CI_NODE_RANK": "0",
                    "SGLANG_HOST_IP": "198.51.100.10",
                    "MASTER_PORT": "29511",
                },
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(rank0.returncode, 0, rank0.stderr)
            self.assertEqual(rank0.stdout.strip(), "198.51.100.10:29511")

            rank1 = subprocess.run(
                ["bash", str(script), str(ranks)],
                env={
                    **os.environ,
                    "PPU_CI_NODE_RANK": "1",
                    "RECIPE_RENDEZVOUS_TIMEOUT_S": "5",
                    "RECIPE_RENDEZVOUS_POLL_S": "1",
                },
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(rank1.returncode, 0, rank1.stderr)
            self.assertEqual(rank1.stdout.strip(), "198.51.100.10:29511")

            # Peer abort barrier short-circuits waiting workers immediately.
            ranks_abort = Path(raw_root) / "ranks-abort"
            barrier = ranks_abort / "barrier"
            barrier.mkdir(parents=True)
            (barrier / "rank0.done").write_text('{"status":"error"}\n')
            aborted = subprocess.run(
                ["bash", str(script), str(ranks_abort)],
                env={
                    **os.environ,
                    "PPU_CI_NODE_RANK": "1",
                    "RECIPE_RENDEZVOUS_TIMEOUT_S": "30",
                    "RECIPE_RENDEZVOUS_POLL_S": "1",
                },
                capture_output=True,
                text=True,
                timeout=5,
            )
            self.assertNotEqual(aborted.returncode, 0)
            self.assertIn("aborted", aborted.stderr)

    def test_multinode_gid_index_picks_highest_routable_and_rejects_mismatch(self):
        script = REPO_ROOT / "scripts/ci/ppu/recipe_multinode_gid_index.sh"
        with tempfile.TemporaryDirectory() as raw_root:
            sysfs = Path(raw_root) / "infiniband"
            for dev in ("roce0", "roce1"):
                gids = sysfs / dev / "ports/1/gids"
                gids.mkdir(parents=True)
                (gids / "0").write_text("fe80:0000:0000:0000:0211:22ff:fe33:4455\n")
                (gids / "1").write_text("0000:0000:0000:0000:0000:0000:0000:0000\n")
                (gids / "2").write_text("fd03:45c2:0001:0001:0000:0000:0000:0001\n")
                (gids / "3").write_text("fd03:45c2:0001:0001:0000:0000:0000:0001\n")
                (gids / "4").write_text("fe80:45c2:0001:0001:0000:0000:0000:0001\n")
                (gids / "10").write_text("0000:0000:0000:0000:0000:0000:0000:0000\n")
            ok = subprocess.run(
                ["bash", str(script)],
                env={**os.environ, "RECIPE_GID_SYSFS_ROOT": str(sysfs)},
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(ok.returncode, 0, ok.stderr)
            self.assertEqual(ok.stdout.strip(), "3")

            # An extra routable entry on one device still resolves the highest common index.
            (sysfs / "roce1/ports/1/gids/11").write_text(
                "fd03:45c2:0001:0001:0000:0000:0000:0002\n"
            )
            common = subprocess.run(
                ["bash", str(script)],
                env={**os.environ, "RECIPE_GID_SYSFS_ROOT": str(sysfs)},
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(common.returncode, 0, common.stderr)
            self.assertEqual(common.stdout.strip(), "3")

            # Truly disjoint devices (no common routable index) must fail closed.
            (sysfs / "roce1/ports/1/gids/2").unlink()
            (sysfs / "roce1/ports/1/gids/3").unlink()
            mismatch = subprocess.run(
                ["bash", str(script)],
                env={**os.environ, "RECIPE_GID_SYSFS_ROOT": str(sysfs)},
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertNotEqual(mismatch.returncode, 0)

            # Inactive (DOWN) ports, non-bond NICs, link-local-only bonds, and IPv4-mapped
            # management bonds must not block active global-IPv6 RoCE fabric bonds.
            sysfs_bond = Path(raw_root) / "infiniband-bond"
            for dev, state, idx, addr in (
                (
                    "mlx5_bond_1",
                    "4: ACTIVE",
                    "3",
                    "fd03:45c2:0001:0001:0000:0000:0000:0001",
                ),
                (
                    "mlx5_bond_2",
                    "4: ACTIVE",
                    "3",
                    "fd03:45c2:0001:0001:0000:0000:0000:0002",
                ),
                (
                    "mlx5_bond_0",
                    "4: ACTIVE",
                    "1",
                    "0000:0000:0000:0000:0000:ffff:0b00:0001",
                ),
                (
                    "mlx5_bond_ll",
                    "4: ACTIVE",
                    "0",
                    "fe80:0000:0000:0000:0211:22ff:fe33:4455",
                ),
                (
                    "mlx5_bond_down",
                    "1: DOWN",
                    "1",
                    "fd03:45c2:0001:0001:0000:0000:0000:0009",
                ),
                ("mlx5_0", "4: ACTIVE", "1", "fd03:45c2:0001:0001:0000:0000:0000:0008"),
            ):
                gids = sysfs_bond / dev / "ports/1/gids"
                types = sysfs_bond / dev / "ports/1/gid_attrs/types"
                gids.mkdir(parents=True)
                types.mkdir(parents=True)
                (sysfs_bond / dev / "ports/1/state").write_text(state + "\n")
                (gids / idx).write_text(addr + "\n")
                (types / idx).write_text("RoCE v2\n")
            bond_ok = subprocess.run(
                ["bash", str(script)],
                env={**os.environ, "RECIPE_GID_SYSFS_ROOT": str(sysfs_bond)},
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(bond_ok.returncode, 0, bond_ok.stderr)
            self.assertEqual(bond_ok.stdout.strip(), "3")

            # Non-ULA global IPv6 management bond must not block ULA fabric bonds,
            # and a single shifted rail among >=3 fabric bonds falls back to strict majority.
            sysfs_ula = Path(raw_root) / "infiniband-ula"
            for dev, idx, addr in (
                ("mlx5_bond_0", "5", "2408:4001:0001:0001:0000:0000:0000:0001"),
                ("mlx5_bond_1", "3", "fd03:45c2:0001:0001:0000:0000:0000:0001"),
                ("mlx5_bond_2", "3", "fd03:45c2:0001:0001:0000:0000:0000:0002"),
                ("mlx5_bond_3", "5", "fd03:45c2:0001:0001:0000:0000:0000:0003"),
            ):
                gids = sysfs_ula / dev / "ports/1/gids"
                types = sysfs_ula / dev / "ports/1/gid_attrs/types"
                gids.mkdir(parents=True)
                types.mkdir(parents=True)
                (sysfs_ula / dev / "ports/1/state").write_text("4: ACTIVE\n")
                (gids / idx).write_text(addr + "\n")
                (types / idx).write_text("RoCE v2\n")
            ula_ok = subprocess.run(
                ["bash", str(script)],
                env={**os.environ, "RECIPE_GID_SYSFS_ROOT": str(sysfs_ula)},
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(ula_ok.returncode, 0, ula_ok.stderr)
            self.assertEqual(ula_ok.stdout.strip(), "3")

            # Link-local-only table must fail closed.
            sysfs_ll = Path(raw_root) / "infiniband-ll"
            gids = sysfs_ll / "roce0/ports/1/gids"
            gids.mkdir(parents=True)
            (gids / "0").write_text("fe80:0000:0000:0000:0211:22ff:fe33:4455\n")
            ll_res = subprocess.run(
                ["bash", str(script)],
                env={**os.environ, "RECIPE_GID_SYSFS_ROOT": str(sysfs_ll)},
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertNotEqual(ll_res.returncode, 0)

            # Mixed gid_attrs/types table: index 3 is RoCE v2 and index 4 is IB/RoCE v1;
            # must select index 3 (RoCE v2) rather than higher-indexed RoCE v1.
            sysfs_typed = Path(raw_root) / "infiniband-typed"
            for dev in ("roce0", "roce1"):
                gids = sysfs_typed / dev / "ports/1/gids"
                types = sysfs_typed / dev / "ports/1/gid_attrs/types"
                gids.mkdir(parents=True)
                types.mkdir(parents=True)
                (gids / "2").write_text("fd03:45c2:0001:0001:0000:0000:0000:0001\n")
                (types / "2").write_text("IB/RoCE v1\n")
                (gids / "3").write_text("fd03:45c2:0001:0001:0000:0000:0000:0001\n")
                (types / "3").write_text("RoCE v2\n")
                (gids / "4").write_text("fd03:45c2:0001:0001:0000:0000:0000:0002\n")
                (types / "4").write_text("IB/RoCE v1\n")
            typed_ok = subprocess.run(
                ["bash", str(script)],
                env={**os.environ, "RECIPE_GID_SYSFS_ROOT": str(sysfs_typed)},
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(typed_ok.returncode, 0, typed_ok.stderr)
            self.assertEqual(typed_ok.stdout.strip(), "3")

            # Typed table with only RoCE v1 must fail closed.
            for dev in ("roce0", "roce1"):
                (sysfs_typed / dev / "ports/1/gid_attrs/types/3").write_text(
                    "IB/RoCE v1\n"
                )
            v1_only = subprocess.run(
                ["bash", str(script)],
                env={**os.environ, "RECIPE_GID_SYSFS_ROOT": str(sysfs_typed)},
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertNotEqual(v1_only.returncode, 0)

    def test_verify_pod_statuses_requires_matching_digests_across_workers(self):
        import recipe_image

        digest_a = "sha256:" + "a" * 64
        digest_b = "sha256:" + "b" * 64
        env = {
            "PPU_CI_K8S_RUNTIME_DIGEST": f"registry.internal/team/sglang@{digest_a}",
            "PPU_CI_RECIPE_ALLOWED_IMAGE_REPOS": "registry.internal/team/",
        }
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            pod0 = root / "pod-status-worker-0.json"
            pod1 = root / "pod-status-worker-1.json"
            for pod, digest in ((pod0, digest_a), (pod1, digest_a)):
                pod.write_text(
                    json.dumps(
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
                )
            self.assertEqual(
                recipe_image.verify_pod_statuses(
                    [pod0, pod1],
                    env,
                    expected_repo="registry.internal/team/sglang",
                ),
                f"@{digest_a}",
            )
            pod1.write_text(
                json.dumps(
                    {
                        "status": {
                            "containerStatuses": [
                                {
                                    "name": "worker",
                                    "imageID": f"docker-pullable://registry.internal/team/sglang@{digest_b}",
                                }
                            ]
                        }
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "inconsistent runtime image"):
                recipe_image.verify_pod_statuses(
                    [pod0, pod1],
                    env,
                    expected_repo="registry.internal/team/sglang",
                )

    def test_multinode_runner_bidirectional_exit_barrier(self):
        import recipe_runner

        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            ranks_dir = root / "ranks"
            barrier_dir = ranks_dir / "barrier"
            barrier_dir.mkdir(parents=True)
            worker_results = ranks_dir / "rank-1/results"
            worker_results.mkdir(parents=True)

            # Case 1: Rank 0 writes rank0.done -> worker stops process and exits 0.
            sleep_cfg = {
                "argv": [sys.executable, "-c", "import time; time.sleep(60)"],
                "server_env": {},
            }

            def _finish_soon():
                import time

                time.sleep(0.3)
                recipe_runner._barrier_write(
                    barrier_dir, "rank0.done", {"status": "ok"}
                )

            import threading

            t = threading.Thread(target=_finish_soon, daemon=True)
            t.start()
            with patch.object(recipe_runner, "RESULTS", worker_results):
                out = recipe_runner.run_worker_node(
                    sleep_cfg,
                    "fast_full",
                    nnodes=2,
                    node_rank=1,
                    dist_init_addr="198.51.100.10:29500",
                    barrier_dir=barrier_dir,
                    poll_s=0.1,
                )
            t.join(timeout=5)
            self.assertEqual(out["outcome"], "pass")
            exit_rec = json.loads((barrier_dir / "exit-rank-1.json").read_text())
            self.assertEqual(exit_rec["exit_code"], 0)
            self.assertEqual(exit_rec["reason"], "rank0_done")

            # Case 2: Worker crashes before rank0.done -> writes fail-rank-1.json,
            # and rank 0's server() immediately aborts with StartupError("worker_exit").
            barrier2 = root / "ranks2/barrier"
            barrier2.mkdir(parents=True)
            worker_results2 = root / "ranks2/rank-1/results"
            worker_results2.mkdir(parents=True)
            crash_cfg = {
                "argv": [sys.executable, "-c", "raise SystemExit(7)"],
                "server_env": {},
            }
            with patch.object(recipe_runner, "RESULTS", worker_results2):
                with self.assertRaisesRegex(recipe_runner.StartupError, "server_exit"):
                    recipe_runner.run_worker_node(
                        crash_cfg,
                        "fast_full",
                        nnodes=2,
                        node_rank=1,
                        dist_init_addr="198.51.100.10:29500",
                        barrier_dir=barrier2,
                        poll_s=0.05,
                    )
            self.assertTrue((barrier2 / "fail-rank-1.json").is_file())

            rank0_results = root / "ranks2/rank0/results"
            rank0_results.mkdir(parents=True)
            timings = {}
            with patch.object(recipe_runner, "RESULTS", rank0_results):
                with self.assertRaisesRegex(recipe_runner.StartupError, "worker_exit"):
                    with recipe_runner.server(
                        sleep_cfg,
                        "fast_full",
                        timings,
                        30,
                        nnodes=2,
                        node_rank=0,
                        dist_init_addr="198.51.100.10:29500",
                        barrier_dir=barrier2,
                        signal_done_on_exit=True,
                    ):
                        pass
            self.assertTrue((barrier2 / "rank0.done").is_file())
            done_rec = json.loads((barrier2 / "rank0.done").read_text())
            self.assertEqual(done_rec["status"], "error")

            # Case 3: Worker exits 7 BEFORE rank0.done appears, then rank0.done
            # arrives in the same poll window -> must preserve exit_code 7 and
            # never rewrite the crash into exit 0.
            barrier3 = root / "ranks3/barrier"
            barrier3.mkdir(parents=True)
            worker_results3 = root / "ranks3/rank-1/results"
            worker_results3.mkdir(parents=True)
            recipe_runner._barrier_write(barrier3, "rank0.done", {"status": "ok"})
            fast_crash_cfg = {
                "argv": [sys.executable, "-c", "raise SystemExit(7)"],
                "server_env": {},
            }
            # Allow the subprocess to exit before the first poll by sleeping
            # inside a wrapper that runs the worker with a tiny initial race.
            orig_popen = recipe_runner.subprocess.Popen

            def _popen_then_wait(*args, **kwargs):
                proc = orig_popen(*args, **kwargs)
                proc.wait(timeout=5)
                return proc

            with (
                patch.object(recipe_runner, "RESULTS", worker_results3),
                patch.object(
                    recipe_runner.subprocess, "Popen", side_effect=_popen_then_wait
                ),
            ):
                with self.assertRaisesRegex(recipe_runner.StartupError, "server_exit"):
                    recipe_runner.run_worker_node(
                        fast_crash_cfg,
                        "fast_full",
                        nnodes=2,
                        node_rank=1,
                        dist_init_addr="198.51.100.10:29500",
                        barrier_dir=barrier3,
                        poll_s=0.05,
                    )
            fail_rec = json.loads((barrier3 / "fail-rank-1.json").read_text())
            exit_rec3 = json.loads((barrier3 / "exit-rank-1.json").read_text())
            self.assertEqual(fail_rec["exit_code"], 7)
            self.assertEqual(exit_rec3["exit_code"], 7)
            with (
                patch.object(recipe_runner, "RESULTS", rank0_results),
                self.assertRaisesRegex(RuntimeError, "multi-node worker exited"),
            ):
                recipe_runner._wait_workers_exit(barrier3, 2, timeout=2.0, poll_s=0.05)

            # Case 4: Exact mid-window race: first poll() returns None, then worker
            # exits 7 AND rank0.done appears before _stop_process_group() runs ->
            # must still preserve exit_code 7 and fail rank 0 wait.
            barrier4 = root / "ranks4/barrier"
            barrier4.mkdir(parents=True)
            worker_results4 = root / "ranks4/rank-1/results"
            worker_results4.mkdir(parents=True)
            proc_holder = {}

            def _capture_popen(*args, **kwargs):
                proc = orig_popen(*args, **kwargs)
                proc_holder["proc"] = proc
                return proc

            def _race_barrier_names(bdir):
                # Simulate the exact window between poll()==None and barrier read:
                # wait for subprocess to exit with code 7, then publish rank0.done.
                proc_holder["proc"].wait(timeout=5)
                recipe_runner._barrier_write(barrier4, "rank0.done", {"status": "ok"})
                return {"rank0.done"}

            delayed_crash_cfg = {
                "argv": [
                    sys.executable,
                    "-c",
                    "import time; time.sleep(0.15); raise SystemExit(7)",
                ],
                "server_env": {},
            }
            with (
                patch.object(recipe_runner, "RESULTS", worker_results4),
                patch.object(
                    recipe_runner.subprocess, "Popen", side_effect=_capture_popen
                ),
                patch.object(
                    recipe_runner, "_barrier_names", side_effect=_race_barrier_names
                ),
            ):
                with self.assertRaisesRegex(recipe_runner.StartupError, "server_exit"):
                    recipe_runner.run_worker_node(
                        delayed_crash_cfg,
                        "fast_full",
                        nnodes=2,
                        node_rank=1,
                        dist_init_addr="198.51.100.10:29500",
                        barrier_dir=barrier4,
                        poll_s=0.05,
                    )
            fail_rec4 = json.loads((barrier4 / "fail-rank-1.json").read_text())
            exit_rec4 = json.loads((barrier4 / "exit-rank-1.json").read_text())
            self.assertEqual(fail_rec4["exit_code"], 7)
            self.assertEqual(exit_rec4["exit_code"], 7)
            with (
                patch.object(recipe_runner, "RESULTS", rank0_results),
                self.assertRaisesRegex(RuntimeError, "multi-node worker exited"),
            ):
                recipe_runner._wait_workers_exit(barrier4, 2, timeout=2.0, poll_s=0.05)

    @unittest.skipUnless(
        all(shutil.which(tool) for tool in ("jq", "sha256sum")),
        "runner tools are required",
    )
    def test_multinode_cluster_transport_sets_gang_mode_verifies_all_workers_and_cleanup(
        self,
    ):
        digest_ok = "sha256:" + "d" * 64
        digest_bad = "sha256:" + "e" * 64
        source_sha = "a" * 40
        for scenario in (
            "ok",
            "host_access_denied",
            "digest_mismatch",
            "cleanup_leftover",
        ):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as temp:
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
                payloads = workspace / "wheels"
                payloads.mkdir()
                (scripts / "recipe_bundle.py").write_text("# verified separately\n")
                (payloads / "recipe-manifest.json").write_text(
                    json.dumps({"recipe_tree_sha256": "0" * 64})
                )
                summary = {
                    "outcome": "pass",
                    "parser_checked": True,
                    "sse_verified": True,
                    "cold_start_s": 12.5,
                    "queue_s": 0,
                    "layer_override": None,
                    "config_digest": "c" * 64,
                }
                (workspace / "fixture.json").write_text(json.dumps(summary))
                (tools / "git").write_text(
                    '#!/bin/bash\ncase "$1" in -C) echo 96e25147971f5d2a04d828f8ad508cae38a4fa8a;; '
                    f"*) echo {source_sha};; esac\n"
                )
                pod0_json = json.dumps(
                    {
                        "status": {
                            "containerStatuses": [
                                {
                                    "name": "worker",
                                    "imageID": f"docker-pullable://registry.internal/team/sglang@{digest_ok}",
                                }
                            ]
                        }
                    }
                )
                pod1_digest = digest_bad if scenario == "digest_mismatch" else digest_ok
                pod1_json = json.dumps(
                    {
                        "status": {
                            "containerStatuses": [
                                {
                                    "name": "worker",
                                    "imageID": f"docker-pullable://registry.internal/team/sglang@{pod1_digest}",
                                }
                            ]
                        }
                    }
                )
                leftover_line = (
                    'if [[ "$*" == *"get pods "* ]]; then echo "pod/stray-worker-1"; exit 0; fi\n'
                    if scenario == "cleanup_leftover"
                    else ""
                )
                (tools / "kubectl").write_text(
                    "#!/bin/bash\n"
                    f"{leftover_line}"
                    f"if [[ \"$*\" == *'worker-0'* ]]; then echo '{pod0_json}'; exit 0; fi\n"
                    f"if [[ \"$*\" == *'worker-1'* ]]; then echo '{pod1_json}'; exit 0; fi\n"
                )
                (tools / "timeout").write_text('#!/bin/bash\nshift\nexec "$@"\n')
                for tool in tools.iterdir():
                    tool.chmod(0o700)
                (backend / "submit.sh").write_text(
                    'test "$INPUT_NNODES/$INPUT_NPROC_PER_NODE/$INPUT_NODE_SELECTOR" = "2/8/board-type=ZW-M890P" || exit 1\n'
                    'grep -q "privileged: true" <<< "$CONTAINER_OPTIONS" || exit 1\n'
                    'grep -q "host_ipc: true" <<< "$CONTAINER_OPTIONS" || exit 1\n'
                    'grep -q "host_network: true" <<< "$CONTAINER_OPTIONS" || exit 1\n'
                    'jq -e \'.RECIPE_NNODES == "2" and .RECIPE_PPU_COUNT == "8"\' <<< "$USER_ENV_JSON" >/dev/null || exit 1\n'
                    # Match real cluster-run submit.sh: initializes queue_duration_seconds=0
                    # and leaves final gang queue_duration_seconds to monitor.sh/report-job.sh.
                    'printf "job_status=scheduled\\nstart_time=1\\ndeadline=2\\nqueue_duration_seconds=0\\n" >> "$GITHUB_OUTPUT"\n'
                )
                (backend / "monitor.sh").write_text(
                    'mkdir -p "$PPU_CI_WORKER_ARTIFACT_DIR/results" "$PPU_CI_WORKER_ARTIFACT_DIR/ranks/rank-1"\n'
                    'cp fixture.json "$PPU_CI_WORKER_ARTIFACT_DIR/results/summary.json"\n'
                    'echo "worker1 ok" > "$PPU_CI_WORKER_ARTIFACT_DIR/ranks/rank-1/runtime.log"\n'
                    'printf "queue_duration_seconds=120\\njob_status=succeeded\\n" >> "$GITHUB_OUTPUT"\n'
                )
                (backend / "cleanup.sh").write_text(
                    'unlink "$SOURCE_STAGE_DIR"\nrmdir "$(dirname "$SOURCE_STAGE_DIR")"\n'
                )
                env = {
                    **os.environ,
                    "PATH": os.pathsep.join(
                        (
                            str(tools),
                            str(Path(sys.executable).parent),
                            os.environ["PATH"],
                        )
                    ),
                    "TIER": "fast_sanity",
                    "MODEL_KEY": "qwen3.8-2.4t-mxfp4",
                    "RECIPE_PPU_COUNT": "8",
                    "RECIPE_NNODES": "2",
                    "PPU_CI_ALLOW_MULTINODE_HOST_ACCESS": (
                        "false" if scenario == "host_access_denied" else "true"
                    ),
                    "GITHUB_OUTPUT": str(root / "gha-output"),
                    "PPU_PUBLIC_SUMMARY": str(root / "gha-summary"),
                    "GITHUB_WORKSPACE": str(workspace),
                    "RUNNER_TEMP": str(root / "runner-temp"),
                    "GITHUB_RUN_ID": "77",
                    "GITHUB_RUN_ATTEMPT": "1",
                    "PPU_CI_K8S_CONFIG_SHA256": "",
                    "PPU_CI_K8S_LOADED_CONFIG_SHA256": "",
                    "PPU_CI_PRIVATE_RESULTS_ROOT": str(retained),
                    "PPU_CI_K8S_PRIVATE_WORKER_ROOT": str(retained),
                    "PPU_CI_K8S_STAGE_ROOT": str(staging),
                    "PPU_CI_SOURCE_SHA": source_sha,
                    "PPU_CI_JIT_ROOT": str(root / "jit"),
                    "PPU_CI_HF_CACHE_ROOT": str(root / "cache"),
                    "PPU_CI_K8S_RUNTIME_DIGEST": f"registry.internal/team/sglang@{digest_ok}",
                }
                proc = subprocess.run(
                    ["bash", str(scripts / "run_recipe_cluster.sh")],
                    cwd=workspace,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                pub = json.loads((workspace / "ppu-public/recipe.json").read_text())
                if scenario == "ok":
                    self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
                    self.assertEqual(pub["outcome"], "pass")
                    self.assertEqual(pub["runtime_image"], f"@{digest_ok}")
                    self.assertEqual(pub["queue_s"], 120)
                    retained_dir = (
                        retained
                        / "sglang/logs/sglang-for-sail/77-1"
                        / source_sha[:12]
                        / "fast_sanity/qwen3.8-2.4t-mxfp4"
                    )
                    self.assertTrue(
                        (retained_dir / "ranks/rank-1/runtime.log").is_file()
                    )
                elif scenario == "host_access_denied":
                    self.assertNotEqual(proc.returncode, 0)
                    self.assertEqual(pub["outcome"], "fail")
                    self.assertEqual(pub["failure_phase"], "runner_preflight")
                elif scenario == "digest_mismatch":
                    self.assertNotEqual(proc.returncode, 0)
                    self.assertEqual(pub["outcome"], "fail")
                    self.assertEqual(pub["failure_phase"], "image_resolution")
                else:
                    self.assertNotEqual(proc.returncode, 0)
                    self.assertEqual(pub["outcome"], "fail")
                    self.assertEqual(pub["failure_phase"], "cleanup")

    def test_nightly_workflow_declares_serial_multinode_slot_and_finish_gate(self):
        import yaml

        workflow = yaml.load(
            (REPO_ROOT / ".github/workflows/nightly-test-ppu.yml").read_text(
                encoding="utf-8"
            ),
            Loader=yaml.BaseLoader,
        )
        gate_multi = workflow["jobs"]["pr-gate-multinode"]
        self.assertEqual(gate_multi["uses"], "./.github/workflows/ppu-approval.yml")
        self.assertEqual(gate_multi["with"]["label"], "run-ci-ppu-multinode")
        single = workflow["jobs"]["recipe-sanity"]
        self.assertEqual(single["env"]["RECIPE_NNODES"], "1")
        self.assertEqual(single["env"]["PPU_CI_ALLOW_MULTINODE_HOST_ACCESS"], "false")
        plan_job = workflow["jobs"]["plan"]
        self.assertIn("pr-gate-multinode", plan_job["needs"])
        plan_run = next(s["run"] for s in plan_job["steps"] if s.get("id") == "plan")
        self.assertNotIn("--topology", plan_run)
        self.assertIn("run-ci-ppu-multinode", plan_run)
        self.assertIn("prior_single_run", plan_run)
        for job_obj in (single, workflow["jobs"]["recipe-sanity-multinode"]):
            trusted_checkout = next(
                s
                for s in job_obj["steps"]
                if s.get("with", {}).get("path") == ".ppu-trusted-ci"
            )
            self.assertEqual(
                trusted_checkout["with"]["repository"], "${{ job.workflow_repository }}"
            )
            self.assertEqual(trusted_checkout["with"]["ref"], "${{ job.workflow_sha }}")
            cluster_run = next(
                s["run"] for s in job_obj["steps"] if s.get("id") == "cluster"
            )
            self.assertIn(
                ".ppu-trusted-ci/scripts/ci/ppu/run_recipe_cluster.sh", cluster_run
            )
            cleanup_run = next(
                s["run"]
                for s in job_obj["steps"]
                if "cleanup_recipe_cluster.sh" in s.get("run", "")
            )
            self.assertIn(
                ".ppu-trusted-ci/scripts/ci/ppu/cleanup_recipe_cluster.sh", cleanup_run
            )
        multi = workflow["jobs"]["recipe-sanity-multinode"]
        self.assertEqual(multi["runs-on"], "k8s-runner-group-cpu-thead")
        self.assertEqual(str(multi["strategy"]["max-parallel"]), "1")
        self.assertIn("pr-gate-multinode", multi["needs"])
        self.assertIn("always() && !cancelled()", multi["if"])
        self.assertIn("needs.pr-gate-multinode.outputs.approved == 'true'", multi["if"])
        self.assertIn("needs.recipe-sanity.result != 'cancelled'", multi["if"])
        self.assertNotIn("needs.recipe-sanity.result == 'success'", multi["if"])
        self.assertEqual(multi["env"]["RECIPE_NNODES"], "${{ matrix.nnodes || 2 }}")
        self.assertEqual(
            multi["env"]["RECIPE_PPU_COUNT"], "${{ matrix.ppu_count || 8 }}"
        )
        self.assertEqual(multi["env"]["PPU_CI_ALLOW_MULTINODE_HOST_ACCESS"], "true")
        for secret_key in (
            "PPU_CI_K8S_RUNTIME_DIGEST",
            "PPU_CI_RECIPE_IMAGE_REPO",
            "PPU_CI_RECIPE_ALLOWED_IMAGE_REPOS",
        ):
            self.assertEqual(
                multi["env"][secret_key], f"${{{{ secrets.{secret_key} }}}}"
            )
        self.assertIn("recipe-sanity-finish", workflow["jobs"])
        self.assertIn(
            "jobs.recipe-sanity-finish.outputs.result",
            workflow["on"]["workflow_call"]["outputs"]["sanity-result"]["value"],
        )

    def test_multinode_pod_exports_roce_overrides_and_diagnostics_classify_nccl_causes(
        self,
    ):
        pod_script = (REPO_ROOT / "scripts/ci/ppu/run_recipe_pod.sh").read_text(
            encoding="utf-8"
        )
        for expected in (
            'export MASTER_ADDR="${PPU_CI_DIST_INIT_ADDR%:*}"',
            'export MASTER_PORT="${PPU_CI_DIST_INIT_ADDR##*:}"',
            'export NCCL_SOCKET_IFNAME="=${GLOO_SOCKET_IFNAME}"',
            'export PCCL_SOCKET_IFNAME="=${GLOO_SOCKET_IFNAME}"',
            'export NCCL_IB_HCA="$effective_hcas"',
            'export PCCL_IB_HCA="$effective_hcas"',
            'export PCCL_IB_GID_INDEX="$NCCL_IB_GID_INDEX"',
            "export PCCL_MNNVL_ENABLE=0",
            "export NCCL_MNNVL_ENABLE=0",
            "export PCCL_CUMEM_ENABLE=0",
            "export NCCL_CUMEM_ENABLE=0",
            "export PCCL_NET_GDR_LEVEL=LOC",
            "export NCCL_NET_GDR_LEVEL=LOC",
            "export PCCL_DEBUG=WARN",
            "export NCCL_DEBUG=WARN",
        ):
            self.assertIn(expected, pod_script)
        self.assertEqual(
            diagnostic_codes(
                "multinode: rank=0/2 dist_init_addr=10.0.0.1:29500 gid=3 iface=bond0\n"
                "DistBackendError: ncclCommInitRank unhandled system error cuMem"
            ),
            [
                "nccl_error",
                "nccl_comm_init",
                "nccl_system_error",
                "nccl_cumem",
                "gid_idx_3",
            ],
        )

    def test_failsafe_cleanup_verifies_zero_leftover_resources_after_cancel(self):
        script = REPO_ROOT / "scripts/ci/ppu/cleanup_recipe_cluster.sh"
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            runner_temp = root / "runner-temp"
            runner_temp.mkdir()
            ws = root / "ws"
            infra = ws / ".ppu-infra/cluster-run/scripts"
            infra.mkdir(parents=True)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            stage_file = root / "stage/source.tar.gz"
            stage_file.parent.mkdir()
            stage_file.write_text("tarball", encoding="utf-8")

            # Case 1a: neither state file nor verified marker -> fail-closed missing_state.
            missing = subprocess.run(
                ["bash", str(script)],
                env={
                    **os.environ,
                    "RUNNER_TEMP": str(runner_temp),
                    "GITHUB_WORKSPACE": str(ws),
                },
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertNotEqual(missing.returncode, 0)
            missing_payload = json.loads(missing.stdout.splitlines()[0])
            self.assertFalse(missing_payload["cleanup_verified"])
            self.assertEqual(missing_payload["source"], "missing_state")

            # Case 1b: no state file, but orchestrator wrote verified marker -> pass.
            marker = runner_temp / "ppu-cleanup-verified.json"
            marker.write_text(
                '{"cleanup_verified": true, "schema": "ppu-cleanup/v1", "source": "orchestrator_trap"}\n',
                encoding="utf-8",
            )
            noop = subprocess.run(
                ["bash", str(script)],
                env={
                    **os.environ,
                    "RUNNER_TEMP": str(runner_temp),
                    "GITHUB_WORKSPACE": str(ws),
                },
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(noop.returncode, 0, noop.stderr)
            self.assertTrue(json.loads(noop.stdout)["cleanup_verified"])
            self.assertEqual(json.loads(noop.stdout)["source"], "orchestrator_trap")
            marker.unlink()

            # Case 2: state file present (orchestrator killed mid-cancel) -> failsafe cleans up.
            (infra / "cleanup.sh").write_text(
                '#!/usr/bin/env bash\nrm -f -- "$SOURCE_STAGE_DIR"\nexit 0\n',
                encoding="utf-8",
            )
            (infra / "cleanup.sh").chmod(0o755)
            kubectl = fake_bin / "kubectl"
            kubectl.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            kubectl.chmod(0o755)
            state = runner_temp / "ppu-cleanup-state.json"
            state.write_text(
                json.dumps(
                    {
                        "job_name": "sglang-recipe-deadbeef01234567",
                        "namespace": "ppu-sched",
                        "nnodes": 2,
                        "source_stage_dir": str(stage_file),
                    }
                ),
                encoding="utf-8",
            )
            cleaned = subprocess.run(
                ["bash", str(script)],
                env={
                    **os.environ,
                    "PATH": f"{fake_bin}:{os.environ['PATH']}",
                    "RUNNER_TEMP": str(runner_temp),
                    "GITHUB_WORKSPACE": str(ws),
                },
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(cleaned.returncode, 0, cleaned.stderr)
            payload = json.loads(cleaned.stdout)
            self.assertTrue(payload["cleanup_verified"])
            self.assertEqual(payload["source"], "always_failsafe")
            self.assertEqual(payload["leftover_pods"], 0)
            self.assertEqual(payload["leftover_services"], 0)
            self.assertEqual(payload["leftover_podgroups"], 0)
            self.assertEqual(payload["leftover_staging"], 0)
            self.assertFalse(state.exists())

            # Case 3: leftover pod remains -> failsafe exits non-zero and preserves state file.
            kubectl.write_text(
                '#!/usr/bin/env bash\nif [[ "$*" == *"get pods"* ]]; then echo "pod/stray-worker-0"; fi\nexit 0\n',
                encoding="utf-8",
            )
            state.write_text(
                json.dumps(
                    {
                        "job_name": "sglang-recipe-deadbeef01234567",
                        "namespace": "ppu-sched",
                        "nnodes": 2,
                        "source_stage_dir": str(stage_file),
                    }
                ),
                encoding="utf-8",
            )
            leaked = subprocess.run(
                ["bash", str(script)],
                env={
                    **os.environ,
                    "PATH": f"{fake_bin}:{os.environ['PATH']}",
                    "RUNNER_TEMP": str(runner_temp),
                    "GITHUB_WORKSPACE": str(ws),
                },
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertNotEqual(leaked.returncode, 0)
            payload_leaked = json.loads(leaked.stdout.splitlines()[0])
            self.assertFalse(payload_leaked["cleanup_verified"])
            self.assertEqual(payload_leaked["leftover_pods"], 1)
            self.assertTrue(state.exists())

    def test_pp_stage_shard_selection_and_synchronized_warm(self):
        import concurrent.futures
        from unittest import mock

        import recipe_runner

        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            snap = root / "snapshot"
            snap.mkdir()
            (snap / "config.json").write_text(
                json.dumps({"num_hidden_layers": 92}), encoding="utf-8"
            )
            weight_map = {
                "model.embed_tokens.weight": "shard-0.safetensors",
                "model.layers.0.mlp.weight": "shard-0.safetensors",
                "model.layers.45.mlp.weight": "shard-1.safetensors",
                "model.layers.46.mlp.weight": "shard-2.safetensors",
                "model.layers.91.mlp.weight": "shard-3.safetensors",
                "model.norm.weight": "shard-3.safetensors",
                "lm_head.weight": "shard-3.safetensors",
                "mtp.layers.0.weight": "shard-mtp.safetensors",
            }
            for fname in set(weight_map.values()):
                (snap / fname).write_bytes(b"x" * 1024)
            (snap / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": weight_map}), encoding="utf-8"
            )

            r0_shards, r0_layers, total = recipe_runner.select_pp_stage_shards(
                snap, pp_size=2, pp_rank=0
            )
            r1_shards, r1_layers, _ = recipe_runner.select_pp_stage_shards(
                snap, pp_size=2, pp_rank=1
            )
            self.assertEqual(total, 5)
            self.assertEqual(r0_layers, (0, 46))
            self.assertEqual(r1_layers, (46, 92))
            self.assertEqual(
                [p.name for p in r0_shards],
                ["shard-0.safetensors", "shard-1.safetensors"],
            )
            self.assertEqual(
                [p.name for p in r1_shards],
                ["shard-2.safetensors", "shard-3.safetensors"],
            )

            # Two ranks warm disjoint halves and synchronize via barrier_dir.
            barrier = root / "barrier"
            results_dir = root / "art/results"
            results_dir.mkdir(parents=True)
            flags = ["--tp-size", "8", "--pp-size", "2"]
            with mock.patch.object(recipe_runner, "RESULTS", results_dir):
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    f0 = pool.submit(
                        recipe_runner.warm_pp_stage_shards,
                        snap,
                        flags=flags,
                        nnodes=2,
                        node_rank=0,
                        barrier_dir=barrier,
                        max_workers=2,
                        poll_s=0.05,
                    )
                    f1 = pool.submit(
                        recipe_runner.warm_pp_stage_shards,
                        snap,
                        flags=flags,
                        nnodes=2,
                        node_rank=1,
                        barrier_dir=barrier,
                        max_workers=2,
                        poll_s=0.05,
                    )
                    t0 = f0.result(timeout=10)
                    t1 = f1.result(timeout=10)
            self.assertGreater(t0, 0.0)
            self.assertGreater(t1, 0.0)
            self.assertTrue((barrier / "warm-rank-0.json").is_file())
            self.assertTrue((barrier / "warm-rank-1.json").is_file())
            self.assertIn(
                "pp_warm_done", (results_dir / "page-cache-warm.log").read_text()
            )

    def test_probes_respects_always_thinking_override_and_chat_template_kwargs(self):
        from unittest import mock

        import recipe_runner

        captured = []

        class FakeHeaders:
            def __init__(self, ctype):
                self._ctype = ctype

            def get_content_type(self):
                return self._ctype

        class FakeResp:
            def __init__(self, body, ctype):
                self._lines = body
                self.headers = FakeHeaders(ctype)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b"".join(self._lines)

            def __iter__(self):
                return iter(self._lines)

        def fake_request(path, data=None, timeout=15):
            payload = json.loads(data.decode())
            captured.append(payload)
            if payload.get("stream"):
                return FakeResp(
                    [
                        b'data: {"choices":[{"delta":{"content":"42"},"finish_reason":"stop"}]}\n',
                        b"data: [DONE]\n",
                    ],
                    "text/event-stream",
                )
            return FakeResp(
                [
                    json.dumps(
                        {
                            "choices": [
                                {
                                    "message": {
                                        "content": "42",
                                        "reasoning_content": "6 * 7 = 42",
                                    },
                                    "finish_reason": "stop",
                                }
                            ]
                        }
                    ).encode()
                ],
                "application/json",
            )

        with tempfile.TemporaryDirectory() as raw_root:
            results_dir = Path(raw_root) / "art/results"
            results_dir.mkdir(parents=True)
            with (
                mock.patch.object(recipe_runner, "RESULTS", results_dir),
                mock.patch.object(recipe_runner, "request", side_effect=fake_request),
            ):
                recipe_runner.probes(
                    {
                        "prompt": "Compute 6 * 7",
                        "expected_substring": "42",
                        "max_tokens": 1024,
                        "thinking": None,
                        "chat_template_kwargs": {"reasoning_effort": "low"},
                    },
                    fast=False,
                )
                recipe_runner.probes(
                    {
                        "prompt": "Compute 6 * 7",
                        "expected_substring": "42",
                        "max_tokens": 128,
                    },
                    fast=False,
                )
        self.assertNotIn("thinking", captured[0])
        self.assertEqual(
            captured[0]["chat_template_kwargs"], {"reasoning_effort": "low"}
        )
        self.assertEqual(captured[2]["thinking"], {"type": "disabled"})

    def test_qwen38_24t_mxfp4_golden_baseline_matches_policy_contract(self):
        model = "qwen3.8-2.4t-mxfp4"
        policy_path = DEFAULT_POLICY_ROOT / f"models/{model}.yaml"
        policy = _load_yaml(policy_path)
        config = resolve_recipe(
            REPO_ROOT / policy["recipe"], policy_path=policy_path, tier="model_sanity"
        )
        golden_path = (
            REPO_ROOT
            / "test/ppu/golden"
            / f"accuracy_{config['golden_model_key']}_890p.json"
        )
        golden = json.loads(golden_path.read_text())
        stage = next(
            s
            for s in policy["tiers"]["model_sanity"]["stages"]
            if s["kind"] == "evalscope"
        )
        self.assertEqual(golden["schema_version"], 2)
        self.assertEqual(golden["benchmark"], "gsm8k")
        self.assertEqual(golden["chip"], "890p")
        self.assertEqual(golden["model"], config["golden_model_key"])
        self.assertEqual(golden["checkpoint"], config["hf_model_id"])
        self.assertEqual(golden["serving"]["ppu_count"], 16)
        self.assertEqual(golden["serving"]["nnodes"], 2)
        self.assertEqual(golden["generation"], stage["generation"])
        self.assertEqual(golden["dataset"]["dataset_id"], stage["dataset"])
        self.assertEqual(golden["dataset"]["num_expected"], 1319)
        self.assertEqual(golden["tolerance"], {"mode": "band", "abs": 0.02})
        self.assertEqual(golden["score"], 0.9757)
        self.assertEqual(golden["score_num"], 1287)

    def test_orchestrator_aborts_before_submit_on_cleanup_state_write_failure(self):
        script = (REPO_ROOT / "scripts/ci/ppu/run_recipe_cluster.sh").read_text(
            encoding="utf-8"
        )
        # Durable cleanup_state write must happen BEFORE tar and MUST NOT use
        # a single `&&` chain that suppresses left-hand write failures.
        self.assertLess(
            script.index('> "${cleanup_state}.tmp"'),
            script.index('tar -czf "$SOURCE_STAGE_DIR"'),
        )
        self.assertNotIn('> "${cleanup_state}.tmp" &&', script)

        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            workspace = root / "ws"
            scripts_dir = workspace / "scripts/ci/ppu"
            scripts_dir.mkdir(parents=True)
            for name in (
                "run_recipe_cluster.sh",
                "recipe_image.py",
                "recipe_bundle.py",
                "recipe_report.py",
                "recipe_publish.py",
                "recipe_retention.py",
                "recipe_diagnostics.py",
                "ppu_recipe.py",
            ):
                src = REPO_ROOT / "scripts/ci/ppu" / name
                dst = scripts_dir / name
                dst.write_bytes(src.read_bytes())
                if src.stat().st_mode & 0o111:
                    dst.chmod(0o755)
            (workspace / "test/ppu/policy/models").mkdir(parents=True)
            (workspace / "test/ppu/policy/models/qwen3.8-2.4t-mxfp4.yaml").write_text(
                (
                    REPO_ROOT / "test/ppu/policy/models/qwen3.8-2.4t-mxfp4.yaml"
                ).read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
            subprocess.run(
                ["git", "config", "user.email", "ci@example.com"],
                cwd=workspace,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "CI"], cwd=workspace, check=True
            )
            subprocess.run(["git", "add", "."], cwd=workspace, check=True)
            subprocess.run(
                ["git", "commit", "-q", "-m", "init"], cwd=workspace, check=True
            )
            source_sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=workspace, text=True
            ).strip()

            infra = workspace / ".ppu-infra/cluster-run/scripts"
            infra.mkdir(parents=True)
            submit_marker = root / "submit-called"
            (infra / "submit.sh").write_text(
                f'#!/usr/bin/env bash\ntouch "{submit_marker}"\nexit 0\n',
                encoding="utf-8",
            )
            (infra / "submit.sh").chmod(0o755)
            (infra / "cleanup.sh").write_text("#!/usr/bin/env bash\nexit 0\n")
            (infra / "cleanup.sh").chmod(0o755)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            (fake_bin / "kubectl").write_text("#!/usr/bin/env bash\nexit 0\n")
            (fake_bin / "kubectl").chmod(0o755)
            (fake_bin / "git").write_text(
                "#!/usr/bin/env bash\n"
                'if [[ "$1" == "-C" && "$3" == "rev-parse" ]]; then echo 96e25147971f5d2a04d828f8ad508cae38a4fa8a; exit 0; fi\n'
                'exec /usr/bin/git "$@"\n'
            )
            (fake_bin / "git").chmod(0o755)

            retained = root / "retained"
            retained.mkdir(mode=0o700)
            staging = root / "staging"
            staging.mkdir(mode=0o700)
            ro_temp = root / "ro-temp"
            ro_temp.mkdir(mode=0o500)
            digest_ok = "sha256:" + "a" * 64
            try:
                proc = subprocess.run(
                    ["bash", str(scripts_dir / "run_recipe_cluster.sh")],
                    cwd=workspace,
                    env={
                        **os.environ,
                        "PATH": f"{fake_bin}:{os.environ['PATH']}",
                        "TIER": "fast_sanity",
                        "MODEL_KEY": "qwen3.8-2.4t-mxfp4",
                        "RECIPE_PPU_COUNT": "8",
                        "RECIPE_NNODES": "2",
                        "PPU_CI_ALLOW_MULTINODE_HOST_ACCESS": "true",
                        "PPU_CI_SKIP_WHEEL_INSTALL": "true",
                        "PPU_CI_RUNTIME_IMAGE_DIGEST": digest_ok,
                        "PPU_CI_K8S_RUNTIME_DIGEST": f"registry.internal/team/sglang@{digest_ok}",
                        "PPU_CI_RECIPE_IMAGE_REPO": "registry.internal/team/sglang",
                        "PPU_CI_RECIPE_ALLOWED_IMAGE_REPOS": "registry.internal/team/",
                        "PPU_PUBLIC_SUMMARY": str(root / "summary.md"),
                        "GITHUB_WORKSPACE": str(workspace),
                        "RUNNER_TEMP": str(ro_temp),
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
                    },
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
            finally:
                ro_temp.chmod(0o700)
            self.assertNotEqual(proc.returncode, 0)
            self.assertFalse(submit_marker.exists())
            self.assertEqual(list(staging.glob("**/source.tar.gz")), [])

    def test_retain_rejects_intermediate_rank_symlinks_and_archives_warmup_logs(self):
        import recipe_retention

        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            outside = root / "outside"
            outside.mkdir()
            secret = outside / "runtime.log"
            secret.write_text("SECRET_OUTSIDE_JOB\n", encoding="utf-8")

            src = root / "scratch"
            (src / "results").mkdir(parents=True)
            (src / "launcher.log").write_text("launcher\n", encoding="utf-8")
            (src / "provenance.json").write_text("{}\n", encoding="utf-8")
            (src / "results/summary.json").write_text(
                json.dumps({"outcome": "pass"}), encoding="utf-8"
            )
            (src / "results/page-cache-warm.log").write_text(
                "pp_warm_done rank=0/2 shards=106/213\n", encoding="utf-8"
            )
            # Legitimate rank-1 warm log + symlinked rank-2 directory pointing outside.
            (src / "ranks/rank-1/results").mkdir(parents=True)
            (src / "ranks/rank-1/results/page-cache-warm.log").write_text(
                "pp_warm_done rank=1/2 shards=107/213\n", encoding="utf-8"
            )
            (src / "ranks/rank-2").symlink_to(outside)

            archive_root = root / "archive"
            archive_root.mkdir(mode=0o700)
            sha = "a" * 40
            recipe_retention.retain(
                {
                    "PPU_CI_SOURCE_SHA": sha,
                    "TIER": "model_sanity",
                    "MODEL_KEY": "qwen3.8-2.4t-mxfp4",
                    "GITHUB_RUN_ID": "88",
                    "GITHUB_RUN_ATTEMPT": "1",
                    "PPU_CI_PRIVATE_RESULTS_ROOT": str(archive_root),
                    "PPU_CI_ARTIFACT_DIR": str(src),
                    "RUNNER_TEMP": str(root / "runner-temp"),
                    "GITHUB_WORKSPACE": str(root / "workspace"),
                    "RECIPE_OUTCOME": "success",
                }
            )
            dest = (
                archive_root
                / "sglang/logs/sglang-for-sail/88-1"
                / sha[:12]
                / "model_sanity/qwen3.8-2.4t-mxfp4"
            )
            index = json.loads((dest / "index.json").read_text(encoding="utf-8"))
            self.assertIn("page-cache-warm.log", index["files"])
            self.assertIn("ranks/rank-1/results/page-cache-warm.log", index["files"])
            self.assertNotIn("ranks/rank-2/runtime.log", index["files"])
            self.assertFalse((dest / "ranks/rank-2/runtime.log").exists())

    def test_gang_snapshot_resolution_locks_single_revision_across_ranks(self):
        import concurrent.futures
        from unittest import mock

        import recipe_runner

        rev_a = "a" * 40
        rev_b = "b" * 40
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            barrier = root / "barrier"
            state = {"current": rev_a}

            def fake_cached(repo_id, revision):
                resolved = state["current"] if revision is None else revision
                return f"/cache/snapshots/{resolved}", resolved

            with mock.patch.object(
                recipe_runner, "cached_model_path", side_effect=fake_cached
            ):
                p0, r0 = recipe_runner._resolve_gang_snapshot(
                    "T-HEAD/Qwen3.8-2.4T-A95B-FP8-mxfp4-fp8-030",
                    None,
                    nnodes=2,
                    node_rank=0,
                    barrier_dir=barrier,
                    poll_s=0.02,
                )
                # Mutate refs/main between rank 0 and rank 1 resolution.
                state["current"] = rev_b
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    p1, r1 = pool.submit(
                        recipe_runner._resolve_gang_snapshot,
                        "T-HEAD/Qwen3.8-2.4T-A95B-FP8-mxfp4-fp8-030",
                        None,
                        nnodes=2,
                        node_rank=1,
                        barrier_dir=barrier,
                        poll_s=0.02,
                    ).result(timeout=5)
            self.assertEqual((p0, r0), (p1, r1))
            self.assertEqual(r1, rev_a)

    def test_stop_process_group_reaps_surviving_children_after_leader_sigkill(self):
        import signal
        import time

        import recipe_runner

        with tempfile.TemporaryDirectory() as raw_root:
            pid_file = Path(raw_root) / "child.pid"
            # Leader spawns a child in the same process group and waits.
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    (
                        "import os, subprocess, sys, time; "
                        f"c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
                        f"open({str(pid_file)!r}, 'w').write(str(c.pid)); "
                        "time.sleep(60)"
                    ),
                ],
                start_new_session=True,
            )
            deadline = time.monotonic() + 5.0
            while not pid_file.exists() or not pid_file.read_text().strip():
                if time.monotonic() > deadline:
                    proc.kill()
                    self.fail("child pid file not created")
                time.sleep(0.05)
            child_pid = int(pid_file.read_text().strip())
            # Kill ONLY the group leader, leaving its child alive in the group.
            os.kill(proc.pid, signal.SIGKILL)
            proc.wait(timeout=5)
            os.kill(child_pid, 0)  # Child is still alive before cleanup.

            sent, rc = recipe_runner._stop_process_group(proc)
            self.assertFalse(sent)
            self.assertEqual(rc, -signal.SIGKILL)
            # Child must be dead after _stop_process_group.
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                try:
                    os.kill(child_pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.05)
            else:
                os.kill(child_pid, signal.SIGKILL)
                self.fail("child process survived after _stop_process_group")

    def test_pp_layer_indices_match_sglang_runtime_remainder_and_partition_override(
        self,
    ):
        import recipe_runner

        # 5 layers across PP2 -> runtime puts remainder on later stages: [0, 2) and [2, 5).
        self.assertEqual(
            recipe_runner._pp_layer_indices(5, pp_size=2, pp_rank=0), (0, 2)
        )
        self.assertEqual(
            recipe_runner._pp_layer_indices(5, pp_size=2, pp_rank=1), (2, 5)
        )
        # Custom SGLANG_PP_LAYER_PARTITION override is honored.
        env = {"SGLANG_PP_LAYER_PARTITION": "1,4"}
        self.assertEqual(
            recipe_runner._pp_layer_indices(5, pp_size=2, pp_rank=0, server_env=env),
            (0, 1),
        )
        self.assertEqual(
            recipe_runner._pp_layer_indices(5, pp_size=2, pp_rank=1, server_env=env),
            (1, 5),
        )

    def test_recent_failures_ignores_healthy_multinode_run_and_surfaces_prior_oom(self):
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            base = root / "sglang/logs/sglang-for-sail"
            # Run 100: failed with OOM.
            r100 = base / "100-1" / ("a" * 12) / "model_sanity/qwen3.8-2.4t-mxfp4"
            r100.mkdir(parents=True)
            (r100 / "index.json").write_text("{}")
            (r100 / "runtime.log").write_text("CUDA out of memory\n", encoding="utf-8")
            # Run 101: later healthy run with outcome=pass and rank-1.status=0.
            r101 = base / "101-1" / ("b" * 12) / "model_sanity/qwen3.8-2.4t-mxfp4"
            (r101 / "ranks").mkdir(parents=True)
            (r101 / "index.json").write_text("{}")
            (r101 / "summary.json").write_text(
                json.dumps({"outcome": "pass"}), encoding="utf-8"
            )
            (r101 / "ranks/rank-1.status").write_text("0\n", encoding="utf-8")

            records = recent_failures(root, 102, model_filter="qwen3.8-2.4t-mxfp4")
            self.assertEqual(
                records,
                [
                    {
                        "run_id": 100,
                        "tier": "model_sanity",
                        "model": "qwen3.8-2.4t-mxfp4",
                        "codes": ["cuda_oom", "worker_oom"],
                    }
                ],
            )

    def test_recent_failures_reports_cleanup_failed_run_even_when_worker_summary_passed(
        self,
    ):
        import recipe_retention

        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            archive_root = root / "archive"
            archive_root.mkdir(mode=0o700)
            base = archive_root / "sglang/logs/sglang-for-sail"
            # Run 100: older OOM.
            r100 = base / "100-1" / ("a" * 12) / "model_sanity/qwen3.8-2.4t-mxfp4"
            r100.mkdir(parents=True)
            (r100 / "index.json").write_text(
                json.dumps({"job_outcome": "fail"}), encoding="utf-8"
            )
            (r100 / "runtime.log").write_text("CUDA out of memory\n", encoding="utf-8")

            # Run 102: worker summary passed, but later cluster cleanup failed
            # (RECIPE_OUTCOME=failure) so index.json records job_outcome=fail.
            src = root / "scratch-102"
            (src / "results").mkdir(parents=True)
            (src / "launcher.log").write_text(
                "Command failed during cleanup\n", encoding="utf-8"
            )
            (src / "recipe-phase.txt").write_text("cleanup\n", encoding="utf-8")
            (src / "results/summary.json").write_text(
                json.dumps({"outcome": "pass"}), encoding="utf-8"
            )
            recipe_retention.retain(
                {
                    "PPU_CI_SOURCE_SHA": "c" * 40,
                    "TIER": "model_sanity",
                    "MODEL_KEY": "qwen3.8-2.4t-mxfp4",
                    "GITHUB_RUN_ID": "102",
                    "GITHUB_RUN_ATTEMPT": "1",
                    "PPU_CI_PRIVATE_RESULTS_ROOT": str(archive_root),
                    "PPU_CI_ARTIFACT_DIR": str(src),
                    "RUNNER_TEMP": str(root / "runner-temp"),
                    "GITHUB_WORKSPACE": str(root / "workspace"),
                    "RECIPE_OUTCOME": "failure",
                }
            )
            records = recent_failures(
                archive_root, 103, model_filter="qwen3.8-2.4t-mxfp4"
            )
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["run_id"], 102)
            self.assertIn("worker_exit", records[0]["codes"])

    def test_failsafe_cleanup_cleans_empty_stage_dir_and_is_idempotent_on_retry(self):
        script = REPO_ROOT / "scripts/ci/ppu/cleanup_recipe_cluster.sh"
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            runner_temp = root / "runner-temp"
            runner_temp.mkdir()
            ws = root / "ws"
            infra = ws / ".ppu-infra/cluster-run/scripts"
            infra.mkdir(parents=True)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            (infra / "cleanup.sh").write_text("#!/usr/bin/env bash\nexit 0\n")
            (infra / "cleanup.sh").chmod(0o755)
            (fake_bin / "kubectl").write_text("#!/usr/bin/env bash\nexit 0\n")
            (fake_bin / "kubectl").chmod(0o755)

            job_name = "sglang-recipe-deadbeef01234567"
            empty_stage_dir = root / f"{job_name}.abcdef12"
            empty_stage_dir.mkdir()
            state = runner_temp / "ppu-cleanup-state.json"
            state.write_text(
                json.dumps(
                    {
                        "job_name": job_name,
                        "namespace": "ppu-sched",
                        "nnodes": 2,
                        "source_stage_dir": str(empty_stage_dir / "source.tar.gz"),
                    }
                ),
                encoding="utf-8",
            )
            env = {
                **os.environ,
                "PATH": f"{fake_bin}:{os.environ['PATH']}",
                "RUNNER_TEMP": str(runner_temp),
                "GITHUB_WORKSPACE": str(ws),
            }
            first = subprocess.run(
                ["bash", str(script)],
                env=env,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            self.assertTrue(json.loads(first.stdout)["cleanup_verified"])
            self.assertFalse(empty_stage_dir.exists())
            self.assertFalse(state.exists())

            # Immediate retry must succeed via the persisted verified marker.
            second = subprocess.run(
                ["bash", str(script)],
                env=env,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
            self.assertTrue(json.loads(second.stdout)["cleanup_verified"])

    def test_select_explicit_filter_ignores_unrelated_unsupported_topology(self):
        from unittest import mock

        import recipe_select

        with tempfile.TemporaryDirectory() as raw_root:
            policy_root = Path(raw_root) / "policy"
            models_dir = policy_root / "models"
            models_dir.mkdir(parents=True)
            glm_policy = DEFAULT_POLICY_ROOT / "models/glm-5.3.yaml"
            (models_dir / "glm-5.3.yaml").write_text(
                glm_policy.read_text(encoding="utf-8"), encoding="utf-8"
            )
            (models_dir / "disagg-demo.yaml").write_text(
                glm_policy.read_text(encoding="utf-8"), encoding="utf-8"
            )
            orig_topo = recipe_select.cluster_topology

            def fake_topo(path, tier):
                if path.stem == "disagg-demo":
                    raise ValueError(
                        "unsupported cluster topology for policy disagg-demo"
                    )
                return orig_topo(path, tier)

            with (
                mock.patch.object(recipe_select, "DEFAULT_POLICY_ROOT", policy_root),
                mock.patch.object(recipe_select, "validate_paths"),
                mock.patch.object(
                    recipe_select, "cluster_topology", side_effect=fake_topo
                ),
            ):
                selected = recipe_select.select("model_sanity", "glm-5.3")
                self.assertEqual(
                    selected, {"include": [{"model": "glm-5.3", "ppu_count": 8}]}
                )
                with self.assertRaises(ValueError):
                    recipe_select.select("model_sanity", "disagg-demo")

    def test_run_worker_node_reaps_child_on_early_exit_and_propagates_rank0_error(self):
        import signal
        import time
        from unittest import mock

        import recipe_runner

        # 1) Leader SIGKILLed early -> run_worker_node must still reap child.
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            results_dir = root / "art/results"
            results_dir.mkdir(parents=True)
            barrier = root / "barrier"
            barrier.mkdir()
            pid_file = root / "child.pid"
            launcher_cmd = [
                sys.executable,
                "-c",
                (
                    "import os, signal, subprocess, sys, time; "
                    "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
                    f"open({str(pid_file)!r}, 'w').write(str(c.pid)); "
                    "os.kill(os.getpid(), signal.SIGKILL)"
                ),
            ]
            with mock.patch.object(recipe_runner, "RESULTS", results_dir):
                with self.assertRaises(recipe_runner.StartupError):
                    recipe_runner.run_worker_node(
                        {"argv": launcher_cmd, "server_env": {}},
                        "fast_full",
                        nnodes=2,
                        node_rank=1,
                        dist_init_addr="127.0.0.1:29500",
                        barrier_dir=barrier,
                        poll_s=0.05,
                    )
            child_pid = int(pid_file.read_text().strip())
            exit_doc = json.loads((barrier / "exit-rank-1.json").read_text())
            self.assertEqual(exit_doc["exit_code"], -signal.SIGKILL)
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                try:
                    os.kill(child_pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.05)
            else:
                os.kill(child_pid, signal.SIGKILL)
                self.fail("rank-1 child survived run_worker_node early exit")

        # 2) rank0.done with status=error -> live rank-1 returns fail (not pass).
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            results_dir = root / "art/results"
            results_dir.mkdir(parents=True)
            barrier = root / "barrier"
            barrier.mkdir()
            (barrier / "rank0.done").write_text(
                json.dumps({"status": "error"}), encoding="utf-8"
            )
            with mock.patch.object(recipe_runner, "RESULTS", results_dir):
                outcome = recipe_runner.run_worker_node(
                    {
                        "argv": [sys.executable, "-c", "import time; time.sleep(60)"],
                        "server_env": {},
                    },
                    "fast_full",
                    nnodes=2,
                    node_rank=1,
                    dist_init_addr="127.0.0.1:29500",
                    barrier_dir=barrier,
                    poll_s=0.05,
                )
            self.assertEqual(outcome["outcome"], "fail")
            exit_doc = json.loads((barrier / "exit-rank-1.json").read_text())
            self.assertEqual(exit_doc["exit_code"], 1)
            self.assertEqual(exit_doc["reason"], "rank0_error")

    def test_server_tears_down_local_process_when_rank0_barrier_write_fails(self):
        import errno
        import time
        from unittest import mock

        import recipe_runner

        spawned = {}
        real_popen = subprocess.Popen

        def tracking_popen(*args, **kwargs):
            proc = real_popen(*args, **kwargs)
            spawned["proc"] = proc
            return proc

        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            results_dir = root / "art/results"
            results_dir.mkdir(parents=True)
            barrier = root / "barrier"
            barrier.mkdir()

            class FakeResp:
                status = 200

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return False

            with (
                mock.patch.object(recipe_runner, "RESULTS", results_dir),
                mock.patch.object(recipe_runner, "request", return_value=FakeResp()),
                mock.patch.object(
                    recipe_runner.subprocess, "Popen", side_effect=tracking_popen
                ),
                mock.patch.object(
                    recipe_runner,
                    "_barrier_write",
                    side_effect=OSError(errno.ENOSPC, "No space left on device"),
                ),
            ):
                with self.assertRaises(OSError):
                    with recipe_runner.server(
                        {
                            "argv": [
                                sys.executable,
                                "-c",
                                "import time; time.sleep(60)",
                            ],
                            "server_env": {},
                        },
                        "fast_full",
                        {},
                        10,
                        nnodes=2,
                        node_rank=0,
                        dist_init_addr="127.0.0.1:29500",
                        barrier_dir=barrier,
                        signal_done_on_exit=True,
                    ):
                        pass
            proc = spawned["proc"]
            deadline = time.monotonic() + 3.0
            while proc.poll() is None and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertIsNotNone(proc.poll())

    def test_failure_reason_attributes_failed_peer_rank_over_rank0_startup(self):
        import recipe_diagnostics

        with tempfile.TemporaryDirectory() as raw_root:
            src = Path(raw_root)
            (src / "worker-phase.txt").write_text("server_startup\n", encoding="utf-8")
            (src / "ranks/barrier").mkdir(parents=True)
            (src / "ranks/rank-1").mkdir(parents=True)
            (src / "ranks/rank-1.status").write_text("1\n", encoding="utf-8")
            (src / "ranks/barrier/fail-rank-1.json").write_text(
                json.dumps({"rank": 1, "exit_code": 1}), encoding="utf-8"
            )
            (src / "ranks/rank-1/worker-phase.txt").write_text(
                "multinode_network\n", encoding="utf-8"
            )
            self.assertEqual(
                recipe_diagnostics.failure_reason(src), "multinode_network"
            )
            # Cleanup precedence remains highest.
            (src / "recipe-phase.txt").write_text("cleanup\n", encoding="utf-8")
            self.assertEqual(recipe_diagnostics.failure_reason(src), "cleanup")

    def test_gid_selector_emits_matched_hca_list_when_majority_excludes_outlier(self):
        script = REPO_ROOT / "scripts/ci/ppu/recipe_multinode_gid_index.sh"
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            sysfs = root / "infiniband"
            for name, slot in (
                ("mlx5_bond_0", 3),
                ("mlx5_bond_1", 3),
                ("mlx5_bond_2", 5),
            ):
                gids = sysfs / name / "ports/1/gids"
                types = sysfs / name / "ports/1/gid_attrs/types"
                gids.mkdir(parents=True)
                types.mkdir(parents=True)
                (sysfs / name / "ports/1/state").write_text("4: ACTIVE\n")
                (gids / str(slot)).write_text(
                    "fd03:0000:0000:0000:0000:0000:0000:0001\n"
                )
                (types / str(slot)).write_text("RoCE v2\n")
            hca_out = root / "hca.txt"
            proc = subprocess.run(
                ["bash", str(script), "--hca-list-file", str(hca_out)],
                env={**os.environ, "RECIPE_GID_SYSFS_ROOT": str(sysfs)},
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(proc.stdout.strip(), "3")
            self.assertEqual(hca_out.read_text().strip(), "=mlx5_bond_0,mlx5_bond_1")

            # Verify run_recipe_pod.sh HCA list intersection against NCCL global
            # prefix & per-entry port semantics.
            pod_text = (REPO_ROOT / "scripts/ci/ppu/run_recipe_pod.sh").read_text(
                encoding="utf-8"
            )
            hca_block = pod_text[
                pod_text.index('if [[ -s "$hca_list_file"') : pod_text.index(
                    "    GLOO_SOCKET_IFNAME="
                )
            ]

            def nccl_accepted(value: str) -> list[str]:
                invert = value.startswith("^")
                value = value.removeprefix("^")
                exact = value.startswith("=")
                value = value.removeprefix("=")
                tokens = [entry.partition(":") for entry in value.split(",") if entry]

                def selected(dev_name: str, port_num: int) -> bool:
                    return any(
                        (dev_name == dev if exact else dev_name.startswith(dev))
                        and (not suffix or int(suffix) == port_num)
                        for dev, _, suffix in tokens
                    )

                return [
                    f"{dev_name}:{port_num}"
                    for dev_name in ("mlx5_bond_0", "mlx5_bond_1", "mlx5_bond_2")
                    for port_num in (1, 2)
                    if selected(dev_name, port_num) != invert
                ]

            for existing, expected_export, expected_matches in (
                (
                    "",
                    "=mlx5_bond_0,mlx5_bond_1",
                    [
                        "mlx5_bond_0:1",
                        "mlx5_bond_0:2",
                        "mlx5_bond_1:1",
                        "mlx5_bond_1:2",
                    ],
                ),
                ("=mlx5_bond_0:1", "=mlx5_bond_0:1", ["mlx5_bond_0:1"]),
                (
                    "mlx5_bond_:1",
                    "=mlx5_bond_0:1,mlx5_bond_1:1",
                    ["mlx5_bond_0:1", "mlx5_bond_1:1"],
                ),
                (
                    "^mlx5_bond_1",
                    "=mlx5_bond_0",
                    ["mlx5_bond_0:1", "mlx5_bond_0:2"],
                ),
            ):
                res = subprocess.run(
                    [
                        "bash",
                        "-c",
                        hca_block + '\nprintf "%s" "$NCCL_IB_HCA"',
                    ],
                    env={
                        **os.environ,
                        "hca_list_file": str(hca_out),
                        "RECIPE_GID_SYSFS_ROOT": str(sysfs),
                        "NCCL_IB_HCA": existing,
                        "PCCL_IB_HCA": "",
                    },
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self.assertEqual(res.returncode, 0, res.stderr)
                self.assertEqual(res.stdout, expected_export)
                self.assertEqual(nccl_accepted(res.stdout), expected_matches)

    def test_trusted_cluster_entrypoints_and_isolated_python_helpers(self):
        import re

        import recipe_image
        import yaml

        pr_wf = yaml.load(
            (REPO_ROOT / ".github/workflows/pr-test-ppu.yml").read_text(
                encoding="utf-8"
            ),
            Loader=yaml.BaseLoader,
        )
        nightly_wf = yaml.load(
            (REPO_ROOT / ".github/workflows/nightly-test-ppu.yml").read_text(
                encoding="utf-8"
            ),
            Loader=yaml.BaseLoader,
        )
        # Ordinary PR cluster preflight must check out the trusted workflow SHA, never candidate SHA.
        pr_cfg = pr_wf["jobs"]["ppu-cluster-config"]
        self.assertEqual(
            pr_cfg["steps"][0]["with"]["repository"], "${{ job.workflow_repository }}"
        )
        self.assertEqual(pr_cfg["steps"][0]["with"]["ref"], "${{ job.workflow_sha }}")
        self.assertIn("python3 -I -B", pr_cfg["steps"][-1]["run"])

        pr_legacy = pr_wf["jobs"]["ppu-legacy-sanity"]
        self.assertEqual(
            pr_legacy["env"]["PPU_CI_ALLOW_MULTINODE_HOST_ACCESS"], "false"
        )
        legacy_trusted = next(
            s
            for s in pr_legacy["steps"]
            if s.get("with", {}).get("path") == ".ppu-trusted-ci"
        )
        self.assertEqual(
            legacy_trusted["with"]["repository"], "${{ job.workflow_repository }}"
        )
        self.assertEqual(legacy_trusted["with"]["ref"], "${{ job.workflow_sha }}")
        self.assertIn(
            ".ppu-trusted-ci/scripts/ci/ppu/run_recipe_cluster.sh",
            next(s["run"] for s in pr_legacy["steps"] if s.get("id") == "cluster"),
        )
        self.assertIn(
            "python3 -I -B",
            nightly_wf["jobs"]["cluster-preflight"]["steps"][-1]["run"],
        )

        # Every python3 invocation in host-side shell scripts must run isolated (-I -B).
        for script_name in ("run_recipe_cluster.sh", "cleanup_recipe_cluster.sh"):
            text = (REPO_ROOT / "scripts/ci/ppu" / script_name).read_text(
                encoding="utf-8"
            )
            self.assertEqual(re.findall(r"\bpython3\s+(?!-I\s+-B\b)", text), [])

        # Host-side Python helpers must not import poisoned modules from candidate CWD.
        with tempfile.TemporaryDirectory() as raw_root:
            poison_cwd = Path(raw_root)
            (poison_cwd / "sitecustomize.py").write_text(
                "raise RuntimeError('poisoned sitecustomize')\n", encoding="utf-8"
            )
            (poison_cwd / "recipe_diagnostics.py").write_text(
                "raise RuntimeError('poisoned sibling')\n", encoding="utf-8"
            )
            for helper in ("recipe_publish.py", "recipe_image.py", "recipe_bundle.py"):
                proc = subprocess.run(
                    [
                        sys.executable,
                        "-I",
                        "-B",
                        str(REPO_ROOT / "scripts/ci/ppu" / helper),
                        "--help",
                    ],
                    cwd=poison_cwd,
                    env={**os.environ, "PYTHONPATH": str(poison_cwd)},
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self.assertEqual(proc.returncode, 0, proc.stderr)

            # Ordinary worker pod verification rejects host namespaces and privileged mode.
            digest = "sha256:" + "a" * 64
            env = {
                "PPU_CI_K8S_RUNTIME_DIGEST": f"registry.internal/team/sglang@{digest}",
                "PPU_CI_RECIPE_ALLOWED_IMAGE_REPOS": "registry.internal/team/",
                "PPU_CI_ALLOW_MULTINODE_HOST_ACCESS": "false",
            }
            pod_file = poison_cwd / "pod.json"
            base_status = {
                "containerStatuses": [
                    {
                        "name": "worker",
                        "imageID": f"docker-pullable://registry.internal/team/sglang@{digest}",
                    }
                ]
            }
            pod_file.write_text(
                json.dumps(
                    {
                        "spec": {
                            "hostNetwork": False,
                            "containers": [
                                {
                                    "name": "worker",
                                    "securityContext": {"privileged": False},
                                }
                            ],
                        },
                        "status": base_status,
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                recipe_image.verify_pod_status(pod_file, env), f"@{digest}"
            )
            for bad_spec in (
                {"hostNetwork": True},
                {"hostIPC": True},
                {
                    "containers": [
                        {"name": "worker", "securityContext": {"privileged": True}}
                    ]
                },
            ):
                pod_file.write_text(
                    json.dumps({"spec": bad_spec, "status": base_status}),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, "ordinary worker pod"):
                    recipe_image.verify_pod_status(pod_file, env)
                self.assertEqual(
                    recipe_image.verify_pod_status(
                        pod_file,
                        {**env, "PPU_CI_ALLOW_MULTINODE_HOST_ACCESS": "true"},
                    ),
                    f"@{digest}",
                )

    @unittest.skipUnless(shutil.which("jq"), "jq is required")
    def test_plan_incremental_multinode_label_dedupes_only_when_single_node_scheduled(
        self,
    ):
        import yaml

        wf = yaml.load(
            (REPO_ROOT / ".github/workflows/nightly-test-ppu.yml").read_text(
                encoding="utf-8"
            ),
            Loader=yaml.BaseLoader,
        )
        plan_step = next(
            s["run"] for s in wf["jobs"]["plan"]["steps"] if s.get("id") == "plan"
        )
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()

        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            runs_json = root / "runs.json"
            jobs_json = root / "jobs.json"
            # Mock gh CLI enforcing real gh 2.92.0 positional-arg check (`--jq --arg` fails).
            (fake_bin / "gh").write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                'if [[ "${1:-}" != "api" ]]; then exit 2; fi\n'
                "shift\n"
                "endpoint=''\n"
                "slurp=0\n"
                "while (( $# > 0 )); do\n"
                '  case "$1" in\n'
                "    --paginate) shift ;;\n"
                "    --slurp) slurp=1; shift ;;\n"
                "    --jq) shift 2 ;;\n"
                "    --*) echo 'unknown or unsupported flag' >&2; exit 1 ;;\n"
                '    *) if [[ -n "$endpoint" ]]; then echo "accepts 1 arg(s), received extra" >&2; exit 1; fi; endpoint="$1"; shift ;;\n'
                "  esac\n"
                "done\n"
                'if [[ "$endpoint" == *"/workflows/nightly-test-ppu.yml/runs"* ]]; then cat "'
                + str(runs_json)
                + '"; exit 0; fi\n'
                'if [[ "$endpoint" == *"/actions/runs/100/jobs"* ]]; then\n'
                '  if (( slurp == 1 )); then printf "[%s]\\n" "$(cat "'
                + str(jobs_json)
                + '")"; else cat "'
                + str(jobs_json)
                + '"; fi\n'
                "  exit 0\n"
                "fi\n"
                "echo '{}' \n",
                encoding="utf-8",
            )
            (fake_bin / "gh").chmod(0o755)

            def run_plan(label: str, multinode_approved: bool) -> list[int]:
                out = root / f"out-{label}-{multinode_approved}"
                out.write_text("", encoding="utf-8")
                env = {
                    **os.environ,
                    "PATH": f"{fake_bin}:{os.pathsep.join((str(Path(sys.executable).parent), os.environ['PATH']))}",
                    "TIER": "model_sanity",
                    "INPUT_REF": "",
                    "INPUT_TARGET_REF": "",
                    "MODEL_FILTER": "",
                    "IMAGE_TAG": "",
                    "IMAGE_DIGEST": "",
                    "APPROVED_SHA": sha,
                    "PR_HEAD": sha,
                    "EVENT_NAME": "pull_request_target",
                    "EVENT_ACTION": "labeled",
                    "TRIGGER_LABEL": label,
                    "MULTINODE_APPROVED": str(multinode_approved).lower(),
                    "MULTINODE_APPROVED_SHA": sha if multinode_approved else "",
                    "SKIP_BUILD": "true",
                    "WHEEL_RUN": "",
                    "WHEEL_ID": "111",
                    "GITHUB_RUN_ID": "123",
                    "GITHUB_SHA": sha,
                    "GITHUB_REPOSITORY": "t-head/sglang-for-sail",
                    "GITHUB_OUTPUT": str(out),
                }
                proc = subprocess.run(
                    ["bash", "-c", plan_step],
                    cwd=REPO_ROOT,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
                parsed = dict(
                    line.split("=", 1) for line in out.read_text().splitlines()
                )
                return [
                    len(json.loads(parsed[k])["include"])
                    for k in ("models", "multinode_models")
                ]

            # 1) Nightly label schedules 11 single-node + 0 multi-node.
            runs_json.write_text('{"workflow_runs":[]}\n', encoding="utf-8")
            jobs_json.write_text('{"jobs":[]}\n', encoding="utf-8")
            self.assertEqual(run_plan("run-ci-ppu-nightly", False), [11, 0])

            # 2) Prior non-cancelled run whose single-node jobs were skipped must NOT suppress single-node.
            runs_json.write_text(
                json.dumps({"workflow_runs": [{"id": 100, "conclusion": "success"}]}),
                encoding="utf-8",
            )
            jobs_json.write_text(
                json.dumps(
                    {
                        "jobs": [
                            {"name": "Model Sanity (glm-5.3)", "conclusion": "skipped"}
                        ]
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(run_plan("run-ci-ppu-multinode", True), [11, 1])

            # 3) Prior run with active/completed single-node jobs suppresses duplicate 11 single-node jobs.
            jobs_json.write_text(
                json.dumps(
                    {
                        "jobs": [
                            {"name": "Model Sanity (glm-5.3)", "conclusion": None},
                            {
                                "name": "Model Sanity (qwen3.8-2.4t-mxfp4, 2x8)",
                                "conclusion": "skipped",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(run_plan("run-ci-ppu-multinode", True), [0, 1])

    def test_recent_failures_ignores_passing_image_resolution_archive_and_catches_real_failures(
        self,
    ):
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            store = root / "store"
            store.mkdir(mode=0o700)
            model = "glm-5.3"
            for run_id, worker, overall, phase, msg, has_prov in (
                (100, "fail", "failure", "cluster_worker", "CUDA out of memory", True),
                (101, "pass", "success", "cluster_worker", "normal", True),
                (102, "pass", "success", "image_resolution", "normal", True),
                (103, "pass", "failure", "cleanup", "Permission denied", True),
            ):
                src = root / f"src-{run_id}"
                (src / "results").mkdir(parents=True)
                (src / "ranks").mkdir(parents=True)
                (src / "launcher.log").write_text(msg + "\n", encoding="utf-8")
                (src / "recipe-phase.txt").write_text(phase + "\n", encoding="utf-8")
                (src / "results/summary.json").write_text(
                    json.dumps({"outcome": worker}), encoding="utf-8"
                )
                (src / "ranks/rank-1.status").write_text("0\n", encoding="utf-8")
                if has_prov:
                    (src / "provenance.json").write_text(
                        json.dumps(
                            {
                                "runtime_image": fast_result()["runtime_image"],
                                "recipe_tree_sha256": "a" * 64,
                            }
                        ),
                        encoding="utf-8",
                    )
                pub = root / f"pub-{run_id}.json"
                env = {
                    "PPU_CI_SOURCE_SHA": "a" * 40,
                    "TIER": "model_sanity",
                    "MODEL_KEY": model,
                    "GITHUB_RUN_ID": str(run_id),
                    "GITHUB_RUN_ATTEMPT": "1",
                    "PPU_CI_PRIVATE_RESULTS_ROOT": str(store),
                    "PPU_CI_ARTIFACT_DIR": str(src),
                    "RUNNER_TEMP": str(root / "tmp"),
                    "GITHUB_WORKSPACE": str(root / "ws"),
                    "RECIPE_OUTCOME": overall,
                }
                with (
                    patch.dict(os.environ, env),
                    patch.object(sys, "argv", ["recipe_report", str(pub)]),
                ):
                    report_recipe()
                records = recent_failures(store, run_id + 1, model_filter=model)
                if run_id in (100, 101, 102):
                    # Passing run 102 at phase=image_resolution must NOT shadow run 100's OOM.
                    self.assertEqual(len(records), 1)
                    self.assertEqual(records[0]["run_id"], 100)
                    self.assertIn("cuda_oom", records[0]["codes"])
                else:
                    self.assertEqual(len(records), 1)
                    self.assertEqual(records[0]["run_id"], 103)
                    self.assertIn("permission_denied", records[0]["codes"])

    def test_failure_reason_preserves_rank0_initiating_failure_over_propagated_peer_exit(
        self,
    ):
        import recipe_diagnostics
        import recipe_runner

        pod_text = (REPO_ROOT / "scripts/ci/ppu/run_recipe_pod.sh").read_text(
            encoding="utf-8"
        )
        fallback = pod_text[
            pod_text.index("write_barrier_fallback()") : pod_text.index(
                "# The runtime image"
            )
        ]

        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            barrier = root / "ranks/barrier"
            rank1 = root / "ranks/rank-1"
            barrier.mkdir(parents=True)
            (rank1 / "results").mkdir(parents=True)
            (root / "recipe-phase.txt").write_text("cluster_worker\n", encoding="utf-8")
            # Rank 0 fails first at eval_environment and its pod trap records rank0.done=error.
            (root / "worker-phase.txt").write_text(
                "eval_environment\n", encoding="utf-8"
            )
            r0 = subprocess.run(
                ["bash", "-c", fallback + '\nwrite_barrier_fallback "1"'],
                env={
                    **os.environ,
                    "node_rank": "0",
                    "barrier_dir": str(barrier),
                    "ranks_dir": str(barrier.parent),
                },
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(r0.returncode, 0, r0.stderr)

            # Rank 1 waiting for snapshot at model_cache_check observes rank 0's error and exits 1.
            with patch.object(recipe_runner, "RESULTS", rank1 / "results"):
                recipe_runner.phase("model_cache_check")
                with self.assertRaisesRegex(
                    RuntimeError, "rank 0 failed before publishing model snapshot"
                ):
                    recipe_runner._resolve_gang_snapshot(
                        "fixture/model",
                        None,
                        nnodes=2,
                        node_rank=1,
                        barrier_dir=barrier,
                        timeout_s=1,
                    )
            r1 = subprocess.run(
                ["bash", "-c", fallback + '\nwrite_barrier_fallback "1"'],
                env={
                    **os.environ,
                    "node_rank": "1",
                    "barrier_dir": str(barrier),
                    "ranks_dir": str(barrier.parent),
                },
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(r1.returncode, 0, r1.stderr)
            self.assertFalse((barrier / "fail-rank-1.json").exists())
            self.assertEqual(
                recipe_diagnostics.failure_reason(root), "eval_environment"
            )

    @unittest.skipUnless(
        all(shutil.which(tool) for tool in ("jq", "sha256sum")),
        "runner tools are required",
    )
    def test_cleanup_state_persisted_before_stage_dir_and_retained_on_marker_write_failure(
        self,
    ):
        import signal
        import time

        cluster = (REPO_ROOT / "scripts/ci/ppu/run_recipe_cluster.sh").read_text(
            encoding="utf-8"
        )
        failsafe = REPO_ROOT / "scripts/ci/ppu/cleanup_recipe_cluster.sh"
        # State rename MUST precede staging directory allocation.
        self.assertLess(
            cluster.index('mv -- "${cleanup_state}.tmp" "$cleanup_state"'),
            cluster.index('mkdir -m 0700 -- "$stage_dir"'),
        )

        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            stages = root / "stage"
            runtemp = root / "runner"
            bins = root / "bin"
            infra = root / ".ppu-infra/cluster-run/scripts"
            for folder in (stages, runtemp, bins, infra):
                folder.mkdir(parents=True)
            (infra / "cleanup.sh").write_text("#!/usr/bin/env bash\nexit 0\n")
            (infra / "cleanup.sh").chmod(0o700)
            (bins / "kubectl").write_text("#!/usr/bin/env bash\nexit 0\n")
            (bins / "kubectl").chmod(0o700)

            state = runtemp / "ppu-cleanup-state.json"
            marker = runtemp / "ppu-cleanup-verified.json"
            probe = root / "entered"
            env = {
                **os.environ,
                "PATH": f"{bins}:{os.environ['PATH']}",
                "raw": str(root / "scratch"),
                "stage_root": str(stages),
                "cleanup_state": str(state),
                "cleanup_verified_marker": str(marker),
                "infra": str(infra),
                "INPUT_NAMESPACE": "ppu-sched",
                "INPUT_NNODES": "2",
                "RUNNER_TEMP": str(runtemp),
                "GITHUB_WORKSPACE": str(root),
                "PROBE": str(probe),
            }

            # 1) Hard kill while jq is building state -> zero staging directories allocated.
            stage_snippet = cluster[
                cluster.index("job_hash=$(printf") : cluster.index("tar -czf")
            ]
            (bins / "jq").write_text(
                '#!/usr/bin/env bash\ntouch "$PROBE"\nexec sleep 12\n'
            )
            (bins / "jq").chmod(0o700)
            proc = subprocess.Popen(
                ["bash", "-c", "set -euo pipefail\n" + stage_snippet],
                env=env,
                start_new_session=True,
            )
            try:
                for _ in range(200):
                    if probe.exists():
                        break
                    time.sleep(0.01)
                self.assertTrue(probe.exists())
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            self.assertEqual(list(stages.iterdir()), [])
            (bins / "jq").unlink()
            Path(str(state) + ".tmp").unlink(missing_ok=True)

            # 2) Run full pre-tar allocation, then interrupt before tar -> failsafe cleans empty stage_dir.
            alloc = subprocess.run(
                ["bash", "-c", "set -euo pipefail\n" + stage_snippet],
                env=env,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(alloc.returncode, 0, alloc.stderr)
            self.assertEqual(len(list(stages.iterdir())), 1)
            self.assertTrue(state.is_file())
            clean1 = subprocess.run(
                ["bash", str(failsafe)],
                env=env,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(clean1.returncode, 0, clean1.stdout + clean1.stderr)
            self.assertTrue(json.loads(clean1.stdout)["cleanup_verified"])
            self.assertEqual(list(stages.iterdir()), [])
            marker.unlink()

            # 3) Simulated marker rename failure in finish() retains cleanup_state and sets rc=1.
            state.write_text(
                json.dumps(
                    {
                        "job_name": "sglang-recipe-deadbeef01234567",
                        "namespace": "ppu-sched",
                        "nnodes": 2,
                        "source_stage_dir": str(
                            stages
                            / "sglang-recipe-deadbeef01234567.01234567/source.tar.gz"
                        ),
                    }
                ),
                encoding="utf-8",
            )
            verify_fn = cluster[
                cluster.index("write_cleanup_verified_marker()") : cluster.index(
                    "cleanup_staging_fallback()"
                )
            ]
            success_part = cluster[
                cluster.index("        if (( cleanup_ok == 0 ))") : cluster.index(
                    "    else\n        if ! write_cleanup_verified_marker no_cluster_resources"
                )
            ]
            (bins / "mv").write_text("#!/usr/bin/env bash\nexit 1\n")
            (bins / "mv").chmod(0o700)
            trap_res = subprocess.run(
                [
                    "bash",
                    "-c",
                    "set +e\n"
                    + verify_fn
                    + "\nverify_cluster_cleanup() { return 0; }\nphase() { :; }\ncleanup_ok=0\nrc=0\n"
                    + success_part
                    + '\nprintf "%s" "$rc"',
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=10,
            )
            (bins / "mv").unlink()
            self.assertEqual(trap_res.returncode, 0, trap_res.stderr)
            self.assertEqual(trap_res.stdout, "1")
            self.assertTrue(state.is_file())
            self.assertFalse(marker.exists())

            # Subsequent failsafe recovers cleanly from the retained state.
            clean2 = subprocess.run(
                ["bash", str(failsafe)],
                env=env,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(clean2.returncode, 0, clean2.stdout + clean2.stderr)
            self.assertTrue(json.loads(clean2.stdout)["cleanup_verified"])
            self.assertFalse(state.exists())
            self.assertTrue(marker.is_file())


if __name__ == "__main__":
    unittest.main()
