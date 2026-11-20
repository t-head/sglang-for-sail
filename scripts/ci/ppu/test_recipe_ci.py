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
)
from recipe_bundle import sha256, wheel_inventory
from recipe_cluster_config import load_settings
from recipe_cluster_diagnostics import diagnostic_codes
from recipe_publish import publish, validate
from recipe_report import legacy_result
from recipe_runner import (
    cached_model_path,
    metrics_zero,
    probes,
    read_eval_reviews,
    run,
)
from recipe_select import select


def fast_result():
    return {
        "schema": "ppu-sanity/v1",
        "model": "glm-5.3",
        "tier": "fast_sanity",
        "source_sha": "a" * 40,
        "recipe_tree_sha256": "0" * 64,
        "config_digest": "c" * 64,
        "outcome": "pass",
        "runtime_image": "download.t-head.cn/docker_test/sglang@sha256:" + "d" * 64,
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
        settings = {"prompt": "Compute 6 * 7.", "max_tokens": 16}
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
                with patch("recipe_runner.request", side_effect=(response, stream)):
                    probes(settings, fast=True)
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
                    {"model": "deepseek-v4-flash"},
                    {"model": "deepseek-v4.1-flash"},
                    {"model": "glm-5.3"},
                    {"model": "minimax-m3"},
                ]
            },
        )

    def test_nightly_exact_filter(self):
        self.assertEqual(
            select("model_sanity", "glm-5.3"), {"include": [{"model": "glm-5.3"}]}
        )
        with self.assertRaises(ValueError):
            select("model_sanity", "glm")
        with self.assertRaises(ValueError):
            select("model_sanity", "glm-5.3,glm-5.3")

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

    def test_minimax_nightly_uses_published_quantized_reference(self):
        self.assertEqual(
            select("model_sanity", "minimax-m3"),
            {"include": [{"model": "minimax-m3"}]},
        )
        policy_path = DEFAULT_POLICY_ROOT / "models/minimax-m3.yaml"
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
        self.assertEqual(reference["score_num"], 1284)
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

        for model, ready_timeout in (("glm-5.3", 1350), ("deepseek-v4-flash", 1000)):
            with (
                self.subTest(model=model),
                tempfile.TemporaryDirectory() as temp,
                patch.dict(os.environ, {"PPU_RECIPE_CONTAINER_MODEL": temp}),
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
                self.assertIn("dummy", config["flags"])
                self.assertNotIn("--json-model-override-args", config["flags"])
                probes.assert_called_once()
                self.assertIsNone(result["layer_override"])
                self.assertNotIn("fallback_reason", result)

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
                self.assertEqual(flags[flags.index("--attn-dp-size") + 1], "8")
                self.assertEqual(flags[flags.index("--moe-a2a-backend") + 1], "deepep")
                page_size = int(flags[flags.index("--page-size") + 1])
                chunk_size = int(flags[flags.index("--chunked-prefill-size") + 1])
                self.assertEqual(chunk_size % (8 * page_size), 0)

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
        for model, tampered in (
            ("glm-5.3", False),
            ("deepseek-v4.1-flash", False),
            ("glm-5.3", True),
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
                                    "imageID": "docker-pullable://"
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
                    'test "$INPUT_NNODES/$INPUT_NPROC_PER_NODE/$INPUT_NODE_SELECTOR" = "0/8/board-type=ZW-M890P" || exit 1\n'
                    'test -f "$SOURCE_STAGE_DIR" || exit 1\n'
                    'jq -e --arg cache "$PPU_CI_HF_CACHE_ROOT" --arg model "$MODEL_KEY" --arg revision "$PPU_CI_GLM53_MODEL_REVISION" \'.PPU_CI_HF_CACHE_ROOT == $cache and .MODEL_KEY == $model and (if $model == "glm-5.3" then .PPU_CI_HF_MODEL_REVISION == $revision else (has("PPU_CI_HF_MODEL_REVISION") | not) end)\' <<< "$USER_ENV_JSON" >/dev/null || exit 1\n'
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
                    "PPU_CI_RECIPE_RUNTIME_IMAGE": result["runtime_image"],
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
                (original_log,) = retained.glob("cluster.*/launcher.log")
                self.assertIn(injected, original_log.read_text())


if __name__ == "__main__":
    unittest.main()
