# SPDX-License-Identifier: Apache-2.0
"""Run recipe sanity checks or the light performance pilot."""

import argparse
import contextlib
import http.client
import importlib.util
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from ppu_recipe import (
    GSM8K_GOLDEN_BAND_ABS,
    REPO_ROOT,
    _load_yaml,
    resolve_recipe,
    validate_policy_structure,
)

HOST = "http://127.0.0.1:9985"
MODEL = "recipe-model"
RESULTS = Path(os.environ.get("PPU_CI_ARTIFACT_DIR", "/artifacts")) / "results"
MODEL_SANITY_READY_TIMEOUTS = {
    # GLM's 141-shard cold load took 63 minutes on TP4 before CUDA graph capture and DeepGEMM warmup.
    "zai-org/GLM-5.3": 6600,
    # GLM Flash loaded 27 of 62 shards in 39 minutes.
    "zai-org/GLM-5.3-Flash": 6600,
    # MiniMax MXFP4/FP8 loaded 26 of 59 shards in 39 minutes.
    "MiniMaxAI/MiniMax-M3": 6600,
    # Both real-weight TP8 jobs were still loading MoE weights at 2400 seconds.
    "deepseek-ai/DeepSeek-V4-Pro": 6000,
    "moonshotai/Kimi-K2.6": 6000,
    # The first Qwen TP4 run loaded 93 of 131 shards in 39 minutes.
    "Qwen/Qwen3.8-Flash-Next": 5100,
}


def phase(name):
    (RESULTS.parent / "worker-phase.txt").write_text(name + "\n")


def request(path, data=None, timeout=15):
    headers = {"Content-Type": "application/json"} if data is not None else {}
    return urlopen(Request(HOST + path, data=data, headers=headers), timeout=timeout)


class StartupError(RuntimeError):
    pass


@contextlib.contextmanager
def server(config, label, timings, ready_timeout):
    with (RESULTS / f"sglang_serve_{label}.log").open("wb") as log:
        phase("server_startup")
        start = time.monotonic()
        process = subprocess.Popen(
            config["argv"],
            env={**os.environ, **config["server_env"]},
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            while time.monotonic() - start < ready_timeout:
                if process.poll() is not None:
                    phase("server_exit")
                    raise StartupError("server_exit")
                try:
                    with request("/health", timeout=3) as health:
                        if health.status != 200:
                            continue
                    with request("/health_generate", timeout=10) as generated:
                        if generated.status == 200:
                            break
                except (URLError, TimeoutError, http.client.HTTPException, OSError):
                    time.sleep(1)
            else:
                phase("readiness_timeout")
                raise StartupError("readiness_timeout")
            timings["cold_start_s"] = round(time.monotonic() - start, 3)
            yield
        finally:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)


def probes(cfg, *, fast):
    phase("chat_probe")
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": cfg["prompt"]}],
        "temperature": 0.0,
        "seed": cfg.get("seed", 42),
        "max_tokens": cfg["max_tokens"],
    }
    if fast:
        payload.update(
            (key, cfg[key]) for key in ("temperature", "top_p", "top_k") if key in cfg
        )
    else:
        payload["thinking"] = {"type": "disabled"}
    with request(
        "/v1/chat/completions", json.dumps(payload).encode(), timeout=90
    ) as response:
        answer = json.load(response)["choices"][0]
    (RESULTS / "chat-probe-response.json").write_text(json.dumps(answer) + "\n")
    message = answer["message"]
    content = message.get("content")
    text_fields = (content, message.get("reasoning_content")) if fast else (content,)
    if (
        not any(isinstance(field, str) and field.strip() for field in text_fields)
        or not answer["finish_reason"]
    ):
        raise ValueError("incomplete non-streaming response")
    if not fast and not re.search(
        r"(?<!\d)" + re.escape(cfg["expected_substring"]) + r"(?!\d)", content
    ):
        raise ValueError("deterministic sanity answer did not match")

    payload["stream"] = True
    token_seen = False
    finish_seen = False
    done_seen = False
    with request(
        "/v1/chat/completions", json.dumps(payload).encode(), timeout=90
    ) as response:
        if response.headers.get_content_type() != "text/event-stream":
            raise ValueError("expected SSE")
        for raw_line in response:
            line = raw_line.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                done_seen = True
                break
            choice = json.loads(data)["choices"][0]
            token_seen |= any(
                bool(choice["delta"].get(field))
                for field in ("content", "reasoning_content")
            )
            finish_seen |= bool(choice.get("finish_reason"))
    if not (token_seen and finish_seen and done_seen):
        raise ValueError("incomplete SSE response")


def metrics_zero(text, names):
    values = {name: [] for name in names}
    for line in text.splitlines():
        match = re.fullmatch(r"([\w:]+)(?:\{.*\})?\s+(\S+)(?:\s+\d+)?", line)
        if match and match[1] in values:
            values[match[1]].append(float(match[2]))
    return all(
        samples and all(value == 0 for value in samples) for samples in values.values()
    )


def wait_zero(*names):
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        with request("/metrics") as response:
            if metrics_zero(response.read().decode(), names):
                return
        time.sleep(1)
    raise TimeoutError("required metrics did not reach zero")


def flush():
    phase("cache_flush")
    wait_zero("sglang:num_running_reqs", "sglang:num_queue_reqs")
    with request("/flush_cache?timeout=10", b"") as response:
        if response.status != 200:
            raise ValueError("cache flush rejected")
    wait_zero("sglang:num_used_tokens")


def read_eval_reviews(directory, expected, score, count):
    # EvalScope's reviewed rows are the gate, together with the aggregate report.
    (source,) = directory.glob("**/reviews/*/gsm8k_main.jsonl")
    rows = {}
    with source.open() as contents:
        for line in contents:
            if not line.strip():
                continue
            row = json.loads(line)
            marking = row["sample_score"]
            question = marking.get("sample_id", row.get("index"))
            if type(question) not in (str, int) or not str(question).strip():
                raise ValueError("unstable question ID")
            question = str(question)
            if question in rows:
                raise ValueError("duplicate question ID")
            result = marking["score"]
            value = result["value"]
            if isinstance(value, dict):
                value = value.get("accuracy", value.get("acc"))
            if (
                result["status"] != "success"
                or result.get("main_score_name") not in (None, "accuracy", "acc")
                or type(value) not in (int, float)
                or value not in (0, 1)
            ):
                raise ValueError("missing binary question score")
            rows[question] = bool(value)
    correct = sum(rows.values())
    if not (
        len(rows) == expected == count
        and math.isfinite(score)
        and 0 <= score <= 1
        and correct == round(score * count)
        and abs(correct / count - score) <= 1 / count
    ):
        raise ValueError("question-level results did not reconcile")
    return rows, correct


def evaluate(stage, config):
    from evalscope import TaskConfig, run_task

    phase("gsm8k_eval")
    dataset = _load_yaml(REPO_ROOT / "test/ppu/policy/datasets/gsm8k.yaml")
    golden_path = (
        REPO_ROOT
        / "test/ppu/golden"
        / f"accuracy_{config['golden_model_key']}_890p.json"
    )
    golden = json.loads(golden_path.read_text())
    expected = dataset["num_expected"]
    dataset_id = f"{dataset['id']}@{dataset['version']}"
    dataset_options = {
        key: dataset[key] for key in ("subset_list", "few_shot_num", "few_shot_random")
    }
    if (
        dataset["scoring"] != "evalscope-default"
        or expected != 1319
        or stage["dataset"] != dataset_id
        or stage["limit"] is not None
        or golden["schema_version"] != 2
        or golden["benchmark"] != "gsm8k"
        or golden["chip"] != "890p"
        or golden["model"] != config["golden_model_key"]
        or golden["checkpoint"] != config["hf_model_id"]
        or golden["serving"]["ppu_count"] != config["ppu_count"]
        or golden["generation"] != stage["generation"]
        or golden["dataset"]
        != {**dataset_options, "num_expected": expected, "dataset_id": dataset_id}
        or golden["tolerance"] != {"mode": "band", "abs": GSM8K_GOLDEN_BAND_ABS}
    ):
        raise ValueError("golden evaluation contract mismatch")
    directory = RESULTS / "evalscope"
    report = run_task(
        task_cfg=TaskConfig(
            model=MODEL,
            api_url=HOST + "/v1",
            api_key="EMPTY",
            eval_type="openai_api",
            datasets=["gsm8k"],
            dataset_args={
                "gsm8k": {
                    **dataset_options,
                    "filters": dataset["filters"],
                    "dataset_id": os.environ.get("PPU_CI_GSM8K_PATH", "/assets/gsm8k"),
                }
            },
            generation_config=stage["generation"],
            eval_batch_size=stage["eval_batch_size"],
            seed=stage["seed"],
            limit=None,
            work_dir=str(directory),
        )
    )["gsm8k"]
    score = float(
        report.score
        if getattr(report, "score", None) is not None
        else report.metrics[0].score
    )
    count = (
        report.num
        if getattr(report, "num", None) is not None
        else report.metrics[0].num
    )
    rows, correct = read_eval_reviews(directory, expected, score, count)
    with (RESULTS / "gsm8k_per_question.jsonl").open("w") as stream:
        for question, answer in rows.items():
            stream.write(
                json.dumps({"question_id": question, "correct": answer}) + "\n"
            )
    return {
        "gsm8k_score": score,
        "gsm8k_correct": correct,
        "gsm8k_count": count,
        "golden_score": golden["score"],
        "accuracy_within_band": abs(score - golden["score"]) <= GSM8K_GOLDEN_BAND_ABS,
        "baseline_established": True,
    }


def bench_module_name():
    if (REPO_ROOT / "python/sglang/benchmark/serving.py").is_file():
        return "sglang.benchmark.serving"
    return (
        "sglang.benchmark.serving"
        if importlib.util.find_spec("sglang") is not None
        and importlib.util.find_spec("sglang.benchmark.serving") is not None
        else "sglang.bench_serving"
    )


def bench(stage, config, concurrency, requests, label):
    phase("micro_e2e")
    output = RESULTS / f"bench-{label}-{concurrency}.jsonl"
    spec = stage["input"]
    args = [
        "python3",
        "-I",
        "-B",
        "-m",
        bench_module_name(),
        "--backend",
        "sglang",
        "--host",
        "127.0.0.1",
        "--port",
        "9985",
        "--model",
        config["resolved_model"],
        "--served-model-name",
        MODEL,
        "--tokenizer",
        config["resolved_tokenizer"],
        "--dataset-name",
        "random-ids",
        "--random-input-len",
        str(spec["input_length"]),
        "--random-output-len",
        str(spec["output_length"]),
        "--random-range-ratio",
        str(spec["range_ratio"]),
        "--num-prompts",
        str(requests),
        "--max-concurrency",
        str(concurrency),
        "--warmup-requests",
        "0",
        "--seed",
        "42",
        "--ready-check-timeout-sec",
        "0",
        "--output-file",
        str(output),
    ]
    with (RESULTS / f"bench-{label}-{concurrency}.log").open("wb") as log:
        subprocess.run(
            args, stdout=log, stderr=subprocess.STDOUT, timeout=900, check=True
        )
    (row,) = output.read_text().splitlines()
    parsed = json.loads(row)
    if parsed["completed"] != requests or parsed["max_concurrency"] != concurrency:
        raise ValueError("native benchmark did not complete every request")


def run_evaluation(stage, config):
    eval_site = os.environ.get("PPU_CI_EVAL_SITE")
    if not eval_site:
        return evaluate(stage, config)
    args = [
        stage,
        {key: config[key] for key in ("golden_model_key", "hf_model_id", "ppu_count")},
    ]
    launcher = (
        "import runpy, sys; sys.path[:0] = sys.argv[1:3]; del sys.argv[1:3]; "
        "runpy.run_module('recipe_runner', run_name='__main__')"
    )
    phase("gsm8k_eval")
    with (RESULTS / "evalscope.log").open("wb") as log:
        subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                "-B",
                "-c",
                launcher,
                eval_site,
                str(Path(__file__).resolve().parent),
                "--eval-worker",
            ],
            input=json.dumps(args).encode(),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return json.loads((RESULTS / "gsm8k-result.json").read_text())


def cached_model_path(repo_id, revision, *, require_weights=True):
    from huggingface_hub import snapshot_download

    cache = Path(os.environ["HF_HUB_CACHE"])
    if revision is None:
        from huggingface_hub.file_download import repo_folder_name
        from huggingface_hub.utils import validate_repo_id

        validate_repo_id(repo_id)
        repo_dir = cache / repo_folder_name(repo_id=repo_id, repo_type="model")
        revision = (repo_dir / "refs/main").read_text().strip()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("expected a pinned model snapshot")
    snapshot = Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=cache,
            local_files_only=True,
        )
    )
    if snapshot.name != revision:
        raise ValueError("expected a pinned model snapshot")
    config_path = snapshot / "config.json"
    if not config_path.is_file():
        raise ValueError("offline model snapshot lacks config metadata")
    if not (snapshot / "tokenizer.json").is_file():
        config = json.loads(config_path.read_text())
        kimi_tokenizer_files = (
            "tokenizer_config.json",
            "tiktoken.model",
            "tokenization_kimi.py",
            "tool_declaration_ts.py",
        )
        if config.get("model_type") != "kimi_k25" or any(
            not (snapshot / name).is_file() or (snapshot / name).stat().st_size == 0
            for name in kimi_tokenizer_files
        ):
            raise ValueError("offline model snapshot lacks tokenizer metadata")
    if require_weights:
        index = json.loads((snapshot / "model.safetensors.index.json").read_text())
        shards = set(index["weight_map"].values())
        if not shards or any(
            Path(name).is_absolute()
            or ".." in Path(name).parts
            or not (snapshot / name).is_file()
            or (snapshot / name).stat().st_size == 0
            for name in shards
        ):
            raise ValueError("offline model snapshot has missing shards")
    return str(snapshot.resolve(strict=True)), revision


def check_one_batch_results(output, case, *, engine="one_batch"):
    shape = case["input"]
    expected = {
        (shape["batch_size"], length, shape["output_len"])
        for length in shape["input_len"]
    }
    threshold = case["threshold"]
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    measured = set()
    measurements = []
    passed = True
    for row in rows:
        key = (row["batch_size"], row["input_len"], row["output_len"])
        if row["run_name"] != case["id"] or key not in expected or key in measured:
            raise ValueError(f"{case['id']}: unexpected or duplicate one_batch result")
        if (
            engine == "one_batch_server"
            and row.get("total_output_tokens") != key[0] * key[2]
        ):
            raise ValueError(f"{case['id']}: server did not generate the requested tokens")
        measured.add(key)
        value = row.get(threshold["metric"])
        valid_throughput = type(value) in (int, float) and 0 <= value < math.inf
        if valid_throughput:
            measurements.append(
                {
                    "batch_size": key[0],
                    "input_len": key[1],
                    "output_len": key[2],
                    "tokens_per_s": value,
                }
            )
        passed &= valid_throughput and value >= threshold["min_tokens_per_s"]
    if measured != expected:
        raise ValueError(
            f"{case['id']}: missing one_batch results for {sorted(expected - measured)}"
        )
    return passed, measurements


def light_perf_worker_args(config, environ):
    argv = list(config["argv"])
    nodes = config["nodes"]
    rank = int(environ.get("NODE_RANK", "0"))
    if nodes != int(environ.get("NNODES", "1")) or not 0 <= rank < nodes:
        raise ValueError(f"light_perf requires {nodes} nodes and a valid node rank")
    if nodes > 1:
        argv[argv.index("--node-rank") + 1] = str(rank)
        argv[argv.index("--dist-init-addr") + 1] = (
            f"{environ['MASTER_ADDR']}:{environ['MASTER_PORT']}"
        )
        if rank:
            argv[4] = "sglang.launch_server"
    return argv, rank


def wait_for_light_perf_marker(path, timeout):
    deadline = time.monotonic() + timeout
    while not path.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for light_perf marker {path.name}")
        time.sleep(1)


def run_light_perf_peer(argv, env, log, leader_done, peer_done):
    process = subprocess.Popen(
        argv,
        env={**env, "SGLANG_BLOCK_NONZERO_RANK_CHILDREN": "1"},
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        wait_for_light_perf_marker(leader_done, 3700)
    finally:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
    peer_done.touch()


def run_light_perf(policy_path, assets_map, queue_s):
    policy = _load_yaml(policy_path)
    validate_policy_structure(policy, context=str(policy_path))
    recipe = REPO_ROOT / policy["recipe"]
    settings = policy["tiers"]["light_perf"]
    cases = settings["cases"]
    snapshot_revision = None
    if assets_map is None:
        phase("model_cache_check")
        ppu = _load_yaml(recipe)["ppu"]
        model_path, snapshot_revision = cached_model_path(
            ppu["hf_model_id"],
            os.environ.get("PPU_CI_HF_MODEL_REVISION"),
            require_weights=not settings["dummy_weights"],
        )
        assets_map = RESULTS.parent / "light-perf-assets.json"
        assets_map.write_text(
            json.dumps({name: model_path for name in ppu["assets"].values()})
        )
    plans = [
        resolve_recipe(
            recipe,
            policy_path=policy_path,
            tier="light_perf",
            preset=case["preset"],
            assets_map_path=assets_map,
            cache_revision=snapshot_revision,
        )
        for case in cases
    ]
    allocated_width = os.environ.get("RECIPE_PPU_COUNT")
    expected_width = int(allocated_width) if allocated_width is not None else None
    for config in plans:
        nodes = config["nodes"]
        total = config["ppu_count"]
        if config["model_id"] != policy["model_id"]:
            raise ValueError("light_perf requires the policy model")
        if expected_width is not None and total != expected_width * nodes:
            raise ValueError(
                f"light_perf preset {config['preset']!r} requires "
                f"{total} PPU devices, allocation has {expected_width * nodes}"
            )
        if not all(
            Path(config[key]).is_dir() for key in ("resolved_model", "resolved_tokenizer")
        ):
            raise ValueError("light_perf requires local model and tokenizer directories")

    output_dir = Path(tempfile.mkdtemp(prefix="one-batch-", dir=RESULTS.resolve()))
    results = {}
    for case, config in zip(cases, plans):
        case_id = case["id"]
        worker_args, node_rank = light_perf_worker_args(config, os.environ)
        env = {**os.environ, **config["server_env"]}
        multi_node = config["nodes"] > 1
        if multi_node:
            shared_results = Path(os.environ["PPU_CI_WORKER_ARTIFACT_DIR"])
            leader_done = shared_results / f"{case_id}.leader-done"
            peer_done = shared_results / f"{case_id}.peer-done"
        shape = case["input"]
        output = output_dir / f"{case_id}.jsonl"
        phase(case_id)
        log_path = output.with_suffix(".log")
        if node_rank:
            with log_path.open("wb") as log:
                run_light_perf_peer(worker_args, env, log, leader_done, peer_done)
            continue
        argv = [
            *worker_args,
            "--batch-size",
            str(shape["batch_size"]),
            "--input-len",
            *map(str, shape["input_len"]),
            "--output-len",
            str(shape["output_len"]),
            "--run-name",
            case_id,
            "--result-filename",
            str(output),
        ]
        if config["benchmark_engine"] == "one_batch_server":
            argv.extend(["--dataset-name", "random-ids"])
        try:
            with log_path.open("wb") as log:
                subprocess.run(
                    argv,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=3600,
                    check=True,
                )
        finally:
            if multi_node:
                leader_done.touch()
        if multi_node:
            wait_for_light_perf_marker(peer_done, 45)
        passed, measurements = check_one_batch_results(
            output, case, engine=config["benchmark_engine"]
        )
        results[case_id] = {
            "passed": passed,
            "config_digest": config["config_digest"],
            "results": str(output.relative_to(RESULTS.resolve())),
            "throughput": {
                "metric": case["threshold"]["metric"],
                "rows": measurements,
            },
        }
    return {
        "queue_s": queue_s,
        "cases": results,
        "outcome": "pass" if all(case["passed"] for case in results.values()) else "fail",
    }


def run(policy_path, tier, queue_s):
    policy = _load_yaml(policy_path)
    recipe = REPO_ROOT / policy["recipe"]
    assets = RESULTS.parent / "assets.json"
    recipe_ppu = _load_yaml(recipe)["ppu"]
    revision = os.environ.get("PPU_CI_HF_MODEL_REVISION")
    uses_cache_ref = revision is None and "PPU_CI_HF_CACHE_ROOT" in os.environ
    if revision is None and not uses_cache_ref:
        model_path = os.environ["PPU_RECIPE_CONTAINER_MODEL"]
    else:
        phase("model_cache_check")
        model_path, snapshot_revision = cached_model_path(
            recipe_ppu["hf_model_id"], revision
        )
    assets.write_text(
        json.dumps({name: model_path for name in recipe_ppu["assets"].values()})
    )

    resolve = dict(
        policy_path=policy_path,
        tier=tier,
        assets_map_path=assets,
        served_model_name=MODEL,
    )
    if uses_cache_ref:
        resolve["cache_revision"] = snapshot_revision
    config = resolve_recipe(recipe, **resolve)
    expected_width = int(os.environ.get("RECIPE_PPU_COUNT", "8"))
    if (
        config["ppu_count"] != expected_width
        or policy["model_id"] != config["model_id"]
    ):
        raise ValueError("incorrect recipe identity or device width")
    result = {
        "queue_s": queue_s,
        "config_digest": config["config_digest"],
        "parser_checked": True,
    }
    if tier == "fast_sanity":
        sanity = policy["tiers"][tier]
        cold_total = 0
        start = time.monotonic()
        has_layer_override = config["layer_override"] is not None
        full_ready_timeout = 1000
        if policy["model_id"] in {"zai-org/GLM-5.3", "Qwen/Qwen3.8-Flash-Next"}:
            # Cold graph capture and the first kernel compilation need extra time.
            full_ready_timeout = 1350
        try:
            with server(
                config,
                "fast_2l" if has_layer_override else "fast_full",
                result,
                420 if has_layer_override else full_ready_timeout,
            ):
                probes(sanity["probe"], fast=True)
        except StartupError as error:
            if not sanity.get("fallback_full_layers") or not has_layer_override:
                raise
            cold_total = time.monotonic() - start
            result["fallback_reason"] = str(error)
            config = resolve_recipe(recipe, **resolve, layer_override=None)
            with server(config, "fast_full", result, full_ready_timeout):
                probes(sanity["probe"], fast=True)
        result.update(
            config_digest=config["config_digest"],
            layer_override=config["layer_override"],
            cold_start_s=round(cold_total + result["cold_start_s"], 3),
            sse_verified=True,
        )
    else:
        stages = {stage["kind"]: stage for stage in policy["tiers"][tier]["stages"]}
        ready_timeout = MODEL_SANITY_READY_TIMEOUTS.get(policy["model_id"], 2400)
        with server(config, "model_sanity", result, ready_timeout):
            probes(stages["sanity"]["probe"], fast=False)
            result["sse_verified"] = True
            start = time.monotonic()
            result.update(run_evaluation(stages["evalscope"], config))
            result["eval_s"] = round(time.monotonic() - start, 3)
            stage = stages["bench"]
            if (
                stage["concurrencies"] != [1, 16]
                or stage["request_count_by_concurrency"] != {1: 8, 16: 32}
                or stage["cache_mode"] != "sanity_throughput_guard"
                or stage["engine"] != "sglang_bench"
            ):
                raise ValueError("incorrect micro-performance scope")
            start = time.monotonic()
            for concurrency in (1, 16):
                flush()
                result["kv_gauge_zero_verified"] = True
                bench(
                    stage,
                    config,
                    concurrency,
                    stage["warmup_count_by_concurrency"][concurrency],
                    "warmup",
                )
                flush()
                bench(
                    stage,
                    config,
                    concurrency,
                    stage["request_count_by_concurrency"][concurrency],
                    "measure",
                )
                result[f"c{concurrency}_passed"] = True
            result["bench_s"] = round(time.monotonic() - start, 3)
            result["cache_mode"] = stage["cache_mode"]
    result["outcome"] = "pass" if result.get("accuracy_within_band", True) else "fail"
    return result


def main():
    if sys.argv[1:] == ["--eval-worker"]:
        result = evaluate(*json.load(sys.stdin))
        (RESULTS / "gsm8k-result.json").write_text(json.dumps(result) + "\n")
        return 0
    parser = argparse.ArgumentParser()
    parser.add_argument("policy", type=Path)
    parser.add_argument("tier", choices=("fast_sanity", "model_sanity", "light_perf"))
    parser.add_argument("--queue-s", type=float, required=True)
    parser.add_argument(
        "--assets-map", type=Path, help="Local assets for light_perf (default: offline cache)"
    )
    args = parser.parse_args()
    if args.assets_map is not None and args.tier != "light_perf":
        parser.error("--assets-map is only supported for light_perf")
    if args.tier == "light_perf" and args.assets_map is None and not os.environ.get(
        "HF_HUB_CACHE"
    ):
        parser.error("light_perf requires --assets-map or an offline HF_HUB_CACHE")
    os.umask(0o077)
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.tier == "light_perf":
        result = run_light_perf(args.policy, args.assets_map, args.queue_s)
    else:
        result = run(args.policy, args.tier, args.queue_s)
    (RESULTS / "summary.json").write_text(json.dumps(result, sort_keys=True) + "\n")
    return int(result["outcome"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
