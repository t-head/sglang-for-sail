# SPDX-License-Identifier: Apache-2.0
"""Run Fast Sanity or Model Sanity inside an isolated candidate container."""

import argparse
import contextlib
import json
import math
import os
import re
import signal
import subprocess
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from ppu_recipe import REPO_ROOT, _load_yaml, resolve_recipe

HOST = "http://127.0.0.1:9985"
MODEL = "recipe-model"
RESULTS = Path(os.environ.get("PPU_CI_ARTIFACT_DIR", "/artifacts")) / "results"


def phase(name):
    (RESULTS.parent / "recipe-phase.txt").write_text(name + "\n")


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
                    raise StartupError("server_exit")
                try:
                    with request("/health", timeout=3) as health:
                        if health.status != 200:
                            continue
                    with request("/health_generate", timeout=10) as generated:
                        if generated.status == 200:
                            break
                except (URLError, TimeoutError):
                    time.sleep(1)
            else:
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
    with request(
        "/v1/chat/completions", json.dumps(payload).encode(), timeout=90
    ) as response:
        answer = json.load(response)["choices"][0]
    content = answer["message"].get("content")
    if (
        not isinstance(content, str)
        or not content.strip()
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
        or golden["tolerance"] != {"mode": "band", "abs": 0.02}
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
        "accuracy_within_band": abs(score - golden["score"]) <= 0.02,
        "baseline_established": True,
    }


def bench(stage, config, concurrency, requests, label):
    phase("micro_e2e")
    output = RESULTS / f"bench-{label}-{concurrency}.jsonl"
    spec = stage["input"]
    args = [
        "python3",
        "-I",
        "-B",
        "-m",
        "sglang.bench_serving",
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


def run(policy_path, tier, queue_s):
    policy = _load_yaml(policy_path)
    recipe = REPO_ROOT / policy["recipe"]
    assets = RESULTS.parent / "assets.json"
    model_asset = _load_yaml(recipe)["ppu"]["assets"]
    assets.write_text(
        json.dumps(
            {
                name: os.environ["PPU_RECIPE_CONTAINER_MODEL"]
                for name in model_asset.values()
            }
        )
    )

    resolve = dict(
        policy_path=policy_path,
        tier=tier,
        assets_map_path=assets,
        served_model_name=MODEL,
    )
    config = resolve_recipe(recipe, **resolve)
    if config["ppu_count"] != 8 or policy["model_id"] != config["model_id"]:
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
        try:
            with server(config, "fast_2l", result, 420):
                probes(sanity["probe"], fast=True)
        except StartupError as error:
            if (
                not sanity.get("fallback_full_layers")
                or config["layer_override"] is None
            ):
                raise
            cold_total = time.monotonic() - start
            result["fallback_reason"] = str(error)
            config = resolve_recipe(recipe, **resolve, layer_override=None)
            with server(config, "fast_full", result, 1000):
                probes(sanity["probe"], fast=True)
        result.update(
            config_digest=config["config_digest"],
            layer_override=config["layer_override"],
            cold_start_s=round(cold_total + result["cold_start_s"], 3),
            sse_verified=True,
        )
    else:
        stages = {stage["kind"]: stage for stage in policy["tiers"][tier]["stages"]}
        with server(config, "model_sanity", result, 2400):
            probes(stages["sanity"]["probe"], fast=False)
            result["sse_verified"] = True
            start = time.monotonic()
            result.update(evaluate(stages["evalscope"], config))
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
    parser = argparse.ArgumentParser()
    parser.add_argument("policy", type=Path)
    parser.add_argument("tier", choices=("fast_sanity", "model_sanity"))
    parser.add_argument("--queue-s", type=float, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    RESULTS.mkdir(parents=True, exist_ok=True)
    result = run(args.policy, args.tier, args.queue_s)
    (RESULTS / "summary.json").write_text(json.dumps(result, sort_keys=True) + "\n")
    return int(result["outcome"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
