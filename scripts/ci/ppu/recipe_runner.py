# SPDX-License-Identifier: Apache-2.0
"""Run Fast Sanity or Model Sanity inside an isolated candidate container."""

from __future__ import annotations

import argparse
import concurrent.futures
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
import threading
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from ppu_recipe import GSM8K_GOLDEN_BAND_ABS, REPO_ROOT, _load_yaml, resolve_recipe

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
    # 213-shard cold load (1.27 TiB) across 2 nodes plus DeepGEMM/graph warmup.
    "Qwen/Qwen3.8-2.4T-A95B": 10800,
}


def phase(name):
    (RESULTS.parent / "worker-phase.txt").write_text(name + "\n")


def request(path, data=None, timeout=15):
    headers = {"Content-Type": "application/json"} if data is not None else {}
    return urlopen(Request(HOST + path, data=data, headers=headers), timeout=timeout)


class StartupError(RuntimeError):
    pass


def _barrier_write(barrier_dir: Path, filename: str, payload: dict) -> None:
    barrier_dir.mkdir(parents=True, exist_ok=True)
    tmp = barrier_dir / f".{filename}.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, barrier_dir / filename)


def _barrier_names(barrier_dir: Path | None) -> set[str]:
    if barrier_dir is None or not barrier_dir.is_dir():
        return set()
    # os.listdir issues readdir on NFS, avoiding stale negative dentry caches.
    return set(os.listdir(barrier_dir))


def _peer_failures(barrier_dir: Path | None, *, own_rank: int) -> list[str]:
    own_name = f"fail-rank-{own_rank}.json"
    return sorted(
        name
        for name in _barrier_names(barrier_dir)
        if name.startswith("fail-rank-") and name.endswith(".json") and name != own_name
    )


def _rank0_done_status(barrier_dir: Path | None) -> str | None:
    if barrier_dir is None or "rank0.done" not in _barrier_names(barrier_dir):
        return None
    try:
        done_doc = json.loads((barrier_dir / "rank0.done").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "error"
    return "ok" if done_doc.get("status") == "ok" else "error"


def _kill_process_group_members(pgid: int, *, grace_s: float = 1.0) -> None:
    """Terminate any surviving members of a process group after leader exit."""
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        try:
            os.killpg(pgid, 0)
        except (ProcessLookupError, PermissionError):
            return
        time.sleep(0.05)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _stop_process_group(process: subprocess.Popen) -> tuple[bool, int | None]:
    """Terminate process group; return (signal_sent, returncode)."""
    if process.poll() is not None:
        orig_rc = process.returncode
        # Even if the leader already exited, reap any surviving child processes
        # that remain in its session/process group.
        _kill_process_group_members(process.pid)
        return (False, orig_rc)
    signal_sent = True
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        signal_sent = False
    try:
        rc = process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        rc = process.wait(timeout=5)
    _kill_process_group_members(process.pid)
    return (signal_sent, rc)


def _flag_value(flags: list[str], name: str) -> str | None:
    value = None
    prefix = f"{name}="
    for idx, token in enumerate(flags):
        if token == name and idx + 1 < len(flags):
            value = flags[idx + 1]
        elif token.startswith(prefix):
            value = token[len(prefix) :]
    return value


def _pp_layer_indices(
    num_layers: int,
    *,
    pp_size: int,
    pp_rank: int,
    server_env: dict[str, str] | None = None,
) -> tuple[int, int]:
    """Mirror sglang.srt.distributed.utils.get_pp_indices."""
    partition_str = (server_env or {}).get("SGLANG_PP_LAYER_PARTITION")
    if partition_str is None:
        partition_str = os.getenv("SGLANG_PP_LAYER_PARTITION")
    if pp_size == 1:
        partition_str = None
    if partition_str is not None:
        partitions = [int(layer) for layer in partition_str.split(",")]
        if len(partitions) != pp_size or sum(partitions) != num_layers:
            raise ValueError(
                f"invalid SGLANG_PP_LAYER_PARTITION={partition_str!r} for "
                f"pp_size={pp_size}, num_layers={num_layers}"
            )
        start = sum(partitions[:pp_rank])
        return start, start + partitions[pp_rank]
    base_layers = num_layers // pp_size
    remainder = num_layers % pp_size
    # SGLang assigns remainder layers to the last `remainder` stages.
    if pp_rank >= pp_size - remainder:
        partitions_without_extra = pp_size - remainder
        start = pp_rank * (base_layers + 1) - partitions_without_extra
        return start, start + base_layers + 1
    start = pp_rank * base_layers
    return start, start + base_layers


def select_pp_stage_shards(
    snapshot: Path,
    *,
    pp_size: int,
    pp_rank: int,
    server_env: dict[str, str] | None = None,
) -> tuple[list[Path], tuple[int, int], int]:
    """Return the safetensors shards needed by this pipeline-parallel stage."""
    config_path = snapshot / "config.json"
    index_path = snapshot / "model.safetensors.index.json"
    if not config_path.is_file() or not index_path.is_file():
        return [], (0, 0), 0
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    num_layers = cfg.get("num_hidden_layers")
    if num_layers is None and isinstance(cfg.get("text_config"), dict):
        num_layers = cfg["text_config"].get("num_hidden_layers")
    if not isinstance(num_layers, int) or num_layers < pp_size:
        return [], (0, 0), 0
    start, end = _pp_layer_indices(
        num_layers, pp_size=pp_size, pp_rank=pp_rank, server_env=server_env
    )

    weight_map = json.loads(index_path.read_text(encoding="utf-8")).get(
        "weight_map", {}
    )
    all_files = set(weight_map.values())
    layer_re = re.compile(r"(?:^|\.)layers\.(\d+)\.")
    selected_names: set[str] = set()
    for tensor_name, rel_file in weight_map.items():
        if "mtp" in tensor_name:
            continue
        match = layer_re.search(tensor_name)
        if match is not None:
            layer_idx = int(match.group(1))
            if start <= layer_idx < end:
                selected_names.add(rel_file)
            continue
        if "embed_tokens" in tensor_name:
            if pp_rank == 0:
                selected_names.add(rel_file)
        elif "lm_head" in tensor_name or tensor_name.endswith("norm.weight"):
            if pp_rank == pp_size - 1:
                selected_names.add(rel_file)
        else:
            selected_names.add(rel_file)

    paths: list[Path] = []
    for rel_file in sorted(selected_names):
        rel = Path(rel_file)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"invalid shard path in safetensors index: {rel_file}")
        candidate = snapshot / rel
        if candidate.is_file():
            paths.append(candidate)
    return paths, (start, end), len(all_files)


def warm_pp_stage_shards(
    snapshot: Path,
    *,
    flags: list[str],
    nnodes: int,
    node_rank: int,
    barrier_dir: Path | None,
    server_env: dict[str, str] | None = None,
    max_workers: int = 8,
    timeout_s: float = 10800.0,
    poll_s: float = 2.0,
) -> float:
    """Warm only this node's PP-stage shards into the Linux OS page cache.

    Under PP=nnodes, each node only loads 1/nnodes of the layers. Reading only
    those shards sequentially in 16 MiB blocks cuts shared-NFS traffic in half
    compared to full-checkpoint prefetch and avoids per-node load-time skew
    before distributed KV-cache sizing collectives run.
    """
    pp_raw = _flag_value(flags, "--pp-size")
    if nnodes <= 1 or pp_raw is None or not pp_raw.isdigit() or int(pp_raw) != nnodes:
        return 0.0
    shards, (start_layer, end_layer), total_shards = select_pp_stage_shards(
        snapshot, pp_size=nnodes, pp_rank=node_rank, server_env=server_env
    )
    if not shards:
        return 0.0

    phase("page_cache_warm")
    start = time.monotonic()
    cancel = threading.Event()

    def _on_signal(_signum, _frame):
        cancel.set()

    is_main = threading.current_thread() is threading.main_thread()
    prev_term = signal.signal(signal.SIGTERM, _on_signal) if is_main else None
    prev_int = signal.signal(signal.SIGINT, _on_signal) if is_main else None
    try:
        chunk_bytes = 16 * 1024 * 1024

        def _read_shard(path: Path) -> int:
            read_total = 0
            with path.open("rb") as stream:
                while not cancel.is_set():
                    if barrier_dir is not None and (
                        _peer_failures(barrier_dir, own_rank=node_rank)
                        or (
                            node_rank > 0
                            and _rank0_done_status(barrier_dir) is not None
                        )
                    ):
                        cancel.set()
                        break
                    block = stream.read(chunk_bytes)
                    if not block:
                        break
                    read_total += len(block)
            return read_total

        total_bytes = 0
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(max_workers, len(shards))
        ) as executor:
            futures = [executor.submit(_read_shard, path) for path in shards]
            for fut in concurrent.futures.as_completed(futures):
                total_bytes += fut.result()

        if cancel.is_set():
            if barrier_dir is not None and (
                _peer_failures(barrier_dir, own_rank=node_rank)
                or (node_rank > 0 and _rank0_done_status(barrier_dir) is not None)
            ):
                phase("worker_exit")
                raise StartupError("worker_exit")
            raise StartupError("page_cache_warm_cancelled")

        warm_only_s = time.monotonic() - start
        if barrier_dir is not None:
            _barrier_write(
                barrier_dir,
                f"warm-rank-{node_rank}.json",
                {
                    "rank": node_rank,
                    "shards": len(shards),
                    "total_shards": total_shards,
                    "elapsed_s": round(warm_only_s, 2),
                },
            )
            expected = {f"warm-rank-{r}.json" for r in range(nnodes)}
            deadline = start + timeout_s
            while time.monotonic() < deadline:
                if cancel.is_set():
                    raise StartupError("page_cache_warm_cancelled")
                if _peer_failures(barrier_dir, own_rank=node_rank) or (
                    node_rank > 0 and _rank0_done_status(barrier_dir) is not None
                ):
                    phase("worker_exit")
                    raise StartupError("worker_exit")
                if expected.issubset(_barrier_names(barrier_dir)):
                    break
                time.sleep(poll_s)
            else:
                phase("readiness_timeout")
                raise StartupError("page_cache_warm_timeout")

        elapsed_s = time.monotonic() - start
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / "page-cache-warm.log").write_text(
            f"pp_warm_done rank={node_rank}/{nnodes} layers={start_layer}..{end_layer - 1} "
            f"shards={len(shards)}/{total_shards} bytes={total_bytes} "
            f"warm_s={warm_only_s:.2f} total_s={elapsed_s:.2f}\n",
            encoding="utf-8",
        )
        return elapsed_s
    except Exception:
        if barrier_dir is not None:
            _barrier_write(
                barrier_dir,
                f"fail-rank-{node_rank}.json",
                {"rank": node_rank, "exit_code": 1, "reason": "page_cache_warm"},
            )
        raise
    finally:
        if is_main:
            signal.signal(signal.SIGTERM, prev_term)
            signal.signal(signal.SIGINT, prev_int)


def _multinode_argv(
    base_argv: list[str], *, nnodes: int, node_rank: int, dist_init_addr: str
) -> list[str]:
    if nnodes <= 1:
        return list(base_argv)
    if not dist_init_addr or ":" not in dist_init_addr:
        raise ValueError("multi-node execution requires PPU_CI_DIST_INIT_ADDR")
    return [
        *base_argv,
        "--nnodes",
        str(nnodes),
        "--node-rank",
        str(node_rank),
        "--dist-init-addr",
        dist_init_addr,
    ]


def _wait_workers_exit(
    barrier_dir: Path, nnodes: int, *, timeout: float = 60.0, poll_s: float = 1.0
) -> None:
    expected = {f"exit-rank-{rank}.json" for rank in range(1, nnodes)}
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        names = _barrier_names(barrier_dir)
        if _peer_failures(barrier_dir, own_rank=0):
            phase("worker_exit")
            raise RuntimeError("a multi-node worker exited with failure")
        if expected.issubset(names):
            for name in sorted(expected):
                record = json.loads((barrier_dir / name).read_text(encoding="utf-8"))
                if record.get("exit_code") != 0:
                    phase("worker_exit")
                    raise RuntimeError(f"worker {name} exited non-zero")
            return
        time.sleep(poll_s)
    phase("worker_exit")
    raise TimeoutError("timed out waiting for multi-node workers to exit")


def run_worker_node(
    config: dict,
    label: str,
    *,
    nnodes: int,
    node_rank: int,
    dist_init_addr: str,
    barrier_dir: Path,
    poll_s: float = 2.0,
) -> dict:
    argv = _multinode_argv(
        config["argv"],
        nnodes=nnodes,
        node_rank=node_rank,
        dist_init_addr=dist_init_addr,
    )
    terminated_by_signal = {"signum": None}

    def _on_signal(signum, _frame):
        terminated_by_signal["signum"] = signum

    is_main = threading.current_thread() is threading.main_thread()
    prev_term = signal.signal(signal.SIGTERM, _on_signal) if is_main else None
    prev_int = signal.signal(signal.SIGINT, _on_signal) if is_main else None
    try:
        with (RESULTS / f"sglang_serve_{label}.log").open("wb") as log:
            phase("server_startup")
            process = subprocess.Popen(
                argv,
                env={**os.environ, **config["server_env"]},
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                while True:
                    # Check process liveness FIRST so a real non-zero exit that
                    # happened before rank0.done is never rewritten to exit 0.
                    rc = process.poll()
                    if rc is not None:
                        names = _barrier_names(barrier_dir)
                        done_ok = False
                        if rc == 0 and "rank0.done" in names:
                            try:
                                done_doc = json.loads(
                                    (barrier_dir / "rank0.done").read_text(
                                        encoding="utf-8"
                                    )
                                )
                                done_ok = done_doc.get("status") == "ok"
                            except (OSError, json.JSONDecodeError):
                                done_ok = False
                        if done_ok:
                            _barrier_write(
                                barrier_dir,
                                f"exit-rank-{node_rank}.json",
                                {
                                    "rank": node_rank,
                                    "exit_code": 0,
                                    "reason": "rank0_done",
                                },
                            )
                            return {"outcome": "pass", "node_rank": node_rank}
                        phase("server_exit")
                        exit_code = rc if rc != 0 else 1
                        _barrier_write(
                            barrier_dir,
                            f"fail-rank-{node_rank}.json",
                            {
                                "rank": node_rank,
                                "exit_code": exit_code,
                                "reason": "server_exit",
                            },
                        )
                        _barrier_write(
                            barrier_dir,
                            f"exit-rank-{node_rank}.json",
                            {
                                "rank": node_rank,
                                "exit_code": exit_code,
                                "reason": "server_exit",
                            },
                        )
                        raise StartupError("server_exit")
                    names = _barrier_names(barrier_dir)
                    if "rank0.done" in names:
                        done_status = _rank0_done_status(barrier_dir)
                        signal_sent, stop_rc = _stop_process_group(process)
                        if done_status != "ok":
                            phase("worker_exit")
                            _barrier_write(
                                barrier_dir,
                                f"exit-rank-{node_rank}.json",
                                {
                                    "rank": node_rank,
                                    "exit_code": 1,
                                    "reason": "rank0_error",
                                },
                            )
                            return {"outcome": "fail", "node_rank": node_rank}
                        expected_signal_rcs = {
                            0,
                            -signal.SIGTERM,
                            -signal.SIGKILL,
                            128 + signal.SIGTERM,
                            128 + signal.SIGKILL,
                        }
                        if not signal_sent and stop_rc != 0:
                            phase("server_exit")
                            exit_code = stop_rc if stop_rc not in (None, 0) else 1
                            _barrier_write(
                                barrier_dir,
                                f"fail-rank-{node_rank}.json",
                                {
                                    "rank": node_rank,
                                    "exit_code": exit_code,
                                    "reason": "server_exit",
                                },
                            )
                            _barrier_write(
                                barrier_dir,
                                f"exit-rank-{node_rank}.json",
                                {
                                    "rank": node_rank,
                                    "exit_code": exit_code,
                                    "reason": "server_exit",
                                },
                            )
                            raise StartupError("server_exit")
                        if stop_rc not in expected_signal_rcs:
                            phase("server_exit")
                            exit_code = stop_rc if stop_rc not in (None, 0) else 1
                            _barrier_write(
                                barrier_dir,
                                f"fail-rank-{node_rank}.json",
                                {
                                    "rank": node_rank,
                                    "exit_code": exit_code,
                                    "reason": "server_exit",
                                },
                            )
                            _barrier_write(
                                barrier_dir,
                                f"exit-rank-{node_rank}.json",
                                {
                                    "rank": node_rank,
                                    "exit_code": exit_code,
                                    "reason": "server_exit",
                                },
                            )
                            raise StartupError("server_exit")
                        _barrier_write(
                            barrier_dir,
                            f"exit-rank-{node_rank}.json",
                            {
                                "rank": node_rank,
                                "exit_code": 0,
                                "reason": "rank0_done",
                            },
                        )
                        return {"outcome": "pass", "node_rank": node_rank}
                    if terminated_by_signal["signum"] is not None:
                        _stop_process_group(process)
                        code = 128 + terminated_by_signal["signum"]
                        _barrier_write(
                            barrier_dir,
                            f"fail-rank-{node_rank}.json",
                            {
                                "rank": node_rank,
                                "exit_code": code,
                                "reason": "signal",
                            },
                        )
                        _barrier_write(
                            barrier_dir,
                            f"exit-rank-{node_rank}.json",
                            {
                                "rank": node_rank,
                                "exit_code": code,
                                "reason": "signal",
                            },
                        )
                        return {"outcome": "fail", "node_rank": node_rank}
                    if _peer_failures(barrier_dir, own_rank=node_rank):
                        phase("worker_exit")
                        _stop_process_group(process)
                        _barrier_write(
                            barrier_dir,
                            f"exit-rank-{node_rank}.json",
                            {
                                "rank": node_rank,
                                "exit_code": 1,
                                "reason": "peer_failed",
                            },
                        )
                        return {"outcome": "fail", "node_rank": node_rank}
                    time.sleep(poll_s)
            finally:
                # Always reap any surviving members of the launcher's process
                # group even when the group leader already exited early.
                _stop_process_group(process)
    finally:
        if is_main:
            signal.signal(signal.SIGTERM, prev_term)
            signal.signal(signal.SIGINT, prev_int)


@contextlib.contextmanager
def server(
    config,
    label,
    timings,
    ready_timeout,
    *,
    nnodes=1,
    node_rank=0,
    dist_init_addr="",
    barrier_dir=None,
    signal_done_on_exit=False,
):
    argv = _multinode_argv(
        config["argv"],
        nnodes=nnodes,
        node_rank=node_rank,
        dist_init_addr=dist_init_addr,
    )
    status = {"ok": False}
    with (RESULTS / f"sglang_serve_{label}.log").open("wb") as log:
        phase("server_startup")
        start = time.monotonic()
        process = subprocess.Popen(
            argv,
            env={**os.environ, **config["server_env"]},
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            while time.monotonic() - start < ready_timeout:
                if barrier_dir is not None and _peer_failures(
                    barrier_dir, own_rank=node_rank
                ):
                    phase("worker_exit")
                    raise StartupError("worker_exit")
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
            status["ok"] = True
        finally:
            wait_error = None
            try:
                if barrier_dir is not None and signal_done_on_exit:
                    # Publish rank0.done BEFORE stopping rank 0's server, and on the
                    # normal path wait for live workers to stop cleanly while rank 0
                    # is still up so workers never see a premature disconnect.
                    _barrier_write(
                        barrier_dir,
                        "rank0.done",
                        {"status": "ok" if status["ok"] else "error"},
                    )
                    if status["ok"] and nnodes > 1:
                        try:
                            _wait_workers_exit(barrier_dir, nnodes)
                        except Exception as exc:
                            wait_error = exc
            finally:
                _stop_process_group(process)
            if wait_error is not None:
                raise wait_error


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
        if "thinking" in cfg:
            if cfg["thinking"] is not None:
                payload["thinking"] = cfg["thinking"]
        else:
            payload["thinking"] = {"type": "disabled"}
        if "chat_template_kwargs" in cfg:
            payload["chat_template_kwargs"] = cfg["chat_template_kwargs"]
    with request(
        "/v1/chat/completions", json.dumps(payload).encode(), timeout=180
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
        "/v1/chat/completions", json.dumps(payload).encode(), timeout=180
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
    has_golden = golden_path.is_file()
    golden = json.loads(golden_path.read_text()) if has_golden else None
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
    ):
        raise ValueError("golden evaluation contract mismatch")
    if has_golden and (
        golden["schema_version"] != 2
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
    if not has_golden:
        return {
            "gsm8k_score": score,
            "gsm8k_correct": correct,
            "gsm8k_count": count,
            "accuracy_within_band": False,
            "baseline_established": False,
        }
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


def cached_model_path(repo_id, revision):
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


def _resolve_gang_snapshot(
    hf_model_id: str,
    revision: str | None,
    *,
    nnodes: int,
    node_rank: int,
    barrier_dir: Path | None,
    timeout_s: float = 1800.0,
    poll_s: float = 0.5,
) -> tuple[str, str]:
    """Ensure every node in a multi-node gang resolves the exact same snapshot."""
    if nnodes <= 1 or revision is not None or barrier_dir is None:
        return cached_model_path(hf_model_id, revision)
    manifest = "snapshot-revision.json"
    if node_rank == 0:
        try:
            model_path, snapshot_revision = cached_model_path(hf_model_id, None)
            _barrier_write(
                barrier_dir,
                manifest,
                {"hf_model_id": hf_model_id, "revision": snapshot_revision},
            )
            return model_path, snapshot_revision
        except Exception:
            _barrier_write(
                barrier_dir,
                "fail-rank-0.json",
                {"rank": 0, "exit_code": 1, "reason": "model_cache_check"},
            )
            raise
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        names = _barrier_names(barrier_dir)
        if _peer_failures(barrier_dir, own_rank=node_rank) or (
            _rank0_done_status(barrier_dir) is not None
        ):
            raise RuntimeError("rank 0 failed before publishing model snapshot")
        if manifest in names:
            payload = json.loads((barrier_dir / manifest).read_text(encoding="utf-8"))
            locked_revision = payload.get("revision")
            if payload.get("hf_model_id") != hf_model_id or not isinstance(
                locked_revision, str
            ):
                raise ValueError("mismatched gang snapshot metadata")
            return cached_model_path(hf_model_id, locked_revision)
        time.sleep(poll_s)
    raise TimeoutError("timed out waiting for rank 0 snapshot revision")


def run(policy_path, tier, queue_s):
    policy = _load_yaml(policy_path)
    recipe = REPO_ROOT / policy["recipe"]
    assets = RESULTS.parent / "assets.json"
    recipe_ppu = _load_yaml(recipe)["ppu"]
    expected_nnodes = int(
        os.environ.get("PPU_CI_NNODES", os.environ.get("RECIPE_NNODES", "1"))
    )
    expected_width = int(os.environ.get("RECIPE_PPU_COUNT", "8"))
    node_rank = int(
        os.environ.get("PPU_CI_NODE_RANK", os.environ.get("NODE_RANK", "0"))
    )
    if expected_nnodes < 1 or not (0 <= node_rank < expected_nnodes):
        raise ValueError("incorrect recipe identity or device width")

    dist_init_addr = os.environ.get("PPU_CI_DIST_INIT_ADDR", "")
    barrier_dir = None
    if expected_nnodes > 1:
        ranks_dir = os.environ.get("PPU_CI_RANKS_DIR", "")
        if not ranks_dir or not dist_init_addr:
            raise ValueError(
                "multi-node execution requires ranks dir and rendezvous address"
            )
        barrier_dir = Path(ranks_dir) / "barrier"

    revision = os.environ.get("PPU_CI_HF_MODEL_REVISION")
    uses_cache_ref = revision is None and "PPU_CI_HF_CACHE_ROOT" in os.environ
    snapshot_wait_timeout = (
        MODEL_SANITY_READY_TIMEOUTS.get(policy["model_id"], 2400)
        if tier == "model_sanity"
        else 1800
    )
    if revision is None and not uses_cache_ref:
        model_path = os.environ["PPU_RECIPE_CONTAINER_MODEL"]
    else:
        phase("model_cache_check")
        model_path, snapshot_revision = _resolve_gang_snapshot(
            recipe_ppu["hf_model_id"],
            revision,
            nnodes=expected_nnodes,
            node_rank=node_rank,
            barrier_dir=barrier_dir,
            timeout_s=float(snapshot_wait_timeout),
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
    if (
        config["nodes"] != expected_nnodes
        or config["devices_per_node"] != expected_width
        or config["ppu_count"] != expected_nnodes * expected_width
        or policy["model_id"] != config["model_id"]
    ):
        raise ValueError("incorrect recipe identity or device width")

    warm_s = 0.0
    if tier == "model_sanity" and expected_nnodes > 1:
        warm_s = warm_pp_stage_shards(
            Path(model_path),
            flags=config["flags"],
            nnodes=expected_nnodes,
            node_rank=node_rank,
            barrier_dir=barrier_dir,
            server_env=config["server_env"],
        )

    if expected_nnodes > 1 and node_rank > 0:
        label = (
            "fast_2l"
            if tier == "fast_sanity" and config["layer_override"] is not None
            else ("fast_full" if tier == "fast_sanity" else "model_sanity")
        )
        return run_worker_node(
            config,
            label,
            nnodes=expected_nnodes,
            node_rank=node_rank,
            dist_init_addr=dist_init_addr,
            barrier_dir=barrier_dir,
        )

    result = {
        "queue_s": queue_s,
        "config_digest": config["config_digest"],
        "parser_checked": True,
    }
    server_kwargs = {}
    if expected_nnodes > 1:
        server_kwargs = dict(
            nnodes=expected_nnodes,
            node_rank=0,
            dist_init_addr=dist_init_addr,
            barrier_dir=barrier_dir,
            signal_done_on_exit=True,
        )
    if tier == "fast_sanity":
        sanity = policy["tiers"][tier]
        cold_total = 0
        start = time.monotonic()
        has_layer_override = config["layer_override"] is not None
        if (
            expected_nnodes > 1
            and sanity.get("fallback_full_layers")
            and has_layer_override
        ):
            raise ValueError(
                "multi-node fast_sanity does not support fallback_full_layers"
            )
        full_ready_timeout = 1800 if expected_nnodes > 1 else 1000
        if policy["model_id"] in {"zai-org/GLM-5.3", "Qwen/Qwen3.8-Flash-Next"}:
            # Cold graph capture and the first kernel compilation need extra time.
            full_ready_timeout = 1350
        try:
            with server(
                config,
                "fast_2l" if has_layer_override else "fast_full",
                result,
                420 if has_layer_override else full_ready_timeout,
                **server_kwargs,
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
        with server(config, "model_sanity", result, ready_timeout, **server_kwargs):
            if warm_s > 0:
                result["cold_start_s"] = round(result["cold_start_s"] + warm_s, 3)
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
    parser.add_argument("tier", choices=("fast_sanity", "model_sanity"))
    parser.add_argument("--queue-s", type=float, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    RESULTS.mkdir(parents=True, exist_ok=True)
    result = run(args.policy, args.tier, args.queue_s)
    node_rank = int(
        os.environ.get("PPU_CI_NODE_RANK", os.environ.get("NODE_RANK", "0"))
    )
    if node_rank == 0:
        (RESULTS / "summary.json").write_text(json.dumps(result, sort_keys=True) + "\n")
    return int(result["outcome"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
