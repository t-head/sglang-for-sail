# SPDX-License-Identifier: Apache-2.0
"""Report fixed diagnostic codes from the task's retained private evidence."""

import json
import os
import re
import stat

from recipe_publish import MODEL_POLICY_ROOT
from recipe_retention import RETENTION_SUBPATH

RULES = {
    "image_pull": r"ErrImagePull|ImagePullBackOff|failed to pull image|manifest unknown",
    "registry_auth": r"status code: (401|403)|pull access denied|authorization failed|insufficient_scope",
    "checkout_ownership": r"detected dubious ownership",
    "source_archive": r"tar: .*Cannot open|source archive.*(missing|not found)",
    "permission_denied": r"permission denied|operation not permitted",
    "capacity_ppu": r"Insufficient .*ppu",
    "capacity_cpu": r"Insufficient cpu",
    "capacity_memory": r"Insufficient memory",
    "worker_oom": r"OOMKilled|out of memory",
    "cuda_oom": r"CUDA out of memory|OutOfMemoryError|Not enough memory",
    "wheel_install": r"not a supported wheel|No matching distribution|Failed building wheel",
    "python_import": r"ModuleNotFoundError|ImportError|undefined symbol|No module named",
    "missing_triton_kernels": r"No module named ['\"]?triton_kernels",
    "invalid_server_args": r"unrecognized arguments|argument .+: invalid choice",
    "deepep_dispatch_limit": r"num_max_dispatch_tokens_per_rank|SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK",
    "deepep_error": r"deep_ep|DeepEP is not available|DeepEP buffer cannot",
    "watchdog": r"Watchdog (?:caught|timeout|triggered)|triggered watchdog",
    "nccl_error": r"NCCL error|DistBackendError|TCPStore|ProcessGroupNCCL",
    "tcpstore_error": r"TCPStore|Socket Timeout|Connection refused|ECONNREFUSED|Connect timeout",
    "nccl_comm_init": r"ncclCommInitRank|ncclGroupEnd|ncclCommInit",
    "nccl_system_error": r"unhandled system error|ncclSystemError",
    "nccl_invalid_usage": r"invalid usage|ncclInvalidUsage",
    "nccl_cumem": r"cuMem|NCCL_CUMEM|PCCL_CUMEM|VMM",
    "pccl_mnnvl_imex": r"alixpu-caps-imex|ptg-imex-ctl|ncclMnnvlCheck|mnnvl\.cc",
    "pccl_socket_connect": r"Call to connect .* failed|socketStartConnect|socketFinalizeConnect|socketConnect|bootstrap.*failed|bootstrapNetInit",
    "pccl_socket_iface": r"No interface found|Bootstrap : no interface|Unable to find interface",
    "pccl_ib_error": r"ibv_open_device|ibv_create_cq|ibv_create_qp|ibv_reg_mr|ibv_modify_qp|NET/IB.*failed|No IB devices|unable to find.*HCA",
    "ib_open_device": r"ibv_open_device|Unable to open device",
    "ib_create_cq": r"ibv_create_cq",
    "ib_create_qp": r"ibv_create_qp|mlx5dv_create_qp",
    "ib_reg_mr": r"ibv_reg_mr|ibv_reg_mr_iova2",
    "ib_reg_dmabuf": r"ibv_reg_dmabuf_mr|DMA-BUF MR registration",
    "ib_no_device": r"No IB devices|NET/IB : No device found",
    "ib_post_send_recv": r"ibv_post_send|ibv_post_recv",
    "ib_async_event": r"async fatal event",
    "ib_mismatched_if": r"Mismatched user defined interface",
    "net_ib_connect_cc": r"net_ib/connect\.cc",
    "net_ib_reg_cc": r"net_ib/reg\.cc",
    "net_ib_p2p_cc": r"net_ib/p2p\.cc",
    "p2p_transport_cc": r"transport/p2p\.cc",
    "p2p_ipc_socket": r"ipcsocket\.cc|ipcSocket",
    "p2p_cuda_ipc": r"cudaIpc|ncclP2pImportShareableBuffer|p2pMap",
    "proxy_connect_fail": r"Proxy Call to rank .+ failed \(Connect\)",
    "proxy_setup_fail": r"Proxy Call to rank .+ failed \(Setup\)",
    "proxy_progress_fail": r"Proxy Call to rank .+ failed \(Progress\)",
    "proxy_uds_fail": r"pcclProxyCallBlockingUDS|Proxy Service UDS",
    "pynccl_caller": r"pynccl\.py|pynccl_wrapper\.py",
    "torch_pg_caller": r"ProcessGroupNCCL",
    "err_eperm": r"errno=1\b|Operation not permitted",
    "err_enoent": r"errno=2\b|No such file or directory",
    "err_efault": r"errno=14\b|Bad address",
    "err_einval": r"errno=22\b|Invalid argument",
    "err_enotsup": r"errno=95\b|Operation not supported|Not supported",
    "err_etimedout": r"errno=110\b|Connection timed out|ETIMEDOUT",
    "err_ehostunreach": r"errno=113\b|No route to host",
    "pccl_shm_error": r"ncclShmOpen|shm_open|shmutils\.cc|No space left on device|u2mm_shm",
    "pccl_no_transport": r"No transport found",
    "pccl_p2p_error": r"p2p\.cc|P2P.*failed|cudaIpc|ipcSocket|ipcsocket\.cc",
    "pccl_proxy_error": r"proxy\.cc|Proxy.*failed",
    "pccl_oom_memlock": r"RLIMIT_MEMLOCK|Cannot allocate memory|ENOMEM",
    "pccl_gid_invalid": r"sgid_index|query_gid|GID.*invalid",
    "gid_idx_0": r"multinode: .* gid=0\b",
    "gid_idx_1": r"multinode: .* gid=1\b",
    "gid_idx_2": r"multinode: .* gid=2\b",
    "gid_idx_3": r"multinode: .* gid=3\b",
    "gid_idx_other": r"multinode: .* gid=(?:[4-9]|[1-9][0-9]+)\b",
    "pp_error": r"pp_rank|pipeline.?parallel|send_tensor_dict|recv_tensor_dict",
    "roce_gid": r"IBV_WC_RETRY_EXC_ERR|ibv_modify_qp|exposes? no routable (?:RoCE v2 )?GID|resolved (?:index .+ but .+ resolved|no common RoCE v2 GID)",
    "gid_no_active": r"found no active RDMA device",
    "gid_no_routable": r"exposes? no routable (?:RoCE v2 )?GID",
    "gid_no_common": r"resolved no common RoCE v2 GID",
    "gloo_error": r"gloo/transport/tcp|Unable to resolve hostname to a \(local\) address|GLOO_SOCKET_IFNAME|matching no non-loopback interface",
    "gloo_timeout_err": r"Timed out waiting [0-9]+ms for (?:recv|send)",
    "gloo_peer_closed": r"Connection closed by peer",
    "kv_config_err": r"kv_cache_configurator\.py|size_plan|allocate_memory_pool|get_available_gpu_memory",
    "post_load_fn_err": r"prepare_moe_topk|warmup_bf16_gemms|install_megakernel|allocate_mega_moe_buffers|check_quantized_moe_compatibility|dist_barrier_after_load",
    "wt_time_lt_1800s": r"Load weight end\. elapsed=(?:[0-9]{1,3}|1[0-7][0-9]{2})\.",
    "wt_time_1800_3600s": r"Load weight end\. elapsed=(?:1[89][0-9]{2}|2[0-9]{3}|3[0-5][0-9]{2})\.",
    "wt_time_3600_7200s": r"Load weight end\. elapsed=(?:3[6-9][0-9]{2}|[4-6][0-9]{3}|7[01][0-9]{2})\.",
    "wt_time_gt_7200s": r"Load weight end\. elapsed=(?:7[2-9][0-9]{2}|[89][0-9]{3}|[1-9][0-9]{4})\.",
    "kv_oom_or_small": r"Not enough memory|memory is not enough|KV cache.*negative|negative.*kv",
    "mnnvl_error": r"mnnvl\.cc|MNNVL|IMEX",
    "rendezvous_timeout": r"rendezvous: node .+ waited|rendezvous: rank 0 could not determine",
    "assertion_error": r"AssertionError",
    "value_error": r"ValueError:",
    "runtime_error": r"RuntimeError:",
    "attribute_error": r"AttributeError:",
    "type_error": r"TypeError:",
    "key_error": r"KeyError:",
    "sigkill": r"Signals\.SIGKILL|exitcode: -9|exit code 137",
    "sigabrt": r"Signals\.SIGABRT|exitcode: -6|Segmentation fault|exit code (134|139)",
    "startup_timeout": r"server did not become ready|startup timed out",
    "stage_dist_init_begin": r"Init torch distributed begin",
    "stage_dist_init_end": r"Init torch distributed ends",
    "stage_load_weight_begin": r"Load weight begin",
    "stage_pp_warm_done": r"pp_warm_done\b",
    "stage_weight_loading": r"Loading safetensors checkpoint shards|Multi-thread loading shards",
    "mt_shard_gt_0": r"\b(?:[1-9]|[1-9]\d|1\d\d|2\d\d)/213\b",
    "mt_shard_gt_50": r"\b(?:[5-9]\d|1\d\d|2\d\d)/213\b",
    "mt_shard_gt_100": r"\b(?:1\d\d|2\d\d)/213\b",
    "mt_shard_gt_150": r"\b(?:1[5-9]\d|2\d\d)/213\b",
    "mt_shard_213": r"\b213/213\b",
    "stage_weight_0_24pct": r"(?:Loading safetensors checkpoint shards|Multi-thread loading shards):\s*(?:[0-9]|1[0-9]|2[0-4])%",
    "stage_weight_25_49pct": r"(?:Loading safetensors checkpoint shards|Multi-thread loading shards):\s*(?:2[5-9]|[34][0-9])%",
    "stage_weight_50pct": r"(?:Loading safetensors checkpoint shards|Multi-thread loading shards):\s*(?:[5-9][0-9]|100)%",
    "stage_weight_50_74pct": r"(?:Loading safetensors checkpoint shards|Multi-thread loading shards):\s*(?:[56][0-9]|7[0-4])%",
    "stage_weight_75_99pct": r"(?:Loading safetensors checkpoint shards|Multi-thread loading shards):\s*(?:7[5-9]|[89][0-9])%",
    "stage_weight_100pct": r"(?:Loading safetensors checkpoint shards|Multi-thread loading shards):\s*100%",
    "stage_load_weight_end": r"Load weight end",
    "stage_kv_alloc": r"KV Cache is allocated|Memory pool end",
    "stage_cuda_graph": r"Capture cuda graph begin",
    "stage_cuda_graph_done": r"Capture cuda graph end",
    "stage_server_ready": r"The server is fired up and ready to roll",
    "stage_http_listen": r"Uvicorn running on|Application startup complete",
    "stage_health_200": r'GET /health HTTP/1\.[01]" 200',
    "stage_health_gen": r"/health_generate",
    "stage_health_gen_200": r'GET /health_generate HTTP/1\.[01]" 200',
    "stage_health_gen_500": r'GET /health_generate HTTP/1\.[01]" 5\d\d',
    "stage_warmup_timeout": r"Read timed out.*(?:/generate|/health_generate)|warmup timed out",
    "stage_prefill_batch": r"Prefill batch\.",
    "stage_decode_batch": r"Decode batch\.",
    "watchdog_soft": r"watchdog timeout \(soft=True",
    "watchdog_hard": r"watchdog timeout \(soft=False",
    "watchdog_nccl": r"WorkNCCL|Watchdog caught collective",
    "heartbeat_timeout": r"heartbeat monitor|FlightRecorder",
    "gloo_timeout": r"gloo[^\n]*timed? out|Timed out waiting",
    "traceback": r"Traceback \(most recent call last\)",
    "param_not_found": r"not found in params_dict",
    "monitored_barrier_err": r"didn't finish loading|monitored_barrier",
    "process_weights_err": r"process_weights_after_loading",
    "tcpstore_closed": r"Broken pipe|Connection reset by peer|Socket Timeout|TCPStore[^\n]*(?:timed? out|closed|refused|reset)",
    "sigterm_exit": r"SIGTERM|exitcode: -15|Terminated",
    "mmap_or_io_err": r"Bus error|SIGBUS|Input/output error|Stale file handle|ESTALE",
    "worker_exit": r"container.+exit.?code|worker pods? failed|Command failed",
    "deadline": r"DeadlineExceeded|PPU scheduling timed out|workload timeout",
    "configuration_key": r"KeyError: 'PPU_CI_[A-Z0-9_]+'",
}

RANK1_RULES = {
    "r1_dist_init_end": r"Init torch distributed ends",
    "r1_load_weight_begin": r"Load weight begin",
    "r1_pp_warm_done": r"pp_warm_done\b",
    "r1_weight_loading": r"Loading safetensors checkpoint shards",
    "r1_weight_50pct": r"Loading safetensors checkpoint shards:\s*(?:[5-9][0-9]|100)%",
    "r1_weight_100pct": r"Loading safetensors checkpoint shards:\s*100%",
    "r1_load_weight_end": r"Load weight end",
    "r1_wt_time_lt_1800s": r"Load weight end\. elapsed=(?:[0-9]{1,3}|1[0-7][0-9]{2})\.",
    "r1_wt_time_1800_3600s": r"Load weight end\. elapsed=(?:1[89][0-9]{2}|2[0-9]{3}|3[0-5][0-9]{2})\.",
    "r1_wt_time_3600_7200s": r"Load weight end\. elapsed=(?:3[6-9][0-9]{2}|[4-6][0-9]{3}|7[01][0-9]{2})\.",
    "r1_wt_time_gt_7200s": r"Load weight end\. elapsed=(?:7[2-9][0-9]{2}|[89][0-9]{3}|[1-9][0-9]{4})\.",
    "r1_gloo_timeout_err": r"Timed out waiting [0-9]+ms for (?:recv|send)",
    "r1_gloo_peer_closed": r"Connection closed by peer",
    "r1_kv_config_err": r"kv_cache_configurator\.py|size_plan|allocate_memory_pool|get_available_gpu_memory",
    "r1_kv_alloc": r"KV Cache is allocated|Memory pool end",
    "r1_cuda_graph": r"Capture cuda graph begin",
    "r1_cuda_graph_done": r"Capture cuda graph end",
    "r1_server_ready": r"The server is fired up and ready to roll",
    "r1_watchdog": r"Watchdog (?:caught|timeout|triggered)|triggered watchdog",
    "r1_watchdog_soft": r"watchdog timeout \(soft=True",
    "r1_watchdog_hard": r"watchdog timeout \(soft=False",
    "r1_traceback": r"Traceback \(most recent call last\)",
    "r1_nccl_error": r"NCCL error|DistBackendError|TCPStore|ProcessGroupNCCL",
    "r1_tcpstore_error": r"TCPStore|Socket Timeout|Connection refused|ECONNREFUSED|Connect timeout",
    "r1_sigabrt": r"Signals\.SIGABRT|exitcode: -6|Segmentation fault|exit code (134|139)",
    "r1_sigkill": r"Signals\.SIGKILL|exitcode: -9|exit code 137",
    "r1_oom": r"OOMKilled|out of memory|OutOfMemoryError",
}

_CHUNK_BYTES = 262144
_OVERLAP_BYTES = 512
_MAX_SCAN_BYTES = 32 * 1024 * 1024


def diagnostic_codes(text, rules=None):
    active_rules = RULES if rules is None else rules
    return [
        name for name, pattern in active_rules.items() if re.search(pattern, text, re.I)
    ]


def _safe_subdir(root, parts):
    current = root
    for part in parts:
        current = current / part
        try:
            if not stat.S_ISDIR(current.lstat().st_mode):
                return None
        except OSError:
            return None
    return current


def _open_regular_file(path):
    try:
        pre_info = path.lstat()
    except OSError:
        return None, 0
    if not stat.S_ISREG(pre_info.st_mode):
        return None, 0
    try:
        fd = os.open(
            path,
            os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0),
        )
    except OSError:
        return None, 0
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            os.close(fd)
            return None, 0
        return os.fdopen(fd, "rb"), info.st_size
    except OSError:
        os.close(fd)
        return None, 0


def _scan_file_codes(path, rules=None):
    active_rules = RULES if rules is None else rules
    stream, size = _open_regular_file(path)
    if stream is None:
        return set()
    found = set()
    remaining = dict(active_rules)
    with stream:
        if size > _MAX_SCAN_BYTES:
            stream.seek(size - _MAX_SCAN_BYTES)
        carry = ""
        scanned = 0
        while remaining and scanned < _MAX_SCAN_BYTES:
            chunk = stream.read(_CHUNK_BYTES)
            if not chunk:
                break
            scanned += len(chunk)
            text = carry + chunk.decode(errors="replace")
            for name in list(remaining):
                if re.search(remaining[name], text, re.I):
                    found.add(name)
                    del remaining[name]
            carry = text[-_OVERLAP_BYTES:]
    return found


def _read_regular_text(path, limit=64):
    stream, _ = _open_regular_file(path)
    if stream is None:
        return ""
    with stream:
        return stream.read(limit).decode(errors="replace")


def _previous_runs(parent, current_run):
    try:
        children = list(parent.iterdir())
    except OSError:
        return []
    previous = []
    for child in children:
        if not re.fullmatch(r"[0-9]+-[0-9]+", child.name):
            continue
        run_id, attempt = map(int, child.name.split("-"))
        if run_id != current_run and _safe_subdir(parent, (child.name,)) is not None:
            previous.append((run_id, attempt, child))
    return sorted(previous, reverse=True)


def _archive_roots(run):
    yield run
    try:
        children = list(run.iterdir())
    except OSError:
        return
    for child in children:
        if re.fullmatch(r"[0-9a-f]{12}", child.name):
            scope = _safe_subdir(run, (child.name,))
            if scope is not None:
                yield scope


def _retained_models(run, models):
    for scope in _archive_roots(run):
        for tier in ("fast_sanity", "model_sanity"):
            tier_dir = _safe_subdir(scope, (tier,))
            if tier_dir is None:
                continue
            for model in models:
                directory = _safe_subdir(tier_dir, (model,))
                if directory is None:
                    continue
                try:
                    index_info = (directory / "index.json").lstat()
                except OSError:
                    continue
                if stat.S_ISREG(index_info.st_mode):
                    yield tier, model, directory


def _iter_safe_rank_logs(ranks_dir):
    try:
        entries = sorted(ranks_dir.iterdir())
    except OSError:
        return
    for entry in entries:
        if not re.fullmatch(r"rank-[0-9]+", entry.name):
            continue
        rank_dir = _safe_subdir(ranks_dir, (entry.name,))
        if rank_dir is None:
            continue
        is_rank1 = entry.name == "rank-1"
        try:
            rank_children = sorted(rank_dir.iterdir())
        except OSError:
            continue
        for item in rank_children:
            if item.suffix == ".log":
                yield item, is_rank1
        results_dir = _safe_subdir(rank_dir, ("results",))
        if results_dir is not None:
            try:
                result_children = sorted(results_dir.iterdir())
            except OSError:
                continue
            for item in result_children:
                if item.suffix == ".log":
                    yield item, is_rank1


def _read_json_dict(path, limit=65536):
    stream, size = _open_regular_file(path)
    if stream is None:
        return None
    if size > limit:
        stream.close()
        return None
    with stream:
        try:
            payload = json.loads(stream.read(limit).decode(errors="replace"))
        except (ValueError, OSError):
            return None
    return payload if isinstance(payload, dict) else None


def _archive_outcome(directory):
    index_doc = _read_json_dict(directory / "index.json", limit=262144)
    if index_doc is not None:
        job_outcome = index_doc.get("job_outcome")
        if job_outcome in ("pass", "fail"):
            return job_outcome
    phase_val = _read_regular_text(directory / "recipe-phase.txt").strip()
    if phase_val in ("cleanup", "image_resolution"):
        return "fail"
    summary_doc = _read_json_dict(directory / "summary.json")
    if summary_doc is None:
        return None
    outcome = summary_doc.get("outcome")
    return outcome if isinstance(outcome, str) else None


def _archive_codes(directory):
    outcome = _archive_outcome(directory)
    if outcome == "pass":
        return set()
    codes = set()
    for filename in (
        "launcher.log",
        "runtime.log",
        "pod-status.json",
        "sglang_serve_fast_full.log",
        "sglang_serve_fast_2l.log",
        "sglang_serve_model_sanity.log",
        "page-cache-warm.log",
        "bench-warmup-1.log",
        "bench-measure-1.log",
        "bench-warmup-16.log",
        "bench-measure-16.log",
        "evalscope.log",
    ):
        codes.update(_scan_file_codes(directory / filename))
    ranks_dir = _safe_subdir(directory, ("ranks",))
    if ranks_dir is not None:
        for path, is_rank1 in _iter_safe_rank_logs(ranks_dir):
            codes.update(_scan_file_codes(path))
            if is_rank1:
                codes.update(_scan_file_codes(path, RANK1_RULES))
        status_val = _read_regular_text(ranks_dir / "rank-1.status").strip()
        if status_val == "0":
            # Only attach healthy rank-1 exit status as context on a failed run;
            # never let a healthy worker exit turn a clean archive into a failure.
            if outcome == "fail" or codes:
                codes.add("r1_status_0")
        elif status_val:
            codes.add("r1_status_nonzero")
    return codes


def recent_failures(root, current_run, model_filter="", fallback_root=None):
    roots = [root]
    if fallback_root is not None and fallback_root != root:
        roots.append(fallback_root)
    models = sorted(
        policy.stem
        for policy in MODEL_POLICY_ROOT.glob("*.yaml")
        if re.fullmatch(r"[a-z0-9][a-z0-9.-]*", policy.stem)
    )
    selected = {m.strip() for m in (model_filter or "").split(",") if m.strip()}
    if selected:
        models = [m for m in models if m in selected]
    for candidate_root in roots:
        try:
            if not stat.S_ISDIR(candidate_root.lstat().st_mode):
                continue
        except OSError:
            continue
        for parts in (RETENTION_SUBPATH, ("sglang-for-sail",)):
            parent = _safe_subdir(candidate_root, parts)
            if parent is None:
                continue
            found_archive = False
            for run_id, _, run in _previous_runs(parent, current_run):
                codes_by_model = {}
                for tier, model, directory in _retained_models(run, models):
                    found_archive = True
                    codes = _archive_codes(directory)
                    if codes:
                        codes_by_model.setdefault((tier, model), set()).update(codes)
                if codes_by_model:
                    return [
                        {
                            "run_id": run_id,
                            "tier": tier,
                            "model": model,
                            "codes": sorted(codes),
                        }
                        for (tier, model), codes in sorted(codes_by_model.items())
                    ]
            if found_archive:
                return []
    return []


def publish_recent_failures(root, current_run, model_filter="", fallback_root=None):
    records = recent_failures(
        root, current_run, model_filter=model_filter, fallback_root=fallback_root
    )
    if records:
        print(json.dumps({"prior_cluster_diagnostics": records}, sort_keys=True))
