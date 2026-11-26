"""External GLM-5.3-Flash FP8 PD radix parity; never launches models or a router.

Run on an idle, dedicated deployment with CP8/EP8 P, TP8/DP8/EP8 D,
BF16 KV, Triton KDA, and native MTP steps/top-k/draft tokens 3/1/4:
  python test_glm_cp_ep_radix.py --prefill-url http://P:30000 \
      --decode-url http://D:30000 --router-url http://R:8000 \
      --model-path /exact/GLM-5.3-Flash-FP8 --deployment-evidence /local/observed.json
Redirect stdout to a JSON file for reproduction. Different worker checkpoint paths
can be supplied with --decode-model-path; --tokenizer-path is the client's copy of
the SAME checkpoint, not a substitute model. No small-model or TCP fallback exists.

The evidence JSON must contain prefill/decode objects with is_ppu=true,
device_sm=89, MOONCAKE_PROTOCOL="rdma", MC_FORCE_TCP=null, model_path,
config_sha256 and source (a worker environment/device/log observation collected
by the operator). Include router_workers as an operator snapshot of the router's
P/D topology; this checkout has no /workers HTTP endpoint. config_sha256 is the
SHA-256 of the worker checkpoint config.json serialized as canonical JSON.
HTTP cannot independently certify physical RDMA; this evidence is recorded as
external observation, not as a transport test. Do not derive it from the client's
environment. SGLANG_ADMIN_API_KEY is used only for worker /flush_cache calls;
SGLANG_API_KEY remains the normal worker/router credential.

Each length gets cold (flush P+D), hot (no flush), and flushDecodeOnly (flush D,
retain P) generations, checking token parity and finite aligned logprobs.
Logprob deltas are diagnostic; model accuracy is validated separately by GSM8K.
The page size is 64 * checkpoint index_kpool. Prefill prefix matching leaves the
final input token for logits, while Decode may reuse the entire prompt on hot
requests. A zero-hit assertion applies only to flushDecodeOnly when no full
Prefill page is reusable. D-only flush isolates P reuse and the real KV/linear-
state transfer to D. --self-test checks only the client oracle on CPU, NOT
inference/cache/RDMA.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

ARCHITECTURE = "Glm5NextForConditionalGeneration"
LENGTHS = (256, 257, 4096, 4097)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def json_ready(value):
    # Preserve invalid server NaN/Inf values in a *valid* failing repro JSON.
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    return value


def request(url, method="GET", payload=None, timeout=600, *, admin=False):
    headers = {"Content-Type": "application/json"}
    # Optional authentication is not copied into the reproduction report.
    key = (
        os.environ.get("SGLANG_ADMIN_API_KEY", os.environ.get("SGLANG_API_KEY"))
        if admin
        else os.environ.get("SGLANG_API_KEY")
    )
    if key:
        headers["Authorization"] = f"Bearer {key}"
    body = None if payload is None else json.dumps(payload).encode()
    try:
        with urlopen(
            Request(url, data=body, headers=headers, method=method), timeout=timeout
        ) as response:
            text = response.read().decode()
    except HTTPError as exc:
        raise RuntimeError(
            f"{method} {url}: HTTP {exc.code}: {exc.read().decode()}"
        ) from exc
    # /flush_cache and /health return plain text; /generate must return JSON.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def make_input_ids(tokenizer, length):
    suffix = tokenizer.encode(
        "\nThe access code is 739391. Copy the access code: 739391\n"
        "The access code is 739391. Copy the access code:",
        add_special_tokens=False,
    )
    filler = tokenizer.encode("These are background notes. ", add_special_tokens=False)
    require(filler and 0 < len(suffix) < length, "invalid tokenizer prompt fragments")
    count = length - len(suffix)
    return (filler * ((count + len(filler) - 1) // len(filler)))[:count] + suffix


def validate_response(output, length, new_tokens):
    require(isinstance(output, dict), "generate did not return an object")
    meta = output["meta_info"]
    ids = output["output_ids"]
    require(meta["prompt_tokens"] == length, "input_ids were truncated/retokenized")
    require(len(ids) == new_tokens, "generation ended early or returned incomplete IDs")
    require(
        all(type(token) is int and token >= 0 for token in ids), "invalid output IDs"
    )
    require(meta["completion_tokens"] == len(ids), "completion count differs from IDs")
    require(meta["finish_reason"]["type"] == "length", "generation aborted/ended early")
    rows = meta["output_token_logprobs"]
    require(len(rows) == len(ids), "missing output logprobs (including PD first token)")
    for token, row in zip(ids, rows, strict=True):
        require(len(row) >= 2 and row[1] == token, "logprob token IDs are not aligned")
        require(
            isinstance(row[0], (int, float)) and math.isfinite(row[0]),
            "non-finite/missing output logprob",
        )
    cached = meta["cached_tokens"]
    require(
        type(cached) is int and 0 <= cached <= length, "invalid/double-counted cache hit"
    )
    return cached


def parity(reference, candidate):
    require(
        reference["output_ids"] == candidate["output_ids"], "output token parity failed"
    )
    expected = reference["meta_info"]["output_token_logprobs"]
    actual = candidate["meta_info"]["output_token_logprobs"]
    require(
        len(expected) == len(actual) and len(expected) > 0, "logprob lengths differ"
    )
    errors = []
    for index, (a, b) in enumerate(zip(expected, actual, strict=True)):
        require(a[1] == b[1], f"logprob token parity failed at {index}")
        require(math.isfinite(a[0]) and math.isfinite(b[0]), "non-finite logprob")
        errors.append(abs(a[0] - b[0]))
    return {"tokens_equal": True, "max_abs_logprob_error": max(errors)}


def preflight(args, report):
    from transformers import AutoTokenizer
    from transformers.utils.hub import cached_file

    client_model = args.tokenizer_path or args.model_path
    config_path = cached_file(client_model, "config.json")
    config = json.loads(Path(config_path).read_text())
    config_sha256 = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    require(
        config["architectures"] == [ARCHITECTURE], "not the GLM-5.3-Flash architecture"
    )
    text_config = config.get("text_config", config)
    quant = (
        config.get("quantization_config")
        or config.get("compression_config")
        or text_config.get("quantization_config")
        or text_config.get("compression_config", {})
    )
    groups = quant.get("config_groups") or {}
    channelwise_fp8 = (
        quant.get("quant_method") == "compressed-tensors"
        and bool(groups)
        and all(
            (group.get("weights") or {}).get("type") == "float"
            and (group.get("weights") or {}).get("num_bits") == 8
            and (group.get("weights") or {}).get("strategy") == "channel"
            and (group.get("input_activations") or {}).get("type") == "float"
            and (group.get("input_activations") or {}).get("num_bits") == 8
            and (group.get("input_activations") or {}).get("strategy") == "token"
            and (group.get("input_activations") or {}).get("dynamic") is True
            for group in groups.values()
        )
    )
    require(
        quant.get("quant_method") == "fp8" or channelwise_fp8,
        "checkpoint must use floating-point W8A8, not FP4/BF16",
    )
    kpool = text_config.get("index_kpool", config.get("index_kpool", 1))
    require(type(kpool) is int and kpool > 0, "invalid checkpoint index_kpool")
    report["checkpoint"] = {
        "model_path": args.model_path,
        "decode_model_path": args.decode_model_path or args.model_path,
        "tokenizer_path": client_model,
        "config": config,
        "config_sha256": config_sha256,
        "index_kpool": kpool,
    }
    evidence = json.loads(Path(args.deployment_evidence).read_text())
    report["external_deployment_observations"] = evidence
    report["rdma_verification"] = (
        "external observations; not independently verifiable over HTTP"
    )
    fields = (
        "model_path",
        "tp_size",
        "ep_size",
        "attn_cp_size",
        "cp_strategy",
        "enable_prefill_cp",
        "enable_cp_tp_group_sharing",
        "nnodes",
        "pp_size",
        "dp_size",
        "attn_dp_size",
        "enable_dp_attention",
        "disaggregation_mode",
        "disaggregation_transfer_backend",
        "page_size",
        "disable_radix_cache",
        "enable_hierarchical_cache",
        "enable_pd_role_switch",
        "enable_unified_memory",
        "disaggregation_decode_enable_radix_cache",
        "quantization",
        "kv_cache_dtype",
        "linear_attn_backend",
        "linear_attn_prefill_backend",
        "linear_attn_decode_backend",
        "linear_attn_verify_backend",
        "chunked_prefill_size",
        "dcp_size",
        "enable_kv_cache_sharding",
        "speculative_algorithm",
        "speculative_eagle_topk",
        "speculative_draft_model_path",
        "speculative_num_steps",
        "speculative_num_draft_tokens",
        "mamba_radix_cache_strategy",
        "version",
        "random_seed",
    )
    router_workers = evidence["router_workers"]
    report["router_workers"] = router_workers
    workers = router_workers["workers"]
    require(len(workers) == 2, "router must have exactly one P and one D worker")
    actual_routes = {
        worker["worker_type"]: worker["url"].rstrip("/") for worker in workers
    }
    require(
        actual_routes == {"prefill": args.prefill_url, "decode": args.decode_url},
        "router does not route to the supplied P/D URLs; flush isolation is invalid",
    )
    report["workers"] = {}
    for role, url in (("prefill", args.prefill_url), ("decode", args.decode_url)):
        observed = evidence[role]
        require(
            observed["is_ppu"] is True and observed["device_sm"] == 89,
            f"{role}: not PPU SM89",
        )
        require(observed["MOONCAKE_PROTOCOL"] == "rdma", f"{role}: not observed RDMA")
        require(observed["MC_FORCE_TCP"] is None, f"{role}: forced TCP")
        require(bool(observed["source"]), f"{role}: missing observation provenance")
        request(url + "/health", timeout=args.timeout)
        info = request(url + "/server_info", timeout=args.timeout)
        model = request(url + "/model_info", timeout=args.timeout)
        report["workers"][role] = {
            "server_info": {f: info.get(f) for f in fields},
            "model_info": model,
        }
        require(
            model["architectures"] == [ARCHITECTURE],
            f"{role}: wrong running architecture",
        )
        expected_path = (
            (args.decode_model_path or args.model_path)
            if role == "decode"
            else args.model_path
        )
        require(
            model["model_path"] == expected_path, f"{role}: wrong running checkpoint"
        )
        require(
            observed["model_path"] == expected_path
            and observed["config_sha256"] == config_sha256,
            f"{role}: worker checkpoint/config fingerprint differs from the client",
        )
        require(
            model["load_format"] != "dummy",
            f"{role}: dummy weights are not precision coverage",
        )
        # None means checkpoint auto-detection; the exact checkpoint above is FP8.
        require(
            info["quantization"] in (None, "fp8", "compressed-tensors"),
            f"{role}: runtime overrides FP8",
        )
        expected = {
            "tp_size": 8,
            "ep_size": 8,
            "nnodes": 1,
            "pp_size": 1,
            "dp_size": 1,
            "disaggregation_mode": role,
            "disaggregation_transfer_backend": "mooncake",
            "disable_radix_cache": False,
            "enable_hierarchical_cache": False,
            "enable_pd_role_switch": False,
            "enable_unified_memory": False,
            "dcp_size": 1,
            "enable_kv_cache_sharding": False,
            "kv_cache_dtype": "bfloat16",
            "linear_attn_prefill_backend": "triton",
            "speculative_num_steps": 3,
            "speculative_eagle_topk": 1,
            "speculative_num_draft_tokens": 4,
        }
        if role == "prefill" and getattr(args, "allow_prefill_no_spec", False):
            expected.update(
                speculative_num_steps=None,
                speculative_eagle_topk=None,
                speculative_num_draft_tokens=None,
            )
        if role == "prefill":
            expected.update(
                attn_cp_size=8,
                cp_strategy="interleave",
                enable_prefill_cp=True,
                enable_cp_tp_group_sharing=True,
                page_size=64 * kpool,
            )
        else:
            expected.update(
                attn_cp_size=1,
                attn_dp_size=8,
                enable_dp_attention=True,
                disaggregation_decode_enable_radix_cache=True,
            )
        for field, value in expected.items():
            require(
                info[field] == value,
                f"{role}: {field}={info[field]!r}, expected {value!r}",
            )
        effective_decode = (
            info["linear_attn_decode_backend"] or info["linear_attn_backend"]
        )
        effective_verify = info["linear_attn_verify_backend"] or (
            effective_decode if effective_decode == "flashinfer" else "triton"
        )
        require(
            effective_decode == "triton" and effective_verify == "triton",
            f"{role}: effective linear-attention decode/verify is not Triton",
        )
        algorithm = info["speculative_algorithm"]
        if not (role == "prefill" and getattr(args, "allow_prefill_no_spec", False)):
            require(
                algorithm in ("NEXTN", "EAGLE"), f"{role}: native MTP is required"
            )
        require(
            info["speculative_draft_model_path"] in (None, expected_path),
            f"{role}: non-native draft",
        )
    return (
        AutoTokenizer.from_pretrained(client_model, trust_remote_code=True),
        64 * kpool,
    )


def run(args, report):
    tokenizer, page_size = preflight(args, report)
    report["cases"] = []
    report["flushes"] = []

    def flush(role, url):
        result = request(
            url + "/flush_cache?timeout=30", "POST", timeout=args.timeout, admin=True
        )
        report["flushes"].append({"role": role, "response": result})

    for length in LENGTHS:
        ids = make_input_ids(tokenizer, length)
        payload = {
            "input_ids": ids,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": args.new_tokens,
                "ignore_eos": True,
            },
            "return_logprob": True,
            # Output-only logprobs: requesting prompt logprobs at zero disables reuse.
            "logprob_start_len": -1,
            "stream": False,
        }
        case = {"length": length, "request": payload, "phases": {}}
        report["cases"].append(case)
        flush("prefill", args.prefill_url)
        flush("decode", args.decode_url)
        reference = None
        for phase in ("cold", "hot", "flushDecodeOnly"):
            if phase == "flushDecodeOnly":
                flush("decode", args.decode_url)  # Deliberately NEVER flush P here.
            start = time.monotonic()
            output = request(
                args.router_url + "/generate", "POST", payload, args.timeout
            )
            record = {"response": output, "elapsed_s": time.monotonic() - start}
            case["phases"][
                phase
            ] = record  # Preserve even a failing response for repro.
            cached = validate_response(output, length, args.new_tokens)
            if phase == "cold":
                require(
                    cached == 0, "cold generation hit cache; deployment is not isolated"
                )
                reference = output
            elif (length - 1) // page_size:
                require(
                    cached > 0,
                    f"{phase}: no reusable page; radix/builder/state restore regressed",
                )
            elif phase == "flushDecodeOnly":
                require(cached == 0, f"{phase}: hit reported without a reusable page")
            record["parity_vs_cold"] = parity(reference, output)
    for role, url in (("prefill", args.prefill_url), ("decode", args.decode_url)):
        request(url + "/health", timeout=args.timeout)
    report["status"] = "passed"


def self_test():
    # These are protocol examples for the oracle, not simulated model outputs.
    example = {
        "output_ids": [7, 8],
        "meta_info": {
            "prompt_tokens": 257,
            "completion_tokens": 2,
            "cached_tokens": 256,
            "finish_reason": {"type": "length"},
            "output_token_logprobs": [[-0.2, 7], [-0.3, 8]],
        },
    }
    validate_response(example, 257, 2)
    full_decode_hit = copy.deepcopy(example)
    full_decode_hit["meta_info"]["prompt_tokens"] = 256
    full_decode_hit["meta_info"]["cached_tokens"] = 256
    validate_response(full_decode_hit, 256, 2)
    parity(example, copy.deepcopy(example))
    changed_logprob = copy.deepcopy(example)
    changed_logprob["meta_info"]["output_token_logprobs"][0][0] = -2.0
    result = parity(example, changed_logprob)
    require(result["max_abs_logprob_error"] > 0, "logprob delta was not recorded")

    def change_token(output):
        output["output_ids"][0] = 9
        output["meta_info"]["output_token_logprobs"][0][1] = 9

    mutations = (
        change_token,
        lambda x: x["meta_info"]["output_token_logprobs"].pop(),
        lambda x: x["meta_info"]["output_token_logprobs"][0].__setitem__(
            0, float("nan")
        ),
        lambda x: x["meta_info"].__setitem__("cached_tokens", 258),
    )
    for mutate in mutations:
        bad = copy.deepcopy(example)
        mutate(bad)
        try:
            validate_response(bad, 257, 2)
            parity(example, bad)
        except ValueError:
            continue
        raise RuntimeError("oracle accepted a corrupt protocol example")
    print(
        json.dumps(
            {
                "status": "passed",
                "self_test": "client oracle only",
                "rejected_cases": len(mutations),
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    for name in (
        "prefill-url",
        "decode-url",
        "router-url",
        "model-path",
        "deployment-evidence",
    ):
        parser.add_argument("--" + name)
    parser.add_argument("--decode-model-path")
    parser.add_argument("--allow-prefill-no-spec", action="store_true")
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    for name in (
        "prefill_url",
        "decode_url",
        "router_url",
        "model_path",
        "deployment_evidence",
    ):
        if not getattr(args, name):
            parser.error("--" + name.replace("_", "-") + " is required")
    report = {
        "schema_version": 1,
        "validation_scope": "cache/token smoke; logprob deltas are diagnostic",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "status": "failed",
    }
    try:
        require(
            args.new_tokens >= 2, "at least two tokens are needed to exercise decode"
        )
        require(
            math.isfinite(args.timeout) and args.timeout > 0,
            "timeout must be finite and positive",
        )
        for name in ("prefill_url", "decode_url", "router_url"):
            url = getattr(args, name).rstrip("/")
            parsed = urlsplit(url)
            require(
                parsed.scheme in ("http", "https")
                and parsed.netloc
                and not parsed.username
                and not parsed.password,
                "invalid URL (credentials must use SGLANG_API_KEY)",
            )
            require(
                not parsed.query and not parsed.fragment,
                "base URLs must not contain query/fragment",
            )
            setattr(args, name, url)
        require(
            len({args.prefill_url, args.decode_url, args.router_url}) == 3,
            "P/D/router must be distinct URLs",
        )
        run(args, report)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(json_ready(report), indent=2, allow_nan=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
