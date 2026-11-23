# SGLang PPU Tests

This directory holds the PPU (zhenwu_m890 / 890P and OAM-810E) test assets for the
SGLang fork: declarative serving **recipes**, per-model test **policies**, GSM8K
**golden** accuracy references, and the device-facing **e2e / unit / module**
pytest suites.

The recipe + policy pairs drive the GitHub Actions PPU CI
([`.github/workflows/pr-test-ppu.yml`](../../.github/workflows/pr-test-ppu.yml))
through the resolver in
[`scripts/ci/ppu/ppu_recipe.py`](../../scripts/ci/ppu/ppu_recipe.py). The runner
logic, cluster provisioning and CI labels are documented in
[`scripts/ci/ppu/README.md`](../../scripts/ci/ppu/README.md); this file documents
the test tree and the resolved serve-flag matrix.

## Directory layout

| Path | Schema / kind | Purpose |
| --- | --- | --- |
| [`recipes/`](recipes) | `ppu-serving/v1` YAML | Declarative serve config per model: `base_args`, `base_env`, `features`, `variants`, `hardware_overrides`, `presets`, topology and HF checkpoint id. |
| [`policy/models/`](policy/models) | `ppu-test-policy/v1` YAML | Per-model CI policy: `hardware`, `accuracy_gate`, and `tiers` (`fast_sanity`, `model_sanity`). Selects a recipe and its stage pipeline. |
| [`policy/datasets/`](policy/datasets) | `ppu-dataset/v1` YAML | Eval dataset definitions referenced as `<id>@<version>` (currently [`gsm8k@2026-11`](policy/datasets/gsm8k.yaml)). |
| [`golden/`](golden) | `schema_version: 2` JSON | Reference GSM8K score + tolerance band per `golden_model_key`, named `accuracy_<key>_890p.json`. |
| [`e2e/`](e2e) | pytest / `unittest` | Device end-to-end server tests (the CI *legacy single-device sanity* set). Launch a real server via `popen_launch_server` and check GSM8K / crash regressions. |
| [`unit/`](unit) | pytest / `unittest` | Kernel- and contract-level unit tests (QSA backends, DSA indexer, DSV4 MHC, migration contracts, ...). |
| [`module/`](module) | pytest / `unittest` | Module integration tests (DeepGEMM MoE W4A8, optimal fused-MoE, DeepEP hooks). |
| [`qsa_validation.md`](qsa_validation.md) | notes | Historical QSA backend compatibility and hardware validation evidence. |
| `test_kda_pla_replayssm.py`, `test_kda_replayssm_pool.py` | pytest | KDA / PLA replay-SSM regression tests at the tree root. |

## Configuration resolution rules

`ppu_recipe.resolve_recipe()` folds a recipe and an optional policy tier into the
effective `sglang.launch_server` argv, server env and a `config_digest`. Layers
apply in this order (last wins, with inverse-flag clearing and JSON deep-merge):

1. `model.base_args` / `model.base_env`
2. `variants.<variant>` (default: `default`)
3. `hardware_overrides.<hardware>` (`zhenwu_m890` or `zhenwu_810e`)
4. enabled `features` — `ppu.default_features` overridden by `presets.<preset>.select.features`
5. `presets.<preset>` (`tp`, `args`, `env`, `remove_args`, `json_args`)
6. policy switches — radix cache (`--disable-radix-cache`), dummy weights
   (`--load-format dummy`) and `layer_override`
   (`--json-model-override-args {"num_hidden_layers": N}`)
7. the runner always appends `--enable-metrics`, then owns
   `--model-path`, `--tokenizer-path`, `--served-model-name`, `--host`, `--port`

Guardrails enforced by the validator:

- **Runner-owned flags** (`--model-path`, `--load-format`, `--enable-metrics`,
  `--disable-radix-cache`, `--host`, `--port`, ...) must not appear in a recipe.
- **Env allowlist**: server env keys must start with `VLLM_`, `SGLANG_`, `SGL_`,
  `NCCL_`, `TORCH_`, `OMP_`, `DG_`, `HGGC_`, `MC_` or `MOONCAKE_`. External
  `DG_*` / `HGGC_*` / `SGLANG_NSA_DUAL_STREAM` keys must be declared in
  `ppu.external_env`. Runtime-mode keys (`CUDA_VISIBLE_DEVICES`, `HF_HOME`,
  `VLLM_SERVER_DEV_MODE`, ...) are forbidden in recipes.
- **Layer override** may only come from a test policy, never from a recipe.
- **`model_sanity` stages** must include one each of `sanity`, `evalscope` and
  `bench`, all with `cache: enabled` (shared server); `bench` with cache enabled
  requires a `cache_mode`.
- **Redaction**: any internal host, mirror, absolute path, IP literal, credential
  or prohibited token in a recipe/policy/dataset fails validation.

Regenerate / verify (the tables below are derived from these commands and must be
kept in sync when a recipe or policy changes):

```bash
# Structure + redaction check for every recipe/policy/dataset under test/ppu
python3 -B scripts/ci/ppu/ppu_recipe.py validate --all --sglang-parser skip

# Resolve one row of the matrix (effective argv / env / digest)
python3 -B scripts/ci/ppu/ppu_recipe.py resolve \
  --recipe test/ppu/recipes/deepseek-ai/DeepSeek-V4-Flash.yaml \
  --policy test/ppu/policy/models/deepseek-v4-flash.yaml \
  --tier fast_sanity

# CI-selector and runner selftests
python3 -B -m unittest discover -s scripts/ci/ppu -p 'test_recipe*.py'
bash scripts/ci/ppu/run_selftests.sh   # Linux with flock/git/jq/huggingface-hub
```

## PR CI (`fast_sanity`)

Every policy declaring a `fast_sanity` tier runs on each PR touching PPU scope.
All rows start directly with **full-layer dummy weights** (`--load-format dummy`);
the policy controls the radix cache. Both non-streaming and streaming probes must
complete. `env:` shows the resolved server environment.

| Policy | Recipe | Cards | Preset | Features | Effective `sglang.launch_server` flags |
| --- | --- | ---: | --- | --- | --- |
| [`deepseek-v4-flash`](policy/models/deepseek-v4-flash.yaml) | [`DeepSeek-V4-Flash.yaml`](recipes/deepseek-ai/DeepSeek-V4-Flash.yaml) | 8 | `default` | `deepep` | `--tp-size 8 --page-size 256 --mem-fraction-static 0.8 --chunked-prefill-size 10240 --cuda-graph-max-bs-decode 64 --disable-prefill-cuda-graph --reasoning-parser deepseek-v4 --disable-custom-all-reduce --disable-shared-experts-fusion --trust-remote-code --attn-dp-size 8 --enable-dp-lm-head --moe-a2a-backend deepep --deepep-mode auto --moe-dense-tp-size 1 --dist-timeout 3600 --watchdog-timeout 3600 --enable-metrics --load-format dummy`<br>env: `DG_USE_MOE_DYNAMIC_TILE=1`, `HGGC_EMBEDDED_COPY_THRESHOLD=0x10000`, `NCCL_CUMEM_ENABLE=1`, `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=512`, `SGLANG_SAIL_DEEPEP_RECV_HOOK=0` |
| [`deepseek-v4-flash-tp8`](policy/models/deepseek-v4-flash-tp8.yaml) | [`DeepSeek-V4-Flash.yaml`](recipes/deepseek-ai/DeepSeek-V4-Flash.yaml) | 8 | `tp8` | `none` | `--tp-size 8 --page-size 256 --mem-fraction-static 0.8 --chunked-prefill-size 10240 --cuda-graph-max-bs-decode 64 --disable-prefill-cuda-graph --reasoning-parser deepseek-v4 --disable-custom-all-reduce --disable-shared-experts-fusion --trust-remote-code --enable-metrics --load-format dummy`<br>env: `DG_USE_MOE_DYNAMIC_TILE=1`, `HGGC_EMBEDDED_COPY_THRESHOLD=0x10000`, `SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=0` |
| [`deepseek-v4-pro`](policy/models/deepseek-v4-pro.yaml) | [`DeepSeek-V4-Pro.yaml`](recipes/deepseek-ai/DeepSeek-V4-Pro.yaml) | 8 | `default` | `none` | `--tp-size 8 --attention-backend dsv4 --page-size 256 --context-length 16384 --mem-fraction-static 0.9 --chunked-prefill-size 8192 --cuda-graph-max-bs-decode 16 --disable-prefill-cuda-graph --kv-cache-dtype fp8_e4m3 --reasoning-parser deepseek-v4 --disable-custom-all-reduce --disable-shared-experts-fusion --trust-remote-code --enable-metrics --load-format dummy`<br>env: `DG_USE_MOE_DYNAMIC_TILE=1`, `HGGC_EMBEDDED_COPY_THRESHOLD=0x10000` |
| [`deepseek-v4.1-flash`](policy/models/deepseek-v4.1-flash.yaml) | [`DeepSeek-V4.1-Flash.yaml`](recipes/deepseek-ai/DeepSeek-V4.1-Flash.yaml) | 8 | `default` | `none` | `--tp-size 8 --attention-backend dsv4 --page-size 256 --mem-fraction-static 0.8 --chunked-prefill-size 10240 --cuda-graph-max-bs-decode 64 --disable-prefill-cuda-graph --reasoning-parser deepseek-v4 --disable-custom-all-reduce --disable-shared-experts-fusion --trust-remote-code --enable-metrics --load-format dummy`<br>env: `DG_USE_MOE_DYNAMIC_TILE=1`, `HGGC_EMBEDDED_COPY_THRESHOLD=0x10000` |
| [`glm-5.3-flash`](policy/models/glm-5.3-flash.yaml) | [`GLM-5.3-Flash.yaml`](recipes/zai-org/GLM-5.3-Flash.yaml) | 2 | `default` | `none` | `--tp-size 2 --context-length 16384 --page-size 64 --dsa-prefill-backend flashmla_sparse --dsa-decode-backend flashmla_kv --kv-cache-dtype bfloat16 --disable-prefill-cuda-graph --disable-custom-all-reduce --trust-remote-code --enable-metrics --load-format dummy` |
| [`glm-5.3-tp4`](policy/models/glm-5.3-tp4.yaml) | [`GLM-5.3-TP4.yaml`](recipes/zai-org/GLM-5.3-TP4.yaml) | 4 | `default` | `none` | `--tp-size 4 --context-length 16384 --page-size 64 --mem-fraction-static 0.85 --chunked-prefill-size 8192 --cuda-graph-max-bs-decode 64 --disable-custom-all-reduce --trust-remote-code --enable-metrics --load-format dummy`<br>env: `DG_USE_MOE_DYNAMIC_TILE=1`, `HGGC_EMBEDDED_COPY_THRESHOLD=0x10000`, `SGLANG_NSA_DUAL_STREAM=1` |
| [`glm-5.3`](policy/models/glm-5.3.yaml) | [`GLM-5.3.yaml`](recipes/zai-org/GLM-5.3.yaml) | 8 | `default` | `none` | `--tp-size 8 --page-size 64 --mem-fraction-static 0.85 --chunked-prefill-size 8192 --cuda-graph-max-bs-decode 64 --disable-custom-all-reduce --trust-remote-code --enable-metrics --load-format dummy`<br>env: `DG_USE_MOE_DYNAMIC_TILE=1`, `HGGC_EMBEDDED_COPY_THRESHOLD=0x10000`, `SGLANG_NSA_DUAL_STREAM=1` |
| [`kimi-k2.6`](policy/models/kimi-k2.6.yaml) | [`Kimi-K2.6.yaml`](recipes/moonshotai/Kimi-K2.6.yaml) | 8 | `default` | `none` | `--tp-size 8 --context-length 16384 --mem-fraction-static 0.9 --prefill-attention-backend fa3 --decode-attention-backend flashmla --cuda-graph-max-bs-decode 16 --disable-prefill-cuda-graph --disable-custom-all-reduce --trust-remote-code --enable-metrics --load-format dummy`<br>env: `DG_USE_MOE_DYNAMIC_TILE=1`, `HGGC_EMBEDDED_COPY_THRESHOLD=0x10000` |
| [`minimax-m3-fp8`](policy/models/minimax-m3-fp8.yaml) | [`MiniMax-M3-FP8.yaml`](recipes/MiniMaxAI/MiniMax-M3-FP8.yaml) | 8 | `default` | `none` | `--tp-size 8 --trust-remote-code --mem-fraction-static 0.9 --attention-backend fa3 --page-size 128 --kv-cache-dtype fp8_e4m3 --moe-runner-backend deep_gemm --enable-metrics --load-format dummy` |
| [`minimax-m3`](policy/models/minimax-m3.yaml) | [`MiniMax-M3.yaml`](recipes/MiniMaxAI/MiniMax-M3.yaml) | 8 | `default` | `none` | `--tp-size 8 --trust-remote-code --mem-fraction-static 0.9 --attention-backend fa3 --page-size 128 --kv-cache-dtype fp8_e4m3 --moe-runner-backend deep_gemm --enable-metrics --load-format dummy` |
| [`qwen3.8-27b`](policy/models/qwen3.8-27b.yaml) | [`Qwen3.8-27B.yaml`](recipes/Qwen/Qwen3.8-27B.yaml) | 2 | `default` | `none` | `--tp-size 2 --dtype bfloat16 --context-length 16384 --disable-custom-all-reduce --trust-remote-code --enable-metrics --load-format dummy` |
| [`qwen3.8-flash-next-tp4`](policy/models/qwen3.8-flash-next-tp4.yaml) | [`Qwen3.8-Flash-Next-TP4.yaml`](recipes/Qwen/Qwen3.8-Flash-Next-TP4.yaml) | 4 | `default` | `spec` (NEXTN) | `--tp-size 4 --context-length 16384 --attention-backend triton --mem-fraction-static 0.9 --max-running-requests 128 --chunked-prefill-size 8192 --speculative-algorithm NEXTN --speculative-num-steps 2 --speculative-eagle-topk 1 --linear-attn-prefill-backend triton --linear-attn-decode-backend triton --linear-attn-verify-backend triton --mamba-ssm-dtype bfloat16 --ple-offload-embedding --reasoning-parser auto --trust-remote-code --disable-radix-cache --enable-metrics --load-format dummy` |
| [`qwen3.8-flash-next`](policy/models/qwen3.8-flash-next.yaml) | [`Qwen3.8-Flash-Next.yaml`](recipes/Qwen/Qwen3.8-Flash-Next.yaml) | 8 | `default` | `deepep`, `spec` (NEXTN) | `--tp-size 8 --attn-dp-size 8 --enable-dp-lm-head --moe-a2a-backend deepep --moe-dense-tp-size 1 --attention-backend fa3 --mem-fraction-static 0.9 --max-running-requests 128 --chunked-prefill-size 8192 --speculative-algorithm NEXTN --speculative-num-steps 2 --speculative-eagle-topk 1 --speculative-num-draft-tokens 3 --linear-attn-prefill-backend triton --linear-attn-decode-backend triton --linear-attn-verify-backend triton --mamba-ssm-dtype bfloat16 --ple-offload-embedding --reasoning-parser auto --dist-timeout 3600 --watchdog-timeout 3600 --trust-remote-code --disable-radix-cache --enable-metrics --load-format dummy`<br>env: `NCCL_CUMEM_ENABLE=1`, `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=512`, `SGLANG_SAIL_DEEPEP_RECV_HOOK=0` |

## Nightly CI (`model_sanity`)

Policies declaring a `model_sanity` tier run once per model against **real
weights** on a single shared server process. Every model runs the same three
stages:

1. `sanity` — a deterministic probe prompt (`6 * 7`, expected substring `42`).
2. `eval-gsm8k` — EvalScope on [`gsm8k@2026-11`](policy/datasets/gsm8k.yaml)
   (main split, 4-shot, all 1,319 questions, `max_tokens=10240`, temperature 1.0),
   reconciled per scored sample against the golden band.
3. `bench-micro` — the native SGLang microbench at concurrency `1` and `16`
   (`cache_mode: sanity_throughput_guard`). These same-server numbers are
   diagnostic, not an acceptance gate.

The accuracy gate is `golden_band` with `golden_band_abs: 0.02` against the
matching [`golden/`](golden) reference (`golden_source: framework_repo`). Flags
below omit `--load-format dummy`; `env:` is identical to the `fast_sanity` row
unless noted.

| Policy | Recipe | Cards | Preset | Golden key | Effective `sglang.launch_server` flags |
| --- | --- | ---: | --- | --- | --- |
| [`deepseek-v4-flash`](policy/models/deepseek-v4-flash.yaml) | [`DeepSeek-V4-Flash.yaml`](recipes/deepseek-ai/DeepSeek-V4-Flash.yaml) | 8 | `default` | [`deepseek_v4_mxfp4`](golden/accuracy_deepseek_v4_mxfp4_890p.json) | `--tp-size 8 --page-size 256 --mem-fraction-static 0.8 --chunked-prefill-size 10240 --cuda-graph-max-bs-decode 64 --disable-prefill-cuda-graph --reasoning-parser deepseek-v4 --disable-custom-all-reduce --disable-shared-experts-fusion --trust-remote-code --attn-dp-size 8 --enable-dp-lm-head --moe-a2a-backend deepep --deepep-mode auto --moe-dense-tp-size 1 --dist-timeout 3600 --watchdog-timeout 3600 --enable-metrics` |
| [`deepseek-v4-flash-tp8`](policy/models/deepseek-v4-flash-tp8.yaml) | [`DeepSeek-V4-Flash.yaml`](recipes/deepseek-ai/DeepSeek-V4-Flash.yaml) | 8 | `tp8` | [`deepseek_v4_mxfp4`](golden/accuracy_deepseek_v4_mxfp4_890p.json) | `--tp-size 8 --page-size 256 --mem-fraction-static 0.8 --chunked-prefill-size 10240 --cuda-graph-max-bs-decode 64 --disable-prefill-cuda-graph --reasoning-parser deepseek-v4 --disable-custom-all-reduce --disable-shared-experts-fusion --trust-remote-code --enable-metrics` |
| [`deepseek-v4-pro`](policy/models/deepseek-v4-pro.yaml) | [`DeepSeek-V4-Pro.yaml`](recipes/deepseek-ai/DeepSeek-V4-Pro.yaml) | 8 | `default` | [`deepseek_v4_pro_mxfp4`](golden/accuracy_deepseek_v4_pro_mxfp4_890p.json) | `--tp-size 8 --attention-backend dsv4 --page-size 256 --context-length 16384 --mem-fraction-static 0.9 --chunked-prefill-size 8192 --cuda-graph-max-bs-decode 16 --disable-prefill-cuda-graph --kv-cache-dtype fp8_e4m3 --reasoning-parser deepseek-v4 --disable-custom-all-reduce --disable-shared-experts-fusion --trust-remote-code --enable-metrics` |
| [`glm-5.3-flash`](policy/models/glm-5.3-flash.yaml) | [`GLM-5.3-Flash.yaml`](recipes/zai-org/GLM-5.3-Flash.yaml) | 2 | `default` | [`glm_5_3_flash_mxfp4`](golden/accuracy_glm_5_3_flash_mxfp4_890p.json) | `--tp-size 2 --context-length 16384 --page-size 64 --dsa-prefill-backend flashmla_sparse --dsa-decode-backend flashmla_kv --kv-cache-dtype bfloat16 --disable-prefill-cuda-graph --disable-custom-all-reduce --trust-remote-code --enable-metrics` |
| [`glm-5.3-tp4`](policy/models/glm-5.3-tp4.yaml) | [`GLM-5.3-TP4.yaml`](recipes/zai-org/GLM-5.3-TP4.yaml) | 4 | `default` | [`glm_5_3_mxfp4_tp4`](golden/accuracy_glm_5_3_mxfp4_tp4_890p.json) | `--tp-size 4 --context-length 16384 --page-size 64 --mem-fraction-static 0.85 --chunked-prefill-size 8192 --cuda-graph-max-bs-decode 64 --disable-custom-all-reduce --trust-remote-code --enable-metrics` |
| [`glm-5.3`](policy/models/glm-5.3.yaml) | [`GLM-5.3.yaml`](recipes/zai-org/GLM-5.3.yaml) | 8 | `default` | [`glm_5_3_mxfp4`](golden/accuracy_glm_5_3_mxfp4_890p.json) | `--tp-size 8 --page-size 64 --mem-fraction-static 0.85 --chunked-prefill-size 8192 --cuda-graph-max-bs-decode 64 --disable-custom-all-reduce --trust-remote-code --enable-metrics` |
| [`kimi-k2.6`](policy/models/kimi-k2.6.yaml) | [`Kimi-K2.6.yaml`](recipes/moonshotai/Kimi-K2.6.yaml) | 8 | `default` | [`kimi_k2_6_mxfp4`](golden/accuracy_kimi_k2_6_mxfp4_890p.json) | `--tp-size 8 --context-length 16384 --mem-fraction-static 0.9 --prefill-attention-backend fa3 --decode-attention-backend flashmla --cuda-graph-max-bs-decode 16 --disable-prefill-cuda-graph --disable-custom-all-reduce --trust-remote-code --enable-metrics` |
| [`minimax-m3-fp8`](policy/models/minimax-m3-fp8.yaml) | [`MiniMax-M3-FP8.yaml`](recipes/MiniMaxAI/MiniMax-M3-FP8.yaml) | 8 | `default` | [`minimax_m3_fp8`](golden/accuracy_minimax_m3_fp8_890p.json) | `--tp-size 8 --trust-remote-code --mem-fraction-static 0.9 --attention-backend fa3 --page-size 128 --kv-cache-dtype fp8_e4m3 --moe-runner-backend deep_gemm --enable-metrics` |
| [`minimax-m3`](policy/models/minimax-m3.yaml) | [`MiniMax-M3.yaml`](recipes/MiniMaxAI/MiniMax-M3.yaml) | 8 | `default` | [`minimax_m3_mxfp4_fp8`](golden/accuracy_minimax_m3_mxfp4_fp8_890p.json) | `--tp-size 8 --trust-remote-code --mem-fraction-static 0.9 --attention-backend fa3 --page-size 128 --kv-cache-dtype fp8_e4m3 --moe-runner-backend deep_gemm --enable-metrics` |
| [`qwen3.8-flash-next-tp4`](policy/models/qwen3.8-flash-next-tp4.yaml) | [`Qwen3.8-Flash-Next-TP4.yaml`](recipes/Qwen/Qwen3.8-Flash-Next-TP4.yaml) | 4 | `model_sanity` | [`qwen3.8_flash_next_mxfp4_tp4`](golden/accuracy_qwen3.8_flash_next_mxfp4_tp4_890p.json) | `--tp-size 4 --context-length 16384 --attention-backend triton --mem-fraction-static 0.9 --max-running-requests 128 --chunked-prefill-size 8192 --linear-attn-prefill-backend triton --linear-attn-decode-backend triton --mamba-ssm-dtype bfloat16 --ple-offload-embedding --reasoning-parser auto --trust-remote-code --page-size 64 --mamba-radix-cache-strategy extra_buffer --disable-overlap-schedule --enable-metrics` |
| [`qwen3.8-flash-next`](policy/models/qwen3.8-flash-next.yaml) | [`Qwen3.8-Flash-Next.yaml`](recipes/Qwen/Qwen3.8-Flash-Next.yaml) | 8 | `model_sanity` | [`qwen3.8-flash-next-dp8-ep`](golden/accuracy_qwen3.8-flash-next-dp8-ep_890p.json) | `--tp-size 8 --attn-dp-size 8 --enable-dp-lm-head --moe-a2a-backend deepep --moe-dense-tp-size 1 --attention-backend fa3 --mem-fraction-static 0.9 --max-running-requests 128 --chunked-prefill-size 8192 --speculative-algorithm NEXTN --speculative-num-steps 2 --speculative-eagle-topk 1 --speculative-num-draft-tokens 3 --linear-attn-prefill-backend triton --linear-attn-decode-backend triton --linear-attn-verify-backend triton --mamba-ssm-dtype bfloat16 --ple-offload-embedding --reasoning-parser auto --dist-timeout 3600 --watchdog-timeout 3600 --trust-remote-code --page-size 64 --mamba-radix-cache-strategy extra_buffer --speculative-draft-model-quantization unquant --enable-metrics` |

> `deepseek-v4.1-flash` and `qwen3.8-27b` declare only a `fast_sanity` tier, so
> they do not appear in the nightly matrix and have no golden reference.

## Device test suites (`e2e`, `unit`, `module`)

These suites run on a real PPU device inside a verified task container and are
independent of the recipe/policy resolver. `e2e` tests are the CI *legacy
single-device sanity* set (dispatched with `run_legacy_sanity=true` on boards
`ZW-M890P` and `OAM-810E`); `unit` and `module` are developer-facing regression
suites. Most read the checkpoint from `MODEL_PATH` (and optionally
`GSM8K_DATA_PATH`) and skip cleanly when no device is present.

```bash
# Kernel / contract unit tests and module integration tests
python3 -m pytest test/ppu/unit -v
python3 -m pytest test/ppu/module -v

# A single device end-to-end server test
MODEL_PATH=/path/to/checkpoint \
  python3 -m pytest test/ppu/e2e/test_glm52_ppu_flashmla_fa3_mtp.py -v -s

# Root-level KDA / PLA replay-SSM regressions
python3 -m pytest test/ppu/test_kda_pla_replayssm.py test/ppu/test_kda_replayssm_pool.py -v
```

## Adding or changing a model

1. Add or edit the serving recipe under [`recipes/`](recipes) (`ppu-serving/v1`).
2. Add or edit the policy under [`policy/models/`](policy/models)
   (`ppu-test-policy/v1`), pointing `recipe:` at the file from step 1 and
   declaring the `fast_sanity` / `model_sanity` tiers.
3. For a `model_sanity` policy, add the matching
   [`golden/`](golden)`accuracy_<golden_model_key>_890p.json` reference.
4. Run `ppu_recipe.py validate --all` and `ppu_recipe.py resolve ...` to confirm
   the effective flags, then update the matrix tables above.
