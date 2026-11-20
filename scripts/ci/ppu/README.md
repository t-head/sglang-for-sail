# SGLang PPU Recipe CI

The release branch owns the SGLang serving recipes, test policies and 890P accuracy references in `test/ppu`. Native SAIL wheels and their Recipe bundle carry the same source revision. The runner verifies the wheel inventory, archive and recipe tree before installation and records the release runtime's manifest digest.

- **Fast Sanity** runs every model with a `fast_sanity` policy. The current models start directly with full-layer dummy weights. Both non-streaming and streaming probes must complete.
- **Model Sanity** runs real weights once per model. It runs GSM8K main (1,319 questions), reconciles each scored sample with the reference band, verifies the KV gauge after a cache flush, and runs the native microbench at concurrency 1 and 16. The same-server performance measurements are diagnostic.
- **Legacy single-device sanity** runs EAGLE3 and DFlash on both board types when explicitly requested. Each test reports its original result and the runner retains the raw output privately.

For release PRs, a maintainer with write access must first apply `run-ci`, then apply `run-ci-ppu` for Fast Sanity or `run-ci-ppu-nightly` for Model Sanity. Reapply the workload label after a new push to approve the new head SHA. Apply the same labels to fork PRs after reviewing the exact revision. Nightly dispatch and reusable inputs also allow an exact model filter and reuse of a candidate build with `skip_wheel_build` and `wheel_artifact_run_id`.

The PR workflow finish job requires Linux infrastructure selftests, the native build, runner preflight and Recipe Fast Sanity for changes in PPU scope. PR-triggered workflows use the protected base branch's workflow and the approved head SHA as the test source. To run the separate single-device workflow on the release branch after it lands, dispatch the same workflow in legacy-only mode.

```bash
gh workflow run pr-test-ppu.yml --repo t-head/sglang-for-sail \
  --ref release/v0.5.23 -f run_legacy_sanity=true
```

The reusable workflow accepts the same boolean input and a candidate `ref`.

## Provision the cluster runner

Register the `k8s-runner-group-cpu-thead` runner in the `sglang-ppu-ci` runner group before enabling the workflows. Restrict the group to this repository and these selected workflows at the protected release ref:

- `t-head/sglang-for-sail/.github/workflows/pr-test-ppu.yml@refs/heads/release/v0.5.23`
- `t-head/sglang-for-sail/.github/workflows/nightly-test-ppu.yml@refs/heads/release/v0.5.23`

Remove the runner's access through the default or other unrestricted groups. Add protected branch refs to the allowlist when deploying these workflows on those branches. The branch must already contain the trusted workflow to run its PR events.

The runner uses the pinned T-Head Cluster Run backend. The scheduler assigns one eight-card 890P Pod for each Recipe job. Each existing single-device test job requests one card on its selected board. The CPU runner passes model and dataset paths privately to the worker from its service environment. A task-specific shared configuration can supply the same variables. The workflow pins the configuration digest and verifies access in a CPU-only job before scheduling PPU Pods. Provision the task configuration and output directory for the runner account before starting jobs. Keep the configuration files at mode 0600 and the private directories at mode 0700.

| Setting | Purpose |
| --- | --- |
| `PPU_CI_HF_CACHE_ROOT` | Shared offline Hugging Face cache root containing `hub/` |
| `PPU_CI_GLM53_MODEL_REVISION` | Pinned GLM-5.3 MXFP4 cache snapshot |
| `PPU_CI_DSV4_FLASH_MODEL_REVISION` | Pinned DeepSeek-V4-Flash MXFP4 cache snapshot |
| `PPU_CI_GSM8K_PATH` | Offline EvalScope GSM8K dataset directory |
| `PPU_CI_JIT_ROOT` | Node-local persistent JIT cache root |
| `PPU_CI_K8S_STAGE_ROOT` | Cluster Run's shared source staging directory in the CPU runner view |
| `PPU_CI_PRIVATE_RESULTS_ROOT` | Private shared results directory in the CPU runner view |
| `PPU_CI_K8S_PRIVATE_WORKER_ROOT` | The same results directory in the worker view |
| `PPU_CI_K8S_RUNTIME_DIGEST` | Optional manifest reference for the designated release runtime tag |
| `PPU_CI_RUNTIME_IMAGE` | Existing single-device test image |
| `PPU_CI_TARGET_MODEL_PATH`, `PPU_CI_EAGLE3_MODEL_PATH`, `PPU_CI_DFLASH_MODEL_PATH`, `PPU_CI_EVAL_CACHE` | Existing single-device assets in the worker view |

The worker keeps `HF_HOME` job-local and points `HF_HUB_CACHE` and `HUGGINGFACE_HUB_CACHE` at the shared cache. The Recipe's `hf_model_id` and pinned revision resolve the checkpoint offline before server launch. The Pod must see the cache and the targets of its snapshot links.

Use a runner configured for reviewed public PR revisions with public-safe runner labels, name and workspace metadata. Kubernetes device requests arbitrate card ownership. The CPU runner and worker must share access to the private per-job directory with mode 0700. The launcher captures Cluster Run diagnostics before exposing scheduler and worker results to GitHub. The evidence gate checks the runtime digest reported by Kubernetes and publishes a typed result. JIT caches for a pinned image can be reused across runs.

For local structure checks, install PyYAML and use these commands. On Linux with `flock`, `git`, `jq` and `huggingface-hub`, also run `bash scripts/ci/ppu/run_selftests.sh`.

```bash
python3 -B scripts/ci/ppu/ppu_recipe.py validate --all --sglang-parser skip
python3 -B -m unittest discover -s scripts/ci/ppu -p 'test_recipe*.py'
```

The release runtime validates these flags again using the installed candidate's `ServerArgs` parser before serving.
