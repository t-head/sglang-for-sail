# SGLang PPU Recipe CI

The release branch owns the SGLang serving recipes, test policies and 890P accuracy references in `test/ppu`. Native SAIL wheels and their Recipe bundle carry the same source revision. The runner verifies the wheel inventory, archive and recipe tree before installation and records the release runtime's manifest digest.

- **Fast Sanity** runs every model with a `fast_sanity` policy. The current models start directly with full-layer dummy weights. The model policy controls prefix caching. Both non-streaming and streaming probes must complete.
- **Model Sanity** runs real weights once per model. It runs GSM8K main (1,319 questions), reconciles each scored sample with the reference band, verifies the KV gauge after a cache flush, and runs the native microbench at concurrency 1 and 16. The same-server performance measurements are diagnostic.
- **Legacy single-device sanity** runs EAGLE3 and DFlash on both board types when explicitly requested. Each test reports its original result and the runner retains the raw output privately.

For release PRs, a maintainer with write access must apply `run-ci` and `run-ci-ppu` for Fast Sanity or `run-ci-ppu-nightly` for Model Sanity. The labels may arrive in either order. A rerun rechecks the PR timeline. After a new push, remove and reapply the workload label to approve the current head. Apply the same labels to fork PRs after reviewing the exact revision. Nightly dispatch and reusable inputs also allow an exact model filter and reuse of a candidate build with `skip_wheel_build` and `wheel_artifact_run_id`.

The PR workflow finish job requires Linux infrastructure selftests, the native build, runner preflight and Recipe Fast Sanity for changes in PPU scope. PR-triggered workflows use the protected base branch's workflow and the approved head SHA as the test source. To run the separate single-device workflow on the release branch after it lands, dispatch the same workflow in legacy-only mode.

```bash
gh workflow run pr-test-ppu.yml --repo t-head/sglang-for-sail \
  --ref release/v0.5.23 -f run_legacy_sanity=true
```

The reusable workflow accepts the same boolean input and a candidate `ref`.

## Provision the cluster runner

Cluster jobs use the T-Head CPU orchestration runner label `k8s-runner-group-cpu-thead`. PR jobs require revision-specific approval before reaching the cluster runner.

For pre-merge validation, dispatch `pr-test-ppu.yml` on a same-repository maintainer branch with `preflight_only=true`. This runs the Linux selftests and the private configuration checks via the direct and reusable workflows. A cross-branch caller can request `tier: cluster_preflight` from the same reusable workflow.

The runner uses the pinned T-Head Cluster Run backend. Each Recipe job requests one 890P Pod with the width resolved from its selected policy and Recipe (2, 4, or 8 cards). The worker verifies that its visible card count and resolved Recipe match this allocation. Each existing single-device test job requests one card on its selected board. The CPU runner passes model and dataset paths privately to the worker from its service environment. A task-specific shared configuration can supply the same variables. The workflow pins the configuration digest and verifies access in a CPU-only job before scheduling PPU Pods. Provision the task configuration and output directory for the runner account before starting jobs. Keep the configuration files at mode 0600 and the private directories at mode 0700.

| Setting | Purpose |
| --- | --- |
| `PPU_CI_HF_CACHE_ROOT` | Shared offline Hugging Face cache root containing `hub/` |
| `PPU_CI_GLM53_MODEL_REVISION` | Pinned GLM-5.3 MXFP4 cache snapshot |
| `PPU_CI_DSV4_FLASH_MODEL_REVISION` | Pinned DeepSeek-V4-Flash MXFP4 cache snapshot |
| `PPU_CI_GSM8K_PATH` | Offline EvalScope GSM8K dataset directory |
| `PPU_CI_JIT_ROOT` | Node-local persistent JIT cache root |
| `PPU_CI_K8S_STAGE_ROOT` | Cluster Run's shared source staging directory in the CPU runner view |
| `PPU_CI_PRIVATE_RESULTS_ROOT` | Private results archive root in the CPU runner view; Recipe runs are archived under `sglang/logs/sglang-for-sail/<run_id>-<attempt>/<tier>/<model_key>` |
| `PPU_CI_K8S_PRIVATE_WORKER_ROOT` | Worker view of the task-scoped private scratch directory (`PPU_CI_K8S_HOST_SCRATCH_ROOT`) |
| `PPU_CI_K8S_RUNTIME_DIGEST` | Optional manifest reference for the designated release runtime tag |
| `PPU_CI_RUNTIME_IMAGE` | Existing single-device test image |
| `PPU_CI_TARGET_MODEL_PATH`, `PPU_CI_EAGLE3_MODEL_PATH`, `PPU_CI_DFLASH_MODEL_PATH`, `PPU_CI_EVAL_CACHE` | Existing single-device assets in the worker view |

The worker keeps `HF_HOME` job-local and points `HF_HUB_CACHE` and `HUGGINGFACE_HUB_CACHE` at the shared cache. The Recipe's `hf_model_id` resolves the checkpoint offline before server launch. Models without an explicit runner pin use the cache's local `refs/main`; the worker resolves it once to a concrete snapshot and includes that revision in the configuration digest. The Pod must see the cache and the targets of its snapshot links. Besides standard `tokenizer.json` snapshots, `kimi_k25` checkpoints can use their complete TikToken files.

Use a runner configured for reviewed public PR revisions with public-safe runner labels, name and workspace metadata. Kubernetes device requests arbitrate card ownership. The CPU runner and worker must share access to the private per-job directory with mode 0700. The launcher captures Cluster Run diagnostics before exposing scheduler and worker results to GitHub. The evidence gate checks the runtime digest reported by Kubernetes and publishes a typed result. JIT caches for a pinned image can be reused across runs.

For local structure checks, install PyYAML and use these commands. On Linux with `flock`, `git`, `jq` and `huggingface-hub`, also run `bash scripts/ci/ppu/run_selftests.sh`.

```bash
python3 -B scripts/ci/ppu/ppu_recipe.py validate --all --sglang-parser skip
python3 -B -m unittest discover -s scripts/ci/ppu -p 'test_recipe*.py'
```

The release runtime validates these flags again using the installed candidate's `ServerArgs` parser before serving.
