# SGLang PPU Recipe CI

The release branch owns the SGLang serving recipes, test policies and 890P accuracy references in `test/ppu`. Native SAIL wheels and their Recipe bundle carry the same source revision. The runner verifies the wheel inventory, archive and recipe tree before installation and records the release runtime's manifest digest.

- **Fast Sanity** runs every model with a `fast_sanity` policy. It uses dummy weights, attempts a two-layer override first and can retry with the full layer count when startup fails. Both non-streaming and streaming probes must complete.
- **Model Sanity** runs real weights once per model. It runs GSM8K main (1,319 questions), reconciles each scored sample with the reference band, verifies the KV gauge after a cache flush, and runs the native microbench at concurrency 1 and 16. The same-server performance measurements are diagnostic.
- The existing single-device EAGLE3 and DFlash tests run separately. They retain their private outputs under the runner's results root.

For release PRs, use `run-ci` with `run-ci-ppu` for Fast Sanity. Apply `run-ci-ppu-nightly` to an eligible ready PR to trigger Model Sanity. Nightly dispatch and reusable inputs also allow an exact model filter and reuse of a candidate build with `skip_wheel_build` and `wheel_artifact_run_id`.

## Provision the runner

The runner service provides these values privately. The run binds the checkpoints and EvalScope dataset read-only inside the container.

| Setting | Purpose |
| --- | --- |
| `PPU_CI_GLM53_MODEL_PATH` | GLM-5.3 MXFP4 checkpoint directory |
| `PPU_CI_DSV4_FLASH_MODEL_PATH` | DeepSeek-V4-Flash MXFP4 checkpoint directory |
| `PPU_CI_GSM8K_PATH` | Offline EvalScope GSM8K dataset directory |
| `PPU_CI_JIT_ROOT` | Node-local persistent JIT cache root |
| `PPU_CI_PRIVATE_RESULTS_ROOT` | Private per-run evidence and checksum manifests |
| `PPU_CI_HOST_PYTHON` | Python 3.10 or newer when the default `python3` is older |

Use a runner configured for reviewed public PR revisions with public-safe runner labels, name and workspace metadata. The host locks all eight devices using the same lock names as the legacy lane. A run publishes a typed result after validation and retains diagnostic output on the runner. The runtime uses no external network.

For local structure checks, install PyYAML and use these commands.

```bash
python3 -B scripts/ci/ppu/ppu_recipe.py validate --all --sglang-parser skip
python3 -B -m unittest discover -s scripts/ci/ppu -p 'test_recipe*.py'
```

The release runtime validates these flags again using the installed candidate's `ServerArgs` parser before serving.
