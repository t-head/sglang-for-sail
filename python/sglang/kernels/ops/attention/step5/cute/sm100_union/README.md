# Step5 SM100 union prefill

This private implementation derives from StepFun commit b3055a1c0e5a17da35bbf36d8f60651e073706c3. See [MANIFEST.json](MANIFEST.json) for the original source hashes and local modifications. Source files retain their copyright and license headers.

The implementation builds local FakeTensor descriptors and uses recursive source fingerprints for the SGLang compilation cache. It owns its dependencies and schedule buffers separately from the public SGLang FA4 implementation.

The schedule reads compact query and output rows. Union metadata supplies the query group count and each group's compact-row offset. `WorkQGlobal` stores per-query positions. `WorkQLocal` and `WorkQLen` locate the query rows within each group. The schedule computes causal limits relative to each union and masks invalid tail loads.

`step5_sparse_prefill` enables this path by default on devices with compute capability 10.x. Set `SPARSE_PREFILL_SM100` in `step5/options.py` to disable it. The backend allocates typed schedule buffers before the decoder layers and sends compact rows directly to the kernel.
