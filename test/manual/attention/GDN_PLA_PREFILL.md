# SAIL PLA GDN prefill

On PPU, select PLA prefill and the existing PLA decode paths with one switch:

```bash
export SGLANG_SAIL_PLA_CUDA=1
sglang serve --model-path <original-model-path> \
  --linear-attn-prefill-backend triton <original-options>
```

`SGLANG_SAIL_PLA_CUDA=1` enables eligible PLA GDN prefill and the existing
PLA decode dispatch. Set it to `0` to use the original Triton paths. PPU
platform initialization already enables this shared switch when unset, so
eligible GDN prefill now uses PLA by default on PPU. The deprecated
`SGLANG_SAIL_FLA_CUDA` alias retains its existing behavior.

The QSA switch `SGLANG_SAIL_QSA_FA3_SPARSE_ATTN` remains independent. Explicit
non-Triton prefill backends keep their dispatch. Existing decode/verify dtype
and ReplaySSM guards still apply; enabling the master switch does not remove
those compatibility fallbacks. Restart the server after changing the switch
so captured graphs and process-local choices are rebuilt.

Requires a PPU PLA build with
`pla.prefill.flashqla.chunk_gated_delta_rule_fwd`, indexed state updates,
`normalize_qk`, `raw_cu_seqlens` and `final_state_dtype`. The adapter lazily imports
this optional extension. It preserves model precision and BF16/FP32 recurrent
state precision, passes activated gates/beta, and normalizes Q/K in PLA. Indexed
states use `[slot,H,V,K]`, not the legacy non-indexed PLA `[batch,H,K,V]` layout.
Sequence offsets and slots are contiguous INT32. Strided state pools use
indexed gather/scatter; untouched slots remain unchanged.

PLA updates final states in place; its final `ht` is not Triton's per-chunk `h`.
Batches requiring intermediate prefix-cache/checkpoint states, read-only scoring,
or unavailable/stale CPU sequence lengths retain Triton. Direct CUDA Graph
capture retains Triton and its existing fixed chunk-plan restrictions rather
than freezing PLA's host plan;
breakable prefill graphs run GDN through their existing eager break and can use
PLA there. Final-state-only prefix tracking can read PLA's updated pool.
Unsupported PLA shapes/dtypes and library errors are reported by PLA, not hidden.

Run inside a configured task container:

```bash
PYTHONPATH=python python3 test/registered/unit/layers/attention/test_gdn_pla_prefill.py -v
PYTHONPATH=python python3 test/registered/kernels/ops/attention/test_gdn_pla_prefill.py -v
```

The CPU adapter contract suite is registered for CPU CI; the numerical and Graph
suite is registered for the PLA-equipped PPU stage-b suite. Numerical comparisons
use `atol=2e-2, rtol=1e-2`; fallback comparisons additionally check exact state
semantics. These tests do not replace original-model startup and output checks.

Porting reference: https://github.com/flytiger-eco/sglang-for-sail/pull/152
The v0.5.18 implementation also used the shared master switch for prefill.
This port preserves the newer read-only scoring and graph metadata contracts.
No extra GDN-prefill selector or temporary route instrumentation is required.
