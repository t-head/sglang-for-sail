# PD transfer buffer lifetime regression

## Change and deployment

This fix starts from `online_base_0914` at
`6e2290ec0fcfe8254f80e181573e33e8bb0d47b5`.

A request becoming Failed does not prove that its transport operations have
finished. Previously an aborted Decode request could return KV, recurrent-state
and metadata slots to their allocators while a Prefill worker still held their
addresses. A later request could reuse those slots before the old write arrived.
Prefill source buffers had the corresponding early-release risk.

The Mooncake path now requires the following lifetime fences:

- Acquire a source lease before a chunk becomes visible to a transfer queue.
  Keep the same lease across staging retries. Cancellation atomically closes
  admission, including when it precedes sender construction.
- Join all submitted transfer futures, including when another future fails.
  A drained ACK requires closed admission and zero remaining leases.
- Register Decode ACK accounting before sending ABORT. Each receiver has a
  nonce, and each expected Prefill rank must acknowledge that nonce. Duplicate
  cancellations preserve accumulated ACKs and retry lost confirmations.
- Preserve Prefill source buffers until TP/CP agree that source reads have
  drained. PP release-ID consensus uses the same gate. Business failure remains
  visible immediately so peer ranks can cancel their own senders.
- Hold Decode destinations through remote drain and local HiCache/staging copy
  completion. Timeout only warns and retries; it never authorizes reuse.
  Quarantine also blocks memory offload and unified-memory compaction.
- Serialize late staging allocation with room teardown. Preserve a failed
  cleanup as a quarantine hold; do not automatically retry a partially completed
  free.

**Update both Prefill and Decode.** A new Decode rejects legacy ACKs without the
nonce, so combining it with an old Prefill leaves cancelled buffers quarantined.
Update Prefill first if a rolling update is required; the full guarantee applies
only once both sides run the fix. Mooncake safety is mandatory even when
`SGLANG_DISAGGREGATION_DEFERRED_DECODE_KV_RELEASE=0`.
`SGLANG_DISAGGREGATION_DEFERRED_DECODE_KV_RELEASE_TIMEOUT` is a warning/retry
interval, not a reclamation timeout.

There is an explicit availability tradeoff: a native transport error does not
provide a proven cancellation/drain fence. Its source lease and destination
buffers remain quarantined. Missing peers/ACKs and failed local cleanup can
also retain capacity until controlled worker recovery. Do not work around this
by force-freeing those slots. An ABORT received before sender creation retains
a small tombstone if that sender never arrives.

## Real transport probe

On a machine with two available GPUs, installed Mooncake, and the fixed checkout
on `PYTHONPATH`:

```bash
python test/manual/disaggregation/test_mooncake_abort_lifetime.py \
  --protocol rdma --source-gpu 0 --destination-gpu 1 --timeout 120
```

Containers need GPU and RDMA device access, network connectivity, and a sufficient
locked-memory limit, for example `--ulimit memlock=-1:-1`. An insufficient limit
can fail RDMA QP creation before a successful transfer occurs.

The probe creates independent P/D processes and registered GPU buffers. It
pauses at two transport-call boundaries: before the real native write, and after
its successful return but before production chunk completion. At both points
the destination must remain unavailable for reuse. Production completion must
then produce the matching real ZMQ ACK, after which the destination can be
reused as B and retain B's marker. `--protocol tcp` explicitly selects CPU
buffers for an alternative transport check; there is no automatic fallback.

This is a model-free integration test of the production lifetime and control
protocol. It does not run the full scheduler/cache allocator, prove native RDMA
cancellation after a timeout, reproduce online incidence, or identify the cause
of a benchmark quality regression. Same-node RDMA also does not establish which
physical link the transfer engine chooses internally.

## Review follow-up: Decode TP consensus and staging rejection

Decode reclamation now uses the Attention TP CPU group. Every rank participates,
including ranks with no local holds. The group checks the same ordered request
identities, then takes the group-wide minimum of remote-drain and local
HiCache/staging-copy readiness. Only that common set is freed, in the same order.
A final collective reports cleanup failures and prevents admission before every
peer has finished freeing its buffers. Queue-identity mismatches, readiness
errors and partial cleanup failures stop the whole group; they are sticky and
require controlled worker recovery, rather than a retry of partially freed
resources.

HiCache restore failures and mismatched metadata room IDs become Failed before
the existing TP poll reduction, in both ordinary and staging paths. This also
prevents one rank from taking a local failure path while another admits the
request.

Staging uses a separate `StagingTransferRejected` exception for local capacity
or buffer rejection proven to precede any gather/native I/O. These rejected
chunks retire their lease normally. Native transport errors and unknown
exceptions still quarantine their leases; local rejection does not establish
that other chunks have drained, so an ACK still requires zero remaining leases.

The regression includes real two-process Gloo collectives for staggered ACKs,
local DMA readiness, ordering, empty/mismatched hold lists, readiness/cleanup
errors, and asymmetric HiCache/metadata failures. Its allocator/cleanup effects
are controlled test doubles; it is not a full GPU scheduler test. Staging tests
check zero gather/native calls for local rejection and retention after a native
error.

For Attention TP > 1, the safety check adds a small CPU collective per resolve
call, plus identity/readiness/outcome collectives when needed. No ITL or
throughput measurement has been performed for this follow-up.

## Validation recorded on 2026-09-20

- Final review follow-up, selected 22-file regression set: **214 passed,
  146 subtests passed**, with the same five known baseline failures deselected.
  This run includes the real two-rank Gloo test and the staging pre-I/O/native
  error distinction. The final log is available locally at
  `/private/tmp/pd-lifetime-validation-20260920/pytest-review-final.log`.
- Initial validation, before the review follow-up: **201 passed, 116 subtests passed**,
  including queued/running cancellation, early ABORT, stale/duplicate ACKs,
  failed futures, aux failure, staging teardown, cross-rank source drain, PP
  release consensus, cleanup failure and compaction/offload guards.
- Five unrelated request-validation cases fail identically on the unmodified
  base commit and were explicitly deselected from that regression run:
  `test_metadata_rejection_does_not_partially_write_buffers`,
  `test_responses_rejects_before_background_acknowledgment`,
  `test_sampling_mask_respects_preferred_params_modes_and_batch`,
  `test_sampling_mask_unset_empty_disabled_and_enabled`, and
  `test_server_limits_are_configurable_and_cli_visible`.
- Initial real Mooncake RDMA probe (not rerun for this review): **PASS**, independent GPU 0 to GPU 1 buffers,
  1 MiB, on the authorized GB300 node. Before completion there was no abort ACK
  and reuse was denied; after completion the ACK allowed reuse and B's marker
  remained intact. The first attempt lacked sufficient container memlock and
  failed QP creation; the successful run used unlimited container memlock.
- Runtime: Mooncake `0.3.13.post1`, FlashInfer `0.6.18`, supplied image digest
  `sha256:999f1fa8ea5cb554b3b2007b23d5c486931f920207fcc561d5576107a878bca6`,
  with the fixed source mounted over the installed Python package.
- Ruff selected checks, isort, formatting and `git diff --check` pass.

No full Kimi-K3/EP32 benchmark, multi-node transport test, or PP GPU run was
performed. PP coverage here is the CPU release-consensus regression. This fix
must not be described as a demonstrated recovery of the DeepSWE score.
