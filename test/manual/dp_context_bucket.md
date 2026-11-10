# Decode DP context-bucket routing: first A/B

This experimental policy balances the **distribution of context lengths** across
DP ranks. It changes request placement; it does not change Attention, MegaMoE,
CUDA Graph, KV allocation, or the running-batch limit.

## Enable on Decode

For the fixed DP8 / TP32 / EP32, PP=1 K3 deployment, replace the Decode load-balancing
option and set the existing snapshot interval on **all Decode nodes**:

```bash
--load-balance-method context_bucket \
--load-snapshot-publish-interval 1
```

Keep Prefill and the other deployment settings unchanged. The base branch defaults
to a snapshot every 15 Decode iterations; use interval 1 in both A/B arms so the
load information stays current and the publication frequency is matched.

Requests must allow automatic Decode DP selection. An explicit `routed_dp_rank`
or `X-Data-Parallel-Rank` still takes priority. With full PD, keep Prefill's routing
and bootstrap binding intact; this option belongs to the Decode server.

## Placement rule

1. The scheduler reports current `Req.seqlen` (input plus generated tokens) for
   running requests and its waiting/admission queues. Lengths are CPU metadata.
2. Inclusive bucket upper bounds are 8K, 16K, 32K, 64K, 128K, 256K, 512K and 1M
   tokens, followed by an overflow bucket. K means 1024 tokens.
3. Among available DP ranks, consider those with at most **one more** running plus
   waiting request than the least-loaded rank. After a dispatch the difference
   can therefore be two; subsequent placements cannot widen it using that budget.
4. Choose the rank with the fewest requests in the **incoming request's bucket**.
   Break ties by current context-token total, outstanding request count, then
   rotating rank order. Reserve the new request immediately in the local budget.

Incoming Decode context is estimated as **prompt length + 1**, including the
first token handed off by Prefill. For `input_embeds`, prompt length comes from
the number of embedding rows: placeholder token IDs do not exist until the
request reaches the scheduler. This prevents exact-boundary prompts, such as
32K and 64K, from being routed using the bucket below their first Decode context.
Later generation can still cross a bucket boundary; snapshots track that growth.

Buckets are balanced independently. Always steering short requests away from a
rank with a long request can leave that rank with too few requests, causing a
request-count rule to send subsequent long requests back to it.

The reported context sum counts each request's full context, including shared
prefixes. It measures neither allocated KV blocks nor predicted remaining output.
The counted queues are waiting, Decode preallocation, transfer and retraction;
grammar compilation and paused requests are not included.

Running at batch 128 does not prevent a rank from receiving another request:
the existing scheduler queues handle admission. No extra global queue, reordering
of each scheduler's queue, or migration of running requests is introduced.

## Check effective load reporting

Query the Decode HTTP endpoint (the server endpoint, rather than a router that
does not forward `/v1/loads`):

```bash
curl -fsS 'http://DECODE_HOST:PORT/v1/loads?include=core' |
  jq '.loads[] | {dp_rank, timestamp, num_running_reqs, num_waiting_reqs,
                  context_length_histogram, num_context_tokens}'
```

Expect eight ranks, advancing timestamps and nine histogram bins per rank.
`null` means no histogram was reported; startup or missing histograms make routing
fall back to request counts. Histogram counts include waiting requests and are
not a substitute for the active `seq_lens` arrays of a sampled graph.

## Compare

- Use the same request file, arrival schedule, output settings and concurrency.
  Baseline: `round_robin` (or the existing policy); treatment: `context_bucket`.
  Keep `--load-snapshot-publish-interval 1` in both.
- Let the active workload turn over before capturing a steady-state trace: this
  policy cannot rebalance a batch whose requests have already been assigned.
- Check actual running batch, queue depth and per-DP histograms. Compare Graph
  duration and ITL P50/P95, output throughput and TTFT together.
- Better DP balance need not lower ITL when a single unsplit long-context MLA
  request dominates the EP group's critical path.

## Validation and limits

The implementation reuses the controller's 20ms snapshot refresh throttle.
Changed snapshots replace local reservations, so requests still in IPC can be
under-counted across a refresh. This is an approximate budget, not an exact
request-lifecycle ledger or a hard KV-capacity check.

This first version requires disaggregated Decode and PP=1. Ordinary/Prefill
extend or chunked batches and other PP microbatches are not fully represented by
these snapshots. Elastic EP / DP expansion is also rejected: the current snapshot
transport is sized to the launch DP count.

CPU tests cover bucket boundaries, burst reservations, snapshot refresh/fallback,
full batches, explicit/inactive rank routing, current generated lengths, queue
accounting, embedding inputs, dynamic arrival/completion at length boundaries,
and real SHM/ZMQ serialization. On a configured SGLang development
environment, run:

```bash
python -m pytest -q \
  test/registered/unit/managers/test_context_bucket.py \
  test/registered/unit/managers/test_data_parallel_controller.py \
  test/registered/unit/managers/test_load_inquirer.py \
  test/registered/unit/managers/test_load_snapshot_backends.py \
  test/registered/unit/server_args/test_server_args.py::TestLoadBalanceMethod
```

Local macOS validation isolates GPU imports and runtime configuration setup;
it is not a full server startup, numerical-accuracy test or GPU performance result.
