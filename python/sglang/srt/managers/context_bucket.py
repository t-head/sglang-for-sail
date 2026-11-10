"""Context-distribution-aware DP routing without moving running requests.

The budget is an estimate: scheduler snapshots replace local reservations,
just as for the existing load-aware policies. It is not a KV admission limit.
"""

from __future__ import annotations

from bisect import bisect_left
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
    from sglang.srt.managers.load_snapshot import LoadSnapshot


# Inclusive upper bounds, followed by an overflow bucket. Keep these fixed so
# schedulers and the controller interpret /v1/loads histograms identically.
CONTEXT_BUCKET_BOUNDS = (
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
    524288,
    1048576,
)


def context_bucket_index(length: int) -> int:
    return bisect_left(CONTEXT_BUCKET_BOUNDS, length)


def estimate_decode_context_length(req: TokenizedGenerateReqInput) -> int:
    """Estimate context when an incoming PD request starts decoding.

    Embedding requests get placeholder input IDs only inside the scheduler.
    Prefill transfers one generated token before Decode starts; include it so
    bucket-boundary prompts match the scheduler's input-plus-output lengths.
    """
    inputs = req.input_embeds if req.input_embeds is not None else req.input_ids
    return len(inputs) + 1


class ContextBucketBudget:
    """Balance length buckets among ranks with similar outstanding counts.

    A candidate may have at most one more outstanding request than the least
    loaded rank. This leaves a little placement freedom without allowing a
    short-context rank to accumulate an arbitrarily deep waiting queue.
    """

    def __init__(self, num_dp_ranks: int):
        self.num_dp_ranks = num_dp_ranks
        self.total_requests = [0] * num_dp_ranks
        self.context_tokens = [0] * num_dp_ranks
        self.context_histograms: list[list[int] | None] = [None] * num_dp_ranks
        self.last_timestamp: list[float | None] = [None] * num_dp_ranks
        self.next_rank = 0

    def update_budget(self, loads: list[LoadSnapshot]) -> None:
        for load in loads:
            rank = load.dp_rank
            if not 0 <= rank < self.num_dp_ranks:
                continue
            if load.timestamp == self.last_timestamp[rank]:
                continue
            self.last_timestamp[rank] = load.timestamp
            self.total_requests[rank] = load.num_running_reqs + load.num_waiting_reqs
            histogram = load.context_length_histogram
            # Missing/older writers must not look like ranks with zero load.
            self.context_histograms[rank] = (
                list(histogram)
                if histogram is not None
                and len(histogram) == len(CONTEXT_BUCKET_BOUNDS) + 1
                else None
            )
            self.context_tokens[rank] = load.num_context_tokens

    def reserve(self, rank: int, estimated_tokens: int) -> None:
        self.total_requests[rank] += 1
        self.context_tokens[rank] += estimated_tokens
        histogram = self.context_histograms[rank]
        if histogram is not None:
            histogram[context_bucket_index(estimated_tokens)] += 1

    def dispatch(self, estimated_tokens: int, active_ranks: list[int]) -> int:
        if not active_ranks:
            raise RuntimeError("Cannot route request: no active DP workers available.")

        min_requests = min(self.total_requests[rank] for rank in active_ranks)
        candidates = [
            rank
            for rank in active_ranks
            if self.total_requests[rank] <= min_requests + 1
        ]
        bucket = context_bucket_index(estimated_tokens)

        def cyclic_order(rank: int) -> int:
            return (rank - self.next_rank) % self.num_dp_ranks

        if all(self.context_histograms[rank] is not None for rank in candidates):
            rank = min(
                candidates,
                key=lambda rank: (
                    # Balance each bucket independently. Always avoiding ranks
                    # with long requests would send their short requests away,
                    # leaving count deficits that attract more long requests.
                    self.context_histograms[rank][bucket],
                    self.context_tokens[rank],
                    self.total_requests[rank],
                    cyclic_order(rank),
                ),
            )
        else:
            # Until all candidates have a histogram, use request counts. Mixing
            # real histograms with missing ones would bias toward unknown ranks.
            rank = min(
                candidates,
                key=lambda rank: (self.total_requests[rank], cyclic_order(rank)),
            )

        self.reserve(rank, estimated_tokens)
        self.next_rank = (rank + 1) % self.num_dp_ranks
        return rank
