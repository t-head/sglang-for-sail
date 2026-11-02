"""Direct L3 support for :class:`UnifiedRadixCache`.

Links the cache's device pools straight to an external KV store, with no host
tier in between. The transport contract and tree-side wrapper live here, while
each backend owns its device-pool layout and physical I/O.

* :class:`UnifiedCacheLinker` -- the transport interface a backend implements.
* :class:`UnifiedCacheLinkerWrapper` -- the tree-side flow that drives it. The
  cache owns one as a plain attribute, keeping the whole external-cache path out
  of the main tree file.

The tree only needs a handful of guarded hooks:

* ``match_prefix``      -> :meth:`UnifiedCacheLinkerWrapper.match`
* ``init_load_back``    -> :meth:`UnifiedCacheLinkerWrapper.load_back`
* ``BackupKV`` actions  -> :meth:`UnifiedCacheLinkerWrapper.offload_nodes`

A load lands in device slots private to the loading request and joins the tree
only when that request inserts it like any computed KV. A failed load therefore
reaches nothing but its own requests, which the scheduler aborts.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import Future
from typing import TYPE_CHECKING, NamedTuple, Optional

import torch

from sglang.srt.mem_cache.allocator.swa import is_swa_req_ring
from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    MatchResult,
)
from sglang.srt.mem_cache.hicache_storage import (
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import (
    ComponentType,
    ExternalLinkerLoadPhase,
    LinkerTransferPhase,
    TreeComponent,
)
from sglang.srt.mem_cache.utils import get_storage_hash_str

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.unified_cache.unified_tree_core_interface import NodeId
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

logger = logging.getLogger(__name__)

_EXTERNAL_LINKER_SUPPORTED_COMPONENTS = frozenset(
    {
        ComponentType.FULL,
        ComponentType.SWA,
    }
)


class LayerWiseLoadCounter:
    """CPU completion counter compatible with KV pools' layer wait hook.

    A failed layer does not raise into the forward that waits on it: the
    forward runs on whatever the slots hold, and :meth:`finish` reports the
    failure afterwards so the scheduler can abort the loading requests before
    their output is used.
    """

    def __init__(
        self,
        num_layers: int,
        on_layer_ready: Optional[Callable[[int, int], None]] = None,
    ):
        self.num_layers = num_layers
        self.on_layer_ready = on_layer_ready
        self.producer_index = -1
        self.consumer_index = -1
        self.futures: dict[int, list[Future]] = {}

    def update_producer(self) -> int:
        self.producer_index += 1
        self.futures[self.producer_index] = [Future() for _ in range(self.num_layers)]
        return self.producer_index

    def set_consumer(self, index: int) -> None:
        self.consumer_index = index

    def complete(self, index: int, layer: int) -> None:
        self.futures[index][layer].set_result(None)

    def fail(self, index: int, error: BaseException) -> None:
        for future in self.futures.get(index, ()):
            if not future.done():
                future.set_exception(error)

    def wait_until(self, threshold: int) -> None:
        index = self.consumer_index
        futures = self.futures.get(index)
        if futures is None:
            return
        # Blocks until the layer resolves; a failure is reported by finish().
        futures[threshold].exception()
        # Runs on failure too: an MLA dedup source must still issue the layer
        # broadcast its peers are waiting for.
        if self.on_layer_ready is not None:
            self.on_layer_ready(index, threshold)

    def finish(self, index: int) -> bool:
        """Wait for every layer of batch ``index``; True when all of them landed."""
        futures = self.futures.get(index)
        if futures is None:
            return True
        # Wait before dropping the entry: the load thread still completes the
        # layers the forward never waited on through it.
        # Do not short-circuit after an error: later layers may still be writing.
        errors = [future.exception() for future in futures]
        landed = all(error is None for error in errors)
        del self.futures[index]
        return landed

    def reset(self) -> None:
        self.producer_index = -1
        self.consumer_index = -1
        self.futures.clear()


class UnifiedCacheLinker(ABC):
    """External KV store reached directly from the device pools."""

    layer_done_counter: object
    log_load_timing: bool = False

    @abstractmethod
    def lookup(self, rid: str, transfers: list[PoolTransfer]) -> list[int]:
        """Return every prefix length (in pages) that is fully restorable.

        A length is included only when *all* pools satisfy their hit policy at
        that exact boundary (contiguous prefix pools, plus each trailing-window
        pool's window ending there). Trailing-window state (SWA / compress
        state) only exists at offloaded node boundaries, so the set is sparse
        and generally non-contiguous -- returning just the local maximum would
        let the tree pick a length that is invalid on another rank.

        Local to this rank; the tree intersects the sets across ranks.
        """

    @abstractmethod
    def load(self, rid: str, transfers: list[PoolTransfer]) -> bool:
        """Queue a load into the given device indices.

        The transfer is executed by the next ``start_layer_wise_loading`` call,
        not here.
        """

    @abstractmethod
    def start_layer_wise_loading(self) -> int:
        """Start queued loads and return the layer-counter consumer index."""

    @abstractmethod
    def finish_layer_wise_loading(self, counter_index: int) -> bool:
        """Wait until the batch started as ``counter_index`` stops writing to
        device memory; True when every layer landed."""

    @abstractmethod
    def cancel_queued_load(self, rid: str) -> bool:
        """Cancel a load that has not started yet."""

    @abstractmethod
    def offload(self, transfers: list[PoolTransfer]) -> bool:
        """Queue every transfer for atomic persistence."""

    @abstractmethod
    def num_completed_offloads(self) -> int:
        """Return the number of completed offloads waiting to be consumed."""

    @abstractmethod
    def pop_completed_offload(self) -> bool:
        """Consume the oldest completed offload and return its result."""

    @abstractmethod
    def reset(self) -> None:
        """Quiesce all transfers and reset backend state before returning."""

    @abstractmethod
    def close(self) -> None:
        """Quiesce all transfers and release backend resources."""


class ExternalCacheHitMarker(NamedTuple):
    """What ``match`` found in the external store, consumed by ``load_back``.

    ``tail_hashes`` are the per-page storage hashes of the restorable tail,
    starting at ``device_hit_len``.
    """

    tail_hashes: list[str]
    device_hit_len: int


class _PendingOffload(NamedTuple):
    lock_node_id: NodeId
    lock_params: DecLockRefParams
    publish_node_ids: list[NodeId]


class UnifiedCacheLinkerWrapper:
    """Drives an external KV store on behalf of one :class:`UnifiedRadixCache`."""

    def __init__(
        self,
        cache: UnifiedRadixCache,
        cache_linker: UnifiedCacheLinker,
    ):
        unsupported = set(cache.tree_components) - _EXTERNAL_LINKER_SUPPORTED_COMPONENTS
        if unsupported:
            names = ", ".join(
                component.name for component in sorted(unsupported, key=int)
            )
            raise ValueError(
                "External cache linker supports only Full and SWA tree "
                f"components; unsupported: {names}"
            )

        self.cache = cache
        self.cache_linker = cache_linker
        swa = cache.components.get(ComponentType.SWA)
        self._skip_swa = swa is not None and is_swa_req_ring(
            cache.token_to_kv_pool_allocator
        )
        self._components = tuple(
            component
            for component in cache._components_tuple
            if not (self._skip_swa and component is swa)
        )
        # rid -> what match found, consumed by the next init_load_back.
        self.hit_markers: dict[str, ExternalCacheHitMarker] = {}
        # Requests loading in the batch being built, then its counter index once
        # the batch starts; finish_loads consumes both after its forward.
        self.inflight_load_rids: list[str] = []
        # Loaded tails stay outside the tree until their forward completes.
        self.private_load_indices: dict[str, torch.Tensor] = {}
        self.inflight_load_index = -1
        # Offloads in flight, each holding a lock on its node until it lands.
        self.pending_offloads: list[_PendingOffload] = []

        cache.tree_core.enable_external_cache_linker = True
        cache.write_through_threshold = 1
        self.pp_coordinator = None
        self.pp_hits = {}
        if getattr(cache, "pp_size", 1) > 1:
            from sglang.srt.distributed.parallel_state import (
                create_custom_parallel_group,
            )
            from sglang.srt.mem_cache.unified_cache.linker_pp import PPLinkerCoordinator

            rank = torch.distributed.get_rank()
            tp_ranks = torch.distributed.get_process_group_ranks(cache.tp_group)
            pp_ranks = torch.distributed.get_process_group_ranks(cache.pp_group)
            ranks = sorted({p + t - rank for p in pp_ranks for t in tp_ranks})
            group = create_custom_parallel_group(ranks, backend="gloo")
            self.pp_coordinator = PPLinkerCoordinator(
                group, self._lookup_hashes, self._wait_local_offload
            )
            cache_linker.log_load_timing = self.pp_coordinator.is_source

    def submit_pp_request(self, req: Req) -> None:
        from sglang.srt.mem_cache.common import _req_radix_key

        req.refresh_fill_ids()
        key = _req_radix_key(
            self.cache,
            req,
            req.full_untruncated_fill_ids,
            req._compute_max_prefix_len(len(req.full_untruncated_fill_ids)),
        )
        key, _ = key.maybe_to_bigram_view(self.cache.tree_core.is_eagle)
        if req.positional_embed_overrides is not None:
            key = key[:0]
        hashes = get_storage_hash_str(
            key.page_aligned(self.cache.page_size), page_size=self.cache.page_size
        )
        self.pp_coordinator.submit(req.rid, hashes)

    def start_pp_lookups(self, rids: Iterable[str]) -> None:
        if self.pp_coordinator is not None:
            for rid in rids:
                self.pp_coordinator.start_lookup(rid)

    def check_pp_ready(self, rid: str) -> bool:
        if self.pp_coordinator is None:
            return True
        sequence = (
            self.pp_coordinator.ready_sequence(rid) if self.cache.pp_rank == 0 else 0
        )
        ready = torch.tensor(sequence, dtype=torch.int64)
        self.cache._all_reduce(ready, torch.distributed.ReduceOp.MIN)
        if not ready.item():
            return False
        self.pp_hits[rid] = self.pp_coordinator.result(rid, int(ready.item()))
        return True

    def defer_pp_request(self, rid: str) -> None:
        if self.pp_coordinator is not None:
            hit = self.pp_hits.pop(rid, None)
            if hit is not None:
                self.pp_coordinator.defer(rid, hit.sequence)
            self.hit_markers.pop(rid, None)

    def _lookup_transfers(self, hashes: list[str]) -> list[PoolTransfer]:
        transfers = []
        for component in self._components:
            transfer = component.build_external_linker_transfer(
                LinkerTransferPhase.LOOKUP, None, hashes
            )
            if transfer is None:
                return []
            transfers.append(transfer)
        return transfers

    def _lookup_hashes(self, rid: str, hashes: list[str]) -> list[int]:
        transfers = self._lookup_transfers(hashes)
        return self.cache_linker.lookup(rid, transfers) if transfers else []

    def _wait_local_offload(self) -> bool:
        deadline = time.monotonic() + 600
        while not self.cache_linker.num_completed_offloads():
            if time.monotonic() >= deadline:
                # Fail the coordinator, never acknowledge a still-active DMA.
                raise TimeoutError("PP linker offload failed to complete within 600s")
            time.sleep(0.01)
        return self.cache_linker.pop_completed_offload()

    @property
    def layer_done_counter(self) -> object:
        return self.cache_linker.layer_done_counter

    def has_hit(self, rid: str) -> bool:
        return rid in self.hit_markers

    # ---- match: probe the remote store and report host_hit_length ----

    def match(self, key: RadixKey, req: Req, result: MatchResult) -> MatchResult:
        cache = self.cache
        key, _ = key.maybe_to_bigram_view(cache.tree_core.is_eagle)
        page = cache.page_size
        device_hit_len = result.device_prefix_len
        if device_hit_len >= len(key):
            return result

        if self.pp_coordinator is not None:
            hit = self.pp_hits.get(req.rid)
            if hit is None:
                return result  # Scheduling policy may probe before admission.
            candidates = [p for p in hit.pages if device_hit_len < p * page <= len(key)]
            if not candidates:
                return result
            end_page = candidates[-1]
            tail_hashes = hit.hashes[device_hit_len // page : end_page]
            hit_pages = len(tail_hashes)
            lookup_transfers = self._lookup_transfers(tail_hashes)
        else:
            tail_hashes = self._tail_hashes(key, result, device_hit_len)
            if not tail_hashes:
                return result
            lookup_transfers = self._lookup_transfers(tail_hashes)
            if not lookup_transfers:
                return result
            hit_pages = self._sync_restorable_prefix(
                self.cache_linker.lookup(req.rid, lookup_transfers),
                num_pages=len(tail_hashes),
                device_hit_pages=0,
            )
        if hit_pages == 0:
            return result
        by_pool = {transfer.name: transfer for transfer in lookup_transfers}
        hit_tokens = hit_pages * page

        swa_transfer = by_pool.get(PoolName.SWA)
        swa_host_hit_length = (
            min(len(swa_transfer.keys), hit_pages) * page
            if swa_transfer is not None
            else 0
        )
        # Mamba keeps a single state slot per node, so a hit is worth one slot.
        mamba_host_hit_length = 1 if PoolName.MAMBA in by_pool else 0

        self.hit_markers[req.rid] = ExternalCacheHitMarker(
            tail_hashes=list(tail_hashes[:hit_pages]),
            device_hit_len=device_hit_len,
        )
        return result._replace(
            last_host_node=result.best_match_node,
            host_hit_length=hit_tokens,
            swa_host_hit_length=max(result.swa_host_hit_length, swa_host_hit_length),
            mamba_host_hit_length=max(
                result.mamba_host_hit_length, mamba_host_hit_length
            ),
        )

    def _sync_restorable_prefix(
        self, restorable: list[int], *, num_pages: int, device_hit_pages: int
    ) -> int:
        """Intersect the per-rank sets of restorable prefix lengths and return the
        longest one, or 0 when the ranks share none beyond the device prefix.

        A rank's set is sparse, so reducing per-rank maxima could land on a
        length that only some ranks can restore. On a 0/1 mask MIN is AND, which
        makes the reduction an intersection.
        """
        mask = torch.zeros(num_pages + 1, dtype=torch.int)
        for pages in restorable:
            if device_hit_pages < pages <= num_pages:
                mask[pages] = 1
        self.cache._all_reduce_attn_groups(mask, torch.distributed.ReduceOp.MIN)
        common = mask.nonzero()
        if common.numel() == 0:
            return 0
        return int(common[-1].item())

    def _tail_hashes(
        self, key: RadixKey, result: MatchResult, device_hit_len: int
    ) -> list[str]:
        """Per-page storage hashes for the device-uncached tail of the prefix."""
        last_hash = None
        if device_hit_len > 0:
            last_hash = self.cache.get_last_hash_value(result.last_device_node)
            if last_hash is None:
                # Without the anchor the tail would hash as if it started at the
                # sequence head, yielding keys that can never match.
                return []
        page = self.cache.page_size
        tail_len = (len(key) - device_hit_len) // page * page
        if tail_len == 0:
            return []
        return get_storage_hash_str(
            key[device_hit_len : device_hit_len + tail_len],
            last_hash,
            page_size=page,
        )

    # ---- init_load_back: remote -> request-private device slots ----

    def load_back(self, req: Req) -> tuple[int, NodeId]:
        """Queue the external hit into device slots owned by ``req``.

        The loaded tail extends the request's prefix but stays out of the
        tree (``req.last_node`` is unchanged): the request inserts it like
        computed KV after its forward, once :meth:`finish_loads` has confirmed
        the load.
        """
        cache = self.cache
        hit = self.hit_markers.pop(req.rid, None)
        if hit is None:
            return 0, req.last_node

        device_hit_len = hit.device_hit_len
        tail_hashes = hit.tail_hashes
        prefix_len = device_hit_len + len(tail_hashes) * cache.page_size

        # Build per-component linker transfers.
        component_transfers: list[tuple[TreeComponent, PoolTransfer]] = []
        for component in self._components:
            transfer = component.build_external_linker_transfer(
                LinkerTransferPhase.LOAD, None, tail_hashes
            )
            if transfer is None:
                self._update_load(
                    ExternalLinkerLoadPhase.ABORT,
                    req,
                    component_transfers,
                    prefix_len,
                )
                if self.pp_coordinator is not None:
                    raise RuntimeError("PP linker could not allocate an agreed load")
                return 0, req.last_node
            component_transfers.append((component, transfer))

        full_transfer = component_transfers[0][1]
        assert full_transfer.name == PoolName.KV
        # Queue before PREPARE touches the request, so a refused load only has
        # the allocations to undo before falling back to recompute. The result
        # depends on the transfers alone, so every rank takes the same branch.
        try:
            queued = self.cache_linker.load(
                req.rid, [transfer for _, transfer in component_transfers]
            )
        except BaseException:
            self._update_load(
                ExternalLinkerLoadPhase.ABORT, req, component_transfers, prefix_len
            )
            raise
        if not queued:
            self._update_load(
                ExternalLinkerLoadPhase.ABORT, req, component_transfers, prefix_len
            )
            if self.pp_coordinator is not None:
                raise RuntimeError("PP linker backend refused an agreed load")
            return 0, req.last_node
        self._update_load(
            ExternalLinkerLoadPhase.PREPARE,
            req,
            component_transfers,
            prefix_len,
        )

        # Components omitted from the linker do not run their PREPARE hook.
        # Keep a non-restorable SWA range as tombstones instead of rebuilding
        # it from an uninitialized FULL-to-SWA mapping when the tail is inserted.
        if self._skip_swa:
            if req.kv is None:
                from sglang.srt.managers.schedule_batch import ReqKvInfo

                req.kv = ReqKvInfo(kv_allocated_len=prefix_len)
            req.kv.set_evicted_seqlen(
                ComponentType.SWA,
                max(req.kv.get_evicted_seqlen(ComponentType.SWA), prefix_len),
            )

        # match placed the SWA branch point before this tail was known; a point
        # the tail covers is gone, since the request inserts the whole tail.
        if (
            req.swa_branching_seqlen is not None
            and req.swa_branching_seqlen <= prefix_len
        ):
            req.swa_branching_seqlen = None
        self.private_load_indices[req.rid] = full_transfer.device_indices
        self.inflight_load_rids.append(req.rid)
        return len(full_transfer.device_indices), req.last_node

    def _update_load(
        self,
        phase: ExternalLinkerLoadPhase,
        req: Req,
        component_transfers: list[tuple[TreeComponent, PoolTransfer]],
        prefix_len: int,
    ) -> list[PoolTransfer]:
        if not component_transfers:
            return []
        full = component_transfers[0][1]
        result = []
        transfers = (
            reversed(component_transfers)
            if phase == ExternalLinkerLoadPhase.ABORT
            else component_transfers
        )
        for component, transfer in transfers:
            transfer = component.update_external_linker_load(
                phase, req, full, transfer, prefix_len
            )
            if transfer is not None:
                result.append(transfer)
        return result

    # ---- offload: device -> remote, driven by the write-through chain ----

    def offload_nodes(self, node_ids: Sequence[NodeId]) -> None:
        """Persist a write-through chain, skipping nodes already in the store."""
        for node_id in node_ids:
            transfers = self.cache.tree_core.build_external_linker_offload_transfers(
                node_id
            )
            if transfers is not None:
                if self._skip_swa:
                    transfers = [t for t in transfers if t.name != PoolName.SWA]
                self._offload_node(node_id, transfers)

    def _offload_node(self, node_id: NodeId, transfers: list[PoolTransfer]) -> None:
        cache = self.cache
        lock_params = cache.inc_lock_ref(node_id).to_dec_params()
        try:
            queued = self.cache_linker.offload(transfers)
        except BaseException:
            cache.dec_lock_ref(node_id, lock_params)
            raise
        if not queued:
            cache.dec_lock_ref(node_id, lock_params)
            return

        cache.tree_core.mark_external_linker_offload_pending(node_id)
        self.pending_offloads.append(_PendingOffload(node_id, lock_params, [node_id]))
        if self.pp_coordinator is not None:
            self.pp_coordinator.offload_submitted()

    def replace_pending_offload_node(
        self, ack_id: NodeId, old_node_id: NodeId, new_node_ids: list[NodeId]
    ) -> None:
        for index, pending in enumerate(self.pending_offloads):
            if pending.lock_node_id != ack_id:
                continue
            publish_node_ids = []
            for node_id in pending.publish_node_ids:
                if node_id == old_node_id:
                    publish_node_ids.extend(new_node_ids)
                else:
                    publish_node_ids.append(node_id)
            self.pending_offloads[index] = pending._replace(
                publish_node_ids=publish_node_ids
            )
            return

    def num_completed_offloads(self) -> int:
        backend = self.pp_coordinator or self.cache_linker
        return min(backend.num_completed_offloads(), len(self.pending_offloads))

    def take_completed_offloads(self, finish_count: int) -> list[bool]:
        assert finish_count <= len(self.pending_offloads)
        if self.pp_coordinator is not None:
            return self.pp_coordinator.take_completed_offloads(finish_count)
        return [self.cache_linker.pop_completed_offload() for _ in range(finish_count)]

    def commit_completed_offloads(self, successes: Sequence[bool]) -> None:
        assert len(successes) <= len(self.pending_offloads)
        for success in successes:
            pending = self.pending_offloads.pop(0)
            self.cache.tree_core.finish_external_linker_offload(
                pending.publish_node_ids, pending.lock_node_id, success
            )
            self.cache.dec_lock_ref(pending.lock_node_id, pending.lock_params)

    def start_layer_wise_loading(self) -> int:
        self.inflight_load_index = self.cache_linker.start_layer_wise_loading()
        return self.inflight_load_index

    def finish_loads(self) -> list[str]:
        """Wait for the loads of the batch that just ran; return the rids whose
        KV did not land, identically on every rank.

        Runs after that batch's forward, so no slot it wrote to has been freed
        or inserted yet. Every rank holds the same ``inflight_load_rids``
        because ``match`` reduces the hit across ranks, so either all ranks or
        none enter the reduction. The verdict is reduced because an MLA dedup
        peer only receives the broadcast and never sees the source's error.
        """
        rids, self.inflight_load_rids = self.inflight_load_rids, []
        if not rids:
            return []
        index, self.inflight_load_index = self.inflight_load_index, -1
        landed = torch.tensor(
            [int(self.cache_linker.finish_layer_wise_loading(index))],
            dtype=torch.int,
        )
        self.cache._all_reduce_attn_groups(landed, torch.distributed.ReduceOp.MIN)
        for rid in rids:
            self.private_load_indices.pop(rid, None)
        if landed.item():
            logger.info(
                "External linker stage load completed: requests=%d rids=%s",
                len(rids),
                rids,
            )
            return []
        logger.error(
            "External linker load failed; aborting %d request(s): %s", len(rids), rids
        )
        return rids

    # ---- lifecycle ----

    def reset(self) -> None:
        if self.pp_coordinator is not None:
            self.pp_coordinator.reset()
        self.pp_hits.clear()
        self.cache_linker.reset()
        self.hit_markers.clear()
        self.inflight_load_rids.clear()
        self.private_load_indices.clear()
        self.inflight_load_index = -1
        self._release_pending_offloads()

    def _release_pending_offloads(self) -> None:
        for pending in self.pending_offloads:
            self.cache.tree_core.finish_external_linker_offload(
                pending.publish_node_ids, pending.lock_node_id, False
            )
            self.cache.dec_lock_ref(pending.lock_node_id, pending.lock_params)
        self.pending_offloads.clear()

    def release_request(self, rid: str) -> None:
        if self.pp_coordinator is not None:
            self.pp_coordinator.release(rid)
        self.pp_hits.pop(rid, None)
        self.hit_markers.pop(rid, None)
        # A queued load targets slots the request owns and is about to free.
        # A started one is finished by finish_loads before anything is freed.
        if rid in self.inflight_load_rids and self.cache_linker.cancel_queued_load(rid):
            self.inflight_load_rids.remove(rid)
            self.private_load_indices.pop(rid, None)

    def close(self) -> None:
        if self.pp_coordinator is not None:
            self.pp_coordinator.close()
        self.pp_hits.clear()
        self.cache_linker.close()
        self.inflight_load_rids.clear()
        self.private_load_indices.clear()
        self._release_pending_offloads()
