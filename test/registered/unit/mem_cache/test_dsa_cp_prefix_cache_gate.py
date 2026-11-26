"""Unit tests for the DSA prefill-CP prefix-cache alignment gate (0.5.23 line).

A hybrid/mamba request matches its prefix with ``cow_mamba=True``, which may
allocate a mamba slot and stage copy-on-write metadata. When the alignment gate
drops the reuse, all of that must be rolled back: lingering COW fields, a leaked
slot, or a stale ``mamba_branching_seqlen`` would silently corrupt the
recomputed prefix.

Ported from the 0.5.21 change set and re-based on this line's architecture:
- every request-side prefix match now funnels through
  ``sglang.srt.mem_cache.common.match_kv_cache`` (the scheduler's
  ``init_next_round_input``, the waiting-queue policy and the PD decode
  prealloc queue), so the gate is exercised there instead of at three separate
  call sites;
- ``MatchResult`` reports ``device_prefix_len`` (an int) instead of
  ``device_indices``, so the reuse-length accounting is asserted against the new
  field;
- the companion HiCache pump stubs are out of scope on this line (the cache
  class they targeted no longer exists), so the ChunkCache unit test that shipped
  with the 0.5.21 change is deliberately not carried over.

Covers:
- ``rollback_matched_mamba_state`` directly (fresh vs. pre-held vs. session slot);
- the PD bypass predicates (only interleave prefill-CP can un-bypass);
- reuse-length accounting for HiCache host extensions;
- ``drop_misaligned_dsa_cp_prefix_reuse`` / ``enforce_dsa_cp_prefix_cache_alignment``,
  including the retained-slot mark that makes the recompute clear the old
  recurrent state;
- the gate inside ``match_kv_cache`` (drop rolls the slot back, aligned hits
  are kept untouched);
- the deferred mamba COW/clear collection shape invariant;
- the resolved page geometry behind ``dsa_cp_prefix_cache_alignment`` (lcm, not
  a product; an inactive call must not poison a later active one);
- the once-per-process misaligned-hit warning;
- the switch surface: only ``SGLANG_DSA_CP_PREFIX_CACHE`` is honored on this
  line, the 0.5.21 variable name is inert (no alias by decision);
- the LayerSplit broadcast scratch invalidation and the prefetch transfer-coord
  fallbacks added by the 0.5.23 migration batch;
- the host-pool write-through invalidating a non-owner rank's scratch copy, and
  packed MTP draft transfer ids following the live target layer count;
- ``require_dsa_cp_hicache_layout`` (layer_first + kernel required with
  ``--enable-dsa-cache-layer-split``, inert otherwise).
"""

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import os
import unittest
import unittest.mock
from array import array
from contextlib import ExitStack
from types import SimpleNamespace

import torch

from sglang.srt.disaggregation import utils as dutils
from sglang.srt.disaggregation.utils import require_dsa_cp_hicache_layout
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import ReqKvInfo, ScheduleBatch
from sglang.srt.mem_cache.base_prefix_cache import (
    MatchResult,
    drop_misaligned_dsa_cp_prefix_reuse,
    dsa_cp_prefix_cache_alignment,
    dsa_cp_prefix_reuse_length,
    enforce_dsa_cp_prefix_cache_alignment,
    is_dsa_cp_prefix_hit_misaligned,
    rollback_matched_mamba_state,
)
from sglang.srt.mem_cache.common import match_kv_cache
from sglang.srt.mem_cache.dsa_cache_layer_split import LayerSplitDSATokenToKVPool
from sglang.srt.mem_cache.kv_cache_builder import require_dsa_cp_page_size
from sglang.srt.mem_cache.pool_host.dsa import DSAIndexerPoolHost
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost

_ALIGNMENT_SYMBOL = (
    "sglang.srt.mem_cache.base_prefix_cache.dsa_cp_prefix_cache_alignment"
)


class _TrackingMambaAllocator:
    def __init__(self, slot: int = 5):
        self.slot = slot
        self.alloc_calls = 0
        self.free_calls = 0
        self.live = set()

    def alloc(self) -> torch.Tensor:
        self.alloc_calls += 1
        # 0-dim, matching every production producer (``mid[0]``,
        # ``dst_index[0]``); ``_collect_deferred_mamba_cow_and_clear`` relies on
        # that shape to build its 1-D gather index.
        idx = torch.tensor(self.slot, dtype=torch.int64)
        self.live.add(self.slot)
        return idx

    def free(self, indices: torch.Tensor) -> None:
        self.free_calls += 1
        self.live.discard(int(indices[0]))


def _tree_cache(allocator) -> SimpleNamespace:
    return SimpleNamespace(req_to_token_pool=SimpleNamespace(mamba_allocator=allocator))


class TestRollbackMatchedMambaState(CustomTestCase):
    def test_freshly_allocated_slot_is_freed(self):
        allocator = _TrackingMambaAllocator()
        kv = ReqKvInfo(
            mamba_pool_idx=allocator.alloc(),
            mamba_cow_src_index=torch.tensor(2, dtype=torch.int64),
            mamba_needs_clear=True,
        )
        req = SimpleNamespace(kv=kv, session=None)

        rollback_matched_mamba_state(
            _tree_cache(allocator), req, mamba_held_before_match=False
        )

        self.assertIsNone(req.kv.mamba_pool_idx)
        self.assertIsNone(req.kv.mamba_cow_src_index)
        self.assertFalse(req.kv.mamba_needs_clear)
        self.assertEqual(allocator.free_calls, 1)
        self.assertEqual(len(allocator.live), 0)

    def test_pre_held_slot_is_kept(self):
        allocator = _TrackingMambaAllocator()
        kv = ReqKvInfo(
            mamba_pool_idx=torch.tensor(3, dtype=torch.int64),
            mamba_cow_src_index=torch.tensor(2, dtype=torch.int64),
            mamba_needs_clear=True,
        )
        req = SimpleNamespace(kv=kv, session=None)

        rollback_matched_mamba_state(
            _tree_cache(allocator), req, mamba_held_before_match=True
        )

        self.assertIsNotNone(req.kv.mamba_pool_idx)
        self.assertIsNone(req.kv.mamba_cow_src_index)
        self.assertFalse(req.kv.mamba_needs_clear)
        self.assertEqual(allocator.free_calls, 0)

    def test_session_slot_is_never_freed(self):
        allocator = _TrackingMambaAllocator()
        kv = ReqKvInfo(mamba_pool_idx=torch.tensor(4, dtype=torch.int64))
        req = SimpleNamespace(kv=kv, session=SimpleNamespace())

        rollback_matched_mamba_state(_tree_cache(allocator), req)

        self.assertIsNotNone(req.kv.mamba_pool_idx)
        self.assertEqual(allocator.free_calls, 0)


class TestBypassPredicates(CustomTestCase):
    """Only an interleave PD prefill-CP worker may keep the radix cache on."""

    def _enter(
        self,
        mode="prefill",
        enable_cp=True,
        strategy="interleave",
        switch=False,
        cp_size=8,
    ):
        stack = ExitStack()
        self.addCleanup(stack.close)
        # Production reads these two through the runtime-context *accessors*
        # (get_parallel() / get_disagg()), so a stub must be a callable that
        # returns the bag, not the bag itself.
        stack.enter_context(
            unittest.mock.patch.object(
                dutils,
                "get_disagg",
                unittest.mock.Mock(
                    return_value=SimpleNamespace(disaggregation_mode=mode)
                ),
            )
        )
        stack.enter_context(
            unittest.mock.patch.object(
                dutils,
                "get_parallel",
                unittest.mock.Mock(
                    return_value=SimpleNamespace(
                        enable_prefill_cp=enable_cp,
                        cp_strategy=strategy,
                        attn_cp_size=cp_size,
                    )
                ),
            )
        )
        stack.enter_context(dutils.envs.SGLANG_DSA_CP_PREFIX_CACHE.override(switch))

    def _bypass(self, **kwargs):
        # is_dsa is an argument of the predicate under test, not of _enter.
        is_dsa = kwargs.pop("is_dsa", True)
        self._enter(**kwargs)
        return dutils.should_bypass_dsa_cp_prefix_cache(is_dsa=is_dsa)

    def test_interleave_is_supported(self):
        self._enter()
        self.assertTrue(dutils.supports_dsa_cp_prefix_cache())

    def test_zigzag_is_not_supported(self):
        self._enter(strategy="zigzag")
        self.assertFalse(dutils.supports_dsa_cp_prefix_cache())

    def test_unified_deployment_is_not_supported(self):
        self._enter(mode="null")
        self.assertFalse(dutils.supports_dsa_cp_prefix_cache())

    def test_decode_mode_is_not_supported(self):
        # Page geometry is only validated on prefill workers; a decode worker
        # must keep its existing bypass even with the switch on.
        self._enter(mode="decode")
        self.assertFalse(dutils.supports_dsa_cp_prefix_cache())
        self.assertTrue(self._bypass(mode="decode", switch=True))

    def test_switch_off_keeps_bypass(self):
        self.assertTrue(self._bypass())

    def test_switch_on_unbypasses_interleave(self):
        self.assertFalse(self._bypass(switch=True))

    def test_switch_on_keeps_bypass_for_non_interleave(self):
        self.assertTrue(self._bypass(switch=True, strategy="zigzag"))

    def test_switch_on_without_prefill_cp_keeps_todays_behavior(self):
        # Not a CP deployment at all: the switch is inert, no bypass happened
        # before this feature and none happens after it.
        self.assertFalse(self._bypass(switch=True, enable_cp=False))

    def test_non_dsa_never_bypasses(self):
        self.assertFalse(self._bypass(is_dsa=False))

    def test_cp_width_one_is_not_a_supported_deployment(self):
        self._enter(cp_size=1)
        self.assertFalse(dutils.supports_dsa_cp_prefix_cache())

    def test_cp_width_one_keeps_bypass_even_with_switch_on(self):
        # The platform can silently drop the CP width to one (the DSA-family
        # override recomputes it from tp_size). Without a real CP shard the
        # runtime alignment guard is inert, so the switch must stay inert too
        # rather than keep the radix cache on behind a marker that promises
        # protection that is not there.
        self.assertTrue(self._bypass(switch=True, cp_size=1))


class _ZeroingTreeCache:
    """Minimal cache stub for ``zero_match_result`` / rollback paths."""

    def __init__(self, allocator=None):
        # ``_GateTreeCache.match_prefix`` stages a slot through the same
        # allocator the production rollback reads back
        # (``tree_cache.req_to_token_pool.mamba_allocator``); keep a direct
        # handle too so the stub is internally consistent.
        self.allocator = allocator
        self.req_to_token_pool = SimpleNamespace(mamba_allocator=allocator)
        self.root = object()

    def supports_prefix_sharing(self) -> bool:
        return True

    def root_node_handle(self, extra_key=None):
        return self.root


def _match(device_len: int, host_len: int = 0, full_len: int = 0) -> MatchResult:
    node = object()
    return MatchResult(
        device_prefix_len=device_len,
        last_device_node=node,
        last_host_node=node,
        best_match_node=node,
        host_hit_length=host_len,
        mamba_branching_seqlen=9,
        full_kv_hit_length=full_len,
    )


class TestDsaCpReuseLength(CustomTestCase):
    """The gate judges the whole reusable prefix, not just its device part."""

    def test_host_extension_counts_towards_alignment(self):
        self.assertEqual(dsa_cp_prefix_reuse_length(_match(8, host_len=2)), 10)

    def test_full_kv_hit_length_without_device_hit(self):
        self.assertEqual(dsa_cp_prefix_reuse_length(_match(0, full_len=6)), 6)

    def test_aligned_total_passes(self):
        with unittest.mock.patch(_ALIGNMENT_SYMBOL, return_value=4):
            self.assertFalse(is_dsa_cp_prefix_hit_misaligned(_match(8, host_len=8)))

    def test_gate_inactive_when_alignment_is_one(self):
        with unittest.mock.patch(_ALIGNMENT_SYMBOL, return_value=1):
            self.assertFalse(is_dsa_cp_prefix_hit_misaligned(_match(3)))


class TestDropMisalignedReuse(CustomTestCase):
    def _req(self, allocator):
        return SimpleNamespace(
            kv=ReqKvInfo(mamba_pool_idx=allocator.alloc()), session=None
        )

    def test_misaligned_drop_rolls_back_staged_slot(self):
        allocator = _TrackingMambaAllocator()
        tree = _ZeroingTreeCache(allocator)
        req = self._req(allocator)
        req.kv.mamba_cow_src_index = torch.tensor(2, dtype=torch.int64)
        req.kv.mamba_needs_clear = True
        match = _match(3)

        with unittest.mock.patch(_ALIGNMENT_SYMBOL, return_value=4):
            dropped = drop_misaligned_dsa_cp_prefix_reuse(
                tree, req, match, mamba_held_before_match=False, extra_key=None
            )

        self.assertEqual(dsa_cp_prefix_reuse_length(dropped), 0)
        self.assertIs(dropped.last_device_node, tree.root)
        # The staged mamba state is undone, not just the match.
        self.assertIsNone(req.kv.mamba_pool_idx)
        self.assertIsNone(req.kv.mamba_cow_src_index)
        self.assertFalse(req.kv.mamba_needs_clear)
        self.assertEqual(allocator.free_calls, 1)
        # A dropped match must not keep a branching hint either.
        self.assertIsNone(dropped.mamba_branching_seqlen)

    def test_aligned_match_is_returned_untouched(self):
        allocator = _TrackingMambaAllocator()
        tree = _ZeroingTreeCache(allocator)
        req = self._req(allocator)
        match = _match(8)

        with unittest.mock.patch(_ALIGNMENT_SYMBOL, return_value=4):
            kept = drop_misaligned_dsa_cp_prefix_reuse(tree, req, match)

        self.assertIs(kept, match)
        self.assertEqual(allocator.free_calls, 0)
        self.assertIsNotNone(req.kv.mamba_pool_idx)

    def test_non_cow_enforce_drops_reuse_only(self):
        allocator = _TrackingMambaAllocator()
        tree = _ZeroingTreeCache(allocator)
        req = self._req(allocator)
        # 6 device + 1 host = 7 tokens, not a multiple of the 4-token alignment.
        match = _match(6, host_len=1)

        with unittest.mock.patch(_ALIGNMENT_SYMBOL, return_value=4):
            dropped = enforce_dsa_cp_prefix_cache_alignment(tree, match)

        self.assertEqual(dsa_cp_prefix_reuse_length(dropped), 0)
        self.assertIsNone(dropped.mamba_branching_seqlen)
        # A non-cow match staged nothing, so nothing may be freed here.
        self.assertEqual(allocator.free_calls, 0)
        self.assertIsNotNone(req.kv.mamba_pool_idx)

    def test_retained_session_state_is_cleared_before_recomputation(self):
        # Carried over from the 0.5.21 change set: when the slot survives the
        # rollback (the request already held it before this match), the pool
        # never marks it for clearing, so the drop path must do it -- otherwise
        # the recompute reads the previous round's recurrent state as if it were
        # the initial state.
        allocator = _TrackingMambaAllocator()
        tree = _ZeroingTreeCache(allocator)
        kv = ReqKvInfo(mamba_pool_idx=torch.tensor(4, dtype=torch.int64))
        req = SimpleNamespace(kv=kv, session=None)
        match = _match(device_len=6)

        with unittest.mock.patch(_ALIGNMENT_SYMBOL, return_value=4):
            dropped = drop_misaligned_dsa_cp_prefix_reuse(
                tree,
                req,
                match,
                mamba_held_before_match=True,
            )

        self.assertEqual(dsa_cp_prefix_reuse_length(dropped), 0)
        # A pre-held slot is kept (no double free) ...
        self.assertEqual(req.kv.mamba_pool_idx.item(), 4)
        self.assertEqual(allocator.free_calls, 0)
        # ... but it is now marked for clearing, and the stale COW hint is gone.
        self.assertTrue(req.kv.mamba_needs_clear)
        self.assertIsNone(req.kv.mamba_cow_src_index)

    def test_cleared_slot_is_collected_as_a_clear_index(self):
        # The other half of the same contract: the batch collector must turn the
        # mark written above into a clear index, so the slot really gets zeroed
        # before the recomputed prefix is written.
        allocator = _TrackingMambaAllocator()
        req = SimpleNamespace(
            kv=ReqKvInfo(
                mamba_pool_idx=torch.tensor(4, dtype=torch.int64),
                mamba_needs_clear=True,
            )
        )
        batch = SimpleNamespace()

        ScheduleBatch._collect_deferred_mamba_cow_and_clear(batch, [req])

        self.assertEqual(batch.mamba_clear_indices.tolist(), [4])
        self.assertIsNone(batch.mamba_cow_src_indices)
        self.assertFalse(req.kv.mamba_needs_clear)
        self.assertEqual(allocator.free_calls, 0)


class _GateTreeCache(_ZeroingTreeCache):
    """Cache stub whose match stages mamba COW state like the real one."""

    def __init__(self, allocator, hit_len: int, mamba_branching_seqlen: int):
        super().__init__(allocator)
        self.hit_len = hit_len
        self.mamba_branching_seqlen = mamba_branching_seqlen

    def supports_mamba(self) -> bool:
        return True

    def match_prefix(self, params):
        req = params.req
        if params.cow_mamba and not req.kv.holds_mamba:
            req.kv.mamba_pool_idx = self.allocator.alloc()
            req.kv.mamba_cow_src_index = torch.tensor(2, dtype=torch.int64)
            req.kv.mamba_needs_clear = True
        return MatchResult(
            device_prefix_len=self.hit_len,
            last_device_node=self.root,
            last_host_node=self.root,
            best_match_node=self.root,
            host_hit_length=0,
            mamba_branching_seqlen=self.mamba_branching_seqlen,
        )


def _req_stub() -> SimpleNamespace:
    return SimpleNamespace(
        kv=ReqKvInfo(),
        session=None,
        extra_key=None,
        cache_salt=None,
        origin_input_ids=array("q", [1, 2, 3, 4, 5, 6, 7, 8]),
        output_ids=array("q", []),
        prefix_len=0,
        num_matched_prefix_tokens=0,
        last_node=None,
        last_host_node=None,
        best_match_node=None,
        host_hit_length=0,
        swa_host_hit_length=0,
        mamba_host_hit_length=0,
        swa_branching_seqlen=None,
        mamba_branching_seqlen=None,
        _compute_max_prefix_len=lambda n: n,
    )


class TestGateInMatchKvCache(CustomTestCase):
    """The single mount point every request-side match goes through."""

    def _call(self, tree, req, *, cow_mamba=True):
        with (
            unittest.mock.patch(_ALIGNMENT_SYMBOL, return_value=4),
            unittest.mock.patch(
                "sglang.srt.mem_cache.common._req_radix_key", return_value=object()
            ),
        ):
            return match_kv_cache(req, tree, req.origin_input_ids, cow_mamba=cow_mamba)

    def test_misaligned_hit_rolls_back_mamba_state(self):
        allocator = _TrackingMambaAllocator()
        tree = _GateTreeCache(allocator, hit_len=3, mamba_branching_seqlen=7)
        req = _req_stub()

        self._call(tree, req)

        # Gate dropped the reuse: no COW residue, no leaked slot, no hint.
        self.assertIsNone(req.kv.mamba_cow_src_index)
        self.assertFalse(req.kv.mamba_needs_clear)
        self.assertIsNone(req.kv.mamba_pool_idx)
        self.assertIsNone(req.mamba_branching_seqlen)
        self.assertEqual(allocator.alloc_calls, 1)
        self.assertEqual(allocator.free_calls, 1)
        self.assertEqual(len(allocator.live), 0)
        # Match zeroed to the root, so the request recomputes.
        self.assertEqual(int(req.prefix_len), 0)
        self.assertIs(req.last_node, tree.root)

    def test_pre_held_slot_survives_the_gate(self):
        # Later prefill rounds already hold a slot: the gate must clear only the
        # staged COW hints, never free the slot another round allocated.
        allocator = _TrackingMambaAllocator()
        tree = _GateTreeCache(allocator, hit_len=3, mamba_branching_seqlen=7)
        req = _req_stub()
        req.kv.mamba_pool_idx = torch.tensor(3, dtype=torch.int64)

        self._call(tree, req)

        # The slot is retained, so the drop path must also arm the clear
        # that the recompute consumes (see mamba_needs_clear producer in
        # HybridReqToTokenPool.alloc, which only marks fresh slots).
        self.assertIsNotNone(req.kv.mamba_pool_idx)
        self.assertIsNone(req.kv.mamba_cow_src_index)
        self.assertTrue(req.kv.mamba_needs_clear)
        self.assertIsNone(req.mamba_branching_seqlen)
        self.assertEqual(allocator.alloc_calls, 0)
        self.assertEqual(allocator.free_calls, 0)

    def test_aligned_hit_keeps_match_and_mamba_state(self):
        allocator = _TrackingMambaAllocator()
        tree = _GateTreeCache(allocator, hit_len=4, mamba_branching_seqlen=7)
        req = _req_stub()

        self._call(tree, req)

        self.assertEqual(int(req.prefix_len), 4)
        self.assertIsNotNone(req.kv.mamba_pool_idx)
        self.assertEqual(allocator.free_calls, 0)
        self.assertEqual(req.mamba_branching_seqlen, 7)

    def test_non_cow_match_drops_reuse_without_touching_slots(self):
        # A non-cow match stages nothing: the enforced branch must drop the
        # unaligned reuse but must not allocate or free any mamba slot.
        allocator = _TrackingMambaAllocator()
        tree = _GateTreeCache(allocator, hit_len=3, mamba_branching_seqlen=7)
        req = _req_stub()

        self._call(tree, req, cow_mamba=False)

        self.assertEqual(int(req.prefix_len), 0)
        self.assertEqual(allocator.alloc_calls, 0)
        self.assertEqual(allocator.free_calls, 0)
        self.assertIsNone(req.kv.mamba_pool_idx)


class TestCollectDeferredMambaState(CustomTestCase):
    """Batch-level invariant: deferred COW/clear matches held 0-dim slots."""

    def test_staged_cow_uses_0dim_slot(self):
        batch = SimpleNamespace()
        kv = ReqKvInfo(
            mamba_pool_idx=torch.tensor(5, dtype=torch.int64),
            mamba_cow_src_index=torch.tensor(7, dtype=torch.int64),
        )
        req = SimpleNamespace(kv=kv)

        ScheduleBatch._collect_deferred_mamba_cow_and_clear(batch, [req])

        self.assertEqual(batch.mamba_cow_src_indices.tolist(), [7])
        self.assertEqual(batch.mamba_cow_dst_indices.tolist(), [5])
        self.assertIsNone(batch.mamba_clear_indices)
        self.assertIsNone(req.kv.mamba_cow_src_index)
        self.assertFalse(req.kv.mamba_needs_clear)

    def test_staged_clear_uses_0dim_slot(self):
        batch = SimpleNamespace()
        kv = ReqKvInfo(
            mamba_pool_idx=torch.tensor(3, dtype=torch.int64),
            mamba_needs_clear=True,
        )
        req = SimpleNamespace(kv=kv)

        ScheduleBatch._collect_deferred_mamba_cow_and_clear(batch, [req])

        self.assertIsNone(batch.mamba_cow_src_indices)
        self.assertEqual(batch.mamba_clear_indices.tolist(), [3])

    def test_staged_cow_without_slot_fails(self):
        batch = SimpleNamespace()
        kv = ReqKvInfo(
            mamba_pool_idx=None,
            mamba_cow_src_index=torch.tensor(7, dtype=torch.int64),
        )
        req = SimpleNamespace(kv=kv)

        with self.assertRaises(AssertionError):
            ScheduleBatch._collect_deferred_mamba_cow_and_clear(batch, [req])


class TestPageSizePrecondition(CustomTestCase):
    """The startup precondition must follow the platform's ``64 * index_kpool``
    page derivation (0.5.22+), not the 0.5.21 literal 64. These cases exercise
    the real rejection logic in ``require_dsa_cp_page_size``, which is what the
    prefill worker calls before printing the feature-hit marker.
    """

    def test_kpool_one_still_requires_64(self):
        self.assertEqual(require_dsa_cp_page_size(64, 1), 64)

    def test_kpool_four_requires_256(self):
        # GLM5.3 on 0.5.23: index_kpool=4 -> derived logical page 256.
        self.assertEqual(require_dsa_cp_page_size(256, 4), 256)

    def test_page_size_128_with_kpool_four_is_rejected(self):
        # 128 // 4 == 32 would break the 64-token index-page requirement.
        with self.assertRaises(ValueError) as ctx:
            require_dsa_cp_page_size(128, 4)
        message = str(ctx.exception)
        for needle in ("page_size=128", "index_kpool=4", "expected=256"):
            self.assertIn(needle, message)

    def test_legacy_literal_64_with_kpool_four_is_rejected(self):
        # The pre-migration precondition is now itself the counterexample.
        with self.assertRaises(ValueError):
            require_dsa_cp_page_size(64, 4)

    def test_alignment_guard_agrees_with_expected_page_size(self):
        # Inputs are mocked only: the alignment helper recomputes from the
        # active context and must agree with the startup precondition.
        with ExitStack() as stack:
            patch = unittest.mock.patch
            stack.enter_context(envs.SGLANG_DSA_CP_PREFIX_CACHE.override(True))
            stack.enter_context(
                patch(
                    "sglang.srt.layers.attention.dsa.utils."
                    "is_dsa_prefill_cp_interleave",
                    return_value=True,
                )
            )
            stack.enter_context(
                patch(
                    "sglang.srt.runtime_context.get_schedule",
                    return_value=SimpleNamespace(page_size=256),
                )
            )
            stack.enter_context(
                patch(
                    "sglang.srt.runtime_context.process_model_config",
                    return_value=SimpleNamespace(
                        hf_config=SimpleNamespace(index_kpool=4)
                    ),
                )
            )
            alignment = dsa_cp_prefix_cache_alignment()
        self.assertEqual(alignment, 256)
        self.assertEqual(alignment, require_dsa_cp_page_size(256, 4))


class TestResolvedDsaCpAlignment(CustomTestCase):
    """The gate must follow the *resolved* page geometry, and must not be
    process-cached: an inactive call may not poison a later active one.

    Ported from the 0.5.21 ``TestResolvedDsaCpAlignment``; on this line the
    alignment is recomputed from the active context on every call
    (``dsa_cp_prefix_cache_alignment`` reads ``get_schedule()`` and
    ``process_model_config()`` lazily), so the 0.5.21 ``cache_clear`` cleanup
    has no counterpart to assert -- the "does not poison" case below is the
    behaviour that cleanup used to protect.
    """

    def _active_context(self, stack: ExitStack, page_size, index_kpool: int):
        stack.enter_context(envs.SGLANG_DSA_CP_PREFIX_CACHE.override(True))
        stack.enter_context(
            unittest.mock.patch.object(
                dutils,
                "get_disagg",
                unittest.mock.Mock(
                    return_value=SimpleNamespace(disaggregation_mode="prefill")
                ),
            )
        )
        stack.enter_context(
            unittest.mock.patch.object(
                dutils,
                "get_parallel",
                unittest.mock.Mock(
                    return_value=SimpleNamespace(
                        enable_prefill_cp=True,
                        cp_strategy="interleave",
                        attn_cp_size=8,
                    )
                ),
            )
        )
        stack.enter_context(
            unittest.mock.patch(
                "sglang.srt.runtime_context.get_schedule",
                return_value=SimpleNamespace(page_size=page_size),
            )
        )
        stack.enter_context(
            unittest.mock.patch(
                "sglang.srt.runtime_context.process_model_config",
                return_value=SimpleNamespace(
                    hf_config=SimpleNamespace(index_kpool=index_kpool)
                ),
            )
        )

    def test_absent_page_size_falls_back_to_one_and_follows_kpool(self):
        # A schedule without a page_size (operator input may leave it unset)
        # must not crash the gate and must still honour the indexer granularity.
        with ExitStack() as stack:
            self._active_context(stack, None, 4)
            self.assertEqual(dsa_cp_prefix_cache_alignment(), 4)

    def test_alignment_is_the_lcm_not_the_product(self):
        # page_size=128 with index_kpool=4: 128 already divides by 4, so the
        # required alignment is 128 -- the 0.5.21 product form would have asked
        # for 512 and dropped reuse that the kpool path could serve.
        with ExitStack() as stack:
            self._active_context(stack, 128, 4)
            self.assertEqual(dsa_cp_prefix_cache_alignment(), 128)

    def test_disabled_call_does_not_poison_later_enabled_call(self):
        with envs.SGLANG_DSA_CP_PREFIX_CACHE.override(False):
            self.assertEqual(dsa_cp_prefix_cache_alignment(), 1)
        with ExitStack() as stack:
            self._active_context(stack, 256, 4)
            self.assertEqual(dsa_cp_prefix_cache_alignment(), 256)

    def test_unsupported_deployment_keeps_alignment_inert(self):
        # Same switch, but a decode worker: the guard must stay out of the way.
        with ExitStack() as stack:
            self._active_context(stack, 256, 4)
            stack.enter_context(
                unittest.mock.patch.object(
                    dutils,
                    "get_disagg",
                    unittest.mock.Mock(
                        return_value=SimpleNamespace(disaggregation_mode="decode")
                    ),
                )
            )
            self.assertEqual(dsa_cp_prefix_cache_alignment(), 1)


class TestMisalignedWarningIsOnce(CustomTestCase):
    """A misaligned hit is expected during bring-up; log it once per process.

    This is the 0.5.23 form of the 0.5.21 module-level "warned once" global: the
    flag now lives in the runtime context resources, so it follows the context
    instead of the import.
    """

    def _hit(self):
        with ExitStack() as stack:
            stack.enter_context(unittest.mock.patch(_ALIGNMENT_SYMBOL, return_value=4))
            stack.enter_context(
                unittest.mock.patch(
                    "sglang.srt.runtime_context.get_context",
                    return_value=SimpleNamespace(
                        resources=SimpleNamespace(dsa_cp_misaligned_warned=False)
                    ),
                )
            )
            with unittest.mock.patch(
                "sglang.srt.mem_cache.base_prefix_cache.logger"
            ) as box:
                self.assertTrue(is_dsa_cp_prefix_hit_misaligned(_match(6)))
                self.assertTrue(is_dsa_cp_prefix_hit_misaligned(_match(10)))
        return box.warning.call_count

    def test_repeated_misaligned_hits_warn_once(self):
        # Both hits above must be reported as misaligned, but only the first one
        # may emit the warning; a per-request warning would bury the log.
        self.assertEqual(self._hit(), 1)

    def test_aligned_hit_does_not_warn(self):
        with ExitStack() as stack:
            stack.enter_context(unittest.mock.patch(_ALIGNMENT_SYMBOL, return_value=4))
            stack.enter_context(
                unittest.mock.patch(
                    "sglang.srt.runtime_context.get_context",
                    return_value=SimpleNamespace(
                        resources=SimpleNamespace(dsa_cp_misaligned_warned=False)
                    ),
                )
            )
            with unittest.mock.patch(
                "sglang.srt.mem_cache.base_prefix_cache.logger"
            ) as box:
                self.assertFalse(is_dsa_cp_prefix_hit_misaligned(_match(8)))
        self.assertEqual(box.warning.call_count, 0)


class TestSwitchNameIsThe0523NameOnly(CustomTestCase):
    """The opt-in is ``SGLANG_DSA_CP_PREFIX_CACHE`` and nothing else.

    Registered decision for the migration: the 0.5.21 variable name is *not*
    carried over as an alias on this line, so an old customer script setting the
    0.5.21 name must keep the default behaviour rather than silently enabling
    the experimental path. If a real dependency on the old name is reported,
    widening it is a separate, evidence-backed change.
    """

    def test_old_0521_variable_name_is_inert(self):
        with unittest.mock.patch.dict(
            os.environ, {"SGLANG_ENABLE_DSA_CP_PREFIX_CACHE": "1"}
        ):
            self.assertFalse(dutils.dsa_cp_prefix_cache_enabled())

    def test_no_alias_attribute_is_exposed(self):
        self.assertFalse(hasattr(envs, "SGLANG_ENABLE_DSA_CP_PREFIX_CACHE"))
        self.assertTrue(hasattr(envs, "SGLANG_DSA_CP_PREFIX_CACHE"))

    def test_0523_name_is_the_one_that_unbypasses(self):
        with envs.SGLANG_DSA_CP_PREFIX_CACHE.override(True):
            self.assertTrue(dutils.dsa_cp_prefix_cache_enabled())


class TestLayerSplitScratchInvalidation(CustomTestCase):
    """Every rank must drop its broadcast scratch copy on a host write-through.

    Covers ``invalidate_kv_buffer_for_layer`` (added together with the KV-side
    write-through invalidation) and keeps it distinct from the index-side twin.
    The stubs are plain attribute bags: the method under test only reads the
    broadcast bookkeeping, so no pool, device or process group is needed.
    """

    def _pool(self, *, pending_layer=None, remote_layer=None):
        calls = []
        pool = SimpleNamespace(
            pending_remote_kv_layer_id=pending_layer,
            remote_kv_layer_id=remote_layer,
            pending_remote_kv_broadcast=pending_layer is not None,
            index_key_cache=SimpleNamespace(
                invalidate=lambda layer: calls.append(("index", layer))
            ),
            _finalize_pending_kv_broadcast=lambda *, set_remote_layer_id: calls.append(
                ("finalize", set_remote_layer_id)
            ),
        )
        pool.calls = calls
        return pool

    def test_invalidate_kv_drops_pending_broadcast_without_publishing_it(self):
        pool = self._pool(pending_layer=7, remote_layer=7)

        LayerSplitDSATokenToKVPool.invalidate_kv_buffer_for_layer(pool, 7)

        self.assertIn(("finalize", False), pool.calls)
        self.assertIsNone(pool.remote_kv_layer_id)

    def test_invalidate_kv_leaves_other_layers_scratch_copy_alone(self):
        pool = self._pool(remote_layer=3)

        LayerSplitDSATokenToKVPool.invalidate_kv_buffer_for_layer(pool, 7)

        self.assertEqual(pool.calls, [])
        self.assertEqual(pool.remote_kv_layer_id, 3)

    def test_invalidate_kv_publishes_nothing_when_only_a_pending_broadcast_exists(
        self,
    ):
        pool = self._pool(pending_layer=7)

        LayerSplitDSATokenToKVPool.invalidate_kv_buffer_for_layer(pool, 7)

        self.assertEqual(pool.calls, [("finalize", False)])
        self.assertIsNone(pool.remote_kv_layer_id)

    def test_index_invalidation_is_a_separate_call(self):
        pool = self._pool(remote_layer=7)

        LayerSplitDSATokenToKVPool.invalidate_index_buffer_for_layer(pool, 7)

        self.assertEqual(pool.calls, [("index", 7)])
        self.assertEqual(pool.remote_kv_layer_id, 7)


class TestLayerSplitPrefetchWaits(CustomTestCase):
    """A prefetch may not skip the wait just because a caller left the transfer
    coordinates unset (the non-comm path reads a layer another rank owns)."""

    def _pool(self, *, counter, comm=None):
        broadcasts = []
        pool = SimpleNamespace(
            remote_kv_layer_id=None,
            pending_remote_kv_broadcast=False,
            pending_remote_kv_layer_id=None,
            _finalize_pending_kv_broadcast=lambda **_: None,
            _local_layer_idx=lambda layer_id: layer_id - 100,
            kv_buffer=["slot-0", "slot-1", "slot-2"],
            _is_layer_owned=lambda layer_id: False,
            layer_transfer_counter=counter,
            layer_broadcast_comm=comm,
            remote_kv_buffer="remote-scratch",
            _broadcast_tensor_from_owner=lambda *a, **k: broadcasts.append((a, k)),
        )
        pool.broadcasts = broadcasts
        return pool

    def test_unset_counter_falls_back_to_the_pools_own_counter(self):
        counter = unittest.mock.Mock()
        pool = self._pool(counter=counter)

        LayerSplitDSATokenToKVPool.prefetch_kv_buffer(pool, 102)

        counter.wait_until.assert_called_once_with(2)
        self.assertEqual(len(pool.broadcasts), 1)
        self.assertEqual(pool.remote_kv_layer_id, 102)

    def test_explicit_coordinates_win_over_the_fallback(self):
        pool_counter = unittest.mock.Mock()
        explicit = unittest.mock.Mock()
        pool = self._pool(counter=pool_counter)

        LayerSplitDSATokenToKVPool.prefetch_kv_buffer(pool, 102, explicit, 5)

        explicit.wait_until.assert_called_once_with(5)
        pool_counter.wait_until.assert_not_called()

    def test_no_counter_anywhere_still_broadcasts(self):
        # Nothing to wait for: the synchronous broadcast is still issued, so the
        # fallback cannot swallow the prefetch.
        pool = self._pool(counter=None)

        LayerSplitDSATokenToKVPool.prefetch_kv_buffer(pool, 102)

        self.assertEqual(len(pool.broadcasts), 1)
        self.assertEqual(pool.remote_kv_layer_id, 102)

    def test_already_remote_layer_is_a_no_op(self):
        counter = unittest.mock.Mock()
        pool = self._pool(counter=counter)
        pool.remote_kv_layer_id = 102

        LayerSplitDSATokenToKVPool.prefetch_kv_buffer(pool, 102)

        counter.wait_until.assert_not_called()
        self.assertEqual(pool.broadcasts, [])


class TestHostWriteThroughInvalidatesScratch(CustomTestCase):
    """The write-through entry must invalidate before the ownership early-return,
    otherwise a non-owner rank keeps serving a pre-write scratch copy."""

    def _device_pool(self, recorder, *, start_layer=8):
        return SimpleNamespace(
            start_layer=start_layer,
            layer_num=9,
            invalidate_index_buffer_for_layer=recorder,
            invalidate_kv_buffer_for_layer=recorder,
        )

    def test_dsa_host_pool_invalidates_on_a_non_owner_rank(self):
        seen = []
        host = SimpleNamespace(
            _is_device_layer_sharded=lambda device_pool: True,
            _is_device_layer_owned=lambda device_pool, layer_id: False,
        )

        DSAIndexerPoolHost.load_to_device_per_layer(
            host,
            self._device_pool(lambda layer: seen.append(layer)),
            None,
            None,
            3,
            "kernel",
        )

        # start_layer + layer_id: the scratch buffer is indexed by absolute layer.
        self.assertEqual(seen, [11])

    def test_dsa_host_pool_skips_invalidation_when_layers_are_not_sharded(self):
        seen = []
        host = SimpleNamespace(
            _is_device_layer_sharded=lambda device_pool: False,
            _is_device_layer_owned=lambda device_pool, layer_id: False,
        )

        DSAIndexerPoolHost.load_to_device_per_layer(
            host,
            self._device_pool(lambda layer: seen.append(layer)),
            None,
            None,
            3,
            "kernel",
        )

        self.assertEqual(seen, [])

    def test_mla_host_pool_invalidates_kv_scratch_before_returning_early(self):
        seen = []
        host = SimpleNamespace(
            _is_dummy=False,
            _is_device_layer_sharded=lambda device_pool: True,
            _target_layer_indices=set(),
        )

        MLATokenToKVPoolHost.load_to_device_per_layer(
            host,
            self._device_pool(lambda layer: seen.append(layer)),
            None,
            None,
            3,
            "kernel",
        )

        self.assertEqual(seen, [11])

    def test_mla_host_pool_skips_invalidation_when_layers_are_not_sharded(self):
        seen = []
        host = SimpleNamespace(
            _is_dummy=False,
            _is_device_layer_sharded=lambda device_pool: False,
            _target_layer_indices=set(),
        )

        MLATokenToKVPoolHost.load_to_device_per_layer(
            host,
            self._device_pool(lambda layer: seen.append(layer)),
            None,
            None,
            3,
            "kernel",
        )

        self.assertEqual(seen, [])


class TestDraftTransferIdsFollowTargetLayerNum(CustomTestCase):
    """Packed MTP draft layers are addressed host-pool relative.

    The 0.5.21 line used the *device* pool's layer count here; with layer sharding
    that id is the rank's full layer count, not the number of live target layers,
    so the draft ids collided with the target range. This asserts the ids the host
    pool now produces, with a device pool whose ``layer_num`` deliberately
    disagrees.
    """

    def _host(self, *, sharded: bool, target_layer_num: int, drafts):
        calls = []
        host = SimpleNamespace(
            _is_device_layer_sharded=lambda device_pool: sharded,
            _owned_device_layer_ids=lambda device_pool: [0, 1],
            mtp_draft_device_pools=drafts,
            target_layer_num=target_layer_num,
            _backup_from_device_per_layer=lambda pool, h, d, layer_id, io_backend, **kw: (
                calls.append((pool, layer_id, kw.get("is_draft", False)))
            ),
        )
        return host, calls

    def test_draft_ids_start_at_the_live_target_layer_count(self):
        draft_a, draft_b = object(), object()
        host, calls = self._host(
            sharded=True, target_layer_num=2, drafts=(draft_a, draft_b)
        )
        device_pool = SimpleNamespace(layer_num=9)

        DSAIndexerPoolHost.backup_from_device_all_layer(
            host, device_pool, None, None, "kernel"
        )

        self.assertEqual(
            calls,
            [
                (device_pool, 0, False),
                (device_pool, 1, False),
                (draft_a, 2, True),
                (draft_b, 3, True),
            ],
        )

    def test_no_draft_pools_means_no_extra_ids_at_all(self):
        # Control for the loop above: a deployment without packed MTP drafts
        # cannot acquire the new ids, so the change stays inside that scenario.
        host, calls = self._host(sharded=True, target_layer_num=2, drafts=())
        device_pool = SimpleNamespace(layer_num=9)

        DSAIndexerPoolHost.backup_from_device_all_layer(
            host, device_pool, None, None, "kernel"
        )

        self.assertEqual([layer for _, layer, _ in calls], [0, 1])


class TestHiCacheLayoutGate(CustomTestCase):
    """``--enable-dsa-cache-layer-split`` needs a per-layer host layout."""

    def test_layer_split_hicache_layout(self):
        # The combination established on the 0.5.21 line (GLM-5.3 checkpoint):
        # layer_first host layout with the kernel I/O backend.
        require_dsa_cp_hicache_layout(
            cache_layer_split=True,
            hicache_layout="layer_first",
            hicache_io_backend="kernel",
        )
        with self.assertRaisesRegex(ValueError, "layer_first"):
            require_dsa_cp_hicache_layout(
                cache_layer_split=True,
                hicache_layout="page_first",
                hicache_io_backend="kernel",
            )

    def test_page_first_layout_is_rejected(self):
        with self.assertRaises(ValueError):
            require_dsa_cp_hicache_layout(
                cache_layer_split=True,
                hicache_layout="page_first",
                hicache_io_backend="kernel",
            )

    def test_direct_io_backend_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "kernel"):
            require_dsa_cp_hicache_layout(
                cache_layer_split=True,
                hicache_layout="layer_first",
                hicache_io_backend="direct",
            )

    def test_rejection_names_the_two_settings_that_are_wrong(self):
        with self.assertRaises(ValueError) as ctx:
            require_dsa_cp_hicache_layout(
                cache_layer_split=True,
                hicache_layout="page_first_kv_split",
                hicache_io_backend="direct",
            )
        message = str(ctx.exception)
        self.assertIn("page_first_kv_split", message)
        self.assertIn("direct", message)

    def test_gate_inert_without_layer_split(self):
        require_dsa_cp_hicache_layout(
            cache_layer_split=False,
            hicache_layout="page_first",
            hicache_io_backend="direct",
        )

    def test_gate_inert_when_hierarchical_cache_is_off(self):
        # The builder passes None for both fields when HiCache is disabled; a
        # layer-split deployment without HiCache keeps today's behaviour.
        require_dsa_cp_hicache_layout(
            cache_layer_split=True, hicache_layout=None, hicache_io_backend=None
        )


if __name__ == "__main__":
    unittest.main()
