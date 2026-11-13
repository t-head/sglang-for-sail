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
- ``drop_misaligned_dsa_cp_prefix_reuse`` / ``enforce_dsa_cp_prefix_cache_alignment``;
- the gate inside ``match_kv_cache`` (drop rolls the slot back, aligned hits
  are kept untouched);
- the deferred mamba COW/clear collection shape invariant.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
import unittest.mock
from contextlib import ExitStack
from array import array
from types import SimpleNamespace

import torch

from sglang.srt.disaggregation import utils as dutils
from sglang.srt.managers.schedule_batch import ReqKvInfo, ScheduleBatch
from sglang.srt.mem_cache.base_prefix_cache import (
    MatchResult,
    dsa_cp_prefix_cache_alignment,
    drop_misaligned_dsa_cp_prefix_reuse,
    dsa_cp_prefix_reuse_length,
    enforce_dsa_cp_prefix_cache_alignment,
    is_dsa_cp_prefix_hit_misaligned,
    rollback_matched_mamba_state,
)
from sglang.srt.mem_cache.common import match_kv_cache
from sglang.srt.mem_cache.kv_cache_builder import require_dsa_cp_page_size

_ALIGNMENT_SYMBOL = "sglang.srt.mem_cache.base_prefix_cache.dsa_cp_prefix_cache_alignment"


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


class TestRollbackMatchedMambaState(unittest.TestCase):
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


class TestBypassPredicates(unittest.TestCase):
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
                unittest.mock.Mock(return_value=SimpleNamespace(disaggregation_mode=mode)),
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
        stack.enter_context(
            unittest.mock.patch.object(
                dutils.envs,
                "SGLANG_DSA_CP_PREFIX_CACHE",
                SimpleNamespace(get=lambda: switch),
            )
        )

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


class TestDsaCpReuseLength(unittest.TestCase):
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


class TestDropMisalignedReuse(unittest.TestCase):
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


class TestGateInMatchKvCache(unittest.TestCase):
    """The single mount point every request-side match goes through."""

    def _call(self, tree, req, *, cow_mamba=True):
        with (
            unittest.mock.patch(_ALIGNMENT_SYMBOL, return_value=4),
            unittest.mock.patch(
                "sglang.srt.mem_cache.common._req_radix_key", return_value=object()
            ),
        ):
            return match_kv_cache(
                req, tree, req.origin_input_ids, cow_mamba=cow_mamba
            )

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

        self.assertIsNotNone(req.kv.mamba_pool_idx)
        self.assertIsNone(req.kv.mamba_cow_src_index)
        self.assertFalse(req.kv.mamba_needs_clear)
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


class TestCollectDeferredMambaState(unittest.TestCase):
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


class TestPageSizePrecondition(unittest.TestCase):
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
        # Inputs are mocked only: the cached alignment helper must compute 256
        # itself and agree with the startup precondition.
        dsa_cp_prefix_cache_alignment.cache_clear()
        try:
            with ExitStack() as stack:
                patch = unittest.mock.patch
                stack.enter_context(
                    patch(
                        "sglang.srt.environ.envs.SGLANG_DSA_CP_PREFIX_CACHE.get",
                        return_value=True,
                    )
                )
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
        finally:
            dsa_cp_prefix_cache_alignment.cache_clear()
        self.assertEqual(alignment, 256)
        self.assertEqual(alignment, require_dsa_cp_page_size(256, 4))


if __name__ == "__main__":
    unittest.main()
