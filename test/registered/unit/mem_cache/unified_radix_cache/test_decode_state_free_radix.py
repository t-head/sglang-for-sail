"""SGLANG_DECODE_STATE_FREE_RADIX: FULL KV only; each request's Mamba slot is freed."""

import unittest
from array import array
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.managers.req_types import FINISH_LENGTH
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocation import ensure_mamba_capacity
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.registry import keep_full_kv_only
from sglang.srt.mem_cache.static_kv_storage import acquire_rows
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.unified_radix_cache.config import CacheConfig
from sglang.test.kits.unified_radix_cache.fixtures import build_fixture, match_prefix
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")

# The fixture's no-buffer Mamba tree needs page 1; the FULL-only tree does not.
PAGE = 1


def _state_free_cache():
    cfg = CacheConfig(
        components=(ComponentType.FULL, ComponentType.MAMBA),
        page_size=PAGE,
        mamba_cache_size=8,
    )
    hybrid, allocator, req_to_token_pool = build_fixture(cfg)
    params = replace(hybrid.cache_init_params, enable_mamba_extra_buffer=False)
    params.tree_slots = tuple(keep_full_kv_only(params, params.tree_slots))
    return UnifiedRadixCache(params=params), allocator, req_to_token_pool


def _request(req_to_token_pool, allocator, tokens, rid):
    req = Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=array("q", tokens),
        sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
    )
    acquire_rows(req_to_token_pool, req_to_token_pool.state, [req])
    req.output_ids = array("q")
    kv = allocator.alloc(len(tokens))
    req_to_token_pool.write((req.kv.req_pool_idx, slice(0, len(tokens))), kv)
    req.kv.kv_committed_len = len(tokens)
    req.kv.kv_allocated_len = len(tokens)
    req.kv.cache_protected_len = 0
    req.lock = None
    req.extra_key = None
    return req


class TestDecodeStateFreeRadix(CustomTestCase):
    def test_request_only_tree_has_no_mamba_component(self):
        cache, _, _ = _state_free_cache()
        self.assertFalse(cache.is_mamba_enabled)
        self.assertNotIn(ComponentType.MAMBA, cache.components)
        self.assertTrue(cache.request_mamba_only)
        self.assertTrue(cache.supports_mamba())
        self.assertEqual(cache.mamba_evictable_size(), 0)

    def test_state_shortage_does_not_evict_full_kv(self):
        cache, _, pool = _state_free_cache()
        # The shared allocator and request-state APIs are exercised by existing
        # lifecycle tests; pin the no-tree-state eviction decision independently.
        state = SimpleNamespace(
            pending_request_slots=lambda pool, req: 1,
            slots=SimpleNamespace(slots_fit=lambda needed: False),
        )
        fake_pool = SimpleNamespace(holds_state_slots=True, binds_row_units=False)
        with patch.object(
            cache, "evict_for_alloc", side_effect=AssertionError("evicted")
        ):
            self.assertFalse(ensure_mamba_capacity(fake_pool, state, [object()], cache))

    def test_finished_request_is_cached_and_frees_its_slot(self):
        cache, allocator, req_to_token_pool = _state_free_cache()
        mamba_free = req_to_token_pool.mamba_allocator.available_size()
        tokens = list(range(1, 13))
        req = _request(req_to_token_pool, allocator, tokens, "finished")
        self.assertEqual(
            req_to_token_pool.mamba_allocator.available_size(), mamba_free - 1
        )
        req.finished_reason = FINISH_LENGTH(length=0)
        release_kv_cache(req, cache, checkpoint=True)
        cache.unlock(req.lock)
        self.assertFalse(req.kv.holds_mamba)
        self.assertEqual(req_to_token_pool.mamba_allocator.available_size(), mamba_free)
        self.assertEqual(match_prefix(cache, tokens).device_prefix_len, len(tokens))
        cache.sanity_check()

    def test_request_released_before_kv_frees_its_slot(self):
        cache, _, req_to_token_pool = _state_free_cache()
        mamba_free = req_to_token_pool.mamba_allocator.available_size()
        req = Req(
            rid="early",
            origin_input_text="",
            origin_input_ids=array("q", [1, 2, 3]),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
        )
        req.kv.mamba_pool_idx = req_to_token_pool.mamba_allocator.alloc(1)[0]
        self.assertFalse(req.kv.holds_kv)
        release_kv_cache(req, cache, checkpoint=False)
        self.assertEqual(req_to_token_pool.mamba_allocator.available_size(), mamba_free)


if __name__ == "__main__":
    unittest.main()
