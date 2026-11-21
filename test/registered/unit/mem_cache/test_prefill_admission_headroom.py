"""CPU tests for the allocator's empty-pool prompt admission bound."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers import tp_worker  # noqa: E402
from sglang.srt.mem_cache.allocator.base import (  # noqa: E402
    BaseTokenToKVPoolAllocator,
    max_input_tokens_for_memory,
)
from sglang.srt.mem_cache.allocator.page_interleave import (  # noqa: E402
    PageInterleavePoolAllocator,
)
from sglang.srt.mem_cache.allocator.paged import (  # noqa: E402
    PagedTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.allocator.swa import (  # noqa: E402
    SWATokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.allocator.token import (  # noqa: E402
    TokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.prefill_budget import SWAPrefillBudget  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _TokenBudget:
    # Use the production method without allocating a KV pool or a GPU tensor.
    max_new_tokens_for_memory = BaseTokenToKVPoolAllocator.max_new_tokens_for_memory

    def __init__(self, page_size):
        self.page_size = page_size


class TestPrefillAdmissionHeadroom(CustomTestCase):
    def worker_info(
        self,
        allocator,
        *,
        context_len=8192,
        block_size=None,
        shards=None,
        chunk_size=2048,
        is_multimodal=False,
    ):
        schedule = SimpleNamespace(
            max_prefill_tokens=2048,
            max_queued_requests=None,
            chunked_prefill_size=chunk_size,
        )
        worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                token_to_kv_pool_allocator=allocator,
                effective_logical_max_total_num_tokens=8192,
                logical_max_total_num_tokens=8192,
                max_running_requests=8,
                forward_stream=None,
                req_to_token_pool=SimpleNamespace(size=8, max_context_len=context_len),
                token_to_kv_pool=SimpleNamespace(size=8192),
            ),
            model_config=SimpleNamespace(
                context_len=context_len, is_multimodal=is_multimodal
            ),
            dllm_algorithm=(
                SimpleNamespace(block_size=block_size)
                if block_size is not None
                else None
            ),
            random_seed=0,
            device="cpu",
            sliding_window_size=128,
        )
        with (
            patch.object(tp_worker, "get_schedule", return_value=schedule),
            (
                patch.object(
                    tp_worker, "page_interleave_shard_size", return_value=shards
                )
                if shards is not None
                else nullcontext()
            ),
        ):
            return tp_worker.TpModelWorker.get_worker_info(worker)

    def swa_allocator(self, *, ring=False, swa_size=8192):
        # Do not construct a pool or allocate any device tensors.
        allocator = object.__new__(SWATokenToKVPoolAllocator)
        allocator.page_size = 256
        allocator._swa_req_ring = ring
        allocator._size_full = 8192
        allocator._size_swa = swa_size
        allocator.full_attn_allocator = SimpleNamespace(available_size=lambda: 8192)
        allocator.swa_attn_allocator = SimpleNamespace(available_size=lambda: swa_size)
        return allocator

    def remaining(self, allocator, input_tokens, max_new_tokens, capacity):
        return allocator.max_new_tokens_for_memory(
            input_tokens,
            max_new_tokens,
            token_capacity=capacity,
            sliding_window_size=None,
            chunk_size=None,
        )

    def limit(self, allocator, max_input_tokens, capacity):
        return max_input_tokens_for_memory(
            allocator,
            max_input_tokens,
            token_capacity=capacity,
            sliding_window_size=None,
            chunk_size=None,
        )

    def test_prompt_that_cannot_fit_returns_none(self):
        allocator = _TokenBudget(256)
        # The previous method returned zero, so Scheduler did not abort it.
        self.assertIsNone(self.remaining(allocator, 8185, 1, 8192))
        self.assertIsNone(self.remaining(allocator, 7934, 1, 8192))

    def test_exact_zero_generation_budget_is_not_an_impossible_prompt(self):
        allocator = _TokenBudget(256)
        self.assertEqual(self.remaining(allocator, 256, 16, 513), 0)
        self.assertIsNone(self.remaining(allocator, 257, 0, 513))

    def test_positive_generation_clipping_is_unchanged(self):
        allocator = _TokenBudget(256)
        self.assertEqual(self.remaining(allocator, 256, 1024, 1024), 511)
        self.assertEqual(self.remaining(allocator, 256, 7, 1024), 7)

    def test_published_strict_limit_leaves_one_output_token(self):
        allocator = _TokenBudget(256)
        inclusive = self.limit(allocator, 8185, 8192)
        self.assertEqual(inclusive, 7680)
        published_strict_limit = inclusive + 1
        self.assertEqual(published_strict_limit, 7681)
        self.assertEqual(self.remaining(allocator, inclusive, 1, 8192), 1)
        self.assertIsNone(self.remaining(allocator, published_strict_limit, 1, 8192))
        self.assertLess(inclusive + 1 + allocator.page_size, 8192)

    def test_existing_smaller_input_limit_is_preserved(self):
        allocator = _TokenBudget(256)
        for existing_limit in (1, 128, 2043, 4090, 7681):
            with self.subTest(existing_limit=existing_limit):
                self.assertEqual(
                    self.limit(allocator, existing_limit - 1, 8192) + 1,
                    existing_limit,
                )

    def test_page_rounding_boundaries(self):
        for page_size in (1, 16, 256):
            allocator = _TokenBudget(page_size)
            for capacity in (2 * page_size + 1, 2 * page_size + 2, 8192):
                with self.subTest(page_size=page_size, capacity=capacity):
                    inclusive = self.limit(allocator, capacity, capacity)
                    self.assertGreaterEqual(inclusive, 0)
                    self.assertEqual(
                        self.remaining(allocator, inclusive, 1, capacity), 1
                    )
                    next_result = self.remaining(allocator, inclusive + 1, 1, capacity)
                    self.assertTrue(next_result is None or next_result < 1)

    def test_no_legal_input_domain(self):
        allocator = _TokenBudget(256)
        self.assertEqual(self.limit(allocator, -1, 8192), -1)
        self.assertEqual(self.limit(allocator, 8192, 257), -1)

    def test_search_uses_the_real_allocator_hook_and_arguments(self):
        allocator = _TokenBudget(256)
        allocator.max_new_tokens_for_memory = Mock(
            wraps=allocator.max_new_tokens_for_memory
        )
        inclusive = max_input_tokens_for_memory(
            allocator,
            8185,
            token_capacity=8192,
            sliding_window_size=128,
            chunk_size=2048,
        )
        self.assertEqual(inclusive, 7680)
        self.assertLess(allocator.max_new_tokens_for_memory.call_count, 20)
        for call in allocator.max_new_tokens_for_memory.call_args_list:
            self.assertEqual(call.args[1], 1)
            self.assertEqual(
                call.kwargs,
                {
                    "token_capacity": 8192,
                    "sliding_window_size": 128,
                    "chunk_size": 2048,
                },
            )

    def test_worker_publishes_the_strict_admissible_input_limit(self):
        info = self.worker_info(self.swa_allocator())
        self.assertEqual(info[4:6], (8191, 7681))
        self.assertEqual(info[:4], (8192, 2048, 8, None))
        self.assertEqual(info[6:], (0, "cpu", None, 8, 8192, 8192))

    def test_worker_preserves_context_and_dllm_restrictions(self):
        allocator = self.swa_allocator()
        self.assertEqual(self.worker_info(allocator, context_len=2048)[5], 2042)
        self.assertEqual(self.worker_info(allocator, block_size=32)[5], 7649)

    def test_worker_does_not_apply_token_formula_to_unverified_allocators(self):
        cases = (
            (self.swa_allocator(ring=True), 1),
            (SimpleNamespace(page_size=256), 1),
            (self.swa_allocator(), 2),
        )
        for allocator, shards in cases:
            with self.subTest(allocator=type(allocator).__name__, shards=shards):
                self.assertEqual(
                    self.worker_info(allocator, shards=shards)[4:6], (8191, 8186)
                )

    def test_worker_paged_limits_use_real_inherited_budget(self):
        for page_size, expected_limit in ((16, 8161), (128, 7937), (256, 7681)):
            with self.subTest(page_size=page_size):
                allocator = object.__new__(PagedTokenToKVPoolAllocator)
                allocator.page_size = page_size
                info = self.worker_info(allocator)
                self.assertEqual(info[4:6], (8191, expected_limit))
                self.assertEqual(info[:4], (8192, 2048, 8, None))
                self.assertEqual(info[6:], (0, "cpu", None, 8, 8192, 8192))
                self.assertIs(
                    type(allocator).max_new_tokens_for_memory,
                    BaseTokenToKVPoolAllocator.max_new_tokens_for_memory,
                )
                self.assertEqual(
                    self.remaining(allocator, expected_limit - 1, 1, 8192), 1
                )
                next_result = self.remaining(allocator, expected_limit, 1, 8192)
                self.assertTrue(next_result is None or next_result < 1)

    def test_worker_paged_preserves_context_and_dllm_bounds(self):
        allocator = object.__new__(PagedTokenToKVPoolAllocator)
        allocator.page_size = 256
        self.assertEqual(self.worker_info(allocator, context_len=2048)[5], 2042)
        self.assertEqual(self.worker_info(allocator, block_size=32)[5], 7649)

    def test_worker_page_interleave_uses_shard_adjusted_budget(self):
        for shards, limit in ((2, 15617), (4, 31489)):
            with self.subTest(shards=shards):
                allocator = object.__new__(PageInterleavePoolAllocator)
                allocator.page_size = 256
                allocator.shard_size = shards
                info = self.worker_info(allocator, context_len=131072)
                self.assertEqual(info[4:6], (8192 * shards - 1, limit))
                adjusted_capacity = 8192 * shards - 256 * (shards - 1)
                self.assertEqual(
                    self.remaining(allocator, limit - 1, 1, adjusted_capacity), 1
                )
                self.assertIsNone(
                    self.remaining(allocator, limit, 1, adjusted_capacity)
                )
                self.assertLess(limit - 1 + 1 + 256 * shards, 8192 * shards)

    def test_worker_page_interleave_subclass_keeps_existing_limit(self):
        class SpecializedInterleave(PageInterleavePoolAllocator):
            pass

        allocator = object.__new__(SpecializedInterleave)
        allocator.page_size = 256
        allocator.shard_size = 2
        with patch.object(
            tp_worker,
            "max_input_tokens_for_memory",
            side_effect=AssertionError("unverified specialized interleave search"),
        ):
            self.assertEqual(
                self.worker_info(allocator, context_len=131072)[4:6], (16383, 16378)
            )

    def test_worker_specialized_paged_subclass_keeps_existing_limit(self):
        class SpecializedPaged(PagedTokenToKVPoolAllocator):
            pass

        allocator = object.__new__(SpecializedPaged)
        allocator.page_size = 256
        with patch.object(
            tp_worker,
            "max_input_tokens_for_memory",
            side_effect=AssertionError("unverified specialized input search"),
        ):
            self.assertEqual(self.worker_info(allocator)[4:6], (8191, 8186))

    def test_worker_token_page_one_keeps_existing_limit(self):
        allocator = object.__new__(TokenToKVPoolAllocator)
        allocator.page_size = 1
        with patch.object(
            tp_worker,
            "max_input_tokens_for_memory",
            side_effect=AssertionError("token page-one limit must stay unchanged"),
        ):
            self.assertEqual(self.worker_info(allocator)[4:6], (8191, 8186))
        self.assertEqual(self.remaining(allocator, 8185, 1, 8192), 1)

    def swa_budget(self, allocator, *, window=128):
        tree = SimpleNamespace(
            sliding_window_size=window,
            full_evictable_size=lambda: 0,
            swa_evictable_size=lambda: 0,
        )
        return SWAPrefillBudget(allocator, tree)

    def check_swa_prefill(self, budget, length, *, chunk_size, max_new_tokens=1):
        return budget.check_prefill(
            extend_input_len=length,
            total_tokens=length + max_new_tokens + budget.page_size,
            max_new_tokens=max_new_tokens,
            input_tokens=length,
            swa_host_hit_length=0,
            chunk_limit=chunk_size,
        )

    def test_swa_disabled_chunking_limits_published_input(self):
        allocator = self.swa_allocator(swa_size=4096)
        budget = self.swa_budget(allocator)
        self.assertEqual(
            self.check_swa_prefill(budget, 7680, chunk_size=None), (False, None)
        )
        for disabled in (None, 0, -1):
            with self.subTest(chunk_size=disabled):
                self.assertEqual(
                    self.worker_info(allocator, chunk_size=disabled)[5], 3585
                )
                self.assertIsNone(
                    allocator.max_new_tokens_for_memory(
                        7680,
                        1,
                        token_capacity=8192,
                        sliding_window_size=128,
                        chunk_size=disabled,
                    )
                )
        self.assertEqual(
            self.check_swa_prefill(budget, 3584, chunk_size=None), (True, None)
        )
        self.assertEqual(
            self.check_swa_prefill(budget, 3585, chunk_size=None), (False, None)
        )

    def test_swa_chunking_keeps_long_prompt_and_allows_chunk_shrink(self):
        allocator = self.swa_allocator(swa_size=4096)
        for chunk_size, expected_chunk in ((2048, 2048), (8192, 3584)):
            with self.subTest(chunk_size=chunk_size):
                self.assertEqual(
                    self.worker_info(allocator, chunk_size=chunk_size)[5], 7681
                )
                budget = self.swa_budget(allocator)
                self.assertEqual(
                    self.check_swa_prefill(budget, 7680, chunk_size=chunk_size),
                    (True, expected_chunk),
                )
                self.assertTrue(
                    budget.can_allocate_prefill(
                        paged_input=expected_chunk,
                        extend_input_len=expected_chunk,
                        max_new_tokens=1,
                        chunk_limit=expected_chunk,
                    )
                )

    def test_swa_generation_headroom_is_clipped_and_exact_zero_preserved(self):
        allocator = self.swa_allocator(swa_size=1024)
        self.assertEqual(self.remaining(allocator, 256, 1024, 8192), 511)
        allocator._size_swa = 513
        self.assertEqual(self.remaining(allocator, 256, 16, 8192), 0)
        allocator._size_swa = 512
        self.assertIsNone(self.remaining(allocator, 256, 0, 8192))

    def test_swa_ring_and_specialized_subclass_keep_base_hook(self):
        class SpecializedSWA(SWATokenToKVPoolAllocator):
            pass

        for allocator in (
            self.swa_allocator(ring=True, swa_size=256),
            object.__new__(SpecializedSWA),
        ):
            with self.subTest(allocator=type(allocator).__name__):
                allocator.page_size = 256
                self.assertEqual(self.remaining(allocator, 7680, 1, 8192), 1)

    def test_swa_chunk_too_small_pool_is_not_made_schedulable(self):
        allocator = self.swa_allocator(swa_size=256)
        for chunk_size in (None, 2048, 8192):
            with self.subTest(chunk_size=chunk_size):
                self.assertIsNone(
                    allocator.max_new_tokens_for_memory(
                        7680,
                        1,
                        token_capacity=8192,
                        sliding_window_size=128,
                        chunk_size=chunk_size,
                    )
                )
                self.assertEqual(
                    self.check_swa_prefill(
                        self.swa_budget(allocator), 7680, chunk_size=chunk_size
                    ),
                    (False, None),
                )

    def test_effective_chunking_matches_disabled_and_multimodal_paths(self):
        plain = SimpleNamespace(is_multimodal=False)
        with patch.object(
            tp_worker,
            "get_resolved_model_impl",
            side_effect=AssertionError("ordinary model does not need backend lookup"),
        ):
            for disabled in (None, 0, -1):
                self.assertIsNone(
                    tp_worker.get_effective_chunked_prefill_size(disabled, plain)
                )
            self.assertEqual(
                tp_worker.get_effective_chunked_prefill_size(2048, plain), 2048
            )
        allocator = self.swa_allocator(swa_size=4096)
        with patch.object(
            tp_worker,
            "get_resolved_model_impl",
            return_value=tp_worker.ModelImpl.TRANSFORMERS,
        ):
            self.assertEqual(self.worker_info(allocator, is_multimodal=True)[5], 3585)
        with patch.object(
            tp_worker,
            "get_resolved_model_impl",
            return_value=tp_worker.ModelImpl.SGLANG,
        ):
            self.assertEqual(self.worker_info(allocator, is_multimodal=True)[5], 7681)


if __name__ == "__main__":
    unittest.main()
