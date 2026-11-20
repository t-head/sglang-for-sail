"""Bound Prefill warmup requests by the real serving input limits."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.entrypoints import warmup
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

LEGACY_SIZES = [
    64,
    128,
    192,
    256,
    384,
    512,
    768,
    1024,
    1536,
    2048,
    3072,
    4096,
    6144,
    8192,
    12288,
    16384,
    24576,
    32768,
]


class FakeManager:
    def __init__(self, **limits):
        self.__dict__.update(limits)
        self.requests = []

    async def generate_request(self, request, context):
        length = len(request.input_ids)
        reserved = getattr(self, "num_reserved_tokens", 0) or 0
        if self.context_len is not None:
            assert length + reserved < self.context_len
            assert (
                length + reserved + request.sampling_params["max_new_tokens"]
                <= self.context_len
            )
        if self.max_req_input_len is not None:
            assert length < self.max_req_input_len
        assert request.sampling_params == {"max_new_tokens": 1, "temperature": 0.0}
        assert all(0 <= token < 2**16 for token in request.input_ids)
        self.requests.append(request)
        yield {"done": True}


class TestPrefillWarmupBounds(CustomTestCase):
    def test_large_context_preserves_original_domain(self):
        manager = SimpleNamespace(
            context_len=131072, num_reserved_tokens=8, max_req_input_len=131065
        )
        self.assertEqual(warmup._prefill_warmup_sizes(manager), LEGACY_SIZES)

    def test_none_limits_fall_back_to_original_domain(self):
        self.assertEqual(warmup._prefill_warmup_sizes(SimpleNamespace()), LEGACY_SIZES)
        manager = SimpleNamespace(
            context_len=None, num_reserved_tokens=None, max_req_input_len=None
        )
        self.assertEqual(warmup._prefill_warmup_sizes(manager), LEGACY_SIZES)

    def test_context_reservation_and_strict_boundary(self):
        for context, reserved in [(8192, 0), (2048, 6), (65, 0), (1, 0), (10, 10)]:
            with self.subTest(context=context, reserved=reserved):
                sizes = warmup._prefill_warmup_sizes(
                    SimpleNamespace(
                        context_len=context,
                        num_reserved_tokens=reserved,
                        max_req_input_len=None,
                    )
                )
                self.assertTrue(all(size + reserved < context for size in sizes))
                self.assertEqual(
                    sizes[-1:] or [],
                    [context - reserved - 1] if context - reserved > 1 else [],
                )

    def test_input_limit_is_not_reserved_twice(self):
        manager = SimpleNamespace(
            context_len=8192, num_reserved_tokens=10, max_req_input_len=2000
        )
        sizes = warmup._prefill_warmup_sizes(manager)
        self.assertEqual(sizes[-1], 1999)
        self.assertTrue(all(size < 2000 for size in sizes))
        self.assertEqual(sizes, sorted(set(sizes)))

    def test_small_input_limits(self):
        for limit, expected in [(0, []), (1, []), (2, [1]), (64, [63]), (65, [64])]:
            with self.subTest(limit=limit):
                self.assertEqual(
                    warmup._prefill_warmup_sizes(
                        SimpleNamespace(context_len=None, max_req_input_len=limit)
                    ),
                    expected,
                )

    def test_complete_async_requests_obey_both_limits(self):
        for mode in ("null", "prefill", "decode"):
            manager = FakeManager(
                context_len=8192, num_reserved_tokens=6, max_req_input_len=8187
            )
            with patch.object(
                warmup.tqdm, "tqdm", side_effect=lambda rows, **kwargs: rows
            ):
                asyncio.run(warmup.prefill_shapes(mode, manager))
            self.assertEqual(
                [len(r.input_ids) for r in manager.requests],
                warmup._prefill_warmup_sizes(manager),
            )
            self.assertEqual(len(manager.requests[-1].input_ids), 8185)
            if mode != "null":
                self.assertTrue(
                    all(
                        r.bootstrap_room == 0
                        and r.bootstrap_host == warmup.FAKE_BOOTSTRAP_HOST
                        for r in manager.requests
                    )
                )

    def test_empty_legal_domain_issues_no_request(self):
        manager = FakeManager(
            context_len=1, num_reserved_tokens=0, max_req_input_len=None
        )
        with patch.object(warmup.tqdm, "tqdm", side_effect=lambda rows, **kwargs: rows):
            asyncio.run(warmup.prefill_shapes("null", manager))
        self.assertEqual(manager.requests, [])


if __name__ == "__main__":
    unittest.main()
