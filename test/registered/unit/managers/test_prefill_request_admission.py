"""Prompt limits must reject or truncate before requests enter the queue."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.configs.model_config import ModelImpl  # noqa: E402
from sglang.srt.disaggregation.utils import DisaggregationMode  # noqa: E402
from sglang.srt.managers import scheduler as scheduler_module  # noqa: E402
from sglang.srt.managers import tp_worker  # noqa: E402
from sglang.srt.managers.io_struct import (  # noqa: E402
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.schedule_batch import FINISH_ABORT  # noqa: E402
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.managers.utils import validate_input_length  # noqa: E402
from sglang.srt.mem_cache.allocator.paged import (  # noqa: E402
    PagedTokenToKVPoolAllocator,
)
from sglang.srt.runtime_context import get_context, get_parallel  # noqa: E402
from sglang.srt.sampling.sampling_params import SamplingParams  # noqa: E402
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm  # noqa: E402

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestPrefillRequestAdmission(CustomTestCase):
    def setUp(self):
        self.enterContext(
            get_context().override_server_args(speculative_algorithm=None)
        )
        self.enterContext(get_parallel().override(tp_rank=0, pp_rank=0))

    def scheduler(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.session_controller = {}
        scheduler.enable_session_radix_cache = False
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        scheduler.model_config = SimpleNamespace(hf_eos_token_id=None, vocab_size=128)
        scheduler.metrics_reporter = SimpleNamespace(enable_metrics=False)
        scheduler.kv_events_publisher = SimpleNamespace(enable_kv_cache_events=False)
        scheduler.tokenizer = None
        scheduler.dllm_config = None
        scheduler.spec_algorithm = SpeculativeAlgorithm.NONE
        scheduler.tree_cache = SimpleNamespace(cache_controller=None)
        scheduler.grammar_manager = SimpleNamespace(
            process_req_with_grammar=Mock(return_value=False)
        )
        scheduler._maybe_namespace_elastic_radix_cache = Mock()
        scheduler.validate_dllm_request = Mock(return_value=None)
        scheduler.max_req_input_len = 7681
        scheduler.max_req_len = 8191
        scheduler.max_total_num_tokens = 8192
        scheduler.max_new_tokens_limit = None
        scheduler.page_size = 256
        scheduler.kv_shard_widening = 1
        scheduler.sliding_window_size = None
        scheduler.chunked_prefill_size = None
        allocator = object.__new__(PagedTokenToKVPoolAllocator)
        allocator.page_size = 256
        scheduler.token_to_kv_pool_allocator = allocator
        scheduler.queued = []
        scheduler._add_request_to_queue = scheduler.queued.append
        scheduler.budget_input_lengths = []

        def initialize(req):
            scheduler.budget_input_lengths.append(len(req.origin_input_ids))
            Scheduler.init_req_max_new_tokens(scheduler, req)

        scheduler.init_req_max_new_tokens = initialize
        return scheduler

    def request(self, length, *, embedding=False):
        fields = dict(
            rid="admission-test",
            input_text=None,
            input_ids=array("q", [1]) * length,
            mm_inputs=None,
            token_type_ids=None,
            sampling_params=SamplingParams(max_new_tokens=1),
        )
        if embedding:
            return TokenizedEmbeddingReqInput(**fields)
        return TokenizedGenerateReqInput(
            **fields,
            input_embeds=None,
            return_logprob=False,
            logprob_start_len=-1,
            top_logprobs_num=0,
            token_ids_logprob=None,
            stream=False,
            bootstrap_port=1,
        )

    def dispatch(self, scheduler, request, *, truncate=False, embedding=False):
        with (
            patch.object(
                scheduler_module,
                "get_serving",
                return_value=SimpleNamespace(allow_auto_truncate=truncate),
            ),
            patch.object(
                scheduler_module,
                "get_device",
                return_value=SimpleNamespace(mlx_enable_sampling=False),
            ),
        ):
            handler = (
                scheduler.handle_embedding_request
                if embedding
                else scheduler.handle_generate_request
            )
            handler(request)
        self.assertEqual(len(scheduler.queued), 1)
        return scheduler.queued[0]

    def test_generation_truncates_before_allocator_budget(self):
        for length in (7681, 7936, 8185):
            with self.subTest(length=length):
                scheduler = self.scheduler()
                req = self.dispatch(scheduler, self.request(length), truncate=True)
                self.assertIsNone(req.to_finish)
                self.assertEqual(len(req.origin_input_ids), 7680)
                self.assertEqual(scheduler.budget_input_lengths, [7680])
                self.assertEqual(req.sampling_params.max_new_tokens, 1)

    def test_generation_rejects_oversized_prompt_without_truncation(self):
        scheduler = self.scheduler()
        req = self.dispatch(scheduler, self.request(7936))
        self.assertIsInstance(req.to_finish, FINISH_ABORT)
        self.assertIn("Input length", req.to_finish.message)
        self.assertEqual(scheduler.budget_input_lengths, [1])

    def test_embedding_rejects_oversized_prompt_instead_of_queueing_work(self):
        for length in (7681, 7936):
            with self.subTest(length=length):
                scheduler = self.scheduler()
                req = self.dispatch(
                    scheduler, self.request(length, embedding=True), embedding=True
                )
                self.assertIsInstance(req.to_finish, FINISH_ABORT)
                self.assertIn("Input length", req.to_finish.message)
                self.assertEqual(list(req.origin_input_ids), [0])

    def test_embedding_auto_truncates_strictly_below_limit(self):
        scheduler = self.scheduler()
        req = self.dispatch(
            scheduler,
            self.request(7936, embedding=True),
            truncate=True,
            embedding=True,
        )
        self.assertIsNone(req.to_finish)
        self.assertEqual(len(req.origin_input_ids), 7680)

    def test_legal_generation_and_embedding_inputs_are_unchanged(self):
        for embedding in (False, True):
            with self.subTest(embedding=embedding):
                scheduler = self.scheduler()
                request = self.request(7680, embedding=embedding)
                req = self.dispatch(scheduler, request, embedding=embedding)
                self.assertIsNone(req.to_finish)
                self.assertEqual(req.origin_input_ids, request.input_ids)
                if not embedding:
                    self.assertEqual(scheduler.budget_input_lengths, [7680])

    def test_truncation_rejects_empty_admission_domain(self):
        for limit in (0, 1):
            with self.subTest(limit=limit):
                req = SimpleNamespace(origin_input_ids=array("q", [1, 2]))
                self.assertIsNotNone(validate_input_length(req, limit, True))
                self.assertEqual(list(req.origin_input_ids), [1, 2])

    def test_scheduler_uses_same_effective_chunk_limit_as_publisher(self):
        for configured in (None, 0, -1, 2048):
            for multimodal in (False, True):
                with self.subTest(configured=configured, multimodal=multimodal):
                    scheduler = Scheduler.__new__(Scheduler)
                    scheduler.model_config = SimpleNamespace(is_multimodal=multimodal)
                    schedule = SimpleNamespace(
                        chunked_prefill_size=configured,
                        prefill_decode_interval=0,
                        prefill_decode_interval_min_bs=1,
                        enable_mixed_chunk=True,
                    )
                    with (
                        patch.object(
                            scheduler_module, "get_schedule", return_value=schedule
                        ),
                        patch.object(
                            tp_worker,
                            "get_resolved_model_impl",
                            return_value=ModelImpl.TRANSFORMERS,
                        ),
                        patch.object(
                            scheduler_module,
                            "get_resolved_model_impl",
                            return_value=ModelImpl.TRANSFORMERS,
                        ),
                    ):
                        expected = tp_worker.get_effective_chunked_prefill_size(
                            configured, scheduler.model_config
                        )
                        scheduler.init_chunked_prefill()
                    self.assertEqual(scheduler.chunked_prefill_size, expected)
                    self.assertEqual(scheduler.is_mixed_chunk, expected is not None)
                    self.assertIsNone(scheduler.chunked_req)


if __name__ == "__main__":
    unittest.main()
