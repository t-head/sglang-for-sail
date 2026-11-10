from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST
from sglang.srt.layers.logits_processor import LogitsProcessorOutput, SamplingMaskStatus
from sglang.srt.managers.schedule_batch import FINISH_ABORT, ReqKvInfo
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.base_prefix_cache import (
    CacheRequestHandle,
    CacheRequestOutcome,
)
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@pytest.fixture(autouse=True)
def _rank_zero():
    with get_parallel().override(tp_rank=0):
        yield


class _Req:
    def __init__(self, *, inflight_middle_chunks: int, allocated: bool = True):
        self.rid = "aborted-prefill"
        self.cache_request_handle = CacheRequestHandle(self.rid, 0)
        self.inflight_middle_chunks = inflight_middle_chunks
        self.kv = ReqKvInfo(
            req_pool_idx=1 if allocated else None,
            kv_allocated_len=1 if allocated else 0,
            mamba_pool_idx=object() if allocated else None,
        )
        self.metadata_buffer_index = 7 if allocated else -1
        self.pending_bootstrap = allocated
        self.disagg_kv_sender = Mock()
        self.disagg_kv_sender.poll.return_value = KVPoll.Failed
        self.disagg_kv_sender.is_source_release_safe.return_value = False
        self.bootstrap_room = 123
        self.to_finish = FINISH_ABORT() if allocated else None
        self.finished_reason = None if allocated else FINISH_ABORT()
        self.discard_output_reason = None
        self.return_logprob = False
        self.return_routed_experts = False
        self.return_sampling_mask = False
        self.grammar = None
        self.output_ids = []
        self.origin_input_ids = list(range(100))
        self.extend_end = None
        self.time_stats = SimpleNamespace(
            set_prefill_finished_time=Mock(),
            set_last_chunked_prefill_finish_time=Mock(),
            set_completion_time=Mock(),
            trace_ctx=Mock(),
        )

    def finished(self):
        return self.finished_reason is not None

    def update_finish_state(self):
        self.finished_reason = self.to_finish
        self.to_finish = None


class _Scheduler(SchedulerDisaggregationPrefillMixin):
    def __init__(self):
        self.batch_result_processor = SimpleNamespace(
            snapshot_auxiliary_output_starts=Mock(return_value=[]),
            move_logprobs_to_cpu=Mock(),
            consume_auxiliary_output=Mock(),
        )
        self.spec_algorithm = SimpleNamespace(is_eagle=lambda: False)
        self.tree_cache = Mock()
        self.disagg_prefill_inflight_queue = []
        self.disagg_prefill_pending_chunk_rids = {"aborted-prefill"}
        self.send_kv_chunk = Mock()
        self.output_streamer = Mock()
        self.metrics_reporter = SimpleNamespace(report_prefill_stats=Mock())
        self.metrics_reporter.enable_metrics = False
        self.maybe_send_health_check_signal = Mock()
        self.req_to_metadata_buffer_idx_allocator = Mock()
        self.enable_hicache_storage = True
        self.chunked_req = None
        self.enable_overlap = False
        self.scheduler_stage_metrics = None
        self.attn_cp_cpu_group = None
        self.attn_tp_cpu_group = None
        self.ps = SimpleNamespace(tp_rank=0, pp_rank=0)

    def _release_aborted_request(self, req):
        if self.enable_hicache_storage:
            self.tree_cache.finish(req.cache_request_handle, CacheRequestOutcome.ABORT)


def _poll_queue(scheduler):
    with (
        patch(
            "sglang.srt.disaggregation.prefill.poll_and_all_reduce_attn_cp_tp_group",
            side_effect=lambda senders, *_: [sender.poll() for sender in senders],
        ),
        patch("sglang.srt.disaggregation.prefill.dist.all_reduce"),
    ):
        return scheduler.process_disagg_prefill_inflight_queue()


def _drain(scheduler, req):
    req.disagg_kv_sender.is_source_release_safe.return_value = True
    return _poll_queue(scheduler)


def _batch(req):
    return SimpleNamespace(
        reqs=[req],
        spec_info=None,
        prefill_stats=None,
        dp_cooperation_info=None,
    )


def _result():
    return GenerationBatchResult(next_token_ids=torch.tensor([11]))


def _free_req(req, _tree_cache, *, checkpoint):
    # Callers assert the expected checkpoint flag on the mock itself.
    req.kv.req_pool_idx = None
    req.kv.mark_kv_released()
    req.kv.mamba_pool_idx = None


@patch("sglang.srt.disaggregation.prefill.release_kv_cache", side_effect=_free_req)
@patch("sglang.srt.disaggregation.prefill.checkpoint_kv_cache")
def test_aborted_final_result_releases_hybrid_cache(
    checkpoint_kv_cache, release_kv_cache
):
    scheduler = _Scheduler()
    req = _Req(inflight_middle_chunks=0)

    scheduler.process_batch_result_disagg_prefill(_batch(req), _result())

    # Resolving the final GPU result does not prove that RDMA finished reading.
    assert scheduler.disagg_prefill_inflight_queue == [req]
    release_kv_cache.assert_not_called()
    scheduler.req_to_metadata_buffer_idx_allocator.free.assert_not_called()
    assert _poll_queue(scheduler) == []
    release_kv_cache.assert_not_called()
    _drain(scheduler, req)

    release_kv_cache.assert_called_once_with(
        req, scheduler.tree_cache, checkpoint=False
    )
    checkpoint_kv_cache.assert_not_called()
    req.disagg_kv_sender.abort.assert_called_with()
    scheduler.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(7)
    scheduler.tree_cache.finish.assert_called_once_with(
        req.cache_request_handle, CacheRequestOutcome.ABORT
    )
    scheduler.output_streamer.stream_output.assert_any_call([req], False, None)
    assert scheduler.output_streamer.stream_output.call_count == 2
    scheduler.send_kv_chunk.assert_not_called()
    assert req.output_ids == []
    assert req.finished()
    assert req.metadata_buffer_index == -1
    assert req.rid not in scheduler.disagg_prefill_pending_chunk_rids


@patch("sglang.srt.disaggregation.prefill.release_kv_cache", side_effect=_free_req)
def test_aborted_middle_result_releases_after_last_chunk(release_kv_cache):
    scheduler = _Scheduler()
    req = _Req(inflight_middle_chunks=1)
    req.extend_end = 50

    scheduler.process_batch_result_disagg_prefill(_batch(req), _result())

    assert req.inflight_middle_chunks == 0
    release_kv_cache.assert_not_called()
    _drain(scheduler, req)
    release_kv_cache.assert_called_once_with(
        req, scheduler.tree_cache, checkpoint=False
    )
    scheduler.output_streamer.stream_output.assert_called_once_with([req], False, None)


@patch("sglang.srt.disaggregation.prefill.release_kv_cache", side_effect=_free_req)
def test_aborted_middle_result_waits_for_inflight_chunk(release_kv_cache):
    scheduler = _Scheduler()
    req = _Req(inflight_middle_chunks=1)
    req.extend_end = len(req.origin_input_ids)

    scheduler.process_batch_result_disagg_prefill(_batch(req), _result())

    release_kv_cache.assert_not_called()
    scheduler.output_streamer.stream_output.assert_not_called()

    scheduler.process_batch_result_disagg_prefill(_batch(req), _result())

    release_kv_cache.assert_not_called()
    _drain(scheduler, req)
    release_kv_cache.assert_called_once_with(
        req, scheduler.tree_cache, checkpoint=False
    )
    scheduler.output_streamer.stream_output.assert_called_once_with([req], False, None)


@patch("sglang.srt.disaggregation.prefill.release_kv_cache")
def test_delayed_result_ignores_already_retired_request(release_kv_cache):
    scheduler = _Scheduler()
    req = _Req(inflight_middle_chunks=0, allocated=False)

    scheduler.process_batch_result_disagg_prefill(_batch(req), _result())

    release_kv_cache.assert_not_called()
    req.disagg_kv_sender.abort.assert_not_called()
    scheduler.output_streamer.stream_output.assert_not_called()


@patch("sglang.srt.disaggregation.prefill.release_kv_cache", side_effect=_free_req)
def test_sender_abort_failure_retains_buffers_until_retry_and_drain(release_kv_cache):
    scheduler = _Scheduler()
    req = _Req(inflight_middle_chunks=0)
    req.disagg_kv_sender.abort.side_effect = RuntimeError("transport is down")

    scheduler.process_batch_result_disagg_prefill(_batch(req), _result())

    release_kv_cache.assert_not_called()
    scheduler.req_to_metadata_buffer_idx_allocator.free.assert_not_called()
    # Even an optimistic safe result must not override failed cancellation.
    assert _drain(scheduler, req) == []
    release_kv_cache.assert_not_called()
    req.disagg_kv_sender.abort.side_effect = None
    _drain(scheduler, req)
    release_kv_cache.assert_called_once_with(
        req, scheduler.tree_cache, checkpoint=False
    )
    scheduler.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(7)
    scheduler.output_streamer.stream_output.assert_called_with([req], False, None)
    assert req.finished()


@patch("sglang.srt.disaggregation.prefill.release_kv_cache", side_effect=_free_req)
@patch("sglang.srt.disaggregation.prefill.checkpoint_kv_cache")
def test_grammar_rejection_retires_prefill_before_transfer(
    checkpoint_kv_cache, release_kv_cache
):
    scheduler = _Scheduler()
    req = _Req(inflight_middle_chunks=0)
    req.to_finish = None
    req.grammar = Mock()
    req.grammar.accept_token.side_effect = ValueError("invalid token")

    scheduler.process_batch_result_disagg_prefill(_batch(req), _result())

    req.grammar.accept_token.assert_called_once_with(11)
    assert req.grammar.finished
    assert req.finished()
    release_kv_cache.assert_not_called()
    _drain(scheduler, req)
    release_kv_cache.assert_called_once_with(
        req, scheduler.tree_cache, checkpoint=False
    )
    checkpoint_kv_cache.assert_not_called()
    scheduler.send_kv_chunk.assert_not_called()
    scheduler.output_streamer.stream_output.assert_called_once_with([req], False, None)
    assert scheduler.disagg_prefill_inflight_queue == []


def test_aborted_result_releases_mamba_allocated_before_kv():
    scheduler = _Scheduler()
    scheduler.enable_hicache_storage = False
    scheduler.tree_cache.supports_mamba.return_value = True
    scheduler.tree_cache.req_to_token_pool.mamba_allocator.free = Mock()
    req = _Req(inflight_middle_chunks=0, allocated=False)
    req.kv.mamba_pool_idx = torch.tensor([3])
    req.to_finish = FINISH_ABORT()
    req.finished_reason = None

    scheduler.process_batch_result_disagg_prefill(_batch(req), _result())

    scheduler.tree_cache.req_to_token_pool.mamba_allocator.free.assert_not_called()
    _drain(scheduler, req)
    scheduler.tree_cache.req_to_token_pool.mamba_allocator.free.assert_called_once()
    assert req.kv.mamba_pool_idx is None
    scheduler.output_streamer.stream_output.assert_called_once_with([req], False, None)


@patch("sglang.srt.disaggregation.prefill.release_kv_cache", side_effect=_free_req)
def test_failed_kv_load_is_aborted_instead_of_sent_to_decode(release_kv_cache):
    scheduler = _Scheduler()
    req = _Req(inflight_middle_chunks=0)
    req.to_finish = None
    req.discard_output_reason = FINISH_ABORT("External KV cache load failed", 503)
    result = GenerationBatchResult(
        next_token_ids=torch.tensor([11]),
        logits_output=LogitsProcessorOutput(next_token_logits=None),
    )

    scheduler.process_batch_result_disagg_prefill(_batch(req), result)

    assert req.finished_reason.status_code == 503
    assert req.output_ids == []
    # Source buffers stay owned until every reader has drained.
    release_kv_cache.assert_not_called()
    _drain(scheduler, req)
    release_kv_cache.assert_called_once_with(
        req, scheduler.tree_cache, checkpoint=False
    )
    # Cancellation is re-sent at the drain poll; abort() is idempotent.
    req.disagg_kv_sender.abort.assert_called_with()
    scheduler.send_kv_chunk.assert_not_called()


@pytest.mark.parametrize("transport_error", [False, True])
@pytest.mark.parametrize(
    "status,http_status,err_type",
    [
        (SamplingMaskStatus.OVERFLOW, 400, "BadRequestError"),
        (SamplingMaskStatus.INVALID, 500, "InternalServerError"),
    ],
)
@patch("sglang.srt.disaggregation.prefill.release_kv_cache", side_effect=_free_req)
def test_sampling_mask_abort_preserves_error_and_releases_once(
    release_kv_cache, status, http_status, err_type, transport_error
):
    """Retain ownership while cancelling, preserving the original API error."""
    scheduler = _Scheduler()
    scheduler.batch_result_processor.get_sampling_mask_finish_reason = lambda **kwargs: (
        SchedulerBatchResultProcessor.get_sampling_mask_finish_reason(None, **kwargs)
    )
    req = _Req(inflight_middle_chunks=0)
    req.to_finish = None
    req.return_sampling_mask = True
    req.time_stats.trace_ctx = Mock()
    if transport_error:
        req.disagg_kv_sender.abort.side_effect = RuntimeError("transport is down")
    result = GenerationBatchResult(
        next_token_ids=torch.tensor([11]),
        logits_output=LogitsProcessorOutput(
            next_token_logits=None, next_token_sampling_mask_status=[status]
        ),
    )

    with get_context().override_server_args(sampling_mask_max_tokens=64):
        scheduler.process_batch_result_disagg_prefill(_batch(req), result)
        scheduler.process_batch_result_disagg_prefill(_batch(req), result)

    release_kv_cache.assert_not_called()
    assert scheduler.disagg_prefill_inflight_queue == [req]
    req.disagg_kv_sender.abort.side_effect = None
    _drain(scheduler, req)

    assert req.finished_reason.status_code == http_status
    assert req.finished_reason.err_type == err_type
    assert req.output_ids == []
    assert not req.kv.holds_kv and not req.kv.holds_mamba
    assert req.metadata_buffer_index == -1
    assert not req.pending_bootstrap
    assert req.rid not in scheduler.disagg_prefill_pending_chunk_rids
    release_kv_cache.assert_called_once_with(
        req, scheduler.tree_cache, checkpoint=False
    )
    assert req.disagg_kv_sender.abort.call_count == 2
    scheduler.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(7)
    scheduler.tree_cache.finish.assert_called_once_with(
        req.cache_request_handle, CacheRequestOutcome.ABORT
    )
    scheduler.output_streamer.stream_output.assert_called_once_with([req], False, None)
    scheduler.send_kv_chunk.assert_not_called()


class TestPrefillCompleteResult(CustomTestCase):
    def setUp(self):
        super().setUp()
        override = get_context().override_server_args(
            disaggregation_decode_allocation_policy="prefill_complete"
        )
        override.install()
        self.addCleanup(override.restore)

    @patch("sglang.srt.disaggregation.prefill.checkpoint_kv_cache")
    def test_ready_observes_accepted_token_and_speculative_sidecars(self, cache):
        scheduler = _Scheduler()
        scheduler.spec_algorithm = SimpleNamespace(is_eagle=lambda: True)
        req = _Req(inflight_middle_chunks=0)
        req.to_finish = None
        req.time_stats.set_prefill_transfer_queue_entry_time = Mock()
        draft = SimpleNamespace(
            hidden_states=torch.tensor([[1.0, 2.0]]),
            topk_p=torch.tensor([[0.75, 0.25]]),
            topk_index=torch.tensor([[3, 4]]),
            dsa_topk_indices=torch.tensor([[5, 6]]),
        )
        batch = _batch(req)
        batch.spec_info = draft
        result = GenerationBatchResult(
            next_token_ids=torch.tensor([11]), next_draft_input=draft
        )
        observed = []

        def ready():
            observed.append(
                (
                    list(req.output_ids),
                    req.hidden_states_tensor.tolist(),
                    req.output_dsa_topk_indices.tolist(),
                    req in scheduler.disagg_prefill_inflight_queue,
                )
            )

        req.disagg_kv_sender.mark_prefill_complete.side_effect = ready
        scheduler.process_batch_result_disagg_prefill(batch, result)
        self.assertEqual(observed, [([11], [1.0, 2.0], [5, 6], True)])
        self.assertTrue(req.pending_bootstrap)
        scheduler.send_kv_chunk.assert_not_called()

    @patch("sglang.srt.disaggregation.prefill.release_kv_cache", side_effect=_free_req)
    def test_aborted_final_result_never_publishes_ready(self, release):
        scheduler = _Scheduler()
        req = _Req(inflight_middle_chunks=0)
        scheduler.process_batch_result_disagg_prefill(_batch(req), _result())
        self.assertTrue(req.finished())
        # Source buffers stay owned until every reader has drained.
        self.assertTrue(req.kv.holds_kv)
        _drain(scheduler, req)
        self.assertFalse(req.kv.holds_kv)
        self.assertEqual(scheduler.disagg_prefill_inflight_queue, [])
        req.disagg_kv_sender.mark_prefill_complete.assert_not_called()

    @patch("sglang.srt.disaggregation.prefill.should_force_retry", return_value=True)
    def test_discarded_attempt_never_publishes_ready(self, force_retry):
        scheduler = _Scheduler()
        req = _Req(inflight_middle_chunks=0)
        req.to_finish = None
        retried = []
        scheduler.optimistic_release_and_requeue = lambda r: retried.append(r)
        scheduler.process_batch_result_disagg_prefill(_batch(req), _result())
        self.assertEqual(retried, [req])
        self.assertEqual(req.output_ids, [])
        self.assertEqual(scheduler.disagg_prefill_inflight_queue, [])
        req.disagg_kv_sender.mark_prefill_complete.assert_not_called()


class TestPrefillSourceDrain(CustomTestCase):
    @patch("sglang.srt.disaggregation.prefill.release_kv_cache", side_effect=_free_req)
    def test_peer_failure_does_not_free_a_local_transfer(self, release):
        scheduler = _Scheduler()
        req = _Req(inflight_middle_chunks=0)
        req.to_finish = None
        req.pending_bootstrap = False
        scheduler.disagg_prefill_inflight_queue = [req]

        # The global MIN reports Failed while this rank is still reading.
        self.assertEqual(_poll_queue(scheduler), [])
        self.assertTrue(req.kv.holds_kv)
        self.assertTrue(req.kv.holds_mamba)
        self.assertEqual(req.metadata_buffer_index, 7)
        req.disagg_kv_sender.failure_exception.assert_not_called()
        req.disagg_kv_sender.clear.assert_not_called()
        release.assert_not_called()
        req.disagg_kv_sender.abort.assert_called_once()

        self.assertEqual(_drain(scheduler, req), [req])
        self.assertFalse(req.kv.holds_kv)
        self.assertFalse(req.kv.holds_mamba)
        self.assertEqual(req.metadata_buffer_index, -1)
        self.assertEqual(scheduler.disagg_prefill_inflight_queue, [])
        self.assertEqual(_poll_queue(scheduler), [])
        # A transport failure, not a cancellation, keeps its computed prefix.
        release.assert_called_once_with(req, scheduler.tree_cache, checkpoint=True)
        req.disagg_kv_sender.clear.assert_called_once()

    @patch("sglang.srt.disaggregation.prefill.release_kv_cache", side_effect=_free_req)
    def test_local_drain_waits_for_other_source_ranks(self, release):
        scheduler = _Scheduler()
        req = _Req(inflight_middle_chunks=0)
        scheduler.queue_aborted_prefill_transfer(req)
        req.disagg_kv_sender.is_source_release_safe.return_value = True

        with (
            patch(
                "sglang.srt.disaggregation.prefill.poll_and_all_reduce_attn_cp_tp_group",
                return_value=[KVPoll.Failed],
            ),
            patch(
                "sglang.srt.disaggregation.prefill.dist.all_reduce",
                side_effect=lambda values, **_: values.zero_(),
            ),
        ):
            self.assertEqual(scheduler.process_disagg_prefill_inflight_queue(), [])
        release.assert_not_called()
        self.assertEqual(req.metadata_buffer_index, 7)
        self.assertEqual(_drain(scheduler, req), [req])
        release.assert_called_once()

    @patch("sglang.srt.disaggregation.prefill.release_kv_cache", side_effect=_free_req)
    def test_terminal_transfer_waits_for_pending_gpu_result(self, release):
        scheduler = _Scheduler()
        req = _Req(inflight_middle_chunks=0)
        scheduler.queue_aborted_prefill_transfer(req)
        scheduler.enable_overlap = True
        scheduler.result_queue = [(_batch(req), _result())]

        self.assertEqual(_drain(scheduler, req), [])
        release.assert_not_called()
        scheduler.result_queue.clear()
        self.assertEqual(_drain(scheduler, req), [req])
        release.assert_called_once()

    @patch("sglang.srt.disaggregation.prefill.release_kv_cache", side_effect=_free_req)
    def test_cached_success_does_not_bypass_source_drain(self, release):
        scheduler = _Scheduler()
        req = _Req(inflight_middle_chunks=0)
        req.to_finish = None
        req.pending_bootstrap = False
        req.bootstrap_host = FAKE_BOOTSTRAP_HOST
        req.time_stats.set_prefill_kv_transfer_finish_time = Mock()
        req.disagg_kv_sender.poll.return_value = KVPoll.Success
        scheduler.disagg_prefill_inflight_queue = [req]

        self.assertEqual(_poll_queue(scheduler), [])
        release.assert_not_called()
        req.disagg_kv_sender.clear.assert_not_called()
        self.assertEqual(_drain(scheduler, req), [req])
        release.assert_called_once()
        req.disagg_kv_sender.clear.assert_called_once()

    @patch("sglang.srt.disaggregation.prefill.release_kv_cache", side_effect=_free_req)
    def test_bootstrap_failure_preserves_metadata_until_drain(self, release):
        scheduler = _Scheduler()
        req = _Req(inflight_middle_chunks=0)
        req.to_finish = None

        scheduler.handle_bootstrap_failure(req)
        self.assertEqual(scheduler.disagg_prefill_inflight_queue, [req])
        self.assertFalse(req.pending_bootstrap)
        self.assertEqual(req.metadata_buffer_index, 7)
        req.disagg_kv_sender.failure_exception.assert_not_called()
        release.assert_not_called()
        self.assertEqual(_drain(scheduler, req), [req])
        self.assertEqual(req.metadata_buffer_index, -1)
        release.assert_called_once()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
