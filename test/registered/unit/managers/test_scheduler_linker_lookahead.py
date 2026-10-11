"""An in-flight PP Linker lookup inside the window skips a request instead of ending admission."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.schedule_policy import AddReqResult
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.runtime_context import get_context

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class _Req:
    def __init__(self, rid):
        self.rid = rid
        self.kv = SimpleNamespace(holds_kv=False)
        self.beam_group = None
        self.token_indices_to_pool = None
        self.inflight_middle_chunks = 0

    def init_next_round_input(self, tree_cache=None):
        pass


class _Adder:
    def __init__(self, *args, **kwargs):
        self.can_run_list = []
        self.preempt_list = []
        self.new_chunked_req = None
        self.rem_chunk_tokens = 0

    def budget_state(self):
        return AddReqResult.CONTINUE

    def add_one_req(self, req, has_chunked_req, truncation_align_size):
        self.can_run_list.append(req)
        return AddReqResult.CONTINUE


class _Linker:
    """Lookups for "1" and "2" are still in flight; the others are ready."""

    def __init__(self):
        self.started = []
        self.pp_coordinator = object()

    def start_pp_lookups(self, rids):
        self.started.extend(rids)

    def check_pp_ready(self, rid):
        return rid not in ("1", "2")


class _BatchBuilt(Exception):
    pass


class TestLinkerLookahead(CustomTestCase):
    def _admit(self, waiting, micro_batch_size, s=None):
        s = s or self._scheduler()
        linker = s.tree_cache.linker
        s.waiting_queue = waiting

        def capture(can_run_list, *args, **kwargs):
            raise _BatchBuilt(list(can_run_list))

        with (
            get_context().override_server_args(
                pp_max_micro_batch_size=micro_batch_size
            ),
            patch.object(ScheduleBatch, "init_new", side_effect=capture),
            patch("sglang.srt.managers.scheduler.set_time_batch"),
        ):
            with self.assertRaises(_BatchBuilt) as built:
                s._get_new_batch_prefill_raw(
                    prefill_delayer_single_pass=None,
                    running_batch=ScheduleBatch(reqs=[]),
                )
        return built.exception.args[0], linker.started

    def _scheduler(self):
        linker = _Linker()
        s = Scheduler.__new__(Scheduler)
        s.grammar_manager = SimpleNamespace(has_waiting_grammars=lambda: False)
        s.enable_priority_preemption = False
        s.is_hybrid_swa = False
        s.chunked_req = None
        s.linker_lookup_window = 0
        s.min_free_slots_delayer = None
        s.policy = Mock()
        s.processed_tokens_counter = 0
        s.chunked_prefill_size = 8192
        s.pp_min_chunked_prefill_size = None
        s.dynamic_chunk_sizer = None
        s.tp_worker = SimpleNamespace(
            model_runner=SimpleNamespace(attn_backend=SimpleNamespace())
        )
        s.get_prefill_adder_class = lambda: _Adder
        s.page_size = 1
        s.tree_cache = SimpleNamespace(
            buffer_pipeline=None, storage_prefetch_retries=None, linker=linker
        )
        s.token_to_kv_pool_allocator = None
        s.new_token_ratio_tracker = SimpleNamespace(current=0.0)
        s.max_prefill_tokens = 1 << 20
        s.is_mixed_chunk = False
        s.priority_scheduling_preemption_threshold = 0
        s.max_prefill_bs = 0
        s.max_running_requests = 8
        s.dllm_config = None
        s.enable_lora = False
        s.disaggregation_mode = DisaggregationMode.PREFILL
        s.disagg_prefill_bootstrap_queue = None
        s.enable_hicache_storage = False
        s.enable_lmcache = False
        s.truncation_align_size = None
        s.model_config = None
        s.enable_overlap = False
        s.spec_algorithm = None
        s.req_to_token_pool = SimpleNamespace(available_size=lambda: 8)
        s.beam_coordinator = SimpleNamespace(pending_member_rows=lambda batch: 0)
        return s

    def test_inflight_lookups_inside_the_window_are_skipped(self):
        waiting = [_Req(str(i)) for i in range(4)]
        admitted, started = self._admit(waiting, micro_batch_size=4)
        self.assertEqual(admitted, [waiting[0], waiting[3]])
        self.assertEqual(started, ["0", "1", "2", "3"])

    def test_window_is_logged_only_when_it_changes(self):
        s = self._scheduler()
        with self.assertLogs("sglang.srt.managers.scheduler", level="INFO") as logs:
            for micro_batch_size in (4, 4, 2):
                waiting = [_Req(str(i)) for i in range(4)]
                self._admit(waiting, micro_batch_size, s)
        lines = [line for line in logs.output if "lookup window" in line]
        self.assertEqual(len(lines), 2)
        self.assertIn("window: 4 requests", lines[0])
        self.assertIn("window: 2 requests", lines[1])
        self.assertEqual(s.linker_lookup_window, 2)

    def test_requests_beyond_the_window_end_admission(self):
        waiting = [_Req(str(i)) for i in range(4)]
        admitted, started = self._admit(waiting, micro_batch_size=2)
        self.assertEqual(admitted, [waiting[0]])
        self.assertEqual(started, ["0", "1"])


if __name__ == "__main__":
    unittest.main()
