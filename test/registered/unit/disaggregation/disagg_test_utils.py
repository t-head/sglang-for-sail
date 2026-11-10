"""Small CPU-only transport boundary helpers shared by disaggregation tests."""

import threading
from collections import defaultdict
from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(
    est_time=0,
    suite="base-a-test-cpu",
    disabled="helper module - exported CPU transport boundary, not a test",
)


class CopyTransport:
    """Execute Mooncake transfer descriptors against registered CPU buffers."""

    def __init__(self, sources, destinations):
        self.sources = sources
        self.destinations = destinations
        self.bytes_sent = 0

    @staticmethod
    def _region(buffers, address, size):
        for buffer in buffers:
            raw = np.asarray(buffer).view(np.uint8).reshape(-1)
            offset = address - raw.ctypes.data
            if 0 <= offset and offset + size <= raw.nbytes:
                return raw[offset : offset + size]
        raise AssertionError("transfer descriptor exceeds a registered CPU buffer")

    def __call__(self, _session, blocks):
        for source, destination, size in blocks:
            self._region(self.destinations, destination, size)[:] = self._region(
                self.sources, source, size
            )
            self.bytes_sent += size
        return 0


def make_decode_kv_manager(test_case, **overrides):
    """A decode-mode CommonKVManager built without ``__init__`` (which needs
    torch.distributed, ZMQ and a bootstrap server); the handshake bookkeeping
    comes from the production initializer, so it cannot drift from it."""
    from sglang.srt.disaggregation.common.conn import CommonKVManager

    mgr = CommonKVManager.__new__(CommonKVManager)
    mgr.connection_lock = threading.Lock()
    mgr.connection_pool = {}
    mgr.prefill_info_table = {}
    mgr.addr_to_rooms_tracker = defaultdict(set)
    mgr.request_status = {}
    mgr.required_prefill_response_num_table = {}
    mgr.prefill_response_tracker = defaultdict(set)
    mgr.failure_records = {}
    mgr.failure_lock = threading.Lock()
    mgr._deferred_abort_ack_tracker = {}
    mgr._deferred_abort_generation = 0
    # update_status resets these when a room's lifecycle (re)starts.
    mgr._deferred_ack_targets = {}
    mgr._deferred_ack_poisoned_rooms = set()
    mgr._staging_outstanding = defaultdict(int)
    mgr.enable_deferred_decode_kv_release = False
    mgr.defer_decode_allocation = False
    mgr.deferred_bootstrap = None
    mgr.enable_staging = False
    mgr.is_mla_backend = False
    mgr.is_hybrid_mla_backend = False
    mgr.attn_tp_size = 1
    mgr.dcp_size = 1
    mgr.dcp_kv_layout = "token"
    mgr.kv_cache_dtype_str = "bfloat16"
    mgr.kv_args = SimpleNamespace(page_size=64)
    mgr.local_ip = "127.0.0.1"
    mgr.rank_port = 17000
    mgr.waiting_timeout = 300
    for name, value in overrides.items():
        setattr(mgr, name, value)
    mgr._init_decode_handshake_state()
    test_case.addCleanup(mgr._bootstrap_executor.shutdown, wait=True)
    test_case.addCleanup(mgr._parallel_info_executor.shutdown, wait=True)
    return mgr


def prefill_info_stub(**overrides):
    """What ``try_ensure_parallel_info`` caches for a 1-rank prefill."""
    info = dict(
        attn_tp_size=1,
        attn_cp_size=1,
        pp_size=1,
        target_tp_rank=0,
        target_tp_ranks=[0],
        target_cp_ranks=[0],
        target_pp_ranks=[0],
        required_dst_info_num=1,
        required_prefill_response_num=1,
        decode_allocation_policy="early",
    )
    info.update(overrides)
    return SimpleNamespace(**info)


def complete_receiver_setup(receiver, timeout=5.0):
    """Wait out the receiver's in-flight bootstrap jobs, then let it publish
    the handshake on this (the scheduler's) thread."""
    for _, part in receiver._setup_parts or ():
        if isinstance(part, Future):
            part.result(timeout=timeout)
    receiver._advance_setup()
