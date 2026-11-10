"""CPU protocol contracts for Page DCP capability and destination registration."""

import asyncio
import json
import struct
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlsplit

import numpy as np
import test_register_to_bootstrap as bootstrap_tests
from disagg_test_utils import complete_receiver_setup, make_decode_kv_manager

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import (
    CommonKVBootstrapServer,
    CommonKVManager,
    ParallelInfoState,
)
from sglang.srt.disaggregation.mooncake.conn import (
    KVArgsRegisterInfo,
    MooncakeKVManager,
    MooncakeKVReceiver,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _run_control_message(manager, start_listener, message):
    """Feed one wire message through the real control loop, then stop at recv."""

    manager.server_socket = MagicMock(spec=["recv_multipart"])
    manager.server_socket.recv_multipart.side_effect = [message, SystemExit]
    with patch(
        "sglang.srt.disaggregation.mooncake.conn.threading.Thread", autospec=True
    ) as thread:
        start_listener()
        try:
            thread.call_args.kwargs["target"]()
        except SystemExit:
            pass


class TestPrefillPageCapability(CustomTestCase):
    def setUp(self):
        override = get_context().override_server_args(
            load_balance_method="follow_bootstrap_room", port=30000
        )
        override.install()
        self.addCleanup(override.restore)

    @patch("sglang.srt.disaggregation.common.conn.requests.put")
    def test_page_support_follows_prefill_configuration(self, mock_put):
        mock_put.return_value = MagicMock(status_code=200)
        for layout, backend, speculative, dcp_size, supported in (
            ("token", "mooncake", None, 1, True),
            ("token", "nixl", None, 1, False),
            ("token", "mooncake", "EAGLE", 1, True),
            ("token", "mooncake", None, 2, False),
        ):
            with self.subTest(
                layout=layout,
                backend=backend,
                speculative=speculative,
                dcp_size=dcp_size,
            ):
                override = get_context().override_server_args(
                    dcp_kv_layout=layout,
                    disaggregation_transfer_backend=backend,
                    speculative_algorithm=speculative,
                )
                override.install()
                try:
                    manager = bootstrap_tests.TestRegisterToBootstrap()._make_manager()
                    manager.dcp_size = dcp_size
                    manager.register_to_bootstrap()
                finally:
                    override.restore()
                self.assertIs(
                    mock_put.call_args.kwargs["json"]["supports_dcp_page"], supported
                )
                self.assertNotIn("dcp_kv_layout", mock_put.call_args.kwargs["json"])


class _RouteRequest:
    def __init__(self, *, data=None, query=None):
        self._data = data
        self.query = query or {}

    async def json(self):
        return self._data


class TestBootstrapDcpPageSupport(CustomTestCase):
    def _make_server(self, *, dp_size=1, tp_size=1):
        server = object.__new__(CommonKVBootstrapServer)
        server.attn_tp_size = tp_size
        server.attn_cp_size = 1
        server.dp_size = dp_size
        server.pp_size = 1
        server.page_size = 16
        server.kv_cache_dtype = "auto"
        server.supports_dcp_page = None
        server.follow_bootstrap_room = True
        server.enable_dsa_cache_layer_split = False
        server.prefill_http_port = None
        server.prefill_port_table = {}
        server._registered_count = 0
        server.lock = asyncio.Lock()
        return server

    @staticmethod
    def _registration(*, dp_rank=0, tp_rank=0, supports_dcp_page=None):
        data = {
            "attn_tp_size": 2,
            "attn_tp_rank": tp_rank,
            "attn_cp_size": 1,
            "attn_cp_rank": 0,
            "attn_dp_size": 1,
            "attn_dp_rank": dp_rank,
            "pp_size": 1,
            "pp_rank": 0,
            "system_dp_size": 1,
            "system_dp_rank": 0,
            "rank_ip": "127.0.0.1",
            "rank_port": 12345 + tp_rank,
            "page_size": 16,
            "kv_cache_dtype": "auto",
        }
        if supports_dcp_page is not None:
            data["supports_dcp_page"] = supports_dcp_page
        return data

    @staticmethod
    def _static_query():
        return {
            "prefill_dp_rank": "-1",
            "prefill_cp_rank": "-1",
            "target_tp_rank": "-1",
            "target_pp_rank": "-1",
        }

    def test_route_only_returns_page_support_when_requested(self):
        """Preserve the legacy response schema unless page support is requested."""
        legacy_info = {
            "attn_tp_size": 2,
            "attn_cp_size": 1,
            "dp_size": 1,
            "pp_size": 1,
            "page_size": 16,
            "kv_cache_dtype": "auto",
            "follow_bootstrap_room": True,
            "enable_dsa_cache_layer_split": False,
            "prefill_http_port": None,
            "decode_allocation_policy": "early",
            "target_tp_rank": None,
            "target_tp_ranks": None,
            "target_cp_ranks": None,
            "target_pp_ranks": None,
            "required_dst_info_num": None,
            "required_prefill_response_num": None,
        }
        for supported in (None, False, True):
            with self.subTest(supported=supported):
                server = self._make_server(tp_size=2)
                for rank in range(2):
                    asyncio.run(
                        server._handle_route_put(
                            _RouteRequest(
                                data=self._registration(
                                    tp_rank=rank, supports_dcp_page=supported
                                )
                            )
                        )
                    )

                for query_fields, support_fields in (
                    ({}, {}),
                    ({"include_dcp_page_support": "0"}, {}),
                    (
                        {"include_dcp_page_support": "1"},
                        {"supports_dcp_page": supported},
                    ),
                ):
                    with self.subTest(query_fields=query_fields):
                        response = asyncio.run(
                            server._handle_route_get(
                                _RouteRequest(
                                    query={**self._static_query(), **query_fields}
                                )
                            )
                        )
                        self.assertEqual(response.status, 200)
                        data = json.loads(response.text)
                        self.assertEqual(data, {**legacy_info, **support_fields})
                        if support_fields:
                            self.assertIs(data["supports_dcp_page"], supported)

    @staticmethod
    def _drain_parallel_info(manager, bootstrap_addr):
        # The topology GET runs on an executor; step until it has landed.
        for _ in range(1000):
            state = manager.try_ensure_parallel_info(bootstrap_addr)
            if state != ParallelInfoState.PENDING:
                return state
            time.sleep(0.005)
        raise AssertionError("topology fetch never completed")

    @patch("sglang.srt.disaggregation.common.conn._get_bootstrap_session")
    def test_decode_checks_page_support_before_caching_prefill_info(self, mock_session):
        bootstrap_addr = "127.0.0.1:30000"
        prefill_info = {
            "attn_tp_size": 1,
            "attn_cp_size": 1,
            "dp_size": 1,
            "pp_size": 1,
            "page_size": 16,
            "kv_cache_dtype": "auto",
            "follow_bootstrap_room": True,
        }
        for decode_layout, support_fields, accepted in (
            ("page", {}, False),
            ("page", {"supports_dcp_page": None}, False),
            ("page", {"supports_dcp_page": False}, False),
            ("page", {"supports_dcp_page": True}, True),
            ("token", {}, True),
            ("token", {"supports_dcp_page": False}, True),
        ):
            with self.subTest(
                decode_layout=decode_layout, support_fields=support_fields
            ):
                manager = make_decode_kv_manager(
                    self,
                    kv_args=SimpleNamespace(page_size=16),
                    kv_cache_dtype_str="auto",
                    dcp_kv_layout=decode_layout,
                )
                manager._resolve_rank_mapping = MagicMock()
                # The fetch goes through a per-address Session (#39).
                manager._fetch_parallel_info_payload = MagicMock(
                    return_value={**prefill_info, **support_fields}
                )

                # The topology fetch runs on the manager's executor; the first
                # call submits it and _drain_parallel_info waits for it.
                manager.try_ensure_parallel_info(bootstrap_addr)
                if accepted:
                    self.assertEqual(
                        self._drain_parallel_info(manager, bootstrap_addr),
                        ParallelInfoState.READY,
                    )
                    cached_info = manager.prefill_info_table[bootstrap_addr]
                    self.assertEqual(
                        cached_info.supports_dcp_page,
                        support_fields.get("supports_dcp_page", False),
                    )
                    self.assertEqual(
                        manager.try_ensure_parallel_info(bootstrap_addr),
                        ParallelInfoState.READY,
                    )
                else:
                    with self.assertRaisesRegex(
                        RuntimeError, "DCP page transfer support"
                    ):
                        self._drain_parallel_info(manager, bootstrap_addr)
                    self.assertEqual(manager.prefill_info_table, {})
                    manager._resolve_rank_mapping.assert_not_called()

                manager._fetch_parallel_info_payload.assert_called_once_with(
                    bootstrap_addr
                )

    def test_route_lookup_requests_page_support(self):
        session = MagicMock()
        session.get.return_value = MagicMock(status_code=200, json=lambda: {})
        with patch(
            "sglang.srt.disaggregation.common.conn._get_bootstrap_session",
            return_value=session,
        ):
            CommonKVManager._fetch_parallel_info_payload("127.0.0.1:30000")
        query = parse_qs(urlsplit(session.get.call_args.args[0]).query)
        self.assertEqual(query.get("include_dcp_page_support"), ["1"])

    def test_token_and_page_share_cached_registration_and_room_metadata(self):
        bootstrap_addr = "127.0.0.1:30000"
        override = get_context().override_server_args()
        override.install()
        self.addCleanup(override.restore)
        for layout in ("token", "page"):
            with self.subTest(layout=layout):
                manager = make_decode_kv_manager(
                    self,
                    kv_args=SimpleNamespace(
                        kv_data_ptrs=[4096],
                        aux_data_ptrs=[],
                        state_data_ptrs=[],
                        state_item_lens=[],
                        state_dim_per_tensor=[],
                        state_layer_ids=[],
                        kv_layer_ids=[0],
                        engine_rank=0,
                        kv_item_lens=[64],
                        page_size=16,
                    ),
                    attn_tp_size=2,
                    dcp_size=2,
                    dcp_rank=0,
                    dcp_kv_layout=layout,
                    is_mla_backend=True,
                    enable_staging=False,
                    defer_decode_allocation=False,
                    request_status={},
                    local_ip="127.0.0.1",
                    rank_port=30001,
                    get_session_id=lambda: "session",
                    prefill_info_table={
                        bootstrap_addr: SimpleNamespace(
                            pp_size=1,
                            attn_cp_size=1,
                            target_tp_rank=0,
                            target_tp_ranks=[0],
                            target_cp_ranks=[0],
                            target_pp_ranks=[0],
                            required_dst_info_num=1,
                            required_prefill_response_num=1,
                            decode_allocation_policy="early",
                        )
                    },
                )
                socket = MagicMock()
                with (
                    patch.object(
                        MooncakeKVReceiver,
                        "_get_bootstrap_info_from_server",
                        return_value={"rank_ip": "127.0.0.1", "rank_port": 30000},
                    ) as lookup,
                    patch.object(
                        MooncakeKVReceiver,
                        "_connect_to_bootstrap_server",
                        return_value=(socket, threading.Lock()),
                    ),
                ):
                    for room in (9, 10):
                        receiver = MooncakeKVReceiver(manager, bootstrap_addr, room)
                        receiver.init(0)
                        complete_receiver_setup(receiver)
                        self.assertEqual(
                            manager.request_status[room], KVPoll.WaitingForInput
                        )
                        receiver.send_metadata(
                            np.array([1], dtype=np.int32), aux_index=0
                        )

                lookup.assert_called_once()
                messages = [
                    call.args[0] for call in socket.send_multipart.call_args_list
                ]
                self.assertEqual(
                    [len(message) for message in messages],
                    [21, 10, 10],
                )
                registration = KVArgsRegisterInfo.from_zmq(messages[0])
                self.assertEqual(registration.dcp_kv_layout, layout)
                self.assertEqual(registration.dst_kv_item_lens, [64])
                self.assertEqual(messages[0][19], struct.pack("Q", 64))
                self.assertEqual(messages[0][20], layout.encode("ascii"))
                self.assertEqual(
                    KVArgsRegisterInfo.from_zmq(
                        messages[0][:19] + [layout.encode("ascii")]
                    ).dcp_kv_layout,
                    layout,
                )

    def test_prefill_registers_decode_layout_and_validates_geometry(self):
        manager = object.__new__(MooncakeKVManager)
        manager.dcp_kv_layout = "token"
        manager.disaggregation_mode = DisaggregationMode.PREFILL
        manager.server_args = SimpleNamespace(enable_pd_role_switch=False)
        manager._worker_threads = []
        manager._stopped = False
        manager.kv_args = SimpleNamespace(
            page_size=4, num_draft_entries=0, kv_item_lens=[8]
        )
        manager.dcp_size = 1
        manager.is_mla_backend = True
        manager.is_hybrid_mla_backend = False
        manager.decode_kv_args_table = {}
        manager.transfer_infos = {}
        manager.local_ip = "127.0.0.1"
        manager.rank_port = 30000
        manager._send_multipart_locked = MagicMock()
        manager.session_lock = threading.Lock()
        manager.failed_sessions = {"session"}
        manager.session_failures = {"session": 1}
        manager.state_layout_rejections = {}
        manager.state_strides_validated = set()
        manager._init_dcp_pack_buffers_once = MagicMock()
        message = [
            b"None",
            b"127.0.0.1",
            b"30001",
            b"session",
            struct.pack("Q", 4096),
            b"",
            b"",
            b"0",
            b"2",
            b"12",
            b"",
            b"",
            struct.pack("I", 0),
            b"",
            b"",
            b"",
            b"2",
            b"0",
            b"",
            b"page",
        ]

        with self.assertRaisesRegex(RuntimeError, "KV geometry differs"):
            _run_control_message(manager, manager.start_prefill_thread, message)

        self.assertEqual(manager.decode_kv_args_table, {})
        manager._send_multipart_locked.assert_not_called()

        message[9] = b"8"
        _run_control_message(manager, manager.start_prefill_thread, message)
        self.assertEqual(
            manager.decode_kv_args_table["session"].dcp_kv_layout,
            "page",
        )
        manager._init_dcp_pack_buffers_once.assert_not_called()
        self.assertNotIn("session", manager.failed_sessions)
        self.assertNotIn("session", manager.session_failures)

        token_message = message[:19]
        token_message[3] = b"token-session"
        _run_control_message(manager, manager.start_prefill_thread, token_message)
        manager._init_dcp_pack_buffers_once.assert_called_once_with(
            2, include_draft=True
        )
        self.assertEqual(manager.decode_kv_args_table["session"].dcp_kv_layout, "page")
        self.assertEqual(
            manager.decode_kv_args_table["token-session"].dcp_kv_layout, "token"
        )


if __name__ == "__main__":
    unittest.main()
