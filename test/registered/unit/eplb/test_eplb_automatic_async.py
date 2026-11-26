"""Automatic async selection must preserve optional dependencies and rank agreement."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

import argparse
import tempfile
import unittest
from contextlib import ExitStack
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.eplb import eplb_manager as manager_module
from sglang.srt.eplb.eplb_manager import EPLBManager
from sglang.test.test_utils import CustomTestCase


def _runtime(
    *,
    enabled=True,
    elastic=None,
    ep=2,
    tp=2,
    pp=1,
    device="cuda",
    host_mirror=False,
    chunk=None,
    dispatch="static",
):
    stack = ExitStack()
    stack.enter_context(
        patch.object(
            manager_module,
            "get_exec",
            return_value=SimpleNamespace(
                moe=SimpleNamespace(
                    enable_eplb=enabled,
                    enable_eplb_async=host_mirror,
                    ep_dispatch_algorithm=dispatch,
                    eplb_rebalance_layers_per_chunk=chunk,
                    elastic_ep_backend=elastic,
                )
            ),
        )
    )
    stack.enter_context(
        patch.object(
            manager_module,
            "get_parallel",
            return_value=SimpleNamespace(ep_size=ep, tp_size=tp, pp_size=pp),
        )
    )
    stack.enter_context(
        patch.object(
            manager_module, "get_device", return_value=SimpleNamespace(device=device)
        )
    )
    return stack


def _select_on_rank(rank, init_file):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        with (
            _runtime(),
            patch(
                "sglang.srt.distributed.parallel_state.get_moe_ep_group",
                return_value=SimpleNamespace(cpu_group=dist.group.WORLD),
            ),
            patch(
                "sglang.srt.eplb.eplb_async_migrator.AsyncExpertMigrator"
            ) as migrator,
        ):
            for missing_rank in (None, 0, 1):
                engine = None if rank == missing_rank else object()
                manager = EPLBManager.__new__(EPLBManager)
                manager._async_migrator = None
                manager._rebalance_layers_per_chunk = 1
                migrator.reset_mock()
                with patch(
                    "sglang.srt.eplb.eplb_copy_engine.ExpertCopyEngine.try_create",
                    return_value=engine,
                ):
                    manager._init_async_migrator()
                assert (manager._async_migrator is not None) == (missing_rank is None)
                assert migrator.call_count == int(missing_rank is None)
        from sglang.srt.eplb.eplb_copy_engine import (
            PeerStagingUnavailable,
            _agree_on_staging,
        )

        with unittest.TestCase().assertRaisesRegex(PeerStagingUnavailable, "EP rank 1"):
            with _agree_on_staging(dist.group.WORLD):
                if rank == 1:
                    raise RuntimeError("peer access denied")
    finally:
        dist.destroy_process_group()


class TestAutomaticAsyncEPLB(CustomTestCase):
    def test_cli_needs_no_async_flag(self):
        from sglang.srt.server_args import ServerArgs

        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        args = parser.parse_args(
            [
                "--model-path",
                "unused",
                "--enable-eplb",
                "--eplb-rebalance-num-iterations",
                "1",
                "--expert-distribution-recorder-buffer-size",
                "1",
            ]
        )
        self.assertTrue(args.enable_eplb)
        self.assertEqual(args.eplb_rebalance_num_iterations, 1)
        self.assertFalse(args.enable_eplb_async)

    def test_supported_config(self):
        with _runtime():
            self.assertIsNone(manager_module.eplb_async_unavailable_reason())

    def test_unsupported_config_preserves_existing_path(self):
        for kwargs in (
            {"enabled": False},
            {"host_mirror": True},
            {"chunk": 2},
            {"dispatch": "lp"},
            {"elastic": "mooncake"},
            {"elastic": "nccl"},
            {"ep": 2, "tp": 4},
            {"pp": 2},
            {"device": "cpu"},
            {"device": "xpu"},
        ):
            with self.subTest(**kwargs), _runtime(**kwargs):
                self.assertIsNotNone(manager_module.eplb_async_unavailable_reason())

    def test_unsupported_manager_does_not_enter_new_collectives(self):
        manager = EPLBManager.__new__(EPLBManager)
        manager._async_migrator = None
        with _runtime(elastic="nccl"), patch.object(dist, "all_reduce") as collective:
            manager._init_async_migrator()
            self.assertIsNone(manager._async_migrator)
            collective.assert_not_called()

    def test_two_rank_agreement_when_one_engine_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_select_on_rank, args=(f"{directory}/gloo",), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
