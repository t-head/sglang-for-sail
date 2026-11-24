"""Persistent planner IPC, staging reuse, and failure propagation."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

import dataclasses
import os
import signal
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.eplb.eplb_async_migrator import plan_layer_pull
from sglang.srt.eplb.eplb_cpu_worker import (
    METADATA_UPLOAD_FIELDS,
    CPUPlanner,
    PlannerConfig,
    allocate_upload_buffers,
    compute_plan,
    decode_layer_plan,
)
from sglang.test.test_utils import CustomTestCase


class TestCPUPlanner(CustomTestCase):
    def test_process_reuse_staging_and_errors(self):
        torch.set_num_threads(1)
        config = PlannerConfig(16, 4, 1, 1, 0, "auto", True, False)
        old_map = torch.arange(16).remainder(12).repeat(3, 1)
        counts = torch.arange(36).reshape(3, 12).float() + 1
        reference = compute_plan(config, counts, old_map, [0, 1, 2])
        buffers = allocate_upload_buffers(reference.metadata)
        pointers = {name: tensor.data_ptr() for name, tensor in buffers.items()}
        worker = CPUPlanner(buffers)
        snapshots = []
        try:
            for step in range(4):
                loads = counts.roll(step, -1)
                expected = compute_plan(config, loads, old_map, [0, 1, 2])
                actual = worker.submit(config, loads, old_map, [0, 1, 2]).result(
                    timeout=30
                )
                self.assertNotEqual(actual.worker_pid, os.getpid())
                self.assertEqual(actual.worker_pid, worker.pid)
                for name in METADATA_UPLOAD_FIELDS:
                    torch.testing.assert_close(
                        getattr(actual.metadata, name),
                        getattr(expected.metadata, name),
                        rtol=0,
                        atol=0,
                    )
                    self.assertEqual(
                        getattr(actual.metadata, name).data_ptr(), pointers[name]
                    )
                for layer in range(3):
                    self.assertEqual(
                        decode_layer_plan(actual.pull_plans[layer]),
                        plan_layer_pull(
                            old_map[layer].tolist(),
                            expected.metadata.physical_to_logical_map[layer].tolist(),
                            4,
                            4,
                            0,
                        ),
                    )
                snapshots.append(
                    (
                        actual.metadata.physical_to_logical_map_cpu,
                        expected.metadata.physical_to_logical_map_cpu.clone(),
                    )
                )
                for snapshot, expected_snapshot in snapshots:
                    torch.testing.assert_close(
                        snapshot, expected_snapshot, rtol=0, atol=0
                    )
            bad = dataclasses.replace(config, num_physical_experts=17)
            with self.assertRaisesRegex(RuntimeError, "evenly distributed"):
                worker.submit(bad, counts, old_map, [0, 1, 2]).result(timeout=30)
            self.assertEqual(
                worker.submit(config, counts, old_map, [0, 1, 2])
                .result(timeout=30)
                .worker_pid,
                worker.pid,
            )
        finally:
            worker.shutdown()
        self.assertIsNotNone(worker._process.poll())

    @unittest.skipUnless(os.name == "posix", "POSIX worker transport")
    def test_decode_can_advance_while_planner_is_busy(self):
        worker = CPUPlanner()
        # The timer also releases the worker if submit ever becomes blocking.
        resume = threading.Timer(5, os.kill, args=(worker.pid, signal.SIGCONT))
        try:
            os.kill(worker.pid, signal.SIGSTOP)
            resume.start()
            config = PlannerConfig(4, 2, 1, 1, 0, "auto", False, False)
            future = worker.submit(
                config, torch.ones(1, 3), torch.tensor([[0, 1, 2, 0]]), [0]
            )
            self.assertFalse(future.done())
            self.assertTrue(resume.is_alive())
            os.kill(worker.pid, signal.SIGCONT)
            self.assertEqual(future.result(timeout=30).worker_pid, worker.pid)
        finally:
            resume.cancel()
            os.kill(worker.pid, signal.SIGCONT)
            worker.shutdown()

    def test_hung_exchange_can_be_cancelled_or_closed(self):
        from sglang.srt.eplb.eplb_async_migrator import AsyncExpertMigrator
        from sglang.srt.runtime_context import get_context

        config = PlannerConfig(4, 2, 1, 1, 0, "auto", False, False)
        counts, old = torch.ones(1, 3), torch.tensor([[0, 1, 2, 0]])
        metadata = compute_plan(config, counts, old, [0]).metadata
        popen = subprocess.Popen
        with (
            tempfile.TemporaryDirectory() as folder,
            get_context().override_server_args(device="cpu"),
        ):
            for action in ("reset", "close"):
                with self.subTest(action=action):
                    accepted = Path(folder) / action

                    def start_stalled(command, **kwargs):
                        # Use the production IPC protocol. Stall only after the
                        # child has decoded the request and entered compute_plan.
                        setup = (
                            "import time, pathlib\n"
                            "import sglang.srt.eplb.eplb_cpu_worker as worker\n"
                            "def stall(*args):\n"
                            f"    pathlib.Path({str(accepted)!r}).touch()\n"
                            "    time.sleep(120)\n"
                            "worker.compute_plan = stall\n"
                        )
                        return popen([*command[:-1], setup + command[-1]], **kwargs)

                    migrator = AsyncExpertMigrator(None)
                    with patch.object(subprocess, "Popen", side_effect=start_stalled):
                        worker = CPUPlanner()
                    migrator._planner_executor = worker
                    future = worker.submit(config, counts, old, [0])
                    job = SimpleNamespace(
                        placement_future=future, old_metadata=metadata
                    )
                    try:
                        deadline = time.monotonic() + 30
                        while not accepted.exists() and time.monotonic() < deadline:
                            time.sleep(0.01)
                        self.assertTrue(accepted.exists())
                        # An old, unbounded join must fail the test without hanging CI.
                        rescue = threading.Timer(25, worker._process.kill)
                        started = time.monotonic()
                        rescue.start()
                        try:
                            if action == "reset":
                                migrator.cancel(job)
                            else:
                                migrator.close()
                            self.assertLess(
                                time.monotonic() - started, 25, "shutdown needed rescue"
                            )
                        finally:
                            rescue.cancel()
                            rescue.join()
                        self.assertIsNotNone(worker._process.poll())
                        with self.assertRaises(RuntimeError):
                            future.result(timeout=1)
                        if action == "reset":
                            restarted = migrator._planner_executor.submit(
                                config, counts, old, [0]
                            )
                            self.assertNotEqual(
                                restarted.result(timeout=30).worker_pid, worker.pid
                            )
                    finally:
                        if worker._process.poll() is None:
                            worker._process.kill()
                        migrator.close()

    @unittest.skipUnless(os.name == "posix", "POSIX worker transport")
    def test_dead_worker_propagates(self):
        worker = CPUPlanner()
        try:
            os.kill(worker.pid, signal.SIGKILL)
            config = PlannerConfig(4, 2, 1, 1, 0, "auto", False, False)
            with self.assertRaises(RuntimeError):
                worker.submit(
                    config, torch.ones(1, 3), torch.tensor([[0, 1, 2, 0]]), [0]
                ).result(timeout=30)
        finally:
            worker.shutdown()


if __name__ == "__main__":
    unittest.main()
