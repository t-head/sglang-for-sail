from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

import gc
import os
import tempfile
import time
import traceback
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.distributed
import torch.multiprocessing as mp

from sglang.srt.eplb.eplb_async_migrator import plan_layer_pull
from sglang.test.test_utils import CustomTestCase, find_available_port

_WORLD_SIZE = 2
_NUM_LOCAL = 2
_NUM_LOGICAL = 3
_NUM_LAYERS = 2
_OLD_MAPS = [[0, 1, 2, 0], [2, 1, 0, 1]]
_ROW = 4


def _fill(tensor, physical_to_logical_map, rank):
    lo = rank * _NUM_LOCAL
    for i in range(_NUM_LOCAL):
        tensor[i] = float(physical_to_logical_map[lo + i])


def _expected(physical_to_logical_map, rank):
    t = torch.empty(_NUM_LOCAL, _ROW)
    _fill(t, physical_to_logical_map, rank)
    return t


class _SharedMemoryTransport:
    def __init__(self, shared, rank, read_delay):
        self._shared = shared
        self._rank = rank
        self._read_delay = read_delay
        self.reads = []

    def prepare(self, weights):
        self.staging = [torch.empty_like(t) for t in weights]
        return self.staging

    def transfer(self, weights, plan):
        layer_id = next(
            layer
            for layer, local in self._shared[self._rank].items()
            if local is weights
        )
        time.sleep(self._read_delay)
        for src, dst in plan.weight2buffer_copies:
            for weight, buf in zip(weights, self.staging, strict=True):
                buf[dst].copy_(weight[src])
        for rank, pairs in plan.reads.items():
            for src, dst in pairs:
                for i, buf in enumerate(self.staging):
                    buf[dst].copy_(self._shared[rank][layer_id][i][src])
                self.reads.append((rank, layer_id, src, dst))


def _run_rank(rank, master_port, shared, writer):
    try:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(master_port)
        torch.distributed.init_process_group(
            backend="gloo", rank=rank, world_size=_WORLD_SIZE
        )

        from sglang.srt.eplb.eplb_async_migrator import AsyncExpertMigrator
        from sglang.srt.eplb.expert_location import (
            ExpertLocationMetadata,
            _compute_logical_to_all_physical_map,
        )
        from sglang.srt.runtime_context import get_context, get_parallel

        def metadata_for(maps):
            physical_to_logical_map = torch.tensor(maps, dtype=torch.int64)
            return ExpertLocationMetadata._init_raw(
                ep_size=_WORLD_SIZE,
                physical_to_logical_map=physical_to_logical_map,
                logical_to_all_physical_map=_compute_logical_to_all_physical_map(
                    physical_to_logical_map,
                    num_logical_experts=_NUM_LOGICAL,
                    ep_size=_WORLD_SIZE,
                    moe_ep_rank=None,
                ),
            )

        with (
            get_context().override_server_args(
                device="cpu",
                ep_size=_WORLD_SIZE,
                tp_size=_WORLD_SIZE,
                nnodes=1,
                ep_dispatch_algorithm=None,
            ),
            get_parallel().override(tp_rank=rank, tp_size=_WORLD_SIZE),
        ):
            weights_of_layer = shared[rank]
            old_metadata = metadata_for(_OLD_MAPS)
            from sglang.srt.eplb.eplb_cpu_worker import PlannerConfig, compute_plan

            config = PlannerConfig(
                num_physical_experts=4,
                ep_size=2,
                num_groups=1,
                num_nodes=1,
                rank=rank,
                algorithm="auto",
                static_dispatch=False,
                prefer_same_node=False,
            )
            counts = torch.tensor([[1, 100, 1], [100, 1, 1]], dtype=torch.int32)
            expected_plan = compute_plan(
                config,
                counts,
                old_metadata.physical_to_logical_map_cpu,
                list(range(_NUM_LAYERS)),
            )
            new_maps = expected_plan.metadata.physical_to_logical_map.tolist()
            assert new_maps != _OLD_MAPS

            transport = _SharedMemoryTransport(
                shared, rank, read_delay=0.5 if rank == 1 else 0.0
            )
            migrator = AsyncExpertMigrator(transport)
            migrator.prepare(
                weights_of_layer,
                cpu_group=torch.distributed.group.WORLD,
                metadata=old_metadata,
            )
            job = migrator.start(
                logical_count=counts,
                old_metadata=old_metadata,
                routed_experts_weights_of_layer=weights_of_layer,
                planner_config=config,
            )
            passes = 0
            while not migrator.placement_ready(job):
                passes += 1
                assert passes < 2000, "placement never agreed ready"
                time.sleep(0.005)
            migrator.build_metadata(job)

            polls_per_layer = []
            for layer_id in job.layer_ids:
                migrator.issue_layer(job, layer_id)
                polls = 0
                while not migrator.layer_ready(job, layer_id):
                    polls += 1
                    assert polls < 2000, "layer never agreed ready"
                    for expected, actual in zip(
                        [_expected(_OLD_MAPS[layer_id], rank)] * 2,
                        weights_of_layer[layer_id],
                        strict=True,
                    ):
                        assert torch.equal(expected, actual), "install happened early"
                    time.sleep(0.005)
                polls_per_layer.append(polls)
                migrator.install_layer(job, layer_id)
                for expected, actual in zip(
                    [_expected(new_maps[layer_id], rank)] * 2,
                    weights_of_layer[layer_id],
                    strict=True,
                ):
                    assert torch.equal(expected, actual), (
                        f"rank {rank} layer {layer_id}: {actual} != {expected}"
                    )
                assert (
                    old_metadata.physical_to_logical_map_cpu[layer_id].tolist()
                    == new_maps[layer_id]
                )
                for later in job.layer_ids[job.layers_installed :]:
                    assert (
                        old_metadata.physical_to_logical_map_cpu[later].tolist()
                        == _OLD_MAPS[later]
                    ), "a later layer's map changed before its install"

            assert job.layers_installed == _NUM_LAYERS
            assert all(p >= 1 for p in polls_per_layer), polls_per_layer
            expected_remote = sum(
                len(pairs)
                for layer in job.plans.values()
                for pairs in layer.reads.values()
            )
            assert len(transport.reads) == expected_remote
            if rank == 1:
                migrator._copy_error = OSError("injected read failure")
            with unittest.TestCase().assertRaisesRegex(
                RuntimeError, "Expert copy failed"
            ):
                migrator.layer_ready(job, job.layer_ids[-1])
            migrator.cancel(job)
            migrator.close()
        torch.distributed.barrier()
        writer.send(True)
    except Exception:
        traceback.print_exc()
        writer.send(False)


class TestRunnerTeardown(CustomTestCase):
    def test_close_and_recreate_before_distributed_teardown(self):
        from sglang.srt.distributed import parallel_state
        from sglang.srt.eplb import eplb_async_migrator, eplb_manager
        from sglang.srt.eplb.eplb_copy_engine import ExpertCopyEngine
        from sglang.srt.eplb.eplb_cpu_worker import METADATA_UPLOAD_FIELDS
        from sglang.srt.managers.scheduler import Scheduler
        from sglang.srt.model_executor.model_runner import ModelRunner
        from sglang.srt.runtime_context import get_context

        class CPUTransport(ExpertCopyEngine):
            def prepare(self, weights):
                self.copy_group = torch.distributed.new_group(backend="gloo")
                return [torch.empty_like(weight) for weight in weights]

        metadata = SimpleNamespace(
            num_local_physical_experts=2,
            **{
                field: torch.zeros(2, dtype=torch.int64)
                for field in METADATA_UPLOAD_FIELDS
            },
        )
        with tempfile.TemporaryDirectory() as folder:
            torch.distributed.init_process_group(
                "gloo",
                init_method=(Path(folder) / "rdzv").as_uri(),
                rank=0,
                world_size=1,
            )
            self.addCleanup(torch.distributed.destroy_process_group)
            with (
                get_context().override_server_args(
                    device="cpu",
                    enable_eplb=True,
                    expert_distribution_recorder_buffer_size=1,
                ),
                patch.object(
                    eplb_manager,
                    "get_global_expert_distribution_recorder",
                    return_value=Mock(recording=True),
                ),
                patch.object(
                    eplb_manager,
                    "get_global_expert_location_metadata",
                    return_value=metadata,
                ),
                patch.object(
                    eplb_manager, "eplb_async_unavailable_reason", return_value=None
                ),
                patch.object(
                    parallel_state,
                    "get_moe_ep_group",
                    return_value=SimpleNamespace(
                        cpu_group=torch.distributed.group.WORLD
                    ),
                ),
                patch.object(
                    eplb_async_migrator,
                    "expert_storage_layout",
                    return_value=("cpu-test-layout",),
                ),
                patch.object(
                    ExpertCopyEngine,
                    "try_create",
                    new=lambda group: CPUTransport(group, None),
                ),
            ):
                for explicit_close in (True, False):
                    runner = ModelRunner.__new__(ModelRunner)
                    runner.is_draft_worker = False
                    runner.model_config = None
                    runner.model = SimpleNamespace(
                        routed_experts_weights_of_layer={0: [torch.zeros(2, 4)]}
                    )
                    runner.maybe_init_eplb_manager()
                    runner.eplb_manager.on_model_loaded(runner.model)
                    migrator = runner.eplb_manager._async_migrator
                    thread = migrator._copy_thread
                    process = migrator._planner_executor._process
                    group = migrator._transport.copy_group
                    staging = weakref.ref(migrator._staging[0])
                    del migrator
                    self.assertTrue(thread.is_alive())
                    self.assertIsNone(process.poll())
                    if explicit_close:
                        scheduler = Mock()
                        scheduler.tp_worker.model_runner = runner
                        scheduler.hisparse_coordinator = (
                            scheduler.decode_offload_manager
                        ) = None
                        scheduler.server_args.enable_eplb = True
                        Scheduler.release_host_resources(scheduler)
                        runner.close_eplb()  # idempotent
                        self.assertIsNone(runner.eplb_manager)
                        del scheduler
                    del runner
                    gc.collect()
                    self.assertFalse(thread.is_alive())
                    self.assertIsNotNone(process.poll())
                    self.assertIsNone(staging())
                    with self.assertRaises((KeyError, ValueError)):
                        torch.distributed.get_process_group_ranks(group)
                    self.assertTrue(torch.distributed.is_initialized())


class TestPlanLayerPull(CustomTestCase):
    def test_same_gpu_then_free_rider_then_nearest_holder(self):
        old = [0, 1, 2, 3, 4, 5, 6, 7]
        new = [1, 0, 4, 4, 0, 0, 2, 7]
        plan0 = plan_layer_pull(old, new, 2, 2, rank=0)
        self.assertEqual(plan0.weight2buffer_copies, [(1, 0), (0, 1)])
        self.assertEqual(plan0.reads, {})
        plan1 = plan_layer_pull(old, new, 2, 2, rank=1)
        self.assertEqual(plan1.reads, {2: [(0, 0)]})
        self.assertEqual(plan1.buffer2weight_copies, [(0, 0), (0, 1)])
        plan3 = plan_layer_pull(old, new, 2, 2, rank=3)
        self.assertEqual(plan3.reads, {1: [(0, 0)]})
        plan2 = plan_layer_pull(old, new, 2, 2, rank=2)
        self.assertEqual(plan2.reads, {0: [(0, 0)]})
        self.assertEqual(plan2.buffer2weight_copies, [(0, 0), (0, 1)])


class TestEplbAsyncMigrator(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        mp.set_start_method("spawn", force=True)

    def test_two_ranks_pull_and_install_layer_by_layer(self):
        shared = {}
        for rank in range(_WORLD_SIZE):
            shared[rank] = {}
            for layer_id in range(_NUM_LAYERS):
                tensors = []
                for _ in range(2):
                    t = torch.empty(_NUM_LOCAL, _ROW)
                    _fill(t, _OLD_MAPS[layer_id], rank)
                    tensors.append(t.share_memory_())
                shared[rank][layer_id] = tensors

        master_port = find_available_port(23456)
        reader, writer = mp.Pipe(duplex=False)
        processes = [
            mp.Process(target=_run_rank, args=(rank, master_port, shared, writer))
            for rank in range(_WORLD_SIZE)
        ]
        for p in processes:
            p.start()
        for _ in range(_WORLD_SIZE):
            self.assertTrue(reader.recv(), "a rank failed; see its traceback above")
        for p in processes:
            p.join()


class TestPackedExpertStorage(CustomTestCase):
    def test_permuted_scale_raw_transfer(self):
        import ctypes

        from sglang.srt.eplb.eplb_async_migrator import _expert_storage_bytes

        source = (
            torch.arange(3 * 8 * 4, dtype=torch.int32).reshape(3, 8, 4).transpose(1, 2)
        )
        destination = torch.empty_like(source)
        self.assertFalse(source.is_contiguous())
        self.assertEqual(source.stride(), destination.stride())
        nbytes = _expert_storage_bytes(source)
        ctypes.memmove(destination[0].data_ptr(), source[2].data_ptr(), nbytes)
        torch.testing.assert_close(destination[0], source[2])
        self.assertEqual(nbytes, source[0].numel() * source.element_size())

    def test_pushes_use_destination_rank_and_slot(self):
        import ctypes
        from unittest.mock import Mock, patch

        from sglang.srt.eplb.eplb_async_migrator import LayerPullPlan
        from sglang.srt.eplb.eplb_copy_engine import ExpertCopyEngine

        def copy(src, dst, sizes, stream):
            for source, target, size in zip(src, dst, sizes, strict=True):
                ctypes.memmove(target, source, size)

        engine = ExpertCopyEngine(None, Mock(enqueue_memcpy_batch=copy))
        engine.rank = 0
        engine.copy_group = object()
        engine.stream = Mock(cuda_stream=1)
        weights = torch.tensor([[10, 11], [20, 21], [30, 31]], dtype=torch.uint8)
        # Keep guards on either side of the destination regions.
        staging = [torch.full((10,), 255, dtype=torch.uint8) for _ in range(2)]
        engine.regions = [(2, 2)]
        engine.peer_bases = [t.data_ptr() for t in staging]
        engine.requests = torch.empty(3, 2, dtype=torch.int32)
        engine.peer_requests = [torch.empty_like(engine.requests) for _ in staging]

        def gather(out, local, group):
            self.assertEqual(local.tolist(), [[-1, -1], [-1, -1], [1, 0]])
            out[0].copy_(local)
            out[1].copy_(torch.tensor([[0, 2], [-1, -1], [0, 0]]))

        plan = LayerPullPlan([(1, 0)], [], {1: [(0, 2)]})
        with patch.object(torch.distributed, "all_gather", side_effect=gather):
            engine.transfer([weights], plan)
        self.assertEqual(
            staging[0].tolist(), [255, 255, 20, 21, 255, 255, 255, 255, 255, 255]
        )
        self.assertEqual(
            staging[1].tolist(), [255, 255, 30, 31, 255, 255, 10, 11, 255, 255]
        )
        engine.stream.synchronize.assert_called_once()

    def test_reject_unsupported_layouts(self):
        from sglang.srt.eplb.eplb_async_migrator import _expert_storage_bytes

        for tensor in (
            torch.zeros(3, 8, 4)[:, ::2],
            torch.zeros(3, 8, 4)[::2],
            torch.zeros(1, 8, 4).expand(3, -1, -1),
        ):
            with self.assertRaises(ValueError):
                _expert_storage_bytes(tensor)


if __name__ == "__main__":
    unittest.main()
