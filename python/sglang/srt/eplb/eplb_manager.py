from __future__ import annotations

import logging
import time
import weakref
from typing import TYPE_CHECKING, Any, Callable, List

import torch.cuda
import torch.distributed as dist
from torch import nn

from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import (
    ExpertLocationMetadata,
    format_expert_location_layout,
    format_expert_location_layout_diff,
    get_global_expert_location_metadata,
)
from sglang.srt.eplb.expert_location_updater import ExpertLocationUpdater
from sglang.srt.runtime_context import get_device, get_exec, get_model, get_parallel

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig

logger = logging.getLogger(__name__)


def eplb_async_unavailable_reason() -> str | None:
    """Check the topology and device supported by the async expert transport."""
    if not get_exec().moe.enable_eplb:
        return "EPLB is disabled"
    if get_exec().moe.enable_eplb_async:
        return "the host-mirror async backend is enabled"
    if get_exec().moe.ep_dispatch_algorithm == "lp":
        return "LP routing requires solver updates at the placement boundary"
    if get_exec().moe.eplb_rebalance_layers_per_chunk not in (None, 1):
        return "an explicit multi-layer chunk uses the existing rebalance path"
    if get_exec().moe.elastic_ep_backend is not None:
        return "elastic EP requires its existing expert update path"
    if get_parallel().ep_size != get_parallel().tp_size:
        return "async expert transfers require EP size equal to TP size"
    if get_parallel().pp_size != 1:
        return "async expert transfers do not support pipeline parallelism"
    if str(get_device().device).split(":")[0] != "cuda":
        return "async expert transfers require a CUDA device"
    return None


class EPLBManager:
    def __init__(
        self,
        *,
        model_config: ModelConfig,
        get_model: Callable[[], nn.Module],
        get_expert_location_updater: Callable[[], ExpertLocationUpdater],
        get_expert_backup_client: Callable[[], Any],
        get_weight_updater: Callable[[], Any],
        get_forward_pass_id: Callable[[], int],
    ):
        super().__init__()
        # These collaborators are set on ModelRunner AFTER EPLBManager is
        # constructed (model load, expert_backup_client, weight_updater), so
        # they are read through getters at rebalance time, not captured here.
        self._model_config = model_config
        self._get_model = get_model
        self._get_expert_location_updater = get_expert_location_updater
        self._get_expert_backup_client = get_expert_backup_client
        self._get_weight_updater = get_weight_updater
        self._get_forward_pass_id = get_forward_pass_id
        self._rebalance_layers_per_chunk = (
            get_exec().moe.eplb_rebalance_layers_per_chunk
        )
        self._rebalance_num_iterations = get_exec().moe.eplb_rebalance_num_iterations
        self._rebalance_disabled_reason = None
        self._rebalance_disabled_logged = False

        # Otherwise, the circular buffer will contain stale data. If the case is needed, it can be implemented.
        assert (
            get_exec().moe.eplb_rebalance_num_iterations
            >= get_exec().moe.expert_distribution_recorder_buffer_size
        ), (
            "eplb_rebalance_num_iterations must be greater than expert_distribution_recorder_buffer_size"
        )

        if not get_global_expert_distribution_recorder().recording:
            get_global_expert_distribution_recorder().start_record()

        logger.info(
            f"[EPLBManager] system started, will rebalance per {self._rebalance_num_iterations} iterations."
        )

        self._async_migrator = None
        self._async_job = None
        self._finalize_migrator = None
        self._init_async_migrator()

        self._prepared_rebalance_metadata = None
        self._prepared_rebalance_apply_event = None
        self._prepared_update_layer_ids_chunks = None
        self._pending_logical_count = None
        self._pending_logical_count_ready_event = None
        self._prepare_stream = (
            torch.cuda.Stream()
            if get_exec().moe.enable_eplb_async and torch.cuda.is_available()
            else None
        )
        self._async_main_generator = self._async_entrypoint()
        self._main_generator = self._entrypoint()

    def _init_async_migrator(self):
        from sglang.srt.distributed.parallel_state import get_moe_ep_group

        reason = eplb_async_unavailable_reason()
        if reason is not None:
            logger.debug(
                "[EPLBManager] asynchronous expert copies disabled: %s", reason
            )
            return

        from sglang.srt.eplb.eplb_async_migrator import AsyncExpertMigrator
        from sglang.srt.eplb.eplb_copy_engine import ExpertCopyEngine

        group = get_moe_ep_group()
        engine = ExpertCopyEngine.try_create(group)
        ready = torch.tensor([int(engine is not None)], dtype=torch.int32)
        dist.all_reduce(ready, op=dist.ReduceOp.MIN, group=group.cpu_group)
        if not ready.item():
            logger.warning(
                "[EPLBManager] copy engine unavailable; using synchronous EPLB"
            )
            return
        self._async_migrator = AsyncExpertMigrator(engine)

    def on_model_loaded(self, model: nn.Module):
        if self._async_migrator is None:
            return
        from sglang.srt.distributed.parallel_state import get_moe_ep_group
        from sglang.srt.eplb.eplb_async_migrator import expert_storage_layout
        from sglang.srt.eplb.eplb_copy_engine import PeerStagingUnavailable

        group = get_moe_ep_group().cpu_group
        weights = getattr(model, "routed_experts_weights_of_layer", None)
        metadata = get_global_expert_location_metadata()
        layout = (
            expert_storage_layout(weights, metadata.num_local_physical_experts)
            if metadata is not None
            else None
        )
        layouts = [None] * dist.get_world_size(group)
        dist.all_gather_object(layouts, layout, group=group)
        if layout is None or any(item != layout for item in layouts):
            logger.info(
                "Copy-engine EPLB requires matching packed expert storage across EP ranks; using synchronous EPLB"
            )
            self._async_migrator = None
            return
        # Bind only the migrator: the manager's model getters retain the runner.
        # Graceful process exit closes explicitly before distributed teardown.
        self._finalize_migrator = weakref.finalize(self, self._async_migrator.close)
        self._finalize_migrator.atexit = False
        try:
            self._async_migrator.prepare(weights, cpu_group=group, metadata=metadata)
        except PeerStagingUnavailable as exc:
            logger.warning("Cannot map peer staging; using synchronous EPLB: %s", exc)
            self._finalize_migrator()
            self._async_migrator = None
            return
        except BaseException:
            self._finalize_migrator()
            raise
        logger.info(
            "[EPLBManager] C++ copy engine ready for asynchronous layer-wise rebalance"
        )

    def close(self):
        if self._finalize_migrator is not None:
            self._finalize_migrator()
        self._async_migrator = None
        self._async_job = None
        self._main_generator.close()
        self._async_main_generator.close()

    def _rebalance_async(self):
        logger.info("[EPLBManager] rebalance start (async)")
        passes = 0

        dump_record_output = get_global_expert_distribution_recorder().dump_record(
            output_mode="object"
        )
        if not self._check_rebalance_needed(
            dump_record_output["average_utilization_rate_over_window"]
        ):
            return

        model = self._get_model()
        model_config = self._model_config
        from sglang.srt.eplb.eplb_cpu_worker import PlannerConfig

        job = self._async_migrator.start(
            logical_count=dump_record_output["logical_count"],
            old_metadata=get_global_expert_location_metadata(),
            routed_experts_weights_of_layer=model.routed_experts_weights_of_layer,
            planner_config=PlannerConfig.from_runtime(model_config),
        )
        self._async_job = job

        while not self._async_migrator.placement_ready(job):
            yield
            passes += 1

        new_expert_location_metadata = self._async_migrator.build_metadata(job)
        self._log_rebalance_layout_before_update(
            new_expert_location_metadata,
            update_layer_ids=job.layer_ids,
        )

        for layer_id in job.layer_ids:
            self._async_migrator.issue_layer(job, layer_id)
            while True:
                yield
                passes += 1
                if self._async_migrator.layer_ready(job, layer_id):
                    break
            self._async_migrator.install_layer(job, layer_id)

        self._async_job = None

        self._log_rebalance_layout_after_update(update_layer_ids=job.layer_ids)
        logger.info(
            "[EPLBManager] rebalance end (async) layers=%d passes=%d wall=%.3fs",
            len(job.layer_ids),
            passes,
            time.time() - job.time_start,
        )

    def on_forward_pass_end(self):
        if get_exec().moe.enable_eplb_async:
            next(self._async_main_generator)
            return
        next(self._main_generator)

    def reset_generator(self):
        if self._async_migrator is not None:
            self._async_migrator.cancel(self._async_job)
            self._async_job = None
        self._prepared_rebalance_metadata = None
        self._prepared_rebalance_apply_event = None
        self._prepared_update_layer_ids_chunks = None
        self._pending_logical_count = None
        self._pending_logical_count_ready_event = None
        self._async_main_generator = self._async_entrypoint()
        self._main_generator = self._entrypoint()

    def disable_rebalance(self, reason: str):
        self._rebalance_disabled_reason = reason
        self._rebalance_disabled_logged = False
        self.reset_generator()

    def enable_rebalance(self):
        self._rebalance_disabled_reason = None
        self._rebalance_disabled_logged = False
        self.reset_generator()

    # can be more complex if needed
    def _async_entrypoint(self):
        while True:
            if self._rebalance_disabled_reason is not None:
                yield
                continue
            forward_pass_id = self._get_forward_pass_id()
            if self._prepared_rebalance_metadata is not None:
                self._apply_prepared_async_rebalance()
                logger.info("[EPLBManager] async rebalance end")
            if self._pending_logical_count is not None:
                self._prepare_async_rebalance()
            if (
                forward_pass_id % self._rebalance_num_iterations == 0
                and self._pending_logical_count is None
                and self._prepared_rebalance_metadata is None
            ):
                logger.info("[EPLBManager] async rebalance start")
                self._start_async_rebalance_logical_count_fetch()
            yield

    def _entrypoint(self):
        while True:
            for _ in range(self._rebalance_num_iterations):
                yield

            yield from self.rebalance()

    def rebalance(self):
        if self._rebalance_disabled_reason is not None:
            if not self._rebalance_disabled_logged:
                logger.debug(
                    "[EPLBManager] rebalance disabled: %s",
                    self._rebalance_disabled_reason,
                )
                self._rebalance_disabled_logged = True
            return

        elastic_state = ElasticEPStateManager.instance()
        is_post_scale_rebalance = elastic_state is not None and elastic_state.has_scaled
        # A failed later scale leaves the previously committed world serving.
        if is_post_scale_rebalance and (
            elastic_state.pending_ep_size is not None
            or elastic_state.scale_phase not in ("serving_expanded", "failed")
        ):
            return

        if self._async_migrator is not None and not is_post_scale_rebalance:
            yield from self._rebalance_async()
            return

        mode = "async" if get_exec().moe.enable_eplb_async else "sync"
        logger.info(f"[EPLBManager] rebalance start mode={mode}")

        enable_timing = self._rebalance_layers_per_chunk is None

        if enable_timing:
            torch.get_device_module().synchronize()
            time_start = time.time()

        dump_record_output = get_global_expert_distribution_recorder().dump_record(
            output_mode="object"
        )
        logical_count = dump_record_output["logical_count"]
        average_utilization_rate_over_window = dump_record_output[
            "average_utilization_rate_over_window"
        ]

        # Check whether rebalancing is needed
        if not self._check_rebalance_needed(average_utilization_rate_over_window):
            return

        expert_location_metadata = self._compute_expert_location_metadata(
            logical_count,
            broadcast_over_world=is_post_scale_rebalance,
        )

        from sglang.srt.model_executor.model_runner_components.moe_ep_setup import (
            init_lplb_solvers,
        )

        update_layer_ids_chunks = self._compute_update_layer_ids_chunks()
        all_update_layer_ids = [
            layer_id for chunk in update_layer_ids_chunks for layer_id in chunk
        ]
        self._log_rebalance_layout_before_update(
            expert_location_metadata,
            update_layer_ids=all_update_layer_ids,
        )
        for chunk_layer_ids in update_layer_ids_chunks:
            if len(update_layer_ids_chunks) > 1:
                yield
            update_expert_location_with_recovery(
                expert_location_updater=self._get_expert_location_updater(),
                model=self._get_model(),
                new_expert_location_metadata=expert_location_metadata,
                update_layer_ids=chunk_layer_ids,
                tp_rank=(
                    self._elastic_global_rank()
                    if is_post_scale_rebalance
                    else get_parallel().tp_rank
                ),
                use_flat_topology=is_post_scale_rebalance,
                expert_backup_client=self._get_expert_backup_client(),
                update_weights_from_disk_callable=self._get_weight_updater().update_weights_from_disk,
                ep_dispatch_algorithm=get_exec().moe.ep_dispatch_algorithm,
                init_lplb_solvers_callable=lambda: init_lplb_solvers(
                    model_config=self._model_config
                ),
            )
            if is_post_scale_rebalance:
                # P2P waits only synchronize participating peers. Ranks without
                # moves must also install this chunk before NIXL resumes.
                dist.barrier()

        self._log_rebalance_layout_after_update(update_layer_ids=all_update_layer_ids)

        msg = f"[EPLBManager] rebalance end"
        if enable_timing:
            torch.get_device_module().synchronize()
            time_end = time.time()
            msg += f" time={time_end - time_start:.3f}s"
        logger.info(msg)

    def _compute_expert_location_metadata(
        self, logical_count, *, broadcast_over_world: bool
    ) -> ExpertLocationMetadata:
        if not broadcast_over_world:
            return ExpertLocationMetadata.init_by_eplb(
                self._model_config,
                logical_count,
            )

        current_metadata = get_global_expert_location_metadata()
        assert current_metadata is not None
        # One owner prevents process-local launch topology from influencing
        # the mapping chosen for the expanded world.
        if dist.get_rank() == 0:
            computed_metadata = ExpertLocationMetadata.init_by_eplb(
                self._model_config,
                logical_count,
                # Arbitrary append topologies may not preserve node divisibility.
                use_flat_topology=True,
            )
            physical_to_logical_map = (
                computed_metadata.physical_to_logical_map.contiguous()
            )
        else:
            physical_to_logical_map = torch.empty_like(
                current_metadata.physical_to_logical_map
            )

        dist.broadcast(physical_to_logical_map, src=0)
        return ExpertLocationMetadata.init_by_mapping(
            self._model_config,
            physical_to_logical_map,
            moe_ep_rank=self._elastic_global_rank(),
        )

    def _elastic_global_rank(self) -> int:
        return get_parallel().tp_rank + get_parallel().ep_join_rank_offset

    def _prepare_async_rebalance(self):
        if self._pending_logical_count is None:
            return
        if (
            self._pending_logical_count_ready_event is not None
            and not self._pending_logical_count_ready_event.query()
        ):
            return

        logical_count = self._pending_logical_count
        logical_count_ready_event = self._pending_logical_count_ready_event
        self._pending_logical_count = None
        self._pending_logical_count_ready_event = None

        (
            self._prepared_rebalance_metadata,
            self._prepared_rebalance_apply_event,
        ) = self._init_async_prepare_expert_location_metadata(
            logical_count,
            logical_count_ready_event=logical_count_ready_event,
        )
        self._prepared_update_layer_ids_chunks = self._compute_update_layer_ids_chunks()

    def _start_async_rebalance_logical_count_fetch(self):
        dump_record_output = get_global_expert_distribution_recorder().dump_record(
            output_mode="object"
        )
        logical_count = dump_record_output["logical_count"]
        average_utilization_rate_over_window = dump_record_output[
            "average_utilization_rate_over_window"
        ]

        if not self._check_rebalance_needed(average_utilization_rate_over_window):
            self._pending_logical_count = None
            self._pending_logical_count_ready_event = None
            self._prepared_rebalance_metadata = None
            self._prepared_rebalance_apply_event = None
            self._prepared_update_layer_ids_chunks = None
            return

        self._pending_logical_count = logical_count
        self._pending_logical_count_ready_event = (
            self._record_logical_count_ready_event(logical_count)
        )

    def _record_logical_count_ready_event(self, logical_count: torch.Tensor):
        if logical_count.device.type != "cuda":
            return None
        ready_event = torch.cuda.Event()
        torch.cuda.current_stream(device=logical_count.device).record_event(ready_event)
        return ready_event

    def _init_async_prepare_expert_location_metadata(
        self,
        logical_count: torch.Tensor,
        logical_count_ready_event=None,
    ):
        if self._prepare_stream is None:
            return (
                ExpertLocationMetadata.init_by_eplb(self._model_config, logical_count),
                None,
            )
        with torch.cuda.stream(self._prepare_stream):
            if logical_count_ready_event is not None:
                self._prepare_stream.wait_event(logical_count_ready_event)
            metadata = ExpertLocationMetadata.init_by_eplb(
                self._model_config, logical_count
            )
            apply_event = torch.cuda.Event()
            apply_event.record(self._prepare_stream)
        return metadata, apply_event

    def _apply_prepared_async_rebalance(self):
        if self._prepared_rebalance_metadata is None:
            return
        if self._prepared_rebalance_apply_event is not None:
            self._prepared_rebalance_apply_event.synchronize()
        for update_layer_ids in self._prepared_update_layer_ids_chunks:
            self._get_expert_location_updater().update(
                self._get_model().routed_experts_weights_of_layer,
                self._prepared_rebalance_metadata,
                update_layer_ids=update_layer_ids,
                nnodes=get_parallel().nnodes,
                rank=get_parallel().tp_rank,
            )
        self._prepared_rebalance_metadata = None
        self._prepared_rebalance_apply_event = None
        self._prepared_update_layer_ids_chunks = None

    def _check_rebalance_needed(self, average_utilization_rate_over_window):
        if average_utilization_rate_over_window is None:
            return True

        if (
            average_utilization_rate_over_window
            > get_exec().moe.eplb_min_rebalancing_utilization_threshold
        ):
            logger.info(
                f"[EPLBManager] Skipped ep rebalancing: current GPU utilization {average_utilization_rate_over_window:.2f} > minimum rebalance threshold {get_exec().moe.eplb_min_rebalancing_utilization_threshold:.2f}"
            )
            return False

        return True

    def _compute_update_layer_ids_chunks(self) -> List[List[int]]:
        routed_experts_weights_of_layer = getattr(
            self._get_model(), "routed_experts_weights_of_layer", None
        )
        if not routed_experts_weights_of_layer:
            return []
        all_layer_ids = sorted(list(routed_experts_weights_of_layer.keys()))
        chunk_size = self._rebalance_layers_per_chunk or 1000000
        return list(_chunk_list(all_layer_ids, chunk_size=chunk_size))

    def _should_log_expert_location_metadata(self) -> bool:
        return (
            get_parallel().tp_rank == 0
            and envs.SGLANG_LOG_EXPERT_LOCATION_METADATA.get()
        )

    def _log_rebalance_layout_before_update(
        self,
        new_expert_location_metadata: ExpertLocationMetadata,
        update_layer_ids: List[int],
    ):
        if not self._should_log_expert_location_metadata():
            return

        old_expert_location_metadata = get_global_expert_location_metadata()
        logger.info(
            "[EPLBManager] rebalance layout before:\n%s",
            format_expert_location_layout(
                old_expert_location_metadata,
                layer_ids=update_layer_ids,
            ),
        )
        logger.info(
            "[EPLBManager] rebalance layout target:\n%s",
            format_expert_location_layout(
                new_expert_location_metadata,
                layer_ids=update_layer_ids,
            ),
        )
        logger.info(
            "[EPLBManager] rebalance layout diff:\n%s",
            format_expert_location_layout_diff(
                old_expert_location_metadata,
                new_expert_location_metadata,
                layer_ids=update_layer_ids,
            ),
        )

    def _log_rebalance_layout_after_update(self, update_layer_ids: List[int]):
        if not self._should_log_expert_location_metadata():
            return

        logger.info(
            "[EPLBManager] rebalance layout after:\n%s",
            format_expert_location_layout(
                get_global_expert_location_metadata(),
                layer_ids=update_layer_ids,
            ),
        )


def update_expert_location_with_recovery(
    *,
    expert_location_updater: ExpertLocationUpdater,
    model: nn.Module,
    new_expert_location_metadata: ExpertLocationMetadata,
    update_layer_ids: List[int],
    tp_rank: int,
    use_flat_topology: bool = False,
    expert_backup_client,
    update_weights_from_disk_callable,
    ep_dispatch_algorithm: str,
    init_lplb_solvers_callable,
):
    p2p_missing_logical_experts = expert_location_updater.update(
        model.routed_experts_weights_of_layer,
        new_expert_location_metadata,
        update_layer_ids=update_layer_ids,
        nnodes=get_parallel().nnodes,
        rank=tp_rank,
        use_flat_topology=use_flat_topology,
    )

    if len(p2p_missing_logical_experts) > 0:
        # Load the missing expert weights from disk
        if callable(getattr(model, "generate_weight_name_filter", None)):
            # Filter and load only missing expert weights
            weight_name_filter = model.generate_weight_name_filter(
                p2p_missing_logical_experts
            )
        else:
            # Do a full reload from disk/DRAM
            logger.info(
                "[Elastic EP] Model does not implement generate_weight_name_filter. "
                "Performing full weight reload."
            )
            weight_name_filter = None

        if expert_backup_client is not None and expert_backup_client.use_backup:
            # Load the missing weights from the DRAM backup
            expert_backup_client.update_weights(weight_name_filter)
        else:
            # Load the missing weights from disk
            update_weights_from_disk_callable(
                get_model().model_path,
                get_model().load_format,
                weight_name_filter=weight_name_filter,
            )

    # Re-init LPLB solvers after expert location update
    if ep_dispatch_algorithm == "lp":
        init_lplb_solvers_callable()


def _chunk_list(items: List, chunk_size):
    for start_index in range(0, len(items), chunk_size):
        yield items[start_index : start_index + chunk_size]
