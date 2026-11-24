"""Peer copies from execution weights into a single layer of symmetric staging."""

import logging
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


class PeerStagingUnavailable(RuntimeError):
    pass


@contextmanager
def _agree_on_staging(group):
    """All peers must finish the local stage before entering the next collective."""
    error = None
    try:
        yield
    except (RuntimeError, OSError, NotImplementedError) as exc:
        error = str(exc)
    errors = [None] * dist.get_world_size(group)
    dist.all_gather_object(errors, error, group=group)
    for rank, message in enumerate(errors):
        if message is not None:
            raise PeerStagingUnavailable(f"EP rank {rank}: {message}")


class ExpertCopyEngine:
    def __init__(self, group, extension):
        self.group = group
        self.extension = extension

    @classmethod
    def try_create(cls, group):
        from sglang.srt.runtime_context import get_parallel
        from sglang.srt.utils.common import is_mnnvl_fabric_device

        try:
            import torch.distributed._symmetric_memory as symm
            from torch.utils.cpp_extension import load

            if not all(hasattr(symm, name) for name in ("empty", "rendezvous")):
                return None
            if get_parallel().nnodes > 1 and not is_mnnvl_fabric_device():
                return None
            if get_parallel().nnodes == 1:
                device = torch.cuda.current_device()
                if any(
                    not torch.cuda.can_device_access_peer(device, peer)
                    for peer in range(torch.cuda.device_count())
                    if peer != device
                ):
                    return None
            source = (
                Path(__file__).resolve().parents[2]
                / "kernels/jit/csrc/eplb/copy_engine.cu"
            )
            ext = load(
                "sglang_eplb_copy_engine", [str(source)], extra_cuda_cflags=["-O3"]
            )
        except (ImportError, OSError, RuntimeError) as exc:
            logger.warning("Cannot initialize the EPLB copy engine: %s", exc)
            return None
        return cls(group, ext)

    def prepare(self, weights):
        import torch.distributed._symmetric_memory as symm

        total = 0
        self.regions = []
        for t in weights:
            total = (total + 255) // 256 * 256
            slot_bytes = t.stride(0) * t.element_size()
            self.regions.append((total, slot_bytes))
            total += t.shape[0] * slot_bytes
        self.slots = weights[0].shape[0]
        self.rank = self.group.rank_in_group
        world = self.group.world_size
        with _agree_on_staging(self.group.cpu_group):
            self.arena = symm.empty(total, dtype=torch.uint8, device=weights[0].device)
        with _agree_on_staging(self.group.cpu_group):
            self.handle = symm.rendezvous(self.arena, group=self.group.device_group)
            self.peer_bases = [
                self.handle.get_buffer(r, (total,), torch.uint8).data_ptr()
                for r in range(world)
            ]
            self.stream = torch.cuda.Stream(device=weights[0].device)
        # The worker and the forward thread may enter collectives concurrently.
        self.copy_group = dist.new_group(
            ranks=self.group.ranks, backend="gloo", use_local_synchronization=True
        )
        self.requests = torch.full((self.slots, 2), -1, dtype=torch.int32, device="cpu")
        self.peer_requests = [torch.empty_like(self.requests) for _ in range(world)]
        return [
            self.arena[offset : offset + t.numel() * t.element_size()]
            .view(t.dtype)
            .as_strided(t.shape, t.stride())
            for t, (offset, _) in zip(weights, self.regions, strict=True)
        ]

    def transfer(self, weights, plan):
        self.requests.fill_(-1)
        for source_rank, pairs in plan.reads.items():
            for src, dst in pairs:
                self.requests[dst, 0] = source_rank
                self.requests[dst, 1] = src

        # Each worker has finished its preceding install before entering here.
        # A sender can now reuse every receiver's staging, including slow peers.
        dist.all_gather(self.peer_requests, self.requests, group=self.copy_group)
        src_ptrs, dst_ptrs, sizes = [], [], []
        moves = [(self.rank, src, dst) for src, dst in plan.weight2buffer_copies]
        for peer, requests in enumerate(self.peer_requests):
            for dst, (owner, src) in enumerate(requests.tolist()):
                if owner == self.rank:
                    moves.append((peer, src, dst))
        bases = [t.data_ptr() for t in weights]
        for peer, src, dst in moves:
            for base, (offset, size) in zip(bases, self.regions, strict=True):
                src_ptrs.append(base + src * size)
                dst_ptrs.append(self.peer_bases[peer] + offset + dst * size)
                sizes.append(size)
        if sizes:
            self.extension.enqueue_memcpy_batch(
                src_ptrs, dst_ptrs, sizes, self.stream.cuda_stream
            )
            self.stream.synchronize()

    def close(self):
        if hasattr(self, "copy_group"):
            if dist.is_initialized():
                dist.destroy_process_group(self.copy_group)
            del self.copy_group
