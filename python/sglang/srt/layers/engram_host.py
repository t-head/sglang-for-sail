"""State and prefetch for the opt-in Shared Host Engram path."""

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.ops.quantization.fp4_fake_quant import FP8_BLOCK_SIZE
from sglang.kernels.ops.embeddings.engram_host import (
    engram_shared_host_gather,
    fused_engram_shared_host_gate,
)

if TYPE_CHECKING:
    from sglang.srt.layers.engram import Engram, EngramEmbedding
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)
_prefetch_streams: dict[torch.device, torch.cuda.Stream] = {}


class SharedHostEngram:
    def __init__(self):
        self._scale_lut: Optional[torch.Tensor] = None
        self._scale_codes: Optional[torch.Tensor] = None
        self._prefetched: Optional[
            tuple[torch.Tensor, torch.Tensor, torch.cuda.Event]
        ] = None

    def finish_load(self, scales: Optional[torch.Tensor], label: str):
        """Replace the host per-row f32 scales with uint8 codes in device memory
        when they take at most 256 distinct values, so a lookup reads only the
        payload row from the host. The codes reproduce every scale exactly."""
        self._prefetched = None
        self._scale_lut = self._scale_codes = None
        if scales is None or scales.numel() == 0 or not torch.cuda.is_available():
            return
        device = torch.device("cuda", torch.cuda.current_device())
        src = scales
        n, step = src.numel(), 1 << 26
        lut = torch.unique(
            torch.cat(
                [torch.unique(src[i : i + step].to(device)) for i in range(0, n, step)]
            )
        )
        if lut.numel() > 256:
            return
        codes = torch.empty(n, dtype=torch.uint8, device=device)
        for i in range(0, n, step):
            chunk = src[i : i + step].to(device)
            code = torch.searchsorted(lut, chunk)
            if bool((code >= lut.numel()).any()):
                return
            restored = lut[code]
            if not torch.equal(restored.view(torch.int32), chunk.view(torch.int32)):
                return
            codes[i : i + step] = code.to(torch.uint8)
        self._scale_lut, self._scale_codes = lut, codes
        logger.info(
            "engram table %s: %s row scales as uint8 codes over %s values",
            label,
            n,
            lut.numel(),
        )

    def _gather(
        self, embed: "EngramEmbedding", indices: torch.Tensor, out: torch.Tensor
    ) -> None:
        weight_ptr, scale_ptr = embed._table_ptrs()
        if self._scale_codes is not None:
            scale_ptr = self._scale_codes.data_ptr()
        engram_shared_host_gather(
            weight_ptr,
            scale_ptr,
            indices.reshape(-1),
            out.view(-1, embed.dim),
            embed.dim,
            FP8_BLOCK_SIZE,
            rows=embed.weight.shape[0],
            per_row_scale=embed._per_channel_scale,
            int8_payload=embed._int8_payload,
            scale_lut=self._scale_lut,
        )

    def lookup(self, embed: "EngramEmbedding", indices: torch.Tensor) -> torch.Tensor:
        pre, self._prefetched = self._prefetched, None
        if indices.shape[0] == 0:
            return embed.new_rows(indices)
        if (
            pre is not None
            and pre[0].data_ptr() == indices.data_ptr()
            and pre[0].shape == indices.shape
            and pre[0].stride() == indices.stride()
        ):
            torch.cuda.current_stream(indices.device).wait_event(pre[2])
            return pre[1]
        out = embed.new_rows(indices)
        self._gather(embed, indices, out)
        return out

    def prefetch(
        self,
        embed: "EngramEmbedding",
        ids: torch.Tensor,
        stream: torch.cuda.Stream,
    ) -> None:
        # Keep the output owned by the caller/consumer stream. The caller must
        # make the prefetch stream wait for that stream before calling us (see
        # prefetch_engram_host_rows). record_stream below protects side-stream
        # writes; lookup waits for completion before the owner stream consumes it.
        # Moving allocation to the side stream would also require tracking the
        # consumer stream's use before this storage could safely be recycled.
        out = embed.new_rows(ids)
        with torch.cuda.stream(stream):
            self._gather(embed, ids, out)
            done = torch.cuda.Event()
            done.record(stream)
        # The batch or a weight reload may discard these tensors before the side
        # stream finishes. Keep their allocations valid until its work completes.
        ids.record_stream(stream)
        out.record_stream(stream)
        if self._scale_codes is not None:
            self._scale_codes.record_stream(stream)
        if self._scale_lut is not None:
            self._scale_lut.record_stream(stream)
        self._prefetched = (ids, out, done)

    @staticmethod
    def gate(
        x: torch.Tensor,
        kv: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        eps: float,
        clamp_value: float,
    ) -> Optional[torch.Tensor]:
        if (
            x.ndim == 3
            and kv.shape == (x.shape[0], (x.shape[1] + 1) * x.shape[2])
            and x.dtype == kv.dtype
            and x.dtype in (torch.bfloat16, torch.float32)
            and q_weight.dtype in (torch.bfloat16, torch.float32)
            and k_weight.dtype in (torch.bfloat16, torch.float32)
            and all(t.is_contiguous() for t in (x, kv, q_weight, k_weight))
        ):
            return fused_engram_shared_host_gate(
                x, kv, q_weight, k_weight, eps, clamp_value
            )
        return None


def prefetch_engram_host_rows(
    engrams: list[Optional["Engram"]],
    hash_ids: torch.Tensor,
    forward_batch: "ForwardBatch",
) -> None:
    layers = [
        (layer.embed._host_opt, layer.embed, layer.layer_hash_index)
        for layer in engrams
        if layer is not None and getattr(layer.embed, "_host_opt", None) is not None
    ]
    for opt, _, _ in layers:
        opt._prefetched = None
    if (
        not layers
        or hash_ids.ndim != 3
        or hash_ids.shape[0] == 0
        or not hash_ids.is_cuda
        or not forward_batch.forward_mode.is_extend()
        or torch.cuda.is_current_stream_capturing()
    ):
        return
    stream = _prefetch_streams.get(hash_ids.device)
    if stream is None:
        stream = torch.cuda.Stream(device=hash_ids.device)
        _prefetch_streams[hash_ids.device] = stream
    stream.wait_stream(torch.cuda.current_stream(hash_ids.device))
    for opt, embed, col in layers:
        opt.prefetch(embed, hash_ids[:, col], stream)
