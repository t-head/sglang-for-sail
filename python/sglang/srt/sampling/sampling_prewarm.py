"""Prepare the selected sampler primitives without token synchronization."""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sglang.srt.layers.sampler import Sampler


def prewarm_flashinfer_sampling(
    sampler: "Sampler", *, vocab_size: int, device: str
) -> None:
    """Run both filtered FlashInfer paths on private one-row probabilities.

    The caller selects the normal execution stream, outside graph capture. This
    does not call Sampler.forward, synchronize tokens, or reserve a measured
    sampling headroom. DFLASH's separate peak-memory rehearsal is unchanged.
    """
    device = torch.device(device)
    if device.type != "cuda" or torch.version.hip is not None:
        return
    if vocab_size <= 0:
        raise ValueError("Sampling prewarm requires a positive vocabulary size")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Sampling prewarm must run outside graph capture")

    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()

    # Sampling must not advance either the CPU or this device's request RNG.
    # Do not seed another device or switch the caller's execution stream.
    with torch.random.fork_rng(devices=[device_index]):
        logits = torch.full((1, vocab_size), -100.0, dtype=torch.float32, device=device)
        active = min(64, vocab_size)
        logits[0, :active] = (
            4.0 - torch.arange(active, dtype=torch.float32, device=device) / 8.0
        )
        probs = torch.softmax(logits / 0.8, dim=-1)
        positions = torch.zeros(1, dtype=torch.int64, device=device)
        sampling_info = SamplingBatchInfo(
            temperatures=torch.full((1, 1), 0.8, dtype=torch.float32, device=device),
            top_ps=torch.full((1,), 0.9, dtype=torch.float32, device=device),
            top_ks=torch.full(
                (1,), min(32, vocab_size), dtype=torch.int32, device=device
            ),
            min_ps=torch.zeros(1, dtype=torch.float32, device=device),
            is_all_greedy=False,
            is_any_greedy=False,
            need_top_p_sampling=True,
            need_top_k_sampling=True,
            need_min_p_sampling=False,
            vocab_size=vocab_size,
        )
        # The joint top-k/top-p sampler and the min-p path use distinct APIs.
        # Pass tensor-valued thresholds, as normal batched requests do.
        sampler._sample_from_probs(
            probs, sampling_info, positions, simple_sampling_case=False
        )
        sampling_info.need_min_p_sampling = True
        sampling_info.min_ps.fill_(0.05)
        sampler._sample_from_probs(
            probs, sampling_info, positions, simple_sampling_case=False
        )
        torch.cuda.current_stream(device).synchronize()
