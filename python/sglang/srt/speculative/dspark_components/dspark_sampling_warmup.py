"""Prepare DSPARK classic/bonus primitives with private FP32 probabilities."""

import torch


def prewarm_dspark_sampling(
    *, gamma, verify_slots, vocab_size, device, stream, block_verification=False
):
    """Prepare current constexpr dimensions; do not claim all batch classes.

    One-hot FP32 probabilities bypass SoftmaxTemp deliberately. The normal
    target sampler's prewarm is retained separately by the worker override.
    Returned tensors are private outputs, never worker buffers or static IO.
    """
    if type(gamma) is not int or gamma < 1 or verify_slots != gamma + 1:
        raise ValueError("DSPARK prewarm requires gamma>=1 and slots=gamma+1")
    if type(verify_slots) is not int or type(vocab_size) is not int or vocab_size < 1:
        raise ValueError("DSPARK prewarm requires integer slots and positive vocab")
    if type(block_verification) is not bool:
        raise ValueError("block_verification must be the resolved boolean mode")
    device = torch.device(device)
    if device.type != "cuda" or device.index is None or torch.version.hip is not None:
        raise ValueError("DSPARK prewarm requires an explicit CUDA/PPU device")
    if stream is None or torch.device(stream.device) != device:
        raise ValueError("pass the current Scheduler stream on the target device")
    with torch.cuda.device(device):
        if torch.cuda.current_stream(device).cuda_stream != stream.cuda_stream:
            raise ValueError("DSPARK prewarm cannot switch the serving stream")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("DSPARK prewarm must run outside graph capture")
        from sglang.kernels.ops.speculative.dspark.dspark_accept import (
            gather_two_level_bonus_triton,
        )
        from sglang.kernels.ops.speculative.reject_sampling import (
            chain_speculative_sampling_triton,
        )

        with torch.inference_mode(), torch.random.fork_rng(devices=[device.index]):
            target_probs = torch.zeros(
                (1, verify_slots, vocab_size), dtype=torch.float32, device=device
            )
            draft_probs = torch.zeros(
                (1, gamma, vocab_size), dtype=torch.float32, device=device
            )
            target_probs[..., 0] = 1.0
            draft_probs[..., 0] = 1.0
            candidates = torch.zeros(
                (1, verify_slots), dtype=torch.int64, device=device
            )
            retrieve_index = torch.arange(
                verify_slots, dtype=torch.int64, device=device
            ).view(1, verify_slots)
            retrieve_next_token = retrieve_index + 1
            retrieve_next_token[:, -1] = -1
            retrieve_next_sibling = torch.full_like(retrieve_index, -1)
            predicts = torch.full((verify_slots,), -1, dtype=torch.int32, device=device)
            accept_indices = torch.full(
                (1, verify_slots), -1, dtype=torch.int32, device=device
            )
            num_correct_drafts = torch.full((1,), -1, dtype=torch.int32, device=device)
            uniform_samples = torch.full(
                (1, gamma), 0.5, dtype=torch.float32, device=device
            )
            final_uniform = torch.full((1,), 0.5, dtype=torch.float32, device=device)
            chain_speculative_sampling_triton(
                predicts=predicts,
                accept_index=accept_indices,
                accept_token_num=num_correct_drafts,
                candidates=candidates,
                retrive_index=retrieve_index,
                retrive_next_token=retrieve_next_token,
                retrive_next_sibling=retrieve_next_sibling,
                uniform_samples=uniform_samples,
                uniform_samples_for_final_sampling=final_uniform,
                target_probs=target_probs,
                draft_probs=draft_probs,
                threshold_single=1.0,
                threshold_acc=1.0,
                deterministic=True,
                block_verification=block_verification,
            )
            bonus_tokens = gather_two_level_bonus_triton(
                accept_index=accept_indices,
                predicts=predicts,
                correct_len=num_correct_drafts,
            )
            stream.synchronize()
            return predicts, accept_indices, num_correct_drafts, bonus_tokens
