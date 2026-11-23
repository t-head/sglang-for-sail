"""Runtime Engram-hash num_tokens alignment must not recompile the kernel."""

import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.embeddings.engram_hash import (
    MODE_DECODE,
    _engram_hash_kernel,
    engram_hash_ids,
)
from sglang.test.ci.ci_register import register_cuda_ci, register_ppu_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_ppu_ci(est_time=15, suite="stage-a-test-1-gpu-ppu")


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA-compatible device")
class TestEngramHashRowAlignment(CustomTestCase):
    @torch.inference_mode()
    def test_aligned_unaligned_reuses_kernel(self):
        torch.manual_seed(27)
        n, layers, heads, vocab = 4, 2, 8, 256
        device = "cuda"
        max_tokens = 129  # unaligned runtime num_tokens; 128 is 16-aligned

        ids = torch.randint(0, vocab, (max_tokens,), device=device)
        positions = torch.arange(max_tokens, device=device)
        history = torch.randint(
            0, vocab, (max_tokens + 4, n - 1), device=device, dtype=torch.int32
        )
        token_map = torch.randperm(vocab, device=device)
        multipliers = torch.randint(0, 2**48, (layers, n), device=device) * 2 + 1
        primes = torch.arange(
            16000001, 16000001 + layers * (n - 1) * heads, device=device
        ).reshape(layers, n - 1, heads)
        offsets = primes.flatten(1).cumsum(1) - primes.flatten(1)

        launched = []
        run = _engram_hash_kernel.run

        def record_kernel(*args, **kwargs):
            kernel = run(*args, **kwargs)
            launched.append(kernel)
            return kernel

        def call(num_tokens):
            return engram_hash_ids(
                ids[:num_tokens],
                positions[:num_tokens],
                mode=MODE_DECODE,
                history=history,
                token_map=token_map,
                multipliers=multipliers,
                primes=primes,
                offsets=offsets,
                pad_id=2,
            )

        # Observe the compiled handles returned by real Triton launches, not
        # filenames in a backend-specific disk cache. Keep all pointer layouts
        # and constexpr arguments fixed; only runtime num_tokens changes alignment.
        with patch.object(_engram_hash_kernel, "run", side_effect=record_kernel):
            out_unaligned, tokens_unaligned = call(max_tokens)
            out_aligned, tokens_aligned = call(max_tokens - 1)
            torch.cuda.synchronize()

        self.assertEqual(len(launched), 2)
        self.assertIsNotNone(launched[0])
        self.assertIs(
            launched[1],
            launched[0],
            "runtime num_tokens alignment recompiled the Engram hash kernel",
        )
        # Prefix math must be identical for the shared rows.
        torch.testing.assert_close(
            out_unaligned[: max_tokens - 1].cpu(), out_aligned.cpu(), rtol=0, atol=0
        )
        torch.testing.assert_close(
            tokens_unaligned[: max_tokens - 1].cpu(),
            tokens_aligned.cpu(),
            rtol=0,
            atol=0,
        )


if __name__ == "__main__":
    unittest.main()
