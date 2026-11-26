"""#335: native PPU BF16 vs raw-FP8 512+0 NoPE, identical seeded operands.

Prepared timings exclude Q cast/padding and index preparation. End-to-end
attention timings include those operations (and KPool length scanning), but
exclude KV-cache population: BF16 and raw FP8 KV are already resident. BF16
pads heads to 64 as the DSA baseline does; Q8 only pads heads not divisible by 16.
No model/server latency or throughput claim is made by this microbenchmark.
"""

from __future__ import annotations

import math
import statistics
import unittest

import torch

from sglang.kernels.jit.benchmark.marker import do_bench
from sglang.srt.utils import is_sail
from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.test_utils import CustomTestCase

register_ppu_ci(est_time=120, suite="stage-a-test-1-gpu-ppu")

FP8 = torch.float8_e4m3fn
DIM = 512
SCALE = 1.0 / math.sqrt(256)
# (query rows, KV rows, logical query heads, real top-k width)
CASES = (
    (1, 4096, 1, 2048),
    (17, 8192, 8, 2051),
    (437, 16384, 16, 2176),
    (17, 32768, 32, 2048),
    (437, 65536, 64, 2051),
    (4096, 8192, 8, 2051),
)


def _event_us(fn, warmup=5, iterations=20, repeats=3):
    """Warm the actual callable, then time it with device events, not wall time."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        result = do_bench(
            fn,
            use_cuda_graph=False,
            warmup_iters=warmup,
            replay_iters=iterations,
            estimated_time_ms=None,
            metrics=(0.5,),
            disable_log_bandwidth=True,
        )
        times.append(result.times[0] * 1e6)
    return statistics.median(times)


def _inputs(rows, kv_rows, heads, width):
    seed = 335 + rows + kv_rows + heads + width
    gen = torch.Generator(device="cuda").manual_seed(seed)
    # One BF16 source for BOTH providers, never independent Q8 random draws.
    q = (torch.randn(rows, heads, DIM, generator=gen, device="cuda") * 0.5).to(
        torch.bfloat16
    )
    kv = (torch.randn(kv_rows, 1, DIM, generator=gen, device="cuda") * 0.5).to(
        torch.bfloat16
    )
    ids = torch.randint(
        1, kv_rows, (rows, width), generator=gen, dtype=torch.int32, device="cuda"
    )
    if width > 2048:
        ids[:, 1024:2048] = -1
        # Real KPool tail IDs following holes, not a contiguous valid prefix.
        ids[:, 2048:] = torch.randint(
            1,
            kv_rows,
            (rows, width - 2048),
            generator=gen,
            dtype=torch.int32,
            device="cuda",
        )
    return seed, q, kv, ids


def _pad_indices(ids):
    padded_width = ((ids.shape[-1] + 127) // 128) * 128
    if padded_width == ids.shape[-1]:
        return ids
    out = ids.new_full((ids.shape[0], padded_width), -1)
    out[:, : ids.shape[-1]].copy_(ids)
    return out


def _pad_bf16_query(q):
    if q.shape[1] == 64:
        return q
    out = q.new_zeros((q.shape[0], 64, DIM))
    out[:, : q.shape[1]].copy_(q)
    return out


class BenchDSANativeQ8PPU(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not is_sail() or torch.cuda.get_device_capability() != (8, 9):
            raise unittest.SkipTest("Native Q8 benchmarks require PPU0015 (SM89)")
        from sglang.kernels.ops.attention.ppu_flash_mla import (
            require_sparse_fp8_attention,
        )

        require_sparse_fp8_attention()

    def test_different_shapes(self):
        from sglang.kernels.ops.attention.metadata import q8kv8_topk_length_from_indices
        from sglang.kernels.ops.attention.mla_qk_prep import concat_and_cast_q_fp8_pad
        from sglang.kernels.ops.attention import flash_mla_sparse_quantized_fwd
        from sglang.srt.hardware_backend.ppu.attention.flash_mla import (
            flash_mla_sparse_fwd,
        )
        from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend

        print("Cold-L2 timings in us; KV population/cast excluded (resident KV cache).")
        for rows, kv_rows, heads, width in CASES:
            with self.subTest(rows=rows, kv_rows=kv_rows, heads=heads, width=width):
                seed, q, kv_bf16, ids = _inputs(rows, kv_rows, heads, width)
                kv_fp8 = kv_bf16.to(FP8)
                indices = _pad_indices(ids)
                kpool = width - 2048 + 1
                lengths = q8kv8_topk_length_from_indices(indices) if kpool > 1 else None
                q_bf16 = _pad_bf16_query(q)
                padded_heads_q8 = heads if heads % 16 == 0 else 64
                q_fp8 = torch.zeros(
                    rows, padded_heads_q8, DIM, dtype=FP8, device="cuda"
                )
                rope = q.new_empty((rows, heads, 0))
                concat_and_cast_q_fp8_pad(q_fp8, q, rope, heads)

                backend = DeepseekSparseAttnBackend.__new__(DeepseekSparseAttnBackend)
                backend.dsa_index_kpool = kpool
                backend.flashmla_padding = 64
                backend._ppu_q8_topk_capacity = indices.shape[-1]
                backend._ppu_q8_max_num_tokens = rows
                backend._ppu_q8_buffers = {}

                def bf16_prepared():
                    return flash_mla_sparse_fwd(
                        q_bf16,
                        kv_bf16,
                        indices.unsqueeze(1),
                        SCALE,
                        d_v=DIM,
                        topk_length=lengths,
                    )[0][:, :heads]

                def bf16_end_to_end():
                    prepared_ids = _pad_indices(ids)
                    prepared_lengths = (
                        q8kv8_topk_length_from_indices(prepared_ids)
                        if kpool > 1
                        else None
                    )
                    return flash_mla_sparse_fwd(
                        _pad_bf16_query(q),
                        kv_bf16,
                        prepared_ids.unsqueeze(1),
                        SCALE,
                        d_v=DIM,
                        topk_length=prepared_lengths,
                    )[0][:, :heads]

                def q8_prepared():
                    return flash_mla_sparse_quantized_fwd(
                        q_fp8,
                        kv_fp8,
                        indices.unsqueeze(1),
                        SCALE,
                        d_v=DIM,
                        topk_length=lengths,
                    )[:, :heads]

                def q8_end_to_end():
                    # The production method performs Q cast, head/table padding,
                    # and the real KPool scan on reusable attention buffers.
                    return backend._forward_ppu_native_q8(
                        q_nope=q,
                        q_rope=rope,
                        kv_cache=kv_fp8,
                        page_table_1=ids,
                        sm_scale=SCALE,
                        v_head_dim=DIM,
                        topk_length=None,
                        reuse_buffers=True,
                    )

                bf16 = bf16_prepared()
                q8 = q8_prepared()
                for out in (bf16, q8, bf16_end_to_end(), q8_end_to_end()):
                    self.assertEqual(tuple(out.shape), (rows, heads, DIM))
                    self.assertTrue(torch.isfinite(out).all().item())
                torch.testing.assert_close(
                    bf16_end_to_end(), bf16, atol=5e-3, rtol=2e-2
                )
                torch.testing.assert_close(q8_end_to_end(), q8, atol=5e-3, rtol=2e-2)
                # Native-provider difference includes FP8 operand loss, NOT just
                # kernel error. The companion numerical UT isolates the latter.
                mae = (q8.float() - bf16.float()).abs().mean().item()
                timings = {
                    name: _event_us(fn)
                    for name, fn in (
                        ("bf16_prepared", bf16_prepared),
                        ("bf16_end_to_end_pad_scan", bf16_end_to_end),
                        ("q8_prepared_cast_excluded", q8_prepared),
                        ("q8_end_to_end_qcast_pad_scan", q8_end_to_end),
                    )
                }
                print(
                    f"seed={seed} R={rows} KV={kv_rows} H={heads} K={width} "
                    f"padded_H_bf16=64 padded_H_q8={padded_heads_q8} "
                    f"native_q8_vs_bf16_mae={mae:.6g} "
                    + " ".join(f"{name}={value:.3f}" for name, value in timings.items())
                    + f" prepared_speedup={timings['bf16_prepared'] / timings['q8_prepared_cast_excluded']:.3f}x"
                    + f" end_to_end_speedup={timings['bf16_end_to_end_pad_scan'] / timings['q8_end_to_end_qcast_pad_scan']:.3f}x"
                )


if __name__ == "__main__":
    unittest.main()
