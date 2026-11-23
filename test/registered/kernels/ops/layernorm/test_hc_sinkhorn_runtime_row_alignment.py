"""Fused mHC Reduce/Sinkhorn must not specialize on runtime row alignment."""

import json
import unittest

import torch
import triton

from sglang.kernels.ops.layernorm.mhc import _hc_mix_reduce_sinkhorn_kernel
from sglang.test.ci.ci_register import register_cuda_ci, register_ppu_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_ppu_ci(est_time=45, suite="stage-a-test-1-gpu-ppu")


# No allocation, dtype, pointer, launch option or constexpr changes within a
# comparison. Include the unit-value class separately from alignment classes.
ROWS = (16, 17, 32, 788, 2052, 1)
CAPACITY = max(ROWS) + 32
HC, MIX, ITERS, K = 4, 24, 20, 20480
RMS_EPS, HC_EPS = 1e-6, 1e-6
GUARD, SENTINEL = 32, 1234567.0


def reference(part_mix, part_sq, residual, scale, base, m, slices):
    """Independent FP64 form of split-K reduction and Sinkhorn normalization."""
    mix = part_mix[: slices * m * MIX].reshape(slices, m, MIX).double()
    sq = part_sq[: slices * m].reshape(slices, m).double()
    if residual is not None:
        mix = mix + residual[: slices * m * MIX].reshape(slices, m, MIX).double()
    reduced = mix.sum(0) * torch.rsqrt(sq.sum(0)[:, None] / K + RMS_EPS)
    pre = torch.sigmoid(reduced[:, :HC] * scale[0] + base[:HC]) + HC_EPS
    post = 2 * torch.sigmoid(reduced[:, HC : 2 * HC] * scale[1] + base[HC : 2 * HC])
    matrix = (reduced[:, 2 * HC :] * scale[2] + base[2 * HC :]).reshape(m, HC, HC)
    matrix = torch.exp(matrix - matrix.amax(-1, keepdim=True))
    matrix = matrix / matrix.sum(-1, keepdim=True) + HC_EPS
    matrix = matrix / (matrix.sum(-2, keepdim=True) + HC_EPS)
    for _ in range(ITERS - 1):
        matrix = matrix / (matrix.sum(-1, keepdim=True) + HC_EPS)
        matrix = matrix / (matrix.sum(-2, keepdim=True) + HC_EPS)
    return pre.float(), post.float(), matrix.float()


def handle_labels(handles):
    """Identity labels, not disk-cache keys or guessed compile counters."""
    labels, seen = {}, []
    for m, handle in handles.items():
        index = next((i for i, previous in enumerate(seen) if handle is previous), None)
        if index is None:
            index = len(seen)
            seen.append(handle)
        labels[str(m)] = index
    return labels


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA-compatible device")
class TestHcSinkhornRuntimeRowAlignment(CustomTestCase):
    @torch.inference_mode()
    def test_production_reduce_sinkhorn_numerics_guards_and_reuse(self):
        production = _hc_mix_reduce_sinkhorn_kernel
        self.assertIn("m", production.do_not_specialize_on_alignment)
        # Same real production Python body, never a copied toy kernel. These
        # independent JIT objects demonstrate the original policy and the
        # broader value+alignment policy without modifying the imported object.
        control = triton.jit(production.fn)
        full_dynamic = triton.jit(
            production.fn,
            do_not_specialize=["m"],
            do_not_specialize_on_alignment=["m"],
        )
        policies = {
            "production": production,
            "original_jit": control,
            "full_dynamic": full_dynamic,
        }
        observations = []
        for slices in (1, 4):
            generator = torch.Generator().manual_seed(52327 + slices)
            mix_cpu = torch.randn(slices * CAPACITY * MIX, generator=generator) * 0.2
            residual_cpu = torch.randn(mix_cpu.shape, generator=generator) * 0.03
            sq_cpu = (torch.rand(slices * CAPACITY, generator=generator) + 0.5) * (
                K / slices
            )
            scale_cpu = torch.tensor([0.7, 1.3, 0.9], dtype=torch.float32)
            base_cpu = torch.randn(MIX, generator=generator) * 0.1
            mix, sq, residual, scale, base = [
                value.cuda()
                for value in (mix_cpu, sq_cpu, residual_cpu, scale_cpu, base_cpu)
            ]
            storage = [
                torch.empty(
                    CAPACITY * width + 2 * GUARD, device="cuda", dtype=torch.float32
                )
                for width in (HC, HC, HC * HC)
            ]
            outputs = [value[GUARD:-GUARD] for value in storage]
            grouped = {}
            for has_residual in (False, True):
                mode_handles = {}
                production_outputs = {}
                for policy, kernel in policies.items():
                    handles = {}
                    for m in ROWS:
                        with self.subTest(
                            slices=slices, residual=has_residual, policy=policy, rows=m
                        ):
                            for value in storage:
                                value.fill_(SENTINEL)
                            handle = kernel.run(
                                mix,
                                sq,
                                scale,
                                base,
                                *outputs,
                                m,
                                1.0 / K,
                                RMS_EPS,
                                MIX=MIX,
                                HC=HC,
                                NUM_SLICES=slices,
                                ITERS=ITERS,
                                EPS=HC_EPS,
                                part_mix_residual_ptr=residual
                                if has_residual
                                else None,
                                num_warps=1,
                                grid=(CAPACITY,),
                                warmup=False,
                            )
                            self.assertIsNotNone(handle)
                            handles[m] = handle
                            torch.cuda.synchronize()
                            expected = reference(
                                mix_cpu,
                                sq_cpu,
                                residual_cpu if has_residual else None,
                                scale_cpu,
                                base_cpu,
                                m,
                                slices,
                            )
                            actual = [
                                out[: m * width].reshape(want.shape).cpu().clone()
                                for out, width, want in zip(
                                    outputs, (HC, HC, HC * HC), expected
                                )
                            ]
                            if policy == "production":
                                production_outputs[m] = actual
                            else:
                                for got, want in zip(actual, production_outputs[m]):
                                    torch.testing.assert_close(
                                        got, want, rtol=0, atol=0
                                    )
                            for value, out, width, want in zip(
                                storage, outputs, (HC, HC, HC * HC), expected
                            ):
                                torch.testing.assert_close(
                                    out[: m * width].reshape(want.shape).cpu(),
                                    want,
                                    rtol=1e-5,
                                    atol=2e-6,
                                )
                                self.assertTrue(
                                    torch.all(value[:GUARD] == SENTINEL).item()
                                )
                                self.assertTrue(
                                    torch.all(
                                        value[GUARD + m * width :] == SENTINEL
                                    ).item()
                                )
                    mode_handles[policy] = handles
                    observations.append(
                        {
                            "slices": slices,
                            "residual": has_residual,
                            "policy": policy,
                            "handle_identity_labels": handle_labels(handles),
                        }
                    )
                # Only alignment is removed in the proposed minimal patch;
                # m=1 may remain its own intentional value specialization.
                for m in ROWS[:-1]:
                    self.assertIs(
                        mode_handles["production"][m], mode_handles["production"][16]
                    )
                self.assertIsNot(
                    mode_handles["original_jit"][16], mode_handles["original_jit"][17]
                )
                for m in ROWS:
                    self.assertIs(
                        mode_handles["full_dynamic"][m],
                        mode_handles["full_dynamic"][16],
                    )
                grouped[has_residual] = mode_handles
            # None is a separate constexpr mode, never evidence of redundant
            # row compilation. Every policy must retain this distinction.
            for policy in policies:
                self.assertIsNot(grouped[False][policy][16], grouped[True][policy][16])
            for got, want in zip(
                (mix, sq, residual, scale, base),
                (mix_cpu, sq_cpu, residual_cpu, scale_cpu, base_cpu),
            ):
                torch.testing.assert_close(got.cpu(), want, rtol=0, atol=0)
        print("SINKHORN_ROW_ALIGNMENT " + json.dumps(observations, sort_keys=True))


if __name__ == "__main__":
    unittest.main(verbosity=2)
