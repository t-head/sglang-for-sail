import unittest

import torch

from sglang.kernels.ops.moe.kimi_k3_situ_and_mul_quant import (
    situ_and_mul_masked_post_quant,
    situ_and_mul_masked_post_quant_int8,
)
from sglang.srt.utils import get_device_sm, is_sail
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

_GROUP_SIZE = 128
_BETA = 4.0
_LINEAR_BETA = 25.0
_TOPK = 16


def _situ_reference(gate_up):
    gate, up = gate_up.chunk(2, dim=-1)
    gate = gate.float()
    up = up.float()
    return (
        _BETA
        * torch.tanh(gate / _BETA)
        * torch.sigmoid(gate)
        * _LINEAR_BETA
        * torch.tanh(up / _LINEAR_BETA)
    )


def _unpack_ue8m0_scales(packed, num_groups):
    num_experts, groups_per_word, num_tokens = packed.shape
    exponents = packed.contiguous().view(torch.uint8)
    exponents = exponents.view(num_experts, groups_per_word, num_tokens, 4)
    exponents = exponents.permute(0, 2, 1, 3).reshape(
        num_experts, num_tokens, num_groups
    )
    return torch.exp2(exponents.float() - 127.0)


class TestSituAndMulQuant(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        if get_device_sm() < 100:
            raise unittest.SkipTest("Kimi K3 compute kernels require SM100a+")

    def test_situ_mul_quant(self):
        torch.cuda.manual_seed_all(3)
        num_experts, num_tokens, hidden_size, topk = 8, 32, 1024, 16
        gate_up = (
            torch.randn(
                num_experts,
                num_tokens,
                2 * hidden_size,
                device="cuda",
                dtype=torch.float32,
            )
            * 2.0
        ).to(torch.bfloat16)
        masked_m = torch.randint(
            0,
            num_tokens + 1,
            (num_experts,),
            device="cuda",
            dtype=torch.int32,
        )
        masked_m[0] = 0
        masked_m[-1] = num_tokens

        output = torch.full(
            (num_experts, num_tokens, hidden_size),
            0x7F,
            device="cuda",
            dtype=torch.uint8,
        ).view(torch.float8_e4m3fn)
        num_groups = hidden_size // _GROUP_SIZE
        output_scale = torch.zeros(
            (num_experts, num_groups // 4, num_tokens),
            device="cuda",
            dtype=torch.int32,
        )

        situ_and_mul_masked_post_quant(
            input=gate_up,
            output=output,
            output_scale=output_scale,
            quant_group_size=_GROUP_SIZE,
            masked_m=masked_m,
            beta=_BETA,
            linear_beta=_LINEAR_BETA,
            scale_ue8m0=True,
            topk=topk,
            transposed=True,
        )

        scales = _unpack_ue8m0_scales(output_scale, num_groups)
        expanded_scales = scales.repeat_interleave(_GROUP_SIZE, dim=-1)
        dequantized = output.float() * expanded_scales
        expected = _situ_reference(gate_up)
        error_bound = expanded_scales * 17.0
        raw_output = output.view(torch.uint8)
        for expert in range(num_experts):
            valid_tokens = int(masked_m[expert].item())
            self.assertTrue(
                bool(
                    (
                        (
                            dequantized[expert, :valid_tokens]
                            - expected[expert, :valid_tokens]
                        ).abs()
                        <= error_bound[expert, :valid_tokens]
                    ).all()
                )
            )
            self.assertTrue(bool((raw_output[expert, valid_tokens:] == 0x7F).all()))


def _situ_int8_reference(gate_up):
    """Per-token absmax INT8 quantization of the SiTU reference."""
    values = _situ_reference(gate_up)
    absmax = values.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10)
    scale = absmax / 127.0
    return values, scale


class TestSituAndMulQuantInt8(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        # This kernel targets SAIL, whose reported compute capability is below
        # SM100; only require SM100a+ on non-SAIL accelerators.
        if not is_sail() and get_device_sm() < 100:
            raise unittest.SkipTest("Kimi K3 compute kernels require SM100a+")

    def _run_and_check(self, num_experts, num_tokens, hidden_size, expected_m):
        torch.cuda.manual_seed_all(3)
        gate_up = (
            torch.randn(
                num_experts,
                num_tokens,
                2 * hidden_size,
                device="cuda",
                dtype=torch.float32,
            )
            * 2.0
        ).to(torch.bfloat16)
        masked_m = torch.randint(
            0, num_tokens + 1, (num_experts,), device="cuda", dtype=torch.int32
        )
        masked_m[0] = 0  # fully masked out
        masked_m[1] = num_tokens  # fully populated; the rest stay uneven

        # Sentinels let us assert masked rows are left untouched.
        output = torch.full(
            (num_experts, num_tokens, hidden_size),
            -128,
            device="cuda",
            dtype=torch.int8,
        )
        output_scale = torch.full(
            (num_experts, num_tokens, 1), -1.0, device="cuda", dtype=torch.float32
        )

        situ_and_mul_masked_post_quant_int8(
            gate_up,
            output,
            output_scale,
            masked_m,
            beta=_BETA,
            linear_beta=_LINEAR_BETA,
            topk=_TOPK,
            expected_m=expected_m,
        )
        torch.cuda.synchronize()

        expected, expected_scale = _situ_int8_reference(gate_up)
        dequantized = output.float() * output_scale.float()
        # The kernel uses fast-math tanh/sigmoid; allow drift of up to two
        # quantization steps plus a small relative term.
        error_bound = expected_scale * 2.0 + expected.abs() * 0.01
        for expert in range(num_experts):
            valid = int(masked_m[expert].item())
            self.assertTrue(
                bool(
                    (
                        (dequantized[expert, :valid] - expected[expert, :valid]).abs()
                        <= error_bound[expert, :valid]
                    ).all()
                )
            )
            self.assertTrue(bool((output[expert, :valid].float().abs() <= 127).all()))
            self.assertTrue(
                torch.allclose(
                    output_scale[expert, :valid, 0],
                    expected_scale[expert, :valid, 0],
                    rtol=0.02,
                    atol=0,
                )
            )
            self.assertTrue(bool((output[expert, valid:] == -128).all()))
            self.assertTrue(bool((output_scale[expert, valid:, 0] == -1.0).all()))

    def test_int8_values_and_scales(self):
        # hidden 1024 -> 128 threads, i.e. the per-token absmax needs the
        # cross-warp CTA reduction.
        self._run_and_check(
            num_experts=8, num_tokens=64, hidden_size=1024, expected_m=None
        )

    def test_int8_grid_stride_iterations(self):
        # expected_m=1 -> one block per expert, so every token beyond the first
        # is produced by the grid-stride loop iteration.
        self._run_and_check(
            num_experts=8, num_tokens=64, hidden_size=1024, expected_m=1
        )


class TestSituMulQuantContig(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if get_device_sm() < 100:
            raise unittest.SkipTest("UE8M0 activation scales are Blackwell-only")

    def test_ue8m0_scale_is_the_one_used_to_quantize(self):
        import triton

        from sglang.srt.layers.moe.moe_runner.deep_gemm import (
            _cast_to_e8m0_with_rounding_up,
            _situ_mul_quant_contig_kernel,
        )

        torch.manual_seed(0)
        rows, n = 64, 2048
        kg = n // _GROUP_SIZE
        gate_up = (torch.randn(rows, 2 * n, device="cuda") * 3).bfloat16()
        q = torch.empty(rows, n, device="cuda", dtype=torch.float8_e4m3fn)
        scale = torch.empty(rows, kg, device="cuda")
        _situ_mul_quant_contig_kernel[(rows,)](
            gate_up,
            q,
            scale,
            n,
            kg,
            _BETA,
            _LINEAR_BETA,
            GROUP=_GROUP_SIZE,
            KG_POW2=triton.next_power_of_2(kg),
            num_warps=8,
        )
        e8m0 = _cast_to_e8m0_with_rounding_up(scale.unsqueeze(0)).squeeze(0)
        e8m0 = e8m0.contiguous().view(torch.uint8).view(rows, -1)[:, :kg]
        read_scale = (e8m0.to(torch.int32) << 23).view(torch.float32)
        torch.testing.assert_close(read_scale, scale, rtol=0, atol=0)
        dequant = q.float() * read_scale.repeat_interleave(_GROUP_SIZE, 1)
        ref = _situ_reference(gate_up)
        self.assertLess(((dequant - ref).norm() / ref.norm()).item(), 0.05)


if __name__ == "__main__":
    unittest.main()
