"""PR334: no-shift GLM H4096 boundary precision and carried-state regression.

The model method, original hc_post/hc_pre chain and independent Torch oracle
must agree. Reverting FP32 collapse variance, carrying an old pre, mutating
inputs, or replaying stale graph inputs must fail. No full-model weights needed.
"""

import unittest
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.kernels.mhc import (
    GLM_MHC_ROWS,
    GLM_MHC_SEEDS,
    GlmMHCFixture,
    assert_glm_mhc_close,
    glm_mhc_next_inputs,
    glm_mhc_ppu_context,
)
from sglang.test.test_utils import CustomTestCase

register_ppu_ci(est_time=120, suite="stage-a-test-1-gpu-ppu")


class TestGlmMegaMHCPPU(CustomTestCase):
    def assert_readonly(self, tensors, snapshots):
        for index, (tensor, snapshot) in enumerate(zip(tensors, snapshots)):
            self.assertTrue(torch.equal(tensor, snapshot), f"input {index} was written")

    def check_two_boundaries(self, rows, seed):
        fixture = GlmMHCFixture(rows, seed)
        tensors = fixture.readonly_tensors()
        snapshots = [tensor.clone() for tensor in tensors]
        native, reference, actual = (
            fixture.baseline(),
            fixture.reference(),
            fixture.mega(),
        )
        assert_glm_mhc_close(self, native, reference)
        assert_glm_mhc_close(self, actual, reference)
        assert_glm_mhc_close(self, actual, native)
        self.assert_readonly(tensors, snapshots)
        del snapshots

        next_inputs = glm_mhc_next_inputs(actual)
        tensors = fixture.readonly_tensors(next_inputs)
        snapshots = [tensor.clone() for tensor in tensors]
        native2 = fixture.baseline(next_inputs)
        reference2 = fixture.reference(next_inputs)
        actual2 = fixture.mega(next_inputs)
        assert_glm_mhc_close(self, native2, reference2)
        assert_glm_mhc_close(self, actual2, reference2)
        assert_glm_mhc_close(self, actual2, native2)
        self.assert_readonly(tensors, snapshots)

    def test_all_shapes_two_boundaries(self):
        with glm_mhc_ppu_context():
            for rows in GLM_MHC_ROWS:
                for seed in GLM_MHC_SEEDS:
                    with self.subTest(rows=rows, seed=seed):
                        self.check_two_boundaries(rows, seed)

    def test_fp32_variance_bf16_numerator(self):
        # Random Gaussian rows alone allow both denominator variants through
        # the .002 row gate. This non-tautological kernel test separates them.
        with glm_mhc_ppu_context():
            fixture = GlmMHCFixture(17, 0)
            fixture.set_variance_discriminator()
            reference = fixture.reference()
            native, actual = fixture.baseline(), fixture.mega()
            assert_glm_mhc_close(self, native, reference)
            assert_glm_mhc_close(self, actual, reference)
            pre = fixture.layer.hc_ffn_base[:4].sigmoid() + 1e-6
            collapse = torch.ones_like(reference["y"], dtype=torch.float32) * (
                0.5 * pre.sum()
            )
            bf = collapse.bfloat16().float()
            wrong_bf16_variance = (
                bf * torch.rsqrt(bf.square().mean(1) + 1e-5)[:, None]
            ).bfloat16()
            wrong_fp32_numerator = (
                collapse * torch.rsqrt(collapse.square().mean(1) + 1e-5)[:, None]
            ).bfloat16()
            self.assertTrue(torch.all(reference["y"] == 0.99609375).item())
            for wrong in (wrong_bf16_variance, wrong_fp32_numerator):
                self.assertFalse(torch.equal(reference["y"], wrong))
            torch.testing.assert_close(actual["y"], reference["y"], rtol=0, atol=0)
            torch.testing.assert_close(native["y"], reference["y"], rtol=0, atol=0)

    def test_missing_variance_capability_declines_readonly(self):
        with glm_mhc_ppu_context():
            import deep_gemm

            from sglang.kernels.ops.layernorm.mhc_mega import _mega_mhc_kernel

            fixture = GlmMHCFixture(17, 1)
            inputs = fixture.inputs
            tensors = fixture.readonly_tensors()
            snapshots = [tensor.clone() for tensor in tensors]
            # Only the installed wheel's capability is mocked, not the dispatch
            # or numerical implementation. 17 rows exclude the legacy <=16 tier.
            try:
                with patch.object(deep_gemm._C, "mega_mhc_h4096_fp32_variance", False):
                    _mega_mhc_kernel.cache_clear()
                    result = fixture.layer.hc_ffn_post_pre(
                        inputs["x"],
                        inputs["residual"].flatten(1),
                        inputs["comb"].flatten(1),
                        inputs["post"],
                        fixture.norm.weight,
                        1e-5,
                    )
                    self.assertIsNone(result)
                    self.assert_readonly(tensors, snapshots)
            finally:
                _mega_mhc_kernel.cache_clear()
            # Restoring the wheel metadata must recover actual fused numerics.
            assert_glm_mhc_close(self, fixture.mega(), fixture.reference())

    def test_dispatch_guards_and_scoped_opt_out(self):
        from sglang.srt.environ import envs
        from sglang.srt.runtime_context import get_exec, get_parallel

        with glm_mhc_ppu_context():
            fixture = GlmMHCFixture(17, 1)
            inputs = fixture.inputs

            def attempt(weight, eps):
                return fixture.layer.hc_ffn_post_pre(
                    inputs["x"],
                    inputs["residual"].flatten(1),
                    inputs["comb"].flatten(1),
                    inputs["post"],
                    weight,
                    eps,
                )

            # Seventeen rows exclude the legacy <=16 fallback, so every declined
            # guard must return None rather than reaching either fused backend.
            with get_parallel().override(
                tp_size=2, moe_ep_size=2, attn_cp_size=2, attn_dp_size=1, tp_group=None
            ):
                self.assertIsNone(attempt(fixture.norm.weight, 1e-5))
            with get_exec().deterministic.override(
                enable_deterministic_inference=True
            ):
                self.assertIsNone(attempt(fixture.norm.weight, 1e-5))
            with envs.SGLANG_DISABLE_GLM_MEGA_MHC.override(True):
                self.assertIsNone(attempt(fixture.norm.weight, 1e-5))
            for weight, eps in (
                (None, None),
                (fixture.norm.weight, None),
                (None, 1e-5),
            ):
                with self.subTest(weight=weight is not None, eps=eps):
                    self.assertIsNone(attempt(weight, eps))

            # The default remains the intended fused path after the opt-out ends.
            assert_glm_mhc_close(self, fixture.mega(), fixture.reference())

    def check_graph(self, rows):
        fixture = GlmMHCFixture(rows, 0)

        def two_boundaries():
            first = fixture.mega()
            return first, fixture.mega(glm_mhc_next_inputs(first))

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        # DeepGEMM's per-stream barriers must be initialized on the capture stream.
        with torch.cuda.stream(stream):
            for _ in range(20):
                two_boundaries()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = two_boundaries()
        graph.replay()
        torch.cuda.synchronize()
        previous = captured[0]["y"].clone()
        for seed in (100, 101):
            gen = torch.Generator(device="cuda").manual_seed(seed)
            for name, tensor in fixture.inputs.items():
                fresh = torch.randn(
                    tensor.shape,
                    device=tensor.device,
                    dtype=tensor.dtype,
                    generator=gen,
                )
                if name == "post":
                    fresh = fresh.sigmoid() * 2
                elif name == "comb":
                    fresh = fresh.softmax(-1)
                tensor.copy_(fresh)
            tensors = fixture.readonly_tensors()
            snapshots = [tensor.clone() for tensor in tensors]
            graph.replay()
            torch.cuda.synchronize()
            self.assertFalse(
                torch.equal(previous, captured[0]["y"]), "stale graph output"
            )
            for boundary in range(2):
                inputs = (
                    fixture.inputs
                    if boundary == 0
                    else glm_mhc_next_inputs(captured[0])
                )
                native = fixture.baseline(inputs)
                reference = fixture.reference(inputs)
                eager = fixture.mega(inputs)
                assert_glm_mhc_close(self, captured[boundary], native)
                assert_glm_mhc_close(self, captured[boundary], reference)
                for name in eager:
                    torch.testing.assert_close(
                        captured[boundary][name], eager[name], rtol=0, atol=0
                    )
            self.assert_readonly(tensors, snapshots)
            previous.copy_(captured[0]["y"])

    def test_graph_replay_new_inputs(self):
        # Physical multi-rank communication is outside this rank-local kernel UT.
        for cp_size, dp_size, shapes in (
            (4, 1, (1, 16, 17, 65, 80, 4096)),
            (8, 1, (65,)),
            (1, 8, (65,)),
        ):
            with glm_mhc_ppu_context(cp_size, dp_size):
                for rows in shapes:
                    with self.subTest(cp=cp_size, dp=dp_size, rows=rows):
                        self.check_graph(rows)


if __name__ == "__main__":
    unittest.main()
