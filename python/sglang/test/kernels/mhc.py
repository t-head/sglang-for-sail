"""Pure-torch references for the mHC (hyper-connection) kernels."""

import math
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

GLM_MHC_ROWS = (1, 16, 17, 65, 80, 2048, 4096, 8192, 12500, 25000)
GLM_MHC_SEEDS = (0, 1)


@contextmanager
def glm_mhc_ppu_context(cp_size=4, dp_size=1):
    """Rank-local boundary fixture; no process groups or model checkpoint needed."""
    if not torch.cuda.is_available():
        raise unittest.SkipTest("PPU SM89 is required")
    from sglang.srt.runtime_context import get_context, get_parallel, get_platform

    platform = get_platform()
    if not platform.is_ppu or platform.device_sm != 89:
        raise unittest.SkipTest("GLM MegaMHC requires PPU SM89")
    import deep_gemm

    from sglang.kernels.ops.layernorm.mhc_mega import _mega_mhc_kernel

    # A compatible device with an old wheel must FAIL, not silently skip or
    # compare the legacy fused boundary against itself (especially rows <= 16).
    if not getattr(deep_gemm._C, "mega_mhc_h4096_fp32_variance", False):
        raise RuntimeError("DeepGEMM lacks mega_mhc_h4096_fp32_variance")
    if _mega_mhc_kernel(non_shifted=True) is None:
        raise RuntimeError("GLM requires enabled JIT DeepGEMM mega_mhc")
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with (
            get_context().override_server_args(
                tp_size=cp_size * dp_size,
                ep_size=cp_size * dp_size,
                attn_cp_size=cp_size,
                attn_dp_size=dp_size,
                enable_deterministic_inference=False,
            ),
            get_parallel().override(
                attn_cp_size=cp_size, attn_dp_size=dp_size, tp_group=None
            ),
            patch(
                "sglang.srt.models.glm5_next.apply_mhc_post_pre_boundary",
                side_effect=AssertionError("valid GLM operands must use MegaMHC"),
            ),
            torch.no_grad(),
        ):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32


class GlmMHCFixture:
    """The actual GLM model calls and original hc_post/hc_pre chain, H=4096."""

    def __init__(self, rows, seed):
        from sglang.srt.layers.layernorm import RMSNorm
        from sglang.srt.models.glm5_next import Glm5NextDecoderLayer

        gen = torch.Generator(device="cuda").manual_seed(seed)

        def rand(shape, dtype=torch.float32):
            return torch.randn(shape, device="cuda", dtype=dtype, generator=gen)

        self.inputs = dict(
            x=rand((rows, 4096), torch.bfloat16),
            residual=rand((rows, 4, 4096), torch.bfloat16),
            post=rand((rows, 4)).sigmoid() * 2,
            comb=rand((rows, 4, 4)).softmax(-1),
        )
        # Allocate only the decoder's mHC parameters, not attention/MoE weights.
        self.layer = Glm5NextDecoderLayer.__new__(Glm5NextDecoderLayer)
        torch.nn.Module.__init__(self.layer)
        self.layer.config = SimpleNamespace(
            mhc=True,
            hc_mult=4,
            hc_sinkhorn_iters=20,
            rms_norm_eps=1e-5,
            hc_eps=1e-6,
        )
        self.layer.hc_ffn_fn = rand((24, 16384)) * 0.01
        self.layer.hc_ffn_scale = torch.tensor([0.5, 0.25, 0.25], device="cuda")
        self.layer.hc_ffn_base = rand((24,)) * 0.1
        self.norm = RMSNorm(4096, eps=1e-5).cuda().bfloat16()
        self.norm.weight.data.copy_((1 + rand((4096,)) * 0.1).bfloat16())

    def readonly_tensors(self, inputs=None):
        inputs = self.inputs if inputs is None else inputs
        return (
            *inputs.values(),
            self.layer.hc_ffn_fn,
            self.layer.hc_ffn_scale,
            self.layer.hc_ffn_base,
            self.norm.weight,
        )

    def baseline(self, inputs=None):
        inputs = self.inputs if inputs is None else inputs
        x, residual, post, comb = (inputs[k] for k in ("x", "residual", "post", "comb"))
        rows = x.shape[0]
        residual = self.layer.hc_post(x, residual.flatten(1), comb.flatten(1), post)
        y, comb, post, norm_fused = self.layer.hc_ffn_pre(
            residual, self.norm.weight, 1e-5
        )
        if not norm_fused:
            y = self.norm(y)
        return dict(
            residual=residual.view(rows, 4, 4096),
            y=y,
            post=post.view(rows, 4),
            comb=comb.view(rows, 4, 4),
        )

    def mega(self, inputs=None):
        inputs = self.inputs if inputs is None else inputs
        x, residual, post, comb = (inputs[k] for k in ("x", "residual", "post", "comb"))
        result = self.layer.hc_ffn_post_pre(
            x, residual.flatten(1), comb.flatten(1), post, self.norm.weight, 1e-5
        )
        if result is None:
            raise AssertionError("valid GLM operands declined the MegaMHC boundary")
        y, residual, comb, post, norm_fused = result
        if not norm_fused:
            raise AssertionError("GLM MegaMHC must include RMSNorm")
        return dict(
            residual=residual.view(x.shape[0], 4, 4096),
            y=y,
            post=post,
            comb=comb.view(x.shape[0], 4, 4),
        )

    def reference(self, inputs=None):
        """Independent FP32 collapse variance with a BF16 numerator and bounded scratch."""
        inputs = self.inputs if inputs is None else inputs
        rows = inputs["x"].shape[0]
        outputs = dict(
            residual=torch.empty_like(inputs["residual"]),
            y=torch.empty_like(inputs["x"]),
            post=torch.empty_like(inputs["post"]),
            comb=torch.empty_like(inputs["comb"]),
        )
        for start in range(0, rows, 1024):
            sl = slice(start, start + 1024)
            x, residual, post, comb = (
                inputs[k][sl] for k in ("x", "residual", "post", "comb")
            )
            updated = (
                x.float().unsqueeze(1) * post.unsqueeze(-1)
                + torch.einsum("tij,tih->tjh", comb, residual.float())
            ).bfloat16()
            flat = updated.flatten(1).float()
            mixes = (flat @ self.layer.hc_ffn_fn.T) * torch.rsqrt(
                flat.square().mean(1) + 1e-5
            ).unsqueeze(1)
            pre, next_post, next_comb = hc_split_sinkhorn_torch(
                mixes.unsqueeze(0), self.layer.hc_ffn_scale, self.layer.hc_ffn_base
            )
            pre, next_post, next_comb = pre[0], next_post[0], next_comb[0]
            collapse = updated[:, 0].float() * pre[:, 0, None]
            for i in range(1, 4):
                collapse = torch.addcmul(
                    collapse, updated[:, i].float(), pre[:, i, None]
                )
            y = (
                collapse.bfloat16().float()
                * torch.rsqrt(collapse.square().mean(1) + 1e-5)[:, None]
                * self.norm.weight.float()
            ).bfloat16()
            for name, value in dict(
                residual=updated, y=y, post=next_post, comb=next_comb
            ).items():
                outputs[name][sl].copy_(value)
        return outputs

    def set_variance_discriminator(self):
        """Constant collapse ~1.0027: the two incorrect RMS variants round to 1,
        while FP32 variance with BF16 numerator rounds to 0.99609375.
        """
        self.inputs["x"].zero_()
        self.inputs["residual"].fill_(0.5)
        self.inputs["comb"].copy_(torch.eye(4, device="cuda"))
        self.layer.hc_ffn_fn.zero_()
        self.layer.hc_ffn_scale.zero_()
        self.layer.hc_ffn_base.zero_()
        p = 0.50135 - 1e-6
        self.layer.hc_ffn_base[:4].fill_(math.log(p / (1 - p)))
        self.norm.weight.fill_(1)


def glm_mhc_next_inputs(outputs):
    return dict(
        x=outputs["y"],
        residual=outputs["residual"],
        post=outputs["post"],
        comb=outputs["comb"],
    )


def assert_glm_mhc_close(case, actual, expected):
    """BF16 stores may straddle one rounding bin; retain the FP32 mixing bound."""
    case.assertEqual(actual.keys(), expected.keys())
    for name, want in expected.items():
        got = actual[name]
        case.assertEqual(got.shape, want.shape, name)
        case.assertEqual(got.dtype, want.dtype, name)
        for start in range(0, got.shape[0], 1024):
            a, b = got[start : start + 1024].float(), want[start : start + 1024].float()
            case.assertTrue(torch.isfinite(a).all().item(), name)
            case.assertTrue(torch.isfinite(b).all().item(), name)
            row_error = (a - b).flatten(1).norm(dim=1) / b.flatten(1).norm(
                dim=1
            ).clamp_min(1e-20)
            limit = (
                torch.finfo(got.dtype).eps / 2 if got.dtype == torch.bfloat16 else 0.002
            )
            case.assertLess(
                row_error.max().item(), limit, f"{name}, rows starting {start}"
            )
            torch.testing.assert_close(
                a, b, rtol=0.015, atol=limit, msg=lambda detail: f"{name}: {detail}"
            )


def hc_split_sinkhorn_torch(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
):
    """Pure-torch equivalent of ``mhc.hc_split_sinkhorn_kernel``.

    Layout mirrors the kernel exactly: the flattened ``mixes`` row holds ``pre``
    (hc), ``post`` (hc) and the ``comb`` matrix (hc * hc) consecutively.
    """
    b, s, _ = mixes.size()
    hc = hc_mult
    flat = mixes.reshape(-1, (2 + hc) * hc).float()
    scale = hc_scale.float()
    base = hc_base.float()

    pre = torch.sigmoid(flat[:, :hc] * scale[0] + base[:hc]) + eps
    post = 2 * torch.sigmoid(flat[:, hc : 2 * hc] * scale[1] + base[hc : 2 * hc])

    comb = flat[:, 2 * hc :] * scale[2] + base[2 * hc :]
    comb = comb.reshape(-1, hc, hc)

    # Initial row softmax (numerically stabilized) then column normalize.
    row_max = comb.amax(dim=2, keepdim=True)
    comb = torch.exp(comb - row_max)
    comb = comb / comb.sum(dim=2, keepdim=True) + eps
    comb = comb / (comb.sum(dim=1, keepdim=True) + eps)

    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=2, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=1, keepdim=True) + eps)

    pre = pre.reshape(b, s, hc).to(mixes.dtype)
    post = post.reshape(b, s, hc).to(mixes.dtype)
    comb = comb.reshape(b, s, hc, hc).to(mixes.dtype)
    return pre, post, comb
