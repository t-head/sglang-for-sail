"""CPU checks for PPU-only Kimi-K3 backend restrictions."""

import argparse
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch, sentinel

import pytest
import torch

from sglang.kernels.jit.utils import arch as jit_arch
from sglang.kernels.ops.attention import kda_fused_decode
from sglang.srt.arg_groups import attention_hook
from sglang.srt.arg_groups import overrides as overrides_module
from sglang.srt.environ import envs
from sglang.srt.layers.attention.linear import kda_backend
from sglang.srt.layers.attention.linear.kernels import kda_flashkda
from sglang.srt.layers.attention.linear.utils import LinearAttnKernelBackend
from sglang.srt.layers.attention.mamba.mamba2_metadata import ForwardMetadata
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16_moe import (
    CompressedTensorsWNA16DeepGemmMoE,
)
from sglang.srt.runtime_context import get_context, override_platform
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.fixture(autouse=True)
def _runtime_context():
    with get_context().override_server_args():
        yield


class TestPpuKimiK3Constraints(CustomTestCase):
    def test_kda_dispatcher_forces_unsupported_backends_to_triton_on_ppu(self):
        triton_kernel = SimpleNamespace(supports_packed_decode=False)
        with (
            patch.object(kda_backend, "is_sail", return_value=True),
            patch.object(kda_backend, "TritonKDAKernel", return_value=triton_kernel),
            patch.object(kda_backend, "rank0_log"),
        ):
            dispatcher = kda_backend.KDAKernelDispatcher(
                LinearAttnKernelBackend.FLASHINFER,
                LinearAttnKernelBackend.NVIDIA_KDA,
                LinearAttnKernelBackend.NV_CUTEDSL,
            )

        self.assertIs(dispatcher.decode_kernel, triton_kernel)
        self.assertIs(dispatcher.extend_kernel, triton_kernel)
        self.assertIs(dispatcher.verify_kernel, triton_kernel)

    def test_kda_dispatcher_uses_one_external_flashkda_kernel(self):
        triton_kernel = SimpleNamespace(supports_packed_decode=False)
        external_kernel = object()
        with (
            patch.object(kda_backend, "is_sail", return_value=True),
            patch.object(kda_backend, "is_ppu_flashkda_supported", return_value=True),
            patch.object(kda_backend, "TritonKDAKernel", return_value=triton_kernel),
            patch.object(
                kda_flashkda, "PPUFlashKDAKernel", return_value=external_kernel
            ) as kernel_class,
            patch.object(kda_backend, "rank0_log"),
        ):
            dispatcher = kda_backend.KDAKernelDispatcher(
                LinearAttnKernelBackend.FLASHKDA,
                LinearAttnKernelBackend.FLASHKDA,
                LinearAttnKernelBackend.FLASHKDA,
                explicit_flashkda_prefill=True,
            )

        self.assertIs(dispatcher.decode_kernel, external_kernel)
        self.assertIs(dispatcher.extend_kernel, external_kernel)
        self.assertIs(dispatcher.verify_kernel, external_kernel)
        kernel_class.assert_called_once_with()

    def test_ppu_flashkda_arguments_parse_on_cpu(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        args = parser.parse_args(
            [
                "--mamba-ssm-accum-dtype",
                "float32",
                "--linear-attn-decode-backend",
                "flashkda",
                "--linear-attn-prefill-backend",
                "flashkda",
                "--linear-attn-verify-backend",
                "flashkda",
            ]
        )
        self.assertEqual(args.mamba_ssm_accum_dtype, "float32")
        self.assertEqual(args.linear_attn_decode_backend, "flashkda")
        self.assertEqual(args.linear_attn_prefill_backend, "flashkda")
        self.assertEqual(args.linear_attn_verify_backend, "flashkda")

    def test_replayssm_mode_is_validated_at_startup(self):
        with envs.SGLANG_USE_PPU_KDA_REPLAYSSM.override("invalid"):
            with self.assertRaisesRegex(ValueError, "off, native, or flashkda"):
                attention_hook.handle_linear_attn_backend(
                    ServerArgs(model_path="dummy")
                )
        with envs.SGLANG_USE_PPU_KDA_REPLAYSSM.override("native"):
            with self.assertRaisesRegex(ValueError, "enable-linear-replayssm-spec"):
                attention_hook.handle_linear_attn_backend(
                    ServerArgs(model_path="dummy")
                )

    def test_native_replayssm_rejects_long_windows_at_startup(self):
        args = ServerArgs(
            model_path="dummy",
            speculative_algorithm="NEXTN",
            enable_linear_replayssm_spec=True,
            linear_replayssm_cache_len=17,
            mamba_ssm_dtype="float32",
        )
        model_config = SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=["KimiK3LinearForCausalLM"],
                linear_attn_config={"kda_layers": [0], "gate_lower_bound": -5.0},
            )
        )
        with (
            envs.SGLANG_USE_PPU_KDA_REPLAYSSM.override("native"),
            override_platform(is_ppu=True, device_sm=89, is_sm100=False),
            patch.object(attention_hook, "model_config_of", return_value=model_config),
            patch(
                "sglang.srt.configs.hybrid_arch.hybrid_kda_config",
                return_value=SimpleNamespace(
                    kda_lower_bound=-5.0, linear_attn_config={}
                ),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "between 1 and 16"):
                attention_hook.handle_linear_attn_backend(args)

    def test_flashkda_abi_is_checked_at_startup_and_default_off_stays_lazy(self):
        args = ServerArgs(
            model_path="dummy",
            linear_attn_decode_backend="flashkda",
            enable_linear_replayssm_spec=True,
            speculative_algorithm="NEXTN",
        )
        incomplete = SimpleNamespace(
            DECODE_SUPPORTS_LOWER_BOUND=True,
            SUPPORTS_BF16_STATE_ACCUMULATION=True,
            verify_strided=lambda *args: None,
            verify_replayssm=None,
        )
        with (
            envs.SGLANG_USE_PPU_KDA_REPLAYSSM.override("flashkda"),
            override_platform(is_ppu=True, device_sm=89, is_sm100=False),
            patch.object(kda_flashkda, "_load_flash_kda", return_value=incomplete),
        ):
            with self.assertRaisesRegex(ValueError, "verify_replayssm ABI"):
                attention_hook.handle_linear_attn_backend(args)

        with (
            envs.SGLANG_USE_PPU_KDA_REPLAYSSM.override("off"),
            override_platform(is_ppu=True, device_sm=89, is_sm100=False),
            patch.object(kda_flashkda, "_load_flash_kda") as load_flash_kda,
        ):
            attention_hook.handle_linear_attn_backend(ServerArgs(model_path="dummy"))
            load_flash_kda.assert_not_called()

    def test_ppu_disables_pdl(self):
        with patch.object(jit_arch, "is_sail_runtime", return_value=True):
            self.assertFalse(jit_arch.is_arch_support_pdl.__wrapped__())

    def test_ppu_preserves_megamoe_a2a_selection(self):
        with patch("sglang.srt.utils.common.is_sail", return_value=True):
            self.assertEqual(
                overrides_module._a2a_backend_overrides(
                    SimpleNamespace(moe_a2a_backend="megamoe", enable_waterfill=False)
                ),
                {},
            )

    def test_ppu_w4a16_deep_gemm_moe_accepts_k3_situ(self):
        scheme = object.__new__(CompressedTensorsWNA16DeepGemmMoE)
        scheme.moe_runner_config = SimpleNamespace(activation="situ")
        scheme.runner = Mock()
        scheme.runner.run.return_value = sentinel.combine_input
        scheme.get_deep_gemm_quant_info = Mock(return_value=sentinel.quant_info)
        result = scheme.apply_weights(sentinel.layer, sentinel.dispatch_output)
        self.assertIs(result, sentinel.combine_input)
        scheme.get_deep_gemm_quant_info.assert_called_once_with(sentinel.layer)
        scheme.runner.run.assert_called_once_with(
            sentinel.dispatch_output, sentinel.quant_info
        )


from sglang.srt.models import kimi_k3


def test_kda_dispatcher_forces_unsupported_backends_to_triton_on_ppu():
    triton_kernel = type("TritonKernel", (), {"supports_packed_decode": False})()
    with (
        patch.object(kda_backend, "is_sail", return_value=True),
        patch.object(kda_backend, "TritonKDAKernel", return_value=triton_kernel),
        patch.object(kda_backend, "rank0_log"),
    ):
        dispatcher = kda_backend.KDAKernelDispatcher(
            LinearAttnKernelBackend.FLASHINFER,
            LinearAttnKernelBackend.NVIDIA_KDA,
            LinearAttnKernelBackend.NV_CUTEDSL,
        )

    assert dispatcher.decode_kernel is triton_kernel
    assert dispatcher.extend_kernel is triton_kernel
    assert dispatcher.verify_kernel is triton_kernel


def test_kda_dispatcher_honors_flashkda_prefill_backend_on_ppu():
    triton_kernel = type("TritonKernel", (), {"supports_packed_decode": False})()
    flashkda_kernel = object()
    with (
        patch.object(kda_backend, "is_sail", return_value=True),
        patch.object(kda_backend, "TritonKDAKernel", return_value=triton_kernel),
        patch.object(
            kda_flashkda, "FlashKDAKernel", return_value=flashkda_kernel
        ) as kernel_class,
        patch.object(kda_backend, "rank0_log"),
    ):
        dispatcher = kda_backend.KDAKernelDispatcher(
            LinearAttnKernelBackend.TRITON,
            LinearAttnKernelBackend.FLASHKDA,
            LinearAttnKernelBackend.TRITON,
        )

    assert dispatcher.decode_kernel is triton_kernel
    assert dispatcher.extend_kernel is flashkda_kernel
    assert dispatcher.verify_kernel is triton_kernel
    kernel_class.assert_called_once_with()


def test_kda_extend_dispatcher_normalizes_kernel_return_contract():
    """Aligned-track batches request no intermediate states; CuteDSL still
    returns ``(out, None)`` while Triton/FlashKDA return a bare tensor. Both
    must come back as ``(out, h-or-None)`` so ``forward_extend`` can unpack
    unconditionally instead of leaving a tuple in tensor ops."""
    triton_kernel = type("TritonKernel", (), {"supports_packed_decode": False})()
    out = torch.zeros(1, 2, 4)
    with (
        patch.object(kda_backend, "is_sail", return_value=False),
        patch.object(kda_backend, "TritonKDAKernel", return_value=triton_kernel),
        patch.object(kda_backend, "rank0_log"),
    ):
        dispatcher = kda_backend.KDAKernelDispatcher(
            LinearAttnKernelBackend.TRITON,
            LinearAttnKernelBackend.TRITON,
            LinearAttnKernelBackend.TRITON,
        )

    extend_kwargs = dict(
        ssm_states=sentinel.ssm_states,
        cache_indices=sentinel.cache_indices,
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        return_intermediate_states=False,
    )

    dispatcher.extend_kernel = SimpleNamespace(
        supports_safe_gate=True, extend=lambda *a, **k: (out, None)
    )
    got, h = dispatcher.extend(
        torch.zeros(1, 2, 4),
        sentinel.k,
        sentinel.v,
        sentinel.g,
        sentinel.beta,
        **extend_kwargs,
    )
    assert got is out
    assert h is None

    dispatcher.extend_kernel = SimpleNamespace(
        supports_safe_gate=True, extend=lambda *a, **k: out
    )
    got, h = dispatcher.extend(
        torch.zeros(1, 2, 4),
        sentinel.k,
        sentinel.v,
        sentinel.g,
        sentinel.beta,
        **extend_kwargs,
    )
    assert got is out
    assert h is None


class TestFlashKDAStridedInputs(CustomTestCase):
    def test_backend_layout_and_state_writeback(self):
        """Packed projection views must avoid PLA copies but remain valid for
        the contiguous-only external backend, including output/state writes.
        """
        num_tokens, num_heads, head_dim = 64, 2, 128
        qkv = (
            torch.arange(num_tokens * 3 * num_heads * head_dim, dtype=torch.float32)
            .reshape(1, num_tokens, 3 * num_heads, head_dim)
            .to(torch.bfloat16)
        )
        q, k, v = qkv.split(num_heads, dim=2)
        gate_beta = (
            torch.arange(num_tokens * num_heads * (head_dim + 1), dtype=torch.float32)
            .reshape(1, num_tokens, -1)
            .to(torch.bfloat16)
        )
        g = gate_beta[..., : num_heads * head_dim].unflatten(-1, (num_heads, head_dim))
        beta = gate_beta[..., num_heads * head_dim :]
        inputs = dict(q=q, k=k, v=v, g=g, beta=beta)
        for tensor in inputs.values():
            self.assertFalse(tensor.is_contiguous())

        initial_pool = torch.arange(5, dtype=torch.float32).view(5, 1, 1, 1)
        initial_pool = initial_pool.expand(5, num_heads, head_dim, head_dim).clone()
        cache_indices = torch.tensor([3, 1], dtype=torch.int32)
        expected_states = initial_pool.clone()
        expected_states[cache_indices] += 1
        expected_output = q + k + v + g + beta.unsqueeze(-1)

        for use_pla, indexed in ((True, True), (True, False), (False, False)):
            with self.subTest(use_pla=use_pla, indexed=indexed):
                states = initial_pool.clone()

                def run_kernel(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    scale,
                    out,
                    A_log,
                    dt_bias,
                    lower_bound,
                    *,
                    initial_state=None,
                    final_state=None,
                    state_pool=None,
                    state_indices=None,
                    **kwargs,
                ):
                    for name, tensor in dict(q=q, k=k, v=v, g=g, beta=beta).items():
                        source = inputs[name]
                        self.assertTrue(torch.equal(tensor.reshape_as(source), source))
                        if use_pla:
                            self.assertEqual(tensor.data_ptr(), source.data_ptr())
                            self.assertEqual(tensor.stride(), source.stride()[1:])
                        else:
                            self.assertTrue(tensor.is_contiguous())

                    self.assertTrue(out.is_contiguous())
                    out.copy_(q + k + v + g + beta.unsqueeze(-1))
                    if indexed:
                        self.assertIs(state_pool, states)
                        self.assertIs(state_indices, cache_indices)
                        self.assertIsNone(initial_state)
                        self.assertIsNone(final_state)
                        state_pool[state_indices] += 1
                    else:
                        self.assertIsNone(state_pool)
                        self.assertTrue(
                            torch.equal(initial_state, initial_pool[cache_indices])
                        )
                        final_state.copy_(initial_state + 1)

                fwd = Mock(side_effect=run_kernel)
                pla_module = ModuleType("pla")
                prefill_module = ModuleType("pla.prefill")
                flashkda_module = ModuleType("pla.prefill.flashkdapro")
                pla_module.prefill = prefill_module
                prefill_module.flashkdapro = flashkda_module
                flashkda_module.flashkda_fwd = fwd

                query_start_loc = torch.tensor([0, 19, 64], dtype=torch.int32)
                metadata = (
                    ForwardMetadata(
                        query_start_loc=query_start_loc,
                        mamba_cache_indices=cache_indices,
                    )
                    if indexed
                    else None
                )
                with (
                    patch.object(kda_flashkda, "is_sail", return_value=use_pla),
                    patch.object(
                        kda_flashkda,
                        "_load_flash_kda",
                        return_value=SimpleNamespace(fwd=fwd),
                    ),
                    patch.dict(
                        sys.modules,
                        {
                            "pla": pla_module,
                            "pla.prefill": prefill_module,
                            "pla.prefill.flashkdapro": flashkda_module,
                        },
                    ),
                ):
                    output = kda_flashkda.FlashKDAKernel()._flashkda_extend(
                        **inputs,
                        ssm_states=states,
                        cache_indices=cache_indices,
                        query_start_loc=query_start_loc,
                        A_log=torch.zeros(1, 1, num_heads, 1),
                        dt_bias=torch.zeros(num_heads * head_dim),
                        lower_bound=-5.0,
                        beta_is_raw=True,
                        extend_seq_lens_cpu=[19, 45],
                        req_pool_indices_cpu=cache_indices.to(torch.int64),
                        pla_prefill_metadata=metadata,
                    )

                fwd.assert_called_once()
                self.assertEqual(output.shape, q.shape)
                self.assertTrue(torch.equal(output, expected_output))
                self.assertTrue(torch.equal(states, expected_states))


@pytest.mark.parametrize(
    "extend_kwargs, device_offsets, expected_raw",
    [
        ({}, [0, 64], None),
        ({"return_intermediate_states": False}, [0, 64], None),
        ({"extend_seq_lens_cpu": [64]}, [0, 64], (0, 64)),
        ({"extend_seq_lens_cpu": [19]}, [0, 19], (0, 19)),
        ({"extend_seq_lens_cpu": [19, 45]}, [0, 19, 64], (0, 19, 64)),
        (
            {
                "cache_offsets": True,
                "extend_seq_lens_cpu": [19, 45],
                "beta_is_raw": True,
            },
            [0, 19, 64],
            (0, 19, 64),
        ),
        (
            {
                "cache_offsets": True,
                "extend_seq_lens_cpu": [32, 32],
                "state_slots": [0, 0],
                "warmup": True,
            },
            [0, 32, 64],
            (0, 32, 64),
        ),
        (
            {
                "cache_offsets": True,
                "extend_seq_lens_cpu": [32, 32],
                "state_slots": [1, 1],
            },
            [0, 32, 64],
            (0, 32, 64),
        ),
        (
            {
                "cache_offsets": True,
                "extend_seq_lens_cpu": [32, 32],
                "beta_is_raw": True,
                "head_major_beta": True,
            },
            [0, 32, 64],
            (0, 32, 64),
        ),
        ({"extend_seq_lens_cpu": (45, 19)}, [0, 45, 64], (0, 45, 64)),
        (
            {"cache_offsets": True, "extend_seq_lens_cpu": [0, 64]},
            [0, 0, 64],
            (0, 0, 64),
        ),
        ({"extend_seq_lens_cpu": [64, 0]}, [0, 64], None),
        ({"extend_seq_lens_cpu": [4096]}, [0, 19], None),
        ({"extend_seq_lens_cpu": [19]}, [0, 64], None),
        ({"extend_seq_lens_cpu": []}, [0, 64], None),
        ({"extend_seq_lens_cpu": [64.0]}, [0, 64], None),
        ({"extend_seq_lens_cpu": [-1, 65]}, [0, 19, 64], None),
        ({"extend_seq_lens_cpu": torch.tensor([64])}, [0, 64], None),
    ],
)
@pytest.mark.parametrize("capturing", [False, True])
def test_ppu_flashkda_dispatches_to_platform_operator(
    extend_kwargs, device_offsets, expected_raw, capturing
):
    num_heads = 12
    head_dim = 128
    packed_tokens = device_offsets[-1]
    q = torch.zeros((1, packed_tokens, num_heads, head_dim), dtype=torch.bfloat16)
    k = torch.zeros_like(q)
    v = torch.zeros_like(q)
    g = torch.zeros_like(q)
    extend_kwargs = extend_kwargs.copy()
    cache_offsets = extend_kwargs.pop("cache_offsets", False)
    warmup = extend_kwargs.pop("warmup", False)
    state_slots = extend_kwargs.pop("state_slots", None)
    head_major_beta = extend_kwargs.pop("head_major_beta", False)
    indexed = (
        cache_offsets
        and not capturing
        and state_slots is None
        and all(length > 0 for length in extend_kwargs["extend_seq_lens_cpu"])
    )
    if head_major_beta:
        beta = torch.full((1, num_heads, packed_tokens), 0.5, dtype=torch.bfloat16)
        beta = beta.transpose(1, 2)
    elif extend_kwargs.get("beta_is_raw"):
        beta = torch.full((1, packed_tokens, 2 * num_heads), 0.5, dtype=torch.bfloat16)[
            ..., :num_heads
        ]
    else:
        beta = torch.full((1, packed_tokens, num_heads), 0.5, dtype=torch.float32)
    num_sequences = len(device_offsets) - 1
    ssm_states = torch.zeros(
        (num_sequences + 1, num_heads, head_dim, head_dim), dtype=torch.float32
    )
    cache_indices = (
        torch.arange(1, num_sequences + 1, dtype=torch.int32)
        if state_slots is None
        else torch.tensor(state_slots, dtype=torch.int32)
    )
    req_pool_indices_cpu = None if warmup else cache_indices.to(torch.int64)
    device_cu = torch.tensor(device_offsets, dtype=torch.int32)
    # CPU backing storage, with device metadata/capture flags mocked explicitly.
    query_start_loc = SimpleNamespace(is_cuda=True, numel=device_cu.numel)
    metadata = (
        ForwardMetadata(
            query_start_loc=query_start_loc, mamba_cache_indices=cache_indices
        )
        if cache_offsets
        else None
    )

    def flashkda_mock(**kwargs):
        kwargs["out"].fill_(0.25)
        if "state_pool" in kwargs:
            kwargs["state_pool"][kwargs["state_indices"]] += 1
        else:
            kwargs["final_state"].copy_(kwargs["initial_state"] + 1)

    flashkda_fwd = Mock(side_effect=flashkda_mock)
    pla_module = ModuleType("pla")
    prefill_module = ModuleType("pla.prefill")
    flashkda_module = ModuleType("pla.prefill.flashkdapro")
    pla_module.prefill = prefill_module
    prefill_module.flashkdapro = flashkda_module
    flashkda_module.flashkda_fwd = flashkda_fwd

    with (
        patch.object(kda_flashkda, "is_sail", return_value=True),
        patch.object(kda_flashkda, "_load_flash_kda") as load_flash_kda,
        patch.object(
            torch.cuda, "is_current_stream_capturing", return_value=capturing
        ) as capture_check,
        patch.object(
            kda_flashkda, "_pla_host_offsets", wraps=kda_flashkda._pla_host_offsets
        ) as build_offsets,
        envs.SGLANG_SAIL_PLA_CUDA.override(True),
        patch.dict(
            sys.modules,
            {
                "pla": pla_module,
                "pla.prefill": prefill_module,
                "pla.prefill.flashkdapro": flashkda_module,
            },
        ),
    ):
        kernel = kda_flashkda.FlashKDAKernel()
        # A second layer must reuse the tuple built by the first eligible layer.
        layers = 2 if indexed else 1
        for _ in range(layers):
            output = kernel.extend(
                q,
                k,
                v,
                g,
                beta,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                A_log=torch.zeros((1, 1, num_heads, 1), dtype=torch.float32),
                dt_bias=torch.zeros(num_heads * head_dim, dtype=torch.float32),
                lower_bound=-5.0,
                req_pool_indices_cpu=req_pool_indices_cpu,
                pla_prefill_metadata=metadata,
                **extend_kwargs,
            )

    assert isinstance(output, torch.Tensor)
    assert output.shape == q.shape
    assert torch.all(output == 0.25)
    expected_states = torch.zeros_like(ssm_states)
    expected_states[cache_indices] = layers
    torch.testing.assert_close(ssm_states, expected_states)
    load_flash_kda.assert_not_called()
    assert flashkda_fwd.call_count == capture_check.call_count == layers
    assert build_offsets.call_count == (0 if capturing else 1)
    if metadata is not None:
        assert metadata.pla_prefill_host_offsets == (expected_raw if indexed else None)
    kwargs = flashkda_fwd.call_args.kwargs
    assert kwargs["q"].shape == (packed_tokens, num_heads, head_dim)
    assert kwargs["beta"].shape == (packed_tokens, num_heads)
    assert kwargs["beta"].dtype == torch.bfloat16
    assert kwargs["A_log"].shape == (num_heads,)
    assert kwargs["dt_bias"].shape == (num_heads, head_dim)
    assert kwargs["cu_seqlens"] is query_start_loc
    if indexed:
        assert kwargs["state_pool"] is ssm_states
        assert kwargs["state_indices"] is cache_indices
        assert "initial_state" not in kwargs
    else:
        assert "state_pool" not in kwargs
    assert kwargs["raw_cu_seqlens"] == (None if capturing else expected_raw)
    if extend_kwargs.get("beta_is_raw"):
        assert kwargs["beta"].data_ptr() == beta.data_ptr()
        assert kwargs["beta"].stride() == beta.stride()[-2:]


@pytest.mark.parametrize(
    "lengths, num_tokens, expected",
    [
        ([8192], 8192, (0, 8192)),
        ([19, 200], 219, (0, 19, 219)),
        ([200, 19], 219, (0, 200, 219)),
        ([0, 19], 19, (0, 0, 19)),
        ([True, 18], 19, None),
        ([torch.tensor(19)], 19, None),
    ],
)
def test_pla_host_offsets_never_reads_device_values(lengths, num_tokens, expected):
    query = SimpleNamespace(
        is_cuda=True,
        numel=lambda: len(lengths) + 1,
        cpu=Mock(side_effect=AssertionError("unexpected D2H")),
        tolist=Mock(side_effect=AssertionError("unexpected D2H")),
        item=Mock(side_effect=AssertionError("unexpected D2H")),
    )
    assert kda_flashkda._pla_host_offsets(lengths, query, num_tokens) == expected
    query.cpu.assert_not_called()
    query.tolist.assert_not_called()
    query.item.assert_not_called()


def test_pla_host_offsets_rebuilds_from_eager_break_lengths():
    query = SimpleNamespace(is_cuda=True, numel=lambda: 3)
    # Breakable attention executes eagerly on replay: recompute from this call's
    # lengths rather than retaining the tuple used during graph warmup.
    lengths = [19, 45]
    assert kda_flashkda._pla_host_offsets(lengths, query, 64) == (0, 19, 64)
    lengths[:] = [45, 19]
    assert kda_flashkda._pla_host_offsets(lengths, query, 64) == (0, 45, 64)


def test_cuda_flashkda_does_not_receive_pla_host_offsets():
    kernel = kda_flashkda.FlashKDAKernel()
    q = torch.zeros((1, 64, 4, 128), dtype=torch.bfloat16)
    v = torch.zeros((1, 4, 64, 128), dtype=torch.bfloat16).transpose(1, 2)
    beta = torch.zeros((1, 64, 4), dtype=torch.bfloat16)
    states = torch.zeros((1, 4, 128, 128), dtype=torch.float32)
    external = SimpleNamespace(fwd=Mock())
    with (
        patch.object(kda_flashkda, "is_sail", return_value=False),
        patch.object(kda_flashkda, "_load_flash_kda", return_value=external),
        patch.object(kda_flashkda, "_pla_host_offsets") as host_offsets,
    ):
        output = kernel._flashkda_extend(
            q,
            q.clone(),
            v,
            q.clone(),
            beta,
            ssm_states=states,
            cache_indices=torch.tensor([0], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 64], dtype=torch.int32),
            A_log=torch.zeros(4),
            dt_bias=torch.zeros(4 * 128),
            lower_bound=-5.0,
            beta_is_raw=True,
            extend_seq_lens_cpu=[64],
        )
    host_offsets.assert_not_called()
    external.fwd.assert_called_once()
    assert output.stride() == v.stride()
    assert external.fwd.call_args.args[2].is_contiguous()
    assert external.fwd.call_args.args[6] is output
    assert external.fwd.call_args.kwargs["cu_seqlens"].dtype == torch.int64
    assert set(external.fwd.call_args.kwargs) == {
        "initial_state",
        "final_state",
        "cu_seqlens",
    }


@pytest.mark.parametrize(
    "track_ssm, lower_bound, is_spec_decode, pla_enabled",
    [
        (True, -5.0, False, True),
        (False, None, False, True),
        (False, -5.0, True, True),
        (False, -5.0, False, False),
    ],
)
def test_ppu_flashkda_fallback_preserves_return_contract(
    track_ssm, lower_bound, is_spec_decode, pla_enabled
):
    kernel = kda_flashkda.FlashKDAKernel()
    output = torch.zeros((1, 64, 12, 128), dtype=torch.bfloat16)
    result = (output, sentinel.intermediate_states) if track_ssm else output
    track_state = sentinel.track_state if track_ssm else None
    track_chunk_idx = sentinel.track_chunk_idx if track_ssm else None
    metadata = ForwardMetadata(
        query_start_loc=sentinel.query_start_loc,
        mamba_cache_indices=sentinel.cache_indices,
    )

    with (
        patch.object(kda_flashkda, "is_sail", return_value=True),
        patch.object(kernel, "_flashkda_extend") as fused_extend,
        envs.SGLANG_SAIL_PLA_CUDA.override(pla_enabled),
        patch.object(kda_flashkda, "_pla_host_offsets") as build_offsets,
        patch(
            "sglang.kernels.ops.attention.fla.kda.chunk_kda", return_value=result
        ) as chunk_kda,
    ):
        actual = kernel.extend(
            sentinel.q,
            sentinel.k,
            sentinel.v,
            sentinel.g,
            sentinel.beta,
            ssm_states=sentinel.ssm_states,
            cache_indices=sentinel.cache_indices,
            query_start_loc=sentinel.query_start_loc,
            A_log=sentinel.A_log,
            dt_bias=sentinel.dt_bias,
            lower_bound=lower_bound,
            beta_is_raw=True,
            is_spec_decode=is_spec_decode,
            return_intermediate_states=track_ssm,
            extend_seq_lens_cpu=[64],
            pla_prefill_metadata=metadata,
            track_state=track_state,
            track_chunk_idx=track_chunk_idx,
        )

    assert actual is result
    fused_extend.assert_not_called()
    build_offsets.assert_not_called()
    assert metadata.pla_prefill_host_offsets is None
    chunk_kda.assert_called_once_with(
        q=sentinel.q,
        k=sentinel.k,
        v=sentinel.v,
        g=sentinel.g,
        beta=sentinel.beta,
        initial_state=sentinel.ssm_states,
        initial_state_indices=sentinel.cache_indices,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=sentinel.query_start_loc,
        A_log=sentinel.A_log,
        dt_bias=sentinel.dt_bias,
        lower_bound=lower_bound,
        beta_is_raw=True,
        output_intermediate_states=track_ssm,
        track_state=track_state,
        track_chunk_idx=track_chunk_idx,
    )


def test_ppu_flashkda_only_bypasses_cuda_sequence_length_fallback():
    kernel = kda_flashkda.FlashKDAKernel()
    short_query_start_loc = torch.tensor([0, 48], dtype=torch.int32)
    long_query_start_loc = torch.tensor([0, 4096], dtype=torch.int32)

    with patch.object(kda_flashkda, "is_sail", return_value=True):
        assert not kernel._should_fall_back(-5.0, False, short_query_start_loc, [48])
        assert not kernel._should_fall_back(-5.0, False, long_query_start_loc, [4096])
        assert kernel._should_fall_back(None, False, short_query_start_loc, [48])
        assert kernel._should_fall_back(-5.0, True, short_query_start_loc, [48])

    with patch.object(kda_flashkda, "is_sail", return_value=False):
        assert kernel._should_fall_back(-5.0, False, short_query_start_loc, [48])
        assert kernel._should_fall_back(-5.0, False, long_query_start_loc, [4096])


def test_ppu_pla_prefill_gate_follows_sail_pla_gate():
    with patch.object(kda_flashkda, "is_sail", return_value=True):
        with envs.SGLANG_SAIL_PLA_CUDA.override(True):
            assert kda_flashkda.ppu_pla_prefill_gate_open()
        with envs.SGLANG_SAIL_PLA_CUDA.override(False):
            assert not kda_flashkda.ppu_pla_prefill_gate_open()

    # PLA prefill is PPU-only; other platforms use the flash_kda package.
    with patch.object(kda_flashkda, "is_sail", return_value=False):
        with envs.SGLANG_SAIL_PLA_CUDA.override(True):
            assert not kda_flashkda.ppu_pla_prefill_gate_open()


def test_closed_pla_gate_falls_back_to_triton_on_ppu():
    kernel = kda_flashkda.FlashKDAKernel()
    with (
        patch.object(kda_flashkda, "is_sail", return_value=True),
        envs.SGLANG_SAIL_PLA_CUDA.override(False),
        patch.object(kernel, "_flashkda_extend") as fused_extend,
        patch(
            "sglang.kernels.ops.attention.fla.kda.chunk_kda",
            return_value=sentinel.triton_output,
        ) as chunk_kda,
    ):
        actual = kernel.extend(
            sentinel.q,
            sentinel.k,
            sentinel.v,
            sentinel.g,
            sentinel.beta,
            ssm_states=sentinel.ssm_states,
            cache_indices=sentinel.cache_indices,
            query_start_loc=sentinel.query_start_loc,
            A_log=sentinel.A_log,
            dt_bias=sentinel.dt_bias,
            lower_bound=-5.0,
        )

    assert actual is sentinel.triton_output
    fused_extend.assert_not_called()
    chunk_kda.assert_called_once()


def _hf_model_config(linear_attn_config):
    return SimpleNamespace(
        hf_config=SimpleNamespace(linear_attn_config=linear_attn_config)
    )


def test_ppu_pla_prefill_default_is_per_runner_and_kda_only():
    with patch.object(kda_backend, "is_sail", return_value=True):
        with envs.SGLANG_SAIL_PLA_CUDA.override(True):
            # KDA runner (target): opts in to the SAIL PLA prefill kernel.
            assert (
                kda_backend.ppu_pla_prefill_default(
                    _hf_model_config({"kda_layers": [0, 1]})
                )
                == "flashkda"
            )
            # Heterogeneous spec draft (GDN) must NOT inherit it: GDN has no
            # FlashKDA kernel and GDNKernelDispatcher rejects that backend.
            assert kda_backend.ppu_pla_prefill_default(_hf_model_config(None)) is None
        # The PLA master gate off turns the default off (same gate as decode).
        with envs.SGLANG_SAIL_PLA_CUDA.override(False):
            assert (
                kda_backend.ppu_pla_prefill_default(
                    _hf_model_config({"kda_layers": [0, 1]})
                )
                is None
            )
    # PLA prefill is PPU-only.
    with (
        patch.object(kda_backend, "is_sail", return_value=False),
        envs.SGLANG_SAIL_PLA_CUDA.override(True),
    ):
        assert (
            kda_backend.ppu_pla_prefill_default(
                _hf_model_config({"kda_layers": [0, 1]})
            )
            is None
        )


def test_pla_prefill_default_resolves_per_runner_and_explicit_backend_wins():
    from sglang.srt.layers.attention.linear import utils as linear_utils

    def _resolve(prefill_default, explicit=None):
        mamba = SimpleNamespace(
            linear_attn_backend="triton",
            linear_attn_prefill_backend=explicit,
            linear_attn_decode_backend=None,
            linear_attn_verify_backend=None,
        )
        with (
            patch.object(
                linear_utils, "get_exec", return_value=SimpleNamespace(mamba=mamba)
            ),
            patch.object(linear_utils, "rank0_log"),
        ):
            return linear_utils.resolve_linear_attn_backends(
                prefill_default=prefill_default
            )

    # KDA runner: the per-runner default reaches the dispatcher.
    assert _resolve("flashkda").prefill is LinearAttnKernelBackend.FLASHKDA
    # GDN draft runner in the same process keeps its own default (Triton).
    assert _resolve(None).prefill.is_triton()
    # Explicit --linear-attn-prefill-backend still wins over the default.
    assert _resolve("flashkda", explicit="triton").prefill.is_triton()


def test_kda_track_path_requests_intermediate_states_only_for_unaligned_rows():
    from sglang.srt.layers.attention.linear.kda_backend import (
        needs_intermediate_states,
    )

    # Chunk-aligned tracked rows snapshot the final state from ssm_states, so
    # they must not force kernels that only exist on the per-chunk-state path.
    assert not needs_intermediate_states(True, torch.empty(0, dtype=torch.int64))
    # Chunk-unaligned tracked rows do read h at the last chunk boundary.
    assert needs_intermediate_states(True, torch.tensor([3], dtype=torch.int64))
    # Untracked batches and metadata without h rows never ask for h.
    assert not needs_intermediate_states(False, torch.tensor([3], dtype=torch.int64))
    assert not needs_intermediate_states(True, None)


def _fused_decode_coverage_inputs(num_heads, dtype=torch.bfloat16):
    head_dim = 128
    seg = num_heads * head_dim
    batch_size = 2
    num_slots = 3
    return (
        torch.empty(batch_size, 3 * seg, dtype=dtype),
        torch.empty(batch_size, seg, dtype=dtype),
        torch.empty(batch_size, num_heads, dtype=dtype),
        torch.empty(num_slots, 3, 3 * seg, dtype=dtype),
        torch.empty(num_slots, num_heads, head_dim, head_dim, dtype=torch.float32),
        torch.arange(batch_size, dtype=torch.int32),
        torch.empty(batch_size, seg, dtype=dtype),
    )


def test_fused_decode_uses_platform_head_and_layout_contracts():
    with (
        patch.object(kda_fused_decode, "is_sail", return_value=True),
        envs.SGLANG_SAIL_PLA_CUDA.override(True),
    ):
        for num_heads in (4, 6, 8, 12, 16, 24, 48, 64, 96):
            assert kda_fused_decode.covered(*_fused_decode_coverage_inputs(num_heads))
        assert not kda_fused_decode.covered(*_fused_decode_coverage_inputs(3))
        legacy = list(_fused_decode_coverage_inputs(12))
        # The existing SAIL route does not inspect the activation layout again.
        legacy[0] = sentinel.prepared_qkv
        assert kda_fused_decode.covered(*legacy)
        inputs = list(_fused_decode_coverage_inputs(4))
        inputs[1] = torch.empty(2, 2 * 4 * 128, dtype=torch.bfloat16)[:, : 4 * 128]
        assert not kda_fused_decode.covered(*inputs)

    with (
        patch.object(kda_fused_decode, "is_sail", return_value=False),
        envs.SGLANG_SAIL_PLA_CUDA.override(False),
    ):
        inputs = list(_fused_decode_coverage_inputs(6))
        assert kda_fused_decode.covered(*inputs)
        inputs[4] = inputs[4].unsqueeze(0)
        assert kda_fused_decode.covered(*inputs)
        assert not kda_fused_decode.covered(
            *_fused_decode_coverage_inputs(6, torch.float16)
        )
        assert not kda_fused_decode.covered(*_fused_decode_coverage_inputs(24))


def test_additional_sail_fused_decode_heads_respect_pla_switch():
    with (
        patch.object(kda_fused_decode, "is_sail", return_value=True),
        envs.SGLANG_SAIL_PLA_CUDA.override(False),
    ):
        for num_heads in (4, 6, 8, 16, 64):
            assert not kda_fused_decode.covered(
                *_fused_decode_coverage_inputs(num_heads)
            )
        assert kda_fused_decode.covered(*_fused_decode_coverage_inputs(12))


@pytest.mark.parametrize(
    "impl,fp16_covered", [("ppu", True), ("ref", False), ("1", False)]
)
def test_additional_sail_fused_decode_heads_use_reference_dtype_contract(
    monkeypatch, impl, fp16_covered
):
    monkeypatch.setenv("KDA_DECODE_IMPL", impl)
    with (
        patch.object(kda_fused_decode, "is_sail", return_value=True),
        envs.SGLANG_SAIL_PLA_CUDA.override(True),
    ):
        assert kda_fused_decode.covered(*_fused_decode_coverage_inputs(8))
        assert (
            kda_fused_decode.covered(*_fused_decode_coverage_inputs(8, torch.float16))
            is fp16_covered
        )


def test_ppu_fused_decode_dispatches_tp4_without_graph_padding_rows():
    num_heads = 24
    head_dim = 128
    seg = num_heads * head_dim
    mixed_qkv, a, b, conv_states, ssm_states, cache_indices, onorm_gate = (
        _fused_decode_coverage_inputs(num_heads)
    )
    cache_indices[1] = -1
    pla_forward = Mock(side_effect=lambda **kwargs: kwargs["out"].zero_())
    pla_module = ModuleType("pla")
    decode_module = ModuleType("pla.decode")
    kda_module = ModuleType("pla.decode.kda")
    pla_module.decode = decode_module
    decode_module.kda = kda_module
    kda_module.fused_kda_decode_mega_forward = pla_forward

    with (
        patch.object(kda_fused_decode, "is_sail", return_value=True),
        patch.dict(
            sys.modules,
            {
                "pla": pla_module,
                "pla.decode": decode_module,
                "pla.decode.kda": kda_module,
            },
        ),
    ):
        output = kda_fused_decode.kda_fused_decode(
            mixed_qkv,
            a,
            b,
            conv_states,
            torch.empty(4, seg, dtype=torch.float32),
            torch.empty(4, seg, dtype=torch.float32),
            torch.empty(4, seg, dtype=torch.float32),
            torch.empty(3 * seg, dtype=torch.float32),
            torch.empty(num_heads, dtype=torch.float32),
            torch.empty(seg, dtype=torch.float32),
            onorm_gate,
            torch.empty(head_dim, dtype=torch.float32),
            ssm_states,
            cache_indices,
            scale=head_dim**-0.5,
            onorm_eps=1e-5,
            lower_bound=-5.0,
            fused_weight=torch.empty(3, 4, seg, dtype=torch.float32),
            actual_batch_size=1,
        )

    assert output.shape == (1, 2, num_heads, head_dim)
    assert torch.count_nonzero(output[:, 1:]).item() == 0
    pla_forward.assert_called_once()
    kwargs = pla_forward.call_args.kwargs
    assert kwargs["x"].shape == (1, 3 * seg)
    assert kwargs["state_indices"].tolist() == [0]
    assert kwargs["out"].shape == (1, 1, num_heads, head_dim)


def _fused_decode_attention(num_heads):
    head_dim = 128
    seg = num_heads * head_dim
    layer = SimpleNamespace(
        conv_weights=torch.empty(3 * seg, 4, dtype=torch.float32),
        bias=None,
        A_log=torch.empty(num_heads, dtype=torch.float32),
        dt_bias=torch.empty(seg, dtype=torch.float32),
        num_v_heads=num_heads,
    )
    attention = SimpleNamespace(
        attn=layer,
        o_norm=SimpleNamespace(
            weight=torch.empty(head_dim, dtype=torch.float32), eps=1e-5
        ),
        _kda_fused_decode_ready=False,
    )
    return attention, layer


def test_ppu_prepares_fused_decode_handoff_for_k3_tp_layouts():
    with (
        patch.object(kimi_k3, "is_sail", return_value=True),
        patch.object(kimi_k3, "_is_hip", False),
    ):
        for num_heads in (12, 24, 48, 96):
            attention, layer = _fused_decode_attention(num_heads)
            kimi_k3.KimiK3DeltaAttention._prepare_fused_decode(attention)

            assert attention._kda_fused_decode_ready
            assert layer._k3_fused_decode_weight.shape == (
                3,
                4,
                num_heads * 128,
            )


def test_non_ppu_keeps_community_fused_decode_preparation_limit():
    with (
        patch.object(kimi_k3, "is_sail", return_value=False),
        patch.object(kimi_k3, "_is_hip", False),
        patch.object(kimi_k3, "rank0_log"),
    ):
        attention, layer = _fused_decode_attention(24)
        kimi_k3.KimiK3DeltaAttention._prepare_fused_decode(attention)
        assert not attention._kda_fused_decode_ready
        assert not hasattr(layer, "_k3_fused_decode_args")

        attention, layer = _fused_decode_attention(12)
        kimi_k3.KimiK3DeltaAttention._prepare_fused_decode(attention)
        assert attention._kda_fused_decode_ready
        assert not hasattr(layer, "_k3_fused_decode_weight")


def test_ppu_disables_pdl_and_preserves_megamoe_selection():
    with patch.object(jit_arch, "is_sail_runtime", return_value=True):
        assert not jit_arch.is_arch_support_pdl.__wrapped__()

    with patch("sglang.srt.utils.common.is_sail", return_value=True):
        assert (
            overrides_module._a2a_backend_overrides(
                SimpleNamespace(moe_a2a_backend="megamoe", enable_waterfill=False)
            )
            == {}
        )


def test_ppu_w4a16_deep_gemm_moe_accepts_k3_situ():
    scheme = object.__new__(CompressedTensorsWNA16DeepGemmMoE)
    scheme.moe_runner_config = SimpleNamespace(activation="situ")
    scheme.runner = Mock()
    scheme.runner.run.return_value = sentinel.combine_input
    scheme.get_deep_gemm_quant_info = Mock(return_value=sentinel.quant_info)

    result = scheme.apply_weights(sentinel.layer, sentinel.dispatch_output)

    assert result is sentinel.combine_input
    scheme.get_deep_gemm_quant_info.assert_called_once_with(sentinel.layer)
    scheme.runner.run.assert_called_once_with(
        sentinel.dispatch_output, sentinel.quant_info
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
