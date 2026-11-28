"""CPU contract checks for FP8 quantization wrappers and native schemas."""

import ast
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.kernels.ops.quantization import fp8_kernel as fp8_module


def test_fp8_quant_allocates_default_output_buffers():
    x = torch.empty((2, 256), dtype=torch.bfloat16)
    with patch.object(fp8_module, "sgl_per_token_quant_fp8", create=True) as quantize:
        q, scale = fp8_module.sglang_per_token_quant_fp8(x)
    assert q.shape == (2, 256)
    assert q.dtype == torch.float8_e4m3fn
    assert scale.shape == (2, 1)
    assert scale.dtype == torch.float32
    quantize.assert_called_once_with(x, q, scale)


def test_fp8_quant_empty_batch_skips_kernel_resolution():
    from sglang.kernels.ops import quantization

    x = torch.empty((0, 256), dtype=torch.bfloat16)
    q = torch.empty((0, 256), dtype=torch.float8_e4m3fn)
    scale = torch.empty((0, 1), dtype=torch.float32)
    with patch.object(quantization, "get_kernel") as resolve:
        quantization.sgl_per_token_quant_fp8(x, q, scale)
    resolve.assert_not_called()


@pytest.mark.parametrize("group_quant", [False, True])
def test_fp8_quant_preserves_output_buffers(group_quant):
    x = torch.empty((2, 256), dtype=torch.bfloat16)
    q = torch.empty((2, 256), dtype=torch.float8_e4m3fn)
    scale = torch.empty((2, 1), dtype=torch.float32)
    quantize = (
        fp8_module.sglang_per_token_group_quant_fp8
        if group_quant
        else fp8_module.sglang_per_token_quant_fp8
    )
    args = (x, 256) if group_quant else (x,)
    with patch.object(fp8_module, "sgl_per_token_quant_fp8", create=True) as aot:
        result_q, result_s = quantize(*args, output_q=q, output_s=scale)
    assert result_q is q
    assert result_s is scale
    aot.assert_called_once_with(x, q, scale)


def test_fp8_quant_preserves_int8_output_path():
    x = torch.empty((2, 256), dtype=torch.bfloat16)
    q = torch.empty((2, 256), dtype=torch.int8)
    scale = torch.empty((2, 1), dtype=torch.float32)
    with (
        patch.object(
            fp8_module, "_per_token_group_quant_8bit_raw", return_value=(q, scale)
        ) as raw,
        patch.object(fp8_module, "sgl_per_token_quant_fp8", create=True) as aot,
    ):
        result_q, result_s = fp8_module.sglang_per_token_group_quant_fp8(
            x, 256, output_q=q, output_s=scale
        )
    assert result_q is q
    assert result_s is scale
    raw.assert_called_once_with(x, 256, dtype=torch.int8, output_q=q, output_s=scale)
    aot.assert_not_called()


@pytest.mark.parametrize(
    "binding",
    ["common_extension.cc", "common_extension_musa.cc", "common_extension_sail.cc"],
)
def test_aot_fp8_quant_schema_uses_three_tensors(binding):
    root = Path(__file__).resolve().parents[4] / "python/sglang/kernels/aot"
    text = (root / "csrc" / binding).read_text()
    schema = re.search(r'm.def\("(sgl_per_token_quant_fp8\([^"\n]+)"\)', text)
    assert schema is not None
    arguments = torch._C.parse_schema(schema.group(1)).arguments
    assert [arg.name for arg in arguments] == ["input", "output_q", "output_s"]
    assert all(str(arg.type) == "Tensor" for arg in arguments)


def test_aot_fp8_quant_python_wrapper_uses_three_tensors():
    path = (
        Path(__file__).resolve().parents[4]
        / "python/sglang/kernels/aot/python/sgl_kernel/gemm.py"
    )
    node = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "sgl_per_token_quant_fp8"
    )
    native = Mock()
    namespace = {
        "torch": SimpleNamespace(
            Tensor=torch.Tensor,
            ops=SimpleNamespace(
                sgl_kernel=SimpleNamespace(
                    sgl_per_token_quant_fp8=SimpleNamespace(default=native)
                )
            ),
        )
    }
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace
    )
    x = torch.empty((2, 256), dtype=torch.bfloat16)
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scales = torch.empty((2, 1), dtype=torch.float32)
    namespace["sgl_per_token_quant_fp8"](x, q, scales)
    native.assert_called_once_with(x, q, scales)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
