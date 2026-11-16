"""Check sparse Prefill backend dispatch without allocating a large workspace."""

import ast
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDsv4SparseWorkspaceDispatch(CustomTestCase):
    def test_compaction_preserves_mega_attention_output_contract(self):
        source = (
            Path(__file__).resolve().parents[5]
            / "python/sglang/srt/layers/attention/deepseek_v4_backend.py"
        )
        candidates = [
            fn
            for fn in ast.walk(ast.parse(source.read_text()))
            if isinstance(fn, ast.FunctionDef) and fn.name == "_forward_prefill_sparse"
        ]
        self.assertEqual(len(candidates), 1)
        future = ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
        module = ast.fix_missing_locations(
            ast.Module(body=[future, *candidates], type_ignores=[])
        )
        compiled = compile(module, str(source), "exec")
        for mega in (True, False):
            with self.subTest(mega=mega):
                self.check_dispatch(compiled, mega)

    def check_dispatch(self, compiled, mega):
        indices = torch.zeros((1, 1), dtype=torch.int32)
        lengths = torch.ones(1, dtype=torch.int32)
        workspace = torch.zeros((1, 1, 512), dtype=torch.bfloat16)
        bf16_out, fp8_out, scales = object(), object(), object()
        fused = Mock(return_value=(fp8_out, scales))
        compact = Mock(return_value=(workspace, indices))
        ordinary = Mock(return_value=(bf16_out, None, None))
        flash = ModuleType("sglang.kernels.ops.attention.flash_mla_interface")
        flash.flash_mla_sparse_fwd = ordinary
        ns = {
            "_is_xpu": False,
            "torch": NS(cuda=NS(is_current_stream_capturing=lambda: False)),
            "get_platform": lambda: NS(is_ppu=True),
            "should_compact_sparse_prefill_kv": lambda rows, count: rows >= 4194304,
            "use_dsv4_q8kv8_sparse_prefill": lambda _: False,
            "compact_sparse_prefill_kv": compact,
            "dequantize_k_cache_paged_dsv4": Mock(),
        }
        backend = NS(
            forward_metadata=NS(
                sparse_prefill_cache=NS(
                    swa_token_ids=NS(shape=(4194304,)),
                    swa_page_size=256,
                    c0_combined_indices=indices,
                    c0_combined_lens=lengths,
                )
            ),
            sparse_prefill_kv_buffer=NS(get=Mock(return_value=workspace)),
            dsv4_prefill_backend="auto",
            head_dim_v=512,
            softmax_scale=0.125,
            _forward_mega_prefill=fused,
        )
        pool = NS(
            get_swa_key_buffer_radix=lambda _: object(),
            get_swa_key_layout=lambda: object(),
        )
        rotary = object() if mega else None
        exec(compiled, ns)
        with patch.dict(sys.modules, {flash.__name__: flash}):
            out = ns["_forward_prefill_sparse"](
                backend, torch.zeros((1, 1, 1, 512)), 0, 0, None, pool,
                NS(positions=torch.zeros(1, dtype=torch.int32)), None, rotary,
            )
        if mega:
            self.assertEqual(out, (fp8_out, scales))
            self.assertIs(fused.call_args.kwargs["cos_sin_cache"], rotary)
            ordinary.assert_not_called()
            compact.assert_not_called()
        else:
            self.assertIs(out, bf16_out)
            compact.assert_called_once()
            fused.assert_not_called()


if __name__ == "__main__":
    unittest.main()
