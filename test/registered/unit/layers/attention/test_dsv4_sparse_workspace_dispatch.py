"""Check sparse Prefill backend dispatch without allocating a large workspace."""

import ast
import sys
import unittest
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDsv4SparseWorkspaceDispatch(CustomTestCase):
    def test_compaction_dispatch_and_output_contracts(self):
        source = (
            Path(__file__).resolve().parents[5]
            / "python/sglang/srt/layers/attention/deepseek_v4_backend.py"
        )
        candidates = [
            fn
            for fn in ast.walk(ast.parse(source.read_text(encoding="utf-8")))
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
        for case in (
            "compact",
            "mega",
            "q8",
            "capture",
            "breakable",
            "piecewise",
            "non_ppu",
            "reuse",
        ):
            with self.subTest(case=case):
                self.check_dispatch(compiled, case)

    def check_dispatch(self, compiled, case):
        mega = case == "mega"
        q8 = case == "q8"
        capture = case == "capture"
        ppu = case != "non_ppu"
        indices = torch.zeros((1, 1), dtype=torch.int32)
        lengths = torch.ones(1, dtype=torch.int32)
        workspace = torch.zeros((1, 1, 512), dtype=torch.bfloat16)
        compact_workspace = torch.ones_like(workspace)
        remapped = torch.ones_like(indices)
        bf16_out, q8_out, fp8_out, scales = (object() for _ in range(4))
        fused = Mock(return_value=(fp8_out, scales))
        compact = Mock(return_value=(compact_workspace, remapped))
        ordinary = Mock(return_value=(bf16_out, None, None))
        quantized = Mock(return_value=(q8_out, None, None))
        flash = ModuleType("sglang.kernels.ops.attention.flash_mla_interface")
        flash.flash_mla_sparse_fwd = ordinary
        flash_q8 = ModuleType("flash_mla")
        flash_q8.flash_mla_sparse_quantized_fwd = quantized
        buffer = NS(
            get=Mock(return_value=workspace),
            can_reuse_bf16=Mock(),
            release_eager_bf16=Mock(),
        )
        ns = {
            "_is_xpu": False,
            "torch": NS(cuda=NS(is_current_stream_capturing=lambda: capture)),
            "get_platform": lambda: NS(is_ppu=ppu),
            "_current_capture_var": NS(
                get=lambda: object() if case == "breakable" else None
            ),
            "is_in_tc_piecewise_cuda_graph": lambda: case == "piecewise",
            "should_compact_sparse_prefill_kv": lambda rows, count: rows >= 4194304,
            "use_dsv4_q8kv8_sparse_prefill": lambda _: q8,
            "_q8kv8_ppu_supported": lambda: ppu,
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
            sparse_prefill_kv_buffer=buffer,
            dsv4_prefill_backend="auto",
            head_dim_v=512,
            softmax_scale=0.125,
            _forward_mega_prefill=fused,
            _q8kv8_sparse_prefill_log_emitted=True,
        )
        pool = NS(
            get_swa_key_buffer_radix=lambda _: object(),
            get_swa_key_layout=lambda: object(),
        )
        rotary = object() if mega else None
        exec(compiled, ns)

        def run():
            return ns["_forward_prefill_sparse"](
                backend,
                torch.zeros((1, 1, 1, 512)),
                0,
                0,
                None,
                pool,
                NS(positions=torch.zeros(1, dtype=torch.int32)),
                None,
                rotary,
            )

        with patch.dict(sys.modules, {flash.__name__: flash, "flash_mla": flash_q8}):
            buffer.can_reuse_bf16.side_effect = lambda rows: (
                buffer.get.called and buffer.get.call_args.args[0] >= rows
            )
            if case == "reuse":
                # A prior non-compacted forward fills a reusable workspace.
                with patch.dict(
                    ns, {"should_compact_sparse_prefill_kv": lambda *_: False}
                ):
                    self.assertIs(run(), bf16_out)
            out = run()
        if mega:
            self.assertEqual(out, (fp8_out, scales))
            self.assertIs(fused.call_args.kwargs["cos_sin_cache"], rotary)
            ordinary.assert_not_called()
            compact.assert_not_called()
            buffer.release_eager_bf16.assert_not_called()
        elif case == "compact":
            self.assertIs(out, bf16_out)
            compact.assert_called_once()
            buffer.release_eager_bf16.assert_called_once_with()
            buffer.get.assert_not_called()
            self.assertIs(ordinary.call_args.kwargs["kv"], compact_workspace)
            torch.testing.assert_close(
                ordinary.call_args.kwargs["indices"], remapped.unsqueeze(1)
            )
        else:
            self.assertIs(out, q8_out if q8 else bf16_out)
            compact.assert_not_called()
            buffer.release_eager_bf16.assert_not_called()
            fused.assert_not_called()
            attention = quantized if q8 else ordinary
            self.assertIs(attention.call_args.kwargs["kv"], workspace)
            torch.testing.assert_close(
                attention.call_args.kwargs["indices"], indices.unsqueeze(1)
            )
            if q8:
                ordinary.assert_not_called()
            self.assertEqual(
                buffer.get.call_args.kwargs["retain_for_graph"],
                ppu and case in ("capture", "breakable", "piecewise"),
            )


if __name__ == "__main__":
    unittest.main()
