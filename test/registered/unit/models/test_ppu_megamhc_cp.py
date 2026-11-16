"""CPU dispatch coverage for the PPU CP Prefill MegaMHC boundary."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestPpuMegaMhcCp(CustomTestCase):
    def test_cp_prefill_does_not_change_decode_dispatch(self):
        path = (
            Path(__file__).resolve().parents[4]
            / "python/sglang/srt/models/deepseek_v4_mhc.py"
        )
        names = {"_can_use_mega_mhc", "_post_fusion"}
        nodes = [
            node
            for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name in names
        ]
        self.assertEqual({node.name for node in nodes}, names)
        future = ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
        tree = ast.fix_missing_locations(
            ast.Module(body=[future, *nodes], type_ignores=[])
        )
        platform = SimpleNamespace(is_ppu=True, is_blackwell=False)
        forward = SimpleNamespace(sp_active=False)
        fused = Mock(return_value=("residual", "normalized", "stats"))
        scope = {
            "get_platform": lambda: platform,
            "get_forward": lambda: forward,
            "get_parallel": lambda: SimpleNamespace(attn_dp_size=1),
            "can_use_mega_mhc_prefill": lambda batch: batch.mode == "extend",
            "can_use_mega_mhc_decode": lambda batch: batch.mode in {"decode", "verify"},
            "try_mhc_mega_boundary": fused,
            "post": lambda *args: "fallback",
            "HcState": lambda *values: values,
            "HcNormed": lambda value: value,
            "HcQuantized": lambda *values: values,
            "fused_mxfp8_rows": lambda rows: False,
            "empty_deepgemm_mxfp8_pair": Mock(
                side_effect=AssertionError("unexpected FP8 allocation in this test")
            ),
        }
        exec(compile(tree, str(path), "exec"), scope)

        cfg = SimpleNamespace(
            cp_prefill=True,
            pre_from_prev=True,
            mult=4,
            hidden=5120,
            rms_eps=1e-6,
            eps=1e-6,
            sinkhorn_iters=20,
        )
        norm = SimpleNamespace(weight=object(), variance_epsilon=1e-6)
        boundary = SimpleNamespace(
            norm_fusable=True,
            hc=SimpleNamespace(
                cfg=cfg, fn=object(), scale=object(), base=object(), norm=norm
            ),
            accepts_mxfp8_deepgemm=False,
            mega_moe_input=None,
        )
        for mode, cp, ppu, sp, rows, expected in (
            ("extend", True, True, False, 128, True),
            ("extend", True, True, False, 0, False),
            ("extend", True, False, False, 128, False),
            ("extend", True, True, True, 128, False),
            ("decode", True, True, False, 32, False),
            ("verify", True, True, False, 128, False),
            ("decode", False, True, False, 32, True),
            ("verify", False, True, False, 128, True),
        ):
            with self.subTest(mode=mode, cp=cp, ppu=ppu, sp=sp, rows=rows):
                cfg.cp_prefill = cp
                platform.is_ppu = ppu
                forward.sp_active = sp
                fused.reset_mock()
                out = scope["_post_fusion"](
                    SimpleNamespace(cfg=cfg),
                    SimpleNamespace(shape=(rows,)),
                    object(),
                    ("pre", "post", "comb"),
                    boundary,
                    SimpleNamespace(mode=mode),
                )
                self.assertEqual(fused.called, expected)
                self.assertEqual(
                    out,
                    ("residual", "pre", "normalized", "stats")
                    if expected
                    else ("fallback", "pre"),
                )


if __name__ == "__main__":
    unittest.main()
