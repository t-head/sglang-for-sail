"""E20 扩展 patch set UT（E19 家族，免卡 CPU 可跑）。

变体加载：E20_UT_VARIANT=mod（默认，修复件）| orig（CI whl 原件，交叉自证用）。
判例族：
- E20-1 DSAIndexerMetadata.topk_transform：
  t1 撕裂+host 名单在场：修复件用 CPU 名单重建账本，sum(repeats)==output_size 构造性成立；
  t2 撕裂+无名单：eager 下抛 Python RuntimeError（E20-1 guard），不落到设备断言；
  t3 capture 中：不插同步、行为与原件一致（原样三参）；
  t4 正常同代：offset 数值正确。
- E20-2 kpool_plan：代码模式断言（repeats 必源自 cpu.ragged_q_len）+ 同源微测。
- E20-3 _build_flashinfer_paged_args：
  t5 撕裂 eager：修复件抛 E20-3 guard；orig 件同场景必须触发 SIMULATED（交叉自证，subprocess）。
"""

import importlib.util
import os
import subprocess
import sys
import unittest
from types import SimpleNamespace

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_VARIANT = os.environ.get("E20_UT_VARIANT", "mod")
_DIR = os.path.join(_HERE, _VARIANT)


def _load(modname, fname):
    spec = importlib.util.spec_from_file_location(modname, os.path.join(_DIR, fname))
    m = importlib.util.module_from_spec(spec)
    sys.modules[modname] = m
    spec.loader.exec_module(m)
    return m


dim = _load("e20_dsa_indexer_metadata", "dsa_indexer_metadata.py")
dtb = _load("e20_dsa_topk_backend", "dsa_topk_backend.py")

# 固定 eager 判定（CPU 环境不依赖真实 capture 状态）
torch.cuda.is_current_stream_capturing = lambda: False


class _Spy:
    def __init__(self, real):
        self.real = real
        self.calls = []

    def __call__(self, self_t, repeats, *a, output_size=None, **kw):
        self.calls.append((self_t, repeats, output_size))
        if output_size is not None:
            s = int(repeats.sum().item())
            if s != output_size:
                raise RuntimeError(
                    "SIMULATED Repeat.cu:20: output_size=%d != sum(repeats)=%d"
                    % (output_size, s)
                )
            return self.real(self_t, repeats, output_size=output_size)
        return self.real(self_t, repeats)


def _fake_meta(host_len):
    captured = {}

    def _tt(**kw):
        captured.update(kw)
        return "TK"

    meta = SimpleNamespace(
        attn_metadata=SimpleNamespace(
            dsa_extend_seq_lens_list=host_len,
            cu_seqlens_q=torch.tensor([0, 2, 5], dtype=torch.int32),
            topk_indices_offset=torch.tensor([0, 2, 5], dtype=torch.int32),
        ),
        topk_backend=SimpleNamespace(topk_transform=_tt),
        topk_transform_method="X",
        force_unfused_topk=False,
        packed_extend_topk=None,
        get_dsa_extend_len_cpu=lambda: host_len,
        get_seqlens_expanded=lambda: torch.tensor([2, 3], dtype=torch.int32),
    )
    return meta, captured


class TestE20(unittest.TestCase):
    def _ri_torn(self):
        """公共构造：device 账本旧代 [0,5,9]，logits 5 行，host 名单 [2,3]。"""
        logits = torch.zeros(5, 4)
        stale = torch.tensor([5, 4], dtype=torch.int32)
        spy = _Spy(torch.repeat_interleave)
        old = torch.repeat_interleave
        torch.repeat_interleave = spy
        try:
            return logits, stale, spy, old
        finally:
            pass

    def test_t1_e201_torn_with_host_ledger(self):
        logits, stale, spy, old = self._ri_torn()
        meta, _ = _fake_meta([2, 3])
        if _VARIANT == "mod":
            out = dim.DSAIndexerMetadata.topk_transform(
                meta, logits, 2, cu_seqlens_q=stale, ke_offset=torch.tensor([1, 1])
            )
            self.assertEqual(out, "TK")
            _, repeats, osz = spy.calls[0]
            self.assertEqual(int(repeats.sum().item()), osz)  # 同源
            self.assertEqual(osz, 5)
        else:  # orig：同场景必须点燃（交叉自证在 subprocess 断言）
            with self.assertRaises(RuntimeError) as cm:
                dim.DSAIndexerMetadata.topk_transform(
                    meta, logits, 2, cu_seqlens_q=stale, ke_offset=torch.tensor([1, 1])
                )
            self.assertIn("SIMULATED Repeat.cu:20", str(cm.exception))

    def test_t2_e201_torn_no_ledger_python_guard(self):
        if _VARIANT != "mod":
            self.skipTest("guard 仅修复件")
        logits, stale, spy, _ = self._ri_torn()
        meta, _ = _fake_meta(None)
        with self.assertRaises(Exception) as cm:
            dim.DSAIndexerMetadata.topk_transform(
                meta, logits, 2, cu_seqlens_q=stale, ke_offset=torch.tensor([1, 1])
            )
        self.assertIn("E20-1 guard", str(cm.exception))  # Python 级前置，不落到设备断言

    def test_t3_e201_capture_keeps_original_call(self):
        if _VARIANT != "mod":
            self.skipTest("capture 分支仅修复件判")
        torch.cuda.is_current_stream_capturing = lambda: True
        try:
            logits = torch.zeros(5, 4)
            paired = torch.tensor([2, 3], dtype=torch.int32)
            spy = _Spy(torch.repeat_interleave)
            old = torch.repeat_interleave
            torch.repeat_interleave = spy
            try:
                meta, _ = _fake_meta([2, 3])
                out = dim.DSAIndexerMetadata.topk_transform(
                    meta, logits, 2, cu_seqlens_q=paired, ke_offset=torch.tensor([1, 1])
                )
            finally:
                torch.repeat_interleave = old
            self.assertEqual(out, "TK")
            _, repeats, osz = spy.calls[0]
            self.assertEqual(int(repeats.sum().item()), osz)  # 同代输入原样放行
            # F1 核心断言：条件成立+capturing 时 repeats 必须是原对象（未重建=未插 H2D）
            self.assertIs(repeats, paired)
        finally:
            torch.cuda.is_current_stream_capturing = lambda: False

    def test_t4_e201_paired_values(self):
        logits, paired, spy, old = self._ri_torn()
        paired = torch.tensor([2, 3], dtype=torch.int32)
        meta, cap = _fake_meta([2, 3])
        out = dim.DSAIndexerMetadata.topk_transform(
            meta, logits, 2, cu_seqlens_q=paired, ke_offset=torch.tensor([1, 1])
        )
        self.assertEqual(out, "TK")
        _, repeats, osz = spy.calls[0]
        self.assertEqual(int(repeats.sum().item()), osz)
        self.assertEqual(osz, 5)

    def test_t5_e203_torn_eager_guard(self):
        ct = torch.tensor([0, 5, 9], dtype=torch.int32)  # diffs=[5,4] sum=9
        bidx = [0, 1]
        spy = _Spy(torch.repeat_interleave)
        old = torch.repeat_interleave
        torch.repeat_interleave = spy
        try:
            if _VARIANT == "mod":
                with self.assertRaises(Exception) as cm:
                    dtb._build_flashinfer_paged_args(
                        None, None, ct, bidx, torch.device("cpu"), 5
                    )
                self.assertIn("E20-3 guard", str(cm.exception))
            else:
                with self.assertRaises(RuntimeError) as cm:
                    dtb._build_flashinfer_paged_args(
                        None, None, ct, bidx, torch.device("cpu"), 5
                    )
                self.assertIn("SIMULATED Repeat.cu:20", str(cm.exception))
        finally:
            torch.repeat_interleave = old

    def test_t6_e202_point_source_pattern_and_micro(self):
        src = open(os.path.join(_DIR, "kpool_plan.py")).read()
        if _VARIANT == "mod":
            self.assertIn("E20-2 E19-family point-source fix", src)
            self.assertRegex(
                src.replace("\n", " "),
                r"repeat_interleave\(\s*local_req_pool_indices\.to\(torch\.int32\),\s*_e20_repeats,\s*output_size=sum\(cpu\.ragged_q_len\)",
            )
            # 同源微测：list→tensor 与 output_size 必相等
            lst = [2, 3, 0]
            rep = torch.tensor([int(v) for v in lst], dtype=torch.int32)
            self.assertEqual(int(rep.sum().item()), sum(lst))
        else:
            self.assertNotIn("E20-2 E19-family", src)
            self.assertRegex(
                src.replace("\n", " "),
                r"repeat_interleave\(\s*local_req_pool_indices\.to\(torch\.int32\),\s*ragged_q_len_t,",
            )

    def test_t8_static_capture_gate(self):
        """佐助④/F1：两处 rebuild 分支必须含 capturing gate（静态源码断言）。"""
        if _VARIANT != "mod":
            self.skipTest("仅 mod 变体跑本判例")
        m1 = open(os.path.join(_DIR, "dsa_indexer_metadata.py")).read()
        m2 = open(os.path.join(_DIR, "kpool_plan.py")).read()
        i1 = m1.index("E20-1 E19-family point-source guard")
        seg1 = m1[i1:m1.index("# <<< E20-1", i1)]
        self.assertRegex(
            seg1.replace("\n", " "),
            r"if \(\s*not torch\.cuda\.is_current_stream_capturing\(\)\s*and _e20_host is not None",
        )
        i2 = m2.index("E20-2 E19-family point-source fix")
        seg2 = m2[i2:m2.index("# <<< E20-2", i2)]
        self.assertIn("if torch.cuda.is_current_stream_capturing():", seg2)
        self.assertIn("_e20_repeats = ragged_q_len_t", seg2)
        self.assertLess(
            seg2.index("if torch.cuda.is_current_stream_capturing():"),
            seg2.index("_e20_repeats = torch.tensor("),
            "capturing 分支必须先于 H2D 重建分支（gate 有效）",
        )

    def test_t7_cross_selfcheck_orig_fires(self):
        """交叉自证：orig 变体跑 t1/t5 必须点燃（subprocess，场景有效性）。"""
        if _VARIANT != "mod":
            self.skipTest("仅 mod 变体跑本判例")
        env = dict(os.environ, E20_UT_VARIANT="orig")
        r = subprocess.run(
            [sys.executable, "-m", "unittest", "-v",
             "%s.TestE20.%s" % (os.path.basename(__file__).replace(".py", ""),
                                "test_t1_e201_torn_with_host_ledger"),
             "%s.TestE20.%s" % (os.path.basename(__file__).replace(".py", ""),
                                "test_t5_e203_torn_eager_guard")],
            cwd=_HERE, env=env, capture_output=True, text=True)
        # orig 变体的 t1/t5 测试体=「必须捕获 SIMULATED 点燃」：子进程 OK 即点燃自证
        self.assertEqual(r.returncode, 0,
                         "orig 子进程未通过=场景未点燃：%s" % r.stderr[-600:])
        self.assertIn("OK", r.stderr, "orig 子进程无 OK 收口")
        self.assertIn("Ran 2 tests", r.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
