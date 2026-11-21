"""E19 M-B v2 点位同源修复 UT（免卡，CPU 可跑；在 v1 基础上按佐助评审意见书 §5 增量）。

v2 相对 v1 的测试增量：
1. 变体文件改为 mamba2_metadata.m_b_v2.py（MB_UT_VARIANT=m_b_v2|orig）。
2. 判据 a)：CPU 名单在场时 repeats 张量 dtype 必须为 torch.long（v2 直接指定，
   不再用 query_start_loc.diff() 探测 dtype）。
3. 判据 c)：新增「CPU 旧、设备新」反向撕裂场景——设备 query_start_loc 是新一代
   [0,2,9]（diff 和=9），CPU 名单是上一代 [2,3]（output_size=5）。补丁版预期：
   不崩溃，且 seq_idx 跟随旧 CPU 名单（[0,0,1,1,1]）。
   —— 该判例把意见书 §4 警示固化为可执行测试：同源化只消灭「断言可点燃」，
   若 CPU 名单本身滞后（改写链/时序问题），补丁后表现为按旧名单的静默错置而非
   崩溃拦截。语义正确性由臂A traceback + 全量验证轮双证，本 UT 只固定行为口径。
交叉自证：orig 件在两个撕裂方向（设备旧、CPU 旧）都必须触发模拟断言。
"""

import importlib.util
import os
import subprocess
import sys
import unittest
from types import SimpleNamespace

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_VARIANT = os.environ.get("MB_UT_VARIANT", "m_b_v2")  # m_b_v2 | orig
_MODULE_PATH = os.path.join(_HERE, "mamba2_metadata.%s.py" % _VARIANT)

_spec = importlib.util.spec_from_file_location("mb2_mamba2_metadata", _MODULE_PATH)
mb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mb)


def _make_args(device_qs, cpu_lens=(2, 3)):
    """构造 prepare_mixed EXTEND 路径入参（CPU 名单默认 [2,3]，output_size=5）。"""
    forward_metadata = SimpleNamespace(
        query_start_loc=torch.tensor(list(device_qs), dtype=torch.int32),
        track_ssm_conv_states=None,
        track_ssm_ssm_states=None,
        track_ssm_seq_idx=None,
        track_ssm_end_locs=None,
        track_ssm_recompute_dst=None,
        has_mamba_track_mask=None,
        mamba_cache_indices=torch.arange(len(cpu_lens), dtype=torch.int64),
        mamba_track_indices=None,
        retrieve_next_token=None,
        retrieve_next_sibling=None,
        retrieve_parent_token=None,
        track_conv_indices=None,
        track_ssm_h_src=None,
        track_ssm_h_dst=None,
        track_ssm_final_src=None,
        track_ssm_final_dst=None,
    )
    forward_batch = SimpleNamespace(
        extend_num_tokens=sum(cpu_lens),
        logical_forward_mode=SimpleNamespace(is_decode=lambda: False),
        forward_mode=SimpleNamespace(is_target_verify=lambda: False),
        spec_info=None,
        extend_seq_lens=torch.tensor(list(cpu_lens), dtype=torch.int32),
        extend_seq_lens_cpu=list(cpu_lens),
        seq_lens=torch.tensor([[p + l] for p, l in zip([0, 0], cpu_lens)], dtype=torch.int32),
        extend_prefix_lens=torch.tensor([0] * len(cpu_lens), dtype=torch.int32),
    )
    return forward_metadata, forward_batch


class _KernelSpy:
    """替代 torch.repeat_interleave：模拟 PPU/CUDA 核的行数断言并记录调用。"""

    def __init__(self, real_fn):
        self.real = real_fn
        self.calls = []

    def __call__(self, self_t, repeats, *a, output_size=None, **kw):
        self.calls.append((self_t, repeats, output_size))
        if a and not isinstance(a[0], (int,)) and not callable(a[0]):
            repeats = a[0] if torch.is_tensor(a[0]) else repeats
        rep_sum = int(repeats.sum().item()) if torch.is_tensor(repeats) else None
        if output_size is not None and rep_sum is not None and rep_sum != output_size:
            raise RuntimeError(
                "SIMULATED Repeat.cu:20 assertion: output_size=%d != sum(repeats)=%d"
                % (output_size, rep_sum)
            )
        if output_size is None:
            return self.real(self_t, repeats)
        return self.real(self_t, repeats, output_size=output_size)


def _run_prepare_mixed(args, spy):
    orig = torch.repeat_interleave
    torch.repeat_interleave = spy
    try:
        fm, fb = args
        meta = mb.Mamba2Metadata.prepare_mixed(fm, 128, fb)
        return meta, spy.calls
    finally:
        torch.repeat_interleave = orig


class TestMBv2PointSource(unittest.TestCase):
    def test_torn_device_stale_no_assertion(self):
        """正向撕裂（设备旧 [0,5,9]、CPU 新 [2,3]）：补丁件不触发行数断言，两值同源。"""
        spy = _KernelSpy(torch.repeat_interleave)
        _meta, calls = _run_prepare_mixed(_make_args([0, 5, 9]), spy)
        self.assertTrue(calls, "repeat_interleave 未被调用")
        _, repeats, out = calls[0]
        self.assertIsNotNone(out)
        self.assertEqual(int(repeats.sum().item()), out, "M-B 同源不成立")
        self.assertEqual(int(out), 5)

    def test_torn_cpu_stale_no_assertion_and_follows_cpu(self):
        """反向撕裂（CPU 旧 [2,3]、设备新 [0,2,9]）：不崩溃且 seq_idx 随旧 CPU ——
        意见书 §4 警示的可执行固化（语义登记：名单本身滞后时补丁后为静默错置）。"""
        spy = _KernelSpy(torch.repeat_interleave)
        meta, calls = _run_prepare_mixed(_make_args([0, 2, 9]), spy)
        _, repeats, out = calls[0]
        self.assertEqual(int(repeats.sum().item()), out)  # 不撕裂
        seq_idx = meta.mixed_metadata.seq_idx
        expect = torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.int32)  # 随 CPU 名单，非设备 9 行
        got = seq_idx.reshape(expect.shape) if seq_idx.dim() == 1 else seq_idx
        self.assertTrue(torch.equal(got.to(torch.int32), expect),
                        "反向场景 seq_idx 未随 CPU 名单: %s" % (got,))

    def test_paired_ledger_values(self):
        """正常场景（同代 [0,2,5] / [2,3]）：seq_idx 值正确 [0,0,1,1,1]。"""
        spy = _KernelSpy(torch.repeat_interleave)
        meta, calls = _run_prepare_mixed(_make_args([0, 2, 5]), spy)
        seq_idx = meta.mixed_metadata.seq_idx
        expect = torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.int32)
        got = seq_idx.reshape(expect.shape) if seq_idx.dim() == 1 else seq_idx
        self.assertTrue(torch.equal(got.to(torch.int32), expect),
                        "seq_idx 值不符: %s" % (got,))

    def test_repeats_dtype_is_long(self):
        """v2 判据 a)：CPU 名单在场时 repeats dtype 必须直接为 torch.long。"""
        spy = _KernelSpy(torch.repeat_interleave)
        _run_prepare_mixed(_make_args([0, 2, 5]), spy)
        _, repeats, out = spy.calls[0]
        self.assertIsNotNone(out)
        self.assertEqual(repeats.dtype, torch.long,
                         "v2 应直接 torch.long，实际 %s" % (repeats.dtype,))

    def test_cpu_none_keeps_two_arg_semantics(self):
        """CPU 名单缺失回退：不传 output_size（保持原两参 auto 语义）。"""
        fm, fb = _make_args([0, 2, 5])
        fb.extend_seq_lens_cpu = None
        spy = _KernelSpy(torch.repeat_interleave)
        _run_prepare_mixed((fm, fb), spy)
        _, repeats, out = spy.calls[0]
        self.assertIsNone(out)
        self.assertTrue(torch.is_tensor(repeats))

    def test_orig_variant_would_fire_on_torn(self):
        """交叉证据：orig 件在两个撕裂方向都必须触发模拟断言（UT 场景有效性自证）。"""
        if _VARIANT != "m_b_v2":
            self.skipTest("仅 m_b_v2 变体跑本判例")
        env = dict(os.environ, MB_UT_VARIANT="orig")
        r = subprocess.run(
            [sys.executable, "-m", "unittest", "-v",
             os.path.basename(__file__).replace(".py", "")],
            cwd=_HERE, env=env, capture_output=True, text=True)
        self.assertIn("SIMULATED Repeat.cu:20", r.stderr,
                      "orig 件未触发撕裂断言，UT 场景构造失效")
        for name in ("test_torn_device_stale_no_assertion",
                     "test_torn_cpu_stale_no_assertion_and_follows_cpu"):
            hit = [ln for ln in r.stderr.splitlines() if name in ln]
            self.assertTrue(hit and any("FAIL" in ln or "ERROR" in ln for ln in hit),
                            "orig 件撕裂方向 %s 未失败，交叉自证不足：%s"
                            % (name, r.stderr[-800:]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
