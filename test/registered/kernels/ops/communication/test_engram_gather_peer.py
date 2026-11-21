"""Peer-gather decode policy, shard ownership, and stable graph outputs."""

import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.communication import engram_gather_peer as peer_module
from sglang.test.ci.ci_register import register_cuda_ci, register_ppu_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
# backend-specific: SAIL forwards the SM80 byte-decoder/SM89 native-FP8 selector.
register_ppu_ci(est_time=30, suite="extra-test-1-gpu-ppu")

_DIM = 256
_BLOCK = 32
_ROWS = (7, 9, 11)
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _e4m3_value(bits):
    """Independent CPU oracle, without an FP8 device conversion."""
    if bits & 127 == 127:
        return float("nan")
    exponent, mantissa = (bits >> 3) & 15, bits & 7
    value = (
        mantissa / 512.0
        if exponent == 0
        else (1 + mantissa / 8.0) * 2 ** (exponent - 7)
    )
    return -value if bits & 128 else value


class TestEngramGatherPeer(CustomTestCase):
    def _check(self, mxfp4, decode_bytes):
        # Three independently allocated shards on one device exercise the peer
        # address/owner arithmetic. The model E2E supplies actual cross-rank maps.
        rows = sum(_ROWS)
        width = _DIM // 2 if mxfp4 else _DIM
        payload = (
            torch.arange(rows * width).view(rows, width)
            + 13 * torch.arange(rows)[:, None]
        ).to(torch.uint8)
        exps = torch.tensor([120, 123, 127, 129, 132, 134, 121, 130], dtype=torch.uint8)
        scales = torch.stack([exps.roll(i % 8) for i in range(rows)])
        if mxfp4:
            nibbles = torch.stack((payload & 15, payload >> 4), dim=-1).flatten(-2)
            magnitudes = torch.tensor(_E2M1)[(nibbles & 7).long()]
            values = torch.where(nibbles >= 8, -magnitudes, magnitudes)
        else:
            values = torch.tensor([_e4m3_value(i) for i in range(256)])[payload.long()]
        expected_table = (
            values * torch.exp2(scales.float() - 127).repeat_interleave(_BLOCK, dim=1)
        ).to(torch.bfloat16)
        expected_table = torch.where(expected_table == 0, 0.0, expected_table)

        weights, scale_shards, w_bases, s_bases, starts = [], [], [], [], []
        first = 0
        for size in _ROWS:
            weight = payload[first : first + size].cuda()
            scale = scales[first : first + size].cuda()
            weights.append(weight)
            scale_shards.append(scale)
            starts.append(first)
            w_bases.append(weight.data_ptr() - first * width)
            s_bases.append(scale.data_ptr() - first * (_DIM // _BLOCK))
            first += size
        starts.append(torch.iinfo(torch.int64).max)
        w_bases.append(0)
        s_bases.append(0)
        start_tensor = torch.tensor(starts, dtype=torch.int64, device="cuda")
        w_bias = torch.tensor(w_bases, dtype=torch.int64, device="cuda")
        s_bias = torch.tensor(s_bases, dtype=torch.int64, device="cuda")

        for dtype in (torch.int32, torch.int64):
            with self.subTest(mxfp4=mxfp4, decode_bytes=decode_bytes, dtype=dtype):
                # A ragged 11-column tile and a noncontiguous token stride.
                storage = torch.zeros(3, 16, dtype=dtype, device="cuda")
                ids = storage[:, :11]
                ids_cpu = (torch.arange(33).view(3, 11) * 7) % rows
                ids_cpu[0, :6] = torch.tensor([0, 6, 7, 15, 16, 26])
                ids.copy_(ids_cpu)
                out = torch.full(
                    (3, 11, _DIM), float("nan"), dtype=torch.bfloat16, device="cuda"
                )

                def launch():
                    peer_module.engram_gather_peer(
                        ids, out, w_bias, s_bias, start_tensor, _DIM, _BLOCK, mxfp4
                    )

                def check():
                    expected = expected_table[ids_cpu].cuda()
                    torch.testing.assert_close(
                        out, expected, rtol=0, atol=0, equal_nan=True
                    )
                    finite = torch.isfinite(expected)
                    self.assertTrue(
                        torch.equal(
                            out.view(torch.int16)[finite],
                            expected.view(torch.int16)[finite],
                        )
                    )

                with (
                    patch.object(peer_module, "is_sail", return_value=True),
                    patch.object(
                        peer_module,
                        "get_device_sm",
                        return_value=80 if decode_bytes else 89,
                    ),
                ):
                    # The validation script starts this test with an empty JIT cache.
                    launch()
                    check()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        launch()
                    address = out.data_ptr()
                    for shift in (1, 5):
                        ids_cpu = (ids_cpu + shift) % rows
                        ids.copy_(ids_cpu)
                        out.fill_(float("nan"))
                        graph.replay()
                        check()
                        self.assertEqual(out.data_ptr(), address)

    def test_fp8_native_decode(self):
        if peer_module.is_sail() and peer_module.get_device_sm() < 89:
            self.skipTest("Native FP8 decode is not supported on pre-SM89 SAIL")
        self._check(mxfp4=False, decode_bytes=False)

    def test_fp8_byte_decode(self):
        self._check(mxfp4=False, decode_bytes=True)

    def test_mxfp4_native_selector(self):
        self._check(mxfp4=True, decode_bytes=False)

    def test_mxfp4_byte_selector(self):
        self._check(mxfp4=True, decode_bytes=True)


if __name__ == "__main__":
    unittest.main()
