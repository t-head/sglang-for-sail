from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from sglang.srt.layers.engram import EngramEmbedding, _scale_region_bytes
from sglang.srt.layers.engram_staging import EngramRowStager
from sglang.test.test_utils import CustomTestCase

DIM = 256
BLOCK = 32


class _FakeHostTable:
    """Stands in for _HostTable in CPU tests: no mmap, no pinning, no barrier."""

    def __init__(self, layout: str):
        self.layout = layout
        self.dirty = False
        self.finished = []

    def finish_load(self, label: str):
        self.finished.append(label)


def _engram(
    total_rows: int,
    row_start: int = 0,
    rows: int = None,
    host_layout: str = None,
) -> EngramEmbedding:
    # Skip EngramEmbedding.__init__ (it needs a live runtime context for
    # get_parallel()) but still give nn.Module its bookkeeping.
    rows = total_rows if rows is None else rows
    embed = object.__new__(EngramEmbedding)
    nn.Module.__init__(embed)
    embed.dim = DIM
    embed.tp_size = total_rows // rows
    embed.rows = rows
    embed.row_start = row_start
    embed.host_table = None if host_layout is None else _FakeHostTable(host_layout)
    embed._per_channel_scale = False
    embed._int8_payload = False
    # 0523 keeps the TP peer shards optional; the table above is the device one.
    embed.peer = None
    # A shared host table covers every row; device and per-rank tables hold a shard.
    table_rows = total_rows if host_layout == "shared" else rows
    embed.weight = nn.Parameter(
        torch.zeros(table_rows, DIM, dtype=torch.float8_e4m3fn), requires_grad=False
    )
    scale_bytes = torch.zeros(_scale_region_bytes(table_rows, DIM), dtype=torch.uint8)
    embed.scale = nn.Parameter(
        scale_bytes[: table_rows * (DIM // BLOCK)]
        .view(torch.float8_e8m0fnu)
        .view(table_rows, DIM // BLOCK),
        requires_grad=False,
    )
    embed._scale_f32 = scale_bytes.view(torch.float32)[:table_rows]
    return embed


def _rows(embed: EngramEmbedding, ids: list[int]) -> torch.Tensor:
    return embed._owned_rows(torch.tensor(ids, dtype=torch.int64))


class TestEngramLoadRows(unittest.TestCase):
    def test_block_scale_lookup(self):
        embed = _engram(4)
        embed.weight.data.copy_(torch.full((4, DIM), 2.0).to(torch.float8_e4m3fn))
        embed.scale.data.view(torch.uint8).fill_(127)  # e8m0 127 is 2**0
        out = _rows(embed, [0, 3, 4])
        self.assertEqual(out.dtype, torch.bfloat16)
        expected = torch.full((DIM,), 2.0, dtype=torch.bfloat16)
        self.assertTrue(torch.equal(out[0], expected))
        self.assertTrue(torch.equal(out[1], expected))
        self.assertTrue(torch.equal(out[2], torch.zeros(DIM, dtype=torch.bfloat16)))

    def test_per_channel_scale_load(self):
        embed = _engram(4)
        embed.weight.data.copy_(torch.full((4, DIM), 2.0).to(torch.float8_e4m3fn))
        embed._load_rows(embed.scale, torch.full((4, 1), 3.5))
        self.assertTrue(embed._per_channel_scale)
        self.assertTrue(torch.equal(embed._scale_f32, torch.full((4,), 3.5)))
        out = _rows(embed, [1])
        self.assertTrue(
            torch.equal(out[0], torch.full((DIM,), 7.0, dtype=torch.bfloat16))
        )

    def test_int8_payload_load(self):
        embed = _engram(4)
        embed._load_rows(embed.scale, torch.full((4, 1), 3.5))
        embed._load_rows(embed.weight, torch.full((4, DIM), 5, dtype=torch.int8))
        self.assertTrue(embed._int8_payload)
        out = _rows(embed, [0])
        self.assertTrue(
            torch.equal(out[0], torch.full((DIM,), 17.5, dtype=torch.bfloat16))
        )

    def test_formats_coexist_across_tables(self):
        blockwise = _engram(4)
        blockwise.weight.data.copy_(torch.full((4, DIM), 2.0).to(torch.float8_e4m3fn))
        blockwise.scale.data.view(torch.uint8).fill_(127)
        channelwise = _engram(4)
        channelwise._load_rows(channelwise.scale, torch.full((4, 1), 3.5))
        channelwise._load_rows(
            channelwise.weight, torch.full((4, DIM), 2.0).to(torch.float8_e4m3fn)
        )
        self.assertFalse(blockwise._per_channel_scale)
        self.assertTrue(channelwise._per_channel_scale)
        self.assertTrue(
            torch.equal(
                _rows(blockwise, [0])[0],
                torch.full((DIM,), 2.0, dtype=torch.bfloat16),
            )
        )
        self.assertTrue(
            torch.equal(
                _rows(channelwise, [0])[0],
                torch.full((DIM,), 7.0, dtype=torch.bfloat16),
            )
        )

    def test_reload_switches_format(self):
        embed = _engram(4)
        embed._load_rows(embed.scale, torch.full((4, 1), 3.5))
        self.assertTrue(embed._per_channel_scale)
        blocks = torch.zeros(4, DIM // BLOCK, dtype=torch.uint8)
        embed._load_rows(embed.scale, blocks.view(torch.float8_e8m0fnu))
        self.assertFalse(embed._per_channel_scale)

    def test_unknown_layout_rejected(self):
        embed = _engram(4)
        with self.assertRaises(ValueError):
            embed._load_rows(embed.scale, torch.ones(4, 2))
        with self.assertRaises(ValueError):
            embed._load_rows(embed.weight, torch.ones(4, DIM))

    def test_scale_region_fits_both_formats(self):
        # dim // 32 = 8 bytes per row already covers one f32 per row.
        self.assertEqual(_scale_region_bytes(5, 256), 40)
        # dim // 32 = 2 bytes per row does not, so the f32 row scale wins.
        self.assertEqual(_scale_region_bytes(5, 64), 20)
        for rows in (0, 1, 3, 7):
            self.assertEqual(_scale_region_bytes(rows, 64) % 4, 0)

    def test_shared_layout_writes_only_own_rows(self):
        embed = _engram(8, row_start=4, rows=2, host_layout="shared")
        embed._load_rows(embed.scale, torch.full((8, 1), 0.5))
        embed._load_rows(
            embed.weight, torch.full((8, DIM), 3.0).to(torch.float8_e4m3fn)
        )
        self.assertTrue(embed._per_channel_scale)
        self.assertTrue(embed.host_table.dirty)
        self.assertTrue(torch.equal(embed._scale_f32[4:6], torch.full((2,), 0.5)))
        self.assertTrue(torch.equal(embed._scale_f32[:4], torch.zeros(4)))
        self.assertTrue(embed.weight.data[4:6].float().eq(3.0).all())
        self.assertTrue(embed.weight.data[:4].float().eq(0.0).all())
        embed.finish_load("layer 7")
        self.assertEqual(embed.host_table.finished, ["layer 7"])

    def test_per_rank_layout_keeps_block_scales(self):
        embed = _engram(8, row_start=4, rows=2, host_layout="per_rank")
        blocks = torch.full((8, DIM // BLOCK), 127, dtype=torch.uint8)
        embed._load_rows(embed.scale, blocks.view(torch.float8_e8m0fnu))
        self.assertFalse(embed._per_channel_scale)
        self.assertTrue(embed.scale.data.view(torch.uint8).eq(127).all())
        # Switching to per-channel writes this rank's rows in the f32 view.
        embed._load_rows(embed.scale, torch.full((8, 1), 0.25))
        self.assertTrue(embed._per_channel_scale)
        self.assertTrue(torch.equal(embed._scale_f32, torch.full((2,), 0.25)))

    def test_channel_scale_row_count_mismatch_rejected(self):
        embed = _engram(4)
        with self.assertRaises(ValueError):
            embed._load_rows(embed.scale, torch.ones(3, 1))

    def test_bf16_row_scale_is_widened(self):
        embed = _engram(4)
        embed._load_rows(embed.scale, torch.full((4,), 2.5, dtype=torch.bfloat16))
        self.assertTrue(embed._per_channel_scale)
        self.assertTrue(torch.equal(embed._scale_f32, torch.full((4,), 2.5)))

    def test_int8_payload_keeps_bytes_and_resets(self):
        embed = _engram(4)
        payload = torch.tensor([[-128, 127, 0, 1] * (DIM // 4)] * 4, dtype=torch.int8)
        embed._load_rows(embed.weight, payload)
        self.assertTrue(embed._int8_payload)
        self.assertTrue(torch.equal(embed.weight.data.view(torch.int8), payload))
        embed._load_rows(embed.weight, torch.zeros(4, DIM, dtype=torch.float8_e4m3fn))
        self.assertFalse(embed._int8_payload)


class TestEngramStagedRows(CustomTestCase):
    def test_staged_rows_follow_format_reload(self):
        self._check_staged_rows(reload_formats=True)

    def _check_staged_rows(self, *, reload_formats):
        """Staging preserves scale format and payload signedness across mixed tables;
        padded rows must remain zero after dequantization.
        """
        ids = torch.tensor([3, 1, 3, 0, 2], dtype=torch.int64)
        formats = [(False, False), (False, True), (True, False), (True, True)]
        embeds = [_engram(4) for _ in formats]

        def load_formats(active_formats, revision):
            for embed, (per_row, int8_payload) in zip(embeds, active_formats):
                if per_row:
                    scales = torch.tensor([0.25, 0.5, 1.5, 3.5]) + revision
                else:
                    exponents = (
                        torch.arange(32, dtype=torch.uint8).view(4, 8) % 4
                        + 125
                        + revision
                    )
                    scales = exponents.view(torch.float8_e8m0fnu)
                embed._load_rows(embed.scale, scales)
                payload = (torch.arange(4 * DIM).view(4, DIM) + revision) % 31 - 15
                dtype = torch.int8 if int8_payload else torch.float8_e4m3fn
                embed._load_rows(embed.weight, payload.to(dtype))

        load_formats(formats, 0)

        buffers = {}

        def gather(
            weight_ptr,
            scale_ptr,
            indices,
            out,
            dim,
            block,
            *,
            row_hi,
            per_row_scale=False,
            int8_payload=False,
        ):
            payload = buffers[weight_ptr]
            scales = buffers[scale_ptr]
            values = payload.view(
                torch.int8 if int8_payload else torch.float8_e4m3fn
            ).float()
            if per_row_scale:
                scale_values = scales.view(torch.float32)
            else:
                scale_values = (
                    scales.view(torch.float8_e8m0fnu)
                    .float()
                    .repeat_interleave(block, dim=1)
                )
            valid = indices < row_hi
            out.zero_()
            out[valid] = (values[indices[valid]] * scale_values[indices[valid]]).to(
                out.dtype
            )

        real_empty = torch.empty

        def unpinned_empty(*args, **kwargs):
            kwargs.pop("pin_memory", None)
            return real_empty(*args, **kwargs)

        hasher = SimpleNamespace(
            hash_windows=lambda *_: (
                ids[:, None, None].expand(-1, len(embeds), 1),
                None,
            )
        )
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: True)
        )
        with (
            patch("torch.cuda.Stream"),
            patch("torch.cuda.Event"),
            patch("torch.cuda.stream", return_value=nullcontext()),
            patch("torch.cuda.current_stream", return_value=MagicMock()),
            patch.object(torch.Tensor, "record_stream"),
            patch("torch.empty", side_effect=unpinned_empty),
            patch.object(EngramRowStager, "_windows", return_value=(None, None)),
            patch("sglang.srt.layers.engram.engram_gather", side_effect=gather),
        ):
            stager = EngramRowStager(hasher, embeds, torch.device("cpu"))
            try:
                rounds = [formats]
                if reload_formats:
                    switched = [(not per_row, not int8) for per_row, int8 in formats]
                    rounds += [switched, switched, formats]
                for revision, active_formats in enumerate(rounds):
                    if revision:
                        load_formats(active_formats, revision)
                    staged = stager.stage(batch)
                    for embed, (rows, scales, event), (per_row, int8_payload) in zip(
                        embeds, staged, active_formats
                    ):
                        with self.subTest(
                            revision=revision,
                            per_row=per_row,
                            int8_payload=int8_payload,
                        ):
                            self.assertEqual(
                                scales.shape, (len(ids), 4 if per_row else DIM // BLOCK)
                            )
                            buffers[rows.data_ptr()] = rows
                            buffers[scales.data_ptr()] = scales
                            out = embed.dequant_staged(
                                rows, scales, event, torch.Size([len(ids) + 2])
                            )
                            torch.testing.assert_close(
                                out[: len(ids)], embed._owned_rows(ids), rtol=0, atol=0
                            )
                            self.assertEqual(out[len(ids) :].count_nonzero().item(), 0)
            finally:
                stager.pool.shutdown()

    def test_staged_rows_reject_missing_fp32_scales(self):
        embed = _engram(4)
        embed._load_rows(embed.scale, torch.ones(4))
        hasher = SimpleNamespace(
            hash_windows=lambda *_: (torch.zeros(1, 1, 1, dtype=torch.int64), None)
        )
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: True)
        )
        with patch("torch.cuda.Stream"):
            stager = EngramRowStager(hasher, [embed], torch.device("cpu"))
        try:
            embed._scale_f32 = None
            with (
                patch.object(stager, "_windows", return_value=(None, None)),
                patch.object(
                    stager,
                    "_gather",
                    side_effect=AssertionError(
                        "invalid scales must be rejected before gathering"
                    ),
                ) as gather,
            ):
                with self.assertRaisesRegex(ValueError, "FP32 scale tensor"):
                    stager.stage(batch)
                gather.assert_not_called()
        finally:
            stager.pool.shutdown()


if __name__ == "__main__":
    unittest.main()
