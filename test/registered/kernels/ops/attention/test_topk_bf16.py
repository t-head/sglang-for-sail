"""BF16 top-k regression for the per-device shared-memory migration.

Both BF16 launchers once retained a malformed legacy helper body, so neither
could compile from an empty CUDA Free JIT cache. Exercise the public operators
for prefill TOPK512/2048 and decode512, including their first launches inside
a graph, then replay with changed scores, lengths, and page tables. Selected
values must exactly match a BF16 reference; ties may choose different indices.
"""

import unittest

import torch

from sglang.kernels.ops.attention.dsv4.topk import (
    _jit_topk_bf16_module,
    _jit_topk_prefill_bf16_module,
    top_k_per_row_prefill_bf16,
    topk_transform_512_bf16,
)
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipIf(
    not torch.cuda.is_available() or is_hip(),
    "BF16 top-k requires a CUDA or SAIL device",
)
class TestTopKBF16(CustomTestCase):
    def make_case(
        self, topk, stride, lengths, *, decode=False, starts=None, page_size=64
    ):
        starts = starts if starts is not None else [0] * len(lengths)
        batch = len(lengths)
        num_pages = (stride + page_size - 1) // page_size
        gen = torch.Generator().manual_seed(topk + stride)
        # Unique physical pages let the checker invert mappings when raw output
        # is omitted. Randomize them to catch accidental identity mappings.
        pages = torch.stack(
            [torch.randperm(4 * num_pages, generator=gen)[:num_pages] for _ in lengths]
        ).to(torch.int32)
        case = {
            "topk": topk,
            "stride": stride,
            "decode": decode,
            "page_size": page_size,
            "scores": torch.empty((batch, stride), dtype=torch.bfloat16, device="cuda"),
            "starts": torch.tensor(starts, dtype=torch.int32, device="cuda"),
            "ends": torch.tensor(
                [s + n for s, n in zip(starts, lengths)],
                dtype=torch.int32,
                device="cuda",
            ),
            "pages": pages.cuda(),
            "out": torch.empty((batch, topk), dtype=torch.int32, device="cuda"),
            "raw": torch.empty((batch, topk), dtype=torch.int32, device="cuda"),
        }
        self.fill_case(case, 0)
        return case

    def fill_case(self, case, seed, pattern="random"):
        gen = torch.Generator().manual_seed(seed)
        scores = torch.full(case["scores"].shape, float("inf"), dtype=torch.bfloat16)
        starts, ends = case["starts"].cpu().tolist(), case["ends"].cpu().tolist()
        for row, (start, end) in enumerate(zip(starts, ends)):
            length = end - start
            if pattern == "random":
                scores[row, start:end] = torch.randn(length, generator=gen)
            else:
                scores[row, start:end] = 1.0
                if pattern == "late_better" and length > case["topk"]:
                    # Same coarse radix bin, with better BF16 values arriving
                    # after more candidates than the shared tie buffer holds.
                    scores[row, end - case["topk"] : end] = 1.0078125
        case["reference"] = scores
        case["scores"].copy_(scores)
        case["out"].fill_(-7)
        case["raw"].fill_(-7)

    def invoke(self, case, raw=True):
        args = (
            case["pages"],
            case["out"],
            case["page_size"],
            case["raw"] if raw else None,
        )
        if case["decode"]:
            topk_transform_512_bf16(case["scores"], case["ends"], *args)
        else:
            top_k_per_row_prefill_bf16(
                case["scores"], case["starts"], case["ends"], *args
            )

    def check_case(self, case, raw=True):
        mapped = case["out"].cpu().long()
        indices = case["raw"].cpu().long() if raw else None
        pages = case["pages"].cpu().long()
        starts, ends = case["starts"].cpu().tolist(), case["ends"].cpu().tolist()
        for row, (start, end) in enumerate(zip(starts, ends)):
            n = min(case["topk"], end - start)
            actual = mapped[row, :n]
            if indices is None:
                inverse = {int(page): i for i, page in enumerate(pages[row])}
                selected = torch.tensor(
                    [
                        inverse[int(index) // case["page_size"]] * case["page_size"]
                        + int(index) % case["page_size"]
                        for index in actual
                    ],
                    dtype=torch.int64,
                )
            else:
                selected = indices[row, :n]
                self.assertTrue(bool((indices[row, n:] == -1).all()))
            self.assertTrue(bool((selected >= 0).all()))
            self.assertTrue(bool((selected < end - start).all()))
            self.assertEqual(selected.unique().numel(), n)
            values = case["reference"][row, start:end].float()
            self.assertTrue(
                torch.equal(
                    values[selected].sort().values, values.topk(n).values.sort().values
                ),
                f"wrong BF16 selection: row={row}, length={end - start}, topk={case['topk']}",
            )
            expected_pages = (
                pages[row, selected // case["page_size"]] * case["page_size"]
                + selected % case["page_size"]
            )
            self.assertTrue(torch.equal(actual, expected_pages))
            self.assertTrue(bool((mapped[row, n:] == -1).all()))

    def test_00_first_launch_graph(self):
        # Compile before capture, but do not launch: shared-memory opt-in must
        # also work on the first kernel call, as during model graph capture.
        for topk, decode in ((512, False), (2048, False), (512, True)):
            if decode:
                _jit_topk_bf16_module()
            else:
                _jit_topk_prefill_bf16_module(topk)
            for stride in (40001,) if decode else (40001, 20001):
                with self.subTest(topk=topk, decode=decode, stride=stride):
                    case = self.make_case(
                        topk, stride, [0, topk - 1, topk + 1, stride - 7], decode=decode
                    )
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    # PyTorch's default capture stream is process-global; an
                    # explicit device stream is required when checking several
                    # devices in one process.
                    stream = torch.cuda.Stream(device=torch.cuda.current_device())
                    with torch.cuda.graph(graph, stream=stream):
                        self.invoke(case)
                    for replay in range(3):
                        # The same graph must consume current data at unchanged
                        # addresses, including lengths and nonidentity mappings.
                        case["ends"][1] = topk - 1 - replay
                        case["pages"].copy_(case["pages"].flip(1))
                        self.fill_case(case, replay + 1)
                        graph.replay()
                        torch.cuda.synchronize()
                        self.check_case(case)

    def test_prefill_numerics(self):
        cases = (
            (1001, [0, 1, 999], [0, 1, 1], "random"),
            (10000, [9998, 5000], [1, 2], "random"),
            (20001, [19998, 16500], [1, 3], "random"),
            (32768, [32768, 16384, 16385], [0, 0, 1], "random"),
            (32769, [32769, 32768, 8193], [0, 1, 3], "random"),
            (40000, [40000, 32768, 32769, 8192], [0, 0, 0, 1], "random"),
            (40001, [39992, 39992, 39992], [1, 3, 7], "random"),
            (12000, [12000], [0], "late_better"),
            (40000, [40000], [0], "late_better"),
            (40000, [40000], [0], "ties"),
        )
        for topk in (512, 2048):
            for stride, lengths, starts, pattern in cases:
                with self.subTest(topk=topk, stride=stride, pattern=pattern):
                    case = self.make_case(topk, stride, lengths, starts=starts)
                    self.fill_case(case, 19, pattern)
                    self.invoke(case)
                    self.check_case(case)
            # Short-row boundary/padding and the optional raw output contract.
            for page_size in (1, 64):
                case = self.make_case(
                    topk, topk + 9, [0, topk - 1, topk, topk + 1], page_size=page_size
                )
                self.invoke(case, raw=False)
                self.check_case(case, raw=False)

    def test_decode_numerics(self):
        for stride in (1001, 10000, 20001, 32769, 40001):
            for pattern in ("random", "late_better", "ties"):
                with self.subTest(stride=stride, pattern=pattern):
                    case = self.make_case(
                        512, stride, [0, 511, 512, 513, stride], decode=True
                    )
                    self.fill_case(case, 23, pattern)
                    self.invoke(case)
                    self.check_case(case)
        case = self.make_case(512, 40001, [0, 511, 40000], decode=True, page_size=1)
        self.invoke(case, raw=False)
        self.check_case(case, raw=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
