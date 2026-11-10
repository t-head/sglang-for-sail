import unittest

import numpy as np

from sglang.srt.disaggregation.common.utils import build_dcp_page_transfer_plan
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _build_plan(*, src, dst, page_size, dcp_size, dcp_rank, **kwargs):
    return build_dcp_page_transfer_plan(
        np.asarray(src, dtype=np.int32),
        np.asarray(dst, dtype=np.int32),
        physical_page_size=page_size,
        dcp_size=dcp_size,
        dcp_rank=dcp_rank,
        **kwargs,
    )


def _enumerated_rows(
    *, src, dst, page_size, dcp_size, dcp_rank, src_page_offset, decode_prefix_len
):
    """Reference every physical row, including the unused end of the last page."""
    rows = []
    for token_offset in range(len(src) * page_size):
        source_chunk_page, source_page_offset = divmod(token_offset, page_size)
        relative_page = src_page_offset + source_chunk_page
        logical_position = (
            decode_prefix_len + relative_page * page_size + source_page_offset
        )
        if (logical_position // page_size) % dcp_size != dcp_rank:
            continue
        destination_page = dst[relative_page // dcp_size]
        rows.append(
            (
                src[source_chunk_page] * page_size + source_page_offset,
                destination_page * page_size + source_page_offset,
            )
        )

    return rows


class TestDcpPageTransferPlan(CustomTestCase):
    def assertPlanMatchesEnumeration(self, *, src, dst, **kwargs):
        plan = _build_plan(src=src, dst=dst, **kwargs)
        expected = _enumerated_rows(src=src, dst=dst, **kwargs)
        page_size = kwargs["page_size"]
        self.assertEqual(
            [
                (int(src_page) * page_size + offset, int(dst_page) * page_size + offset)
                for src_page, dst_page in zip(
                    plan.src_page_indices, plan.dst_page_indices
                )
                for offset in range(page_size)
            ],
            expected,
        )

    def test_fragmented_pages_prefix_offset_and_tail(self):
        # The five source pages are intentionally fragmented.  The fifth page
        # holds a one-token tail and is still sent whole; the offset shifts ownership.
        src = [9, 3, 8, 1, 7]
        dst = [20, 4]
        for rank in range(3):
            with self.subTest(rank=rank):
                self.assertPlanMatchesEnumeration(
                    src=src,
                    dst=dst,
                    page_size=2,
                    dcp_size=3,
                    dcp_rank=rank,
                    src_page_offset=1,
                    decode_prefix_len=6,
                )

    def test_chunk_offsets_keep_owner_phase_through_the_final_tail(self):
        chunks = [
            ([30, 9], 1),
            ([5, 18], 3),
            ([12], 5),
        ]
        dst = [100, 40]
        for rank in range(3):
            for src, src_page_offset in chunks:
                with self.subTest(
                    rank=rank,
                    src_page_offset=src_page_offset,
                ):
                    self.assertPlanMatchesEnumeration(
                        src=src,
                        dst=dst,
                        page_size=3,
                        dcp_size=3,
                        dcp_rank=rank,
                        src_page_offset=src_page_offset,
                        decode_prefix_len=0,
                    )

    def test_rank_with_no_local_page_needs_no_destination_page(self):
        plan = _build_plan(
            src=[13],
            dst=[],
            page_size=2,
            dcp_size=4,
            dcp_rank=3,
            decode_prefix_len=0,
        )
        self.assertEqual(plan.src_page_indices.size, 0)
        self.assertEqual(plan.dst_page_indices.size, 0)

    def test_rejects_misaligned_prefix(self):
        for prefix in (1, 2):
            with (
                self.subTest(prefix=prefix),
                self.assertRaisesRegex(ValueError, "virtual DCP page size"),
            ):
                _build_plan(
                    src=[0],
                    dst=[0],
                    page_size=2,
                    dcp_size=2,
                    dcp_rank=0,
                    decode_prefix_len=prefix,
                )


if __name__ == "__main__":
    unittest.main()
