"""CUDA correctness and capture coverage for Page DCP MLA writers."""

import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.attention.set_mla_kv_concat_q import (
    can_use_set_mla_kv_concat_q_fp8,
    set_mla_kv_concat_q_fp8,
)
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=180, stage="base-b-kernel-unit", runner_config="1-gpu-large")

NOPE_DIM = 512
ROPE_DIM = 64
MLA_DIM = NOPE_DIM + ROPE_DIM
MLA_PAGES = 256


def _make_inputs(batch_size, num_heads, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    kv = torch.randn(
        batch_size, MLA_DIM, generator=generator, device="cuda", dtype=torch.float32
    )
    query = torch.randn(
        batch_size,
        num_heads,
        MLA_DIM,
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )
    kv = kv.mul(0.1).to(torch.bfloat16)
    query = query.mul(0.1).to(torch.bfloat16)
    return (
        kv[:, :NOPE_DIM],
        kv[:, NOPE_DIM:],
        query[..., :NOPE_DIM],
        query[..., NOPE_DIM:],
    )


@unittest.skipUnless(torch.cuda.is_available(), "Page DCP writers require CUDA")
class TestDcpPageMlaWrite(CustomTestCase):
    def test_mla_scatter_concat_fp8_dcp(self):
        """DCP geometry preserves owner-only KV stores and every query."""
        if torch.cuda.get_device_capability()[0] < 9:
            self.skipTest("fused FP8 MLA scatter+concat requires SM90+")
        for world_size, page_size in (
            (2, 0),
            (3, 0),
            (2, 1),
            (2, 64),
            (8, 64),
            (3, 3),
        ):
            self.assertTrue(can_use_set_mla_kv_concat_q_fp8(world_size, page_size))
            stripe = page_size or 1
            locations = sorted(
                {
                    0,
                    1,
                    stripe - 1,
                    stripe,
                    stripe + 1,
                    world_size * stripe - 1,
                    world_size * stripe,
                    world_size * stripe + 1,
                    2 * world_size * stripe + 1,
                }
            )
            k_nope, k_rope, q_nope, q_rope = _make_inputs(
                len(locations), num_heads=3, seed=1
            )
            kv_rows = torch.cat((k_nope, k_rope), dim=-1).to(torch.float8_e4m3fn)
            query_ref = torch.cat((q_nope, q_rope), dim=-1).to(torch.float8_e4m3fn)
            for rank in range(world_size):
                # Independently enumerate this rank's stripes and physical rows.
                owned_slots = [
                    slot
                    for start in range(
                        rank * stripe, max(locations) + 1, world_size * stripe
                    )
                    for slot in range(start, start + stripe)
                ]
                local_rows = {slot: row for row, slot in enumerate(owned_slots)}
                for loc_dtype in (torch.int32, torch.int64):
                    with self.subTest(
                        world_size=world_size,
                        page_size=page_size,
                        rank=rank,
                        loc_dtype=loc_dtype,
                    ):
                        pool = torch.full(
                            (MLA_PAGES, MLA_DIM),
                            0.5,
                            dtype=torch.float8_e4m3fn,
                            device="cuda",
                        )
                        expected = pool.clone()
                        for token, slot in enumerate(locations):
                            if slot in local_rows:
                                expected[local_rows[slot]] = kv_rows[token]
                        loc = torch.tensor(locations, dtype=loc_dtype, device="cuda")
                        query = set_mla_kv_concat_q_fp8(
                            pool,
                            loc,
                            k_nope,
                            k_rope,
                            q_nope,
                            q_rope,
                            dcp_world_size=world_size,
                            dcp_rank=rank,
                            dcp_page_size=page_size,
                        )
                        self.assertTrue(
                            torch.equal(
                                pool.view(torch.uint8), expected.view(torch.uint8)
                            )
                        )
                        self.assertTrue(
                            torch.equal(
                                query.view(torch.uint8), query_ref.view(torch.uint8)
                            )
                        )
                        if (world_size, page_size, loc_dtype) == (8, 64, torch.int64):
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):
                                graph_query = set_mla_kv_concat_q_fp8(
                                    pool,
                                    loc,
                                    k_nope,
                                    k_rope,
                                    q_nope,
                                    q_rope,
                                    dcp_world_size=world_size,
                                    dcp_rank=rank,
                                    dcp_page_size=page_size,
                                )
                            pool.fill_(0.5)
                            graph.replay()
                            self.assertTrue(
                                torch.equal(
                                    pool.view(torch.uint8), expected.view(torch.uint8)
                                )
                            )
                            self.assertTrue(
                                torch.equal(
                                    graph_query.view(torch.uint8),
                                    query_ref.view(torch.uint8),
                                )
                            )

    def test_mla_page_scatter_through_pool(self):
        """The ordinary writer stores only owned rows and preserves padding."""
        for world_size, page_size in ((2, 64), (8, 64), (3, 3)):
            span = world_size * page_size
            locations = torch.cat(
                (
                    torch.zeros(1, dtype=torch.int64, device="cuda"),
                    torch.arange(span, 4 * span, dtype=torch.int64, device="cuda"),
                )
            )
            k_nope, k_rope, _, _ = _make_inputs(locations.numel(), num_heads=1, seed=2)
            with (
                get_context().override_server_args(
                    dcp_size=world_size, dcp_kv_layout="page"
                ),
                get_parallel().override(dcp_enabled=True, attn_dcp_size=world_size),
            ):
                for dtype in (torch.bfloat16, torch.float8_e4m3fn):
                    pool = MLATokenToKVPool(
                        size=3 * page_size,
                        page_size=page_size,
                        dtype=dtype,
                        kv_lora_rank=NOPE_DIM,
                        qk_rope_head_dim=ROPE_DIM,
                        layer_num=1,
                        device="cuda",
                        enable_memory_saver=False,
                    )
                    buffer = pool.get_key_buffer(0)
                    rows = torch.cat((k_nope, k_rope), dim=-1).to(dtype)
                    for rank in range(world_size):
                        for loc_dtype in (torch.int32, torch.int64):
                            with (
                                self.subTest(
                                    world_size=world_size,
                                    page_size=page_size,
                                    rank=rank,
                                    dtype=dtype,
                                    loc_dtype=loc_dtype,
                                ),
                                get_parallel().override(dcp_rank=rank),
                            ):
                                buffer.fill_(0.5)
                                expected = buffer.clone()
                                # Three complete rounds; select this rank's page
                                # in each round, independently of slot arithmetic.
                                expected[page_size:, 0] = (
                                    rows[1:]
                                    .reshape(3, world_size, page_size, MLA_DIM)[:, rank]
                                    .reshape(-1, MLA_DIM)
                                )
                                pool.set_mla_kv_buffer(
                                    SimpleNamespace(layer_id=0),
                                    locations.to(loc_dtype),
                                    k_nope.unsqueeze(1),
                                    k_rope.unsqueeze(1),
                                )
                                self.assertTrue(
                                    torch.equal(
                                        buffer.view(torch.uint8),
                                        expected.view(torch.uint8),
                                    )
                                )


if __name__ == "__main__":
    unittest.main()
