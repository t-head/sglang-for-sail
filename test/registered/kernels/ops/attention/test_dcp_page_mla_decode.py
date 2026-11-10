"""Real Page DCP / token DCP / dense-reference parity on Blackwell."""

import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from sglang.srt.layers.attention.cutedsl_mla_backend import CuteDslMLABackend
from sglang.srt.layers.attention.trtllm_mla_backend import DEFAULT_WORKSPACE_SIZE_MB
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=180, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


@unittest.skipUnless(torch.cuda.is_available(), "DCP layout parity requires CUDA")
class TestCuteDslDcpLayoutParity(CustomTestCase):
    PAGE_SIZE = 64
    DCP_SIZE = 8
    PAGES = [7, 3, 12]
    MAX_LEN = 1100
    NUM_HEADS = 128
    LENGTHS = (1, 50, 64, 65, 512, 513, 1100)
    SCALE = (128 + 64) ** -0.5

    def _make_backend(self, pool, req_to_token, workspace, seq_lens, *, layout):
        backend = object.__new__(CuteDslMLABackend)
        backend.page_size = self.PAGE_SIZE
        backend.dcp_kv_layout = layout
        backend.max_context_len = self.MAX_LEN
        backend.num_draft_tokens = 0
        backend.data_type = backend.q_data_type = torch.bfloat16
        backend.kv_lora_rank = 512
        backend.qk_nope_head_dim = 128
        backend.qk_rope_head_dim = 64
        backend.kv_cache_dim = 576
        backend.token_to_kv_pool = pool
        backend.req_to_token = req_to_token
        backend.workspace_buffer = workspace
        backend.q_indptr_decode = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
        backend.kv_index_translator = SimpleNamespace(
            is_translating=False, full_v2p_table=None
        )
        backend.decode_cuda_graph_metadata = {}
        backend.decode_cuda_graph_kv_indices = torch.full(
            (1, backend._calc_padded_blocks(self.MAX_LEN)),
            -1,
            dtype=torch.int32,
            device="cuda",
        )
        backend._init_cuda_graph_metadata(
            1, 1, ForwardMode.DECODE, seq_lens, torch.device("cuda")
        )
        return backend

    def _reference(self, query, kv):
        # The reference uses CUDA PyTorch math, not an SGLang attention backend.
        keys = kv.float()[None, None].expand(1, self.NUM_HEADS, -1, -1)
        with sdpa_kernel(SDPBackend.MATH):
            output = F.scaled_dot_product_attention(
                query.float().unsqueeze(2),
                keys,
                keys[..., :512],
                scale=self.SCALE,
            )
        lse = torch.logsumexp(query.float() @ kv.float().T * self.SCALE, dim=-1)
        return output.flatten(1), lse

    def _run_layout(self, layout, query, kv, workspace, layer):
        outputs = {length: [] for length in self.LENGTHS}
        lses = {length: [] for length in self.LENGTHS}
        with (
            get_context().override_server_args(
                dcp_size=self.DCP_SIZE, dcp_kv_layout=layout
            ),
            get_parallel().override(dcp_enabled=True, attn_dcp_size=self.DCP_SIZE),
        ):
            pool = MLATokenToKVPool(
                size=max(self.PAGES) * self.PAGE_SIZE,
                page_size=self.PAGE_SIZE,
                dtype=torch.bfloat16,
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                layer_num=1,
                device="cuda",
                enable_memory_saver=False,
            )
            span = self.DCP_SIZE * self.PAGE_SIZE
            allocator = PagedTokenToKVPoolAllocator(
                size=pool.size * self.DCP_SIZE,
                page_size=span,
                dtype=torch.bfloat16,
                device="cuda",
                kvcache=pool,
                need_sort=False,
            )
            allocator.free_pages = torch.tensor(
                self.PAGES
                + [p for p in range(1, max(self.PAGES) + 1) if p not in self.PAGES],
                dtype=torch.int64,
                device="cuda",
            )
            req_to_token = allocator.alloc(len(self.PAGES) * span)[
                : self.MAX_LEN
            ].unsqueeze(0)
            req_indices = torch.zeros(1, dtype=torch.int64, device="cuda")
            stripe = self.PAGE_SIZE if layout == "page" else 1
            for rank in range(self.DCP_SIZE):
                with (
                    self.subTest(layout=layout, rank=rank),
                    get_parallel().override(dcp_rank=rank),
                ):
                    positions = [
                        p
                        for start in range(
                            rank * stripe, self.MAX_LEN, self.DCP_SIZE * stripe
                        )
                        for p in range(start, min(start + stripe, self.MAX_LEN))
                    ]
                    rows = [
                        page * self.PAGE_SIZE + offset
                        for page in self.PAGES
                        for offset in range(self.PAGE_SIZE)
                    ][: len(positions)]
                    row_for_position = dict(zip(positions, rows))
                    buffer = pool.get_key_buffer(0)
                    buffer.fill_(-1)
                    buffer[rows, 0] = kv[positions]
                    seq_lens = torch.ones(1, dtype=torch.int32, device="cuda")
                    loc = torch.empty(1, dtype=torch.int64, device="cuda")
                    k = torch.empty((1, 1, 512), dtype=torch.bfloat16, device="cuda")
                    k_rope = torch.empty(
                        (1, 1, 64), dtype=torch.bfloat16, device="cuda"
                    )
                    batch = SimpleNamespace(
                        forward_mode=ForwardMode.DECODE,
                        batch_size=1,
                        seq_lens=seq_lens,
                        out_cache_loc=loc,
                        out_cache_loc_is_physical=False,
                    )
                    backend = self._make_backend(
                        pool, req_to_token, workspace, seq_lens, layout=layout
                    )

                    def prepare(length):
                        seq_lens.fill_(length)
                        loc.copy_(req_to_token[0, length - 1 : length])
                        k.copy_(kv[length - 1, :512].view(1, 1, 512))
                        k_rope.copy_(kv[length - 1, 512:].view(1, 1, 64))
                        if length - 1 in row_for_position:
                            buffer[row_for_position[length - 1]].zero_()
                        # Non-translating pool: the KV loc plan is not read.
                        backend._apply_cuda_graph_metadata(
                            plan=None,
                            bs=1,
                            req_pool_indices=req_indices,
                            seq_lens=seq_lens,
                            forward_mode=ForwardMode.DECODE,
                        )

                    def forward():
                        return backend.forward_decode(
                            query, k, None, layer, batch, k_rope=k_rope
                        )

                    # Compile both device kernels and the empty-shard fixup before capture.
                    prepare(1)
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            forward()
                    torch.cuda.current_stream().wait_stream(stream)
                    eager = {}
                    for length in self.LENGTHS:
                        prepare(length)
                        output, lse = forward()
                        owned = [p for p in positions if p < length]
                        metadata = backend.forward_decode_metadata
                        self.assertEqual(metadata.seq_lens_k.item(), len(owned))
                        page_count = (len(owned) + self.PAGE_SIZE - 1) // self.PAGE_SIZE
                        self.assertEqual(
                            metadata.block_kv_indices[0, :page_count].tolist(),
                            self.PAGES[:page_count],
                        )
                        if owned:
                            ref_output, ref_lse = self._reference(query, kv[owned])
                            torch.testing.assert_close(
                                output.float(), ref_output, atol=3e-3, rtol=3e-2
                            )
                            torch.testing.assert_close(
                                lse, ref_lse, atol=3e-3, rtol=1e-3
                            )
                        else:
                            self.assertTrue(
                                torch.equal(output, torch.zeros_like(output))
                            )
                            self.assertTrue(torch.isneginf(lse).all())
                        eager[length] = (output.clone(), lse.clone())
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        graph_output, graph_lse = forward()
                    for length in self.LENGTHS:
                        prepare(length)
                        graph.replay()
                        torch.testing.assert_close(graph_output, eager[length][0])
                        torch.testing.assert_close(graph_lse, eager[length][1])
                        if length - 1 in row_for_position:
                            torch.testing.assert_close(
                                buffer[row_for_position[length - 1], 0], kv[length - 1]
                            )
                        outputs[length].append(
                            graph_output.view(1, self.NUM_HEADS, 512).float().clone()
                        )
                        lses[length].append(graph_lse.clone())
            # Rank slices run on one GPU; this checks numerical reassembly, not NCCL.
            results = {}
            for length in self.LENGTHS:
                with self.subTest(layout=layout, length=length, mode="merged"):
                    rank_lses = torch.stack(lses[length])
                    weights = torch.softmax(rank_lses, dim=0)
                    merged = (
                        (torch.stack(outputs[length]) * weights.unsqueeze(-1))
                        .sum(0)
                        .flatten(1)
                    )
                    merged_lse = torch.logsumexp(rank_lses, dim=0)
                    results[length] = (merged, merged_lse)
                    expected, expected_lse = self._reference(query, kv[:length])
                    torch.testing.assert_close(merged, expected, atol=3e-3, rtol=3e-2)
                    torch.testing.assert_close(
                        merged_lse, expected_lse, atol=3e-3, rtol=1e-3
                    )
            return results

    @torch.no_grad()
    def test_page_token_and_dense_attention_agree_in_eager_and_cuda_graph(self):
        """Both layouts must match dense attention, not just each other."""
        if torch.cuda.get_device_capability()[0] != 10:
            self.skipTest("CuteDSL MLA requires a Blackwell SM10.x GPU")
        generator = torch.Generator(device="cuda").manual_seed(42)
        kv = (
            torch.randn(
                (self.MAX_LEN, 576),
                generator=generator,
                device="cuda",
                dtype=torch.bfloat16,
            )
            * 0.1
        )
        query = (
            torch.randn(
                (1, self.NUM_HEADS, 576),
                generator=generator,
                device="cuda",
                dtype=torch.bfloat16,
            )
            * 0.1
        )
        # A value-only position signal makes a wrong causal cutoff conspicuous.
        query[..., 0] = 0
        kv[:, 0] = torch.arange(self.MAX_LEN, device="cuda") * 0.125
        layer = SimpleNamespace(
            layer_id=0,
            tp_q_head_num=self.NUM_HEADS,
            head_dim=576,
            v_head_dim=512,
            scaling=self.SCALE,
            k_scale_float=None,
        )
        workspace = torch.zeros(
            DEFAULT_WORKSPACE_SIZE_MB * 1024 * 1024, dtype=torch.int8, device="cuda"
        )
        page = self._run_layout("page", query, kv, workspace, layer)
        token = self._run_layout("token", query, kv, workspace, layer)
        for length in self.LENGTHS:
            with self.subTest(length=length, mode="page-vs-token"):
                torch.testing.assert_close(
                    page[length][0], token[length][0], atol=3e-3, rtol=3e-2
                )
                torch.testing.assert_close(
                    page[length][1], token[length][1], atol=3e-3, rtol=1e-3
                )


if __name__ == "__main__":
    unittest.main()
