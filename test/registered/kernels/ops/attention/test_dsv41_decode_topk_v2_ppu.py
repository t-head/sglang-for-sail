"""PPU Prefill/Decode BF16 v2 selection, phase isolation and graph replay."""

import builtins
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention import deep_select
from sglang.kernels.ops.attention.dsv4.topk import (
    plan_topk_v2,
    topk_transform_paged_v2,
    topk_transform_ragged_bf16_v2,
    topk_transform_sparse_bf16_v2,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4 import indexer
from sglang.srt.layers.attention.dsv4.v41_indexer import (
    full_topk,
    scoring,
    sparse_table,
)
from sglang.srt.utils import is_sail
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=25, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(is_sail(), "PPU-only BF16 TopK v2")
class TestPPUDecodeTopKV2(CustomTestCase):
    def inputs(self, width=8192, k=512):
        rows = 7
        # Padded row strides catch accidental dense-row indexing.
        scores = torch.randn(rows, width + 16, device="cuda").to(torch.bfloat16)
        scores = scores[:, :width]
        lengths = torch.tensor(
            [0, 1, k - 1, k, min(k + 1, width), width - 1, width],
            dtype=torch.int32,
            device="cuda",
        )
        blocks = torch.stack(
            [torch.randperm(width // 8, device="cuda") for _ in range(rows)]
        ).to(torch.int32)
        output = torch.empty(rows, k, dtype=torch.int32, device="cuda")
        return scores, lengths, blocks, output

    def check_selection(self, scores, lengths, blocks, output, page_size=8):
        inverse = blocks.argsort(dim=1)
        raw = inverse.gather(1, output.long().clamp_min(0) // page_size) * page_size
        raw += output.long().clamp_min(0) % page_size
        for row, length in enumerate(lengths.cpu().tolist()):
            valid = output[row] >= 0
            take = min(output.shape[1], length)
            self.assertEqual(int(valid.sum()), take)
            self.assertTrue(bool((output[row, ~valid] == -1).all()))
            selected = raw[row, valid]
            self.assertEqual(selected.unique().numel(), take)
            self.assertTrue(bool(((selected >= 0) & (selected < length)).all()))
            actual = scores[row, selected].sort().values
            expected = scores[row, :length].topk(take).values.sort().values
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def full_paged_inputs(self, page_size):
        scores, lengths, _, output = self.inputs()
        pages = torch.stack(
            [
                torch.randperm(scores.shape[1] // page_size, device="cuda")
                for _ in range(scores.shape[0])
            ]
        ).to(torch.int32)
        return scores, lengths, pages, output

    @torch.inference_mode()
    def test_full_paged_decode_opt_in_and_existing_fallbacks(self):
        backend = full_topk.FullTopKIndexer(
            token_to_kv_pool=None,
            req_to_token=None,
            use_deep_gemm_prefill=False,
            use_deep_gemm_decode=True,
        )
        for page_size, chunk_size in ((128, None), (256, 3)):
            bf16, lengths, pages, output = self.full_paged_inputs(page_size)
            chunks = (
                [
                    slice(start, min(start + chunk_size, bf16.shape[0]))
                    for start in range(0, bf16.shape[0], chunk_size)
                ]
                if chunk_size is not None
                else [slice(None)]
            )
            metadata = SimpleNamespace(
                compressed_seq_lens=lengths[:, None],
                compressed_page_size=page_size,
                max_compressed_seq_len=bf16.shape[1],
                page_table=pages,
                use_topk_v2=False,
                topk_metadata_chunks=None,
                deep_gemm_metadata=[None] * len(chunks) if chunk_size else None,
                request_indices=None,
                row_chunks=lambda: [(rows, None) for rows in chunks],
            )
            for enabled, sail, verify, dtype, provider in (
                (False, True, False, torch.bfloat16, "legacy_bf16"),
                (True, True, False, torch.bfloat16, "native_bf16"),
                (True, True, True, torch.bfloat16, "legacy_bf16"),
                (True, False, False, torch.bfloat16, "legacy_bf16"),
                (True, True, False, torch.float32, "legacy_fp32"),
            ):
                scores = bf16.to(dtype)
                data = SimpleNamespace(
                    k_cache=None,
                    weights=scores,
                    paged_logits_args=lambda _, rows=slice(None): (
                        scores[rows],
                        lengths[rows],
                        pages[rows],
                    ),
                )
                inputs = SimpleNamespace(
                    paged_metadata=metadata, is_verify=verify, out_page_indices=output
                )
                with (
                    self.subTest(
                        page_size=page_size,
                        chunk_size=chunk_size,
                        provider=provider,
                        enabled=enabled,
                        sail=sail,
                        verify=verify,
                    ),
                    envs.SGLANG_OPT_DSV41_TOPK_V2.override(enabled),
                    patch.object(full_topk, "is_sail", return_value=sail),
                    patch.object(
                        full_topk, "get_deep_gemm_decode_data", return_value=data
                    ),
                    patch.object(
                        full_topk,
                        "deep_gemm_fp4_paged_mqa_logits",
                        side_effect=lambda q, *args, **kwargs: q,
                    ),
                    patch.object(indexer, "topk_transform_paged_bf16") as legacy_bf16,
                    patch.object(indexer, "topk_transform_paged") as legacy_fp32,
                    patch.object(
                        indexer,
                        "topk_transform_paged_bf16_v2",
                        wraps=indexer.topk_transform_paged_bf16_v2,
                    ) as native_bf16,
                ):
                    output.fill_(-99)
                    backend.topk_decode(inputs)
                    for name, mock in (
                        ("legacy_bf16", legacy_bf16),
                        ("native_bf16", native_bf16),
                        ("legacy_fp32", legacy_fp32),
                    ):
                        self.assertEqual(
                            mock.call_count, len(chunks) if name == provider else 0
                        )
                    if provider == "native_bf16":
                        self.check_selection(scores, lengths, pages, output, page_size)

    @torch.inference_mode()
    def test_full_paged_bf16_graph_replays_lengths_and_page_table(self):
        page_size = 128
        scores, lengths, pages, output = self.full_paged_inputs(page_size)
        metadata = SimpleNamespace(
            compressed_seq_lens=lengths[:, None],
            page_table=pages,
            compressed_page_size=page_size,
        )

        def run():
            indexer.topk_transform_paged_from_metadata(
                scores, metadata, output, use_bf16_v2=True
            )

        run()
        self.check_selection(scores, lengths, pages, output, page_size)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        scores.normal_()
        lengths.fill_(1543)
        pages.copy_(pages.roll(1, dims=1))
        graph.replay()
        self.check_selection(scores, lengths, pages, output, page_size)
        lengths.zero_()
        graph.replay()
        self.check_selection(scores, lengths, pages, output, page_size)

    @torch.inference_mode()
    def test_selection_and_slot_mapping(self):
        for width in (2048, 8192, 16384, 32768):
            for k in (8, 512, 2048):
                for distribution in ("random", "ties", "negative_inf"):
                    with self.subTest(width=width, k=k, distribution=distribution):
                        args = self.inputs(width, k)
                        if distribution == "ties":
                            args[0].fill_(0.5)
                        elif distribution == "negative_inf":
                            args[0].fill_(-torch.inf)
                            args[0][:, :3] = 1.0
                        before = args[0].clone()
                        topk_transform_sparse_bf16_v2(*args)
                        self.check_selection(*args)
                        torch.testing.assert_close(args[0], before, rtol=0, atol=0)

    @torch.inference_mode()
    def test_cuda_graph_replay_updates_lengths_and_blocks(self):
        args = self.inputs()
        topk_transform_sparse_bf16_v2(*args)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            topk_transform_sparse_bf16_v2(*args)
        graph.replay()
        self.check_selection(*args)
        args[1].fill_(1024)
        args[2].copy_(args[2].roll(1, dims=1))
        args[0].normal_()
        graph.replay()
        self.check_selection(*args)
        args[1].zero_()
        graph.replay()
        self.check_selection(*args)

    @torch.inference_mode()
    def test_decode_opt_in_does_not_change_verify_or_other_platforms(self):
        with envs.SGLANG_OPT_DSV41_TOPK_V2.override(False):
            envs.SGLANG_OPT_DSV41_TOPK_V2.clear()
            self.assertFalse(envs.SGLANG_OPT_DSV41_TOPK_V2.get())
        args = self.inputs()
        scores, lengths, blocks, output = args
        data = SimpleNamespace(q_fp4=None, q_sf=None, k_cache=None, weights=scores)
        published = SimpleNamespace(
            valid_lens=lengths,
            phys_blocks=blocks,
            blocks=blocks,
            schedule=None,
            ready=torch.cuda.Event(),
        )
        published.ready.record()
        legacy = sparse_table.topk_transform_sparse
        for enabled, sail, verify, expected_v2 in (
            (False, True, False, False),
            (True, True, False, True),
            (True, True, True, False),
            (True, False, False, False),
        ):
            with (
                self.subTest(enabled=enabled, sail=sail, verify=verify),
                envs.SGLANG_OPT_DSV41_TOPK_V2.override(enabled),
                patch.object(sparse_table, "is_sail", return_value=sail),
                patch.object(
                    sparse_table, "get_deep_gemm_decode_data", return_value=data
                ),
                patch.object(sparse_table, "sparse_logits", return_value=scores),
                patch.object(
                    sparse_table, "topk_transform_sparse", wraps=legacy
                ) as old,
                patch.object(
                    sparse_table,
                    "topk_transform_sparse_bf16_v2",
                    wraps=topk_transform_sparse_bf16_v2,
                ) as new,
            ):
                inputs = SimpleNamespace(is_verify=verify, out_page_indices=output)
                backend = SimpleNamespace(token_to_kv_pool=None)
                sparse_table.SparseTableBackend.consume_decode(
                    backend, inputs, published
                )
                self.check_selection(*args)
                self.assertEqual(new.call_count, int(expected_v2))
                self.assertEqual(old.call_count, int(not expected_v2))

    @torch.inference_mode()
    def test_sparse_prefill_opt_in_including_captured_outputs(self):
        for enabled in (False, True):
            args = self.inputs()
            scores, lengths, blocks, output = args
            phys_blocks = blocks.roll(1, dims=1)
            table = SimpleNamespace(
                valid_lens=lengths, blocks=blocks, phys_blocks=phys_blocks
            )
            with (
                self.subTest(enabled=enabled),
                envs.SGLANG_OPT_DSV41_TOPK_V2.override(enabled),
                patch.object(
                    sparse_table, "_prefill_table_logits", return_value=scores
                ),
                patch.object(
                    sparse_table,
                    "topk_transform_sparse_bf16_v2",
                    wraps=topk_transform_sparse_bf16_v2,
                ) as new,
            ):
                sparse_table.select_prefill_table(
                    table=table,
                    data=SimpleNamespace(q_fp4=None, q_sf=None, weights=None),
                    k_cache=None,
                    out_positions=output,
                )
                self.check_selection(*args)
                self.assertEqual(new.call_count, int(enabled))
                # A sliced graph output is not contiguous; the temporary must
                # use the selected kernel and copy only the first 512 columns.
                raw = torch.full((7, 520), -99, dtype=torch.int32, device="cuda")
                page = torch.full_like(raw, -99)
                inputs = SimpleNamespace(
                    indexer=SimpleNamespace(index_topk=512),
                    q=scores,
                    weights=None,
                    layer_id=0,
                    index_page_size=8,
                    out_raw_indices=raw,
                    out_page_indices=page,
                )
                backend = SimpleNamespace(token_to_kv_pool=None)

                def run():
                    sparse_table.SparseTableBackend.consume_prefill_captured(
                        backend, inputs, table
                    )

                def check():
                    for result, mapping in ((raw, blocks), (page, phys_blocks)):
                        self.check_selection(scores, lengths, mapping, result[:, :512])
                        self.assertTrue(bool((result[:, 512:] == -99).all()))

                with (
                    patch.object(
                        sparse_table, "quantize_index_q", return_value=(scores, None)
                    ),
                    patch.object(sparse_table, "get_index_k_cache", return_value=None),
                ):
                    run()
                    check()
                    self.assertEqual(new.call_count, 3 * int(enabled))
                    if enabled:
                        torch.cuda.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            run()
                        lengths.fill_(1024)
                        phys_blocks.copy_(phys_blocks.roll(1, dims=1))
                        graph.replay()
                        check()

    @torch.inference_mode()
    def test_ragged_bf16_windows_offsets_and_graph(self):
        for width in (2048, 8192, 16384, 32768):
            scores, lengths, _, output = self.inputs(width)
            # Exercise all four vector-window alignments and nonzero offsets.
            starts = torch.tensor(
                [0, 1, 2, 3, 1, 2, 3], device="cuda", dtype=torch.int32
            )
            lengths = torch.minimum(lengths, width - starts)
            offsets = torch.arange(7, device="cuda", dtype=torch.int32) * 40000

            def run():
                topk_transform_ragged_bf16_v2(
                    scores,
                    lengths,
                    row_starts=starts,
                    out_offsets=offsets,
                    out_indices=output,
                )

            def check():
                for row in range(7):
                    start, length, offset = (
                        int(starts[row]),
                        int(lengths[row]),
                        int(offsets[row]),
                    )
                    valid = output[row] >= 0
                    selected = output[row, valid].long() - offset
                    self.assertEqual(selected.numel(), min(512, length))
                    self.assertEqual(selected.unique().numel(), selected.numel())
                    self.assertTrue(bool(((selected >= 0) & (selected < length)).all()))
                    self.assertTrue(bool((output[row, ~valid] == -1).all()))
                    torch.testing.assert_close(
                        scores[row, start + selected].sort().values,
                        scores[row, start : start + length]
                        .topk(min(512, length))
                        .values.sort()
                        .values,
                        rtol=0,
                        atol=0,
                    )

            run()
            check()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            scores.normal_()
            lengths.fill_(1024)
            offsets.add_(17)
            graph.replay()
            check()
            lengths.zero_()
            graph.replay()
            check()

    @torch.inference_mode()
    def test_ragged_prefill_dispatch_without_wheel(self):
        scores, lengths, _, output = self.inputs()
        offsets = torch.zeros_like(lengths)
        for enabled, sail in ((False, True), (True, True), (True, False)):
            with (
                self.subTest(enabled=enabled, sail=sail),
                envs.SGLANG_OPT_DSV41_TOPK_V2.override(enabled),
                patch.object(scoring, "is_sail", return_value=sail),
                patch.object(deep_select, "_wheel_available", return_value=False),
                patch.object(
                    scoring,
                    "topk_transform_ragged_bf16_v2",
                    wraps=topk_transform_ragged_bf16_v2,
                ) as new,
            ):
                scoring.topk_ragged(scores, lengths, out=output, offsets=offsets)
                self.assertEqual(new.call_count, int(enabled and sail))
                identity = (
                    torch.arange(8192 // 8, device="cuda", dtype=torch.int32)
                    .expand(7, -1)
                    .contiguous()
                )
                self.check_selection(scores, lengths, identity, output)

    @torch.inference_mode()
    def test_ragged_prefill_keeps_deep_select_wheel(self):
        scores, lengths, _, output = self.inputs()
        offsets = torch.arange(7, device="cuda", dtype=torch.int32) * 40000
        for enabled in (False, True):
            with (
                self.subTest(enabled=enabled),
                envs.SGLANG_OPT_DSV41_TOPK_V2.override(enabled),
                patch.object(deep_select, "_wheel_available", return_value=True),
                patch.object(deep_select, "_wheel_topk") as wheel,
                patch.object(scoring, "topk_transform_ragged_bf16_v2") as native,
            ):
                scoring.topk_ragged(scores, lengths, out=output, offsets=offsets)
                native.assert_not_called()
                wheel.assert_called_once()
                self.assertIs(wheel.call_args.args[0], scores)
                self.assertEqual(wheel.call_args.args[1], output.shape[1])
                self.assertIs(wheel.call_args.kwargs["end"], lengths)
                self.assertIs(wheel.call_args.kwargs["output_idx"], output)
                self.assertIs(wheel.call_args.kwargs["output_idx_offset"], offsets)
                self.assertEqual(wheel.call_args.kwargs["idx_oob_fill_value"], -1)

    @torch.inference_mode()
    def test_fp32_and_bf16_modules_coexist(self):
        args = self.inputs()
        fp32_scores = args[0].float()
        plan = plan_topk_v2(args[1])
        for bf16 in (False, True, False):
            if bf16:
                topk_transform_sparse_bf16_v2(*args)
                self.check_selection(*args)
            else:
                topk_transform_paged_v2(fp32_scores, args[1], args[2], args[3], 8, plan)
                self.check_selection(fp32_scores, *args[1:])

    @torch.inference_mode()
    def test_no_deep_select_import_and_empty_batch(self):
        args = self.inputs()
        import_original = builtins.__import__

        def reject_wheel(name, *positional, **kwargs):
            if name == "deep_select" or name.startswith("deep_select."):
                raise AssertionError(
                    "The native BF16 v2 path must not import the wheel"
                )
            return import_original(name, *positional, **kwargs)

        with (
            patch.object(deep_select, "_wheel_available", return_value=False),
            patch("builtins.__import__", side_effect=reject_wheel),
        ):
            topk_transform_sparse_bf16_v2(*args)
            with envs.SGLANG_OPT_DSV41_TOPK_V2.override(True):
                scoring.topk_ragged(
                    args[0], args[1], out=args[3], offsets=torch.zeros_like(args[1])
                )
        topk_transform_sparse_bf16_v2(*args)
        self.check_selection(*args)
        topk_transform_sparse_bf16_v2(*(arg[:0] for arg in args))


if __name__ == "__main__":
    unittest.main()
