"""PPU Indexer API regressions; enable device tests with SGLANG_TEST_INDEXER_GPU=1."""

import ast
import os
import sys
import unittest
from pathlib import Path
from types import MethodType, ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[3]
ATTENTION = ROOT / "python/sglang/srt/layers/attention"


def load_symbol(path, name, namespace, *, stop_after=None):
    tree = ast.parse(path.read_text())
    node = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name
    )
    if isinstance(node, ast.FunctionDef):
        node.decorator_list = []
        if stop_after is not None:
            # Isolate metadata construction without unrelated compressor initialization.
            stop = next(
                i
                for i, stmt in enumerate(node.body)
                if isinstance(stmt, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == stop_after
                    for target in stmt.targets
                )
            )
            node.body = node.body[: stop + 1] + [
                ast.Return(value=ast.Name(id=stop_after, ctx=ast.Load()))
            ]
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


def api_functions():
    namespace = {"torch": torch}
    path = ATTENTION / "mqa_logits_utils.py"
    for name in (
        "SailPagedMQAMetadata",
        "sail_mqa_logits",
        "sail_paged_mqa_metadata",
        "sail_paged_mqa_logits",
    ):
        load_symbol(path, name, namespace)
    return SimpleNamespace(**namespace)


class TestIndexerAPI(unittest.TestCase):
    def setUp(self):
        self.api = api_functions()
        self.dg = ModuleType("deep_gemm")
        for name in (
            "fp8_fp4_mqa_logits",
            "fp8_mqa_logits",
            "int8_mqa_logits",
            "get_paged_mqa_logits_metadata",
            "fp8_fp4_paged_mqa_logits",
            "fp8_paged_mqa_logits",
            "int8_paged_mqa_logits",
        ):
            setattr(self.dg, name, Mock())
        self.dg.get_num_sms = Mock(return_value=24)
        self.modules = patch.dict(sys.modules, {"deep_gemm": self.dg})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def packed(self, rows=4, paged=False):
        shape = (rows, 1, 32, 64) if paged else (rows, 32, 64)
        return (
            torch.full(shape, 0x22, dtype=torch.uint8),
            torch.full(shape[:-1], 0x7F7F7F7F, dtype=torch.int32),
        )

    def test_mxfp4_nonpaged_new_signature_and_bf16(self):
        q = self.packed()
        kv = (torch.zeros(96, 64, dtype=torch.int8), torch.zeros(96, dtype=torch.int32))
        starts = torch.tensor([0, 20, 40, 70], dtype=torch.int32)
        ends = torch.tensor([9, 37, 66, 95], dtype=torch.int32)
        self.api.sail_mqa_logits(
            q, kv, torch.ones(4, 32), starts, ends, max_seqlen_k=32
        )
        args, kwargs = self.dg.fp8_fp4_mqa_logits.call_args
        self.assertEqual(kwargs, {})
        self.assertEqual(len(args), 6)
        self.assertEqual(args[-1], 32)
        self.assertEqual(args[0][0].dtype, torch.int8)
        self.assertEqual(args[2].dtype, torch.bfloat16)
        self.assertIs(args[3], starts)
        with self.assertRaisesRegex(ValueError, "positive"):
            self.api.sail_mqa_logits(q, kv, torch.ones(4, 32), starts, ends)

    def test_int8_output_selected_by_weights_not_keyword(self):
        for dtype in (torch.float32, torch.bfloat16):
            q = torch.ones(4, 32, 128, dtype=torch.int8)
            kv = (torch.ones(64, 128, dtype=torch.int8), torch.ones(64))
            self.api.sail_mqa_logits(
                q,
                kv,
                torch.ones(4, 32, dtype=dtype),
                torch.zeros(4, dtype=torch.int32),
                torch.ones(4, dtype=torch.int32),
            )
            args, kwargs = self.dg.int8_mqa_logits.call_args
            self.assertEqual(args[2].dtype, dtype)
            self.assertEqual(kwargs, {"clean_logits": False, "max_seqlen_k": 0})

    def test_int8_paged_mtp_keeps_query_rows_and_allows_split_chunks(self):
        namespace = {
            "torch": torch,
            "sail_paged_mqa_logits": self.api.sail_paged_mqa_logits,
            "quantize_query": lambda q, w: (q.to(torch.int8), w.float() * 2),
        }
        paged = load_symbol(
            ROOT / "python/sglang/kernels/ops/attention/dsv4/int8_indexer.py",
            "int8_paged_index_logits",
            namespace,
        )
        q = torch.ones(12, 32, 128)
        weights = torch.ones(12, 32)
        lens = torch.arange(1, 13, dtype=torch.int64)
        table = torch.arange(24, dtype=torch.int32).reshape(12, 2)
        cache = torch.zeros(24, 64 * 132, dtype=torch.uint8)
        self.dg.int8_paged_mqa_logits.side_effect = lambda q, *args, **kwargs: (
            torch.zeros(q.shape[0], 128)
        )
        for rows in (slice(0, 12), slice(0, 4), slice(4, 8), slice(8, 12)):
            output = paged(
                q[rows],
                weights[rows],
                cache,
                lens[rows],
                table[rows],
                object(),
                128,
                64,
                rows_per_request=6,
            )
            args, kwargs = self.dg.int8_paged_mqa_logits.call_args
            meta_args, meta_kwargs = self.dg.get_paged_mqa_logits_metadata.call_args
            count = rows.stop - rows.start
            self.assertEqual(output.shape, (count, 128))
            self.assertEqual(args[0].shape, (count, 1, 32, 128))
            self.assertEqual(args[0].dtype, torch.int8)
            self.assertTrue(torch.equal(args[2], weights[rows] * 2))
            self.assertTrue(torch.equal(args[3], lens[rows].to(torch.int32)[:, None]))
            self.assertTrue(torch.equal(args[4], table[rows]))
            self.assertEqual(meta_args[0].shape, (count, 1))
            self.assertEqual(meta_kwargs["metadata_extra"], (1, 32, 128, torch.int8))
            self.assertEqual(kwargs, {"indices": None})
            self.assertIs(args[5], self.dg.get_paged_mqa_logits_metadata.return_value)

    def test_paged_mtp_owners_metadata_and_padding(self):
        for owners in ([7, 7, 12, 12], [7, 12, 12, 12]):
            q = self.packed(paged=True)
            lens = torch.tensor([3, 5, 7, 0], dtype=torch.int32)
            ids = torch.tensor(owners, dtype=torch.int64)
            self.api.sail_paged_mqa_logits(
                q,
                torch.zeros(2, 64, 1, 68, dtype=torch.uint8),
                torch.ones(4, 32),
                lens,
                torch.zeros(4, 2, dtype=torch.int32),
                object(),
                128,
                indices=ids,
            )
            meta_args, meta_kwargs = self.dg.get_paged_mqa_logits_metadata.call_args
            args, kwargs = self.dg.fp8_fp4_paged_mqa_logits.call_args
            self.assertEqual(meta_kwargs["metadata_extra"], (1, 32, 128, torch.int8))
            self.assertEqual(meta_args[0].shape, (4, 1))
            self.assertEqual(meta_args[2], 24)
            self.assertEqual(args[0][0].shape, (4, 1, 32, 64))
            self.assertEqual(args[2].dtype, torch.bfloat16)
            self.assertEqual(set(kwargs), {"indices"})
            self.assertIs(meta_kwargs["indices"], kwargs["indices"])
            self.assertEqual(kwargs["indices"].tolist(), owners[:3] + [-4])
            self.assertEqual(kwargs["indices"].dtype, torch.int32)
        self.dg.get_num_sms.return_value = 12
        self.api.sail_paged_mqa_logits(
            q,
            torch.zeros(2, 64, 1, 68, dtype=torch.uint8),
            torch.ones(4, 32),
            lens,
            torch.zeros(4, 2, dtype=torch.int32),
            None,
            128,
            indices=ids,
        )
        self.assertEqual(self.dg.get_paged_mqa_logits_metadata.call_args.args[2], 12)

    def test_forward_schedule_reused_without_owner_or_weight_kernels(self):
        q = self.packed(paged=True)
        lens = torch.tensor([3, 5, 7, 0], dtype=torch.int32)
        owners = torch.tensor([7, 7, 12, 12], dtype=torch.int32)
        weights = torch.ones(4, 32, dtype=torch.bfloat16)
        cache = torch.zeros(2, 64, 1, 68, dtype=torch.uint8)
        table = torch.zeros(4, 2, dtype=torch.int32)
        plan = self.api.sail_paged_mqa_metadata(lens, 64, 32, 128, torch.int8, owners)
        self.assertEqual(plan.indices.tolist(), [7, 7, 12, -4])
        self.assertEqual(owners.tolist(), [7, 7, 12, 12])
        self.assertIs(
            self.dg.get_paged_mqa_logits_metadata.call_args.kwargs["indices"],
            plan.indices,
        )
        with (
            patch.object(torch, "arange") as arange,
            patch.object(torch, "where") as where,
        ):
            for _ in range(3):
                self.api.sail_paged_mqa_logits(
                    q, cache, weights, lens, table, plan, 128, indices=owners
                )
                args, kwargs = self.dg.fp8_fp4_paged_mqa_logits.call_args
                self.assertIs(args[2], weights)
                self.assertIs(args[5], plan.tensor)
                self.assertIs(kwargs["indices"], plan.indices)
        self.assertEqual(arange.call_count, 0)
        self.assertEqual(where.call_count, 0)
        self.assertEqual(self.dg.get_paged_mqa_logits_metadata.call_count, 1)
        self.assertEqual(self.dg.fp8_fp4_paged_mqa_logits.call_count, 3)

    def test_grouped_padding_does_not_skip_preceding_valid_rows(self):
        def grouped_schedule(lens, *args, indices, **kwargs):
            bounds = torch.zeros_like(lens)
            start = 0
            for end in range(1, len(indices) + 1):
                if end == len(indices) or indices[end] != indices[start]:
                    bounds[start:end] = lens[end - 1]
                    start = end
            return bounds

        def grouped_logits(q, cache, weights, lens, table, schedule, width, **kwargs):
            out = torch.full((len(lens), width), float("-inf"))
            for row in range(len(lens)):
                count = min(lens[row, 0].item(), schedule[row, 0].item())
                out[row, :count] = row + 1
            return out

        self.dg.get_paged_mqa_logits_metadata.side_effect = grouped_schedule
        self.dg.fp8_fp4_paged_mqa_logits.side_effect = grouped_logits
        owners = torch.tensor([5, 5, 9, 9], dtype=torch.int32)
        for lengths in ([33, 64, 95, 0], [0, 64, 0, 95], [0, 0, 0, 0]):
            with self.subTest(lengths=lengths):
                lens = torch.tensor(lengths, dtype=torch.int32)
                plan = self.api.sail_paged_mqa_metadata(
                    lens, 64, 32, 128, torch.int8, owners
                )
                padding = plan.indices[lens == 0]
                self.assertTrue(bool((padding < 0).all()))
                self.assertEqual(padding.unique().numel(), padding.numel())
                self.assertTrue(torch.equal(plan.indices[lens > 0], owners[lens > 0]))
                self.assertEqual(owners.tolist(), [5, 5, 9, 9])
                out = self.api.sail_paged_mqa_logits(
                    self.packed(paged=True),
                    torch.zeros(2, 64, 1, 68, dtype=torch.uint8),
                    torch.ones(4, 32, dtype=torch.bfloat16),
                    lens,
                    torch.zeros(4, 2, dtype=torch.int32),
                    plan,
                    128,
                    indices=owners,
                )
                self.assertIs(
                    self.dg.fp8_fp4_paged_mqa_logits.call_args.kwargs["indices"],
                    plan.indices,
                )
                for row, count in enumerate(lengths):
                    self.assertTrue(bool((out[row, :count] == row + 1).all()))
                    self.assertTrue(bool(out[row, count:].isneginf().all()))

    def test_schedule_rebuilt_for_changed_compute_configuration(self):
        q = self.packed(paged=True)
        lens = torch.ones(4, dtype=torch.int32)
        owners = torch.arange(4, dtype=torch.int32)
        cache = torch.zeros(2, 64, 1, 68, dtype=torch.uint8)
        table = torch.zeros(4, 2, dtype=torch.int32)
        plan = self.api.sail_paged_mqa_metadata(lens, 64, 32, 128, torch.int8, owners)
        # Changing SM allocation must not reuse an incompatible opaque schedule.
        self.dg.get_num_sms.return_value = 12
        self.api.sail_paged_mqa_logits(
            q, cache, torch.ones(4, 32), lens, table, plan, 128, indices=owners
        )
        self.assertEqual(self.dg.get_paged_mqa_logits_metadata.call_count, 2)
        self.assertEqual(self.dg.get_paged_mqa_logits_metadata.call_args.args[2], 12)
        # Clipped/padded queries and differently tiled heads/pages also rebuild.
        for rows, page_size, heads, dim, dtype in (
            (2, 64, 32, 128, torch.int8),
            (4, 128, 32, 128, torch.int8),
            (4, 64, 64, 128, torch.int8),
            (4, 64, 32, 64, torch.int8),
            (4, 64, 32, 128, torch.float8_e4m3fn),
        ):
            self.dg.get_num_sms.return_value = 24
            self.api.sail_paged_mqa_logits(
                torch.zeros(rows, 1, heads, dim, dtype=dtype),
                torch.zeros(2, page_size, 1, dim + 4, dtype=torch.uint8),
                torch.ones(rows, heads),
                lens[:rows],
                table[:rows],
                plan,
                128,
            )
            self.assertEqual(
                self.dg.get_paged_mqa_logits_metadata.call_args.kwargs[
                    "metadata_extra"
                ],
                (1, heads, dim, dtype),
            )
        self.assertEqual(self.dg.get_paged_mqa_logits_metadata.call_count, 7)

    def test_packed_schedule_without_owners_is_rebuilt(self):
        lens = torch.tensor([33, 64, 95, 0], dtype=torch.int32)
        owners = torch.tensor([5, 5, 9, 9], dtype=torch.int32)
        plan = self.api.sail_paged_mqa_metadata(lens, 64, 32, 128, torch.int8)
        self.api.sail_paged_mqa_logits(
            self.packed(paged=True),
            torch.zeros(2, 64, 1, 68, dtype=torch.uint8),
            torch.ones(4, 32, dtype=torch.bfloat16),
            lens,
            torch.zeros(4, 2, dtype=torch.int32),
            plan,
            128,
            indices=owners,
        )
        self.assertEqual(self.dg.get_paged_mqa_logits_metadata.call_count, 2)
        normalized = self.dg.get_paged_mqa_logits_metadata.call_args.kwargs["indices"]
        self.assertEqual(normalized.tolist(), [5, 5, 9, -4])
        self.assertIs(
            self.dg.fp8_fp4_paged_mqa_logits.call_args.kwargs["indices"], normalized
        )

    def test_legacy_paged_metadata_uses_real_dtype(self):
        for dtype, name in (
            (torch.int8, "int8_paged_mqa_logits"),
            (torch.float8_e4m3fn, "fp8_paged_mqa_logits"),
        ):
            self.api.sail_paged_mqa_logits(
                torch.zeros(4, 1, 32, 128, dtype=dtype),
                torch.zeros(2, 64, 1, 132, dtype=torch.uint8),
                torch.ones(4, 32),
                torch.ones(4, dtype=torch.int32),
                torch.zeros(4, 2, dtype=torch.int32),
                None,
                128,
            )
            self.assertEqual(
                self.dg.get_paged_mqa_logits_metadata.call_args.kwargs[
                    "metadata_extra"
                ],
                (1, 32, 128, dtype),
            )
            self.assertEqual(getattr(self.dg, name).call_args.kwargs, {"indices": None})

    def test_invalid_paged_shape_and_missing_owners_fail_early(self):
        cache, weights = torch.zeros(2, 64, 1, 68, dtype=torch.uint8), torch.ones(4, 32)
        lens, table = (
            torch.ones(4, dtype=torch.int32),
            torch.zeros(4, 2, dtype=torch.int32),
        )
        with self.assertRaisesRegex(ValueError, "real request"):
            self.api.sail_paged_mqa_logits(
                self.packed(paged=True), cache, weights, lens, table, None, 128
            )
        q = (
            torch.zeros(2, 2, 32, 64, dtype=torch.int8),
            torch.zeros(2, 2, 32, dtype=torch.int32),
        )
        with self.assertRaisesRegex(ValueError, "\\[Q, 1"):
            self.api.sail_paged_mqa_logits(
                q, cache, weights, lens, table, None, 128, indices=torch.arange(4)
            )

    def test_dsa_dispatch_uses_capacity_and_no_old_keywords(self):
        dispatch = load_symbol(
            ATTENTION / "dsa/dsa_indexer.py",
            "_get_mqa_logits_dispatch",
            {
                "torch": torch,
                "is_sail": lambda: True,
                "sail_mqa_logits": self.api.sail_mqa_logits,
            },
        )
        model = SimpleNamespace(
            use_fp4=True, use_fp8=False, use_int8=False, bf16_indexer=False
        )
        dispatch(
            model,
            self.packed(),
            (torch.zeros(96, 64, dtype=torch.int8), torch.zeros(96, dtype=torch.int32)),
            torch.ones(4, 32),
            torch.zeros(4, dtype=torch.int32),
            torch.ones(4, dtype=torch.int32),
            clean_logits=False,
            max_seqlen_k=32,
        )
        self.assertEqual(self.dg.fp8_fp4_mqa_logits.call_args.args[-1], 32)
        self.assertNotIn("logits_dtype", self.dg.fp8_fp4_mqa_logits.call_args.kwargs)

    def test_dsa_paged_fixed_and_ragged_mtp_use_expanded_rows(self):
        namespace = {
            "torch": torch,
            "TYPE_CHECKING": False,
            "_is_hip": False,
            "_is_xpu": False,
            "_is_cuda": True,
            "is_sail": lambda: True,
            "_index_k_dcp": lambda: (1, 0),
        }
        path = ATTENTION / "dsa/dsa_indexer.py"
        paged = load_symbol(path, "_get_topk_paged", namespace)
        pad_heads = load_symbol(path, "_pad_heads_for_deep_gemm", namespace)
        cache = torch.zeros(2, 64 * 68, dtype=torch.uint8)
        namespace["get_token_to_kv_pool"] = lambda: SimpleNamespace(index_page_size=64)
        model = SimpleNamespace(
            use_fp4=True,
            bf16_indexer=False,
            head_dim=128,
            sm_count=39,
            index_topk=2,
            _get_index_k_read_buffer=Mock(return_value=cache),
            _mask_init_and_local_tokens=Mock(),
            paged_mqa_logits_backend=SimpleNamespace(
                is_cutedsl=lambda: False, is_aiter=lambda: False
            ),
            _get_paged_mqa_logits_dispatch=Mock(
                return_value=torch.zeros(4, 128, dtype=torch.bfloat16)
            ),
        )
        model._pad_heads_for_deep_gemm = pad_heads
        q = (
            torch.zeros(4, 16, 64, dtype=torch.int8),
            torch.zeros(4, 16, dtype=torch.int32),
        )
        mode = SimpleNamespace(
            is_target_verify=lambda: True, is_draft_extend_v2=lambda: False
        )
        metadata = SimpleNamespace(
            get_page_table_64=lambda: torch.zeros(4, 2, dtype=torch.int32),
            get_seqlens_expanded=lambda: torch.tensor([3, 4, 5, 6], dtype=torch.int32),
            get_seqlens_int32=lambda: torch.tensor([4, 6], dtype=torch.int32),
            get_dsa_extend_len_cpu=lambda: [1] * 4,
            paged_mqa_ctx_lens_2d=torch.tensor([[3, 4], [5, 6]], dtype=torch.int32),
            topk_transform=Mock(return_value=torch.zeros(4, 2, dtype=torch.int32)),
        )
        for row_owners, expected in (
            (None, [0, 0, 1, 1]),
            (torch.tensor([0, 1, 1, 1]), [0, 1, 1, 1]),
        ):
            metadata.get_token_to_batch_idx = lambda: row_owners
            paged(
                model,
                SimpleNamespace(forward_mode=mode),
                0,
                q,
                torch.ones(4, 16, 1),
                metadata,
            )
            args, kwargs = model._get_paged_mqa_logits_dispatch.call_args
            self.assertEqual(args[0][0].shape, (4, 1, 32, 64))
            self.assertEqual(args[0][1].shape, (4, 1, 32))
            self.assertEqual(args[2].shape, (4, 32))
            self.assertEqual(args[3].shape, (4, 1))
            self.assertEqual(args[4].shape, (4, 2))
            self.assertEqual(kwargs["indices"].tolist(), expected)

    def test_paged_metadata_copy_keeps_request_buffer_address(self):
        from dataclasses import dataclass, field, fields

        namespace = {
            "torch": torch,
            "dataclass": dataclass,
            "field": field,
            "fields": fields,
            "is_sail": lambda: True,
            "is_hip": lambda: False,
            "plan_indexer_row_chunks": lambda **kwargs: 2,
            "sail_paged_mqa_metadata": self.api.sail_paged_mqa_metadata,
        }
        path = ATTENTION / "dsv4/metadata.py"
        load_symbol(path, "iter_row_chunks", namespace)
        load_symbol(path, "copy_metadata", namespace)
        metadata_class = load_symbol(path, "PagedIndexerMetadata", namespace)

        def make(owners):
            return metadata_class(
                page_size=256,
                compressed_page_size=64,
                page_table=torch.zeros(5, 2, dtype=torch.int32),
                compressed_seq_lens=torch.tensor([3, 4, 5, 6, 0], dtype=torch.int32),
                use_topk_v2=False,
                rows_per_request=3,
                q_fp8_shape=torch.Size((5, 1, 32, 64)),
                qk_dtype=torch.int8,
                request_indices=torch.tensor(owners, dtype=torch.int32),
            )

        current = make([7, 7, 7, 12, 12])
        newer = make([7, 12, 12, 12, 12])
        pointer = current.request_indices.data_ptr()
        current.copy_(newer)
        self.assertEqual(current.request_indices.data_ptr(), pointer)
        self.assertEqual(current.request_indices.tolist(), [7, 12, 12, 12, 12])
        self.assertEqual(len(current.row_chunks()), 3)
        self.assertEqual(self.dg.get_paged_mqa_logits_metadata.call_count, 6)
        with (
            patch.object(torch, "arange") as arange,
            patch.object(torch, "where") as where,
        ):
            for _ in range(3):
                for rows, plan in current.row_chunks():
                    count = rows.stop - rows.start
                    self.assertEqual(plan.config[:5], (count, 64, 32, 128, torch.int8))
                    expected = current.request_indices[rows].clone()
                    expected[current.compressed_seq_lens[rows] == 0] = -1
                    self.assertTrue(torch.equal(plan.indices, expected))
                    self.api.sail_paged_mqa_logits(
                        self.packed(rows=count, paged=True),
                        torch.zeros(2, 64, 1, 68, dtype=torch.uint8),
                        torch.ones(count, 32, dtype=torch.bfloat16),
                        current.compressed_seq_lens[rows],
                        current.page_table[rows],
                        plan,
                        128,
                        indices=current.request_indices[rows],
                    )
                    self.assertIs(
                        self.dg.fp8_fp4_paged_mqa_logits.call_args.kwargs["indices"],
                        plan.indices,
                    )
        arange.assert_not_called()
        where.assert_not_called()
        self.assertEqual(self.dg.get_paged_mqa_logits_metadata.call_count, 6)
        self.assertEqual(self.dg.fp8_fp4_paged_mqa_logits.call_count, 9)

    def test_graph_schedule_copy_updates_in_place(self):
        metadata = self.api.SailPagedMQAMetadata
        config = (4, 64, 32, 128, torch.int8, 24)
        captured = metadata(
            torch.zeros(25, 2, dtype=torch.int32),
            config,
            torch.tensor([5, 5, 9, -4], dtype=torch.int32),
        )
        replay = metadata(
            torch.ones(25, 2, dtype=torch.int32),
            config,
            torch.tensor([5, -2, 9, -4], dtype=torch.int32),
        )
        pointer = captured.tensor.data_ptr()
        owners_pointer = captured.indices.data_ptr()
        captured.copy_(replay)
        self.assertEqual(captured.tensor.data_ptr(), pointer)
        self.assertEqual(captured.indices.data_ptr(), owners_pointer)
        self.assertTrue(torch.equal(captured.tensor, replay.tensor))
        self.assertTrue(torch.equal(captured.indices, replay.indices))
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            captured.copy_(metadata(replay.tensor, config))

    def test_paged_graph_metadata_refreshes_schedule_and_lengths(self):
        from dataclasses import dataclass, field, fields

        namespace = {
            "torch": torch,
            "dataclass": dataclass,
            "field": field,
            "fields": fields,
            "is_sail": lambda: True,
            "is_hip": lambda: False,
            "plan_indexer_row_chunks": lambda **kwargs: None,
            "sail_paged_mqa_metadata": self.api.sail_paged_mqa_metadata,
        }
        path = ATTENTION / "dsv4/metadata.py"
        load_symbol(path, "iter_row_chunks", namespace)
        load_symbol(path, "copy_metadata", namespace)
        metadata_class = load_symbol(path, "PagedIndexerMetadata", namespace)
        self.dg.get_paged_mqa_logits_metadata.side_effect = (
            lambda lens, *args, **kwargs: lens.clone()
        )
        namespace["iter_row_chunks"] = Mock(side_effect=AssertionError("not chunked"))

        def make(lens, owners, build=True):
            return metadata_class(
                page_size=256,
                compressed_page_size=64,
                page_table=torch.zeros(len(lens), 2, dtype=torch.int32),
                compressed_seq_lens=torch.tensor(lens, dtype=torch.int32),
                request_indices=torch.tensor(owners, dtype=torch.int32),
                q_fp8_shape=torch.Size((len(lens), 1, 32, 64)),
                qk_dtype=torch.int8,
                use_topk_v2=False,
                use_prefill_cuda_graph=True,
                build_paged_mqa_logits_metadata=build,
            )

        captured = make([3, 5, 7, 0], [7, 7, 12, 12])
        replay = make([9, 0, 13, 0], [7, 7, 12, 12])
        pointer = captured.deep_gemm_metadata.tensor.data_ptr()
        owners_pointer = captured.deep_gemm_metadata.indices.data_ptr()
        request_pointer = captured.request_indices.data_ptr()
        captured.copy_(replay)
        self.assertEqual(captured.deep_gemm_metadata.tensor.data_ptr(), pointer)
        self.assertEqual(captured.deep_gemm_metadata.indices.data_ptr(), owners_pointer)
        self.assertEqual(captured.request_indices.data_ptr(), request_pointer)
        self.assertEqual(captured.deep_gemm_metadata.indices.tolist(), [7, -2, 12, -4])
        self.assertTrue(
            torch.equal(
                captured.deep_gemm_metadata.tensor[:, 0], replay.compressed_seq_lens
            )
        )
        self.assertTrue(torch.equal(captured.request_indices, replay.request_indices))
        self.api.sail_paged_mqa_logits(
            self.packed(paged=True),
            torch.zeros(2, 64, 1, 68, dtype=torch.uint8),
            torch.ones(4, 32, dtype=torch.bfloat16),
            captured.compressed_seq_lens,
            captured.page_table,
            captured.deep_gemm_metadata,
            128,
            indices=captured.request_indices,
        )
        self.assertEqual(self.dg.get_paged_mqa_logits_metadata.call_count, 2)
        self.assertIs(
            self.dg.fp8_fp4_paged_mqa_logits.call_args.args[5],
            captured.deep_gemm_metadata.tensor,
        )
        self.assertIs(
            self.dg.fp8_fp4_paged_mqa_logits.call_args.kwargs["indices"],
            captured.deep_gemm_metadata.indices,
        )
        self.assertIsNone(make([], []).deep_gemm_metadata)
        self.assertIsNone(
            make([3, 5, 7, 0], [7, 7, 12, 12], build=False).deep_gemm_metadata
        )
        namespace["iter_row_chunks"].assert_not_called()
        self.assertEqual(self.dg.get_paged_mqa_logits_metadata.call_count, 2)

    def test_true_int8_forward_metadata_does_not_normalize_owners(self):
        from dataclasses import dataclass, field

        namespace = {
            "torch": torch,
            "dataclass": dataclass,
            "field": field,
            "is_sail": lambda: True,
            "plan_indexer_row_chunks": lambda **kwargs: None,
            "sail_paged_mqa_metadata": self.api.sail_paged_mqa_metadata,
        }
        metadata_class = load_symbol(
            ATTENTION / "dsv4/metadata.py", "PagedIndexerMetadata", namespace
        )
        with (
            patch.object(torch, "arange") as arange,
            patch.object(torch, "where") as where,
        ):
            metadata = metadata_class(
                page_size=256,
                compressed_page_size=64,
                page_table=torch.zeros(4, 2, dtype=torch.int32),
                compressed_seq_lens=torch.tensor([33, 64, 95, 0], dtype=torch.int32),
                request_indices=torch.tensor([5, 5, 9, 9], dtype=torch.int32),
                q_fp8_shape=torch.Size((4, 1, 32, 128)),
                qk_dtype=torch.int8,
                use_topk_v2=False,
            )
        arange.assert_not_called()
        where.assert_not_called()
        self.assertIsNone(metadata.deep_gemm_metadata.indices)
        self.assertIsNone(
            self.dg.get_paged_mqa_logits_metadata.call_args.kwargs["indices"]
        )

    def test_fp4_fused_producer_allocates_requested_weight_dtype(self):
        module = SimpleNamespace(forward=Mock())
        factory = Mock(return_value=module)
        namespace = {
            "torch": torch,
            "_is_hip": False,
            "_jit_main_q_indexer_rope_hadamard_fp4_quant_module": factory,
        }
        producer = load_symbol(
            ROOT / "python/sglang/kernels/ops/attention/dsv4/norm_rope.py",
            "fused_q_indexer_rope_hadamard_fp4_quant",
            namespace,
        )
        q = torch.ones(3, 32, 128, dtype=torch.bfloat16)
        weight = torch.ones(3, 32, dtype=torch.bfloat16)
        freqs = torch.ones(8, 32, dtype=torch.complex64)
        positions = torch.arange(3, dtype=torch.int32)
        for dtype in (torch.float32, torch.bfloat16):
            (_, scales), output = producer(
                q, weight, 0.125, freqs, positions, weights_dtype=dtype
            )
            self.assertEqual(output.dtype, dtype)
            self.assertEqual(output.shape, (3, 32, 1))
            self.assertEqual(scales.shape, (3, 32))
            self.assertIs(module.forward.call_args.args[4], output)
            self.assertEqual(factory.call_args.args, (q.dtype, dtype))
        self.assertEqual(module.forward.call_count, 2)

    def test_m890_forced_true_int8_rejected(self):
        for sm, forced in ((80, False), (80, True), (89, False), (89, True)):
            namespace = {
                "is_sail": lambda: True,
                "get_device_sm": lambda: sm,
                "envs": SimpleNamespace(
                    SGLANG_SAIL_DSV4_USE_INT8=SimpleNamespace(get=lambda: forced)
                ),
            }
            path = ATTENTION / "dsv4/indexer_quant.py"
            load_symbol(path, "dsv41_requires_int8_indexer", namespace)
            choose = load_symbol(path, "use_dsv41_int8_indexer", namespace)
            if sm == 89 and forced:
                with self.assertRaisesRegex(ValueError, "M890"):
                    choose()
            else:
                self.assertEqual(choose(), sm < 89)

    def test_bf16_dsa_v2_cast_preserves_scores_and_mapping(self):
        from enum import Enum, IntEnum, auto

        namespace = {
            "torch": torch,
            "is_sail": lambda: True,
            "Enum": Enum,
            "IntEnum": IntEnum,
            "auto": auto,
            "envs": SimpleNamespace(
                SGLANG_DSA_FUSE_TOPK=SimpleNamespace(get=lambda: True),
                SGLANG_OPT_USE_TOPK_V2=SimpleNamespace(get=lambda: True),
            ),
            "_topk_transform_v2_paged": Mock(return_value="mapped"),
        }
        path = ATTENTION / "dsa/dsa_topk_backend.py"
        load_symbol(path, "_sail_topk_fp32", namespace)
        load_symbol(path, "TopkTransformMethod", namespace)
        backend = load_symbol(path, "DSATopKBackend", namespace)
        scores = torch.tensor([[1, 3, 2, 99, 88]], dtype=torch.bfloat16)
        lengths = torch.tensor([3], dtype=torch.int32)
        metadata = SimpleNamespace(
            real_page_table=torch.tensor([[4]], dtype=torch.int32)
        )
        result = backend.SGL_KERNEL.topk_transform(
            scores, lengths, 2, namespace["TopkTransformMethod"].PAGED, metadata
        )
        self.assertEqual(result, "mapped")
        actual = namespace["_topk_transform_v2_paged"].call_args.args[0]
        self.assertEqual(actual.dtype, torch.float32)
        self.assertEqual(actual.stride(0) % 4, 0)
        self.assertTrue(torch.equal(actual, scores.float()))

    def test_ppu_chunk_budget_counts_aligned_rows_without_changing_cuda(self):
        namespace = {
            "torch": torch,
            "MQA_LOGITS_STATIC_SKIP_ELEMS": 8_000_000,
            "logger": Mock(),
            "MQA_LOGITS_ROW_ALIGN_ELEMS": 256,
            "MQA_LOGITS_BYTES_PER_ELEM": 4,
            "ceil_div": lambda x, y: (x + y - 1) // y,
            "is_hip": lambda: False,
        }
        for name in (
            "mqa_logits_needs_budget_check",
            "mqa_logits_row_bytes",
            "mqa_logits_should_chunk",
            "mqa_logits_rows_per_chunk",
        ):
            load_symbol(ATTENTION / "mqa_logits_utils.py", name, namespace)
        planner = load_symbol(
            ATTENTION / "dsv4/metadata.py", "plan_indexer_row_chunks", namespace
        )
        rows, width, budget = 30_000, 257, 35 * 1024 * 1024
        for ppu in (False, True):
            namespace["is_sail"] = lambda: ppu
            expected = budget // 2048 if ppu else None
            self.assertEqual(
                planner(
                    num_rows=rows,
                    num_cols=width,
                    budget_bytes=budget,
                    sm120_row_cap=None,
                ),
                expected,
            )
            namespace["capture_mode"] = SimpleNamespace(is_capture_mode=False)
            namespace["mqa_logits_budget_bytes"] = lambda **kwargs: budget
            chunks = load_symbol(
                ATTENTION / "dsa/dsa_indexer_kpool.py",
                "_mqa_logits_row_chunks",
                namespace,
            )
            with patch.object(
                torch.cuda, "is_current_stream_capturing", return_value=False
            ):
                actual = chunks(
                    num_rows=rows, num_cols=width, device=SimpleNamespace(index=0)
                )
            self.assertEqual(actual[0].stop, expected or rows)
            self.assertEqual(actual[-1].stop, rows)
        int8_chunks = load_symbol(
            ROOT / "python/sglang/kernels/ops/attention/dsv4/int8_indexer.py",
            "int8_logits_rows_per_chunk",
            namespace,
        )
        for width in (1, 128, 257, 1024):
            row_bytes = (width * 4 + 1023) // 1024 * 1024
            self.assertEqual(
                int8_chunks(width, budget), max(1, budget // row_bytes // 4 * 4)
            )

    def test_dsv4_prefill_cp_owners_follow_core_reindex_and_padding(self):
        path = ATTENTION / "deepseek_v4_backend.py"
        parallel = SimpleNamespace(attn_cp_rank=1, attn_cp_size=2)
        namespace = {
            "torch": torch,
            "is_sail": lambda: True,
            "get_parallel": lambda: parallel,
            "is_cp_active": lambda batch: batch.cp_active,
            "use_dsv41_int8_indexer": lambda: False,
            "get_platform": lambda: SimpleNamespace(is_sm120=False),
            "_is_xpu": False,
            "PagedIndexerMetadata": lambda **kwargs: SimpleNamespace(**kwargs),
        }
        apply_cp = load_symbol(path, "apply_cp_reindex", namespace)
        init_indexer = load_symbol(path, "init_forward_metadata_indexer", namespace)
        prefill = load_symbol(
            path,
            "init_forward_metadata_prefill",
            namespace,
            stop_after="indexer_metadata",
        )
        owners = torch.tensor([7, 7, 12, 12, 12, 12, 7, 7], dtype=torch.int32)
        lengths = torch.arange(8, dtype=torch.int32) + 1
        for cp_active, local_index in (
            (False, None),
            (True, None),
            (True, torch.tensor([5, 3, 1, 7])),
        ):
            core = SimpleNamespace(
                seq_lens_casual=lengths.clone(),
                page_table=torch.arange(8, dtype=torch.int32)[:, None],
                c4_topk_lengths_raw=lengths.clone(),
                _CP_REINDEX_FIELDS=(
                    "seq_lens_casual",
                    "page_table",
                    "c4_topk_lengths_raw",
                ),
                _CP_REINDEX_OPTIONAL_FIELDS=(),
                _CP_GLOBAL_FIELDS=(),
                init_flashmla_related=Mock(),
            )
            core.apply_cp_reindex = MethodType(apply_cp, core)
            model = SimpleNamespace(
                has_c4=True,
                index_head_dim=128,
                index_n_heads=32,
                enable_deepseek_v4_fp4_indexer=True,
                page_size=256,
                token_to_kv_pool=SimpleNamespace(request_window=None),
                dsa_topk_backend=SimpleNamespace(should_use_topk_v2=lambda: False),
                req_to_token=torch.zeros(16, 256, dtype=torch.int32),
                expand_prefill_casually=Mock(return_value=(lengths, owners)),
                make_core_attn_metadata=Mock(return_value=core),
            )
            model.init_forward_metadata_indexer = MethodType(init_indexer, model)
            metadata = prefill(
                model,
                256,
                torch.tensor([7, 12]),
                torch.tensor([8, 16]),
                [8, 16],
                torch.zeros(8, dtype=torch.int32),
                6,
                torch.tensor([2, 4]),
                [2, 4],
                forward_batch=SimpleNamespace(cp_active=cp_active),
                cp_metadata=SimpleNamespace(
                    local_index=local_index, per_rank_actual_token=[4, 4]
                ),
            )
            idx = (
                (local_index if local_index is not None else slice(1, None, 2))
                if cp_active
                else slice(None)
            )
            expected = torch.tensor([7, 7, 12, 12, 12, 12, -7, -8])[idx]
            self.assertEqual(metadata.request_indices.tolist(), expected.tolist())
            expected_lens = lengths[idx].masked_fill(expected < 0, 0)
            self.assertTrue(torch.equal(metadata.compressed_seq_lens, expected_lens))
            self.assertEqual(metadata.page_table.shape[0], expected.numel())
            self.assertEqual(metadata.q_fp8_shape, (expected.numel(), 1, 32, 64))
        with self.assertRaisesRegex(ValueError, "match query rows"):
            init_indexer(model, core, request_indices=torch.tensor([7]))


@unittest.skipUnless(
    os.environ.get("SGLANG_TEST_INDEXER_GPU") == "1",
    "M890 device tests require explicit opt-in",
)
class TestIndexerGPU(unittest.TestCase):
    def setUp(self):
        self.api = api_functions()
        self.assertTrue(torch.cuda.is_available())
        self.assertEqual(torch.cuda.get_device_capability(), (8, 9))

    def inputs(self, packed):
        device = "cuda"
        heads, dim, rows, keys = 32, 128, 4, 128
        values = torch.tensor([0.5, 1, 1.5, 2, 3, 4, 6], device=device)[
            torch.arange(keys, device=device) % 7
        ]
        weights = torch.ones(rows, heads, device=device)
        if packed:
            codes = (torch.arange(keys, device=device) % 7 + 1).to(torch.int8)
            k = (codes * 17)[:, None].expand(keys, 64).contiguous()
            q = (
                torch.full((rows, heads, 64), 0x22, device=device, dtype=torch.int8),
                torch.full((rows, heads), 0x7F7F7F7F, device=device, dtype=torch.int32),
            )
            kv = (k, torch.full((keys,), 0x7F7F7F7F, device=device, dtype=torch.int32))
        else:
            q = torch.ones(rows, heads, dim, device=device).to(torch.float8_e4m3fn)
            kv = (
                values[:, None].expand(keys, dim).contiguous().to(torch.float8_e4m3fn),
                torch.ones(keys, device=device),
            )
        reference = (values * heads * dim)[None, :].expand(rows, keys)
        return q, kv, weights, reference

    def test_nonpaged_reference_and_relative_columns(self):
        for packed in (False, True):
            q, kv, w, ref = self.inputs(packed)
            starts = torch.tensor([0, 20, 40, 70], device="cuda", dtype=torch.int32)
            ends = torch.tensor([9, 37, 66, 95], device="cuda", dtype=torch.int32)
            out = self.api.sail_mqa_logits(q, kv, w, starts, ends, max_seqlen_k=32)
            self.assertEqual(out.shape, (4, 32))
            self.assertEqual(out.dtype, torch.bfloat16 if packed else torch.float32)
            self.assertEqual(out.stride(0) * out.element_size() % 1024, 0)
            for row, (start, end) in enumerate(zip(starts.tolist(), ends.tolist())):
                torch.testing.assert_close(
                    out[row, : end - start].float(),
                    ref[row, start:end],
                    rtol=0,
                    atol=0,
                )
        torch.cuda.synchronize()

    def test_dsa_ragged_relative_topk_with_chunk_and_padding(self):
        from contextlib import nullcontext

        q, kv, weights, reference = self.inputs(True)
        device = q[0].device
        namespace = {
            "torch": torch,
            "TYPE_CHECKING": False,
            "_is_hip": False,
            "_is_xpu": False,
            "is_sail": lambda: True,
            "_enable_logits_nan_check": False,
            "sail_mqa_logits": self.api.sail_mqa_logits,
            "mqa_logits_row_bytes": lambda width: (width * 4 + 1023) // 1024 * 1024,
        }
        path = ATTENTION / "dsa/dsa_indexer.py"
        ragged = load_symbol(path, "_get_topk_ragged", namespace)
        dispatch = load_symbol(path, "_get_mqa_logits_dispatch", namespace)
        pad_heads = load_symbol(path, "_pad_heads_for_deep_gemm", namespace)
        topk_path = ATTENTION / "dsa/dsa_topk_backend.py"
        cast_logits = load_symbol(topk_path, "_sail_topk_fp32", namespace)
        select = load_symbol(topk_path, "_topk_transform_v2_ragged", namespace)
        starts = torch.tensor([0, 0, 40, 40], device=device, dtype=torch.int32)
        ends = torch.tensor([9, 17, 66, 75], device=device, dtype=torch.int32)
        lengths = ends - starts
        owners = torch.tensor([0, 0, 1, 1], device=device, dtype=torch.int32)
        pool = SimpleNamespace(
            index_page_size=64, get_index_k_scale_buffer=Mock(return_value=kv)
        )
        namespace["get_token_to_kv_pool"] = lambda: pool
        model = SimpleNamespace(
            use_fp4=True,
            use_fp8=False,
            use_int8=False,
            bf16_indexer=False,
            index_topk=2,
            _mask_init_and_local_tokens=Mock(),
            _with_real_sm_count=nullcontext,
        )
        model._pad_heads_for_deep_gemm = pad_heads
        model._get_mqa_logits_dispatch = MethodType(dispatch, model)
        metadata = SimpleNamespace(
            get_logical_page_table=lambda: torch.zeros(
                2, 2, device=device, dtype=torch.int32
            ),
            get_indexer_kvcache_range=lambda: (starts, ends),
            get_indexer_seq_len_cpu=lambda: torch.tensor([40, 87]),
            get_indexer_seq_len=lambda: torch.tensor([40, 87], device=device),
            get_seqlens_expanded=lambda: lengths,
            get_token_to_batch_idx=lambda: owners,
            attn_metadata=SimpleNamespace(topk_indices_offset=starts),
        )

        def transform(logits, topk, *, ks, ke_offset, **kwargs):
            self.assertTrue(bool((ks == 0).all()))
            offset = kwargs.get("topk_indices_offset_override")
            offset = starts if offset is None else offset
            return select(cast_logits(logits), ke_offset, topk, offset, ks)

        metadata.topk_transform = transform
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: True),
            seq_lens_cpu=torch.tensor([40, 87]),
            extend_seq_lens_cpu=[2, 2],
        )
        valid = (torch.arange(128, device=device)[None, :] >= starts[:, None]) & (
            torch.arange(128, device=device)[None, :] < ends[:, None]
        )
        masked = reference.masked_fill(~valid, float("-inf"))
        best = masked.topk(2, dim=1).values
        q_padded = (torch.cat((q[0], q[0][:1])), torch.cat((q[1], q[1][:1])))
        weights_padded = torch.cat((weights, weights[:1])).unsqueeze(-1)
        for chunked in (False, True):
            namespace["mqa_logits_should_chunk"] = Mock(return_value=(chunked, 2048))
            result = ragged(model, False, batch, 0, q_padded, weights_padded, metadata)
            self.assertEqual(result.shape, (5, 2))
            self.assertTrue(bool((result[-1] == -1).all()))
            self.assertTrue(bool((result[:4] >= starts[:, None]).all()))
            self.assertTrue(bool((result[:4] < ends[:, None]).all()))
            selected = reference.gather(1, result[:4].long())
            torch.testing.assert_close(
                selected.sort(dim=1).values, best.sort(dim=1).values, rtol=0, atol=0
            )
            if not chunked:
                single_pass = selected.clone()
            else:
                torch.testing.assert_close(
                    selected.sort(dim=1).values,
                    single_pass.sort(dim=1).values,
                    rtol=0,
                    atol=0,
                )
        torch.cuda.synchronize()

    def test_paged_reference_and_graph_replay(self):
        from sglang.kernels.ops.attention.dsv4.topk import plan_topk_v2

        namespace = {"torch": torch}
        topk_path = ATTENTION / "dsa/dsa_topk_backend.py"
        cast_logits = load_symbol(topk_path, "_sail_topk_fp32", namespace)
        paged_topk = load_symbol(topk_path, "_topk_transform_v2_paged", namespace)
        for packed in (False, True):
            q, kv, w, ref = self.inputs(packed)
            if packed:
                q = (q[0].unsqueeze(1), q[1].unsqueeze(1))
                scales = kv[1].view(torch.uint8)
            else:
                q = q.unsqueeze(1)
                scales = kv[1].view(torch.uint8)
            width = kv[0].shape[1]
            cache = torch.cat(
                (
                    kv[0].view(torch.uint8).reshape(2, 64 * width),
                    scales.reshape(2, 64 * 4),
                ),
                dim=1,
            ).view(2, 64, 1, width + 4)
            lens = torch.tensor([33, 64, 95, 0], device="cuda", dtype=torch.int32)
            # A nonidentity page table checks logits reads and physical KV slot mapping.
            table = torch.tensor([[1, 0]] * 4, device="cuda", dtype=torch.int32)
            logical_ref = torch.cat((ref[:, 64:], ref[:, :64]), dim=1)
            metadata = SimpleNamespace(
                real_page_table=table,
                page_size=64,
                topk_v2_plan=torch.empty(5, 2, device="cuda", dtype=torch.int32),
            )
            ids = (
                torch.tensor([5, 5, 9, 9], device="cuda", dtype=torch.int32)
                if packed
                else None
            )

            def make_plan():
                return self.api.sail_paged_mqa_metadata(
                    lens, 64, 32, 128, torch.int8 if packed else q.dtype, ids
                )

            def compute(plan=None):
                if plan is None:
                    plan = make_plan()
                return self.api.sail_paged_mqa_logits(
                    q, cache, w, lens, table, plan, 128, indices=ids
                )

            def select(logits):
                plan_topk_v2(lens, out=metadata.topk_v2_plan)
                scores = cast_logits(logits) if packed else logits
                return paged_topk(scores, lens, 2, metadata)

            def check_indices(indices):
                self.assertTrue(bool((indices[-1] == -1).all()))
                positions = indices[:3] ^ 64
                self.assertTrue(bool((positions >= 0).all()))
                self.assertTrue(bool((positions < lens[:3, None]).all()))
                selected = ref[:3].gather(1, indices[:3].long())
                valid = torch.arange(128, device="cuda")[None, :] < lens[:3, None]
                best = logical_ref[:3].masked_fill(~valid, float("-inf")).topk(2).values
                torch.testing.assert_close(
                    selected.sort(dim=1).values,
                    best.sort(dim=1).values,
                    rtol=0,
                    atol=0,
                )

            out = compute()
            valid = torch.arange(128, device="cuda")[None, :] < lens[:, None]
            torch.testing.assert_close(
                out[valid].float(), logical_ref[valid], rtol=0, atol=0
            )
            check_indices(select(out))
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                select(compute())
                select(compute())
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = compute()
                captured_indices = select(captured)
            lens.copy_(torch.tensor([17, 65, 101, 0], device="cuda", dtype=torch.int32))
            if packed:
                ids.copy_(torch.tensor([5, 9, 9, 9], device="cuda", dtype=torch.int32))
            graph.replay()
            valid = torch.arange(128, device="cuda")[None, :] < lens[:, None]
            torch.testing.assert_close(
                captured[valid].float(), logical_ref[valid], rtol=0, atol=0
            )
            check_indices(captured_indices)

            shared_plan = make_plan()
            select(compute(shared_plan))
            torch.cuda.synchronize()
            shared_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(shared_graph):
                shared_out = compute(shared_plan)
                shared_indices = select(shared_out)
            schedule_pointer = shared_plan.tensor.data_ptr()
            owners_pointer = shared_plan.indices.data_ptr() if packed else None
            lens.copy_(torch.tensor([23, 49, 87, 0], device="cuda", dtype=torch.int32))
            if packed:
                ids.copy_(torch.tensor([5, 5, 9, 9], device="cuda", dtype=torch.int32))
            shared_plan.copy_(make_plan())
            self.assertEqual(shared_plan.tensor.data_ptr(), schedule_pointer)
            if packed:
                self.assertEqual(shared_plan.indices.data_ptr(), owners_pointer)
                self.assertEqual(shared_plan.indices.tolist(), [5, 5, 9, -4])
            shared_graph.replay()
            valid = torch.arange(128, device="cuda")[None, :] < lens[:, None]
            torch.testing.assert_close(
                shared_out[valid].float(), logical_ref[valid], rtol=0, atol=0
            )
            check_indices(shared_indices)
        torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main()
