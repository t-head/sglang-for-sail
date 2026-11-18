"""CPU row-order checks; transport and projection are mocked, not GPU accuracy."""

import ast
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
SOURCE = Path(__file__).resolve().parents[5] / "python/sglang/srt"


def load_function(path, name, namespace):
    # Run the actual methods without importing GPU-only backend dependencies.
    nodes = [
        n
        for n in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(n, ast.FunctionDef) and n.name == name
    ]
    assert len(nodes) == 1
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    tree = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace[name]


class TestCpProjectGather(CustomTestCase):
    def test_owned_rows_padding_and_tail_order(self):
        for lengths in ([1], [3, 2], [4, 5, 7], [0, 3, 1], [128, 127]):
            for rank in range(4):
                for scored in (False, True):
                    for tail in (False, True):
                        with self.subTest(
                            lengths=lengths, rank=rank, scored=scored, tail=tail
                        ):
                            self.check_case(
                                lengths, rank, scored, tail, True, True, True
                            )

    def test_disabled_paths_and_non_ppu(self):
        for ppu in (False, True):
            for compressor_on in (False, True):
                for indexer_on in (False, True):
                    with self.subTest(
                        ppu=ppu, compressor=compressor_on, indexer=indexer_on
                    ):
                        self.check_case(
                            [3, 6], 2, True, False, ppu, compressor_on, indexer_on
                        )

    def check_case(self, lengths, rank, scored, tail, ppu, compressor_on, indexer_on):
        total = sum(lengths)
        global_x = torch.arange(total * 8, dtype=torch.float64).reshape(total, 8) / 32
        req = torch.repeat_interleave(torch.arange(len(lengths)), torch.tensor(lengths))
        pos = torch.cat([torch.arange(n) + 101 + i * 2 for i, n in enumerate(lengths)])
        original_indices, start = [], 0
        for n in lengths:
            original_indices.extend(range(start + 7, start + 7 + n))
            start += 7 + n
        owners = (torch.tensor(original_indices) if tail else torch.arange(total)) % 4
        counts = [int((owners == r).sum()) for r in range(4)]
        owned = owners == rank
        local = global_x[owned]
        local_lens = [int((req[owned] == i).sum()) for i in range(len(lengths))]
        physical = max(counts)
        x = torch.full((physical, 8), float("nan"), dtype=torch.float64)
        x[: len(local)] = local
        weight = torch.arange(24, dtype=torch.float64).reshape(8, 3) / 16

        def project(value):
            return value @ weight, value @ (weight + 0.25) if scored else None

        expected_kv, expected_score = project(global_x)
        expected_packed = (
            torch.cat((expected_kv, expected_score), -1) if scored else expected_kv
        )
        wanted = expected_packed if ppu else global_x
        slots, gather_indices = [0] * 4, []
        for owner in owners.tolist():
            gather_indices.append(owner * physical + slots[owner])
            slots[owner] += 1
        batch = NS(
            attn_cp_metadata=NS(
                total_seq_lens=total,
                per_rank_actual_token=[physical] * 4,
                per_rank_logical_token=counts,
                gather_index=torch.tensor(gather_indices) if tail else None,
            ),
            extend_seq_lens_cpu=list(lengths),
            positions=pos,
        )

        def all_gather(destination, value):
            self.assertTrue(torch.equal(value[: len(local)], wanted[owned]))
            self.assertEqual(torch.count_nonzero(value[len(local) :]).item(), 0)
            for r in range(4):
                destination[r * physical : (r + 1) * physical].zero_()
                destination[r * physical : r * physical + counts[r]].copy_(
                    wanted[owners == r]
                )

        actual_gather = load_function(
            SOURCE / "layers/cp/interleave.py",
            "_gather_interleaved_tensor",
            {
                "torch": torch,
                "get_parallel": lambda: NS(attn_cp_group=None),
                "use_symmetric_memory": lambda *a, **k: nullcontext(),
                "is_allocation_symmetric": lambda: False,
                "attn_cp_all_gather_into_tensor": all_gather,
            },
        )
        calls = []

        def gather(value, fb, stream):
            if ppu:
                self.assertEqual(len(value), len(local))
            result = actual_gather(NS(cp_rank=rank, cp_size=4), value, fb)
            self.assertTrue(torch.equal(result, wanted))
            calls.append("gather")
            return result

        def consume(layer, value, requests, positions, projected=None):
            self.assertEqual(projected is not None, ppu)
            kv, score = project(value) if projected is None else projected
            self.assertTrue(torch.equal(kv, expected_kv))
            self.assertTrue(
                torch.equal(score, expected_score) if scored else score is None
            )
            self.assertTrue(torch.equal(requests, req))
            self.assertTrue(torch.equal(positions, pos))
            calls.append("compressor")

        def indexer(*args, **kwargs):
            self.assertTrue(torch.equal(args[1], local))
            self.assertEqual(kwargs["rows_per_request"], local_lens)
            calls.append("indexer")

        fn = load_function(
            SOURCE / "layers/attention/deepseek_v4_backend.py",
            "_forward_low_ratio_sources_cp",
            {
                "torch": torch,
                "get_platform": lambda: NS(is_ppu=ppu),
                "get_parallel": lambda: NS(attn_cp_rank=rank, attn_cp_size=4),
                "_as_int_list": list,
                "token_req_indices": lambda *a, **k: req,
                "interleave_rows_per_request": lambda *a: local_lens,
                "cp_materialize_global_token_order": gather,
            },
        )
        metadata = NS(
            late_layer_tail=NS(
                local_lens_cpu=local_lens, req_global=req, pos_global=pos
            )
            if tail
            else None
        )
        owner = NS(
            forward_metadata=metadata,
            _low_ratio_compress_torch=consume,
            _low_ratio_index_topk=indexer,
        )
        layer = NS(compressor=NS(project=project), indexer=object())
        with patch.object(torch.cuda, "current_stream", return_value=None):
            fn(
                owner,
                layer=layer,
                x=x,
                q_lora=x,
                positions=pos[owned],
                forward_batch=batch,
                run_compressor=compressor_on,
                run_indexer=indexer_on,
            )
        self.assertEqual(
            calls,
            (["gather", "compressor"] if compressor_on else [])
            + (["indexer"] if indexer_on else []),
        )


if __name__ == "__main__":
    unittest.main()
