"""CPU regressions for CP late-layer-tail routing IDs.

The late-layer tail keeps the last tokens of each request. Under prefill CP the
tail rows are selected from the rank's already-sharded hidden states, so their
routing IDs must come from the same layout. Re-sharding the compact tail from
zero shifts token ownership when the tail starts off the original CP boundary.
"""

import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from sglang.srt.layers.attention import deepseek_v4_backend
from sglang.srt.layers.attention.deepseek_v4_backend import (
    DeepseekV4AttnBackend,
    LateLayerTail,
)
from sglang.srt.layers.moe.utils import MoeA2ABackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models import deepseek_v4
from sglang.srt.runtime_context import get_flags, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestEagerTailRows(CustomTestCase):
    def test_compacted_tail_bounds_and_restores_counts(self):
        """An eager 128-row tail must not retain its full 129-row extent."""
        for graph in (False, True):
            with self.subTest(graph=graph):
                global_count = torch.tensor(129, dtype=torch.int32)
                local_count = torch.tensor(30, dtype=torch.int32)
                batch = NS(
                    attn_cp_metadata=None,
                    global_num_token_non_padded_cpu=129,
                    global_num_token_non_padded=global_count,
                    num_token_non_padded=local_count,
                )
                backend = object.__new__(DeepseekV4AttnBackend)
                backend.forward_metadata = object()
                original_metadata = backend.forward_metadata
                backend.tail_forward_metadata = NS(
                    late_layer_tail=NS(
                        real_rows_count=128 if graph else None,
                        extend_seq_lens_cpu=[128],
                        cp_metadata=None,
                    ),
                    core_attn_metadata=NS(),
                )
                backend._bcg_tail_num_token_non_padded = None
                backend.token_to_kv_pool = NS(request_window=None)
                with (
                    patch.object(
                        deepseek_v4_backend, "get_local_dp_buffer_len", return_value=132
                    ),
                    patch.object(deepseek_v4_backend, "set_local_dp_buffer_len"),
                ):
                    saved = backend.enter_late_layer_tail(batch)
                    self.assertEqual(batch.global_num_token_non_padded_cpu, 128)
                    self.assertEqual(batch.global_num_token_non_padded.item(), 128)
                    self.assertEqual(batch.num_token_non_padded.item(), 128)
                    backend.exit_late_layer_tail(saved, batch)
                self.assertIs(backend.forward_metadata, original_metadata)
                self.assertEqual(batch.global_num_token_non_padded_cpu, 129)
                self.assertIs(batch.global_num_token_non_padded, global_count)
                self.assertIs(batch.num_token_non_padded, local_count)


class TestDSV41TailCPRoutingIDs(CustomTestCase):
    def _cp_tail(self, *, total: int):
        token_indices = torch.arange(1, total)
        forward_batch = NS(batch_size=1, positions=torch.arange(total))
        with get_parallel().override(attn_cp_rank=0, attn_cp_size=2):
            return DeepseekV4AttnBackend._late_layer_tail_cp_layout(
                None,
                forward_batch,
                token_indices,
                torch.tensor([total - 1], dtype=torch.int32),
            )

    def test_off_boundary_tail_keeps_original_cp_ownership(self):
        total = 129
        cp_tail = self._cp_tail(total=total)
        full_ids = torch.arange(total)
        local_ids = full_ids[0::2]
        selected = local_ids[cp_tail["local_token_indices"]]
        expected = full_ids[2::2]
        torch.testing.assert_close(selected, expected)

        # The old compact re-shard starts at zero and loses the boundary parity.
        compact = full_ids[1:][0::2]
        self.assertFalse(torch.equal(compact, expected))

    def test_uneven_tail_pads_without_changing_ownership(self):
        total = 128
        cp_tail = self._cp_tail(total=total)
        self.assertEqual(cp_tail["pad_rows"], 1)
        full_ids = torch.arange(total)
        local_ids = full_ids[0::2]
        selected = local_ids[cp_tail["local_token_indices"]]
        torch.testing.assert_close(selected, full_ids[2::2])

    def test_routing_ids_use_local_rows_and_all_gather_for_no_a2a(self):
        local_ids = torch.arange(3)
        with (
            get_parallel().override(attn_cp_size=2),
            patch.object(deepseek_v4, "attn_cp_all_gather_into_tensor") as gather,
        ):
            with get_flags().moe.override(a2a_backend=MoeA2ABackend.DEEPEP):
                input_ids, input_ids_global = deepseek_v4._tail_cp_routing_ids(
                    local_input_ids=local_ids
                )
                torch.testing.assert_close(input_ids, local_ids)
                torch.testing.assert_close(input_ids_global, local_ids)
                gather.assert_not_called()

            def fake_all_gather(out, local):
                # Rank 0 keeps its local rows; rank 1 has distinct IDs.
                out.copy_(torch.cat([local, local + 100]))

            gather.side_effect = fake_all_gather
            with get_flags().moe.override(a2a_backend=MoeA2ABackend.NONE):
                input_ids, input_ids_global = deepseek_v4._tail_cp_routing_ids(
                    local_input_ids=local_ids
                )
            torch.testing.assert_close(input_ids, local_ids)
            torch.testing.assert_close(
                input_ids_global, torch.tensor([0, 1, 2, 100, 101, 102])
            )
            gather.assert_called_once()

    def test_output_indices_keep_global_cp_rows_and_drop_graph_padding(self):
        # CP hidden rows are local; output indices must still address full IDs.
        for global_indices, real_rows, expected in (
            (torch.tensor([2, 4, 6]), None, [2, 4, 6]),
            (None, 2, [1, 2]),
        ):
            with self.subTest(cp=global_indices is not None):
                tail = LateLayerTail(
                    token_indices=torch.tensor([1, 2, 0]),
                    positions=torch.tensor([2, 4, 0]),
                    extend_seq_lens=torch.tensor([2], dtype=torch.int32),
                    extend_seq_lens_cpu=[2],
                    swa_out_cache_loc=torch.zeros(3, dtype=torch.int32),
                    global_token_indices=global_indices,
                    real_rows_count=real_rows,
                )
                torch.testing.assert_close(
                    tail.output_token_indices, torch.tensor(expected)
                )

    def test_mm_preparation_preserves_scheduler_ids_and_cp_kwargs(self):
        ids = torch.tensor([1, deepseek_v4.MM_PAD_SHIFT_VALUE + 7, 2])
        scheduler_ids = ids.clone()
        embeds = torch.arange(12, dtype=torch.float32).view(3, 4)
        positions = torch.arange(3)
        batch = NS(
            input_ids=ids,
            positions=positions,
            forward_mode=ForwardMode.EXTEND,
            contains_mm_inputs=lambda: True,
        )

        def embed_mm(full_ids, forward_batch):
            # The vision boundary must see the original, unsharded hash IDs.
            torch.testing.assert_close(full_ids, scheduler_ids)
            self.assertIs(forward_batch, batch)
            return embeds

        model = NS(
            vision=object(),
            config=NS(image_token_id=99),
            _prepare_mm_embeddings=embed_mm,
        )
        model_ids, mm_embeds = (
            deepseek_v4.DeepseekV4ForCausalLM.prepare_language_model_inputs(
                model, ids, batch
            )
        )
        torch.testing.assert_close(model_ids, torch.tensor([1, 99, 2]))
        torch.testing.assert_close(batch.input_ids, scheduler_ids)
        proxy = object()
        cp_embeds, cp_positions, model_kwargs = (
            deepseek_v4.DeepseekV4ForCausalLM.prepare_cp_inputs(
                model, batch, input_embeds=mm_embeds, pp_proxy_tensors=proxy
            )
        )
        self.assertIs(cp_embeds, embeds)
        self.assertIs(cp_positions, positions)
        self.assertEqual(model_kwargs, {"pp_proxy_tensors": proxy})


if __name__ == "__main__":
    unittest.main()
