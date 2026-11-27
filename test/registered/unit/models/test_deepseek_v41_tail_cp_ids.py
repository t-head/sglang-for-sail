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
    def test_tail_source_kv_keeps_shared_backend_contract(self):
        for metadata in (NS(), NS(real_num_tokens=2)):
            for num_rows in (0, 4):
                with self.subTest(metadata=vars(metadata), num_rows=num_rows):
                    x = torch.arange(num_rows * 2).view(num_rows, 2)
                    positions = torch.arange(num_rows)
                    batch, layer = NS(), NS(layer_id=7)
                    calls = []
                    backend = NS(
                        forward_metadata=metadata,
                        forward_low_ratio_sources=lambda **kwargs: calls.append(kwargs),
                    )
                    DeepseekV4AttnBackend.write_tail_source_kv(
                        backend,
                        layer=layer,
                        x=x,
                        positions=positions,
                        forward_batch=batch,
                    )
                    self.assertEqual(backend.tail_source_kv_layer_id, layer.layer_id)
                    self.assertEqual(len(calls), int(num_rows > 0))
                    if calls:
                        self.assertIs(calls[0]["x"], x)
                        self.assertIs(calls[0]["positions"], positions)
                        self.assertIs(calls[0]["forward_batch"], batch)
                        self.assertIs(calls[0]["layer"], layer)
                        self.assertIsNone(calls[0]["q_lora"])
                        self.assertFalse(calls[0]["run_indexer"])

    def test_contiguous_tail_ignores_mlp_sync_padding(self):
        indices = torch.tensor([4, 5, 6])
        rows = torch.arange(16).view(8, 2)
        selected = deepseek_v4_backend._tail_rows(
            rows, token_indices=indices, contiguous_start=4
        )
        torch.testing.assert_close(selected, rows[indices])
        tail = NS(contiguous_start=4, scatter_indices=None)
        result = deepseek_v4._scatter_tail_rows(tail, selected, 8)
        torch.testing.assert_close(result[4:7], selected)

    def test_eager_metadata_uses_real_rows_and_restores_batch(self):
        batch = NS(
            forward_mode=ForwardMode.EXTEND,
            global_num_token_non_padded_cpu=129,
            positions=torch.arange(132),
            out_cache_loc=torch.arange(132),
        )
        positions, cache = batch.positions, batch.out_cache_loc
        observed = []

        def build(fb, *, real_num_tokens=None):
            observed.append((len(fb.positions), len(fb.out_cache_loc), real_num_tokens))
            return NS(real_num_tokens=real_num_tokens)

        backend = NS(
            mtp_enabled=False,
            enable_decoder_swa_bounded_replay=True,
            _build_forward_metadata=build,
            init_forward_metadata_in_graph=lambda fb: None,
            _build_late_layer_tail_metadata=lambda fb: len(fb.positions),
            _activate_request_window=lambda: None,
        )
        DeepseekV4AttnBackend.init_forward_metadata(backend, batch)
        self.assertEqual(observed, [(129, 129, 129)])
        self.assertEqual(backend.tail_forward_metadata, 129)
        self.assertIs(batch.positions, positions)
        self.assertIs(batch.out_cache_loc, cache)


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
