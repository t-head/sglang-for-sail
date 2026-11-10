"""CPU contracts for the DCP page-layout MLA integration seams."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.kernels.ops.attention import set_mla_kv_concat_q as fused_module
from sglang.srt.layers.attention import cutedsl_mla_backend as cute_module
from sglang.srt.layers.attention import trtllm_mla_backend as trt_module
from sglang.srt.layers.attention.cutedsl_mla_backend import CuteDslMLABackend
from sglang.srt.layers.attention.trtllm_mla_backend import TRTLLMMLABackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestDcpPageMlaWriterContracts(CustomTestCase):
    def test_fused_fp8_wrapper_passes_page_or_token_layout_to_kernel(self):
        backend = object.__new__(TRTLLMMLABackend)
        backend.page_size = 64
        backend.kv_index_translator = SimpleNamespace(is_translating=False)
        backend.token_to_kv_pool = SimpleNamespace(
            get_key_buffer=lambda _layer: torch.zeros((4, 1, 576))
        )
        layer = SimpleNamespace(
            layer_id=0, tp_q_head_num=1, v_head_dim=512, head_dim=576
        )
        q = torch.zeros((1, 512), dtype=torch.bfloat16)
        q_rope = torch.zeros((1, 64), dtype=torch.bfloat16)
        k = torch.zeros((1, 1, 512), dtype=torch.bfloat16)
        k_rope = torch.zeros((1, 1, 64), dtype=torch.bfloat16)

        with (
            patch.object(
                trt_module, "set_mla_kv_concat_q_fp8_covered", return_value=True
            ) as covered,
            patch.object(
                trt_module, "set_mla_kv_concat_q_fp8", return_value=q
            ) as fused,
        ):
            for layout, expected_page_size in (("token", 0), ("page", 64)):
                backend.dcp_kv_layout = layout
                with (
                    self.subTest(layout=layout),
                    get_context().override_server_args(
                        dcp_size=3,
                        dcp_kv_layout="page" if layout == "token" else "token",
                    ),
                    get_parallel().override(dcp_rank=1, attn_dcp_rank=1),
                ):
                    query = backend._set_kv_and_concat_q_fp8_fused(
                        layer, torch.tensor([3]), q, q_rope, k, k_rope
                    )
                    self.assertIs(query, q)
                    self.assertEqual(
                        fused.call_args.kwargs["dcp_page_size"], expected_page_size
                    )
                    self.assertEqual(fused.call_args.kwargs["dcp_world_size"], 3)
                    self.assertEqual(fused.call_args.kwargs["dcp_rank"], 1)
                    self.assertEqual(covered.call_args.kwargs["dcp_world_size"], 3)
                    self.assertEqual(
                        covered.call_args.kwargs["dcp_page_size"], expected_page_size
                    )

    def test_fp8_warmup_and_launch_share_geometry_specializations(self):
        """Warmup and launch share each world/page specialization across ranks."""
        # Keep the real JIT cache and wrapper; replace only compilation/launch.
        module_cache = fused_module.cache_once(
            fused_module.set_mla_kv_concat_q_fp8_module.__wrapped__
        )
        inputs = (
            torch.empty((4, 576), dtype=torch.float8_e4m3fn),
            torch.tensor([2], dtype=torch.int64),
            torch.zeros((1, 512), dtype=torch.bfloat16),
            torch.zeros((1, 64), dtype=torch.bfloat16),
            torch.zeros((1, 1, 512), dtype=torch.bfloat16),
            torch.zeros((1, 1, 64), dtype=torch.bfloat16),
        )
        with (
            patch.object(fused_module, "set_mla_kv_concat_q_fp8_module", module_cache),
            patch.object(fused_module, "load_jit") as load,
            patch.object(fused_module, "is_arch_support_pdl", return_value=False),
            patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)),
        ):
            for count, (world_size, page_size) in enumerate(
                ((1, 0), (2, 0), (3, 64), (8, 64), (3, 3)), start=1
            ):
                with self.subTest(world_size=world_size, page_size=page_size):
                    self.assertTrue(
                        fused_module.can_use_set_mla_kv_concat_q_fp8.__wrapped__(
                            world_size, page_size
                        )
                    )
                    self.assertEqual(load.call_count, count)
                    self.assertEqual(
                        load.call_args.args,
                        (
                            "set_mla_kv_concat_q_fp8",
                            str(world_size),
                            str(page_size),
                            "false",
                        ),
                    )
                    self.assertEqual(
                        load.call_args.kwargs["cuda_wrappers"],
                        [
                            (
                                "set_mla_kv_concat_q_fp8",
                                f"SetMlaKVConcatQFp8Kernel<{world_size}, {page_size}, false>::run",
                            )
                        ],
                    )
                    for rank in range(world_size):
                        query = fused_module.set_mla_kv_concat_q_fp8(
                            *inputs,
                            num_warps=4,
                            dcp_world_size=world_size,
                            dcp_rank=rank,
                            dcp_page_size=page_size,
                        )
                        args = load.return_value.set_mla_kv_concat_q_fp8.call_args.args
                        self.assertEqual(len(args), 9)
                        self.assertEqual(args[-2:], (4, rank))
                        self.assertIs(query, args[6])
                        self.assertEqual(query.shape, (1, 1, 576))
                        self.assertEqual(query.dtype, torch.float8_e4m3fn)
                    self.assertEqual(load.call_count, count)

    def test_backend_keeps_its_layout_after_fp8_warmup(self):
        """Ambient config changes must not reinterpret an existing backend's KV."""
        model_runner = SimpleNamespace(
            model_config=SimpleNamespace(
                num_attention_heads=8,
                get_num_kv_heads=lambda _tp: 1,
                kv_lora_rank=512,
                qk_nope_head_dim=128,
                qk_rope_head_dim=64,
                v_head_dim=512,
            ),
            kv_cache_dtype=torch.float8_e4m3fn,
            dtype=torch.bfloat16,
            page_size=64,
            req_to_token_pool=SimpleNamespace(req_to_token=torch.empty((4, 8))),
            token_to_kv_pool=SimpleNamespace(),
            device="cpu",
            max_running_requests=4,
            is_draft_worker=False,
        )
        with (
            patch.object(
                trt_module.FlashInferMLAAttnBackend, "__init__", return_value=None
            ),
            patch.object(
                trt_module, "global_cute_dsl_workspace_buffer", torch.empty(0)
            ),
            patch.object(
                trt_module,
                "make_persistent_multi_ctas_kv_counter_buffer",
                return_value=torch.empty(0),
            ),
            patch.object(
                trt_module.envs.SGLANG_ENABLE_ASYNC_ASSERT, "get", return_value=False
            ),
            patch.object(
                trt_module, "can_use_set_mla_kv_concat_q_fp8", return_value=True
            ) as warmup,
        ):
            for layout, page_size in (("token", 0), ("page", 64)):
                backend = object.__new__(TRTLLMMLABackend)
                backend.device = "cpu"
                with (
                    self.subTest(layout=layout),
                    get_context().override_server_args(
                        dcp_size=3,
                        dcp_kv_layout=layout,
                        disable_chunked_prefix_cache=False,
                        speculative_num_draft_tokens=None,
                    ),
                    get_parallel().override(dcp_rank=1, attn_dcp_rank=1),
                ):
                    backend.__init__(model_runner, backend="cute-dsl")
                    self.assertTrue(backend._fused_set_kv_concat_q_fp8)
                    warmup.assert_called_with(3, page_size)
                    get_context().override(
                        "test", dcp_kv_layout="page" if layout == "token" else "token"
                    )
                    # N=3, rank=1, S=64: length 65 gives 22 token rows or 1 page row.
                    expected_length = 22 if layout == "token" else 1
                    torch.testing.assert_close(
                        backend._get_dcp_local_seq_lens(torch.tensor([65])),
                        torch.tensor([expected_length], dtype=torch.int32),
                    )
                    self.assertEqual(
                        backend._get_dcp_local_max_seq_len(65), expected_length
                    )


class TestDcpPageMetadataContracts(CustomTestCase):
    def test_eager_page_table_takes_global_lengths_and_metadata_stores_local(self):
        """Page-table creation consumes global lengths, not pre-sharded lengths."""
        backend = object.__new__(TRTLLMMLABackend)
        backend.page_size = 2
        backend.max_context_len = 16
        backend.num_draft_tokens = 2
        backend._kv_shard_pool = None
        backend._fill_dcp_block_kv_indices = MagicMock()
        backend.kv_index_translator = SimpleNamespace(
            is_translating=True, fill_read_table=MagicMock(), copy_page_table=MagicMock()
        )
        cases = (
            ("page", True, ForwardMode.DECODE, [0, 2, 3], [1, 4, 9]),
            ("token", True, ForwardMode.DECODE, [0, 1, 3], [1, 4, 9]),
            ("token", True, ForwardMode.TARGET_VERIFY, [1, 2, 4], [3, 6, 11]),
            ("token", True, ForwardMode.DRAFT_EXTEND_V2, [1, 1, 3], [3, 4, 10]),
            ("token", False, ForwardMode.DECODE, [1, 4, 9], None),
            ("token", False, ForwardMode.TARGET_VERIFY, [3, 6, 11], [3, 6, 11]),
        )
        for layout, enabled, mode, expected_local, expected_global in cases:
            backend.dcp_kv_layout = layout
            batch = SimpleNamespace(
                forward_mode=mode,
                batch_size=3,
                seq_lens=torch.tensor([1, 4, 9], dtype=torch.int32),
                seq_lens_cpu=torch.tensor([1, 4, 9]),
                req_pool_indices=torch.arange(3),
                extend_seq_lens=torch.tensor([1, 3, 2]),
                extend_seq_lens_cpu=[1, 3, 2],
                kv_loc_plan=None,
            )
            with (
                self.subTest(layout=layout, enabled=enabled, mode=mode),
                get_context().override_server_args(
                    dcp_size=3 if enabled else 1, dcp_kv_layout=layout
                ),
                get_parallel().override(dcp_rank=1 if enabled else 0),
                patch.object(
                    backend,
                    "_create_block_kv_indices",
                    wraps=backend._create_block_kv_indices,
                ) as create_table,
            ):
                backend.init_forward_metadata(batch)
                metadata = batch.decode_trtllm_mla_metadata
                torch.testing.assert_close(
                    metadata.seq_lens_k,
                    torch.tensor(expected_local, dtype=torch.int32),
                )
                if expected_global is None:
                    self.assertIsNone(metadata.global_seq_lens_k)
                else:
                    torch.testing.assert_close(
                        metadata.global_seq_lens_k,
                        torch.tensor(expected_global, dtype=torch.int32),
                    )
                torch.testing.assert_close(
                    # (plan, batch_size, max_blocks, req_pool_indices, seq_lens)
                    create_table.call_args.args[4],
                    torch.tensor(
                        expected_global
                        if expected_global is not None
                        else expected_local,
                        dtype=torch.int32,
                    ),
                    check_dtype=False,
                )
                if enabled and mode == ForwardMode.DECODE:
                    self.assertIs(metadata.global_seq_lens_k, batch.seq_lens)
                if enabled:
                    torch.testing.assert_close(
                        backend._fill_dcp_block_kv_indices.call_args.args[2],
                        metadata.seq_lens_k,
                    )

    def test_local_lengths_match_owner_enumeration(self):
        backend = object.__new__(TRTLLMMLABackend)
        for dcp_size in (2, 3, 8):
            for layout, page_size in (
                ("token", 64),
                ("page", 1),
                ("page", 2),
                ("page", 3),
                ("page", 64),
            ):
                backend.page_size = page_size
                backend.dcp_kv_layout = layout
                for rank in range(dcp_size):
                    expected = [0]
                    for position in range(3 * dcp_size * page_size + 1):
                        owner = position if layout == "token" else position // page_size
                        owned = owner % dcp_size == rank
                        expected.append(expected[-1] + owned)
                    with (
                        self.subTest(
                            layout=layout, size=dcp_size, page_size=page_size, rank=rank
                        ),
                        get_context().override_server_args(
                            dcp_size=dcp_size, dcp_kv_layout=layout
                        ),
                        get_parallel().override(dcp_rank=rank),
                    ):
                        torch.testing.assert_close(
                            backend._get_dcp_local_seq_lens(
                                torch.arange(len(expected))
                            ),
                            torch.tensor(expected, dtype=torch.int32),
                        )
                        self.assertEqual(
                            [
                                backend._get_dcp_local_max_seq_len(length)
                                for length in range(len(expected))
                            ],
                            [max(length, 1) for length in expected],
                        )

    def test_graph_metadata_refreshes_local_lengths_for_dynamic_batch(self):
        backend = object.__new__(TRTLLMMLABackend)
        backend.page_size = 2
        backend.max_context_len = 16
        backend._fill_dcp_block_kv_indices = MagicMock()
        backend.decode_cuda_graph_kv_indices = torch.full(
            (3, backend._calc_padded_blocks(16)), -1, dtype=torch.int32
        )

        for layout in ("page", "token"):
            backend.dcp_kv_layout = layout
            backend.decode_cuda_graph_metadata = {}
            for batch_size, seq_lens, page_lens, token_lens in (
                (3, [1, 4, 9], [0, 2, 3], [0, 1, 3]),
                (2, [5, 7], [2, 2], [2, 2]),
                # Replay the existing bucket with a padded length-one row.
                (3, [8, 1, 7], [2, 0, 2], [3, 0, 2]),
            ):
                with (
                    self.subTest(layout=layout, batch_size=batch_size, lens=seq_lens),
                    get_context().override_server_args(
                        dcp_size=3, dcp_kv_layout=layout
                    ),
                    get_parallel().override(dcp_rank=1),
                ):
                    lengths = torch.tensor(seq_lens)
                    if batch_size not in backend.decode_cuda_graph_metadata:
                        backend._init_cuda_graph_metadata(
                            batch_size, batch_size, ForwardMode.DECODE, lengths, "cpu"
                        )
                    metadata = backend.decode_cuda_graph_metadata[batch_size]
                    block_storage = metadata.block_kv_indices.data_ptr()
                    lens_storage = metadata.seq_lens_k.data_ptr()
                    global_lens_storage = metadata.global_seq_lens_k.data_ptr()
                    backend._apply_cuda_graph_metadata(
                        plan=None,
                        bs=batch_size,
                        req_pool_indices=torch.arange(batch_size),
                        seq_lens=lengths,
                        forward_mode=ForwardMode.DECODE,
                    )
                    torch.testing.assert_close(
                        metadata.global_seq_lens_k,
                        torch.tensor(seq_lens, dtype=torch.int32),
                    )
                    torch.testing.assert_close(
                        metadata.seq_lens_k,
                        torch.tensor(
                            page_lens if layout == "page" else token_lens,
                            dtype=torch.int32,
                        ),
                    )
                    self.assertIs(
                        backend._fill_dcp_block_kv_indices.call_args.args[2],
                        metadata.seq_lens_k,
                    )
                    self.assertEqual(
                        metadata.block_kv_indices.data_ptr(), block_storage
                    )
                    self.assertEqual(metadata.seq_lens_k.data_ptr(), lens_storage)
                    self.assertEqual(
                        metadata.global_seq_lens_k.data_ptr(), global_lens_storage
                    )
                    self.assertEqual(
                        metadata.max_seq_len_k, 6 if layout == "page" else 5
                    )


class TestCuteDslDcpWrapperContracts(CustomTestCase):
    def test_single_rank_keeps_the_base_decode_path(self):
        backend = object.__new__(CuteDslMLABackend)
        backend._decode_with_splits = None
        expected = torch.zeros((1, 1, 1, 512))
        for layout in ("token", "page"):
            backend.dcp_kv_layout = layout
            with (
                self.subTest(layout=layout),
                get_context().override_server_args(dcp_size=1, dcp_kv_layout=layout),
                patch.object(
                    backend, "_call_trtllm_batch_decode_mla", return_value=expected
                ),
            ):
                output = backend._run_decode_kernel(
                    query=torch.zeros((1, 1, 1, 576)),
                    kv_cache=torch.empty((1, 1, 64, 576)),
                    block_tables=torch.zeros((1, 1), dtype=torch.int32),
                    seq_lens=torch.ones(1, dtype=torch.int32),
                    max_seq_len=1,
                    layer=SimpleNamespace(),
                    cp_world=1,
                    cp_rank=0,
                )
                self.assertIs(output, expected)

    def test_forward_decode_derives_page_local_and_token_global_coordinates(self):
        backend = object.__new__(CuteDslMLABackend)
        backend.data_type = torch.bfloat16
        backend.q_data_type = torch.bfloat16
        backend.kv_cache_dim = 576
        backend.page_size = 2
        backend.workspace_buffer = object()
        backend.qk_nope_head_dim = 128
        backend.qk_rope_head_dim = 64
        backend.q_indptr_decode = torch.tensor([0, 1], dtype=torch.int32)
        backend.forward_decode_metadata = SimpleNamespace(
            batch_size=1,
            block_kv_indices=torch.zeros((1, 2), dtype=torch.int32),
            seq_lens_k=None,
            global_seq_lens_k=None,
            max_seq_len_k=3,
        )
        backend.token_to_kv_pool = SimpleNamespace(
            get_key_buffer=lambda _layer_id: torch.zeros((4, 576), dtype=torch.bfloat16)
        )
        backend.kv_lora_rank = 512
        layer = SimpleNamespace(
            layer_id=0, tp_q_head_num=1, v_head_dim=512, head_dim=576, scaling=1.0
        )
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            batch_size=1,
            seq_lens=torch.tensor([8], dtype=torch.int32),
        )
        with (
            patch.object(cute_module, "get_in_autotune_dummy_run", return_value=False),
            patch.object(cute_module, "fixup_zero_kv_rows"),
        ):
            calls = []

            def fake_decode(**kwargs):
                calls.append(kwargs)
                return torch.zeros((1, 1, 1, 512)), torch.zeros((1, 1, 1))

            for layout, causal_seq, cp_world, cp_rank, cached_length in (
                ("token", 8, 3, 1, None),
                ("page", 2, 1, 0, None),
                ("token", 8, 3, 1, 3),
                ("page", 2, 1, 0, 2),
            ):
                backend.dcp_kv_layout = layout
                metadata = backend.forward_decode_metadata
                metadata.seq_lens_k = (
                    torch.tensor([cached_length], dtype=torch.int32)
                    if cached_length is not None
                    else None
                )
                metadata.global_seq_lens_k = (
                    forward_batch.seq_lens if cached_length is not None else None
                )
                with self.subTest(layout=layout, cached_length=cached_length):
                    with (
                        patch.object(
                            backend,
                            "_get_dcp_local_seq_lens",
                            wraps=backend._get_dcp_local_seq_lens,
                        ) as local_lengths,
                        patch.object(
                            backend,
                            "_run_decode_kernel",
                            wraps=backend._run_decode_kernel,
                        ) as run_kernel,
                        get_context().override_server_args(
                            dcp_size=3,
                            dcp_kv_layout="page" if layout == "token" else "token",
                        ),
                        get_parallel().override(dcp_rank=1),
                        patch.object(
                            cute_module,
                            "flashinfer",
                            SimpleNamespace(
                                decode=SimpleNamespace(
                                    trtllm_batch_decode_with_kv_cache_mla=fake_decode
                                )
                            ),
                            create=True,
                        ),
                    ):
                        backend.forward_decode(
                            torch.zeros((1, 576), dtype=torch.bfloat16),
                            torch.empty(0),
                            torch.empty(0),
                            layer,
                            forward_batch,
                            save_kv_cache=False,
                        )
                    self.assertEqual(
                        local_lengths.call_count, int(cached_length is None)
                    )
                    self.assertEqual(run_kernel.call_args.kwargs["cp_world"], 3)
                    self.assertEqual(run_kernel.call_args.kwargs["cp_rank"], 1)
                    torch.testing.assert_close(
                        run_kernel.call_args.kwargs["causal_seqs"],
                        forward_batch.seq_lens,
                    )
                    kwargs = calls[-1]
                    self.assertTrue(kwargs["enable_dcp"])
                    self.assertTrue(kwargs["return_lse"])
                    torch.testing.assert_close(
                        kwargs["seq_lens"],
                        torch.tensor([2 if layout == "page" else 3], dtype=torch.int32),
                    )
                    if cached_length is not None:
                        self.assertEqual(
                            kwargs["seq_lens"].data_ptr(),
                            metadata.seq_lens_k.data_ptr(),
                        )
                    self.assertEqual(kwargs["cp_world"], cp_world)
                    self.assertEqual(kwargs["cp_rank"], cp_rank)
                    torch.testing.assert_close(
                        kwargs["causal_seqlens_kv_global"],
                        torch.tensor([causal_seq], dtype=torch.int32),
                    )
                    if layout == "page":
                        self.assertIs(
                            kwargs["causal_seqlens_kv_global"], kwargs["seq_lens"]
                        )
                    elif cached_length is not None:
                        self.assertEqual(
                            kwargs["causal_seqlens_kv_global"].data_ptr(),
                            metadata.global_seq_lens_k.data_ptr(),
                        )


if __name__ == "__main__":
    unittest.main()
