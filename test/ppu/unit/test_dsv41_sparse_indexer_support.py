"""CPU-only sparse-indexer dispatch regression tests.

Execute the actual gate functions with mocked device and DeepGEMM boundaries;
kernel numerics and graph replay still require PPU hardware.
"""

import ast
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

ROOT = Path(__file__).resolve().parents[3]
LAYERS = ROOT / "python/sglang/srt/layers"


def load_function(path, name, namespace):
    nodes = ast.parse(path.read_text()).body
    function = next(
        n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            function,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


class TestDSV41SparseIndexerSupport(unittest.TestCase):
    def supports(self, *, sail, sm, enabled=True, blackwell=False, missing=None):
        deep_gemm = ModuleType("deep_gemm")
        for name in (
            "get_paged_sparse_mqa_logits_metadata",
            "fp8_fp4_paged_sparse_mqa_logits",
            "int8_paged_sparse_mqa_logits",
        ):
            setattr(deep_gemm, name, None if missing == name else Mock())
        gate = load_function(
            LAYERS / "deep_gemm_wrapper/configurer.py",
            "_supports_paged_sparse_mqa_logits",
            {
                "_is_sail": sail,
                "ENABLE_JIT_DEEPGEMM": enabled,
                "DEEPGEMM_BLACKWELL": blackwell,
                "get_device_sm": lambda: sm,
            },
        )
        with patch.dict(sys.modules, {"deep_gemm": deep_gemm}):
            return gate()

    def factory(
        self,
        *,
        sail,
        sm,
        supported=True,
        topk=16,
        block_size=8,
        cp=1,
        prefill=True,
        int8=False,
        index_n_heads=None,
        index_head_dim=None,
        token_to_kv_pool=None,
    ):
        package = "sglang.srt.layers.attention.dsv4.v41_indexer"
        config_name = "sglang.srt.layers.deep_gemm_wrapper.configurer"
        config = ModuleType(config_name)
        config.DEEPGEMM_PAGED_SPARSE_MQA_LOGITS = supported
        dense = ModuleType(package + ".dense_blocks")
        dense.DenseBlocksBackend = Mock()
        sparse = ModuleType(package + ".sparse_table")
        sparse.SparseTableBackend = Mock()
        runtime = ModuleType("sglang.srt.runtime_context")
        runtime.get_parallel = lambda: SimpleNamespace(attn_cp_size=cp)
        quant = ModuleType("sglang.srt.layers.attention.dsv4.indexer_quant")
        quant.use_dsv41_int8_indexer = lambda: int8
        factory = load_function(
            LAYERS / "attention/dsv4/v41_indexer/__init__.py",
            "make_candidate_indexer",
            {
                "__package__": package,
                "is_sail": lambda: sail,
                "get_device_sm": lambda: sm,
                "is_sm100_or_newer": lambda: sm >= 100,
                "_use_deep_gemm_prefill": lambda: prefill,
                "use_dsv41_int8_indexer": lambda: int8,
                "logger": Mock(),
            },
        )
        with patch.dict(
            sys.modules,
            {
                config_name: config,
                dense.__name__: dense,
                sparse.__name__: sparse,
                runtime.__name__: runtime,
                quant.__name__: quant,
            },
        ):
            result = factory(
                token_to_kv_pool=token_to_kv_pool,
                req_to_token=None,
                page_size=64,
                candidate_topk_blocks=topk,
                candidate_block_size=block_size,
                index_n_heads=index_n_heads,
                index_head_dim=index_head_dim,
            )
        return result, sparse.SparseTableBackend, dense.DenseBlocksBackend

    def test_sail_sm89_and_newer_select_sparse_engine(self):
        for sm in (89, 90, 100):
            with self.subTest(sm=sm):
                self.assertTrue(self.supports(sail=True, sm=sm))
                result, constructor, dense = self.factory(sail=True, sm=sm)
                self.assertEqual(
                    result, (constructor.return_value, constructor.return_value)
                )
                constructor.assert_called_once_with(
                    token_to_kv_pool=None,
                    page_size=64,
                    candidate_topk_blocks=16,
                    candidate_block_size=8,
                )

    def test_sail_sm80_int8_selects_sparse_engine(self):
        self.assertTrue(self.supports(sail=True, sm=80))
        # SM80 probes int8_paged_sparse_mqa_logits, not fp8_fp4.
        self.assertTrue(
            self.supports(
                sail=True,
                sm=80,
                missing="fp8_fp4_paged_sparse_mqa_logits",
            )
        )
        for api in (
            "get_paged_sparse_mqa_logits_metadata",
            "int8_paged_sparse_mqa_logits",
        ):
            self.assertFalse(self.supports(sail=True, sm=80, missing=api))
        result, constructor, dense = self.factory(sail=True, sm=80, int8=True)
        self.assertEqual(result, (constructor.return_value, constructor.return_value))
        constructor.assert_called_once_with(
            token_to_kv_pool=None,
            page_size=64,
            candidate_topk_blocks=16,
            candidate_block_size=8,
        )
        # Graceful fallback when the installed DeepGEMM lacks sparse INT8.
        result, constructor, dense = self.factory(
            sail=True, sm=80, int8=True, supported=False
        )
        self.assertEqual(result, (dense.return_value, dense.return_value))
        constructor.assert_not_called()

    def test_sail_sm80_int8_sparse_enforces_dispatch_contracts(self):
        # H=32, D=128, block_size=8, topk_blocks in (0, 2048] % 4 == 0,
        # 128-slot pages (128 * 132 = 16896, 512B aligned) -> sparse.
        pool_128 = SimpleNamespace(index_pools={1: SimpleNamespace(page_size=128)})
        for topk in (4, 16, 2048):
            with self.subTest(valid_topk=topk):
                result, sparse, dense = self.factory(
                    sail=True,
                    sm=80,
                    int8=True,
                    topk=topk,
                    block_size=8,
                    index_n_heads=32,
                    index_head_dim=128,
                    token_to_kv_pool=pool_128,
                )
                self.assertEqual(result, (sparse.return_value, sparse.return_value))
        # Out-of-contract configs must fall back to DenseBlocksBackend instead of
        # tripping SparseTableBackend assertions or C++ TORCH_CHECK failures.
        pool_64 = SimpleNamespace(index_pools={1: SimpleNamespace(page_size=64)})
        invalid_cases = [
            ("heads_64", dict(index_n_heads=64, index_head_dim=128)),
            ("head_dim_64", dict(index_n_heads=32, index_head_dim=64)),
            (
                "block_size_16",
                dict(block_size=16, index_n_heads=32, index_head_dim=128),
            ),
            ("block_size_4", dict(block_size=4, index_n_heads=32, index_head_dim=128)),
            ("topk_not_mult4", dict(topk=6, index_n_heads=32, index_head_dim=128)),
            ("topk_over_2048", dict(topk=2052, index_n_heads=32, index_head_dim=128)),
            ("topk_4096", dict(topk=4096, index_n_heads=32, index_head_dim=128)),
            (
                "unaligned_page_stride",
                dict(
                    index_n_heads=32,
                    index_head_dim=128,
                    token_to_kv_pool=pool_64,
                ),
            ),
        ]
        for label, overrides in invalid_cases:
            with self.subTest(case=label):
                kwargs = dict(
                    sail=True,
                    sm=80,
                    int8=True,
                    topk=16,
                    block_size=8,
                    token_to_kv_pool=pool_128,
                )
                kwargs.update(overrides)
                result, sparse, dense = self.factory(**kwargs)
                self.assertEqual(result, (dense.return_value, dense.return_value))
                sparse.assert_not_called()

    def test_sail_unsupported_sm_and_non_int8_sm80_keep_fallback(self):
        # DeepGEMM check_int8_sparse_mqa_metadata strictly requires capability 8.0.
        for sm in (75, 86, 88):
            with self.subTest(supports_sm=sm):
                self.assertFalse(self.supports(sail=True, sm=sm))
        for sm, int8 in (
            (75, True),
            (86, True),
            (88, True),
            (80, False),
            (86, False),
            (88, False),
        ):
            with self.subTest(sm=sm, int8=int8):
                result, constructor, dense = self.factory(sail=True, sm=sm, int8=int8)
                self.assertEqual(result, (dense.return_value, dense.return_value))
                constructor.assert_not_called()

    def test_deepgemm_disabled_or_missing_api_falls_back_to_dense(self):
        self.assertFalse(self.supports(sail=True, sm=89, enabled=False))
        self.assertFalse(self.supports(sail=True, sm=80, enabled=False))
        for api in (
            "get_paged_sparse_mqa_logits_metadata",
            "fp8_fp4_paged_sparse_mqa_logits",
        ):
            with self.subTest(api=api):
                self.assertFalse(self.supports(sail=True, sm=89, missing=api))
        result, constructor, dense = self.factory(sail=True, sm=89, supported=False)
        self.assertEqual(result, (dense.return_value, dense.return_value))
        constructor.assert_not_called()

    def test_unloadable_deepgemm_does_not_fail_the_probe(self):
        for sm in (80, 89):
            gate = load_function(
                LAYERS / "deep_gemm_wrapper/configurer.py",
                "_supports_paged_sparse_mqa_logits",
                {
                    "_is_sail": True,
                    "ENABLE_JIT_DEEPGEMM": True,
                    "get_device_sm": lambda sm=sm: sm,
                },
            )
            original_import = __import__
            for error in (ImportError, OSError, RuntimeError):
                with self.subTest(sm=sm, error=error):

                    def import_module(name, *args, **kwargs):
                        if name == "deep_gemm":
                            raise error("unavailable DeepGEMM build")
                        return original_import(name, *args, **kwargs)

                    with patch("builtins.__import__", side_effect=import_module):
                        self.assertFalse(gate())

    def test_nvidia_gate_is_preserved(self):
        self.assertFalse(self.supports(sail=False, sm=90))
        result, constructor, dense = self.factory(sail=False, sm=90)
        self.assertEqual(result, (dense.return_value, dense.return_value))
        constructor.assert_not_called()
        self.assertTrue(self.supports(sail=False, sm=100, blackwell=True))
        result, constructor, dense = self.factory(sail=False, sm=100)
        self.assertEqual(result, (constructor.return_value, constructor.return_value))

    def test_disabled_candidate_budget_skips_engine(self):
        result, constructor, dense = self.factory(sail=True, sm=89, topk=0)
        self.assertEqual(result, (dense.return_value, dense.return_value))
        constructor.assert_not_called()

    def test_cp_and_torch_prefill_keep_dense_prefill_sparse_decode(self):
        for sm, int8 in ((89, False), (80, True)):
            for cp, prefill in ((2, True), (1, False)):
                with self.subTest(sm=sm, int8=int8, cp=cp, prefill=prefill):
                    result, sparse, dense = self.factory(
                        sail=True, sm=sm, int8=int8, cp=cp, prefill=prefill
                    )
                    self.assertEqual(result, (dense.return_value, sparse.return_value))

    def test_sm89_int8_format_keeps_dense_for_both_phases(self):
        result, sparse, dense = self.factory(sail=True, sm=89, int8=True)
        self.assertEqual(result, (dense.return_value, dense.return_value))
        sparse.assert_not_called()

    def test_index_weights_dtype_follows_sail_sm(self):
        import functools

        path = ROOT / "python/sglang/kernels/ops/attention/dsv4/index_logits.py"
        fake_torch = SimpleNamespace(bfloat16="bf16", float32="fp32")
        for sm, forced, expected in (
            (80, None, "fp32"),
            (80, True, "bf16"),
            (80, False, "fp32"),
            (89, None, "bf16"),
            (89, False, "bf16"),
        ):
            with self.subTest(sm=sm, forced=forced):
                ns = {
                    "functools": functools,
                    "torch": fake_torch,
                    "is_sail": lambda: True,
                    "get_device_sm": lambda sm=sm: sm,
                    "_is_sm100": lambda: False,
                    "envs": SimpleNamespace(
                        SGLANG_OPT_DSV41_BF16_INDEX_LOGITS=SimpleNamespace(
                            get=lambda forced=forced: forced
                        )
                    ),
                }
                load_function(path, "_auto_bf16_index_logits", ns)
                load_function(path, "index_logits_dtype", ns)
                weights_dtype = load_function(path, "index_weights_dtype", ns)
                self.assertEqual(weights_dtype(), expected)

    def test_sparse_logits_int8_dispatches_and_slices_padding(self):
        path = ROOT / "python/sglang/kernels/ops/attention/dsv4/index_logits.py"
        fake_torch = SimpleNamespace(bfloat16="bf16", float32="fp32", int8="int8")

        class FakeTensor:
            def __init__(self, shape, dtype):
                self.shape = shape
                self.dtype = dtype
                self.is_contig = False
                self.sliced = None

            def to(self, dtype):
                t = FakeTensor(self.shape, dtype)
                return t

            def contiguous(self):
                t = FakeTensor(self.shape, self.dtype)
                t.is_contig = True
                return t

            def __getitem__(self, key):
                self.sliced = key
                return SimpleNamespace(source=self, key=key)

        deep_gemm = ModuleType("deep_gemm")
        padded = FakeTensor((2, 512), "bf16")
        deep_gemm.int8_paged_sparse_mqa_logits = Mock(return_value=padded)
        deep_gemm.fp8_fp4_paged_sparse_mqa_logits = Mock()
        fn = load_function(
            path,
            "sparse_logits",
            {"torch": fake_torch, "CANDIDATE_BLOCK_SIZE": 8},
        )
        q = FakeTensor((2, 1, 32, 128), "int8")
        k_cache = FakeTensor((4, 128, 1, 132), "uint8")
        weights = FakeTensor((2, 32), "fp32")
        schedule = FakeTensor((16,), "uint8")
        with patch.dict(sys.modules, {"deep_gemm": deep_gemm}):
            out = fn(q, None, k_cache, weights, schedule, topk_blocks=16)

        deep_gemm.fp8_fp4_paged_sparse_mqa_logits.assert_not_called()
        deep_gemm.int8_paged_sparse_mqa_logits.assert_called_once()
        args = deep_gemm.int8_paged_sparse_mqa_logits.call_args.args
        self.assertEqual(args[0].shape, (2, 1, 32, 128))
        self.assertTrue(args[0].is_contig)
        self.assertIs(args[1], k_cache)
        self.assertEqual(args[2].dtype, "bf16")
        self.assertTrue(args[2].is_contig)
        self.assertEqual(args[4:], (16, 8))
        self.assertIs(out.source, padded)
        self.assertEqual(out.key, (slice(None), slice(None, 128)))

    def test_flat_index_logits_tiles_supports_int8_tensor_q_on_sail(self):
        path = ROOT / "python/sglang/kernels/ops/attention/dsv4/index_logits.py"
        fake_torch = SimpleNamespace(bfloat16="bf16", float32="fp32", int8="int8")

        class Sliceable:
            def __init__(self, shape, dtype):
                self.shape = shape
                self.dtype = dtype

            def to(self, dtype):
                return Sliceable(self.shape, dtype)

            def __getitem__(self, sl):
                return Sliceable((sl.stop - sl.start, *self.shape[1:]), self.dtype)

            def __add__(self, other):
                return (self, other)

        sail_mqa = Mock(return_value=Sliceable((4, 64), "fp32"))
        ns = {
            "torch": fake_torch,
            "is_sail": lambda: True,
            "ceil_align": lambda x, a: (x + a - 1) // a * a,
            "FLAT_LOGITS_WIDTH_ALIGN": 64,
            "flat_index_logits_rows_per_tile": lambda rows, width, **_: rows,
            "index_weights_dtype": lambda: "fp32",
            "index_logits_dtype": lambda: "fp32",
            "sail_mqa_logits": sail_mqa,
            "envs": SimpleNamespace(
                SGLANG_SPARSE_INDEXER_MAX_LOGITS_MB=SimpleNamespace(get=lambda: 64)
            ),
        }
        tiles_fn = load_function(path, "flat_index_logits_tiles", ns)
        q = Sliceable((4, 32, 128), "int8")
        kv = (Sliceable((32, 128), "int8"), Sliceable((32,), "fp32"))
        weights = Sliceable((4, 32), "bf16")
        starts = Sliceable((4,), "int32")
        lengths = Sliceable((4,), "int32")
        tiles = list(
            tiles_fn(
                q=q,
                kv=kv,
                weights=weights,
                starts=starts,
                lengths=lengths,
                context_lengths=[32],
                budget_bytes=1 << 20,
            )
        )
        self.assertEqual(len(tiles), 1)
        args, kwargs = sail_mqa.call_args
        self.assertNotIsInstance(args[0], tuple)
        self.assertEqual(args[0].dtype, "int8")
        self.assertEqual(args[2].dtype, "fp32")
        self.assertEqual(kwargs, {"max_seqlen_k": 64})

    def test_full_topk_int8_enables_deepgemm_and_drops_request_indices(self):
        import functools

        init_path = LAYERS / "attention/dsv4/v41_indexer/__init__.py"
        quant = ModuleType("sglang.srt.layers.attention.dsv4.indexer_quant")
        quant.use_dsv41_int8_indexer = lambda: True
        ns = {
            "functools": functools,
            "is_sail": lambda: True,
            "is_sm100_or_newer": lambda: False,
            "has_dense_fp4_indexer": lambda: False,
            "has_dense_int8_indexer": lambda: True,
            "FullTopKIndexer": Mock(),
        }
        load_function(init_path, "use_deep_gemm_fp4_indexer", ns)
        load_function(init_path, "_use_deep_gemm_prefill", ns)
        make_full = load_function(init_path, "make_full_topk_indexer", ns)
        with patch.dict(sys.modules, {quant.__name__: quant}):
            make_full(token_to_kv_pool=None, req_to_token=None)
        ns["FullTopKIndexer"].assert_called_once_with(
            token_to_kv_pool=None,
            req_to_token=None,
            use_deep_gemm_prefill=True,
            use_deep_gemm_decode=True,
        )

        # FullTopKIndexer._deep_gemm_decode must pass indices=None for INT8 (q_sf is None).
        full_path = LAYERS / "attention/dsv4/v41_indexer/full_topk.py"
        paged_logits = Mock(return_value=MagicMock())
        topk_meta = Mock()
        data = SimpleNamespace(
            q_fp4=MagicMock(),
            q_sf=None,
            weights=MagicMock(),
            k_cache=MagicMock(),
            paged_logits_args=lambda metadata, rows=slice(None): (
                MagicMock(),
                metadata.compressed_seq_lens,
                metadata.page_table,
            ),
        )
        metadata = SimpleNamespace(
            compressed_seq_lens=MagicMock(),
            page_table=MagicMock(),
            deep_gemm_metadata=object(),
            max_compressed_seq_len=64,
            request_indices=MagicMock(),
        )
        inputs = SimpleNamespace(
            paged_metadata=metadata,
            out_page_indices=MagicMock(),
        )
        full_ns = {
            "get_deep_gemm_decode_data": lambda *_: data,
            "deep_gemm_fp4_paged_mqa_logits": paged_logits,
            "topk_transform_paged_from_metadata": topk_meta,
        }
        tree = ast.parse(full_path.read_text())
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "FullTopKIndexer"
        )
        method = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "_deep_gemm_decode"
        )
        mod = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                method,
            ],
            type_ignores=[],
        )
        ast.fix_missing_locations(mod)
        exec(compile(mod, str(full_path), "exec"), full_ns)
        full_ns["_deep_gemm_decode"](SimpleNamespace(token_to_kv_pool=None), inputs)
        self.assertIsNone(paged_logits.call_args.kwargs["indices"])
        topk_meta.assert_called_once()

    def test_sparse_table_captured_prefill_dispatches_int8(self):
        sparse_path = LAYERS / "attention/dsv4/v41_indexer/sparse_table.py"
        q_int8 = MagicMock(shape=(4, 32, 128))
        folded_w = MagicMock()
        int8_module = ModuleType("sglang.kernels.ops.attention.dsv4.int8_indexer")
        int8_module.quantize_query = Mock(return_value=(q_int8, folded_w))
        k_cache = MagicMock()
        get_cache = Mock(return_value=k_cache)
        prefill_logits = Mock(return_value=MagicMock())
        topk_sparse = Mock()
        ns = {
            "CANDIDATE_BLOCK_SIZE": 8,
            "quantize_index_q": Mock(),
            "get_index_k_cache": get_cache,
            "_prefill_table_logits": prefill_logits,
            "topk_transform_sparse": topk_sparse,
            "torch": MagicMock(),
        }
        tree = ast.parse(sparse_path.read_text())
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "SparseTableBackend"
        )
        method = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "consume_prefill_captured"
        )
        mod = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                method,
            ],
            type_ignores=[],
        )
        ast.fix_missing_locations(mod)
        exec(compile(mod, str(sparse_path), "exec"), ns)

        out_page = MagicMock()
        out_page.__getitem__.return_value.is_contiguous.return_value = True
        out_raw = MagicMock()
        out_raw.__getitem__.return_value.is_contiguous.return_value = True
        inputs = SimpleNamespace(
            indexer=SimpleNamespace(use_int8=True, index_topk=64),
            layer_id=5,
            q=MagicMock(),
            weights=MagicMock(),
            index_page_size=128,
            out_page_indices=out_page,
            out_raw_indices=out_raw,
        )
        published = SimpleNamespace(
            blocks=SimpleNamespace(shape=(4, 16)),
            phys_blocks=MagicMock(),
            valid_lens=MagicMock(),
        )
        backend = SimpleNamespace(token_to_kv_pool="pool")
        with patch.dict(sys.modules, {int8_module.__name__: int8_module}):
            ns["consume_prefill_captured"](backend, inputs, published)

        int8_module.quantize_query.assert_called_once_with(inputs.q, inputs.weights)
        ns["quantize_index_q"].assert_not_called()
        get_cache.assert_called_once_with(
            token_to_kv_pool="pool", layer_id=5, page_size=128, use_int8=True
        )
        prefill_logits.assert_called_once_with(
            published, q_int8, None, folded_w, k_cache
        )
        self.assertEqual(topk_sparse.call_count, 2)

    def test_int8_torch_prefill_accepts_the_gathered_k_tuple(self):
        """gather_int8_index_k returns (k, scales); the row-chunk budget must not
        read .shape from that tuple."""
        scoring_path = LAYERS / "attention/dsv4/v41_indexer/scoring.py"
        tensor = MagicMock()
        int8_module = ModuleType("sglang.kernels.ops.attention.dsv4.int8_indexer")
        int8_module.int8_index_logits = Mock(return_value=tensor)
        int8_module.int8_logits_rows_per_chunk = Mock(return_value=2)
        environ = ModuleType("sglang.srt.environ")
        environ.envs = SimpleNamespace(
            SGLANG_SPARSE_INDEXER_MAX_LOGITS_MB=SimpleNamespace(get=lambda: 1)
        )
        tok = MagicMock()
        tok.numel.return_value = 3

        class Broadcastable:
            def __getitem__(self, key):
                return self

            def __ge__(self, other):
                return self

        request = SimpleNamespace(
            lc=5, k=2, columns=Broadcastable(), tok=tok, lens=tensor
        )
        namespace = {
            "_TORCH_SCORE_BUDGET_BYTES": 1 << 30,
            "ChunkScores": SimpleNamespace,
            "torch": SimpleNamespace(inf=float("inf")),
        }
        score_chunks = load_function(scoring_path, "_score_chunks", namespace)
        with patch.dict(
            sys.modules,
            {int8_module.__name__: int8_module, environ.__name__: environ},
        ):
            chunks = list(
                score_chunks(
                    indexer=SimpleNamespace(use_int8=True),
                    q=tensor,
                    weights=tensor,
                    index_k=(tensor, tensor),
                    request=request,
                )
            )
        int8_module.int8_logits_rows_per_chunk.assert_called_once_with(5, 1 << 20)
        self.assertEqual(len(chunks), 2)

    def test_sm90_decode_respects_platform_and_cache_format(self):
        """SAIL SM90 must use its INT8/DeepGEMM scorer, not Hopper's FP4 scorer."""
        scoring_path = LAYERS / "attention/dsv4/v41_indexer/scoring.py"
        for sail, int8, selected in (
            (True, True, "int8"),
            (True, False, "deep_gemm_fp4"),
            (False, False, "hopper_fp4"),
        ):
            with self.subTest(sail=sail, int8=int8):
                tensor = MagicMock(shape=(1, 64 * (132 if int8 else 68)))
                metadata = MagicMock(max_compressed_seq_len=8, compressed_page_size=64)
                metadata.row_chunks.return_value = [(slice(0, 1), Mock())]
                inputs = MagicMock(
                    compress_ratio=4,
                    req_rows=tensor,
                    positions=tensor,
                    paged_metadata=metadata,
                )
                inputs.indexer.use_int8 = int8
                inputs.indexer.queries.return_value = tensor
                inputs.indexer.head_weights.return_value = tensor
                pool = Mock()
                pool.get_index_k_with_scale_buffer.return_value = tensor
                kernels = {
                    name: MagicMock(name=name)
                    for name in ("int8", "deep_gemm_fp4", "hopper_fp4")
                }
                int8_module = ModuleType(
                    "sglang.kernels.ops.attention.dsv4.int8_indexer"
                )
                fp4_module = ModuleType(
                    "sglang.kernels.ops.attention.dsv4.index_logits"
                )
                int8_module.int8_paged_index_logits = kernels["int8"]
                fp4_module.deep_gemm_fp4_paged_mqa_logits = kernels["deep_gemm_fp4"]
                columns = MagicMock()
                columns.__getitem__.return_value.__ge__.return_value = tensor
                namespace = {
                    "torch": MagicMock(
                        arange=Mock(return_value=columns), inf=float("inf")
                    ),
                    "get_platform": lambda: SimpleNamespace(is_sm90=True),
                    "is_sail": lambda: sail,
                    "DecodeScores": SimpleNamespace,
                    "PagedDecodeScores": SimpleNamespace,
                    "plan_topk_v2": Mock(),
                    "fp4_index_logits_paged": kernels["hopper_fp4"],
                    "fp4_index_logits_decode": Mock(),
                    "quantize_index_q": Mock(return_value=(tensor, tensor)),
                    "get_index_k_cache": Mock(return_value=tensor),
                }
                decode = load_function(scoring_path, "decode_score_chunks", namespace)
                with patch.dict(
                    sys.modules,
                    {
                        int8_module.__name__: int8_module,
                        fp4_module.__name__: fp4_module,
                    },
                ):
                    outputs = list(
                        decode(
                            inputs=inputs,
                            token_to_kv_pool=pool,
                            req_to_token=tensor,
                        )
                    )

                kernels[selected].assert_called_once()
                for name, kernel in kernels.items():
                    if name != selected:
                        kernel.assert_not_called()
                namespace["fp4_index_logits_decode"].assert_not_called()
                self.assertEqual(len(outputs), 1)
                self.assertEqual((outputs[0].bs, outputs[0].lmax), (1, 8))
                scores = kernels[selected].return_value
                if sail:
                    scores = scores.__getitem__.return_value
                self.assertIs(outputs[0].scores, scores)

    def test_c2_target_verify_position0_and_position1_visible_slots(self):
        """Target-verify rows at position=0 and position=1 for C2 must keep
        FlashMLA lengths clamped to 1 while exposing 0 and 1 visible slots to
        both full top-k and sparse-table decode."""
        backend_path = LAYERS / "attention/deepseek_v4_backend.py"
        indexer_path = LAYERS / "attention/dsv4/indexer.py"
        full_path = LAYERS / "attention/dsv4/v41_indexer/full_topk.py"
        sparse_path = LAYERS / "attention/dsv4/v41_indexer/sparse_table.py"

        class FakeIntTensor:
            def __init__(self, data, dtype="int32"):
                self.data = data
                self.dtype = dtype
                self.ndim = 2 if (data and isinstance(data[0], list)) else 1
                self.shape = (
                    (len(data), len(data[0])) if self.ndim == 2 else (len(data),)
                )

            def dim(self):
                return self.ndim

            def numel(self):
                return (
                    self.shape[0] * self.shape[1] if self.ndim == 2 else len(self.data)
                )

            def to(self, dtype):
                return FakeIntTensor(
                    [row.copy() for row in self.data]
                    if self.ndim == 2
                    else self.data.copy(),
                    dtype=dtype,
                )

            def contiguous(self):
                return self

            def reshape(self, *_):
                return self

            def record_stream(self, _stream):
                pass

            def __getitem__(self, key):
                return FakeIntTensor(self.data[key], dtype=self.dtype)

            def __floordiv__(self, other):
                return FakeIntTensor([v // other for v in self.data], dtype=self.dtype)

            def __mod__(self, other):
                return FakeIntTensor([v % other for v in self.data], dtype=self.dtype)

            def __eq__(self, other):
                return [v == other for v in self.data]

            def __lt__(self, other):
                return [v < other for v in self.data]

            def clamp_min(self, minimum):
                return FakeIntTensor(
                    [max(v, minimum) for v in self.data], dtype=self.dtype
                )

            def masked_fill(self, mask, value):
                return FakeIntTensor(
                    [value if m else v for m, v in zip(mask, self.data)],
                    dtype=self.dtype,
                )

        def fake_where(cond, x, y):
            return FakeIntTensor(
                [xv if c else y for c, xv in zip(cond, x.data)], dtype=x.dtype
            )

        fake_torch = SimpleNamespace(
            int32="int32",
            int64="int64",
            int8="int8",
            float8_e4m3fn="fp8",
            float32="fp32",
            bfloat16="bf16",
            Size=tuple,
            where=fake_where,
            cuda=SimpleNamespace(
                current_stream=Mock(return_value=Mock()),
                stream=Mock(return_value=MagicMock()),
                Event=Mock(return_value=Mock()),
            ),
        )

        # Load DSV4AttnMetadata.init_compression_metadata and DeepseekV4AttnBackend.init_forward_metadata_indexer.
        backend_tree = ast.parse(backend_path.read_text())

        def load_method(tree, path, cls_name, method_name, ns):
            cls = next(
                n
                for n in tree.body
                if isinstance(n, ast.ClassDef) and n.name == cls_name
            )
            method = next(
                n
                for n in cls.body
                if isinstance(n, ast.FunctionDef) and n.name == method_name
            )
            mod = ast.Module(
                body=[
                    ast.ImportFrom(
                        module="__future__",
                        names=[ast.alias(name="annotations")],
                        level=0,
                    ),
                    method,
                ],
                type_ignores=[],
            )
            exec(compile(ast.fix_missing_locations(mod), str(path), "exec"), ns)
            return ns[method_name]

        backend_ns = {
            "torch": fake_torch,
            "_pad_last_dim": lambda x: x,
            "is_sail": lambda: True,
            "use_dsv41_int8_indexer": lambda: True,
            "expand_index_page_table": lambda table, **_: table,
            "get_platform": lambda: SimpleNamespace(is_sm120=False),
            "_is_xpu": False,
            "PagedIndexerMetadata": lambda **kwargs: SimpleNamespace(**kwargs),
        }
        load_function(backend_path, "_low_ratio_compression_metadata", backend_ns)
        init_comp = load_method(
            backend_tree,
            backend_path,
            "DSV4AttnMetadata",
            "init_compression_metadata",
            backend_ns,
        )
        init_idx = load_method(
            backend_tree,
            backend_path,
            "DeepseekV4AttnBackend",
            "init_forward_metadata_indexer",
            backend_ns,
        )

        # Target-verify batch for one request: position=0 (seq_len=1) and position=1 (seq_len=2).
        page_table = FakeIntTensor([[3], [3]])
        core = SimpleNamespace(
            page_table=page_table,
            seq_lens_casual=FakeIntTensor([1, 2]),
            raw_out_loc=FakeIntTensor([10, 11], dtype="int64"),
            swa_page_indices=FakeIntTensor([[0], [0]]),
            has_c4=False,
            has_c128=False,
            low_ratios=(1, 2),
        )
        init_comp(core)
        self.assertEqual(core.c2_out_loc.data, [-1, 5])
        self.assertEqual(core.c2_topk_lengths_raw.data, [0, 1])
        self.assertEqual(core.c2_topk_lengths_clamp1.data, [1, 1])

        backend = SimpleNamespace(
            page_size=256,
            index_head_dim=128,
            index_n_heads=32,
            enable_deepseek_v4_fp4_indexer=False,
            token_to_kv_pool=SimpleNamespace(get_index_k_page_size=lambda _r: 128),
            dsa_topk_backend=SimpleNamespace(should_use_topk_v2=lambda: False),
        )
        metadata = init_idx(
            backend,
            core,
            compress_ratio=2,
            rows_per_request=2,
            request_indices=FakeIntTensor([0, 0]),
        )
        self.assertEqual(metadata.compressed_seq_lens.data, [0, 1])
        metadata.deep_gemm_metadata = object()
        metadata.max_compressed_seq_len = 128

        # Execute naive paged top-k semantics via real topk_transform_paged_from_metadata.
        def naive_paged_topk(logits, seq_lens, table, out, page_size, raw=None):
            topk = out.shape[1]
            for r, length in enumerate(seq_lens.data):
                out.data[r] = [
                    table.data[r][i // page_size] * page_size + (i % page_size)
                    if i < length
                    else -1
                    for i in range(topk)
                ]

        idx_ns = {
            "torch": fake_torch,
            "topk_transform_paged": naive_paged_topk,
            "topk_transform_paged_v2": Mock(),
            "topk_transform_paged_bf16": Mock(),
        }
        topk_from_meta = load_function(
            indexer_path, "topk_transform_paged_from_metadata", idx_ns
        )

        # 1) FullTopKIndexer._deep_gemm_decode
        full_tree = ast.parse(full_path.read_text())
        data = SimpleNamespace(
            q_fp4=SimpleNamespace(shape=(2, 1, 32, 128), dtype="int8"),
            q_sf=None,
            weights=MagicMock(),
            k_cache=MagicMock(),
            paged_logits_args=lambda md, rows=slice(None): (
                MagicMock(),
                md.compressed_seq_lens[rows],
                md.page_table[rows],
            ),
        )
        fake_logits = SimpleNamespace(dtype="fp32", record_stream=Mock())
        full_ns = {
            "get_deep_gemm_decode_data": lambda *_: data,
            "deep_gemm_fp4_paged_mqa_logits": Mock(return_value=fake_logits),
            "topk_transform_paged_from_metadata": topk_from_meta,
        }
        full_decode = load_method(
            full_tree, full_path, "FullTopKIndexer", "_deep_gemm_decode", full_ns
        )
        full_out = FakeIntTensor([[-99] * 4, [-99] * 4])
        full_decode(
            SimpleNamespace(token_to_kv_pool=None),
            SimpleNamespace(paged_metadata=metadata, out_page_indices=full_out),
        )
        self.assertEqual(full_out.data[0], [-1, -1, -1, -1])
        self.assertEqual(full_out.data[1], [384, -1, -1, -1])

        # 2) SparseTableBackend.publish_decode
        sparse_tree = ast.parse(sparse_path.read_text())
        cand_lens = Mock(
            side_effect=lambda lens, _k: (
                FakeIntTensor([(v + 7) // 8 for v in lens.data]),
                FakeIntTensor(lens.data.copy()),
            )
        )
        amax_blocks = Mock(return_value=MagicMock(device="cuda"))
        sort_blocks = Mock(return_value=MagicMock())
        build_sched = Mock(return_value=MagicMock())
        sparse_ns = {
            "torch": fake_torch,
            "get_deep_gemm_decode_data": lambda *_: data,
            "deep_gemm_fp4_paged_mqa_logits": Mock(return_value=fake_logits),
            "topk_transform_paged": naive_paged_topk,
            "topk_transform_paged_bf16": Mock(),
            "candidate_row_lens": cand_lens,
            "amax_topk_blocks": amax_blocks,
            "sort_candidate_blocks": sort_blocks,
            "build_sparse_indexer_schedule": build_sched,
            "_SparseTable": SimpleNamespace,
        }
        publish_decode = load_method(
            sparse_tree,
            sparse_path,
            "SparseTableBackend",
            "publish_decode",
            sparse_ns,
        )
        sparse_out = FakeIntTensor([[-99] * 4, [-99] * 4])
        publisher = SimpleNamespace(
            token_to_kv_pool=None,
            alt_stream=Mock(),
            topk_blocks=16,
            _get_request_ids=Mock(return_value=FakeIntTensor([0, 0])),
        )
        table = publish_decode(
            publisher,
            SimpleNamespace(paged_metadata=metadata, out_page_indices=sparse_out),
        )
        self.assertEqual(sparse_out.data[0], [-1, -1, -1, -1])
        self.assertEqual(sparse_out.data[1], [384, -1, -1, -1])
        self.assertEqual(cand_lens.call_args.args[0].data, [0, 1])
        self.assertEqual(table.valid_lens.data, [0, 1])
        self.assertEqual(amax_blocks.call_args.args[1].data, [0, 1])
        self.assertEqual(sort_blocks.call_args.args[1].data, [0, 1])
        self.assertEqual(build_sched.call_args.args[1].data, [0, 1])

        # Verify graph-copy and CP-reindex metadata tables include c2_topk_lengths_raw.
        backend_src = backend_path.read_text()
        self.assertGreaterEqual(backend_src.count('"c2_topk_lengths_raw"'), 3)

    def test_sparse_consumer_requests_bf16_weights_directly(self):
        """Sparse consumers must pass weights_dtype=torch.bfloat16 through
        get_deep_gemm_{decode,prefill}_data -> _index_q_and_weights so fused
        INT8 Q quantization emits bf16 directly without an extra fp32->bf16 cast."""
        scoring_path = LAYERS / "attention/dsv4/v41_indexer/scoring.py"
        sparse_path = LAYERS / "attention/dsv4/v41_indexer/sparse_table.py"
        fake_torch = SimpleNamespace(
            bfloat16="bf16",
            float32="fp32",
            cuda=SimpleNamespace(current_stream=Mock(return_value=Mock())),
        )

        # 1) _index_q_and_weights forwards explicit weights_dtype=bf16 and skips index_weights_dtype().
        fused_q = Mock(return_value=("q_int8", "w_bf16"))
        default_dtype_probe = Mock(return_value="fp32")
        int8_module = ModuleType("sglang.kernels.ops.attention.dsv4.int8_indexer")
        int8_module.index_q_rope_int8_quant_weights = fused_q
        int8_module.quantize_query = Mock()
        logits_module = ModuleType("sglang.kernels.ops.attention.dsv4.index_logits")
        logits_module.index_weights_dtype = default_dtype_probe
        scoring_ns = {"torch": fake_torch, "is_sail": lambda: True}
        q_and_w = load_function(scoring_path, "_index_q_and_weights", scoring_ns)

        q_raw = MagicMock(dtype="bf16")
        q_raw.view.return_value = q_raw
        indexer = SimpleNamespace(
            use_int8=True,
            n_local_heads=32,
            index_head_dim=128,
            rope_head_dim=64,
            head_weight_scale=0.125,
            wq_b=Mock(return_value=(q_raw, None)),
            head_weights_raw=Mock(return_value="w_raw"),
        )
        x = SimpleNamespace(shape=(4, 512))
        q_lora = SimpleNamespace(is_cuda=True)

        with patch.dict(
            sys.modules,
            {
                int8_module.__name__: int8_module,
                logits_module.__name__: logits_module,
            },
        ):
            out = q_and_w(
                indexer=indexer,
                x=x,
                q_lora=q_lora,
                freqs_cis="freqs",
                positions="pos",
                weights_dtype="bf16",
            )
            self.assertEqual(out, ("q_int8", None, "w_bf16"))
            self.assertEqual(fused_q.call_args.kwargs["weights_dtype"], "bf16")
            default_dtype_probe.assert_not_called()

            # Default weights_dtype=None still resolves through index_weights_dtype().
            q_and_w(
                indexer=indexer,
                x=x,
                q_lora=q_lora,
                freqs_cis="freqs",
                positions="pos",
            )
            default_dtype_probe.assert_called_once()
            self.assertEqual(fused_q.call_args.kwargs["weights_dtype"], "fp32")

        # 2) SparseTableBackend.consume_decode / consume_prefill request weights_dtype=bf16.
        sparse_tree = ast.parse(sparse_path.read_text())

        def load_sparse_method(name, ns):
            cls = next(
                n
                for n in sparse_tree.body
                if isinstance(n, ast.ClassDef) and n.name == "SparseTableBackend"
            )
            method = next(
                n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name
            )
            mod = ast.Module(
                body=[
                    ast.ImportFrom(
                        module="__future__",
                        names=[ast.alias(name="annotations")],
                        level=0,
                    ),
                    method,
                ],
                type_ignores=[],
            )
            exec(compile(ast.fix_missing_locations(mod), str(sparse_path), "exec"), ns)
            return ns[name]

        get_decode = Mock(
            return_value=SimpleNamespace(
                q_fp4="q",
                q_sf=None,
                k_cache="k",
                weights=MagicMock(),
            )
        )
        get_prefill = Mock(
            return_value=SimpleNamespace(num_rows=4, write_page_indices=Mock())
        )
        sparse_ns = {
            "torch": fake_torch,
            "get_deep_gemm_decode_data": get_decode,
            "get_deep_gemm_prefill_data": get_prefill,
            "get_index_k_cache": Mock(return_value="k_cache"),
            "select_prefill_table": Mock(),
            "sparse_logits": Mock(return_value="logits"),
            "topk_transform_sparse": Mock(),
        }
        consume_decode = load_sparse_method("consume_decode", sparse_ns)
        consume_prefill = load_sparse_method("consume_prefill", sparse_ns)

        backend = SimpleNamespace(
            token_to_kv_pool=SimpleNamespace(get_index_k_page_size=lambda _r: 128)
        )
        published = SimpleNamespace(
            ready="ev",
            schedule="sched",
            blocks=SimpleNamespace(shape=(4, 16)),
            valid_lens="vl",
            phys_blocks="pb",
        )
        consume_decode(
            backend,
            SimpleNamespace(out_page_indices="out_pages"),
            published,
        )
        get_decode.assert_called_once()
        self.assertEqual(get_decode.call_args.kwargs, {"weights_dtype": "bf16"})

        prefill_inputs = SimpleNamespace(
            reset_outputs=Mock(),
            layer_id=3,
            compress_ratio=1,
            indexer=SimpleNamespace(use_int8=True),
            out_raw_indices=MagicMock(),
        )
        consume_prefill(backend, prefill_inputs, published)
        get_prefill.assert_called_once()
        self.assertEqual(get_prefill.call_args.kwargs, {"weights_dtype": "bf16"})


if __name__ == "__main__":
    unittest.main()
