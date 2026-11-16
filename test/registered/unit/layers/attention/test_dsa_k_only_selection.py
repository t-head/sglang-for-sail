"""CPU-only checks of DSA K-only selection and graph-variant boundaries."""

import ast
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import ClassVar
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, stage="base-a", runner_config="cpu")

LAYERS = Path(__file__).resolve().parents[5] / "python/sglang/srt/layers"


def _load(nodes, source, namespace):
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    exec(compile(module, str(source), "exec"), namespace)
    return namespace


class _Lengths:
    def __init__(self, *values):
        self.values = values

    def numel(self):
        return len(self.values)

    def max(self):
        return SimpleNamespace(item=lambda: max(self.values))


def _batch(mode="decode", cpu=(2048,), device=None):
    return SimpleNamespace(
        forward_mode=SimpleNamespace(
            is_extend_without_speculative=lambda: mode == "prefill",
            is_decode_or_idle=lambda: mode in ("decode", "idle"),
        ),
        seq_lens_cpu=None if cpu is None else _Lengths(*cpu),
        seq_lens=None if device is None else _Lengths(*device),
    )


class TestDsaKOnlySelection(CustomTestCase):
    def _skip(
        self,
        batch,
        *,
        sail=True,
        hip=False,
        capture=False,
        variant=None,
        bf16=False,
    ):
        source = LAYERS / "attention/dsa/dsa_indexer.py"
        cls = next(
            node
            for node in ast.parse(source.read_text()).body
            if isinstance(node, ast.ClassDef) and node.name == "Indexer"
        )
        method = next(
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_should_skip_logits_computation"
        )
        namespace = _load(
            [method],
            source,
            {
                "_is_hip": hip,
                "is_sail": lambda: sail,
                "get_is_capture_mode": lambda: capture,
                "DSA_DENSE": "dense",
            },
        )
        capture_module = ModuleType(
            "sglang.srt.model_executor.runner_utils.capture_mode"
        )
        capture_module.get_capture_attention_variant = lambda: variant
        indexer = SimpleNamespace(
            index_topk=2048,
            bf16_indexer=bf16,
        )
        with patch.dict("sys.modules", {capture_module.__name__: capture_module}):
            return namespace[method.name](indexer, batch)

    def _variants(self, *, sail=True, hip=False, dsa=True, kpool=1):
        source = LAYERS / "attention/graph_variants.py"
        names = {"DsaGraphVariants", "create_attention_graph_variants"}
        nodes = [
            n
            for n in ast.parse(source.read_text()).body
            if getattr(n, "name", None) in names
        ]
        namespace = _load(
            nodes,
            source,
            {
                "__name__": __name__,
                "dataclass": dataclass,
                "ClassVar": ClassVar,
                "DSA_DENSE": "dense",
                "DSA_SPARSE": "sparse",
                "logger": Mock(),
            },
        )
        config = ModuleType("sglang.srt.configs.model_config")
        config.get_dsa_index_kpool = lambda unused: kpool
        config.get_dsa_index_topk = lambda unused: 2048
        config.is_deepseek_dsa = lambda unused: dsa
        utils = ModuleType("sglang.srt.utils")
        utils.is_hip = lambda: hip
        utils.is_sail = lambda: sail
        with patch.dict(
            "sys.modules", {config.__name__: config, utils.__name__: utils}
        ):
            return namespace["create_attention_graph_variants"](object())

    def test_bf16_indexer_keeps_the_full_path(self):
        for mode in ("prefill", "decode", "idle"):
            for capture in (False, True):
                with self.subTest(mode=mode, capture=capture):
                    self.assertFalse(
                        self._skip(
                            _batch(mode), bf16=True, capture=capture, variant="dense"
                        )
                    )

    def test_prefill_length_boundary_and_missing_lengths(self):
        for lengths, expected in (
            ((1, 2048), True),
            ((1, 2049), False),
            ((), False),
            (None, False),
        ):
            with self.subTest(lengths=lengths):
                self.assertEqual(self._skip(_batch("prefill", cpu=lengths)), expected)

    def test_eager_decode_platform_boundary(self):
        for sail, hip in ((True, False), (False, True), (False, False)):
            for mode in ("decode", "idle"):
                for length in (2048, 2049):
                    with self.subTest(sail=sail, hip=hip, mode=mode, length=length):
                        self.assertEqual(
                            self._skip(_batch(mode, cpu=(length,)), sail=sail, hip=hip),
                            (sail or hip) and length <= 2048,
                        )

    def test_decode_device_lengths_fallback_and_unknown_lengths(self):
        self.assertTrue(self._skip(_batch(cpu=None, device=(1024, 2048))))
        self.assertFalse(self._skip(_batch(cpu=(), device=(1024, 2049))))
        self.assertFalse(self._skip(_batch(cpu=None, device=None)))

    def test_capture_uses_variant_instead_of_capture_time_lengths(self):
        for variant in (None, "dense", "sparse"):
            for length in (1024, 4096):
                with self.subTest(variant=variant, length=length):
                    self.assertEqual(
                        self._skip(
                            _batch(cpu=(length,)), capture=True, variant=variant
                        ),
                        variant == "dense",
                    )

    def test_speculative_modes_keep_full_indexer(self):
        self.assertFalse(self._skip(_batch("target_verify", cpu=(128,))))
        self.assertFalse(self._skip(_batch("draft_extend", cpu=(128,))))

    def test_graph_factory_platform_boundary(self):
        for sail, hip in ((True, False), (False, True), (False, False)):
            with self.subTest(sail=sail, hip=hip):
                variants = self._variants(sail=sail, hip=hip)
                self.assertEqual(variants is not None, sail or hip)
                if variants is not None:
                    self.assertEqual(variants.capture_labels, ("dense", "sparse"))

    def test_graph_factory_preserves_kpool_and_non_dsa_exclusions(self):
        for sail, hip in ((True, False), (False, True)):
            self.assertIsNone(self._variants(sail=sail, hip=hip, kpool=2))
            self.assertIsNone(self._variants(sail=sail, hip=hip, dsa=False))

    def test_graph_selector_uses_maximum_length_and_sparse_fallback(self):
        variants = self._variants()
        self.assertEqual(variants.select(_batch(cpu=(1, 2048))), "dense")
        self.assertEqual(variants.select(_batch(cpu=(1, 2049))), "sparse")
        self.assertEqual(variants.select(_batch(cpu=None, device=(2048,))), "dense")
        self.assertEqual(variants.select(_batch(cpu=None)), "sparse")


if __name__ == "__main__":
    unittest.main()
