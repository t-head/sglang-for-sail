"""Structural regressions for runtime-only scalar specialization controls."""

import ast
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


CASES = (
    (
        "attention/dsv4/attn_metadata_kernels.py",
        "_expand_prefill_causally_kernel",
        ("bs", "num_tokens", "total_tokens"),
        ("BLOCK", "BS_P2"),
    ),
    ("memory/common.py", "get_last_loc_kernel", ("num_tokens",), ("BLOCK_SIZE",)),
    (
        "speculative/dspark/markov_sample.py",
        "_markov_sample_finish_kernel",
        ("BS",),
        ("PARTS", "WORLD", "BLOCK"),
    ),
    (
        "speculative/dspark/dspark_accept.py",
        "_finalize_accept_lens_kernel",
        ("bs",),
        ("BLOCK",),
    ),
    (
        "speculative/dspark/commit_swa.py",
        "_committed_swa_locations",
        ("N",),
        ("WIDTH", "MAP_SIZE", "BLOCK"),
    ),
)


class TestDsparkScalarSpecialization(CustomTestCase):
    def kernels(self):
        root = Path(__file__).resolve().parents[4] / "python/sglang/kernels/ops"
        for path, name, scalars, constants in CASES:
            tree = ast.parse((root / path).read_text())
            fn = next(
                n
                for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == name
            )
            yield fn, scalars, constants

    def test_runtime_scalars_have_both_specialization_guards(self):
        for fn, scalars, _ in self.kernels():
            with self.subTest(kernel=fn.name):
                decorator = next(
                    d
                    for d in fn.decorator_list
                    if isinstance(d, ast.Call)
                    and isinstance(d.func, ast.Attribute)
                    and d.func.attr == "jit"
                )
                options = {k.arg: ast.literal_eval(k.value) for k in decorator.keywords}
                self.assertEqual(options["do_not_specialize"], list(scalars))
                self.assertEqual(
                    options["do_not_specialize_on_alignment"], list(scalars)
                )
                arguments = {a.arg: a.annotation for a in fn.args.args}
                self.assertTrue(all(arguments[name] is None for name in scalars))

    def test_block_and_shape_constants_remain_compile_time(self):
        for fn, _, constants in self.kernels():
            with self.subTest(kernel=fn.name):
                arguments = {a.arg: a.annotation for a in fn.args.args}
                for name in constants:
                    annotation = arguments[name]
                    self.assertIsInstance(annotation, ast.Attribute)
                    self.assertEqual(annotation.attr, "constexpr")

    def test_runtime_scalars_do_not_define_static_iteration_shapes(self):
        for fn, scalars, _ in self.kernels():
            with self.subTest(kernel=fn.name):
                for call in ast.walk(fn):
                    if (
                        isinstance(call, ast.Call)
                        and isinstance(call.func, ast.Attribute)
                        and call.func.attr in ("arange", "static_range")
                    ):
                        dependencies = {
                            n.id
                            for a in call.args
                            for n in ast.walk(a)
                            if isinstance(n, ast.Name)
                        }
                        self.assertFalse(dependencies.intersection(scalars))


if __name__ == "__main__":
    unittest.main()
