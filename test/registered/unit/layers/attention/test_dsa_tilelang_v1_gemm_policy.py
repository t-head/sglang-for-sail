"""CPU contract checks for the v1 sparse-MLA specialization's GEMM policy.

The real factory body is executed with a no-op prim_func decorator. No TIR is
lowered and no TileLang/device compiler is imported by these tests.
"""

import ast
import copy
import inspect
import unittest
from collections import Counter
from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Policy(IntEnum):
    FullRow = 1
    FullCol = 2


class _TraceSink:
    """Records the live statement sequence a specialization emits."""

    def __init__(self):
        self.events = []

    def add(self, *event):
        self.events.append(event)
        return None


class _TraceBuf:
    def __init__(self, name, sink):
        self.name = name
        self.sink = sink
        self.dtype = "bfloat16"

    def __getitem__(self, key):
        return self

    def __setitem__(self, key, value):
        self.sink.add("store", self.name)

    def __add__(self, other):
        return self

    __radd__ = __add__
    __sub__ = __add__
    __mul__ = __add__
    __rmul__ = __add__
    __truediv__ = __add__
    __neg__ = lambda self: self

    def __ge__(self, other):
        return _TraceCond()

    __le__ = __ge__
    __gt__ = __ge__
    __lt__ = __ge__
    __eq__ = lambda self, other: _TraceCond()
    __hash__ = None


class _TraceCond:
    def __bool__(self):
        raise AssertionError("a TIR condition must not be evaluated on the host")


class _TraceIndex:
    def __init__(self, name):
        self.name = name

    def __add__(self, other):
        return self

    __radd__ = __add__
    __sub__ = __add__
    __mul__ = __add__
    __rmul__ = __add__
    __floordiv__ = __add__
    __mod__ = __add__

    def __ge__(self, other):
        return _TraceCond()

    __le__ = __ge__
    __gt__ = __ge__
    __lt__ = __ge__
    __eq__ = lambda self, other: _TraceCond()
    __hash__ = None


class _TraceIter:
    """Yields exactly one representative iteration."""

    def __init__(self, values):
        self.values = values

    def __enter__(self):
        return self.values[0] if len(self.values) == 1 else self.values

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        if len(self.values) == 1:
            yield self.values[0]
        else:
            yield self.values


def _label_allocations(factory_ast):
    """Attach each buffer's variable name to its allocation call."""
    main = next(
        node
        for node in factory_ast.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    for node in ast.walk(main):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        value = node.value
        if not isinstance(target, ast.Name):
            continue
        if not (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr in ("alloc_shared", "alloc_fragment")
        ):
            continue
        value.keywords.append(
            ast.keyword(arg="__name__", value=ast.Constant(target.id))
        )
    return ast.fix_missing_locations(factory_ast)


def _trace_namespace(sink):
    def alloc(kind):
        def make(shape, dtype, *, __name__="anonymous"):
            sink.add("alloc", __name__, kind)
            return _TraceBuf(__name__, sink)

        return make

    class _T:
        GemmWarpPolicy = _Policy
        symbolic = staticmethod(lambda name: _TraceIndex(name))
        Tensor = staticmethod(lambda *args: object())
        prim_func = staticmethod(lambda fn: fn)
        alloc_shared = staticmethod(alloc("shared"))
        alloc_fragment = staticmethod(alloc("fragment"))
        fill = staticmethod(lambda dst, src: sink.add("fill", dst.name))
        copy = staticmethod(lambda src, dst: sink.add("copy", src.name, dst.name))
        reduce_max = staticmethod(
            lambda src, dst, **kw: sink.add("reduce", src.name, dst.name)
        )
        reduce_sum = staticmethod(
            lambda src, dst, **kw: sink.add("reduce", src.name, dst.name)
        )
        if_then_else = staticmethod(lambda cond, a, b: _TraceBuf("selection", sink))
        exp2 = staticmethod(lambda x: _TraceBuf("exp2", sink))
        log2 = staticmethod(lambda x: _TraceBuf("log2", sink))
        infinity = staticmethod(lambda x: _TraceBuf("inf", sink))

        @staticmethod
        def gemm(a, b, acc, **kw):
            sink.add("gemm", a.name, b.name, acc.name, str(kw.get("policy")))

        @staticmethod
        def Kernel(*dims, **kw):
            return _TraceIter([_TraceIndex("axis%d" % i) for i in range(len(dims))])

        @staticmethod
        def Pipelined(count, **kw):
            # The extent is part of the emitted statement: the block loop runs
            # `count` times, so recording it lets a test tell a 32-block
            # geometry from a 33-block one instead of discarding the argument.
            sink.add("pipeline", count)
            return _TraceIter([_TraceIndex("pipeline")])

        @staticmethod
        def Parallel(*dims):
            return _TraceIter([_TraceIndex("lane%d" % i) for i in range(len(dims))])

    return _T



class TestDSATileLangV1GemmPolicy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = (
            Path(__file__).resolve().parents[5]
            / "python/sglang/kernels/ops/attention/dsa/tilelang_kernel.py"
        )
        cls.module_ast = ast.parse(source.read_text())
        cls.factory_ast = next(
            node
            for node in cls.module_ast.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "sparse_attention_fwd_kernel_v1"
        )

    def make_specialization(self, *, ppu, heads, dim, tail_dim, topk=128, kv_group=1):
        factory = copy.deepcopy(self.factory_ast)
        factory.decorator_list = []
        module = ast.fix_missing_locations(ast.Module(body=[factory], type_ignores=[]))
        namespace = {
            "is_sail": lambda: ppu,
            "tilelang": SimpleNamespace(
                math=SimpleNamespace(
                    next_power_of_2=lambda n: 1 << (n - 1).bit_length()
                ),
                cdiv=lambda a, b: (a + b - 1) // b,
            ),
            "T": SimpleNamespace(
                symbolic=lambda _: 1,
                Tensor=lambda *_: object(),
                prim_func=lambda fn: fn,
                GemmWarpPolicy=_Policy,
            ),
        }
        exec(compile(module, str(__file__), "exec"), namespace)
        return namespace[factory.name](
            heads, dim, tail_dim, topk, kv_group=kv_group, num_stages=1)

    def test_ppu_uses_full_row_for_main_and_zero_tail_geometries(self):
        for heads in (16, 32, 64, 128):
            for dim, tail_dim in ((256, 0), (512, 64)):
                with self.subTest(heads=heads, dim=dim, tail_dim=tail_dim):
                    kernel = self.make_specialization(
                        ppu=True, heads=heads, dim=dim, tail_dim=tail_dim
                    )
                    self.assertIs(
                        inspect.getclosurevars(kernel).nonlocals["gemm_policy"],
                        _Policy.FullRow,
                    )

    def test_non_ppu_specializations_keep_full_col(self):
        for heads in (16, 32, 64, 128):
            for dim, tail_dim in ((256, 0), (512, 64)):
                with self.subTest(heads=heads, dim=dim, tail_dim=tail_dim):
                    kernel = self.make_specialization(
                        ppu=False, heads=heads, dim=dim, tail_dim=tail_dim
                    )
                    self.assertIs(
                        inspect.getclosurevars(kernel).nonlocals["gemm_policy"],
                        _Policy.FullCol,
                    )

    def main_gemms(self):
        main = next(
            node
            for node in self.factory_ast.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        gemms = [
            node
            for node in ast.walk(main)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "T"
            and node.func.attr == "gemm"
        ]
        self.assertEqual(len(gemms), 3)
        self.assertEqual(
            Counter(node.args[2].id for node in gemms), {"acc_s": 2, "acc_o": 1}
        )
        return gemms

    def test_qk_gemms_share_the_specialization_policy(self):
        for gemm in self.main_gemms():
            keywords = {kw.arg: kw.value for kw in gemm.keywords}
            if gemm.args[2].id != "acc_s":
                continue
            self.assertIsInstance(keywords["policy"], ast.Name)
            self.assertEqual(keywords["policy"].id, "gemm_policy")
            self.assertIsInstance(keywords["transpose_B"], ast.Constant)
            self.assertIs(keywords["transpose_B"].value, True)

    def test_pv_gemm_has_its_own_policy_argument(self):
        # PV is specialised through pv_policy so the padded PPU geometry can
        # diverge from the QK policy; everywhere else pv_policy aliases it.
        pv = [g for g in self.main_gemms() if g.args[2].id == "acc_o"]
        self.assertEqual(len(pv), 1)
        keywords = {kw.arg: kw.value for kw in pv[0].keywords}
        self.assertIsInstance(keywords["policy"], ast.Name)
        self.assertEqual(keywords["policy"].id, "pv_policy")

    def test_pv_policy_aliases_qk_outside_the_gate(self):
        cases = [(False, 16), (False, 32), (False, 64), (False, 128), (True, 128)]
        for ppu, heads in cases:
            for dim, tail_dim in ((256, 0), (512, 64)):
                with self.subTest(ppu=ppu, heads=heads, dim=dim):
                    kernel = self.make_specialization(
                        ppu=ppu, heads=heads, dim=dim, tail_dim=tail_dim
                    )
                    nonlocals = inspect.getclosurevars(kernel).nonlocals
                    self.assertIs(nonlocals["ppu_layout_fix"], False)
                    self.assertIs(nonlocals["pv_policy"], nonlocals["gemm_policy"])

    def test_padded_ppu_geometry_decouples_the_pv_policy(self):
        outcomes = ((512, 64, _Policy.FullCol), (320, 0, _Policy.FullRow))
        for dim, tail_dim, expected in outcomes:
            with self.subTest(dim=dim, tail_dim=tail_dim):
                kernel = self.make_specialization(
                    ppu=True, heads=16, dim=dim, tail_dim=tail_dim
                )
                nonlocals = inspect.getclosurevars(kernel).nonlocals
                self.assertIs(nonlocals["ppu_layout_fix"], True)
                self.assertIs(nonlocals["gemm_policy"], _Policy.FullRow)
                # Both outcomes of the geometry-dependent fallback: the wide
                # tile keeps FullCol, the narrow one falls back to FullRow.
                self.assertIs(nonlocals["pv_policy"], expected)
                self.assertIs(
                    nonlocals["D_tail_padded"], max(tail_dim, 16) if tail_dim else 0
                )


    def trace_statements(self, *, ppu, heads, dim, tail_dim, topk=128):
        """Run the specialization body and return its live statement sequence.

        `topk` is an explicit argument: the geometry the block loop runs for is
        derived from it, and pinning a production statement sequence without
        passing the production topk would trace a different specialization.
        """
        factory = copy.deepcopy(self.factory_ast)
        factory.decorator_list = []
        _label_allocations(factory)
        module = ast.fix_missing_locations(ast.Module(body=[factory], type_ignores=[]))
        sink = _TraceSink()
        namespace = {
            "is_sail": lambda: ppu,
            "tilelang": SimpleNamespace(
                math=SimpleNamespace(
                    next_power_of_2=lambda n: 1 << (n - 1).bit_length()
                ),
                cdiv=lambda a, b: (a + b - 1) // b,
            ),
            "T": _trace_namespace(sink),
        }
        exec(compile(module, str(__file__), "exec"), namespace)
        main = namespace[factory.name](heads, dim, tail_dim, topk, num_stages=1)
        buffers = {"Q": "Q", "KV": "KV", "Indices": "Indices", "Output": "Output"}
        main(*[_TraceBuf(name, sink) for name in buffers.values()])
        return sink.events

    @staticmethod
    def first(events, predicate):
        for index, event in enumerate(events):
            if predicate(event):
                return index
        return -1

    def test_padded_ppu_tail_tile_is_zeroed_before_the_tail_gemm(self):
        # The tail tile has padded head rows the repack never writes; they must
        # be zeroed before the tail QK gemm reads them. This must hold for every
        # tail width, including the ones where D_tail_padded == D_tail and a
        # width comparison alone would skip the initialisation.
        for heads in (8, 16):
            for dim, tail_dim in ((512, 64), (512, 32), (512, 16), (512, 8), (192, 64)):
                with self.subTest(heads=heads, dim=dim, tail_dim=tail_dim):
                    events = self.trace_statements(
                        ppu=True, heads=heads, dim=dim, tail_dim=tail_dim
                    )
                    fill = self.first(
                        events, lambda e: e[0] == "fill" and e[1] == "Q_tail_shared"
                    )
                    gemm = self.first(
                        events, lambda e: e[0] == "gemm" and e[1] == "Q_tail_shared"
                    )
                    self.assertGreaterEqual(fill, 0, "tail tile is never initialised")
                    self.assertGreaterEqual(gemm, 0, "tail tile is never consumed")
                    self.assertLess(fill, gemm, "tail tile read before it is zeroed")

    def test_padded_ppu_repack_initialises_both_q_tiles(self):
        for heads in (8, 16):
            with self.subTest(heads=heads):
                events = self.trace_statements(
                    ppu=True, heads=heads, dim=512, tail_dim=64
                )
                for name in ("Q_shared", "Q_tail_shared"):
                    self.assertIn(("fill", name), events)
                # The repack writes a head-sized slice, never the whole buffer.
                self.assertIn(("copy", "Q", "Q_shared"), events)
                self.assertNotIn(("copy", "Q_shared", "Q"), events)

    def test_specializations_outside_the_gate_keep_the_upstream_statement_set(self):
        # CUDA, and PPU geometries that need no padding, must emit neither the
        # padded repack nor its initialisations.
        cases = [(False, 8, 512, 64), (False, 16, 512, 64), (False, 32, 64, 128),
                 (True, 192, 512, 64)]
        for ppu, heads, dim, tail_dim in cases:
            with self.subTest(ppu=ppu, heads=heads, dim=dim, tail_dim=tail_dim):
                events = self.trace_statements(
                    ppu=ppu, heads=heads, dim=dim, tail_dim=tail_dim
                )
                self.assertNotIn(("fill", "Q_shared"), events)
                self.assertNotIn(("fill", "Q_tail_shared"), events)
                self.assertIn(("copy", "Q", "Q_shared"), events)
                self.assertIn(("copy", "acc_o", "Output"), events)

    def test_statistic_buffers_are_shared_only_inside_the_gate(self):
        gated = [(True, 8, 512, 0), (True, 16, 512, 64), (True, 32, 64, 128),
                 (True, 64, 512, 64)]
        ungated = [(True, 128, 512, 64), (False, 16, 512, 64), (False, 8, 512, 0)]
        for ppu, heads, dim, tail_dim in gated:
            with self.subTest(ppu=ppu, heads=heads):
                classes = {
                    event[1]: event[2]
                    for event in self.trace_statements(
                        ppu=ppu, heads=heads, dim=dim, tail_dim=tail_dim
                    )
                    if event[0] == "alloc"
                }
                self.assertEqual(classes["sumexp"], "shared")
                self.assertEqual(classes["alpha"], "shared")
                self.assertEqual(classes["sumexp_i"], "fragment")
        for ppu, heads, dim, tail_dim in ungated:
            with self.subTest(ppu=ppu, heads=heads):
                classes = {
                    event[1]: event[2]
                    for event in self.trace_statements(
                        ppu=ppu, heads=heads, dim=dim, tail_dim=tail_dim
                    )
                    if event[0] == "alloc"
                }
                self.assertEqual(classes["sumexp"], "fragment")
                self.assertEqual(classes["alpha"], "fragment")

    def test_real_production_specialization_keeps_the_verified_statements(self):
        # dsa_backend._forward_tilelang -> tilelang_sparse_fwd passes
        # num_heads = layer.tp_q_head_num, d_v, tail_dim = dim - d_v,
        # topk = index_topk padded up to a 64-column block, num_stages=1.
        for heads, dim, tail_dim in ((8, 512, 0), (64, 512, 0), (8, 512, 64)):
            for topk in (2048, 2112):
                with self.subTest(heads=heads, tail_dim=tail_dim, topk=topk):
                    events = self.trace_statements(
                        ppu=True, heads=heads, dim=dim, tail_dim=tail_dim, topk=topk
                    )
                    # The captured production service arguments are topk 2112,
                    # which is 33 blocks of 64; 2048 is the 32-block reference
                    # arm. The extent must be the one that was asked for.
                    extents = [e[1] for e in events if e[0] == "pipeline"]
                    self.assertTrue(extents, "no block loop was emitted")
                    self.assertEqual(
                        set(extents), {(topk + 63) // 64},
                        "the traced specialization does not run the block loop "
                        "the topk it claims to cover requires",
                    )
                    kinds = {event[1]: event[2]
                             for event in events if event[0] == "alloc"}
                    self.assertEqual(kinds["sumexp"], "shared")
                    self.assertEqual(kinds["alpha"], "shared")
                    self.assertEqual(kinds["sumexp_i"], "fragment")
                    if tail_dim:
                        fill = self.first(
                            events, lambda e: e[0] == "fill" and e[1] == "Q_tail_shared"
                        )
                        gemm = self.first(
                            events,
                            lambda e: e[0] == "gemm" and e[1] == "Q_tail_shared",
                        )
                        self.assertGreaterEqual(fill, 0)
                        self.assertLess(fill, gemm)
                    else:
                        # No tail tile exists on this path at all; nothing may
                        # reference one, and the padded repack must still zero
                        # the tile it slices.
                        self.assertFalse(
                            [
                                e
                                for e in events
                                if "Q_tail_shared" in e or "K_tail_shared" in e
                            ],
                            "a tail tile appeared on a tail-free specialization",
                        )
                        if heads < 32:
                            self.assertIn(("fill", "Q_shared"), events)

    @staticmethod
    def _eval_host_expression(node, values):
        """Evaluate one host-side expression from the kernel source.

        Only the arithmetic and conditional forms the head-offset computation
        uses are supported; anything else is a test-harness bug and raises, so
        a kernel-side change in shape cannot be silently ignored.
        """
        if isinstance(node, ast.Name):
            return values[node.id]
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -TestDSATileLangV1GemmPolicy._eval_host_expression(
                node.operand, values)
        if isinstance(node, ast.IfExp):
            test = TestDSATileLangV1GemmPolicy._eval_host_expression(node.test, values)
            taken = node.body if test else node.orelse
            return TestDSATileLangV1GemmPolicy._eval_host_expression(taken, values)
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            left = TestDSATileLangV1GemmPolicy._eval_host_expression(node.left, values)
            right = TestDSATileLangV1GemmPolicy._eval_host_expression(
                node.comparators[0], values)
            op = node.ops[0]
            table = {ast.Gt: left > right, ast.Lt: left < right,
                     ast.Eq: left == right, ast.NotEq: left != right,
                     ast.GtE: left >= right, ast.LtE: left <= right}
            return table[type(op)]
        if isinstance(node, ast.BinOp):
            left = TestDSATileLangV1GemmPolicy._eval_host_expression(node.left, values)
            right = TestDSATileLangV1GemmPolicy._eval_host_expression(node.right, values)
            # dispatched lazily: building an eager table would evaluate every
            # operator for every node and a legal `x + 0` would raise on `x // 0`
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.FloorDiv):
                return left // right
            if isinstance(node.op, ast.Mod):
                return left % right
            raise AssertionError(
                "unsupported host operator: %s" % type(node.op).__name__)
        raise AssertionError("unsupported host expression: %s" % ast.dump(node)[:160])

    def kernel_head_ranges(self, *, ppu, heads, dim, tail_dim, topk, block_I=64):
        """Per-block head ranges, read from the kernel's own H0/H1 statements.

        The ranges are evaluated from the emitted source with the specialization
        constants of the geometry under test, so changing the kernel formulas
        changes what this reports - a re-implemented formula in the test could
        not do that, which is the gap this closes.
        """
        main = next(
            node
            for node in self.factory_ast.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )

        gates = [
            node
            for node in ast.walk(main)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "ppu_layout_fix"
        ]

        def assign_of(target, *, gated):
            """The kernel's own assignment of `target` for this specialization.

            Both branches of the device gate assign it with identically shaped
            expressions, so the gate they sit under decides which one is live:
            the device gate is a host-side ``if`` and source order alone would
            hand back the gated formula even for a specialization that never
            takes it.
            """
            matches = []
            for node in ast.walk(main):
                if not (
                    isinstance(node, ast.Assign)
                    and any(
                        isinstance(t, ast.Name) and t.id == target
                        for t in node.targets
                    )
                ):
                    continue
                live = any(
                    (node in gate.body) if gated else (node in gate.orelse)
                    for gate in gates
                )
                if live:
                    matches.append(node)
            if len(matches) != 1:
                raise AssertionError(
                    "expected one live assignment of %s, found %d"
                    % (target, len(matches))
                )
            return matches[0].value

        padded_H = max(1 << (heads - 1).bit_length(), 32 if ppu else 16)
        replicate = (heads // 64) if heads > 64 else 1
        h_per_block = padded_H if replicate == 1 else 64
        base_values = {
            "padded_H": padded_H,
            "H": heads,
            "REPLICATE_H": replicate,
            "H_per_block": h_per_block,
            "block_I": block_I,
            "g_i": 0,
            "bx": 0,
        }
        # what the specialization actually runs: the device gate is host-side
        gated = ppu and replicate == 1
        h0_expr = assign_of("H0", gated=gated)
        h1_expr = assign_of("H1", gated=gated)
        ranges = []
        for block in range(replicate):
            per_block = dict(base_values, bx=block)
            start = self._eval_host_expression(h0_expr, per_block)
            stop = self._eval_host_expression(
                h1_expr, dict(per_block, H0=start))
            ranges.append((start, stop))
        return replicate, h_per_block, ranges

    def test_replicated_192_head_geometry_keeps_upstream_slices(self):
        # A replicated specialization has 64 shared rows per block; the padded
        # repack would ask those buffers for the whole head range instead.
        for ppu in (True, False):
            with self.subTest(ppu=ppu):
                kernel = self.make_specialization(
                    ppu=ppu, heads=192, dim=512, tail_dim=64
                )
                nonlocals = inspect.getclosurevars(kernel).nonlocals
                self.assertIs(nonlocals["ppu_layout_fix"], False)
                self.assertEqual(nonlocals["REPLICATE_H"], 3)
                self.assertEqual(nonlocals["H_per_block"], 64)
                self.assertIs(nonlocals["pv_policy"], nonlocals["gemm_policy"])
                self.assertEqual(nonlocals["D_tail_padded"], 64)

        # Ranges come from the kernel's own H0/H1 statements for this geometry,
        # not from a formula the test carries alongside it.
        for ppu in (True, False):
            with self.subTest(derive_ppu=ppu):
                replicate, rows, derived = self.kernel_head_ranges(
                    ppu=ppu, heads=192, dim=512, tail_dim=64, topk=2112
                )
                self.assertEqual(replicate, 3)
                self.assertEqual(rows, 64)
                self.assertEqual(
                    derived, [(0, 64), (64, 128), (128, 192)],
                    "the replicated path must keep the upstream 64-row slices",
                )
                for start, stop in derived:
                    self.assertLessEqual(
                        stop, 192, "a kernel head slice reaches past the head dim"
                    )
                    self.assertEqual(stop - start, rows)

    def test_padded_geometries_keep_the_head_group_contract(self):
        """A padded head range may not be combined with a grouped KV head.

        The kernel refuses this at specialization time. The refusal is host-side,
        so it has to hold on every device and on both sides of the layout gate:
        a specialization that reaches the generated body with a grouped KV head
        has never been validated. The accepted combinations are kept green as
        well, so a blanket rejection cannot stand in for the contract.
        """
        # 8 heads pad to 32 rows on PPU (inside the layout gate) and to 16 rows
        # elsewhere (outside it); either way the head range is padded.
        for ppu in (True, False):
            for kv_group in (2, 4):
                with self.subTest(rejected=(ppu, kv_group)):
                    with self.assertRaises(AssertionError):
                        self.make_specialization(
                            ppu=ppu, heads=8, dim=512, tail_dim=0,
                            topk=2048, kv_group=kv_group,
                        )
        # combinations whose head range is not padded must still specialize
        for ppu, heads, kv_group in ((True, 64, 2), (True, 32, 1), (False, 32, 1)):
            with self.subTest(accepted=(ppu, heads, kv_group)):
                self.make_specialization(
                    ppu=ppu, heads=heads, dim=512, tail_dim=0,
                    topk=2048, kv_group=kv_group,
                )

    def test_head_group_refusal_follows_this_revisions_padding_floor(self):
        """The head/group refusal compares against this revision's padding floor.

        Upstream pads the head range to 16 rows; this pull request pads it to 32
        rows on PPU so the reduction layout gets enough rows. The refusal reuses
        the same number, so a specialization with 16 key-value heads per group and
        more than one group counts as padded on PPU and is refused there, while the
        same combination stays accepted off PPU, exactly as upstream accepted it.
        This is a real behaviour difference against upstream and it is pinned here
        so it cannot drift silently.
        """
        # the boundary family: 16 key-value heads per group
        for kv_group in (2, 4):
            heads = 16 * kv_group
            with self.subTest(refused_on_ppu=(heads, kv_group)):
                with self.assertRaises(AssertionError):
                    self.make_specialization(
                        ppu=True, heads=heads, dim=512, tail_dim=0,
                        topk=2048, kv_group=kv_group,
                    )
            with self.subTest(still_built_off_ppu=(heads, kv_group)):
                self.make_specialization(
                    ppu=False, heads=heads, dim=512, tail_dim=0,
                    topk=2048, kv_group=kv_group,
                )
        # upstream's own boundary stays where upstream put it: 8 heads per group is
        # padded on both devices, so both devices refuse it
        for kv_group in (2, 4):
            with self.subTest(refused_on_both_devices=(8 * kv_group, kv_group)):
                for ppu in (True, False):
                    with self.assertRaises(AssertionError):
                        self.make_specialization(
                            ppu=ppu, heads=8 * kv_group, dim=512, tail_dim=0,
                            topk=2048, kv_group=kv_group,
                        )

    def test_masked_gather_selects_zero_for_invalid_entries(self):
        """Invalid index entries must gather a zero value, not slot 0's content.

        Two statements are in scope and only these two: the write into the shared
        key-value tile and the write into the shared rope-tail tile. The case pins
        their shape in the kernel source and then confirms they are the gather
        stores the gated specialization actually runs, so a change to either one
        is visible at CPU level. The recorded statement stream is not modified by
        this case - the pin lives in the source, which keeps the statement-level
        equivalence reported in the pull request body honest.
        """
        gathered = ("KV_shared", "K_tail_shared")
        main = next(
            node
            for node in self.factory_ast.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        parents = {}
        for node in ast.walk(main):
            for child in ast.iter_child_nodes(node):
                parents[child] = node

        def inside_the_gate(statement):
            current = statement
            parent = parents.get(current)
            while parent is not None:
                if (
                    isinstance(parent, ast.If)
                    and isinstance(parent.test, ast.Name)
                    and parent.test.id == "ppu_layout_fix"
                    and branch_of(parent, current) == "body"
                ):
                    return True
                current = parent
                parent = parents.get(current)
            return False

        def branch_of(parent, child):
            for field, value in ast.iter_fields(parent):
                if isinstance(value, list) and any(item is child for item in value):
                    return field
            return None

        found = []
        for node in ast.walk(main):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id in gathered
            ):
                continue
            if not (
                isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "if_then_else"
            ):
                continue
            found.append((target.value.id, node, inside_the_gate(node)))

        found.sort(key=lambda item: item[1].lineno)
        # counted before the gate test as well: a selection on any other gather
        # would silently widen the change beyond the two sites under review
        self.assertEqual(
            [gated for _, _, gated in found], [True, True],
            "both selected gathers have to sit inside the device gate",
        )
        found = [(name, node) for name, node, _ in found]
        self.assertEqual(
            [name for name, _ in found],
            ["KV_shared", "K_tail_shared"],
            "expected exactly the two masked gathers to select their value, in "
            "the order the kernel emits them",
        )
        for name, node in found:
            condition, taken, otherwise = node.value.args
            self.assertEqual(
                ast.unparse(condition), "idx >= 0",
                "%s: the mask condition must be the index validity test" % name)
            self.assertEqual(
                ast.unparse(otherwise), "0",
                "%s: an invalid entry has to gather the constant zero" % name)
            self.assertIsInstance(taken, ast.Subscript)
            self.assertIsInstance(taken.value, ast.Name)
            self.assertEqual(
                taken.value.id, "KV", "%s: not a key-value gather" % name)
            clamps = [
                sub
                for sub in ast.walk(taken)
                if isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Attribute)
                and sub.func.attr == "if_then_else"
            ]
            self.assertEqual(
                len(clamps), 1,
                "%s: the gathered read must keep exactly one clamped index" % name)
            self.assertEqual(
                ast.unparse(clamps[0]), "T.if_then_else(idx >= 0, idx, 0)",
                "%s: the safe clamped address must stay in place" % name)

        # the pinned statements are the live gather stores of the gated path
        for tail_dim, expected in (
            (0, ["KV_shared"]),
            (64, ["KV_shared", "K_tail_shared"]),
        ):
            with self.subTest(tail_dim=tail_dim):
                events = self.trace_statements(
                    ppu=True, heads=8, dim=512, tail_dim=tail_dim, topk=2112
                )
                self.assertEqual(
                    [e[1] for e in events if e[0] == "store" and e[1] in gathered],
                    expected,
                    "the gated specialization must write exactly these gather "
                    "destinations",
                )

    def test_repack_path_is_guarded_by_the_gate(self):
        main = next(
            node
            for node in self.factory_ast.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        parents = {}
        for node in ast.walk(main):
            for child in ast.iter_child_nodes(node):
                parents[child] = node

        def branch_of(parent, child):
            """Which list of `parent` holds `child` (`body`, `orelse`, ...)."""
            for field, value in ast.iter_fields(parent):
                if isinstance(value, list) and any(item is child for item in value):
                    return field
            return None

        def under_the_gate(node):
            # A node counts as gated only when the path from it up to the gate
            # enters each enclosing `if` through its `body`. The `else` branch
            # runs with the device gate off, so it is upstream code, not the
            # workaround, and must not be counted as guarded by the gate.
            child = node
            current = parents.get(child)
            while current is not None:
                entered_through_body = branch_of(current, child) == "body"
                if (
                    isinstance(current, ast.If)
                    and isinstance(current.test, ast.Name)
                    and current.test.id == "ppu_layout_fix"
                    and entered_through_body
                ):
                    return True
                child = current
                current = parents.get(current)
            return False

        def slices_head(node):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "copy"
            ):
                return False
            for arg in node.args:
                if not (
                    isinstance(arg, ast.Subscript)
                    and isinstance(arg.slice, ast.Tuple)
                ):
                    continue
                for element in arg.slice.elts:
                    if (
                        isinstance(element, ast.Slice)
                        and isinstance(element.upper, ast.Name)
                        and element.upper.id == "H"
                    ):
                        return True
            return False

        repack_copies = [node for node in ast.walk(main) if slices_head(node)]
        self.assertTrue(repack_copies, "expected the padded repack copies")
        for node in repack_copies:
            self.assertTrue(
                under_the_gate(node), "a head-sliced copy escaped the gate"
            )

        stale_branches = [
            node
            for node in ast.walk(main)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and any(
                isinstance(part, ast.Name) and part.id == "padded_H"
                for part in [node.test.left] + list(node.test.comparators)
            )
        ]
        for node in stale_branches:
            self.assertTrue(
                under_the_gate(node),
                "a padded_H branch must stay inside the device gate",
            )

    def test_production_eight_head_geometry_on_ppu_and_cuda(self):
        # Production wrapper reaches v1 with eight heads, D=512 and no tail.
        kernel = self.make_specialization(
            ppu=True, heads=8, dim=512, tail_dim=0
        )
        nonlocals = inspect.getclosurevars(kernel).nonlocals
        self.assertEqual(nonlocals["padded_H"], 32)
        self.assertEqual(nonlocals["H_per_block"], 32)
        self.assertIs(nonlocals["ppu_layout_fix"], True)
        self.assertIs(nonlocals["gemm_policy"], _Policy.FullRow)
        self.assertIs(nonlocals["pv_policy"], _Policy.FullCol)
        self.assertEqual(nonlocals["D_tail_padded"], 0)

        kernel = self.make_specialization(
            ppu=False, heads=8, dim=512, tail_dim=0
        )
        nonlocals = inspect.getclosurevars(kernel).nonlocals
        self.assertEqual(nonlocals["padded_H"], 16)
        self.assertIs(nonlocals["ppu_layout_fix"], False)
        self.assertIs(nonlocals["pv_policy"], nonlocals["gemm_policy"])

    def test_production_wrapper_still_routes_these_geometries_to_v1(self):
        # The sm<=89 branch, and the sm>89 branch when there is no tail tile,
        # both call the kernel this test pins.
        direct_calls = [
            node
            for node in ast.walk(self.module_ast)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "sparse_attention_fwd_kernel_v1"
        ]
        self.assertGreaterEqual(len(direct_calls), 1)
        # On newer devices the factory is selected by a ternary on tail_dim.
        ternaries = [
            node
            for node in ast.walk(self.module_ast)
            if isinstance(node, ast.IfExp)
            and isinstance(node.body, ast.Name)
            and node.body.id == "sparse_attention_fwd_kernel_v1"
        ]
        self.assertTrue(
            ternaries, "the tail_dim-based v1/v2 selection must stay explicit"
        )
        branches = [
            node
            for node in ast.walk(self.module_ast)
            if isinstance(node, ast.Compare)
            and isinstance(node.left, ast.Call)
            and isinstance(node.left.func, ast.Name)
            and node.left.func.id == "get_device_sm"
            and isinstance(node.comparators[0], ast.Constant)
            and node.comparators[0].value == 89
        ]
        self.assertTrue(branches, "the <=89 device branch must stay explicit")


if __name__ == "__main__":
    unittest.main()
