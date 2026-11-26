"""PR333 dense-chain KDA verify and accepted-prefix SSM-commit benchmark.

Run on PPU SM89 with SGLANG_SAIL_PLA_CUDA=0 and the FlashKDA 0dfef21 wheel
providing verify_replayssm. Reject PLA dispatch rather than label it Triton.
Only BF16 inputs / FP32 states, seven (B, H, T) shapes with T=4/5 are covered;
verify+commit is a single-layer kernel pipeline, NOT serving E2E or prefill.
Adapter output allocation is included; setup, validation, state resets, JIT
compilation and graph capture are excluded from the GPU-event measurements.
"""

import json
import math
import statistics
import unittest

import torch

from sglang.kernels.ops.attention.linear.kda_replayssm_ppu import (
    commit_kda_replayssm_ppu,
    target_verify_kda_replayssm_ppu,
)
from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
    fused_mamba_state_scatter_with_mask,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.linear.kernels.kda_flashkda import PPUFlashKDAKernel
from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel
from sglang.srt.utils.common import is_sail
from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.test_utils import CustomTestCase

register_ppu_ci(est_time=120, suite="stage-a-test-1-gpu-ppu")

_SHAPES = (
    (1, 8, 4),
    (1, 16, 5),
    (8, 16, 4),
    (8, 64, 5),
    (64, 8, 5),
    (64, 64, 4),
    (80, 64, 5),
)
_WARMUP = 10
_SAMPLES = 31


class BenchFlashKDAReplaySSMPPU(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not is_sail() or not torch.cuda.is_available():
            raise unittest.SkipTest("Real PPU SM89 GPU is required")
        if torch.cuda.get_device_capability() != (8, 9):
            raise unittest.SkipTest("Real PPU SM89 GPU is required")
        if envs.SGLANG_SAIL_PLA_CUDA.get():
            raise RuntimeError(
                "Set SGLANG_SAIL_PLA_CUDA=0 for this benchmark: FP32 "
                "TritonKDAKernel.target_verify otherwise dispatches to PLA."
            )

    @torch.inference_mode()
    def test_verify_and_accepted_prefix_commit(self):
        for batch, heads, length in _SHAPES:
            with self.subTest(batch=batch, heads=heads, length=length):
                self._bench_shape(batch, heads, length)

    def _bench_shape(self, batch, heads, length):
        seed = 333 + batch * 100 + heads * 10 + length
        torch.manual_seed(seed)
        slots, dim = batch + 1, 128
        shape = (1, batch * length, heads, dim)
        q, k, v, a = [
            (torch.randn(shape, device="cuda") * 0.1).to(torch.bfloat16)
            for _ in range(4)
        ]
        b = (torch.randn(1, batch * length, heads, device="cuda") * 0.1).to(
            torch.bfloat16
        )
        A_log = torch.randn(heads, device="cuda") * 0.1
        dt_bias = torch.randn(heads, dim, device="cuda") * 0.1
        initial = torch.randn(slots, heads, dim, dim, device="cuda") * 0.05
        triton_state, native_state, flash_state = [initial.clone() for _ in range(3)]

        # Physical committed slots and scratch rows are deliberately NOT request
        # row numbers, and for B>1 they differ from each other. Compact records
        # are indexed by physical state slot, not intermediate_state_indices.
        physical_slots = list(range(batch, 0, -1))
        scratch_rows = physical_slots[1:] + physical_slots[:1]
        accepted_counts = (
            [length] if batch == 1 else [i % (length + 1) for i in range(batch)]
        )
        indices = torch.tensor(physical_slots, dtype=torch.int32, device="cuda")
        scratch_indices = torch.tensor(scratch_rows, dtype=torch.int32, device="cuda")
        accepts = torch.tensor(accepted_counts, dtype=torch.int32, device="cuda")
        accepted_steps = accepts - 1  # -1 skips zero accepts; t stores AFTER token t.
        starts = torch.arange(
            0, (batch + 1) * length, length, device="cuda", dtype=torch.int32
        )
        snapshots = torch.zeros(
            slots, length, heads, dim, dim, device="cuda", dtype=torch.float32
        )
        native_records = [
            torch.zeros(slots, heads, length, dim, device="cuda", dtype=torch.float32)
            for _ in range(3)
        ]
        flash_records = [torch.zeros_like(record) for record in native_records]
        # Pre-built layer views match the production all-layer commit/scatter ABI.
        snapshot_layers = snapshots.unsqueeze(0)
        triton_layers = triton_state.unsqueeze(0)
        native_layers = native_state.unsqueeze(0)
        flash_layers = flash_state.unsqueeze(0)
        native_record_layers = [record.unsqueeze(0) for record in native_records]
        flash_record_layers = [record.unsqueeze(0) for record in flash_records]
        triton = TritonKDAKernel()
        flashkda = PPUFlashKDAKernel()
        common = dict(q=q, k=k, v=v, a=a, b=b, A_log=A_log, dt_bias=dt_bias)
        chain = dict(
            cache_indices=indices,
            query_start_loc=starts,
            intermediate_state_indices=scratch_indices,
            cache_steps=length,
            retrieve_parent_token=None,
            lower_bound=-5.0,
        )

        def triton_verify():
            return triton.target_verify(
                **common,
                **chain,
                ssm_states=triton_state,
                intermediate_states_buffer=snapshots,
            )

        def native_verify():
            return target_verify_kda_replayssm_ppu(
                **common,
                ssm_states=native_state,
                cache_indices=indices,
                correction=native_records[0],
                key=native_records[1],
                decay=native_records[2],
                lower_bound=-5.0,
            )

        def flash_verify():
            # Supplying compact buffers requires the external verify_replayssm
            # ABI; this adapter raises rather than falling back or using PLA.
            return flashkda.target_verify(
                **common,
                **chain,
                ssm_states=flash_state,
                intermediate_states_buffer=None,
                native_correction=flash_records[0],
                native_key=flash_records[1],
                native_decay=flash_records[2],
            )

        def snapshot_commit():
            fused_mamba_state_scatter_with_mask(
                triton_layers,
                snapshot_layers,
                indices,
                accepted_steps,
                src_indices_raw=scratch_indices,
            )

        def native_commit():
            commit_kda_replayssm_ppu(
                native_layers, *native_record_layers, indices, accepts
            )

        def flash_commit():
            # External compact records use the same native production commit.
            commit_kda_replayssm_ppu(
                flash_layers, *flash_record_layers, indices, accepts
            )

        def triton_pipeline():
            output = triton_verify()
            snapshot_commit()
            return output

        def native_pipeline():
            output = native_verify()
            native_commit()
            return output

        def flash_pipeline():
            output = flash_verify()
            flash_commit()
            return output

        operations = {
            "triton_verify": triton_verify,
            "native_verify": native_verify,
            "flashkda_verify": flash_verify,
            "snapshot_copy_commit": snapshot_commit,
            "native_commit": native_commit,
            "flashkda_records_native_commit": flash_commit,
            "triton_verify_commit": triton_pipeline,
            "native_verify_commit": native_pipeline,
            "flashkda_verify_native_commit": flash_pipeline,
        }
        operation_states = dict(
            zip(
                operations,
                (
                    triton_state,
                    native_state,
                    flash_state,
                    triton_state,
                    native_state,
                    flash_state,
                    triton_state,
                    native_state,
                    flash_state,
                ),
            )
        )

        # Untimed eager execution compiles/loads real backends and establishes
        # the full-snapshot reference before any warmup or graph capture.
        reference = triton_verify()
        self._assert_finite(reference)
        self._assert_finite(snapshots)
        self.assertTrue(torch.equal(triton_state, initial))
        for output, state in (
            (native_verify(), native_state),
            (flash_verify(), flash_state),
        ):
            self._assert_output(output, reference)
            self.assertTrue(torch.equal(state, initial))
        self._assert_records(native_records, flash_records)

        # Independent Torch gather of accepted snapshots uses the explicit
        # source row, physical destination and count-1, never src=request row.
        expected_state = initial.clone()
        for slot, row, count in zip(physical_slots, scratch_rows, accepted_counts):
            if count > 0:
                expected_state[slot].copy_(snapshots[row, count - 1])

        def reset(name):
            operation_states[name].copy_(initial)

        def check(name, output):
            state = operation_states[name]
            if output is not None:
                self._assert_output(output, reference)
            if name.endswith("verify"):
                self.assertTrue(torch.equal(state, initial), f"{name} mutated state")
            else:
                self._assert_finite(state)
                torch.testing.assert_close(state, expected_state, rtol=1e-4, atol=2e-6)

        # Validate commits BEFORE collecting timings, including zero accepts and
        # the unused padding slot. Repeat correctness checks on captured outputs.
        for name, operation in operations.items():
            reset(name)
            check(name, operation())
        timings = {
            mode: self._measure_paired(operations, reset, check, mode)
            for mode in ("eager", "graph")
        }
        self._assert_records(native_records, flash_records)
        full_bytes = snapshots.numel() * snapshots.element_size()
        compact_bytes = sum(r.numel() * r.element_size() for r in native_records)
        print(
            json.dumps(
                {
                    "benchmark": "pr333_kda_replayssm_ppu",
                    "scope": "single_layer_dense_verify_and_accepted_ssm_commit_only",
                    "device": torch.cuda.get_device_name(),
                    "shape": {"B": batch, "H": heads, "T": length, "D": dim},
                    "slots": slots,
                    "input_dtype": "bfloat16",
                    "state_dtype": "float32",
                    "seed": seed,
                    "accept_count_histogram": {
                        str(count): accepted_counts.count(count)
                        for count in sorted(set(accepted_counts))
                    },
                    "live_scratch_bytes_per_backend_per_layer": {
                        "triton_full_snapshots": full_bytes,
                        "three_fp32_compact_records": compact_bytes,
                        "full_over_compact": full_bytes / compact_bytes,
                    },
                    "timing": timings,
                    "excludes": "resets,setup,checks,jit,capture,conv_rollback,radix_tracks,prefill,serving",
                },
                sort_keys=True,
                allow_nan=False,
            ),
            flush=True,
        )

    def _assert_finite(self, tensor):
        self.assertTrue(torch.isfinite(tensor).all().item())

    def _assert_output(self, output, reference):
        self._assert_finite(output)
        torch.testing.assert_close(
            output.float(), reference.float(), rtol=0.01, atol=2e-4
        )

    def _assert_records(self, native_records, flash_records):
        for native, external in zip(native_records, flash_records):
            self._assert_finite(native)
            self._assert_finite(external)
            torch.testing.assert_close(native, external, rtol=1e-4, atol=2e-6)
            self.assertEqual(torch.count_nonzero(native[0]).item(), 0)
            self.assertEqual(torch.count_nonzero(external[0]).item(), 0)

    def _measure_paired(self, operations, reset, check, mode):
        names = list(operations)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        # Side-stream warmup resolves lazy initialization before graph capture.
        with torch.cuda.stream(stream):
            for name in names:
                for _ in range(_WARMUP):
                    reset(name)
                    operations[name]()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graphs, captured_outputs = {}, {}
        calls = operations
        if mode == "graph":
            for name in names:
                reset(name)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    captured_outputs[name] = operations[name]()
                graphs[name] = graph
            calls = {name: graph.replay for name, graph in graphs.items()}
            for name in names:
                for _ in range(_WARMUP):
                    reset(name)
                    calls[name]()
            torch.cuda.synchronize()

        # One call per event pair: every mutating operation starts from the SAME
        # checkpoint. Reset copies are stream-ordered BEFORE start, not timed,
        # and never captured. Repeated commit without reset would fold twice.
        events = {
            name: [
                (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                for _ in range(_SAMPLES)
            ]
            for name in names
        }
        # CUDA events initialize lazily on first record. Prime both endpoints
        # so first-use host allocation cannot inflate a measured interval.
        for pairs in events.values():
            for start, end in pairs:
                start.record()
                end.record()
        torch.cuda.synchronize()
        flush = torch.empty(
            5
            * torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).L2_cache_size,
            dtype=torch.uint8,
            device="cuda",
        )
        for trial in range(_SAMPLES):
            offset = trial % len(names)
            for name in names[offset:] + names[:offset]:
                reset(name)
                flush.zero_()
                start, end = events[name][trial]
                start.record()
                calls[name]()
                end.record()
        torch.cuda.synchronize()
        samples = {
            name: [start.elapsed_time(end) * 1000 for start, end in events[name]]
            for name in names
        }
        for values in samples.values():
            self.assertTrue(all(math.isfinite(value) and value > 0 for value in values))
        for name in names:
            reset(name)
            output = calls[name]()
            torch.cuda.synchronize()
            check(name, captured_outputs[name] if mode == "graph" else output)

        pairs = (
            ("triton_verify", "native_verify"),
            ("triton_verify", "flashkda_verify"),
            ("snapshot_copy_commit", "native_commit"),
            ("snapshot_copy_commit", "flashkda_records_native_commit"),
            ("triton_verify_commit", "native_verify_commit"),
            ("triton_verify_commit", "flashkda_verify_native_commit"),
        )
        return {
            "warmup_calls_per_operation": _WARMUP,
            "graph_replay_warmup_calls": _WARMUP if mode == "graph" else 0,
            "paired_samples": _SAMPLES,
            "calls_per_event_pair": 1,
            "l2": "cold; 5x device L2 flushed before each timed call",
            "order": "rotating_within_each_paired_trial",
            "median_us": {
                name: statistics.median(values) for name, values in samples.items()
            },
            "median_paired_speedup_baseline_over_candidate": {
                f"{baseline}/{candidate}": statistics.median(
                    base / candidate_time
                    for base, candidate_time in zip(
                        samples[baseline], samples[candidate]
                    )
                )
                for baseline, candidate in pairs
            },
        }


if __name__ == "__main__":
    # CI launches each registered file in its own process; do not disable PLA
    # for the model smoke tests or the numerical oracle in neighboring files.
    with envs.SGLANG_SAIL_PLA_CUDA.override(False):
        unittest.main()
