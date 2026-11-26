import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.linear.kda_replayssm_ppu import (
    commit_kda_replayssm_ppu,
    target_verify_kda_replayssm_ppu,
)
from sglang.srt.layers.attention.linear.kernels import kda_flashkda
from sglang.srt.layers.attention.linear.kernels.kda_flashkda import PPUFlashKDAKernel
from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel
from sglang.srt.runtime_context import get_context
from sglang.srt.utils.common import is_sail
from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.test_utils import CustomTestCase

register_ppu_ci(est_time=300, suite="stage-a-test-1-gpu-ppu")


class TestFlashKDAReplaySSMPPU(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not is_sail() or not torch.cuda.is_available():
            raise unittest.SkipTest("PPU SM89 kernel is required")
        if torch.cuda.get_device_capability() != (8, 9):
            raise unittest.SkipTest("PPU SM89 kernel is required")

    def test_commit_accept_prefix_and_tracks(self):
        for dtype in (torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                torch.manual_seed(29)
                layers, slots, heads, record_len, dim = 2, 8, 16, 4, 128
                state = (
                    torch.randn(layers, slots, heads, dim, dim, device="cuda") * 0.05
                ).to(dtype)
                correction = (
                    torch.randn(layers, slots, heads, record_len, dim, device="cuda")
                    * 0.01
                )
                key = (
                    torch.randn(layers, slots, heads, record_len, dim, device="cuda")
                    * 0.05
                )
                decay = (
                    torch.rand(layers, slots, heads, record_len, dim, device="cuda")
                    .mul_(0.05)
                    .add_(0.95)
                )
                indices = torch.tensor([1, 2, 3, -1], device="cuda", dtype=torch.int32)
                accepts = torch.tensor(
                    [0, 1, record_len, 2], device="cuda", dtype=torch.int32
                )
                tracks = torch.tensor([-1, 5, 6, -1], device="cuda", dtype=torch.int32)
                steps = torch.tensor([-1, 0, 1, -1], device="cuda", dtype=torch.int32)
                expected = state.clone()
                for slot, count, track, track_step in zip(
                    indices.tolist(), accepts.tolist(), tracks.tolist(), steps.tolist()
                ):
                    if slot < 0 or count <= 0:
                        continue
                    current = state[:, slot].float()
                    for step in range(count):
                        current = (
                            current * decay[:, slot, :, step, None, :]
                            + correction[:, slot, :, step, :, None]
                            * key[:, slot, :, step, None, :]
                        )
                        if track >= 0 and step == track_step:
                            expected[:, track] = current.to(dtype)
                    expected[:, slot] = current.to(dtype)
                commit_kda_replayssm_ppu(
                    state, correction, key, decay, indices, accepts, tracks, steps
                )
                tolerance = 0.001 if dtype == torch.bfloat16 else 2e-6
                torch.testing.assert_close(state, expected, rtol=0, atol=tolerance)

    @torch.inference_mode()
    def test_packed_decode_matches_triton(self):
        for lower_bound in (-5.0, None):
            for dtype in (torch.bfloat16, torch.float32):
                with self.subTest(lower_bound=lower_bound, dtype=dtype):
                    torch.manual_seed(47 + (lower_bound is None))
                    batch, heads, dim, slots = 4, 8, 128, 8
                    mixed_qkv = (
                        torch.randn(batch, 3 * heads * dim, device="cuda") * 0.1
                    ).to(torch.bfloat16)
                    a = (torch.randn(batch, 1, heads * dim, device="cuda") * 0.1).to(
                        torch.bfloat16
                    )
                    b = (torch.randn(batch, 1, heads, device="cuda") * 0.1).to(
                        torch.bfloat16
                    )
                    A_log = torch.randn(heads, device="cuda") * 0.1
                    dt_bias = torch.randn(heads * dim, device="cuda") * 0.1
                    initial = (
                        torch.randn(slots, heads, dim, dim, device="cuda") * 0.05
                    ).to(dtype)
                    # Include a padded row and nonsequential int64 pool indices.
                    indices = torch.tensor(
                        [6, 2, -1, 4], device="cuda", dtype=torch.int64
                    )
                    actual_state, expected_state = initial.clone(), initial.clone()
                    kwargs = dict(
                        mixed_qkv=mixed_qkv,
                        a=a,
                        b=b,
                        A_log=A_log,
                        dt_bias=dt_bias,
                        scale=dim**-0.5,
                        cache_indices=indices,
                        num_v_heads=heads,
                        head_v_dim=dim,
                        lower_bound=lower_bound,
                    )
                    actual = PPUFlashKDAKernel().packed_decode(
                        **kwargs, ssm_states=actual_state
                    )
                    expected = TritonKDAKernel().packed_decode(
                        **kwargs, ssm_states=expected_state
                    )
                    self._assert_extend_close(actual, expected)
                    self._assert_extend_close(actual_state, expected_state)
                    untouched = [0, 1, 3, 5, 7]
                    torch.testing.assert_close(
                        actual_state[untouched],
                        initial[untouched],
                        rtol=0,
                        atol=0,
                    )
                    torch.testing.assert_close(
                        actual[:, 2],
                        torch.zeros_like(actual[:, 2]),
                        rtol=0,
                        atol=0,
                    )

    def test_verify_readonly_output_records_and_accepted_state(self):
        for heads in (8, 16, 64):
            for length in (4, 5):
                for dtype in (torch.bfloat16, torch.float32):
                    with self.subTest(heads=heads, length=length, dtype=dtype):
                        self._check_verify(heads, length, dtype)

    def _check_verify(self, heads, length, dtype):
        torch.manual_seed(31)
        batch, slots, dim = 4, 8, 128
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
        initial = (torch.randn(slots, heads, dim, dim, device="cuda") * 0.05).to(dtype)
        indices = torch.tensor([1, 2, 3, -1], device="cuda", dtype=torch.int32)
        accepts = torch.tensor([0, 1, length, 2], device="cuda", dtype=torch.int32)
        tracks = torch.tensor([-1, 5, 6, -1], device="cuda", dtype=torch.int32)
        steps = torch.tensor([-1, 0, length - 1, -1], device="cuda", dtype=torch.int32)
        lower_bound = -5.0
        expected_out = torch.zeros(shape, device="cuda", dtype=torch.float32)
        expected_records = [
            torch.zeros(slots, heads, length, dim, device="cuda") for _ in range(3)
        ]
        expected_state = initial.clone()
        for request, slot in enumerate(indices.tolist()):
            if slot < 0:
                continue
            current = initial[slot].float()
            for step in range(length):
                token = request * length + step
                query = q[0, token].float()
                query = query / torch.sqrt(query.square().sum(-1, keepdim=True) + 1e-6)
                query = query * dim**-0.5
                key = k[0, token].float()
                key = key / torch.sqrt(key.square().sum(-1, keepdim=True) + 1e-6)
                decay = torch.exp(
                    lower_bound
                    * torch.sigmoid(
                        A_log.exp()[:, None] * (a[0, token].float() + dt_bias)
                    )
                )
                current = current * decay[:, None, :]
                correction = (
                    v[0, token].float() - (current * key[:, None, :]).sum(-1)
                ) * torch.sigmoid(b[0, token].float())[:, None]
                current = current + correction[:, :, None] * key[:, None, :]
                expected_out[0, token] = (current * query[:, None, :]).sum(-1)
                for record, value in zip(expected_records, (correction, key, decay)):
                    record[slot, :, step] = value
                if step + 1 == accepts[request].item():
                    expected_state[slot] = current.to(dtype)
                if tracks[request].item() >= 0 and step == steps[request].item():
                    expected_state[tracks[request]] = current.to(dtype)
        for backend in ("native", "flashkda"):
            with self.subTest(backend=backend):
                state = initial.clone()
                correction, key, decay = [
                    torch.zeros_like(record) for record in expected_records
                ]
                if backend == "native":
                    output = target_verify_kda_replayssm_ppu(
                        q=q,
                        k=k,
                        v=v,
                        a=a,
                        b=b,
                        A_log=A_log,
                        dt_bias=dt_bias,
                        ssm_states=state,
                        cache_indices=indices,
                        correction=correction,
                        key=key,
                        decay=decay,
                        lower_bound=lower_bound,
                    )
                else:
                    output = PPUFlashKDAKernel().target_verify(
                        A_log=A_log,
                        dt_bias=dt_bias,
                        q=q,
                        k=k,
                        v=v,
                        a=a,
                        b=b,
                        ssm_states=state,
                        cache_indices=indices,
                        query_start_loc=torch.arange(
                            0,
                            (batch + 1) * length,
                            length,
                            device="cuda",
                            dtype=torch.int32,
                        ),
                        intermediate_states_buffer=None,
                        intermediate_state_indices=indices,
                        cache_steps=length,
                        retrieve_parent_token=None,
                        lower_bound=lower_bound,
                        native_correction=correction,
                        native_key=key,
                        native_decay=decay,
                    )
                torch.testing.assert_close(state, initial, rtol=0, atol=0)
                torch.testing.assert_close(
                    output.float(), expected_out, rtol=0.01, atol=2e-4
                )
                for actual, expected in zip((correction, key, decay), expected_records):
                    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=2e-6)
                commit_kda_replayssm_ppu(
                    state.unsqueeze(0),
                    correction.unsqueeze(0),
                    key.unsqueeze(0),
                    decay.unsqueeze(0),
                    indices,
                    accepts,
                    tracks,
                    steps,
                )
                tolerance = 5e-4 if dtype == torch.bfloat16 else 2e-6
                torch.testing.assert_close(
                    state, expected_state, rtol=1e-4, atol=tolerance
                )

    @torch.inference_mode()
    def test_extend_raw_safe_gate_accumulation(self):
        for heads in (8, 16, 64):
            for dtype in (torch.bfloat16, torch.float32):
                for accum in ("bfloat16", "float32"):
                    with self.subTest(heads=heads, dtype=dtype, accum=accum):
                        with get_context().override_server_args(
                            mamba_ssm_accum_dtype=accum
                        ):
                            self._check_extend(heads, dtype, accum, tracked=False)

    @torch.inference_mode()
    def test_extend_sparse_radix_checkpoints_and_restore(self):
        for heads in (8, 16, 64):
            for dtype in (torch.bfloat16, torch.float32):
                for accum in ("bfloat16", "float32"):
                    with self.subTest(heads=heads, dtype=dtype, accum=accum):
                        with get_context().override_server_args(
                            mamba_ssm_accum_dtype=accum
                        ):
                            self._check_extend(heads, dtype, accum, tracked=True)

    def _assert_extend_close(self, actual, expected, *, exact=False):
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertTrue(torch.isfinite(actual).all().item())
        if exact:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            return
        # test_kda_prefill.py elementwise bounds plus FlashKDA test_ppu_glm53.py's 2.5% norm bound.
        torch.testing.assert_close(
            actual.float(), expected.float(), rtol=2e-2, atol=3e-2
        )
        relative = (
            actual.float() - expected.float()
        ).norm() / expected.float().norm().clamp_min(1e-12)
        self.assertLess(relative.item(), 0.025)

    def _check_extend(self, heads, dtype, accum, *, tracked):
        torch.manual_seed(1109 + heads)
        lengths = [63, 64, 65, 127, 128, 129, 4097]
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        dim, slots = 128, 10
        shape = (1, offsets[-1], heads, dim)
        raw = [
            (torch.randn(shape, device="cuda") * 0.2).to(torch.bfloat16)
            for _ in range(3)
        ]
        raw.append((torch.randn(shape, device="cuda") * 0.25 - 2).to(torch.bfloat16))
        raw.append(torch.randn(shape[:-1], device="cuda", dtype=torch.bfloat16))
        parameters = dict(
            A_log=(torch.randn(1, 1, heads, 1, device="cuda") * 0.1),
            dt_bias=(torch.randn(heads * dim, device="cuda") * 0.1),
            lower_bound=-5.0,
            beta_is_raw=True,
        )
        initial = (torch.randn(slots, heads, dim, dim, device="cuda") * 0.01).to(dtype)
        indices = torch.tensor([7, 3, 6, 1, 5, 2, 4], device="cuda", dtype=torch.int32)
        unused = torch.tensor([0, 8, 9], device="cuda", dtype=torch.int64)
        # Caller scratch is FP32 [batch,H,V,K]; j is BEFORE local chunk j, -1 is unread.
        chunks = [0, -1, 1, 1, -1, 2, 64]
        track_idx = torch.tensor(chunks, device="cuda", dtype=torch.int32)
        sentinel = torch.full(
            (len(lengths), heads, dim, dim), 7.0, device="cuda", dtype=torch.float32
        )
        candidate, oracle = PPUFlashKDAKernel(), TritonKDAKernel()
        flash_kda = kda_flashkda._load_flash_kda(ppu=True)
        native = not (tracked and accum == "bfloat16" and dtype == torch.float32)

        def run(
            kernel, tensors, state, cache_indices, lens, track=None, *, native_path=None
        ):
            cu = [0]
            for length in lens:
                cu.append(cu[-1] + length)
            kwargs = dict(
                zip(("q", "k", "v", "g", "beta"), (x.clone() for x in tensors))
            )
            kwargs.update(
                parameters,
                ssm_states=state,
                cache_indices=cache_indices,
                query_start_loc=torch.tensor(cu, device="cuda", dtype=torch.int32),
                extend_seq_lens_cpu=lens,
                return_intermediate_states=track is not None,
                track_state=track,
                track_chunk_idx=track_idx if track is not None else None,
                state_chunk_size=64,
            )
            if native_path is None:
                result = kernel.extend(**kwargs)
            else:
                with patch.object(flash_kda, "fwd", wraps=flash_kda.fwd) as fwd:
                    with patch.object(
                        kda_flashkda,
                        "_triton_fallback",
                        wraps=kda_flashkda._triton_fallback,
                    ) as fallback:
                        result = kernel.extend(**kwargs)
                path = "native" if native_path else "fallback"
                self.assertEqual(fwd.call_count, int(native_path), path)
                self.assertEqual(fallback.call_count, int(not native_path), path)
                if native_path:
                    self.assertEqual(
                        fwd.call_args.kwargs["state_accum_dtype"], getattr(torch, accum)
                    )
                fwd.reset_mock()
                fallback.reset_mock()
            if track is not None:
                self.assertIsInstance(result, tuple)
                self.assertEqual(len(result), 2)
                if native_path:
                    self.assertIsNone(result[1])
                result = result[0]
            self.assertIsInstance(result, torch.Tensor)
            return result

        actual_state, expected_state = initial.clone(), initial.clone()
        actual_track = sentinel.clone() if tracked else None
        expected_track = sentinel.clone() if tracked else None
        actual_out = run(
            candidate,
            raw,
            actual_state,
            indices,
            lengths,
            actual_track,
            native_path=native,
        )
        expected_out = run(
            oracle, raw, expected_state, indices, lengths, expected_track
        )
        for row, (begin, end) in enumerate(zip(offsets, offsets[1:])):
            with self.subTest(
                length=lengths[row], path="native" if native else "fallback"
            ):
                self._assert_extend_close(
                    actual_out[:, begin:end],
                    expected_out[:, begin:end],
                    exact=not native,
                )
                self._assert_extend_close(
                    actual_state[indices[row]],
                    expected_state[indices[row]],
                    exact=not native,
                )
        for state in (actual_state, expected_state):
            torch.testing.assert_close(state[unused], initial[unused], rtol=0, atol=0)
        if not tracked:
            return
        for row, chunk in enumerate(chunks):
            with self.subTest(checkpoint_row=row, chunk=chunk):
                if chunk < 0:
                    for track in (actual_track, expected_track):
                        torch.testing.assert_close(
                            track[row], sentinel[row], rtol=0, atol=0
                        )
                elif chunk == 0:
                    for track in (actual_track, expected_track):
                        torch.testing.assert_close(
                            track[row], initial[indices[row]].float(), rtol=0, atol=0
                        )
                else:
                    self._assert_extend_close(
                        actual_track[row], expected_track[row], exact=not native
                    )

        positive_rows = [row for row, chunk in enumerate(chunks) if chunk > 0]
        prefix_raw = [
            torch.cat(
                [
                    x[:, offsets[row] : offsets[row] + chunks[row] * 64]
                    for row in positive_rows
                ],
                dim=1,
            )
            for x in raw
        ]
        prefix_indices = indices[positive_rows]
        prefix_state = initial.float().clone()
        run(
            oracle,
            prefix_raw,
            prefix_state,
            prefix_indices,
            [chunks[row] * 64 for row in positive_rows],
        )
        for row in positive_rows:
            with self.subTest(prefix_oracle_row=row):
                torch.testing.assert_close(
                    expected_track[row],
                    prefix_state[indices[row]],
                    rtol=1e-5,
                    atol=1e-5,
                )
                self._assert_extend_close(actual_track[row], prefix_state[indices[row]])
        torch.testing.assert_close(
            prefix_state[unused], initial[unused].float(), rtol=0, atol=0
        )

        restore_rows = [row for row, chunk in enumerate(chunks) if chunk >= 0]
        restore_indices = indices[restore_rows]
        suffix_lengths = [lengths[row] - chunks[row] * 64 for row in restore_rows]
        suffix_raw = [
            torch.cat(
                [
                    x[:, offsets[row] + chunks[row] * 64 : offsets[row + 1]]
                    for row in restore_rows
                ],
                dim=1,
            )
            for x in raw
        ]
        suffix_oracle = torch.cat(
            [
                expected_out[:, offsets[row] + chunks[row] * 64 : offsets[row + 1]]
                for row in restore_rows
            ],
            dim=1,
        )
        untouched = [
            slot for slot in range(slots) if slot not in restore_indices.tolist()
        ]
        for name, kernel, track in (
            ("flashkda", candidate, actual_track),
            ("triton", oracle, expected_track),
        ):
            with self.subTest(restore_backend=name):
                restored = initial.clone()
                # Match _track_mamba_state_extend's publication into resident storage.
                restored[restore_indices] = track[restore_rows].to(dtype)
                resumed_out = run(
                    kernel,
                    suffix_raw,
                    restored,
                    restore_indices,
                    suffix_lengths,
                    native_path=True if name == "flashkda" else None,
                )
                start = 0
                for row, length in zip(restore_rows, suffix_lengths):
                    with self.subTest(restored_row=row):
                        self._assert_extend_close(
                            resumed_out[:, start : start + length],
                            suffix_oracle[:, start : start + length],
                        )
                        self._assert_extend_close(
                            restored[indices[row]], expected_state[indices[row]]
                        )
                    start += length
                torch.testing.assert_close(
                    restored[untouched], initial[untouched], rtol=0, atol=0
                )


if __name__ == "__main__":
    unittest.main()
