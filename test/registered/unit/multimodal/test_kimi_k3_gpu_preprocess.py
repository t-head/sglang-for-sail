"""K3 preprocess: shared batched pipeline hook contract and backend selection.

The pipeline tests cover the CPU-computable parts of the GPU path (the resize
kernel's agreement with PIL, the transparent-background composite). The
backend-selection tests cover ``SGLANG_K3_IMAGE_PREPROCESS_MODE``, which
chooses between that GPU path and the checkpoint's own CPU processor.
"""

import io
import sys
import threading
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch
from PIL import Image

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.multimodal.media_artifacts import MediaArtifactInput
from sglang.srt.multimodal.processors.kimi_k3 import (
    KimiK3GPUProcessorWrapper,
    KimiK3ImageProcessor,
    _estimate_gpu_preprocess_bytes,
    _K3EncodedImage,
    _KimiK3VisualItem,
    _fill_transparent_bg,
    _probe_encoded_image,
    _resolve_image_preprocess_mode,
)
from sglang.srt.multimodal.processors.kimi_k25 import _resize_bicubic_if_needed
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=14, suite="base-a-test-cpu")


def _natural_image(height: int, width: int) -> np.ndarray:
    """Deterministic natural-image-like content: gradients, hard edges,
    and high-frequency texture (the aliasing-sensitive case)."""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    base = (
        127
        + 60 * np.sin(2 * np.pi * xx / (width / 7.3))
        + 50 * np.cos(2 * np.pi * yy / (height / 5.1))
    )
    edges = 255.0 * ((xx // 9 + yy // 7) % 2)
    tex = 30.0 * np.sin(xx * 12.9898 + yy * 78.233)
    img = np.clip(0.55 * base + 0.30 * edges + 0.15 * (127 + tex), 0, 255)
    return np.stack(
        [img, np.roll(img, 13, axis=0), np.roll(img, 29, axis=1)], axis=-1
    ).astype(np.uint8)


def test_resize_matches_pil_bicubic_golden():
    """The GPU resize must reproduce the checkpoint processor's
    PIL.Image.resize(..., BICUBIC) downscale: PIL antialiases (kernel support
    scales with the ratio) and returns uint8. Without antialias=True the
    difference on textured content reaches tens of pixel levels."""
    arr = _natural_image(1200, 1600)
    for target_w, target_h in ((800, 600), (1120, 840)):
        golden = np.asarray(
            Image.fromarray(arr).resize((target_w, target_h), Image.Resampling.BICUBIC)
        ).astype(np.float32)

        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        ours = (
            _resize_bicubic_if_needed(x, target_h, target_w)
            .squeeze(0)
            .permute(1, 2, 0)
            .numpy()
        )

        diff = np.abs(ours - golden)
        # Integer pixel domain: everything within 1 level, most pixels exact.
        assert diff.max() <= 1.0, f"max |diff|={diff.max()} at {target_w}x{target_h}"
        assert (diff == 0).mean() > 0.7, f"bitwise ratio={(diff == 0).mean():.3f}"


def test_fill_transparent_bg_matches_checkpoint_composite():
    """Composite + truncation must match the checkpoint's numpy reference:
    alpha * rgb + (1 - alpha) * chessboard, then astype(np.uint8)."""
    cfg = {
        "pattern": "chessboard",
        "chessboard_square_size": 8,
        "chessboard_square_on_top_left": True,
        "chessboard_white_value": 255,
        "chessboard_gray_value": 180,
    }
    rgba = _natural_image(32, 40)
    alpha = ((np.mgrid[0:32, 0:40][0] * 6) % 256).astype(np.uint8)
    img = np.concatenate([rgba, alpha[..., None]], axis=-1)

    # Checkpoint reference (media_utils.fill_transparent_bg_with).
    bg = np.ones((32, 40, 3), dtype=np.uint8) * 255
    for y in range(0, 32, 8):
        for x0 in range(0, 40, 8):
            if (y // 8 + x0 // 8) % 2 == 1:
                bg[y : y + 8, x0 : x0 + 8] = 180
    a3 = np.stack([alpha.astype(np.float32) / 255.0] * 3, axis=2)
    golden = (a3 * img[:, :, :3] + (1 - a3) * bg).astype(np.uint8)

    x = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)
    ours = _fill_transparent_bg(x, cfg).squeeze(0).permute(1, 2, 0).numpy()
    assert np.array_equal(ours, golden.astype(np.float32))


def test_fill_transparent_bg_batch_matches_per_image():
    """Compositing a batch must be bitwise identical to per-image calls
    (the batched pipeline applies it to whole resize groups)."""
    torch.manual_seed(0)
    batch = torch.rand(3, 4, 8, 6) * 255.0
    cfg = {"pattern": "chessboard", "chessboard_square_size": 2}

    batched = _fill_transparent_bg(batch, cfg)
    per_image = torch.cat(
        [_fill_transparent_bg(batch[i : i + 1], cfg) for i in range(batch.shape[0])]
    )
    assert torch.equal(batched, per_image)


def test_fill_transparent_bg_rgb_passthrough_batch():
    batch = torch.rand(2, 3, 4, 4) * 255.0
    assert _fill_transparent_bg(batch, {"pattern": "white"}) is batch


def test_fill_transparent_bg_no_config_drops_alpha():
    batch = torch.rand(2, 4, 4, 4) * 255.0
    out = _fill_transparent_bg(batch, None)
    assert out.shape == (2, 3, 4, 4)
    assert torch.equal(out, batch[:, :3])


def _defer_gate(mode: str):
    """Bare object carrying only what _should_defer_gpu_preprocessing reads."""
    gate = type("_Gate", (), {})()
    gate._image_preprocess_mode = mode
    gate.mm_feature_transport = "cpu"
    gate._processor = SimpleNamespace(
        preprocess_config=SimpleNamespace(
            patch_size=14,
            merge_kernel_size=2,
            in_patch_limit=4096,
            patch_limit_on_one_side=64,
            fixed_output_tokens=None,
        )
    )
    gate._should_defer_gpu_preprocessing = (
        KimiK3ImageProcessor._should_defer_gpu_preprocessing.__get__(gate)
    )
    return gate


def test_explicit_modes_opt_out_of_deferral_that_auto_would_take():
    """mode != auto must short-circuit the deferral gate.

    The gate's own precondition is ``mm_feature_transport == "cpu"`` -- the
    exact transport a CPU run uses -- so under mode="cpu" it would otherwise
    win first and hand preprocessing to the vision-DP owner rank's GPU. The
    request would still preprocess on a GPU, just in the model process, and
    the mode would look like it had no effect.
    """
    images = [Image.fromarray(_natural_image(512, 512))]
    with mock.patch(
        "sglang.srt.multimodal.processors.kimi_k3.is_cuda", return_value=True
    ):
        # Same inputs, only the mode differs.
        assert _defer_gate("auto")._should_defer_gpu_preprocessing(images) is True
        assert _defer_gate("cpu")._should_defer_gpu_preprocessing(images) is False
        assert _defer_gate("gpu")._should_defer_gpu_preprocessing(images) is False


def test_unknown_mode_is_rejected():
    """A typo must fail loudly, not silently fall back to an arm."""
    with envs.SGLANG_K3_IMAGE_PREPROCESS_MODE.override("CPU "):
        assert _resolve_image_preprocess_mode() == "cpu"
    with envs.SGLANG_K3_IMAGE_PREPROCESS_MODE.override("cpu_only"):
        with pytest.raises(ValueError, match="must be one of"):
            _resolve_image_preprocess_mode()


def _use_gpu_probe(mode: str):
    probe = type("_Probe", (), {})()
    probe._image_preprocess_mode = mode
    probe._log_preprocess_backend = False
    probe._has_video = KimiK3GPUProcessorWrapper._has_video
    probe._use_gpu = KimiK3GPUProcessorWrapper._use_gpu.__get__(probe)
    return probe


def test_video_keeps_the_gpu_under_cpu_mode():
    """Video has no CPU preprocessing path.

    ``_cpu_call`` routes through the checkpoint's *image* processor, so
    sending a video item there would turn every video request into an error
    rather than a slower request. mode="cpu" must govern images only.
    """
    video = _KimiK3VisualItem(
        media_type="video",
        frames=(Image.fromarray(_natural_image(64, 64)),),
        source_index=0,
        in_patch_limit=1024,
    )
    image = Image.fromarray(_natural_image(64, 64))
    with mock.patch("torch.cuda.is_available", return_value=True):
        assert _use_gpu_probe("cpu")._use_gpu([image]) is False
        assert _use_gpu_probe("cpu")._use_gpu([video]) is True
        # A mixed request still carries the video item.
        assert _use_gpu_probe("cpu")._use_gpu([image, video]) is True


_PREPROCESS_CONFIG = SimpleNamespace(
    patch_size=14,
    merge_kernel_size=2,
    in_patch_limit=16384,
    patch_limit_on_one_side=512,
    fixed_output_tokens=None,
)


def _budget_gate(budget_bytes):
    gate = type("_Gate", (), {})()
    gate._gpu_preprocess_budget_bytes = budget_bytes
    gate._gpu_preprocess_inflight_bytes = 0
    gate._gpu_preprocess_lock = threading.Lock()
    gate._log_preprocess_backend = False
    gate._processor = SimpleNamespace(preprocess_config=_PREPROCESS_CONFIG)
    gate._reserve = KimiK3ImageProcessor._reserve_gpu_preprocess.__get__(gate)
    gate._release = KimiK3ImageProcessor._release_gpu_preprocess.__get__(gate)
    return gate


def test_auto_budget_counts_concurrent_requests_and_never_waits():
    """The budget is shared by in-flight requests, not checked per request.

    Thirty-two processor workers can each hold a request that fits on its
    own; a per-request check would let all of them onto the GPU at once.
    A request that does not fit goes to the CPU immediately -- and so does
    one too big for the whole budget, which must not be let through alone.
    """
    big = _K3EncodedImage(b"", 8192, 8192, 3)
    one = _estimate_gpu_preprocess_bytes([big], _PREPROCESS_CONFIG)
    gate = _budget_gate(int(one * 1.5))

    first, held = gate._reserve([big])
    assert first is True and held == one
    second, held_second = gate._reserve([big])
    assert second is False and held_second == 0
    gate._release(held)
    third, held_third = gate._reserve([big])
    assert third is True
    gate._release(held_third)
    assert gate._gpu_preprocess_inflight_bytes == 0

    assert gate._reserve([big, big])[0] is False
    assert _budget_gate(None)._reserve([big]) == (None, 0)


def test_cpu_choice_never_defers_to_the_model_process_gpu():
    """use_gpu=False must bypass deferral even when the gate would defer.

    Deferral hands the image to the vision-DP owner rank's GPU, so a request
    the budget sent to the CPU would still preprocess on a GPU.
    """
    seen = {}

    def prepare_image_features(images, use_gpu=None):
        seen["images"], seen["use_gpu"] = images, use_gpu
        return [torch.zeros(1)], [(8, 8)], [{}], [(1, 1, 1)]

    stub = type("_Stub", (), {})()
    stub._processor = SimpleNamespace(
        preprocess_config=_PREPROCESS_CONFIG,
        prepare_image_features=prepare_image_features,
    )
    stub._should_defer_gpu_preprocessing = lambda images: True
    stub._make_artifact = lambda **kwargs: kwargs
    batch = KimiK3ImageProcessor.prepare_artifact_batch.__get__(stub)

    image = Image.new("RGB", (8, 8))
    entry = MediaArtifactInput("sha256:x", "key", Modality.IMAGE, image)
    [artifact] = batch([entry], use_gpu=False)
    assert seen == {"images": [image], "use_gpu": False}
    assert artifact.get("deferred") is None


def _oom_wrapper(oom_fallback: bool):
    calls = []

    def gpu_call(*_args):
        raise torch.OutOfMemoryError("CUDA out of memory")

    def cpu_call(text, images, original_input_ids=None, **kwargs):
        calls.append(images)
        return {"backend": "cpu"}

    w = type("_W", (), {})()
    w._image_preprocess_mode = "gpu"
    w._log_preprocess_backend = False
    w._oom_fallback = oom_fallback
    w._gpu_call, w._cpu_call = gpu_call, cpu_call
    w._has_video = KimiK3GPUProcessorWrapper._has_video
    for name in ("_use_gpu", "_can_retry_on_cpu", "_after_gpu_oom"):
        setattr(w, name, getattr(KimiK3GPUProcessorWrapper, name).__get__(w))
    w.__call__ = KimiK3GPUProcessorWrapper.__call__.__get__(w)
    return w, calls


def test_gpu_oom_is_redone_on_the_cpu_instead_of_failing():
    """A wrong budget must cost latency, not a 500.

    Decoded tensors (nvJPEG output) are handed to the CPU processor as PIL,
    the only image type it takes. Video has no CPU path, so it re-raises
    instead of failing later in a CPU-side error; so does a disabled switch.
    """
    chw = torch.zeros(3, 6, 4, dtype=torch.uint8)
    with mock.patch("torch.cuda.is_available", return_value=True), mock.patch(
        "torch.cuda.empty_cache"
    ):
        wrapper, calls = _oom_wrapper(oom_fallback=True)
        assert wrapper.__call__(text="t", images=[chw]) == {"backend": "cpu"}
        [[retried]] = calls
        assert isinstance(retried, Image.Image) and retried.size == (4, 6)

        off, _ = _oom_wrapper(oom_fallback=False)
        with pytest.raises(torch.OutOfMemoryError):
            off.__call__(text="t", images=[chw])

        video = _KimiK3VisualItem(
            media_type="video", frames=(chw,), source_index=0, in_patch_limit=1024
        )
        with pytest.raises(torch.OutOfMemoryError):
            _oom_wrapper(oom_fallback=True)[0].__call__(text="t", images=[video])


def test_header_probe_reads_geometry_and_rejects_garbage():
    buf = io.BytesIO()
    Image.new("RGBA", (37, 23)).save(buf, format="PNG")
    assert _probe_encoded_image(buf.getvalue()) == _K3EncodedImage(
        buf.getvalue(), 37, 23, 4
    )
    # Undecodable input must fall back to the regular decode path, which owns
    # the client-facing error, rather than fail here.
    assert _probe_encoded_image(b"not an image") is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
