"""CPU-only regression tests, with real PyAV and injected TorchCodec failures."""

import importlib.util
import io
import json
import os
import struct
import sys
import tempfile
import threading
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

import numpy as np

try:
    import av
except ImportError:
    av = None


ROOT = Path(os.environ.get("SGLANG_SOURCE_ROOT", Path(__file__).resolve().parents[4]))
UTILS = ROOT / "python/sglang/srt/utils"


def _load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


register_cpu_ci = _load_file(
    "video_metadata_fallback_ci_register",
    ROOT / "python/sglang/test/ci/ci_register.py",
).register_cpu_ci
register_cpu_ci(est_time=5, suite="base-a-test-cpu")


fallback = _load_file(
    "video_metadata_fallback_under_test", UTILS / "video_metadata_fallback.py"
)


def _fps_error(field="averageFpsFromHeader", token="inf"):
    document = '{\n"averageFps": 400,\n"' + field + '": ' + token + "\n}"
    try:
        json.loads(document)
    except json.JSONDecodeError as error:
        return error
    raise AssertionError("Expected malformed metadata")


def _video_bytes(frame_count=12):
    output = io.BytesIO()
    with av.open(output, mode="w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=30)
        stream.width = 32
        stream.height = 24
        stream.pix_fmt = "yuv420p"
        stream.codec_context.thread_count = 1
        stream.codec_context.max_b_frames = 0
        stream.options = {"preset": "ultrafast"}
        for index in range(frame_count):
            pixels = np.full((24, 32, 3), index * 17 % 256, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = index
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return output.getvalue()


def _zero_timestamps(payload):
    output = bytearray(payload)
    containers = {b"moov", b"trak", b"mdia", b"minf", b"stbl"}

    def walk(start, end):
        while start < end:
            size, kind = struct.unpack_from(">I4s", output, start)
            if kind in containers:
                walk(start + 8, start + size)
            elif kind == b"stts":
                count = struct.unpack_from(">I", output, start + 12)[0]
                for index in range(count):
                    struct.pack_into(">I", output, start + 20 + index * 8, 0)
            start += size

    walk(0, len(output))
    return bytes(output)


@unittest.skipIf(av is None, "Install av==16.1.0 to exercise the optional fallback")
class TestVideoMetadataFallback(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.good_video = _video_bytes()
        cls.bad_timestamps = _zero_timestamps(cls.good_video)

    def setUp(self):
        self.environment = mock.patch.dict(
            os.environ, {"SGLANG_VIDEO_METADATA_FALLBACK": "1"}
        )
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.modules = mock.patch.dict(
            sys.modules,
            {"sglang.srt.utils.video_metadata_fallback": fallback},
        )
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def wrapper(self, constructor):
        codec_module = types.ModuleType("torchcodec.decoders")
        codec_module.VideoDecoder = constructor
        with mock.patch.dict(sys.modules, {"torchcodec.decoders": codec_module}):
            return _load_file("video_decoder_under_test", UTILS / "video_decoder.py")

    def test_match_is_narrow(self):
        for token in ("inf", "-inf", "nan", "-nan"):
            for field in ("averageFps", "averageFpsFromHeader"):
                self.assertTrue(
                    fallback.is_nonfinite_fps_json_error(_fps_error(field, token))
                )
        self.assertFalse(fallback.is_nonfinite_fps_json_error(_fps_error("width")))
        self.assertFalse(
            fallback.is_nonfinite_fps_json_error(_fps_error(token="broken"))
        )
        self.assertFalse(fallback.is_nonfinite_fps_json_error(RuntimeError("inf")))
        error = _fps_error()
        self.assertEqual((error.lineno, error.colno, error.pos), (3, 25, 45))

    def test_decode_real_pixels_without_seeking(self):
        decoder = fallback.SequentialVideoDecoder(self.bad_timestamps)
        self.addCleanup(decoder.close)
        with av.open(io.BytesIO(self.bad_timestamps)) as container:
            frames = list(container.decode(video=0))
        self.assertEqual(len({frame.pts for frame in frames}), 1)
        self.assertEqual(len(decoder), len(frames))
        self.assertTrue(np.isfinite(decoder.avg_fps))
        indices = [11, 0, 5, 5]
        actual = decoder.get_frames_at(indices)
        expected = np.stack(
            [frames[index].to_ndarray(format="rgb24") for index in indices]
        )
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(decoder[-1], expected[0])
        self.assertEqual(decoder.get_frames_at([]).shape, (0, 24, 32, 3))
        with self.assertRaises(IndexError):
            decoder.get_frames_at([12])
        decoder.close()
        with self.assertRaisesRegex(ValueError, "closed"):
            decoder.get_frames_at([0])

    def test_path_source_and_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "video.mp4"
            path.write_bytes(self.bad_timestamps)
            decoder = fallback.SequentialVideoDecoder(str(path))
            self.assertEqual(len(decoder), 12)
            decoder.close()
            path.unlink()

    def test_wrapper_falls_back_per_instance(self):
        constructor = mock.Mock(side_effect=_fps_error())
        wrapper = self.wrapper(constructor)
        with wrapper.VideoDecoderWrapper(self.bad_timestamps) as decoder:
            self.assertEqual(decoder._backend, "pyav")
            self.assertEqual(wrapper._BACKEND, "torchcodec")
            self.assertEqual(decoder.get_frames_at([0, 11]).shape, (2, 24, 32, 3))
            self.assertEqual(decoder[0].shape, (24, 32, 3))
            self.assertGreater(decoder.avg_fps, 0)
            self.assertEqual(decoder.source_bytes, self.bad_timestamps)
            tensor = mock.Mock()
            torch_module = types.ModuleType("torch")
            torch_module.from_numpy = mock.Mock(return_value=tensor)
            with mock.patch.dict(
                sys.modules, {"torch": torch_module}
            ), mock.patch.object(
                decoder,
                "_parallel_decode",
                side_effect=AssertionError("Must remain serial"),
            ):
                result = decoder.get_frames_as_tensor([0, 11], num_threads=16)
                self.assertIs(result, tensor.pin_memory.return_value)
        constructor.assert_called_once()

    def test_cpu_retry_can_fall_back(self):
        wrapper = self.wrapper(
            mock.Mock(side_effect=[RuntimeError("CUDA unavailable"), _fps_error()])
        )
        with mock.patch.object(wrapper, "_try_cuda_backend", return_value=True):
            with wrapper.VideoDecoderWrapper(
                self.bad_timestamps, device="cuda"
            ) as decoder:
                self.assertEqual(decoder._backend, "pyav")
        self.assertEqual(wrapper.VideoDecoder.call_count, 2)

    def test_healthy_torchcodec_unchanged(self):
        native = mock.Mock()
        native.metadata.average_fps = 30
        native.get_frames_at.return_value.data.numpy.return_value = np.zeros(
            (1, 24, 32, 3), dtype=np.uint8
        )
        wrapper = self.wrapper(mock.Mock(return_value=native))
        with mock.patch.object(
            fallback,
            "SequentialVideoDecoder",
            side_effect=AssertionError("No fallback"),
        ):
            with wrapper.VideoDecoderWrapper(self.good_video) as decoder:
                self.assertEqual(decoder._backend, "torchcodec")
                self.assertEqual(decoder.avg_fps, 30)
                self.assertEqual(decoder.get_frames_at([0]).shape, (1, 24, 32, 3))

    def test_next_request_still_uses_torchcodec(self):
        native = mock.Mock()
        native.metadata.average_fps = 30
        wrapper = self.wrapper(mock.Mock(side_effect=[_fps_error(), native]))
        with wrapper.VideoDecoderWrapper(self.bad_timestamps) as first:
            self.assertEqual(first._backend, "pyav")
            with wrapper.VideoDecoderWrapper(self.good_video) as second:
                self.assertEqual(second._backend, "torchcodec")
                self.assertEqual(second.avg_fps, 30)
                self.assertEqual(wrapper._BACKEND, "torchcodec")

    def test_healthy_parallel_decode_unchanged(self):
        wrapper = self.wrapper(mock.Mock(return_value=mock.Mock()))
        with wrapper.VideoDecoderWrapper(self.good_video) as decoder:
            with mock.patch.dict(
                sys.modules, {"torch": types.ModuleType("torch")}
            ), mock.patch.object(
                decoder, "_parallel_decode", return_value="parallel-result"
            ) as parallel:
                self.assertEqual(
                    decoder.get_frames_as_tensor([0, 1, 2, 3], num_threads=2),
                    "parallel-result",
                )
                parallel.assert_called_once_with([0, 1, 2, 3], 2)

    def test_decord_path_and_temp_cleanup_unchanged(self):
        wrapper = self.wrapper(mock.Mock())
        wrapper._BACKEND = "decord"
        native = mock.MagicMock()
        pixels = np.zeros((1, 24, 32, 3), dtype=np.uint8)
        native.__len__.return_value = 12
        native.get_avg_fps.return_value = 30
        native.get_batch.return_value.asnumpy.return_value = pixels
        native.__getitem__.return_value.asnumpy.return_value = pixels[0]
        decord = types.ModuleType("decord")
        decord.VideoReader = mock.Mock(return_value=native)
        decord.cpu = mock.Mock(return_value="cpu")
        with mock.patch.dict(sys.modules, {"decord": decord}):
            with wrapper.VideoDecoderWrapper(self.good_video) as decoder:
                path = Path(decoder._tmp_path)
                self.assertTrue(path.exists())
                self.assertEqual(decoder._backend, "decord")
                self.assertEqual(len(decoder), 12)
                self.assertEqual(decoder.avg_fps, 30)
                np.testing.assert_array_equal(decoder.get_frames_at([0]), pixels)
                np.testing.assert_array_equal(decoder[0], pixels[0])
            self.assertFalse(path.exists())

    def test_other_errors_propagate(self):
        for error in (
            _fps_error("width"),
            RuntimeError("OOM"),
            MemoryError("allocation"),
            ImportError("backend"),
        ):
            wrapper = self.wrapper(mock.Mock(side_effect=error))
            with mock.patch.object(
                fallback,
                "SequentialVideoDecoder",
                side_effect=AssertionError("No fallback"),
            ):
                with self.assertRaises(type(error)) as caught:
                    wrapper.VideoDecoderWrapper(self.bad_timestamps)
                self.assertIs(caught.exception, error)

    def test_disabled_fallback_preserves_error(self):
        error = _fps_error()
        wrapper = self.wrapper(mock.Mock(side_effect=error))
        with mock.patch.dict(os.environ, {"SGLANG_VIDEO_METADATA_FALLBACK": "0"}):
            with self.assertRaises(json.JSONDecodeError) as caught:
                wrapper.VideoDecoderWrapper(self.bad_timestamps)
        self.assertIs(caught.exception, error)

    def test_input_and_decode_limits(self):
        limits = {
            "MAX_INPUT_BYTES": "1",
            "MAX_FRAMES": "1",
            "MAX_FRAME_PIXELS": "1",
            "MAX_TOTAL_PIXELS": "1",
        }
        for setting, value in limits.items():
            with self.subTest(setting=setting), mock.patch.dict(
                os.environ, {f"SGLANG_VIDEO_FALLBACK_{setting}": value}
            ):
                with self.assertRaises(ValueError):
                    fallback.SequentialVideoDecoder(self.bad_timestamps)
        with mock.patch.dict(
            os.environ, {"SGLANG_VIDEO_FALLBACK_MAX_OUTPUT_BYTES": "1"}
        ):
            decoder = fallback.SequentialVideoDecoder(self.bad_timestamps)
            with self.assertRaisesRegex(ValueError, "output-byte"):
                decoder.get_frames_at([0])
            decoder.close()

    def test_invalid_configuration(self):
        for value in ("0", "-1", "inf", "nan"):
            with mock.patch.dict(
                os.environ, {"SGLANG_VIDEO_FALLBACK_TIMEOUT_SECONDS": value}
            ):
                with self.assertRaises(ValueError):
                    fallback.SequentialVideoDecoder(self.bad_timestamps)

    def test_slot_timeout_and_release_after_failure(self):
        semaphore = threading.BoundedSemaphore(1)
        with mock.patch.object(fallback, "_DECODE_SLOTS", semaphore), mock.patch.dict(
            os.environ, {"SGLANG_VIDEO_FALLBACK_TIMEOUT_SECONDS": "0.01"}
        ):
            semaphore.acquire()
            try:
                with self.assertRaisesRegex(ValueError, "waiting for a decode slot"):
                    fallback.SequentialVideoDecoder(self.bad_timestamps)
            finally:
                semaphore.release()
            with self.assertRaises(Exception):
                fallback.SequentialVideoDecoder(b"not a video")
            self.assertTrue(semaphore.acquire(blocking=False))
            semaphore.release()

    def test_cooperative_deadline(self):
        with mock.patch.object(fallback.time, "monotonic", side_effect=[0, 0, 0, 20]):
            with self.assertRaisesRegex(ValueError, "time budget"):
                fallback.SequentialVideoDecoder(self.bad_timestamps)

    def test_concurrent_instances(self):
        def decode():
            decoder = fallback.SequentialVideoDecoder(self.bad_timestamps)
            try:
                return decoder.get_frames_at([0, 11])
            finally:
                decoder.close()

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(lambda _: decode(), range(8)))
        for result in results:
            np.testing.assert_array_equal(result, results[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
