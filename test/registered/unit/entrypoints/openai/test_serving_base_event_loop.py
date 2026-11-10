"""Event-loop responsiveness tests for OpenAI request preprocessing."""

import asyncio
import threading
import time
import unittest
from types import SimpleNamespace

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase  # noqa: E402
from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.managers.tokenizer_manager import TokenizerManager  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _BlockingServing(OpenAIServingBase):
    def __init__(self, conversion_started, release_conversion):
        tokenizer_manager = TokenizerManager.__new__(TokenizerManager)
        if hasattr(tokenizer_manager, "init_request_preprocessor"):
            tokenizer_manager.init_request_preprocessor()
        tokenizer_manager.request_logger = SimpleNamespace(
            log_requests=False, log_requests_level=0
        )
        tokenizer_manager.server_args = SimpleNamespace(
            tokenizer_metrics_allowed_custom_labels=None,
        )
        super().__init__(tokenizer_manager)
        self.conversion_started = conversion_started
        self.release_conversion = release_conversion

    def _request_id_prefix(self):
        return "test-"

    def _convert_to_internal_request(self, request, raw_request=None):
        self.conversion_started.set()
        if not self.release_conversion.wait(timeout=5):
            raise TimeoutError("test did not release request conversion")
        return object(), request

    async def _handle_non_streaming_request(
        self, adapted_request, request, raw_request
    ):
        return "ok"


class TestServingBaseEventLoop(CustomTestCase):
    def test_request_conversion_does_not_block_event_loop(self):
        conversion_started = threading.Event()
        loop_progressed = threading.Event()
        release_conversion = threading.Event()
        loop_was_starved = threading.Event()
        conversion_did_not_start = threading.Event()
        serving = _BlockingServing(conversion_started, release_conversion)

        def release_after_loop_progress():
            if not conversion_started.wait(timeout=2):
                conversion_did_not_start.set()
                release_conversion.set()
                return
            if not loop_progressed.wait(timeout=1):
                loop_was_starved.set()
            release_conversion.set()

        watchdog = threading.Thread(target=release_after_loop_progress)
        watchdog.start()

        async def run_request_and_probe_loop():
            request = SimpleNamespace(stream=False)
            request_task = asyncio.create_task(
                serving.handle_request(request, object())
            )
            await asyncio.sleep(0)
            loop_progressed.set()
            return await request_task

        try:
            result = asyncio.run(run_request_and_probe_loop())
        finally:
            release_conversion.set()
            watchdog.join(timeout=2)
            executor = getattr(
                serving.tokenizer_manager, "_request_preprocessor_executor", None
            )
            if executor is not None:
                executor.shutdown(wait=True)

        self.assertEqual(result, "ok")
        self.assertFalse(conversion_did_not_start.is_set())
        self.assertFalse(watchdog.is_alive())
        self.assertFalse(
            loop_was_starved.is_set(),
            "synchronous request conversion blocked the API event loop",
        )


class _SlowConversionServing(_BlockingServing):
    """Conversion takes ``convert_s`` on the preprocessing thread."""

    def __init__(self, convert_s=0.0):
        release = threading.Event()
        release.set()
        super().__init__(threading.Event(), release)
        self.convert_s = convert_s

    def _convert_to_internal_request(self, request, raw_request=None):
        time.sleep(self.convert_s)
        return object(), request


class TestSlowConversionWarning(CustomTestCase):
    logger_name = "sglang.srt.entrypoints.openai.serving_base"

    def _run(self, serving, busy_thread_s=0.0):
        async def run():
            if busy_thread_s:
                # Occupy the single preprocessing thread first: the request's
                # conversion then waits in the queue behind this job.
                busy = asyncio.ensure_future(
                    serving.tokenizer_manager.run_in_request_preprocessor(
                        time.sleep, busy_thread_s
                    )
                )
                await asyncio.sleep(0)
            request = SimpleNamespace(stream=False, rid="rid-1")
            result = await serving.handle_request(request, object())
            if busy_thread_s:
                await busy
            return result

        try:
            return asyncio.run(run())
        finally:
            executor = getattr(
                serving.tokenizer_manager, "_request_preprocessor_executor", None
            )
            if executor is not None:
                executor.shutdown(wait=True)

    def test_slow_thread_conversion_is_reported(self):
        with envs.SGLANG_LOG_SLOW_PREPROCESSING_MS.override(10):
            with self.assertLogs(self.logger_name, level="WARNING") as logs:
                self.assertEqual(self._run(_SlowConversionServing(0.05)), "ok")
        self.assertEqual(len(logs.output), 1)
        self.assertIn("conversion on the in-process thread", logs.output[0])
        self.assertIn("rid=rid-1", logs.output[0])

    def test_queueing_is_not_counted(self):
        with envs.SGLANG_LOG_SLOW_PREPROCESSING_MS.override(10):
            with self.assertNoLogs(self.logger_name, level="WARNING"):
                result = self._run(_SlowConversionServing(0.0), busy_thread_s=0.1)
        self.assertEqual(result, "ok")

    def test_fast_or_disabled_is_silent(self):
        for threshold in (1000, 0):
            with envs.SGLANG_LOG_SLOW_PREPROCESSING_MS.override(threshold):
                with self.assertNoLogs(self.logger_name, level="WARNING"):
                    self.assertEqual(self._run(_SlowConversionServing(0.02)), "ok")


if __name__ == "__main__":
    unittest.main()
