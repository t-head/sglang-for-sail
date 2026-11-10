"""Tests for the process-mode request preprocessor (SGLANG_REQUEST_PREPROCESSOR_MODE=process)."""

import asyncio
import functools
import json
import os
import re
import shutil
import signal
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from fastapi import HTTPException  # noqa: E402
from starlette.datastructures import Headers  # noqa: E402

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest  # noqa: E402
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat  # noqa: E402
from sglang.srt.managers import process_request_preprocessor as prp  # noqa: E402
from sglang.srt.managers.tokenizer_manager import TokenizerManager  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

_READY_TIMEOUT_S = 120


def _chat_request(n_msgs=2, content="hello world", **kwargs) -> ChatCompletionRequest:
    messages = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"{i} {content}"}
        for i in range(n_msgs)
    ]
    messages[-1]["role"] = "user"
    return ChatCompletionRequest(
        model="tiny", messages=messages, max_tokens=16, stream=True, **kwargs
    )


class _RawRequest:
    def __init__(self, headers, client=("10.0.0.1", 1234)):
        self.headers = Headers(headers=headers)
        self.client = client


# ----------------------------------------------------------------------------
# Handlers used by the end-to-end tests. Module level so spawned children can
# import them by reference.
# ----------------------------------------------------------------------------


class ClientReadingServingChat(OpenAIServingChat):
    """Reads raw_request.client, which a preprocessor child does not have."""

    def _convert_to_internal_request(self, request, raw_request=None):
        _ = raw_request.client
        return super()._convert_to_internal_request(request, raw_request)


_GIL_PATTERN = re.compile(r" ?\w+| ?[^\s\w]+|\s+")


@functools.lru_cache(maxsize=1)
def _gil_text() -> str:
    return " ".join(f"w{i % 9973}x{i % 7}" for i in range(2_000_000))


def _hold_gil() -> None:
    # One long C call that keeps the GIL (~100+ ms), like a pre-tokenizer regex
    # over a long prompt.
    _GIL_PATTERN.findall(_gil_text())


class GilHeavyServingChat(OpenAIServingChat):
    def _convert_to_internal_request(self, request, raw_request=None):
        _hold_gil()
        return super()._convert_to_internal_request(request, raw_request)


class FailingTokenizerManager(TokenizerManager):
    @classmethod
    def create_for_request_preprocessing(cls, server_args):
        raise RuntimeError("tokenizer unavailable in this test")


# ----------------------------------------------------------------------------
# Helpers without processes
# ----------------------------------------------------------------------------


class TestHelpers(CustomTestCase):
    def test_token_ids_round_trip(self):
        long_ids = list(range(1000))
        self.assertEqual(prp.pack_token_ids(long_ids)[0], "array")
        self.assertEqual(prp.unpack_token_ids(prp.pack_token_ids(long_ids)), long_ids)
        short = [1, 2, 3]
        self.assertEqual(prp.pack_token_ids(short), ("raw", short))
        batch = [list(range(500)), [7]]
        self.assertEqual(prp.unpack_token_ids(prp.pack_token_ids(batch)), batch)
        mixed = ["a"] * 1000
        self.assertEqual(prp.pack_token_ids(mixed), ("raw", mixed))
        self.assertEqual(prp.unpack_token_ids(prp.pack_token_ids(None)), None)

    def test_snapshot_headers(self):
        self.assertIsNone(prp.snapshot_headers(None))
        raw = [(b"x-a", b"1"), (b"x-a", b"2"), (b"x-b", b"3")]
        request = SimpleNamespace(headers=Headers(raw=raw))
        self.assertEqual(prp.snapshot_headers(request), raw)
        mapping = SimpleNamespace(headers={"X-Smg-Routing-Key": "k"})
        self.assertEqual(prp.snapshot_headers(mapping), [(b"x-smg-routing-key", b"k")])
        self.assertEqual(prp.snapshot_headers(object()), [])

    def test_request_changes_replay(self):
        request = _chat_request(
            chat_template_kwargs={"reasoning_effort": "low", "k": 1}
        )
        expected = request.model_copy(deep=True)
        snapshot = prp._snapshot_request_state(request)

        # What conversions do: in-place dict edits, reassignments, same-value sets.
        request.chat_template_kwargs.pop("reasoning_effort")
        request.reasoning_effort = "low"
        request.skip_special_tokens = request.skip_special_tokens
        for target in (expected,):
            target.chat_template_kwargs.pop("reasoning_effort")
            target.reasoning_effort = "low"
            target.skip_special_tokens = target.skip_special_tokens

        changes = prp._collect_request_changes(request, snapshot)
        self.assertNotIn("messages", changes)
        replayed = _chat_request(
            chat_template_kwargs={"reasoning_effort": "low", "k": 1}
        )
        for name, value in changes.items():
            setattr(replayed, name, value)
        self.assertEqual(replayed.model_dump(), expected.model_dump())
        self.assertEqual(replayed.model_fields_set, expected.model_fields_set)

    def test_rebound_messages_are_reported(self):
        request = _chat_request()
        snapshot = prp._snapshot_request_state(request)
        request.messages = list(request.messages)
        self.assertIn("messages", prp._collect_request_changes(request, snapshot))

    def test_exceptions_cross_the_boundary_with_their_type(self):
        http = prp._local_exception(
            prp._portable_exception(HTTPException(status_code=400, detail="bad"))
        )
        self.assertIsInstance(http, HTTPException)
        self.assertEqual((http.status_code, http.detail), (400, "bad"))

        value_error = ValueError("invalid")
        self.assertIs(prp._portable_exception(value_error), value_error)

        class Unpicklable(ValueError):
            def __init__(self, a, b):
                super().__init__(f"{a}-{b}")

        carried = prp._local_exception(prp._portable_exception(Unpicklable(1, 2)))
        self.assertIs(type(carried), ValueError)
        self.assertEqual(str(carried), "1-2")

        class Opaque(Exception):
            def __init__(self, a, b):
                super().__init__(f"{a}-{b}")

        carried = prp._local_exception(prp._portable_exception(Opaque(1, 2)))
        self.assertIsInstance(carried, RuntimeError)
        self.assertIn("1-2", str(carried))

    def test_child_convert_reports_fallback_and_errors(self):
        class Handler:
            def __init__(self, behaviour):
                self.behaviour = behaviour

            def _convert_to_internal_request(self, request, raw_request):
                if self.behaviour == "client":
                    return raw_request.client
                if self.behaviour == "error":
                    raise ValueError("nope")
                adapted = SimpleNamespace(input_ids=list(range(600)))
                request.reasoning_effort = "high"
                return adapted, request

        handlers = {}
        state = SimpleNamespace(handler=lambda cls: handlers[cls])
        with mock.patch.object(prp, "_CHILD_STATE", state):
            for behaviour in ("client", "error", "ok"):
                cls = type(f"H_{behaviour}", (Handler,), {})
                handlers[cls] = Handler(behaviour)
                result = prp._child_convert(cls, _chat_request(), [(b"a", b"b")])
                if behaviour == "client":
                    self.assertEqual(result[0], "fallback")
                elif behaviour == "error":
                    self.assertEqual(result[0], "error")
                    self.assertIsInstance(result[1], ValueError)
                else:
                    status, adapted, packed, (kind, changes) = result
                    self.assertEqual(status, "ok")
                    self.assertIsNone(adapted.input_ids)
                    self.assertEqual(prp.unpack_token_ids(packed), list(range(600)))
                    self.assertEqual(
                        (kind, changes), ("changes", {"reasoning_effort": "high"})
                    )

    def test_convert_before_ready_warns_once_and_declines(self):
        class Opted:
            pass

        class NotOpted:
            pass

        pre = prp.ProcessRequestPreprocessor.__new__(prp.ProcessRequestPreprocessor)
        pre._pool = object()  # started, warmup not finished
        pre._ready = False
        pre._disabled_reason = None
        pre._handler_classes = (Opted,)
        pre._warned = set()

        async def convert_all():
            return [
                await pre.convert(handler, _chat_request(), None)
                for handler in (Opted(), Opted(), NotOpted())
            ]

        with self.assertLogs(prp.logger, level="WARNING") as logs:
            self.assertEqual(asyncio.run(convert_all()), [None, None, None])
        self.assertEqual(len(logs.output), 1)
        self.assertIn("not ready yet", logs.output[0])


# ----------------------------------------------------------------------------
# End to end with spawned children and a tiny local model
# ----------------------------------------------------------------------------


def _make_tiny_model(path: str) -> None:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=600,
        special_tokens=["<unk>", "<s>", "</s>", "<|im_start|>", "<|im_end|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    corpus = ["hello world, a tiny tokenizer for tests; def f(x): return x"] * 50
    tokenizer.train_from_iterator(corpus, trainer)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="</s>",
    )
    fast.chat_template = (
        "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}"
        "<|im_end|>\n{% endfor %}{% if tools %}<|im_start|>tools\n{{ tools | tojson }}"
        "<|im_end|>\n{% endif %}{% if add_generation_prompt %}<|im_start|>assistant\n"
        "{% endif %}"
    )
    fast.save_pretrained(path)
    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "num_hidden_layers": 2,
        "vocab_size": 600,
        "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-5,
        "torch_dtype": "bfloat16",
    }
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(config, f)


def _wait(predicate, timeout=_READY_TIMEOUT_S):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.1)
    return False


class TestProcessRequestPreprocessorEndToEnd(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.srt.parser.template_manager import TemplateManager
        from sglang.srt.runtime_context import get_model, get_serving, publish
        from sglang.srt.server_args import ServerArgs

        cls.model_dir = tempfile.mkdtemp(prefix="sglang-tiny-")
        _make_tiny_model(cls.model_dir)
        cls.env = mock.patch.dict(
            os.environ,
            {
                "SGLANG_REQUEST_PREPROCESSOR_MODE": "process",
                "SGLANG_REQUEST_PREPROCESSOR_PROCESSES": "1",
            },
        )
        cls.env.start()
        server_args = ServerArgs(
            model_path=cls.model_dir, device="cpu", skip_server_warmup=True
        )
        publish(server_args, role="tokenizer")
        cls.tokenizer_manager = TokenizerManager.create_for_request_preprocessing(
            server_args
        )
        cls.tokenizer_manager.init_request_preprocessor()
        cls.template_manager = TemplateManager()
        cls.template_manager.initialize_templates(
            tokenizer_manager=cls.tokenizer_manager,
            model_path=get_model().model_path,
            chat_template=get_serving().chat_template,
            completion_template=get_serving().completion_template,
        )
        cls.handler_classes = (
            OpenAIServingChat,
            ClientReadingServingChat,
            GilHeavyServingChat,
        )
        cls.tokenizer_manager.maybe_start_process_request_preprocessor(
            cls.template_manager, handler_classes=cls.handler_classes
        )
        cls.preprocessor = cls.tokenizer_manager.process_request_preprocessor
        assert _wait(lambda: cls.preprocessor.ready), cls.preprocessor._disabled_reason

    @classmethod
    def tearDownClass(cls):
        cls.preprocessor.shutdown()
        cls.tokenizer_manager._request_preprocessor_executor.shutdown(wait=True)
        cls.env.stop()
        shutil.rmtree(cls.model_dir, ignore_errors=True)

    def _handler(self, cls=OpenAIServingChat):
        return cls(self.tokenizer_manager, self.template_manager)

    def test_matches_in_process_conversion(self):
        handler = self._handler()
        raw_request = _RawRequest(
            {"x-smg-routing-key": "rk", "x-data-parallel-rank": "2"}
        )
        tools = [
            {
                "type": "function",
                "function": {
                    "name": f"tool_{i}",
                    "parameters": {
                        "type": "object",
                        "properties": {"x": {"type": "string"}},
                    },
                },
            }
            for i in range(3)
        ]
        variants = [
            {},
            {"tools": tools, "tool_choice": "auto"},
            {"chat_template_kwargs": {"reasoning_effort": "low", "foo": 1}},
            {"n_msgs": 300, "content": "long " * 50},
        ]

        async def run():
            for kwargs in variants:
                in_process = _chat_request(**kwargs)
                out_of_process = in_process.model_copy(deep=True)
                expected, expected_request = handler._convert_to_internal_request(
                    in_process, raw_request
                )
                adapted, processed = await handler._run_request_conversion(
                    out_of_process, raw_request
                )
                self.assertIs(processed, out_of_process)
                self.assertEqual(adapted.__dict__, expected.__dict__, kwargs)
                self.assertEqual(processed.model_dump(), expected_request.model_dump())
                self.assertEqual(
                    processed.model_fields_set, expected_request.model_fields_set
                )

        asyncio.run(run())

    def test_conversion_errors_keep_their_type(self):
        handler = self._handler()

        async def run():
            with self.assertRaises(HTTPException) as ctx:
                await handler._run_request_conversion(
                    _chat_request(), _RawRequest({"x-data-parallel-rank": "abc"})
                )
            self.assertEqual(ctx.exception.status_code, 400)

        asyncio.run(run())

    def test_non_header_access_falls_back_to_thread(self):
        handler = self._handler(ClientReadingServingChat)
        raw_request = _RawRequest({})

        async def run():
            with self.assertLogs(prp.logger, level="WARNING") as logs:
                adapted, _ = await handler._run_request_conversion(
                    _chat_request(), raw_request
                )
            self.assertTrue(any("raw_request.client" in m for m in logs.output))
            return adapted

        adapted = asyncio.run(run())
        self.assertGreater(len(adapted.input_ids), 0)

    def test_event_loop_stays_responsive(self):
        handler = self._handler(GilHeavyServingChat)
        _gil_text()  # build the input outside the timed region
        started = time.perf_counter()
        _hold_gil()
        gil_hold_s = time.perf_counter() - started

        async def max_loop_lag(convert):
            lags = []
            done = asyncio.Event()

            async def ticker():
                while not done.is_set():
                    t = time.perf_counter()
                    await asyncio.sleep(0.001)
                    lags.append(time.perf_counter() - t - 0.001)

            task = asyncio.create_task(ticker())
            await asyncio.sleep(0.05)
            lags.clear()
            await convert()
            done.set()
            await task
            return max(lags)

        async def run():
            request = _chat_request()
            in_thread = await max_loop_lag(
                lambda: self.tokenizer_manager.run_in_request_preprocessor(
                    handler._convert_to_internal_request, request, _RawRequest({})
                )
            )
            in_process = await max_loop_lag(
                lambda: handler._run_request_conversion(
                    _chat_request(), _RawRequest({})
                )
            )
            return in_thread, in_process

        in_thread, in_process = asyncio.run(run())
        # Sanity check of the setup: on the thread the GIL-holding call stalls
        # the loop for about its whole duration.
        self.assertGreater(in_thread, gil_hold_s / 2, (in_thread, gil_hold_s))
        self.assertLess(in_process, gil_hold_s / 2, (in_process, gil_hold_s))

    def test_broken_pool_restarts(self):
        handler = self._handler()
        pool = self.preprocessor._pool
        for pid in list(pool._processes):
            os.kill(pid, signal.SIGKILL)

        async def convert():
            return await handler._run_request_conversion(_chat_request(), None)

        adapted, _ = asyncio.run(convert())  # served by the thread fallback
        self.assertGreater(len(adapted.input_ids), 0)
        self.assertTrue(_wait(lambda: self.preprocessor._pool is not pool))
        self.assertTrue(_wait(lambda: self.preprocessor.ready))
        adapted, _ = asyncio.run(convert())
        self.assertGreater(len(adapted.input_ids), 0)


class TestProcessRequestPreprocessorDisables(TestProcessRequestPreprocessorEndToEnd):
    """Startup failures disable the preprocessor instead of failing requests."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    def _start(self, tokenizer_manager, template_manager):
        return prp.ProcessRequestPreprocessor(
            tokenizer_manager=tokenizer_manager,
            template_manager=template_manager,
            handler_classes=(OpenAIServingChat,),
            num_processes=1,
        )

    def test_fingerprint_mismatch_disables(self):
        template_manager = SimpleNamespace(
            chat_template_name="something-else",
            completion_template_name=None,
            jinja_template_content_format=None,
        )
        preprocessor = self._start(self.tokenizer_manager, template_manager)
        try:
            self.assertTrue(_wait(lambda: preprocessor._disabled_reason is not None))
            self.assertIn("chat_template_name", preprocessor._disabled_reason)
            self.assertFalse(preprocessor.accepts(self._handler()))
        finally:
            preprocessor.shutdown()

    def test_child_init_failure_disables(self):
        manager = FailingTokenizerManager.__new__(FailingTokenizerManager)
        manager.server_args = self.tokenizer_manager.server_args
        manager.tokenizer = self.tokenizer_manager.tokenizer
        preprocessor = self._start(manager, self.template_manager)
        try:
            self.assertTrue(_wait(lambda: preprocessor._disabled_reason is not None))
            self.assertIn("failed to start", preprocessor._disabled_reason)
        finally:
            preprocessor.shutdown()

    # The inherited end-to-end tests already ran on the parent class.
    test_matches_in_process_conversion = None
    test_conversion_errors_keep_their_type = None
    test_non_header_access_falls_back_to_thread = None
    test_event_loop_stays_responsive = None
    test_broken_pool_restarts = None


if __name__ == "__main__":
    unittest.main()
