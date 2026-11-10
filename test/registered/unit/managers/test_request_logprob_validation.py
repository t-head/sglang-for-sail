"""Reject unsafe request sizes before dispatch and before streaming headers."""

import argparse
import asyncio
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    CustomTestCase,
    enter_scope,
    maybe_stub_sgl_kernel,
    published_topology,
)

maybe_stub_sgl_kernel()

from sglang.srt.arg_groups.arg_utils import add_cli_args_from_dataclass  # noqa: E402
from sglang.srt.disaggregation.base import KVPoll  # noqa: E402
from sglang.srt.disaggregation.prefill import (  # noqa: E402
    SchedulerDisaggregationPrefillMixin,
)
from sglang.srt.disaggregation.utils import (  # noqa: E402
    MAX_DISAGGREGATION_TOP_LOGPROBS,
    DisaggregationMode,
    InvalidDisaggregationMetadata,
    MetadataBuffers,
)
from sglang.srt.entrypoints import http_server  # noqa: E402
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat  # noqa: E402
from sglang.srt.entrypoints.openai.serving_completions import (  # noqa: E402
    OpenAIServingCompletion,
)
from sglang.srt.entrypoints.openai.serving_responses import (  # noqa: E402
    OpenAIServingResponses,
)
from sglang.srt.managers.io_struct import (  # noqa: E402
    EmbeddingReqInput,
    GenerateReqInput,
)
from sglang.srt.managers.request_validation import (  # noqa: E402
    validate_request_limits,
)
from sglang.srt.managers.schedule_batch import FINISH_ABORT  # noqa: E402
from sglang.srt.managers.tokenizer_manager import TokenizerManager  # noqa: E402
from sglang.srt.runtime_context import get_context  # noqa: E402
from sglang.srt.sampling.sampling_mask import SamplingMaskRows  # noqa: E402
from sglang.srt.sampling.sampling_params import SamplingParams  # noqa: E402
from sglang.srt.server_args import ServerArgs  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def make_manager(mode=DisaggregationMode.PREFILL, vocab_size=256):
    manager = TokenizerManager.__new__(TokenizerManager)
    manager.model_config = SimpleNamespace(vocab_size=vocab_size)
    manager.disaggregation_mode = mode
    manager.server_args = SimpleNamespace(stream_response_default_include_usage=False)
    manager.preferred_sampling_params = {}
    manager.sampling_params_class = SamplingParams
    manager.auto_create_handle_loop = Mock()
    manager._init_req_state = Mock()
    manager._send_one_request = Mock()
    return manager


def make_mask_rows(ids):
    rows = SamplingMaskRows()
    rows.append(np.asarray(ids, dtype=np.int32), np.asarray([-0.5], dtype=np.float32))
    return rows


class TestRequestValidation(CustomTestCase):
    def setUp(self):
        super().setUp()
        override = get_context().override_server_args(
            max_parallel_samples=128,
            max_batch_outputs=1024,
            speculative_algorithm=None,
            sampling_backend="pytorch",
        )
        override.install()
        self.addCleanup(override.restore)
        env_patch = patch.dict(
            os.environ,
            {
                "SGLANG_ENABLE_DISAGG_SAMPLING_MASK": "0",
                "SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES": "0",
            },
        )
        env_patch.start()
        self.addCleanup(env_patch.stop)

    def test_pd_boundaries_and_batch_values(self):
        for mode in (DisaggregationMode.PREFILL, DisaggregationMode.DECODE):
            manager = make_manager(mode)
            for value in (None, 0, 1, 128, [0, 128]):
                with self.subTest(mode=mode, value=value):
                    manager.validate_logprob_params(
                        GenerateReqInput(
                            text=["a", "b"] if isinstance(value, list) else "hello",
                            top_logprobs_num=value,
                        )
                    )
            for value in (129, 130, 10**9, [1, 130]):
                with self.subTest(mode=mode, value=value):
                    with self.assertRaisesRegex(ValueError, "at most 128"):
                        manager.validate_logprob_params(
                            GenerateReqInput(
                                text=["a", "b"] if isinstance(value, list) else "hello",
                                top_logprobs_num=value,
                            )
                        )

    def test_non_pd_uses_vocab_limit(self):
        manager = make_manager(DisaggregationMode.NULL)
        for value in (130, 256, [0, 256]):
            manager.validate_logprob_params(
                GenerateReqInput(
                    text=["a", "b"] if isinstance(value, list) else "hello",
                    top_logprobs_num=value,
                )
            )
        with self.assertRaisesRegex(ValueError, "at most 256"):
            manager.validate_logprob_params(
                GenerateReqInput(text="hello", top_logprobs_num=257)
            )

    def test_pd_also_respects_small_vocab(self):
        manager = make_manager(vocab_size=64)
        manager.validate_logprob_params(GenerateReqInput(top_logprobs_num=64))
        with self.assertRaisesRegex(ValueError, "at most 64"):
            manager.validate_logprob_params(GenerateReqInput(top_logprobs_num=65))

    def test_invalid_types_and_negative_values(self):
        manager = make_manager()
        for value in (-1, True, 1.5, "130", [0, -1], [None]):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "non-negative integer"):
                    manager.validate_logprob_params(
                        GenerateReqInput(top_logprobs_num=value)
                    )

    def test_rejects_logprob_list_shape_mismatches(self):
        manager = make_manager()
        for text, values in (("hello", [1]), (["a", "b"], [1]), (["a"], [])):
            with self.subTest(text=text, values=values):
                with self.assertRaisesRegex(ValueError, "top_logprobs_num must"):
                    manager.validate_logprob_params(
                        GenerateReqInput(text=text, top_logprobs_num=values)
                    )
        manager.validate_logprob_params(
            GenerateReqInput(input_ids=[[1], [2]], top_logprobs_num=[1, 2])
        )

    def test_embedding_requests_are_unaffected(self):
        make_manager().validate_logprob_params(EmbeddingReqInput(text="hello"))

    def test_rejects_before_normalization_or_any_batch_dispatch(self):
        manager = make_manager()
        for value in (130, [1, 130]):
            obj = GenerateReqInput(
                text=["valid", "invalid"],
                top_logprobs_num=value,
                return_logprob=True,
            )
            with patch.object(obj, "normalize_batch_and_arguments") as normalize:
                with self.assertRaisesRegex(ValueError, "at most 128"):
                    asyncio.run(manager.generate_request(obj).__anext__())
                normalize.assert_not_called()
        manager._init_req_state.assert_not_called()
        manager._send_one_request.assert_not_called()

    def test_metadata_allocation_and_validation_share_capacity(self):
        import torch

        with (
            patch("sglang.srt.disaggregation.utils.is_npu", return_value=False),
            patch(
                "sglang.srt.disaggregation.utils.envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL.get",
                return_value=None,
            ),
        ):
            buffers = MetadataBuffers(
                size=1,
                hidden_size=1,
                hidden_states_dtype=torch.float32,
                max_sampling_mask_tokens=0,
            )
        self.assertEqual(
            buffers.output_top_logprobs_val.shape[1], MAX_DISAGGREGATION_TOP_LOGPROBS
        )
        self.assertEqual(
            buffers.output_top_logprobs_idx.shape[1], MAX_DISAGGREGATION_TOP_LOGPROBS
        )

    def test_native_http_400_including_stream_and_valid_followup(self):
        manager = make_manager()

        async def mock_inference(obj, request):
            manager.validate_request_params(obj)
            yield {"text": "ok", "meta_info": {}}

        manager.generate_request = mock_inference
        manager.create_abort_task = Mock(return_value=None)
        app = FastAPI()

        @app.post("/generate")
        async def generate(request: Request):
            return await http_server.generate_request(
                GenerateReqInput(**await request.json()), request
            )

        with (
            patch.object(
                http_server, "_global_state", SimpleNamespace(tokenizer_manager=manager)
            ),
            TestClient(app) as client,
        ):
            for stream in (False, True):
                bad = client.post(
                    "/generate",
                    json={"text": "hello", "top_logprobs_num": 130, "stream": stream},
                )
                self.assertEqual(bad.status_code, 400, bad.text)
                self.assertIn("at most 128", bad.json()["error"]["message"])
                self.assertIn("application/json", bad.headers["content-type"])
                good = client.post(
                    "/generate",
                    json={"text": "hello", "top_logprobs_num": 128, "stream": stream},
                )
                self.assertEqual(good.status_code, 200, good.text)
                self.assertIn("ok", good.text)

    def test_openai_chat_and_completion_return_400_before_streaming(self):
        for cls in (OpenAIServingChat, OpenAIServingCompletion):
            for stream in (False, True):
                with self.subTest(handler=cls.__name__, stream=stream):
                    manager = make_manager()
                    serving = cls.__new__(cls)
                    serving.tokenizer_manager = manager
                    obj = GenerateReqInput(
                        text="hello", return_logprob=True, top_logprobs_num=130
                    )
                    request = SimpleNamespace(
                        stream_options=None,
                        return_input_ids_in_sglext=False,
                        return_output_ids_in_sglext=False,
                    )
                    handler = (
                        serving._handle_streaming_request
                        if stream
                        else serving._handle_non_streaming_request
                    )
                    response = asyncio.run(handler(obj, request, None))
                    self.assertEqual(response.status_code, 400)
                    self.assertIn(b"at most 128", response.body)
                    manager._init_req_state.assert_not_called()
                    manager._send_one_request.assert_not_called()

    def test_responses_rejects_before_background_acknowledgment(self):
        cases = (
            (130, {}, [1], b"at most 128"),
            (0, {"n": 129}, [1], b"n must be an integer"),
            (0, {}, [-1], b"input_ids"),
        )
        for background in (False, True):
            for top_logprobs, params, ids, error in cases:
                manager = make_manager()
                manager.tokenizer = None
                manager.num_reserved_tokens = 0
                serving = OpenAIServingResponses.__new__(OpenAIServingResponses)
                serving.tokenizer_manager = manager
                serving.use_harmony = False
                serving.tool_server = None
                serving.default_sampling_params = {}
                # Background requests need the in-memory response store.
                serving.enable_response_store = True
                serving._validate_model = Mock(return_value=None)
                processed = SimpleNamespace(
                    require_reasoning=False,
                    reasoning_end_token_ids=None,
                    skip_special_tokens=True,
                    stop=None,
                    tool_call_constraint=None,
                    image_data=None,
                    video_data=None,
                    audio_data=None,
                    modalities=None,
                )
                serving._make_request = AsyncMock(
                    return_value=([], ["hello"], [ids], processed)
                )
                serving._generate_with_builtin_tools = Mock()
                request = SimpleNamespace(
                    model="test",
                    request_id="resp_validation",
                    tool_choice="auto",
                    tools=None,
                    stream=False,
                    background=background,
                    store=True,
                    previous_response_id=None,
                    top_logprobs=top_logprobs,
                    stop=None,
                    session_id=None,
                    extra_key=None,
                    cache_salt=None,
                    is_include_output_logprobs=lambda: True,
                    effective_tool_choice=lambda: "auto",
                    routed_dp_rank=None,
                    bootstrap_host=None,
                    bootstrap_port=None,
                    bootstrap_room=None,
                    disagg_prefill_dp_rank=None,
                    priority=None,
                    to_sampling_params=Mock(return_value=params),
                )
                response = asyncio.run(serving.create_responses(request))
                self.assertEqual(response.status_code, 400)
                self.assertIn(error, response.body)
                serving._generate_with_builtin_tools.assert_not_called()

    def test_n_and_total_output_boundaries(self):
        manager = make_manager()
        for n in (1, 128):
            manager.validate_request_params(
                GenerateReqInput(text="hello", sampling_params={"n": n})
            )
        for n in (0, -1, 129, 5_000_000, True, 1.5, "2", None):
            with self.subTest(n=n):
                with self.assertRaisesRegex(ValueError, "n must be an integer"):
                    manager.validate_request_params(
                        GenerateReqInput(text="hello", sampling_params={"n": n})
                    )
        manager.validate_request_params(
            GenerateReqInput(text=["hello"] * 8, sampling_params={"n": 128})
        )
        for batch, n in ((9, 128), (1025, 1)):
            with self.assertRaisesRegex(ValueError, "max-batch-outputs"):
                manager.validate_request_params(
                    GenerateReqInput(text=["hello"] * batch, sampling_params={"n": n})
                )
        with self.assertRaisesRegex(ValueError, "same for all"):
            manager.validate_request_params(
                GenerateReqInput(text=["a", "b"], sampling_params=[{"n": 1}, {"n": 2}])
            )

    def test_mask_list_respects_beam_search_without_parallel_expansion(self):
        manager = make_manager()
        # n counts returned beam sequences, but normalization keeps one request
        # per input. A disabled mask list must remain valid in this case.
        manager.validate_request_params(
            GenerateReqInput(
                text=["a", "b"],
                sampling_params={"n": 2, "beam_width": 2},
                return_sampling_mask=[False, False],
            )
        )
        with self.assertRaisesRegex(ValueError, "Cannot use list return_sampling_mask"):
            manager.validate_request_params(
                GenerateReqInput(
                    text=["a", "b"],
                    sampling_params={"n": 2},
                    return_sampling_mask=[False, False],
                )
            )

    def test_server_limits_are_configurable_and_cli_visible(self):
        parser = argparse.ArgumentParser()
        add_cli_args_from_dataclass(parser, ServerArgs)
        defaults = parser.parse_args(["--model-path", "dummy"])
        self.assertEqual(defaults.max_parallel_samples, 128)
        self.assertEqual(defaults.max_batch_outputs, 1024)
        args = parser.parse_args(
            [
                "--model-path",
                "dummy",
                "--max-parallel-samples",
                "2",
                "--max-batch-outputs",
                "3",
            ]
        )
        validate_request_limits(args.max_parallel_samples, args.max_batch_outputs)
        with get_context().override_server_args(
            max_parallel_samples=args.max_parallel_samples,
            max_batch_outputs=args.max_batch_outputs,
        ):
            manager = make_manager()
            with self.assertRaisesRegex(ValueError, r"\[1, 2\]"):
                manager.validate_request_params(
                    GenerateReqInput(text="hello", sampling_params={"n": 3})
                )
            with self.assertRaisesRegex(ValueError, "max-batch-outputs=3"):
                manager.validate_request_params(
                    GenerateReqInput(text=["a", "b"], sampling_params={"n": 2})
                )
        for n, total in ((0, 1024), (128, -1), (True, 1), (1, 2.0)):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                validate_request_limits(n, total)

    def test_raw_input_ids_range_types_and_batch(self):
        manager = make_manager()
        for ids in ([0, 255], [[0], [255]]):
            manager.validate_request_params(GenerateReqInput(input_ids=ids))
            manager.validate_request_params(EmbeddingReqInput(input_ids=ids))
        for ids in ([-1], [256], [True], [1.5], [], [[]], [[1], [-1]], [1, [2]]):
            with self.subTest(ids=ids):
                with self.assertRaisesRegex(ValueError, "input_ids"):
                    manager.validate_request_params(GenerateReqInput(input_ids=ids))
        # Validate only the original IDs; preprocessing may later replace them
        # with internal multimodal padding IDs outside the vocabulary.
        obj = GenerateReqInput(input_ids=[1], image_data="image")
        manager.validate_request_params(obj)
        self.assertEqual(obj.input_ids, [1])

    def test_bad_batch_member_rejected_before_any_allocation(self):
        manager = make_manager()
        requests = (
            GenerateReqInput(text="hello", sampling_params={"n": 5_000_000}),
            GenerateReqInput(input_ids=[[1], [-1]]),
            GenerateReqInput(text=["a", "b"], return_sampling_mask=[False, True]),
        )
        for obj in requests:
            with patch.object(obj, "normalize_batch_and_arguments") as normalize:
                with self.assertRaises(ValueError):
                    asyncio.run(manager.generate_request(obj).__anext__())
                normalize.assert_not_called()
        manager._init_req_state.assert_not_called()
        manager._send_one_request.assert_not_called()

    def test_empty_input_ids_preserve_session_continuations(self):
        """Existing session history can supply the prompt when no new IDs arrive."""
        manager = make_manager(DisaggregationMode.NULL)
        request = GenerateReqInput(
            input_ids=[],
            session_params={"id": "existing-session"},
            sampling_params={"max_new_tokens": 8},
        )
        manager.validate_request_params(request)
        self.assertEqual(request.input_ids, [])
        self.assertFalse(hasattr(request, "batch_size"))
        request.normalize_batch_and_arguments()
        self.assertTrue(request.is_single)
        self.assertEqual(request.batch_size, 1)
        for params in (None, {}, {"id": ""}):
            with self.subTest(session_params=params):
                with self.assertRaisesRegex(ValueError, "input_ids"):
                    manager.validate_request_params(
                        GenerateReqInput(input_ids=[], session_params=params)
                    )

    def test_sampling_mask_uses_release_enable_flag_and_capacity(self):
        manager = make_manager()
        obj = GenerateReqInput(
            text="hello", return_sampling_mask=True, sampling_params={"top_k": 3}
        )
        with self.assertRaisesRegex(ValueError, "SGLANG_ENABLE_DISAGG_SAMPLING_MASK=1"):
            manager.validate_request_params(obj)
        with (
            patch.dict(os.environ, {"SGLANG_ENABLE_DISAGG_SAMPLING_MASK": "1"}),
            get_context().override_server_args(sampling_mask_max_tokens=4),
        ):
            manager.validate_request_params(obj)
            obj.sampling_params = {"top_k": 4}
            manager.validate_request_params(obj)  # Release permits top_k at the cap.
            obj.sampling_params = {"top_k": 5}
            with self.assertRaisesRegex(ValueError, "sampling-mask-max-tokens"):
                manager.validate_request_params(obj)
            obj.sampling_params = {"temperature": 0}
            manager.validate_request_params(obj)
            obj.sampling_params = {}
            with self.assertRaisesRegex(ValueError, "finite top_k"):
                manager.validate_request_params(obj)

    def test_sampling_mask_respects_preferred_params_modes_and_batch(self):
        manager = make_manager()
        manager.preferred_sampling_params = {"top_k": 3}
        with (
            patch.dict(os.environ, {"SGLANG_ENABLE_DISAGG_SAMPLING_MASK": "1"}),
            get_context().override_server_args(sampling_mask_max_tokens=4),
        ):
            manager.validate_request_params(
                GenerateReqInput(text="hello", return_sampling_mask=True)
            )
            manager.validate_request_params(
                GenerateReqInput(
                    text=["a", "b"],
                    return_sampling_mask=[False, True],
                    sampling_params=[{"top_k": 100}, {"top_k": 3}],
                )
            )
            for kwargs in (
                {"speculative_algorithm": "EAGLE"},
                {"sampling_backend": "ascend"},
            ):
                with get_context().override_server_args(**kwargs):
                    with self.assertRaisesRegex(ValueError, "not supported"):
                        manager.validate_request_params(
                            GenerateReqInput(text="hello", return_sampling_mask=True)
                        )
        unified = make_manager(DisaggregationMode.NULL)
        unified.validate_request_params(
            GenerateReqInput(
                text="hello", return_sampling_mask=True, sampling_params={"top_k": 5}
            )
        )
        with get_context().override_server_args(sampling_mask_max_tokens=4):
            with self.assertRaisesRegex(ValueError, "sampling-mask-max-tokens"):
                unified.validate_request_params(
                    GenerateReqInput(
                        text="hello",
                        return_sampling_mask=True,
                        sampling_params={"top_k": 5},
                    )
                )
        for masks in (1, "true", [True], [False, 1]):
            with self.assertRaisesRegex(ValueError, "return_sampling_mask"):
                manager.validate_request_params(
                    GenerateReqInput(text=["a", "b"], return_sampling_mask=masks)
                )

    def test_metadata_rejection_does_not_partially_write_buffers(self):
        import torch

        with (
            patch("sglang.srt.disaggregation.utils.is_npu", return_value=False),
            patch(
                "sglang.srt.disaggregation.utils.envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL.get",
                return_value=None,
            ),
        ):
            with patch.dict(os.environ, {"SGLANG_ENABLE_DISAGG_SAMPLING_MASK": "1"}):
                buffers = MetadataBuffers(
                    1, 1, torch.float32, max_sampling_mask_tokens=2
                )
        req = SimpleNamespace(
            return_logprob=False,
            return_sampling_mask=True,
            sampling_mask_rows=make_mask_rows([1, 2, 3]),
            sampling_logprobs_mode="selected",
            metadata_buffer_index=0,
            output_ids=[99],
        )
        with self.assertRaisesRegex(InvalidDisaggregationMetadata, "length 3"):
            buffers.set_buf(req)
        self.assertEqual(buffers.output_ids[0, 0].item(), 0)

    def test_native_http_400_for_all_new_guards_and_streams(self):
        manager = make_manager()
        app = FastAPI()

        @app.post("/generate")
        async def generate(request: Request):
            return await http_server.generate_request(
                GenerateReqInput(**await request.json()), request
            )

        with (
            patch.object(
                http_server, "_global_state", SimpleNamespace(tokenizer_manager=manager)
            ),
            TestClient(app) as client,
        ):
            for stream in (False, True):
                for payload in (
                    {"text": "hello", "sampling_params": {"n": 129}},
                    {"input_ids": [-1]},
                    {"text": "hello", "return_sampling_mask": True},
                ):
                    response = client.post(
                        "/generate", json={**payload, "stream": stream}
                    )
                    self.assertEqual(response.status_code, 400, response.text)
                    self.assertIn("application/json", response.headers["content-type"])
        manager._init_req_state.assert_not_called()
        manager._send_one_request.assert_not_called()

    def test_openai_400_for_all_new_guards(self):
        for cls in (OpenAIServingChat, OpenAIServingCompletion):
            for stream in (False, True):
                for obj in (
                    GenerateReqInput(text="hello", sampling_params={"n": 129}),
                    GenerateReqInput(input_ids=[-1]),
                    GenerateReqInput(text="hello", return_sampling_mask=True),
                ):
                    serving = cls.__new__(cls)
                    serving.tokenizer_manager = make_manager()
                    handler = (
                        serving._handle_streaming_request
                        if stream
                        else serving._handle_non_streaming_request
                    )
                    response = asyncio.run(
                        handler(
                            obj,
                            SimpleNamespace(
                                stream_options=None,
                                return_input_ids_in_sglext=False,
                                return_output_ids_in_sglext=False,
                            ),
                            None,
                        )
                    )
                    self.assertEqual(response.status_code, 400)

    def test_metadata_overflow_aborts_one_request_and_releases_prefill_state(self):
        enter_scope(self, published_topology())
        import torch

        from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST

        with patch.dict(os.environ, {"SGLANG_ENABLE_DISAGG_SAMPLING_MASK": "1"}):
            buffers = MetadataBuffers(1, 1, torch.float32, max_sampling_mask_tokens=2)
        sender = Mock()
        sender.poll.return_value = KVPoll.Transferring
        sender.is_source_release_safe.return_value = True
        sender.abort.side_effect = lambda: setattr(
            sender.poll, "return_value", KVPoll.Failed
        )
        sender.failure_exception.side_effect = RuntimeError("transfer aborted")
        req = SimpleNamespace(
            rid="bad-mask",
            return_logprob=False,
            return_routed_experts=False,
            cache_request_handle=None,
            return_sampling_mask=True,
            sampling_mask_rows=make_mask_rows([1, 2, 3]),
            sampling_logprobs_mode="selected",
            metadata_buffer_index=0,
            output_ids=[1],
            start_send_idx=0,
            origin_input_ids=[1, 2],
            extend_end=2,
            disagg_kv_sender=sender,
            finished_reason=None,
            to_finish=None,
            bootstrap_room=7,
            pending_bootstrap=False,
            time_stats=Mock(),
            bootstrap_host=FAKE_BOOTSTRAP_HOST,
            kv=SimpleNamespace(holds_kv=True, holds_mamba=False),
        )
        good_sender = Mock()
        good_sender.poll.return_value = KVPoll.Success
        good_sender.is_source_release_safe.return_value = True
        good = SimpleNamespace(
            rid="good",
            return_logprob=False,
            return_routed_experts=False,
            cache_request_handle=None,
            disagg_kv_sender=good_sender,
            pending_bootstrap=False,
            finished_reason=None,
            to_finish=None,
            time_stats=Mock(),
            bootstrap_host=FAKE_BOOTSTRAP_HOST,
            metadata_buffer_index=1,
        )
        scheduler = SchedulerDisaggregationPrefillMixin()
        scheduler.scheduler_stage_metrics = None
        scheduler.token_to_kv_pool_allocator = SimpleNamespace(page_size=1)
        scheduler.enable_staging = False
        scheduler.kv_checksum_computer = None
        scheduler.disagg_metadata_buffers = buffers
        scheduler.disagg_prefill_pending_chunk_rids = {req.rid}
        scheduler.disagg_prefill_inflight_queue = [req, good]
        scheduler.attn_cp_cpu_group = scheduler.attn_tp_cpu_group = None
        scheduler.tree_cache = Mock()
        scheduler.req_to_metadata_buffer_idx_allocator = Mock()
        scheduler.metrics_reporter = SimpleNamespace(enable_metrics=False)
        scheduler.output_streamer = Mock()
        scheduler.enable_overlap = False
        scheduler._release_aborted_request = Mock()

        scheduler.send_kv_chunk(req, last_chunk=True)
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        self.assertEqual(req.finished_reason.status_code, 400)
        sender.abort.assert_called_once()
        sender.send.assert_not_called()
        self.assertNotIn(req.rid, scheduler.disagg_prefill_pending_chunk_rids)
        self.assertEqual(buffers.output_ids[0, 0].item(), 0)

        with (
            patch(
                "sglang.srt.disaggregation.prefill.poll_and_all_reduce_attn_cp_tp_group",
                side_effect=lambda senders, *groups: [s.poll() for s in senders],
            ),
            patch("sglang.srt.disaggregation.prefill.dist.all_reduce"),
            patch("sglang.srt.disaggregation.prefill.release_kv_cache") as release,
        ):
            done = scheduler.process_disagg_prefill_inflight_queue()
        self.assertEqual(done, [req, good])
        self.assertEqual(release.call_count, 2)
        self.assertEqual(req.finished_reason.status_code, 400)
        self.assertEqual(scheduler.disagg_prefill_inflight_queue, [])
        self.assertEqual(
            scheduler.req_to_metadata_buffer_idx_allocator.free.call_count, 2
        )
        self.assertEqual(req.metadata_buffer_index, -1)
        self.assertEqual(good.metadata_buffer_index, -1)

    def test_runtime_guard_does_not_swallow_unrelated_exceptions(self):
        req = SimpleNamespace(
            start_send_idx=0,
            origin_input_ids=[1],
            extend_end=1,
            disagg_kv_sender=Mock(),
        )
        scheduler = SchedulerDisaggregationPrefillMixin()
        scheduler.token_to_kv_pool_allocator = SimpleNamespace(page_size=1)
        scheduler.kv_checksum_computer = None
        scheduler.disagg_metadata_buffers = Mock()
        scheduler.disagg_metadata_buffers.set_buf.side_effect = RuntimeError(
            "CUDA failure"
        )
        with self.assertRaisesRegex(RuntimeError, "CUDA failure"):
            scheduler.send_kv_chunk(req, last_chunk=True)
        req.disagg_kv_sender.abort.assert_not_called()


if __name__ == "__main__":
    unittest.main()
