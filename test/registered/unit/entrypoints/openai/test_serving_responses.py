import argparse
import asyncio
import sys
import unittest
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from copy import deepcopy
from unittest.mock import AsyncMock, Mock, patch

import orjson
import pytest
from openai.types.responses import (
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
)
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from openai_harmony import Conversation, Message, Role, ToolNamespaceConfig
from utils import (
    MockTemplateManager,
    MockTokenizerManager,
    StreamFixture,
    collect_stream_events,
    engine_chunk,
    event_payloads,
    make_serving,
)
from xgrammar import Grammar, GrammarCompiler, GrammarMatcher, TokenizerInfo
from xgrammar.testing import _is_grammar_accept_string

from sglang.srt.entrypoints.context import (
    HarmonyContext,
    SimpleContext,
)
from sglang.srt.entrypoints.harmony_utils import get_encoding
from sglang.srt.entrypoints.openai.protocol import (
    MessageProcessingResult,
    RequestResponseMetadata,
    ResponsesRequest,
    ResponsesResponse,
)
from sglang.srt.entrypoints.openai.serving_responses import (
    OpenAIServingResponses,
    _build_output_text_logprobs,
    _should_emit_normal_text_as_message,
)
from sglang.srt.environ import ToolStrictLevel, envs
from sglang.srt.function_call.core_types import ToolCallItem
from sglang.srt.function_call.kimik3_format import THINK_CLOSE
from sglang.srt.parser.template_detection import ReasoningToggleConfig
from sglang.srt.runtime_context import get_serving, publish, reset_context
from sglang.srt.sampling.sampling_params import (
    REQUEST_REASONING_END_TOKEN_IDS_KEY,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class InputMessageConstructionTestCase(CustomTestCase):
    def test_previous_response_replays_assistant_text_not_instructions(self):
        serving = make_serving()
        prev_response = Mock(id="resp_prev")
        prev_response.output = [
            ResponseReasoningItem(
                id="rs_prev", summary=[], type="reasoning", content=None, status=None
            ),
            ResponseOutputMessage(
                id="msg_prev",
                content=[
                    ResponseOutputText(
                        text="first answer part",
                        annotations=[],
                        type="output_text",
                        logprobs=None,
                    ),
                    ResponseOutputText(
                        text="second answer part",
                        annotations=[],
                        type="output_text",
                        logprobs=None,
                    ),
                ],
                role="assistant",
                status="completed",
                type="message",
            ),
        ]
        serving.msg_store["resp_prev"] = [
            {"role": "user", "content": "old input"},
            *[item.model_dump(exclude_none=True) for item in prev_response.output],
        ]

        request = ResponsesRequest(
            model="x",
            instructions="Be brief",
            previous_response_id="resp_prev",
            input="new input",
            store=False,
        )

        messages = serving._construct_input_messages(request, prev_response)

        self.assertEqual(
            messages,
            [
                {"role": "system", "content": "Be brief"},
                {"role": "user", "content": "old input"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "first answer part"},
                        {"type": "text", "text": "second answer part"},
                    ],
                },
                {"role": "user", "content": "new input"},
            ],
        )

    def test_stored_tool_turn_matches_client_replay_without_old_instructions(self):
        publish(
            ServerArgs(model_path="dummy", enable_response_store=True), role="tokenizer"
        )
        for stream in (False, True):
            with self.subTest(stream=stream):
                serving = make_serving()
                serving.reasoning_parser = "deepseek-r1"
                serving.tool_call_parser = None
                request = ResponsesRequest(
                    model="x",
                    input="old input",
                    instructions="OLD INSTRUCTION",
                    tools=[{"type": "function", "name": "lookup"}],
                    tool_choice="required",
                    store=True,
                    stream=stream,
                )
                chunk = engine_chunk(
                    '<think>secret plan</think>[{"name":"lookup","parameters":{}}]',
                    finish=True,
                )
                if stream:
                    StreamFixture(serving, request).run([chunk])
                    response = serving.response_store[request.request_id]
                else:
                    context = SimpleContext()
                    context.append_output(chunk)

                    async def empty():
                        if False:
                            yield

                    response = asyncio.run(
                        serving.responses_full_generator(
                            request,
                            {},
                            empty(),
                            context,
                            "x",
                            Mock(),
                            RequestResponseMetadata(request_id=request.request_id),
                            require_reasoning=False,
                        )
                    )
                call = next(
                    item for item in response.output if item.type == "function_call"
                )
                result = {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": "answer 42",
                }
                for instructions in ("NEW INSTRUCTION", None):
                    followup = ResponsesRequest(
                        model="x",
                        previous_response_id=response.id,
                        input=[result],
                        instructions=instructions,
                        store=False,
                    )
                    explicit = ResponsesRequest(
                        model="x",
                        input=[{"role": "user", "content": "old input"}]
                        + [item.model_dump() for item in response.output]
                        + [result],
                        instructions=instructions,
                        store=False,
                    )
                    messages = serving._construct_input_messages(followup, response)
                    self.assertEqual(
                        messages, serving._construct_input_messages(explicit)
                    )
                    self.assertNotIn("OLD INSTRUCTION", str(messages))
                    self.assertIn("secret plan", str(messages))
                    self.assertIn(call.call_id, str(messages))
                self.assertEqual(
                    [item["type"] for item in serving.msg_store[response.id][1:]],
                    ["reasoning", "function_call"],
                )

    def test_harmony_instructions_are_rebuilt_for_each_request(self):
        serving = make_serving()
        serving.use_harmony = True
        previous = Mock(id="resp_previous", output=[])
        first = ResponsesRequest(
            model="x", input="old input", instructions="OLD INSTRUCTION"
        )
        messages = serving._construct_input_messages_with_harmony(first, None)
        serving.msg_store[previous.id] = messages[2:]
        for instructions in ("NEW INSTRUCTION", None):
            request = ResponsesRequest(
                model="x",
                input="next",
                instructions=instructions,
                previous_response_id=previous.id,
            )
            actual = serving._construct_input_messages_with_harmony(request, previous)
            expected = serving._construct_input_messages_with_harmony(request, None)
            self.assertEqual(actual[:2], expected[:2])
            self.assertEqual(actual[2:], messages[2:] + expected[2:])

    def test_harmony_replays_output_text_and_encoded_reasoning(self):
        from sglang.srt.entrypoints.harmony_utils import parse_response_input
        from sglang.srt.entrypoints.openai.responses_adapters import (
            encode_reasoning_state,
        )

        message = parse_response_input(
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "assistant-only secret 42"},
                ],
                "phase": "final_answer",
            },
            [],
        )
        self.assertEqual(message.content[0].text, "assistant-only secret 42")
        reasoning = parse_response_input(
            {
                "type": "reasoning",
                "encrypted_content": encode_reasoning_state("private plan"),
            },
            [],
        )
        self.assertEqual(reasoning.content[0].text, "private plan")
        self.assertEqual(reasoning.channel, "analysis")

    def test_harmony_replays_dict_tool_calls_and_results_in_one_input(self):
        serving = make_serving()
        request = ResponsesRequest(
            model="x",
            input=[
                {
                    "type": "function_call",
                    "name": "lookup",
                    "call_id": "call_1",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": [{"type": "output_text", "text": "result"}],
                },
            ],
            store=False,
        )
        messages = serving._construct_input_messages_with_harmony(request, None)
        self.assertEqual(messages[-1].author.name, "functions.lookup")
        self.assertEqual(messages[-1].content[0].text, "result")

    def test_harmony_message_channels_map_to_phases(self):
        from sglang.srt.entrypoints.harmony_utils import (
            parse_output_message,
            parse_response_input,
        )

        for phase in ("commentary", "final_answer"):
            message = parse_response_input(
                {
                    "role": "assistant",
                    "content": "answer",
                    "phase": phase,
                },
                [],
            )
            (item,) = parse_output_message(message)
            self.assertEqual(item.phase, phase)
            self.assertEqual(item.content[0].text, "answer")

    def test_replay_preserves_different_assistant_phases(self):
        serving = make_serving()
        request = ResponsesRequest(
            model="x",
            input=[
                {"role": "assistant", "content": "working", "phase": "commentary"},
                {"role": "assistant", "content": "answer", "phase": "final_answer"},
            ],
            store=False,
        )
        messages = serving._construct_input_messages(request)
        self.assertEqual([m["phase"] for m in messages], ["commentary", "final_answer"])
        self.assertEqual([m["content"] for m in messages], ["working", "answer"])

    def test_input_parts_normalized_for_chat_templates(self):
        serving = make_serving()
        request = ResponsesRequest(
            model="x",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "what is this?"},
                        {
                            "type": "input_image",
                            "image_url": "http://example.com/cat.png",
                        },
                    ],
                }
            ],
            store=False,
        )

        messages = serving._construct_input_messages(request)

        self.assertEqual(
            messages,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "http://example.com/cat.png",
                                "detail": "auto",
                            },
                        },
                    ],
                }
            ],
        )

    def test_previous_response_id_input_list_does_not_call_copy_module(self):
        serving = make_serving()
        serving.use_harmony = True
        prev = Mock(id="resp_prev")
        prev.output = [
            ResponseFunctionToolCall(
                arguments="{}",
                call_id="call_x",
                name="t",
                type="function_call",
                id="fc_x",
                status="completed",
            )
        ]
        request = ResponsesRequest(
            model="x",
            input=[{"role": "user", "content": "hi"}],
            previous_response_id="resp_prev",
            store=False,
        )
        try:
            serving._construct_input_messages_with_harmony(request, prev)
        except TypeError as exc:
            self.fail(f"copy() module-call regression: {exc}")
        except Exception:
            pass


class ChatToolForwardingTestCase(CustomTestCase):
    def test_make_request_passes_function_tools_to_chat_processing(self):
        serving = make_serving()
        seen = {}

        def fake_process(chat_request, is_multimodal):
            seen["tools"] = chat_request.tools
            seen["tool_choice"] = chat_request.tool_choice
            seen["parallel_tool_calls"] = chat_request.parallel_tool_calls
            return MessageProcessingResult(
                prompt="prompt",
                prompt_ids=[1, 2, 3],
                image_data=None,
                audio_data=None,
                video_data=None,
                modalities=[],
                stop=["</s>"],
                tool_call_constraint=("json_schema", {"type": "object"}),
            )

        serving._process_messages = Mock(side_effect=fake_process)
        request = ResponsesRequest(
            model="x",
            input="call the tool",
            tools=[
                {
                    "type": "function",
                    "name": "lookup",
                    "parameters": {"type": "object"},
                }
            ],
            tool_choice="required",
            parallel_tool_calls=False,
            store=False,
        )

        messages, request_prompts, engine_prompts, processed = asyncio.run(
            serving._make_request(request, None, serving.tokenizer_manager.tokenizer)
        )

        self.assertEqual(messages, [{"role": "user", "content": "call the tool"}])
        self.assertEqual(request_prompts, [[1, 2, 3]])
        self.assertEqual(engine_prompts, [[1, 2, 3]])
        self.assertEqual(seen["tools"][0].function.name, "lookup")
        self.assertEqual(seen["tool_choice"], "required")
        self.assertFalse(seen["parallel_tool_calls"])
        self.assertEqual(processed.tool_call_constraint[0], "json_schema")

    def test_harmony_forced_choices_explain_missing_routing_constraints(self):
        serving = make_serving()
        serving.use_harmony = True
        for choice in ("none", "required", {"type": "function", "name": "lookup"}):
            request = ResponsesRequest(
                model="x",
                input="hi",
                tool_choice=choice,
                tools=[{"type": "function", "name": "lookup"}],
                store=False,
            )
            response = asyncio.run(serving.create_responses(request))
            self.assertEqual(response.status_code, 400)
            self.assertIn(b"recipient", response.body)
            self.assertIn(b"tool_choice", response.body)
        serving.tokenizer_manager.generate_request.assert_not_called()

    def test_required_tool_choice_without_function_tool_returns_400(self):
        serving = make_serving()
        request = ResponsesRequest(
            model="x",
            input="hi",
            tool_choice="required",
            tools=[{"type": "web_search"}, {"type": "mcp"}],
            store=False,
        )
        result = asyncio.run(serving.create_responses(request, raw_request=None))
        self.assertEqual(getattr(result, "status_code", None), 400)

    def test_kimi_k3_request_uses_chat_encoder_fields(self):
        serving = make_serving()
        serving.chat_encoding_spec = "kimi_k3"
        serving.default_chat_template_kwargs = {}
        serving.template_manager.chat_template_name = None
        serving.tokenizer_manager.tokenizer.apply_chat_template.return_value = [4, 5, 6]
        request = ResponsesRequest(
            model="x",
            input="Explain <|kimi_image_placeholder|>",
            tools=[
                {
                    "type": "function",
                    "name": "lookup",
                    "parameters": {"type": "object"},
                }
            ],
            tool_choice="required",
            reasoning={"effort": "high"},
            store=False,
        )

        _, request_prompts, engine_prompts, _ = asyncio.run(
            serving._make_request(request, None, serving.tokenizer_manager.tokenizer)
        )

        call = serving.tokenizer_manager.tokenizer.apply_chat_template.call_args
        self.assertEqual(
            call.args[0][0]["content"], "Explain <| kimi_image_placeholder |>"
        )
        self.assertEqual(call.kwargs["thinking_effort"], "high")
        self.assertEqual(call.kwargs["tool_choice"], "required")
        self.assertEqual(call.kwargs["tools"][0]["function"]["name"], "lookup")
        self.assertEqual(request_prompts, [[4, 5, 6]])
        self.assertEqual(engine_prompts, [[4, 5, 6]])

    def test_k2_output_parser_reuses_effective_template_default(self):
        serving = make_serving()
        serving.reasoning_parser = "k2_horizon"
        serving.default_chat_template_kwargs = {"reasoning_effort": "low"}
        serving.template_manager.chat_template_name = None
        serving.tokenizer_manager.tokenizer.apply_chat_template.return_value = [4, 5, 6]
        request = ResponsesRequest(
            model="IFM/K2-Horizon-7B",
            input="hi",
            # Template kwargs are the final render inputs, so the server default
            # below takes precedence over this API convenience field.
            reasoning={"effort": "medium"},
            store=False,
        )

        asyncio.run(
            serving._make_request(request, None, serving.tokenizer_manager.tokenizer)
        )

        render_call = serving.tokenizer_manager.tokenizer.apply_chat_template.call_args
        self.assertEqual(render_call.kwargs["reasoning_effort"], "low")
        self.assertEqual(request.chat_template_kwargs["reasoning_effort"], "low")

        output_items = serving._make_response_output_items(
            request,
            "work</ifm|think_faster>\nanswer",
            tokenizer=Mock(),
            require_reasoning=True,
        )
        self.assertEqual(output_items[0].content[0].text, "work")
        self.assertEqual(output_items[1].content[0].text, "\nanswer")


class AllowedToolsResponsesTestCase(CustomTestCase):
    def setUp(self):
        super().setUp()
        reset_context()
        self.addCleanup(reset_context)
        self.serving = make_serving()
        self.serving.chat_encoding_spec = "kimi_k3"
        self.serving.tool_call_parser = "kimi_k3"
        self.serving.default_chat_template_kwargs = {}
        self.serving.template_manager.chat_template_name = None
        self.serving.tokenizer_manager.tokenizer.apply_chat_template.return_value = [
            1,
            2,
        ]
        self.internal_request = None

    @staticmethod
    def _request(mode="auto", names=("B",), **kwargs):
        return ResponsesRequest(
            model="x",
            input="Choose a tool.",
            store=False,
            tools=[
                {"type": "function", "name": name, "parameters": {"type": "object"}}
                for name in ("A", "B", "C")
            ],
            tool_choice={
                "type": "allowed_tools",
                "mode": mode,
                "tools": [{"type": "function", "name": name} for name in names],
            },
            **kwargs,
        )

    @staticmethod
    def _call(name="B", *, arguments="", count=1):
        return (
            "<|open|>tools<|sep|>"
            + "".join(
                f'<|open|>call tool="{name}" index="{index}"<|sep|>'
                f"{arguments}<|close|>call<|sep|>"
                for index in range(1, count + 1)
            )
            + "<|close|>tools<|sep|>"
        )

    def _complete(
        self, request, text="No tool needed.", *, chunk_size=1, finish_type="stop"
    ):
        self.internal_request = None

        async def generate(internal_request, raw_request):
            self.internal_request = internal_request
            ends = (
                [*range(chunk_size, len(text), chunk_size), len(text)]
                if request.stream
                else [len(text)]
            )
            for end in ends:
                chunk = engine_chunk(text[:end])
                chunk["meta_info"]["finish_reason"] = (
                    {"type": finish_type} if end == len(text) else None
                )
                yield chunk

        async def complete():
            self.serving.tokenizer_manager.generate_request = generate
            response = await self.serving.create_responses(request)
            if isinstance(response, AsyncIterator):
                return event_payloads(await collect_stream_events(response))
            return response

        return asyncio.run(complete())

    def test_subset_preserves_declarations_and_round_trips_wire_choice(self):
        """Restrict calls without trimming the prompt's tools or echoing auto."""
        for stream in (False, True):
            for mode in ("auto", "required"):
                with self.subTest(stream=stream, mode=mode):
                    request = self._request(mode=mode, stream=stream)
                    result = self._complete(request, self._call())
                    if stream:
                        self.assertEqual(result[-1]["type"], "response.completed")
                        response = result[-1]["response"]
                        self.assertTrue(
                            all(
                                event["response"]["tool_choice"] == request.tool_choice
                                for event in result
                                if "response" in event
                            )
                        )
                    else:
                        self.assertIsInstance(result, ResponsesResponse)
                        response = result.model_dump()
                    self.assertEqual(response["tool_choice"], request.tool_choice)
                    self.assertEqual(
                        [
                            item["name"]
                            for item in response["output"]
                            if item["type"] == "function_call"
                        ],
                        ["B"],
                    )
                    rendered = self.serving.tokenizer_manager.tokenizer.apply_chat_template.call_args
                    self.assertEqual(
                        [tool["function"]["name"] for tool in rendered.kwargs["tools"]],
                        ["A", "B", "C"],
                    )
                    grammar = Grammar.from_structural_tag(
                        self.internal_request.sampling_params["structural_tag"]
                    )
                    self.assertTrue(_is_grammar_accept_string(grammar, self._call()))
                    self.assertFalse(
                        _is_grammar_accept_string(grammar, self._call("C"))
                    )
                    self.assertEqual(
                        _is_grammar_accept_string(grammar, "No tool needed."),
                        mode == "auto",
                    )

    def test_allowlist_preserves_reasoning_parallel_and_strict_options(self):
        self.serving.reasoning_parser = "kimi_k3"
        self.serving.template_manager.reasoning_config = ReasoningToggleConfig(
            toggle_param="thinking", default_enabled=True
        )
        arguments = (
            '<|open|>argument key="count" type="number"<|sep|>2<|close|>argument<|sep|>'
        )
        single = self._call(arguments=arguments)
        multiple = self._call(arguments=arguments, count=2)
        for stream in (False, True):
            for mode, thinking, parallel, strict in (
                ("auto", False, False, False),
                ("auto", True, True, True),
                ("required", False, True, False),
                ("required", True, False, True),
                ("auto", False, True, True),
                ("required", True, True, False),
            ):
                with (
                    self.subTest(
                        stream=stream,
                        mode=mode,
                        thinking=thinking,
                        parallel=parallel,
                        strict=strict,
                    ),
                    envs.SGLANG_TOOL_STRICT_LEVEL.override(ToolStrictLevel.OFF),
                ):
                    request = self._request(
                        mode=mode,
                        stream=stream,
                        parallel_tool_calls=parallel,
                        chat_template_kwargs={"thinking": thinking},
                    )
                    request.tools[1].strict = strict
                    request.tools[1].parameters = {
                        "type": "object",
                        "properties": {"count": {"type": "integer", "minimum": 1}},
                        "required": ["count"],
                        "additionalProperties": False,
                    }
                    text = ("Thinking." + THINK_CLOSE if thinking else "") + (
                        multiple if parallel else single
                    )
                    result = self._complete(request, text)
                    if stream:
                        self.assertEqual(result[-1]["type"], "response.completed")
                        output = result[-1]["response"]["output"]
                    else:
                        self.assertIsInstance(result, ResponsesResponse)
                        output = result.model_dump()["output"]
                    calls = [item for item in output if item["type"] == "function_call"]
                    self.assertEqual(
                        [
                            (call["name"], orjson.loads(call["arguments"]))
                            for call in calls
                        ],
                        [("B", {"count": 2})] * (2 if parallel else 1),
                    )
                    reasoning = "".join(
                        part["text"]
                        for item in output
                        if item["type"] == "reasoning"
                        for part in item["content"]
                    )
                    self.assertEqual(reasoning, "Thinking." if thinking else "")
                    self.assertEqual(self.internal_request.require_reasoning, thinking)
                    grammar = Grammar.from_structural_tag(
                        self.internal_request.sampling_params["structural_tag"]
                    )
                    self.assertTrue(_is_grammar_accept_string(grammar, single))
                    self.assertEqual(
                        _is_grammar_accept_string(grammar, multiple), parallel
                    )
                    self.assertEqual(
                        _is_grammar_accept_string(grammar, self._call()), not strict
                    )

    def test_allowed_call_arguments_and_ids_agree_across_stream_events(self):
        text = (
            '<|open|>tools<|sep|><|open|>call tool="A" index="1"<|sep|>'
            '<|open|>argument key="city" type="string"<|sep|>'
            'Paris "rive"\\Seine\n雪<|close|>argument<|sep|>'
            '<|open|>argument key="days" type="number"<|sep|>2<|close|>argument<|sep|>'
            '<|open|>argument key="metadata" type="object"<|sep|>'
            '{"tags":["windy"],"enabled":true}<|close|>argument<|sep|>'
            '<|close|>call<|sep|><|open|>call tool="B" index="2"<|sep|>'
            '<|open|>argument key="ok" type="boolean"<|sep|>true<|close|>argument<|sep|>'
            "<|close|>call<|sep|><|close|>tools<|sep|>"
        )
        expected = [
            (
                "A",
                {
                    "city": 'Paris "rive"\\Seine\n雪',
                    "days": 2,
                    "metadata": {"tags": ["windy"], "enabled": True},
                },
            ),
            ("B", {"ok": True}),
        ]
        for mode in ("auto", "required"):
            with self.subTest(mode=mode, stream=False):
                full = self._complete(self._request(mode=mode, names=("A", "B")), text)
                self.assertIsInstance(full, ResponsesResponse)
                self.assertTrue(all(item.call_id for item in full.output))
                self.assertEqual(len({item.call_id for item in full.output}), 2)
                self.assertEqual(
                    [(item.name, orjson.loads(item.arguments)) for item in full.output],
                    expected,
                )
            for chunk_size in (1, len(text)):
                with self.subTest(mode=mode, chunk_size=chunk_size):
                    events = self._complete(
                        self._request(mode=mode, names=("A", "B"), stream=True),
                        text,
                        chunk_size=chunk_size,
                    )
                    self.assertEqual(events[-1]["type"], "response.completed")
                    output = events[-1]["response"]["output"]
                    self.assertEqual(
                        [
                            (item["name"], orjson.loads(item["arguments"]))
                            for item in output
                        ],
                        expected,
                    )
                    self.assertEqual(
                        {item["type"] for item in output}, {"function_call"}
                    )
                    for key in ("id", "call_id"):
                        self.assertTrue(all(item[key] for item in output))
                        self.assertEqual(len({item[key] for item in output}), 2)
                    added = [
                        e for e in events if e["type"] == "response.output_item.added"
                    ]
                    done = [
                        e for e in events if e["type"] == "response.output_item.done"
                    ]
                    self.assertEqual([e["item"] for e in done], output)
                    self.assertEqual(len(added), len(output))
                    argument_events = [
                        e
                        for e in events
                        if e["type"]
                        in (
                            "response.function_call_arguments.delta",
                            "response.function_call_arguments.done",
                        )
                    ]
                    self.assertEqual(
                        {e["item_id"] for e in argument_events},
                        {item["id"] for item in output},
                    )
                    for index, item in enumerate(output):
                        for key in ("type", "id", "call_id", "name"):
                            self.assertEqual(added[index]["item"][key], item[key])
                        for event in (added[index], done[index]):
                            self.assertEqual(event["output_index"], index)
                        parts = [
                            e for e in argument_events if e["item_id"] == item["id"]
                        ]
                        self.assertEqual({e["output_index"] for e in parts}, {index})
                        self.assertEqual(
                            "".join(
                                e["delta"]
                                for e in parts
                                if e["type"].endswith(".delta")
                            ),
                            item["arguments"],
                        )
                        self.assertEqual(
                            [
                                e["arguments"]
                                for e in parts
                                if e["type"].endswith(".done")
                            ],
                            [item["arguments"]],
                        )
                    grammar = Grammar.from_structural_tag(
                        self.internal_request.sampling_params["structural_tag"]
                    )
                    self.assertTrue(_is_grammar_accept_string(grammar, text))

    def test_allowlist_preserves_schema_admission_and_rendering(self):
        """Selecting a subset must not rewrite or reject previously accepted schemas."""
        for schema in (
            {"type": "object", "properties": {"n": {"type": "int"}}},
            {"type": "not_a_json_schema_type"},
        ):
            for allowed in (False, True):
                with self.subTest(schema=schema, allowed=allowed):
                    request = self._request()
                    request.tools[2].parameters = deepcopy(schema)
                    if not allowed:
                        request.tool_choice = "auto"
                    result = self._complete(request)
                    self.assertIsInstance(result, ResponsesResponse)
                    self.assertEqual(request.tools[2].parameters, schema)
                    rendered = self.serving.tokenizer_manager.tokenizer.apply_chat_template.call_args
                    self.assertEqual(
                        rendered.kwargs["tools"][2]["function"]["parameters"], schema
                    )

    def test_allowed_tools_k3_template_hints_preserve_choice_and_grammar(self):
        """Responses must forward required/empty allowlist tool-use hints;
        its wire choice and grammar must retain the original allowlist.
        """
        fields = {"tool_choice", "tools", "chat_template_kwargs"}
        for mode, names, template_choice in (
            ("auto", (), "none"),
            ("required", (), "none"),
            ("auto", ("B",), None),
            ("required", ("B",), "required"),
            ("auto", ("A", "B"), None),
            ("required", ("A", "B"), "required"),
        ):
            for stream in (False, True):
                for template_kwargs, expected_hint in (
                    ({"thinking": False}, template_choice),
                    ({"thinking": False, "tool_choice": "auto"}, "auto"),
                ):
                    with self.subTest(
                        mode=mode, stream=stream, names=names, kwargs=template_kwargs
                    ):
                        request = self._request(
                            mode=mode,
                            names=names,
                            stream=stream,
                            chat_template_kwargs=template_kwargs,
                        )
                        original_choice = request.tool_choice
                        original_fields = request.model_dump(include=fields)
                        text = self._call() if names else "No tool needed."
                        result = self._complete(request, text)
                        if stream:
                            self.assertEqual(result[-1]["type"], "response.completed")
                            response = result[-1]["response"]
                        else:
                            self.assertIsInstance(result, ResponsesResponse)
                            response = result.model_dump()
                        rendered = self.serving.tokenizer_manager.tokenizer.apply_chat_template.call_args
                        self.assertEqual(
                            rendered.kwargs.get("tool_choice"), expected_hint
                        )
                        self.assertEqual(
                            [
                                tool["function"]["name"]
                                for tool in rendered.kwargs["tools"]
                            ],
                            [tool["name"] for tool in original_fields["tools"]],
                        )
                        self.assertIs(request.tool_choice, original_choice)
                        self.assertEqual(
                            request.model_dump(include=fields), original_fields
                        )
                        self.assertEqual(response["tool_choice"], original_choice)
                        grammar = Grammar.from_structural_tag(
                            self.internal_request.sampling_params["structural_tag"]
                        )
                        self.assertTrue(_is_grammar_accept_string(grammar, text))
                        self.assertFalse(
                            _is_grammar_accept_string(grammar, self._call("C"))
                        )

    def test_empty_allowlist_constrains_generation_with_or_without_tools(self):
        for stream in (False, True):
            for mode, declared in (
                ("auto", False),
                ("auto", True),
                ("required", False),
                ("required", True),
            ):
                with self.subTest(mode=mode, stream=stream, declared=declared):
                    request = self._request(mode=mode, names=(), stream=stream)
                    if not declared:
                        request.tools = []
                    result = self._complete(request)
                    if stream:
                        self.assertEqual(result[-1]["type"], "response.completed")
                        output = result[-1]["response"]["output"]
                    else:
                        self.assertIsInstance(result, ResponsesResponse)
                        output = result.model_dump()["output"]
                    self.assertEqual([item["type"] for item in output], ["message"])
                    grammar = Grammar.from_structural_tag(
                        self.internal_request.sampling_params["structural_tag"]
                    )
                    self.assertTrue(
                        _is_grammar_accept_string(grammar, "No tool needed.")
                    )
                    self.assertFalse(_is_grammar_accept_string(grammar, self._call()))

    def test_invalid_function_references_fail_before_generation(self):
        for stream in (False, True):
            for mode in ("auto", "required"):
                with self.subTest(stream=stream, mode=mode):
                    response = self._complete(
                        self._request(mode, names=("missing",), stream=stream)
                    )
                    self.assertEqual(response.status_code, 400)
                    self.assertIsNone(self.internal_request)
            with self.subTest(stream=stream, custom_only=True):
                request = self._request(stream=stream)
                request.tools[1].type = "custom"
                response = self._complete(request, self._call())
                self.assertEqual(response.status_code, 400)
                self.assertIsNone(self.internal_request)

    def test_tool_result_continuation_uses_new_allowlist_without_rewriting_history(
        self,
    ):
        self.serving.enable_response_store = True
        for stream in (False, True):
            with self.subTest(stream=stream):
                first = self._request(names=("A",), stream=stream)
                first.store = True
                result = self._complete(first, self._call("A"))
                if stream:
                    self.assertEqual(result[-1]["type"], "response.completed")
                    previous = result[-1]["response"]
                else:
                    self.assertIsInstance(result, ResponsesResponse)
                    previous = result.model_dump()
                previous_id = previous["id"]
                call_id = previous["output"][0]["call_id"]
                history = deepcopy(self.serving.msg_store[previous_id])
                stored_response = self.serving.response_store[previous_id].model_dump()
                self.assertEqual(stored_response["tool_choice"], first.tool_choice)

                request = self._request(previous_response_id=previous_id, stream=stream)
                request.input = [
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": "answer 42",
                    }
                ]
                result = self._complete(request, self._call())
                if stream:
                    self.assertEqual(result[-1]["type"], "response.completed")
                    response = result[-1]["response"]
                else:
                    self.assertIsInstance(result, ResponsesResponse)
                    response = result.model_dump()
                self.assertEqual(response["output"][0]["name"], "B")
                self.assertEqual(response["tool_choice"], request.tool_choice)
                self.assertEqual(
                    self.serving.response_store[previous_id].model_dump(),
                    stored_response,
                )
                self.assertEqual(self.serving.msg_store[previous_id], history)

                rendered = self.serving.tokenizer_manager.tokenizer.apply_chat_template.call_args
                messages = rendered.args[0]
                self.assertEqual(
                    [m["role"] for m in messages], ["user", "assistant", "tool"]
                )
                self.assertEqual(messages[0]["content"], "Choose a tool.")
                historical_call = messages[1]["tool_calls"][0]
                self.assertEqual(historical_call["id"], call_id)
                self.assertEqual(historical_call["function"]["name"], "A")
                self.assertEqual(messages[2]["tool_call_id"], call_id)
                self.assertEqual(messages[2]["content"], "answer 42")
                self.assertEqual(
                    [tool["function"]["name"] for tool in rendered.kwargs["tools"]],
                    ["A", "B", "C"],
                )
                grammar = Grammar.from_structural_tag(
                    self.internal_request.sampling_params["structural_tag"]
                )
                self.assertFalse(_is_grammar_accept_string(grammar, self._call("A")))
                self.assertTrue(_is_grammar_accept_string(grammar, self._call()))

    def test_k3_parser_installs_grammar_without_native_encoder(self):
        """A generic chat template must not disable the selected K3 tool grammar."""
        self.serving.chat_encoding_spec = None
        for mode in ("auto", "required"):
            for stream in (False, True):
                with self.subTest(mode=mode, stream=stream):
                    result = self._complete(
                        self._request(mode=mode, stream=stream), self._call()
                    )
                    if stream:
                        self.assertEqual(result[-1]["type"], "response.completed")
                    else:
                        self.assertIsInstance(result, ResponsesResponse)
                    grammar = Grammar.from_structural_tag(
                        self.internal_request.sampling_params["structural_tag"]
                    )
                    self.assertTrue(_is_grammar_accept_string(grammar, self._call()))
                    self.assertFalse(
                        _is_grammar_accept_string(grammar, self._call("C"))
                    )
                    self.assertEqual(
                        _is_grammar_accept_string(grammar, "No tool needed."),
                        mode == "auto",
                    )

    def test_unsupported_parser_does_not_fall_back_to_auto(self):
        for encoder, parser, harmony in (
            (None, None, False),
            ("kimi_k3", "hermes", False),
            (None, None, True),
        ):
            with self.subTest(encoder=encoder, parser=parser, harmony=harmony):
                self.serving.chat_encoding_spec = encoder
                self.serving.tool_call_parser = parser
                self.serving.use_harmony = harmony
                response = self._complete(self._request())
                self.assertEqual(response.status_code, 400)
                self.assertIn(b"allowed_tools", response.body)
                self.assertIsNone(self.internal_request)

    def test_text_format_conflicts_but_plain_text_does_not(self):
        for stream in (False, True):
            for mode in ("auto", "required"):
                for fmt in (
                    {"type": "json_object"},
                    *[
                        {
                            "type": "json_schema",
                            "name": "answer",
                            "schema": {"type": "object"},
                            "strict": strict,
                        }
                        for strict in (None, False, True)
                    ],
                ):
                    with self.subTest(stream=stream, mode=mode, fmt=fmt):
                        response = self._complete(
                            self._request(
                                mode=mode, stream=stream, text={"format": fmt}
                            )
                        )
                        self.assertEqual(response.status_code, 400)
                        self.assertIn(b"text.format", response.body)
                        self.assertIsNone(self.internal_request)
        result = self._complete(
            self._request(text={"format": {"type": "text"}}), self._call()
        )
        self.assertIsInstance(result, ResponsesResponse)

    def test_constraint_failure_uses_existing_fallback(self):
        for mode in ("auto", "required"):
            for stream in (False, True):
                with self.subTest(mode=mode, stream=stream):
                    request = self._request(mode=mode, stream=stream)
                    request.tools[1].strict = True
                    request.tools[1].parameters = {"type": "array"}
                    response = self._complete(request)
                    if stream:
                        self.assertEqual(response[-1]["type"], "response.completed")
                    else:
                        self.assertIsInstance(response, ResponsesResponse)
                        self.assertEqual(response.status, "completed")
                    self.assertIsNotNone(self.internal_request)
                    self.assertIsNone(
                        self.internal_request.sampling_params.get("structural_tag")
                    )
                    self.assertIsNone(
                        self.internal_request.sampling_params.get("json_schema")
                    )

    def test_allowed_tools_preserves_existing_output_parsing(self):
        """Grammar selection must not add rejection to the native parser's output."""
        text = (
            '<|open|>tools<|sep|><|open|>call tool="A" index="1"<|sep|>'
            '<|open|>argument key="body" type="string"<|sep|><|close|>call<|sep|>'
            '<|open|>call tool="C" index="2"<|sep|><|close|>call<|sep|>'
            "<|close|>argument<|sep|><|close|>call<|sep|><|close|>tools<|sep|>"
        )
        for mode in ("auto", "required"):
            for stream in (False, True):
                with self.subTest(mode=mode, stream=stream):
                    outputs = []
                    for allowed in (False, True):
                        request = self._request(
                            mode=mode, names=("A", "B"), stream=stream
                        )
                        if not allowed:
                            request.tool_choice = mode
                        result = self._complete(request, text)
                        if stream:
                            self.assertEqual(result[-1]["type"], "response.completed")
                            output = result[-1]["response"]["output"]
                        else:
                            self.assertIsInstance(result, ResponsesResponse)
                            output = result.model_dump()["output"]
                        outputs.append(
                            [
                                {
                                    k: v
                                    for k, v in item.items()
                                    if k not in ("id", "call_id")
                                }
                                for item in output
                            ]
                        )
                    self.assertEqual(outputs[0], outputs[1])
                    grammar = Grammar.from_structural_tag(
                        self.internal_request.sampling_params["structural_tag"]
                    )
                    self.assertTrue(_is_grammar_accept_string(grammar, text))

    def test_required_preserves_incomplete_output_on_both_pd_workers(self):
        """Incomplete PD output must retain its finish reason on both workers."""
        for mode in ("prefill", "decode"):
            publish(
                ServerArgs(model_path="dummy", disaggregation_mode=mode),
                role="tokenizer",
            )
            self.serving = OpenAIServingResponses(
                self.serving.tokenizer_manager, self.serving.template_manager
            )
            self.serving.chat_encoding_spec = "kimi_k3"
            self.serving.tool_call_parser = "kimi_k3"
            for stream in (False, True):
                with self.subTest(mode=mode, stream=stream):
                    result = self._complete(
                        self._request(
                            mode="required",
                            stream=stream,
                            max_output_tokens=512,
                            bootstrap_host="127.0.0.1",
                            bootstrap_room=1,
                        ),
                        "<|open|>",
                        finish_type="length",
                    )
                    self.assertTrue(
                        self.internal_request.sampling_params["structural_tag"]
                    )
                    if stream:
                        self.assertEqual(result[-1]["type"], "response.incomplete")
                        self.assertEqual(result[-1]["response"]["status"], "incomplete")
                        self.assertIsNone(result[-1]["response"]["error"])
                    else:
                        self.assertIsInstance(result, ResponsesResponse)
                        self.assertEqual(result.status, "incomplete")
                        self.assertIsNone(result.error)

    def test_required_preserves_length_truncation_before_a_call(self):
        """A truncated grammar-valid prefix must not become a validation error."""
        prefix = '<|open|>tools<|sep|><|open|>call tool="B" index="1"<|sep|>'
        for stream in (False, True):
            with self.subTest(stream=stream):
                result = self._complete(
                    self._request(mode="required", stream=stream),
                    prefix,
                    finish_type="length",
                )
                compiler = GrammarCompiler(
                    TokenizerInfo([bytes([i]) for i in range(256)])
                )
                matcher = GrammarMatcher(
                    compiler.compile_structural_tag(
                        self.internal_request.sampling_params["structural_tag"]
                    )
                )
                self.assertTrue(matcher.accept_string(prefix))
                self.assertFalse(matcher.is_completed())
                if stream:
                    self.assertEqual(result[-1]["type"], "response.incomplete")
                    response = result[-1]["response"]
                else:
                    self.assertIsInstance(result, ResponsesResponse)
                    response = result.model_dump()
                self.assertEqual(response["status"], "incomplete")
                self.assertIsNone(response["error"])
                self.assertFalse(
                    any(item["type"] == "function_call" for item in response["output"])
                )


class ReasoningRequestForwardingTestCase(unittest.TestCase):
    def test_create_responses_uses_processed_reasoning_state(self):
        serving = make_serving()
        serving.reasoning_parser = "deepseek-r1"
        serving.default_chat_template_kwargs = {"thinking": False}
        serving.template_manager.reasoning_config = ReasoningToggleConfig(
            toggle_param="thinking", default_enabled=True
        )
        rendered = MessageProcessingResult(
            prompt="prompt",
            prompt_ids=[1, 2, 3],
            image_data=None,
            audio_data=None,
            video_data=None,
            modalities=[],
            stop=[],
            reasoning_end_token_ids=[41, 42],
        )
        captured = {}

        async def fake_generate(
            request_id,
            request_prompt,
            adapted_request,
            sampling_params,
            context,
            **kwargs,
        ):
            captured["adapted_request"] = adapted_request
            context.append_output(
                {
                    "text": "done",
                    "meta_info": {
                        "prompt_tokens": 3,
                        "completion_tokens": 1,
                        "cached_tokens": 0,
                    },
                }
            )
            yield context

        serving._generate_with_builtin_tools = fake_generate
        request = ResponsesRequest(
            model="x",
            input="answer",
            request_id="resp_reasoning",
            store=False,
        )

        with (
            patch.object(
                serving, "_apply_conversation_template", return_value=rendered
            ),
            patch(
                "sglang.srt.entrypoints.openai.serving_responses.ReasoningParser"
            ) as parser_cls,
        ):
            parser_cls.return_value.parse_non_stream.return_value = (None, "done")
            response = asyncio.run(serving.create_responses(request))

        self.assertEqual(response.status, "completed")
        self.assertFalse(captured["adapted_request"].require_reasoning)
        self.assertEqual(
            captured["adapted_request"].sampling_params["custom_params"][
                REQUEST_REASONING_END_TOKEN_IDS_KEY
            ],
            [41, 42],
        )
        self.assertFalse(parser_cls.call_args.kwargs["force_reasoning"])


class SkipSpecialTokensForwardingTestCase(CustomTestCase):
    """The skip_special_tokens override from _process_messages must reach the
    engine sampling params; muse's channel markers die in detok otherwise."""

    def _create_responses_sampling_params(self, serving):
        serving.default_chat_template_kwargs = None
        rendered = MessageProcessingResult(
            prompt="prompt",
            prompt_ids=[1, 2, 3],
            image_data=None,
            audio_data=None,
            video_data=None,
            modalities=[],
            stop=[],
        )
        captured = {}

        async def fake_generate(
            request_id,
            request_prompt,
            adapted_request,
            sampling_params,
            context,
            **kwargs,
        ):
            captured["sampling_params"] = sampling_params
            context.append_output(
                {
                    "text": "done",
                    "meta_info": {
                        "prompt_tokens": 3,
                        "completion_tokens": 1,
                        "cached_tokens": 0,
                    },
                }
            )
            yield context

        serving._generate_with_builtin_tools = fake_generate
        request = ResponsesRequest(
            model="x",
            input="answer",
            request_id="resp_skip_special",
            store=False,
        )

        with (
            patch.object(
                serving, "_apply_conversation_template", return_value=rendered
            ),
            patch(
                "sglang.srt.entrypoints.openai.serving_responses.ReasoningParser"
            ) as parser_cls,
        ):
            parser_cls.return_value.parse_non_stream.return_value = (None, "done")
            response = asyncio.run(serving.create_responses(request))

        self.assertEqual(response.status, "completed")
        return captured["sampling_params"]

    def test_marker_preserving_parser_disables_skip_special_tokens(self):
        serving = make_serving()
        serving.reasoning_parser = "muse"
        params = self._create_responses_sampling_params(serving)
        self.assertFalse(params["skip_special_tokens"])

    def test_default_parser_keeps_skip_special_tokens(self):
        serving = make_serving()
        params = self._create_responses_sampling_params(serving)
        # The chat request's True is a synthesized default (ResponsesRequest has
        # no such field), so leave it unset for --preferred-sampling-params.
        self.assertNotIn("skip_special_tokens", params)


class InputItemNormalizationTestCase(CustomTestCase):
    def test_function_call_becomes_assistant_tool_call(self):
        normalized = OpenAIServingResponses._normalize_response_message_for_chat(
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_abc",
                "name": "lookup",
                "arguments": '{"key": "val"}',
                "status": "completed",
            }
        )
        self.assertEqual(
            normalized,
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": '{"key": "val"}',
                        },
                    }
                ],
            },
        )

    def test_developer_role_becomes_labelled_system(self):
        normalized = OpenAIServingResponses._normalize_response_message_for_chat(
            {"role": "developer", "content": "Be terse."}
        )
        self.assertEqual(
            normalized,
            {"role": "system", "content": "Developer instructions:\nBe terse."},
        )

    def test_function_call_output_becomes_tool_message(self):
        normalized = OpenAIServingResponses._normalize_response_message_for_chat(
            {
                "type": "function_call_output",
                "call_id": "call_abc",
                "output": "42",
            }
        )
        self.assertEqual(
            normalized,
            {"role": "tool", "tool_call_id": "call_abc", "content": "42"},
        )

    def test_unknown_input_item_type_raises(self):
        with self.assertRaises(ValueError):
            OpenAIServingResponses._normalize_response_message_for_chat(
                {"type": "web_search_call", "id": "ws_1"}
            )


class FullResponseUsageTestCase(CustomTestCase):
    def test_full_response_uses_dict_meta_info_for_usage(self):
        serving = make_serving()
        context = SimpleContext()
        context.last_output = {
            "text": "done",
            "meta_info": {
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "cached_tokens": 3,
                "reasoning_tokens": 2,
            },
        }
        request = ResponsesRequest(
            model="x", input="hello", request_id="resp_usage", store=False
        )
        metadata = RequestResponseMetadata(request_id=request.request_id)

        async def empty_generator():
            for _ in ():
                yield None

        response = asyncio.run(
            serving.responses_full_generator(
                request,
                sampling_params={},
                result_generator=empty_generator(),
                context=context,
                model_name="x",
                tokenizer=serving.tokenizer_manager.tokenizer,
                request_metadata=metadata,
                created_time=123,
                require_reasoning=False,
            )
        )

        self.assertEqual(response.usage.prompt_tokens, 11)
        self.assertEqual(response.usage.completion_tokens, 7)
        self.assertEqual(response.usage.reasoning_tokens, 2)
        self.assertEqual(metadata.final_usage_info, response.usage)


class MultimodalRequestTestCase(CustomTestCase):
    def test_text_only_create_responses_rejects_media_before_generation(self):
        serving = make_serving()
        serving._process_messages = Mock()
        request = ResponsesRequest(
            model="x",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "describe it"},
                        {
                            "type": "input_image",
                            "image_url": "http://example.com/cat.png",
                        },
                    ],
                }
            ],
            store=False,
        )

        response = asyncio.run(serving.create_responses(request))

        self.assertEqual(response.status_code, 400)
        self.assertIn(b"received unsupported content type 'image_url'", response.body)
        serving._process_messages.assert_not_called()
        serving.tokenizer_manager.generate_request.assert_not_called()

    def test_multimodal_create_responses_sends_text_and_media_to_engine(self):
        serving = make_serving(is_multimodal=True)
        captured = {}

        serving._process_messages = Mock(
            return_value=MessageProcessingResult(
                prompt="rendered multimodal prompt",
                prompt_ids=[9, 9, 9],
                image_data=["http://example.com/cat.png"],
                audio_data=None,
                video_data=None,
                modalities=["image"],
                stop=[],
            )
        )

        async def fake_generate(
            request_id,
            request_prompt,
            adapted_request,
            sampling_params,
            context,
            **kwargs,
        ):
            captured["request_prompt"] = request_prompt
            captured["adapted_request"] = adapted_request
            context.append_output(
                {
                    "text": "looks like a cat",
                    "meta_info": {
                        "prompt_tokens": 5,
                        "completion_tokens": 4,
                        "cached_tokens": 0,
                    },
                }
            )
            yield context

        serving._generate_with_builtin_tools = fake_generate
        request = ResponsesRequest(
            model="x",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "describe it"},
                        {
                            "type": "input_image",
                            "image_url": "http://example.com/cat.png",
                        },
                    ],
                }
            ],
            request_id="resp_mm",
            store=False,
        )

        response = asyncio.run(serving.create_responses(request))

        self.assertEqual(response.status, "completed")
        self.assertEqual(captured["request_prompt"], "rendered multimodal prompt")
        self.assertEqual(captured["adapted_request"].text, "rendered multimodal prompt")
        self.assertIsNone(captured["adapted_request"].input_ids)
        self.assertEqual(
            captured["adapted_request"].image_data, ["http://example.com/cat.png"]
        )
        self.assertEqual(captured["adapted_request"].modalities, ["image"])

    def test_multimodal_token_first_specs_route_through_prompt_ids(self):
        """Bug regression: token-first encoders leave prompt == "" with
        non-empty prompt_ids; forwarding the empty text 400s in
        _tokenize_texts, so the multimodal branch must forward prompt_ids."""
        for spec in ("inkling", "kimi_k3"):
            with self.subTest(spec=spec):
                serving = make_serving(is_multimodal=True)
                serving.chat_encoding_spec = spec
                serving._process_messages = Mock(
                    return_value=MessageProcessingResult(
                        prompt="",
                        prompt_ids=[4, 5, 6],
                        image_data=None,
                        audio_data=None,
                        video_data=None,
                        modalities=[],
                        stop=[],
                    )
                )
                request = ResponsesRequest(model="x", input="hi", store=False)

                _, request_prompts, engine_prompts, _ = asyncio.run(
                    serving._make_request(
                        request, None, serving.tokenizer_manager.tokenizer
                    )
                )

                self.assertEqual(engine_prompts, [[4, 5, 6]])
                self.assertEqual(request_prompts, [[4, 5, 6]])


class OutputItemsTestCase(CustomTestCase):
    def setUp(self):
        # qwen3_coder is the default for this class; the one no-native-parser
        # case overrides it.
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")
        self.serving = make_serving()
        self.serving.tool_call_parser = "qwen3_coder"

    def _function_tool_request(self):
        return ResponsesRequest(
            model="x",
            input="weather?",
            store=False,
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object"},
                }
            ],
        )

    def test_function_tool_call_extracted_via_parser(self):
        serving = self.serving
        fake_call = ToolCallItem(
            tool_index=0, name="get_weather", parameters='{"city": "Beijing"}'
        )

        with patch(
            "sglang.srt.entrypoints.openai.serving_responses.FunctionCallParser"
        ) as parser_cls:
            instance = parser_cls.return_value
            instance.has_tool_call.return_value = True
            instance.parse_non_stream.return_value = ("trailing text", [fake_call])
            output_items = serving._make_response_output_items(
                self._function_tool_request(),
                "raw model output with <tool_call>",
                tokenizer=Mock(),
                require_reasoning=False,
            )

        tool_calls = [
            item for item in output_items if isinstance(item, ResponseFunctionToolCall)
        ]
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].name, "get_weather")
        self.assertEqual(tool_calls[0].arguments, '{"city": "Beijing"}')

        message_items = [
            item for item in output_items if isinstance(item, ResponseOutputMessage)
        ]
        self.assertEqual(len(message_items), 1)
        self.assertEqual(message_items[0].content[0].text, "trailing text")

    def test_prose_emitted_before_tool_call_item(self):
        serving = self.serving
        fake_call = ToolCallItem(
            tool_index=0, name="get_weather", parameters='{"city": "Beijing"}'
        )

        with patch(
            "sglang.srt.entrypoints.openai.serving_responses.FunctionCallParser"
        ) as parser_cls:
            instance = parser_cls.return_value
            instance.has_tool_call.return_value = True
            instance.parse_non_stream.return_value = (
                "I'll check the weather.",
                [fake_call],
            )
            output_items = serving._make_response_output_items(
                self._function_tool_request(),
                "raw model output",
                tokenizer=Mock(),
                require_reasoning=False,
            )

        types = [type(item).__name__ for item in output_items]
        self.assertEqual(types, ["ResponseOutputMessage", "ResponseFunctionToolCall"])
        self.assertEqual(output_items[0].phase, "commentary")

    def test_required_tool_choice_parses_json_array_without_native_parser(self):
        serving = self.serving
        serving.tool_call_parser = None
        request = ResponsesRequest(
            model="x",
            input="hi",
            tool_choice="required",
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "parameters": {"type": "object"},
                }
            ],
            store=False,
        )
        raw = '[{"name": "get_weather", "parameters": {"city": "Beijing"}}]'

        output_items = serving._make_response_output_items(
            request, raw, tokenizer=Mock(), require_reasoning=False
        )

        tool_calls = [
            item for item in output_items if isinstance(item, ResponseFunctionToolCall)
        ]
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].name, "get_weather")
        self.assertEqual(tool_calls[0].arguments, '{"city": "Beijing"}')
        self.assertEqual(
            [item for item in output_items if isinstance(item, ResponseOutputMessage)],
            [],
        )

    def test_required_tool_choice_skips_json_fallback_for_native_parser(self):
        """muse reports parses_required_natively, so required output must not
        be pushed through the orjson JSON-array fallback (mirrors chat)."""
        serving = self.serving
        serving.tool_call_parser = "muse"
        request = ResponsesRequest(
            model="x",
            input="hi",
            tool_choice="required",
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "parameters": {"type": "object"},
                }
            ],
            store=False,
        )
        raw = '[{"name": "get_weather", "parameters": {"city": "Beijing"}}]'

        output_items = serving._make_response_output_items(
            request, raw, tokenizer=Mock(), require_reasoning=False
        )

        self.assertEqual(
            [
                item
                for item in output_items
                if isinstance(item, ResponseFunctionToolCall)
            ],
            [],
        )
        message_items = [
            item for item in output_items if isinstance(item, ResponseOutputMessage)
        ]
        self.assertEqual(len(message_items), 1)
        self.assertEqual(message_items[0].content[0].text, raw)

    def test_no_tool_call_extraction_when_tool_choice_none(self):
        serving = self.serving
        request = ResponsesRequest(
            model="x",
            input="hi",
            store=False,
            tool_choice="none",
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "parameters": {"type": "object"},
                }
            ],
        )

        with patch(
            "sglang.srt.entrypoints.openai.serving_responses.FunctionCallParser"
        ) as parser_cls:
            output_items = serving._make_response_output_items(
                request,
                "just a plain answer",
                tokenizer=Mock(),
                require_reasoning=False,
            )
            parser_cls.assert_not_called()

        self.assertEqual(len(output_items), 1)
        self.assertIsInstance(output_items[0], ResponseOutputMessage)


class HarmonyResponsesTestCase(CustomTestCase):
    def test_developer_message_skips_unsupported_tool_types(self):
        from sglang.srt.entrypoints.harmony_utils import get_developer_message
        from sglang.srt.entrypoints.openai.protocol import ResponseTool

        tools = [
            ResponseTool(
                type="function",
                name="get_weather",
                description="Look up weather.",
                parameters={"type": "object"},
            ),
            ResponseTool(type="web_search"),
            ResponseTool(type="namespace", name="codex"),
            ResponseTool(type="mcp"),
        ]
        msg = get_developer_message(instructions="be helpful", tools=tools)
        self.assertIsNotNone(msg)


class StatusFromFinishReasonTestCase(CustomTestCase):
    def test_only_length_maps_to_incomplete(self):
        fn = OpenAIServingResponses._status_from_finish_reason
        self.assertEqual(fn({"type": "length"}), "incomplete")
        self.assertEqual(fn("length"), "incomplete")
        for other in ({"type": "stop"}, {"type": "tool_calls"}, "stop", None):
            self.assertEqual(fn(other), "completed", other)


class BuildOutputTextLogprobsTestCase(CustomTestCase):
    def test_tokens_and_top_logprobs_are_converted(self):
        meta_info = {
            "output_token_logprobs": [(-0.1, 10, "Hello"), (-0.2, 11, " world")],
            "output_top_logprobs": [
                [(-0.1, 10, "Hello"), (-2.0, 12, "Hi")],
                [(-0.2, 11, " world"), (-3.0, 13, " earth")],
            ],
        }
        out = _build_output_text_logprobs(meta_info)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].token, "Hello")
        self.assertEqual(out[0].logprob, -0.1)
        self.assertEqual(out[0].bytes, list("Hello".encode("utf-8")))
        self.assertEqual(len(out[0].top_logprobs), 2)
        self.assertEqual(out[0].top_logprobs[0].token, "Hello")
        self.assertEqual(out[1].token, " world")

    def test_no_top_logprobs_yields_empty_lists(self):
        meta_info = {
            "output_token_logprobs": [(-0.5, 7, "hi")],
            "output_top_logprobs": None,
        }
        out = _build_output_text_logprobs(meta_info)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].top_logprobs, [])


class ChatToolChoiceConversionTestCase(CustomTestCase):
    def test_conversion(self):
        fn = OpenAIServingResponses._chat_tool_choice
        for s in ("auto", "required", "none"):
            self.assertEqual(fn(s), s)
        # Input is an effective_tool_choice() result, so the only object form
        # reaching here is a named function; degrading the rest to "auto"
        # happens there, once, so the echoed and the honored value agree.
        self.assertEqual(
            fn({"type": "function", "name": "get_weather"}),
            {"type": "function", "function": {"name": "get_weather"}},
        )


class ShouldEmitNormalTextTestCase(CustomTestCase):
    def test_whitespace_suppressed_only_while_a_tool_is_open(self):
        emit = _should_emit_normal_text_as_message
        self.assertFalse(emit("", any_tool_call_in_progress=False))
        # whitespace between tool blocks is an inter-call separator, not content
        self.assertFalse(emit("\n", any_tool_call_in_progress=True))
        self.assertTrue(emit("\n", any_tool_call_in_progress=False))
        self.assertTrue(emit("hello", any_tool_call_in_progress=True))


class EnginePassthroughTestCase(CustomTestCase):
    """Both flags cross hops with no type contract, and dropping either fails
    silently."""

    def _capture(self, serving, request, raw_request=None):
        # Let the real _process_messages run: it is the hop that turns
        # skip_special_tokens off, so mocking it would make that assertion vacuous.
        # chat_template_name=None routes it through the tokenizer's template
        # (mocked) instead of the conversation registry, which has no fixture entry.
        serving.default_chat_template_kwargs = {}
        serving.template_manager.chat_template_name = None
        captured = {}

        async def fake_generate(
            request_id,
            request_prompt,
            adapted_request,
            sampling_params,
            context,
            **kwargs,
        ):
            captured["adapted_request"] = adapted_request
            captured["sampling_params"] = sampling_params
            context.append_output(
                {
                    "text": "ok",
                    "meta_info": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "cached_tokens": 0,
                    },
                }
            )
            yield context

        serving._generate_with_builtin_tools = fake_generate
        asyncio.run(serving.create_responses(request, raw_request=raw_request))
        return captured

    def test_pd_routing_fields_forwarded_to_engine(self):
        serving = make_serving()
        raw_request = Mock(headers={"x-data-parallel-rank": "2"}, state=Mock())

        captured = self._capture(
            serving,
            ResponsesRequest(
                model="x",
                input="hi",
                bootstrap_host="10.0.0.1",
                bootstrap_port=8998,
                bootstrap_room=42,
                routed_dp_rank=1,
                disagg_prefill_dp_rank=0,
                store=False,
            ),
            raw_request=raw_request,
        )

        adapted_request = captured["adapted_request"]
        self.assertEqual(adapted_request.bootstrap_host, "10.0.0.1")
        self.assertEqual(adapted_request.bootstrap_port, 8998)
        self.assertEqual(adapted_request.bootstrap_room, 42)
        self.assertEqual(adapted_request.routed_dp_rank, 2)
        self.assertEqual(adapted_request.disagg_prefill_dp_rank, 0)

    def test_require_reasoning_forwarded_when_reasoning_parser_configured(self):
        serving = make_serving()
        serving.reasoning_parser = "deepseek-r1"
        serving.template_manager.reasoning_config = ReasoningToggleConfig(
            toggle_param="thinking", default_enabled=True
        )

        captured = self._capture(
            serving, ResponsesRequest(model="x", input="hi", store=False)
        )

        self.assertTrue(captured["adapted_request"].require_reasoning)

    def test_prefilled_think_template_opens_the_parser(self):
        """``force_reasoning`` is a template property, not a request one, so it
        drives the parser but never the engine flag -- as on the chat path."""
        serving = make_serving()
        serving.reasoning_parser = "deepseek-r1"
        serving.template_manager.force_reasoning = True

        with patch(
            "sglang.srt.entrypoints.openai.serving_responses.ReasoningParser"
        ) as parser_cls:
            parser_cls.return_value.parse_non_stream.return_value = (None, "hi")
            serving._make_response_output_items(
                ResponsesRequest(model="x", input="hi", store=False),
                "hi",
                tokenizer=Mock(),
                require_reasoning=False,
            )

        self.assertTrue(parser_cls.call_args.kwargs["force_reasoning"])

    def test_require_reasoning_false_without_reasoning_parser(self):
        serving = make_serving()
        serving.reasoning_parser = None

        captured = self._capture(
            serving, ResponsesRequest(model="x", input="hi", store=False)
        )

        self.assertFalse(captured["adapted_request"].require_reasoning)

    def test_skip_special_tokens_disabled_for_tool_requests(self):
        # _process_messages turns it off so tool-call markers survive detokenize;
        # create_responses must re-apply it to the engine sampling dict.
        serving = make_serving()
        serving.tool_call_parser = "qwen25"

        captured = self._capture(
            serving,
            ResponsesRequest(
                model="x",
                input="weather",
                store=False,
                tools=[
                    {
                        "type": "function",
                        "name": "get_weather",
                        "parameters": {"type": "object"},
                    }
                ],
            ),
        )

        self.assertFalse(captured["sampling_params"]["skip_special_tokens"])


class CancelIdempotencyTestCase(CustomTestCase):
    def test_cancelling_a_terminal_response_returns_it_not_an_error(self):
        publish(
            ServerArgs(model_path="dummy", enable_response_store=True), role="tokenizer"
        )
        from sglang.srt.entrypoints.openai.protocol import ResponsesResponse

        for status in ("cancelled", "completed"):
            serving = make_serving()
            resp = ResponsesResponse.from_request(
                ResponsesRequest(model="x", input="hi", store=False),
                sampling_params={},
                model_name="x",
                created_time=0,
                output=[],
                status=status,
                usage=None,
            )
            serving.response_store[resp.id] = resp

            out = asyncio.run(serving.cancel_responses(resp.id))

            self.assertIs(out, resp, status)
            self.assertEqual(out.status, status)


class StreamingLogprobsRejectionTestCase(CustomTestCase):
    def test_stream_with_logprobs_include_rejected(self):
        import orjson

        serving = make_serving()
        request = ResponsesRequest(
            model="x",
            input="hi",
            store=False,
            stream=True,
            include=["message.output_text.logprobs"],
        )
        result = asyncio.run(serving.create_responses(request))
        self.assertEqual(result.status_code, 400)
        body = orjson.loads(result.body)
        self.assertIn("streaming mode", body["error"]["message"])


STORE_DISABLED_MESSAGE = (
    "Response store is disabled. Stateful Responses require "
    "--enable-response-store on a standalone server; response storage "
    "is unavailable in PD mode."
)
STORE_PD_MESSAGE = (
    "--enable-response-store is not supported with "
    "--disaggregation-mode=prefill or decode; response storage must "
    "remain disabled in PD mode."
)


@pytest.fixture(autouse=True)
def isolated_response_config():
    reset_context()
    yield
    reset_context()


@pytest.fixture
def response_serving():
    def build(enabled=False, mode="null", harmony=False):
        reset_context()
        publish(
            ServerArgs(
                model_path="dummy",
                enable_response_store=enabled,
                disaggregation_mode=mode,
            ),
            role="tokenizer",
        )
        serving = make_serving()
        serving.use_harmony = harmony
        serving.default_chat_template_kwargs = {}
        serving.template_manager.chat_template_name = None
        serving.template_manager.jinja_template_content_format = "string"
        serving.tokenizer_manager.tokenizer.apply_chat_template.return_value = [1, 2, 3]
        serving.tokenizer_manager.abort_request = Mock()
        serving.reasoning_parser = None
        serving.tool_call_parser = None

        async def generate(*args, **kwargs):
            chunk = engine_chunk("ok", finish=True)
            if harmony:
                chunk["output_ids"] = get_encoding().render_conversation(
                    Conversation.from_messages(
                        [
                            Message.from_role_and_content(
                                Role.ASSISTANT, "ok"
                            ).with_channel("final")
                        ]
                    )
                )
                chunk["meta_info"]["completion_tokens"] = len(chunk["output_ids"])
            yield chunk

        serving.tokenizer_manager.generate_request = Mock(side_effect=generate)
        return serving

    return build


async def create_response_result(serving, request):
    result = await serving.create_responses(request)
    if request.stream:
        payloads = event_payloads([event async for event in result])
        assert payloads[0]["type"] == "response.created"
        assert payloads[-1]["type"] == "response.completed"
        return ResponsesResponse.model_validate(payloads[-1]["response"])
    return result


def assert_response_error(response, param, message=STORE_DISABLED_MESSAGE, status=400):
    assert response.status_code == status
    assert orjson.loads(response.body) == {
        "error": {
            "message": message,
            "type": "invalid_request_error",
            "param": param,
            "code": status,
        }
    }


@pytest.mark.parametrize(
    "mode,enabled", [("null", False), ("null", True), ("decode", True)]
)
def test_response_store_configuration(mode, enabled):
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    argv = ["--model-path", "dummy", "--disaggregation-mode", mode]
    if enabled:
        argv.append("--enable-response-store")
    args = ServerArgs.from_cli_args(parser.parse_args(argv))
    if mode != "null":
        with pytest.raises(ValueError) as error:
            publish(args, role="tokenizer")
        assert str(error.value) == STORE_PD_MESSAGE
    else:
        publish(args, role="tokenizer")
        serving = make_serving()
        assert get_serving().enable_response_store is enabled
        assert serving.enable_response_store is enabled
        assert not serving.is_disaggregated


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_response_store_persistence_and_continuation(response_serving, stream, enabled):
    serving = response_serving(enabled=enabled, harmony=not stream)

    async def run():
        first = await create_response_result(
            serving, ResponsesRequest(model="x", input="first", stream=stream)
        )
        assert first.status == "completed"
        assert first.output[0].content[0].text == "ok"
        assert first.store is True
        assert not serving.background_tasks
        if not enabled:
            assert not serving.response_store and not serving.msg_store
            return
        assert serving.response_store[first.id].output == first.output
        history = list(serving.msg_store[first.id])
        assert len(history) == 2
        second = await create_response_result(
            serving,
            ResponsesRequest(
                model="x",
                input="next",
                previous_response_id=first.id,
                store=False,
                stream=stream,
            ),
        )
        if serving.use_harmony:
            generated_request = (
                serving.tokenizer_manager.generate_request.call_args.args[0]
            )
            prompt = get_encoding().decode(generated_request.input_ids)
        else:
            prompt = str(
                serving.tokenizer_manager.tokenizer.apply_chat_template.call_args.args[
                    0
                ]
            )
        assert all(text in prompt for text in ("first", "ok", "next"))
        assert second.status == "completed"
        assert second.output[0].content[0].text == "ok"
        assert second.store is False
        assert set(serving.response_store) == {first.id}
        assert set(serving.msg_store) == {first.id}
        assert serving.msg_store[first.id] == history

    asyncio.run(run())


@pytest.mark.parametrize(
    "enabled,fields,param,message",
    [
        (
            False,
            {"previous_response_id": ""},
            "previous_response_id",
            STORE_DISABLED_MESSAGE,
        ),
        (False, {}, "background", STORE_DISABLED_MESSAGE),
        (True, {}, "store", "background=true requires store=true."),
    ],
    ids=["empty-predecessor", "disabled-background", "background-without-store"],
)
def test_response_store_admission(response_serving, enabled, fields, param, message):
    serving = response_serving(enabled=enabled, mode="null" if enabled else "prefill")
    serving._make_request = AsyncMock(side_effect=AssertionError("prompt reached"))
    serving.response_store = Mock()
    serving.msg_store = Mock()
    request = ResponsesRequest(
        model="unknown", input="hi", background=True, stream=True, store=None, **fields
    )
    assert_response_error(
        asyncio.run(serving.create_responses(request)), param, message
    )
    serving._make_request.assert_not_called()
    serving.tokenizer_manager.generate_request.assert_not_called()
    assert not serving.response_store.mock_calls
    assert not serving.msg_store.mock_calls
    assert not serving.background_tasks


@pytest.mark.parametrize("enabled", [False, True])
def test_response_store_read_endpoints(response_serving, enabled):
    serving = response_serving(enabled=enabled)

    async def run():
        if not enabled:
            serving.response_store = Mock()
            serving.background_tasks = Mock()
            for method in (serving.retrieve_responses, serving.cancel_responses):
                assert_response_error(await method("bad"), "response_id")
            assert not serving.response_store.mock_calls
            assert not serving.background_tasks.mock_calls
        else:
            response = await create_response_result(
                serving, ResponsesRequest(model="x", input="hi")
            )
            assert await serving.retrieve_responses(response.id) is response
            for response_id, status in (("bad", 400), ("resp_missing", 404)):
                for method in (serving.retrieve_responses, serving.cancel_responses):
                    assert (await method(response_id)).status_code == status
                request = ResponsesRequest(
                    model="x", input="hi", previous_response_id=response_id
                )
                assert (await serving.create_responses(request)).status_code == status
        serving.tokenizer_manager.abort_request.assert_not_called()

    asyncio.run(run())


@pytest.mark.parametrize(
    "outcome", ["success", "failure", "cancel_queued", "cancel_running"]
)
def test_background_task_lifecycle(response_serving, outcome):
    serving = response_serving(enabled=True)
    original_generate = serving.tokenizer_manager.generate_request

    async def run():
        entered, release = asyncio.Event(), asyncio.Event()

        async def generate(*args, **kwargs):
            entered.set()
            await release.wait()
            if outcome == "failure":
                raise ValueError("generation failed")
            async for chunk in original_generate(*args, **kwargs):
                yield chunk

        serving.tokenizer_manager.generate_request = generate
        request = ResponsesRequest(model="x", input="hi", background=True)
        queued = await serving.create_responses(request)
        assert queued.status == "queued"
        assert serving.response_store[queued.id] is queued
        assert request.request_id in serving.msg_store
        task = serving.background_tasks[queued.id]
        if outcome != "cancel_queued":
            await entered.wait()
            assert (await serving.retrieve_responses(queued.id)).status == "in_progress"
        if outcome.startswith("cancel"):
            cancelled = await serving.cancel_responses(queued.id)
            assert cancelled.status == "cancelled"
            serving.tokenizer_manager.abort_request.assert_called_once_with(
                rid=queued.id
            )
            assert task.done()
            if outcome == "cancel_queued":
                assert task.cancelled()
        else:
            release.set()
            await task
            assert serving.response_store[queued.id].status == (
                "failed" if outcome == "failure" else "completed"
            )
        assert not serving.background_tasks

    asyncio.run(run())


def test_active_stream_cancel_and_final_history(response_serving):
    serving = response_serving(enabled=True, harmony=True)

    async def run():
        request = ResponsesRequest(model="x", input="hi", background=True, stream=True)
        stream = await serving.create_responses(request)
        assert "response.created" in await anext(stream)
        assert not serving.background_tasks
        assert (await serving.cancel_responses(request.request_id)).status_code == 404
        serving.tokenizer_manager.abort_request.assert_not_called()
        events = [event async for event in stream]
        assert event_payloads(events)[-1]["type"] == "response.completed"
        assert serving.response_store[request.request_id].status == "completed"
        assert len(serving.msg_store[request.request_id]) == 2

    asyncio.run(run())


INLINE_SYSTEM_TEMPLATE = (
    "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}{% endfor %}"
)


@pytest.mark.parametrize(
    "encoding_spec, chat_template_name, jinja_template, preserve",
    [
        ("kimi_k3", None, None, True),
        ("dsv32", None, INLINE_SYSTEM_TEMPLATE, False),
        (None, None, INLINE_SYSTEM_TEMPLATE, True),
        (None, "llama-3", INLINE_SYSTEM_TEMPLATE, False),
        (None, None, "{{ messages[0]['content'] }}", False),
    ],
)
def test_inline_instruction_keeps_history_prefix(
    encoding_spec, chat_template_name, jinja_template, preserve
):
    """An appended developer instruction stays in place when the renderer supports
    inline system; hoisting it rewrites the prompt prefix and defeats KV reuse."""
    publish(ServerArgs(model_path="dummy"), role="tokenizer")
    tokenizer_manager = MockTokenizerManager()
    tokenizer_manager.tokenizer.chat_template = jinja_template
    template_manager = MockTemplateManager()
    template_manager.chat_template_name = chat_template_name
    with patch.object(
        OpenAIServingResponses,
        "_resolve_chat_encoding_spec",
        return_value=encoding_spec,
    ):
        serving = OpenAIServingResponses(tokenizer_manager, template_manager)
    history = [
        {"role": "user", "content": "Existing conversation"},
        {"role": "assistant", "content": "Previous answer"},
    ]
    appended = [
        {"role": "developer", "content": "New instruction"},
        {"role": "user", "content": "Next turn"},
    ]
    before = serving._construct_input_messages(
        ResponsesRequest(model="x", instructions="Stable instructions", input=history)
    )
    after = serving._construct_input_messages(
        ResponsesRequest(
            model="x", instructions="Stable instructions", input=history + appended
        )
    )
    if preserve:
        assert after[: len(before)] == before
        assert after[len(before)]["role"] == "system"
        assert "New instruction" in after[len(before)]["content"]
    else:
        assert [m["role"] for m in after].count("system") == 1
        assert "New instruction" in after[0]["content"]


@pytest.mark.parametrize("stream", [False, True])
def test_completion_preserves_cancelled_response(response_serving, stream):
    serving = response_serving(enabled=True, harmony=not stream)
    request = ResponsesRequest(model="x", input="hi", stream=stream)
    cancelled = ResponsesResponse.from_request(
        request,
        {},
        model_name="x",
        created_time=0,
        output=[],
        status="cancelled",
        usage=None,
    )
    serving.response_store[request.request_id] = cancelled
    serving.msg_store[request.request_id] = ["original"]
    if stream:
        StreamFixture(serving, request).run([engine_chunk("ok", finish=True)])
    else:
        messages = serving._construct_input_messages_with_harmony(request, None)
        context = HarmonyContext(messages, {})

        async def generate():
            async for chunk in serving.tokenizer_manager.generate_request():
                context.append_output(chunk)
                yield context

        response = asyncio.run(
            serving.responses_full_generator(
                request,
                {},
                generate(),
                context,
                "x",
                Mock(),
                RequestResponseMetadata(request_id=request.request_id),
                require_reasoning=False,
            )
        )
        assert response.status == "completed"
    assert serving.response_store[request.request_id] is cancelled
    assert serving.msg_store[request.request_id] == ["original"]


def test_pd_builtin_tool_admission(response_serving):
    serving = response_serving(mode="prefill", harmony=True)
    serving.tool_server = Mock()
    serving.supports_browsing = True
    request = ResponsesRequest(model="x", input="hi", tools=[{"type": "web_search"}])
    result = asyncio.run(serving.create_responses(request))
    assert result.status_code == 400
    assert orjson.loads(result.body)["error"]["param"] == "tools"
    serving.tool_server.get_tool_session.assert_not_called()
    serving.tokenizer_manager.generate_request.assert_not_called()


def test_standalone_builtin_tools_without_storage(response_serving):
    serving = response_serving(harmony=True)
    tool_session = Mock()
    tool_session.call_tool = AsyncMock(
        return_value=Mock(content=[Mock(text="Search result: 42")])
    )

    @asynccontextmanager
    async def session(name):
        yield tool_session

    turns = iter(
        [
            Message.from_role_and_content(Role.ASSISTANT, '{"query":"answer"}')
            .with_channel("commentary")
            .with_recipient("browser.search"),
            Message.from_role_and_content(
                Role.ASSISTANT, "The answer is 42."
            ).with_channel("final"),
        ]
    )

    async def generate(*args, **kwargs):
        chunk = engine_chunk("", finish=True)
        chunk["output_ids"] = get_encoding().render_conversation(
            Conversation.from_messages([next(turns)])
        )
        if serving.tokenizer_manager.generate_request.call_count == 1:
            # The parser starts inside the prompt's open assistant header.
            chunk["output_ids"] = chunk["output_ids"][2:]
        chunk["meta_info"]["completion_tokens"] = len(chunk["output_ids"])
        yield chunk

    serving.tokenizer_manager.generate_request = Mock(side_effect=generate)
    serving.tool_server = Mock()
    serving.tool_server.get_tool_session = session
    serving.tool_server.get_tool_description.return_value = ToolNamespaceConfig(
        name="browser", description="Browser", tools=[]
    )
    serving.supports_browsing = True
    request = ResponsesRequest(model="x", input="hi", tools=[{"type": "web_search"}])
    response = asyncio.run(create_response_result(serving, request))
    assert isinstance(response, ResponsesResponse), response.body
    tool_session.call_tool.assert_awaited_once_with("search", {"query": "answer"})
    assert response.status == "completed"
    assert response.output[-1].content[0].text == "The answer is 42."
    assert serving.tokenizer_manager.generate_request.call_count == 2
    continuation = serving.tokenizer_manager.generate_request.call_args_list[1].args[0]
    assert "Search result: 42" in get_encoding().decode(continuation.input_ids)
    assert not serving.msg_store and not serving.response_store


def test_pd_tool_continuation_stops_before_side_effect(response_serving):
    serving = response_serving(mode="decode")
    context = Mock()
    context.need_builtin_tool_call.return_value = True
    context.call_tool = AsyncMock()

    async def run():
        async for _ in serving._generate_with_builtin_tools(
            "resp_tool", "hi", Mock(), {}, context
        ):
            pass

    with pytest.raises(ValueError, match="disaggregation"):
        asyncio.run(run())
    context.call_tool.assert_not_awaited()
    assert serving.tokenizer_manager.generate_request.call_count == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
