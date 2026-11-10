"""Unit test for DetokenizerManager.trim_matched_stop.

Under speculative decoding the output handed to trim_matched_stop may carry
content after the matched stop. With no_stop_trim the stop string must be kept
but the trailing over-generation still dropped (`output[:end]`); without it the
stop is removed (`output[:pos]`). Pure CPU: calls the method with a stub self,
so no DetokenizerManager.__init__ / IPC / tokenizer."""

import unittest
from array import array
from itertools import product
from types import SimpleNamespace

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.function_call.kimik3_format import RESPONSE_OPEN
from sglang.srt.managers.detokenizer_manager import DetokenizerManager
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler_components.output_streamer import (
    _GenerationStreamAccumulator,
)
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

GPT_OSS_CALL_TOKEN = 200012


def _trim(output, matched, no_stop_trim, *, gpt_oss=False):
    stub = SimpleNamespace(is_tool_call_parser_gpt_oss=gpt_oss)
    finished_reason = None if matched is None else {"matched": matched}
    return DetokenizerManager.trim_matched_stop(
        stub, output, finished_reason, no_stop_trim
    )


class TestTrimMatchedStop(unittest.TestCase):
    def test_no_finished_reason_returns_output(self):
        self.assertEqual(_trim("abc", None, False), "abc")

    def test_no_matched_returns_output(self):
        stub = SimpleNamespace(is_tool_call_parser_gpt_oss=False)
        self.assertEqual(
            DetokenizerManager.trim_matched_stop(stub, "abc", {}, False), "abc"
        )

    # --- stop string ---
    def test_str_trim_removes_stop(self):
        # no_stop_trim=False: drop the stop string and anything after it.
        self.assertEqual(_trim("ans\n\nQuestion: A", "Question", False), "ans\n\n")

    def test_str_no_trim_keeps_stop_but_drops_trailing(self):
        # no_stop_trim=True: keep through the stop, drop the over-generated tail.
        self.assertEqual(
            _trim("ans\n\nQuestion: A", "Question", True), "ans\n\nQuestion"
        )

    def test_str_not_found_returns_output(self):
        self.assertEqual(_trim("no stop here", "Question", False), "no stop here")

    # --- stop token ---
    def test_token_trim_drops_last(self):
        self.assertEqual(_trim([1, 2, 3], 3, False), [1, 2])

    def test_token_no_trim_keeps_all(self):
        self.assertEqual(_trim([1, 2, 3], 3, True), [1, 2, 3])

    def test_token_zero_is_a_stop_token(self):
        # Token id 0 is falsy; the guard checks `matched is None`, not truthiness.
        self.assertEqual(_trim([123, 0], 0, False), [123])
        self.assertEqual(_trim([123, 0], 0, True), [123, 0])

    def test_matched_none_inside_a_finished_reason_returns_output(self):
        stub = SimpleNamespace(is_tool_call_parser_gpt_oss=False)
        trimmed = DetokenizerManager.trim_matched_stop(
            stub, [123, 0], {"type": "stop", "matched": None}, False
        )
        self.assertEqual(trimmed, [123, 0])

    def test_token_gpt_oss_call_kept(self):
        # gpt-oss tool-call token is also an eos; keep it even when trimming.
        self.assertEqual(
            _trim([1, 2, GPT_OSS_CALL_TOKEN], GPT_OSS_CALL_TOKEN, False, gpt_oss=True),
            [1, 2, GPT_OSS_CALL_TOKEN],
        )


class _ReasoningTokenizer:
    eos_token_id = 5
    additional_stop_token_ids = None
    all_special_ids = [9]
    is_fast = True
    tokens = {
        0: "prompt context",
        1: "需要列出苹果、香蕉、",
        2: "橘子",
        3: "<|close|>think",
        4: "<|sep|>",
        5: "",
        6: RESPONSE_OPEN,
        7: "苹果、香蕉、",
        8: "橘子橘子",
        9: "<special>",
        10: "橘",
        11: "子",
    }

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text)))

    def decode(self, ids, *, skip_special_tokens=False, **kwargs):
        return "".join(
            self.tokens[int(i)]
            for i in ids
            if not (skip_special_tokens and i in self.all_special_ids)
        )

    def batch_decode(self, batches, **kwargs):
        return [self.decode(ids, **kwargs) for ids in batches]


def _make_reasoning_req(
    *, stop="橘子", stop_regex=None, stop_token_ids=None, max_new_tokens=128
):
    tokenizer = _ReasoningTokenizer()
    sampling_params = SamplingParams(
        stop=stop,
        stop_regex=stop_regex,
        stop_token_ids=stop_token_ids,
        max_new_tokens=max_new_tokens,
    )
    sampling_params.normalize(tokenizer=tokenizer)
    req = Req(
        rid="reasoning-stop",
        origin_input_text="",
        origin_input_ids=array("q", [0]),
        sampling_params=sampling_params,
        require_reasoning=True,
        vocab_size=100,
    )
    req.tokenizer = tokenizer
    return req


def _accept_token(req, token):
    req.output_ids.append(token)
    req.update_reasoning_tokens(token, [3, 4])
    req.update_finish_state()


class TestReasoningStop(CustomTestCase):
    """Non-speculative decoding only; speculative compatibility is out of scope."""

    def test_user_stops_wait_until_after_thinking(self):
        """Quoted stops and the thinking terminator must not end generation."""
        for params, matched in (
            ({"stop": "橘子"}, "橘子"),
            ({"stop": None, "stop_regex": "橘."}, "橘."),
            ({"stop": None, "stop_token_ids": {2, 4}}, 2),
        ):
            with self.subTest(params=params):
                req = _make_reasoning_req(**params)
                for token in (1, 2, 3, 4, 6, 7):
                    _accept_token(req, token)
                    self.assertFalse(req.finished())
                    self.assertFalse(req.check_match_stop_str_prefix())
                _accept_token(req, 2)
                self.assertEqual(
                    req.finished_reason.to_json(), {"type": "stop", "matched": matched}
                )
                self.assertEqual(
                    list(req.output_ids_through_stop), [1, 2, 3, 4, 6, 7, 2]
                )

    def test_empty_matching_stops_wait_during_thinking(self):
        for params in ({"stop": ""}, {"stop": None, "stop_regex": "X?"}):
            with self.subTest(params=params):
                req = _make_reasoning_req(**params)
                _accept_token(req, 1)
                self.assertFalse(req.finished())

    def test_eos_still_finishes_thinking(self):
        """System EOS remains effective even if also listed as a user stop token."""
        req = _make_reasoning_req(stop=None, stop_token_ids={2, 5})
        for token in (1, 2):
            _accept_token(req, token)
            self.assertFalse(req.finished())
        _accept_token(req, 5)
        self.assertEqual(req.finished_reason.to_json(), {"type": "stop", "matched": 5})

    def test_length_cap_still_finishes_thinking(self):
        req = _make_reasoning_req(max_new_tokens=2)
        _accept_token(req, 1)
        self.assertFalse(req.finished())
        _accept_token(req, 2)
        self.assertEqual(req.finished_reason.to_json(), {"type": "length", "length": 2})

    def test_stream_withholds_content_stop_prefix(self):
        """Thinking can stream a quoted stop; an incomplete content stop must wait."""
        req = _make_reasoning_req()
        for token in (1, 2, 3, 4, 6):
            _accept_token(req, token)
            self.assertFalse(req.finished())
            self.assertFalse(req.check_match_stop_str_prefix())
        _accept_token(req, 10)
        self.assertFalse(req.finished())
        self.assertTrue(req.check_match_stop_str_prefix())
        _accept_token(req, 11)
        self.assertEqual(
            req.finished_reason.to_json(), {"type": "stop", "matched": "橘子"}
        )

    def test_kimi_stop_keeps_thinking_and_answer_prefix(self):
        """Stop trimming must preserve thinking and stop at the first content match."""
        for stream, no_stop_trim, decoder, (answer_tokens, answer_prefix) in product(
            (False, True),
            (False, True),
            ("batch", "single", "slow"),
            (
                ((7, 2), "苹果、香蕉、"),
                ((7, 8), "苹果、香蕉、"),
                ((7, 10, 11), "苹果、香蕉、"),
                ((2,), ""),
            ),
        ):
            with self.subTest(
                stream=stream,
                no_stop_trim=no_stop_trim,
                decoder=decoder,
                answer_tokens=answer_tokens,
            ):
                req = _make_reasoning_req()
                req.stream = stream
                req.sampling_params.no_stop_trim = no_stop_trim
                req.tokenizer.is_fast = decoder != "slow"
                detokenizer = DetokenizerManager.__new__(DetokenizerManager)
                detokenizer.tokenizer = req.tokenizer
                detokenizer.vocab_size = req.vocab_size
                detokenizer.decode_status = {}
                detokenizer.disable_tokenizer_batch_decode = decoder == "single"
                detokenizer.is_tool_call_parser_gpt_oss = False
                emitted = []
                for token in (9, 1, 2, 3, 4, 6, *answer_tokens):
                    self.assertFalse(req.finished())
                    _accept_token(req, token)
                    accumulator = _GenerationStreamAccumulator(
                        return_logprob=False,
                        return_hidden_states=False,
                        return_routed_experts=False,
                        return_indexer_topk=False,
                        spec_algorithm=SpeculativeAlgorithm.NONE,
                        disaggregation_mode=DisaggregationMode.NULL,
                        default_stream_interval=1,
                        default_force_stream_interval=1000,
                        get_cached_tokens_details=lambda req: None,
                        current_weight_version=None,
                    )
                    accumulator.accept(req=req)
                    payload = accumulator.to_payload(dp_rank=0, is_idle_batch=False)
                    if payload is not None:
                        output = detokenizer.handle_batch_token_id_out(payload)
                        emitted.extend(output.output_strs)

                parser = ReasoningParser(model_type="kimi_k3", force_reasoning=True)
                if stream:
                    reasoning, content = "", ""
                    for chunk in emitted:
                        reasoning_chunk, content_chunk = parser.parse_stream_chunk(
                            chunk
                        )
                        reasoning += reasoning_chunk or ""
                        content += content_chunk or ""
                    reasoning_chunk, content_chunk = parser.parse_stream_end()
                    reasoning += reasoning_chunk or ""
                    content += content_chunk or ""
                else:
                    reasoning, content = parser.parse_non_stream("".join(emitted))
                self.assertEqual(reasoning, "需要列出苹果、香蕉、橘子")
                self.assertEqual(
                    content, answer_prefix + ("橘子" if no_stop_trim else "")
                )
                self.assertEqual(
                    req.finished_reason.to_json(), {"type": "stop", "matched": "橘子"}
                )


if __name__ == "__main__":
    unittest.main()
