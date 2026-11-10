"""Request validation errors must not echo the request back.

FastAPI attaches the offending value to every validation error as ``input``;
for a chat request that is the whole message list, once per Union branch.
Stringifying it ran on the HTTP event loop and stalled token delivery for
every stream on that tokenizer worker. These tests pin that the 400 body stays
small, valid JSON and free of the request content, while still naming the
field that failed.
"""

import asyncio
import json
import unittest

from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from starlette.requests import Request

from sglang.srt.entrypoints import http_server
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# sgl-model-gateway truncates upstream error bodies beyond this many bytes.
GATEWAY_ERROR_BODY_CAP = 2048
MARKER = "QQQQQQQQQQQQQQQQ"


def _handle(path, errors):
    scope = {
        "type": "http",
        "method": "POST",
        "path": path,
        "root_path": "",
        "query_string": b"",
        "headers": [],
    }
    response = asyncio.run(
        http_server.validation_exception_handler(
            Request(scope), RequestValidationError(errors)
        )
    )
    return response.status_code, response.body


class TestValidationErrorResponse(CustomTestCase):
    def test_chat_completions_names_bad_field_without_echoing_request(self):
        body = {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": MARKER * 250_000},
                        {"type": "image_url", "image_url": 123},
                    ],
                }
            ],
        }
        response = TestClient(http_server.app).post("/v1/chat/completions", json=body)

        self.assertEqual(response.status_code, 400)
        self.assertLess(len(response.content), GATEWAY_ERROR_BODY_CAP)
        self.assertNotIn(MARKER, response.text)
        message = response.json()["message"]
        self.assertIn(
            "image_url: Input should be a valid dictionary or object", message
        )
        self.assertNotIn("function-after[", message)

    def test_responses_endpoint_uses_digest(self):
        errors = [
            {
                "type": "string_type",
                "loc": ("body", "input"),
                "msg": "Input should be a valid string",
                "input": [{"content": MARKER * 100_000}],
            }
        ]
        status, raw = _handle("/v1/responses", errors)
        self.assertEqual(status, 400)
        self.assertEqual(
            json.loads(raw)["error"]["message"], "input: Input should be a valid string"
        )

    def test_small_error_is_kept_intact_on_every_route(self):
        errors = [
            {"type": "missing", "loc": ("body", "messages"), "msg": "Field required"}
        ]
        _, raw = _handle("/v1/chat/completions", errors)
        self.assertEqual(json.loads(raw)["message"], "messages: Field required")
        _, raw = _handle("/v1/messages", errors)
        payload = json.loads(raw)
        self.assertEqual(payload["error"]["type"], "invalid_request_error")
        self.assertEqual(payload["error"]["message"], "messages: Field required")

    def test_body_stays_under_gateway_cap_for_wide_characters(self):
        # loc can carry client-controlled dict keys (e.g. logit_bias); CJK is
        # 3 bytes in UTF-8 and control characters are escaped to 6 bytes.
        errors = [
            {
                "type": "int_parsing",
                "loc": ("body", "logit_bias", f"{i}" + "中\x01" * 500),
                "msg": "Input should be a valid integer",
            }
            for i in range(50)
        ]
        status, raw = _handle("/v1/chat/completions", errors)
        self.assertEqual(status, 400)
        self.assertLess(len(raw), GATEWAY_ERROR_BODY_CAP)
        self.assertTrue(json.loads(raw)["message"].endswith("…"))


if __name__ == "__main__":
    unittest.main()
