"""Request logs abbreviate huge grammar constraints (e.g. tool-call structural tags)."""

import json
import os
import unittest
from unittest import mock

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import GenerateReqInput  # noqa: E402
from sglang.srt.utils import request_logger  # noqa: E402
from sglang.srt.utils.request_logger import RequestLogger  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

_STRUCTURAL_TAG = json.dumps({"type": "structural_tag", "tools": ["x" * 50] * 400})


def _request() -> GenerateReqInput:
    return GenerateReqInput(
        rid="rid-1",
        input_ids=[1, 2, 3],
        sampling_params={
            "max_new_tokens": 8,
            "structural_tag": _STRUCTURAL_TAG,
            "regex": "[a-z]+",
        },
    )


def _logged(level: int, fmt: str = "json", env=None):
    logger = None
    captured = []
    with mock.patch.dict(os.environ, env or {}):
        logger = RequestLogger(
            log_requests=True,
            log_requests_level=level,
            log_requests_format=fmt,
            log_requests_target=None,
        )
    with (
        mock.patch.object(
            request_logger,
            "log_json",
            lambda targets, event, data: captured.append(data),
        ),
        mock.patch.object(logger, "_log", captured.append),
    ):
        logger.log_received_request(_request())
    return captured[0]


class TestRequestLogAbbreviation(CustomTestCase):
    def test_level_1_abbreviates_long_grammar_fields(self):
        params = _logged(level=1)["obj"]["sampling_params"]
        self.assertTrue(params["structural_tag"].startswith("<structural_tag: "))
        self.assertIn(f"{len(_STRUCTURAL_TAG)} chars", params["structural_tag"])
        self.assertIn("blake2b=", params["structural_tag"])
        self.assertEqual(params["regex"], "[a-z]+")  # short values stay verbatim
        self.assertEqual(params["max_new_tokens"], 8)

    def test_same_value_same_digest(self):
        first = _logged(level=1)["obj"]["sampling_params"]["structural_tag"]
        second = _logged(level=2)["obj"]["sampling_params"]["structural_tag"]
        self.assertEqual(first, second)

    def test_level_3_and_empty_field_list_log_verbatim(self):
        self.assertEqual(
            _logged(level=3)["obj"]["sampling_params"]["structural_tag"],
            _STRUCTURAL_TAG,
        )
        verbatim = _logged(level=1, env={"SGLANG_LOG_REQUEST_ABBREVIATE_FIELDS": ""})
        self.assertEqual(
            verbatim["obj"]["sampling_params"]["structural_tag"], _STRUCTURAL_TAG
        )

    def test_text_format(self):
        line = _logged(level=1, fmt="text")
        self.assertIn("<structural_tag: ", line)
        self.assertNotIn(_STRUCTURAL_TAG, line)


if __name__ == "__main__":
    unittest.main()
