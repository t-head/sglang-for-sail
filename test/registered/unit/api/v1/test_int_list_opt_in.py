"""Import-time isolation and semantic parity for optional integer-list decoding."""

import importlib.util
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import msgspec

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

SOURCE = Path(__file__).resolve().parents[5] / "python/sglang/api/v1/api_types.py"
FLAG = "SGLANG_OPT_API_INT_LIST"
DECODERS = ("decode_TokenIds", "decode_Int64List", "decode_MatchedTokens")


def load_types(value):
    name = "_int_list_opt_in_test_types"
    spec = importlib.util.spec_from_file_location(name, SOURCE)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(os.environ), patch.dict(sys.modules, {name: module}):
        os.environ.pop(FLAG, None)
        if value is not None:
            os.environ[FLAG] = value
        spec.loader.exec_module(module)
    return module


class TestIntListOptIn(unittest.TestCase):
    def test_disabled_never_defines_or_calls_optimization(self):
        for flag in (None, "0", "", "false", "true", "2"):
            with self.subTest(flag=flag):
                api = load_types(flag)
                self.assertFalse(hasattr(api, "_expect_int_list"))
                with patch.object(msgspec, "convert", side_effect=AssertionError("fast path")):
                    for name in DECODERS:
                        decode = getattr(api, name)
                        values = list(range(1024))
                        self.assertEqual(decode(values), values)
                        self.assertIsNot(decode(values), values)
                        with self.assertRaises(api.JsonContractError):
                            decode([True] * 1024)

    def test_enabled_uses_native_validation(self):
        api = load_types("1")
        with patch.object(msgspec, "convert", wraps=msgspec.convert) as convert:
            for name in DECODERS:
                self.assertEqual(getattr(api, name)(list(range(64))), list(range(64)))
            self.assertEqual(convert.call_count, len(DECODERS))

    def test_enabled_small_lists_keep_original_validation(self):
        api = load_types("1")
        with patch.object(msgspec, "convert", side_effect=AssertionError("fast path")):
            for name in DECODERS:
                self.assertEqual(getattr(api, name)(list(range(63))), list(range(63)))

    def test_flag_is_sampled_at_import(self):
        api = load_types(None)
        with patch.dict(os.environ, {FLAG: "1"}):
            self.assertFalse(hasattr(api, "_expect_int_list"))
            with patch.object(msgspec, "convert", side_effect=AssertionError("fast path")):
                self.assertEqual(api.decode_TokenIds(list(range(64))), list(range(64)))

    def test_enabled_selection_survives_environment_change(self):
        api = load_types("1")
        with patch.dict(os.environ, {FLAG: "0"}):
            with patch.object(msgspec, "convert", wraps=msgspec.convert) as convert:
                for name in DECODERS:
                    values = list(range(64))
                    self.assertEqual(getattr(api, name)(values), values)
                self.assertEqual(convert.call_count, len(DECODERS))

    def test_on_off_values_types_and_errors_match(self):
        off, on = load_types("0"), load_types("1")

        class Integer(int):
            pass

        class ListWithIterator(list):
            def __iter__(self):
                return iter(["bad"])

        cases = [[], list(range(63)), list(range(64)), list(range(131072)),
                 [Integer(1 << 100)] * 65, ListWithIterator(range(100)),
                 (1, 2), range(100), "123", None]
        for invalid in (True, 1.0, "1", None, [], {}):
            for index in (0, 63, 1023):
                values = list(range(1024))
                values[index] = invalid
                cases.append(values)
        for name in DECODERS:
            for i, values in enumerate(cases):
                with self.subTest(decoder=name, case=i):
                    try:
                        expected = getattr(off, name)(values)
                    except off.JsonContractError as error:
                        with self.assertRaises(on.JsonContractError) as actual:
                            getattr(on, name)(values)
                        self.assertEqual(str(error), str(actual.exception))
                    else:
                        actual = getattr(on, name)(values)
                        self.assertEqual(actual, expected)
                        self.assertIsNot(actual, values)
                        for left, right in zip(actual, expected):
                            self.assertIs(left, right)

    def test_nested_generate_requests_match(self):
        off, on = load_types("0"), load_types("1")
        request = {"input_ids": [list(range(1024)), [], list(range(65))]}
        expected = off.GenerateRequest.from_json_value(request).to_json_value()
        self.assertEqual(on.GenerateRequest.from_json_value(request).to_json_value(), expected)
        request["input_ids"][0][-1] = False
        request["input_ids"][2][0] = "later"
        for api in (off, on):
            with self.assertRaisesRegex(api.JsonContractError, "boolean `false`"):
                api.GenerateRequest.from_json_value(request)


if __name__ == "__main__":
    unittest.main()
