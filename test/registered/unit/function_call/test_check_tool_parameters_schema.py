"""Tool-schema compatibility tests without importing the serving/GPU stack."""

import ast
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

from jsonschema import Draft202012Validator, SchemaError

REPO_ROOT = Path(__file__).resolve().parents[4]


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


register_cpu_ci = _load_module(
    "schema_test_ci_register",
    REPO_ROOT / "python/sglang/test/ci/ci_register.py",
).register_cpu_ci
register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _load_validation_from_utils():
    """Load the production validator definitions without optional SGLang deps."""
    path = REPO_ROOT / "python/sglang/srt/function_call/utils.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    selected = []
    found = set()
    names = {
        "_JSON_SCALAR_TYPES",
        "_MAX_RUST_SCHEMA_DEPTH",
        "_TOOL_SCHEMA_VALIDATOR",
        "_RUST_FALLBACK_WARNING_LOCK",
        "_rust_fallback_warning_logged",
        "_is_rust_compatible_json",
        "_warn_rust_fallback",
        "check_tool_parameters_schema",
        "logger",
    }
    for node in tree.body:
        if isinstance(node, ast.Import):
            if any(
                alias.name in {"logging", "math", "threading", "jsonschema_rs"}
                for alias in node.names
            ):
                selected.append(node)
        elif isinstance(node, ast.ImportFrom):
            if node.module in {"typing", "jsonschema", "jsonschema_specifications"}:
                selected.append(node)
        elif isinstance(node, ast.Try):
            # jsonschema_rs is optional in production. Keep its guarded
            # import/initialization so the extracted module follows the same
            # fallback path as the application when the binding is absent.
            if any(
                isinstance(child, ast.Import)
                and any(alias.name == "jsonschema_rs" for alias in child.names)
                for child in node.body
            ):
                selected.append(node)
                found.add("_TOOL_SCHEMA_VALIDATOR")
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            matched = {
                target.id
                for target in targets
                if isinstance(target, ast.Name) and target.id in names
            }
            if matched:
                selected.append(node)
                found.update(matched)
        elif isinstance(node, ast.FunctionDef) and node.name in names:
            selected.append(node)
            found.add(node.name)
    missing = names - found
    if missing:
        raise AssertionError(f"Production validator definitions not found: {missing}")
    module = types.ModuleType("tool_schema_validation_under_test")
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"),
        module.__dict__,
    )
    return module


validation = _load_validation_from_utils()


def _schema(**properties):
    return {"type": "object", "properties": properties, "required": list(properties)}


class TestCheckToolParametersSchema(unittest.TestCase):
    def setUp(self):
        self.python_check = Draft202012Validator.check_schema

    def assert_matches_python(self, schema, expected_fallback=None):
        """Compare both acceptance and the public Python error details."""
        try:
            self.python_check(schema)
        except SchemaError as expected:
            with self.assertRaises(SchemaError) as caught:
                validation.check_tool_parameters_schema(schema)
            actual = caught.exception
            self.assertEqual(str(actual), str(expected))
            for attribute in ("message", "validator", "validator_value"):
                self.assertEqual(
                    getattr(actual, attribute), getattr(expected, attribute)
                )
            self.assertEqual(list(actual.absolute_path), list(expected.absolute_path))
            self.assertEqual(
                list(actual.absolute_schema_path), list(expected.absolute_schema_path)
            )
        else:
            actual = validation.check_tool_parameters_schema(schema)
            self.assertIsInstance(actual, bool)
            if expected_fallback is not None:
                self.assertEqual(actual, expected_fallback)

    @unittest.skipUnless(
        validation._TOOL_SCHEMA_VALIDATOR is not None,
        "jsonschema_rs is not installed",
    )
    def test_valid_and_repeated_schemas_use_real_rust_without_python(self):
        schema = _schema(a={"type": "string"}, b={"type": "integer"})
        with (
            mock.patch.object(
                validation,
                "_TOOL_SCHEMA_VALIDATOR",
                wraps=validation._TOOL_SCHEMA_VALIDATOR,
            ) as rust,
            mock.patch.object(
                validation.Draft202012Validator,
                "check_schema",
                wraps=self.python_check,
            ) as python,
        ):
            self.assertFalse(validation.check_tool_parameters_schema(schema))
            self.assertFalse(validation.check_tool_parameters_schema(schema))
        self.assertEqual(rust.is_valid.call_count, 2)
        python.assert_not_called()

    @unittest.skipUnless(
        validation._TOOL_SCHEMA_VALIDATOR is not None,
        "jsonschema_rs is not installed",
    )
    def test_mutated_schema_is_checked_again(self):
        schema = _schema(a={"type": "string"})
        with (
            mock.patch.object(
                validation,
                "_TOOL_SCHEMA_VALIDATOR",
                wraps=validation._TOOL_SCHEMA_VALIDATOR,
            ) as rust,
            mock.patch.object(
                validation.Draft202012Validator,
                "check_schema",
                wraps=self.python_check,
            ) as python,
        ):
            validation.check_tool_parameters_schema(schema)
            schema["properties"]["a"]["type"] = "integer"
            validation.check_tool_parameters_schema(schema)
            python.assert_not_called()
            schema["properties"]["a"]["type"] = "not-a-type"
            self.assert_matches_python(schema)
        self.assertEqual(rust.is_valid.call_count, 3)
        python.assert_called_once_with(schema)

    def test_invalid_schemas_preserve_python_errors(self):
        cases = {
            "type": _schema(a={"type": "not-a-type"}),
            "required": {"type": "object", "required": "x"},
            "nested": {"$defs": {"bad": {"minLength": -1}}},
            "regex": {"pattern": "["},
            "ref_type": {"$ref": 123},
            "multiple_errors": {"required": "x", "type": "not-a-type"},
        }
        for name, schema in cases.items():
            with (
                self.subTest(name=name),
                mock.patch.object(
                    validation.Draft202012Validator,
                    "check_schema",
                    wraps=self.python_check,
                ) as python,
            ):
                self.assert_matches_python(schema)
                self.assert_matches_python(schema)
                self.assertEqual(python.call_count, 2)

    def test_rust_rejection_cannot_override_python_acceptance(self):
        schema = _schema(a={"type": "string"})
        with (
            mock.patch.object(validation, "_TOOL_SCHEMA_VALIDATOR") as rust,
            mock.patch.object(
                validation.Draft202012Validator,
                "check_schema",
                wraps=self.python_check,
            ) as python,
        ):
            rust.is_valid.return_value = False
            self.assertTrue(validation.check_tool_parameters_schema(schema))
        rust.is_valid.assert_called_once_with(schema)
        python.assert_called_once_with(schema)

    def test_rust_conversion_exceptions_use_python(self):
        for exception in (
            RuntimeError,
            TypeError,
            ValueError,
            OverflowError,
            RecursionError,
        ):
            for schema in (_schema(a={"type": "string"}), {"required": "x"}):
                with (
                    self.subTest(exception=exception, schema=schema),
                    mock.patch.object(validation, "_TOOL_SCHEMA_VALIDATOR") as rust,
                    mock.patch.object(
                        validation.Draft202012Validator,
                        "check_schema",
                        wraps=self.python_check,
                    ) as python,
                ):
                    rust.is_valid.side_effect = exception("conversion failed")
                    self.assert_matches_python(schema)
                    python.assert_called_once_with(schema)

    def test_rust_conversion_exception_warns_once(self):
        schema = _schema(a={"type": "string"})
        previous = validation._rust_fallback_warning_logged
        validation._rust_fallback_warning_logged = False
        self.addCleanup(
            setattr,
            validation,
            "_rust_fallback_warning_logged",
            previous,
        )
        with (
            mock.patch.object(validation, "_TOOL_SCHEMA_VALIDATOR") as rust,
            mock.patch.object(
                validation.Draft202012Validator,
                "check_schema",
                wraps=self.python_check,
            ),
            self.assertLogs(validation.logger, level="WARNING") as logs,
        ):
            rust.is_valid.side_effect = RuntimeError("conversion failed")
            self.assertTrue(validation.check_tool_parameters_schema(schema))
            self.assertTrue(validation.check_tool_parameters_schema(schema))

        self.assertEqual(len(logs.output), 1)
        self.assertIn("RuntimeError", logs.output[0])
        self.assertIn("falling back to Python", logs.output[0])

    def test_unavailable_rust_validator_uses_python(self):
        schema = _schema(a={"type": "string"})
        with (
            mock.patch.object(validation, "_TOOL_SCHEMA_VALIDATOR", None),
            mock.patch.object(
                validation.Draft202012Validator,
                "check_schema",
                wraps=self.python_check,
            ) as python,
        ):
            self.assertTrue(validation.check_tool_parameters_schema(schema))
        python.assert_called_once_with(schema)

    @unittest.skipUnless(
        validation._TOOL_SCHEMA_VALIDATOR is not None,
        "jsonschema_rs is not installed",
    )
    def test_newline_anchor_keeps_python_acceptance(self):
        # Python's regex "$" also matches before a trailing newline; Rust's
        # anchor check is stricter. Exercise the real false-negative fallback.
        for keyword in ("$anchor", "$dynamicAnchor"):
            schema = {keyword: "x\n"}
            with self.subTest(keyword=keyword):
                self.assertFalse(validation._TOOL_SCHEMA_VALIDATOR.is_valid(schema))
                self.assert_matches_python(schema)

    def test_non_json_values_bypass_rust_and_keep_python_semantics(self):
        class DictSubclass(dict):
            pass

        cases = {
            "tuple": {"type": "object", "required": ("x",)},
            "integer_key": {"type": "object", "properties": {1: {}}},
            "set_annotation": {"default": {1, 2}},
            "dict_subclass": DictSubclass(type="object"),
            "nan": {"type": "number", "minimum": float("nan")},
            "infinity": {"type": "number", "maximum": float("inf")},
        }
        for name, schema in cases.items():
            with (
                self.subTest(name=name),
                mock.patch.object(validation, "_TOOL_SCHEMA_VALIDATOR") as rust,
                mock.patch.object(
                    validation.Draft202012Validator,
                    "check_schema",
                    wraps=self.python_check,
                ) as python,
            ):
                self.assert_matches_python(schema)
                rust.is_valid.assert_not_called()
                python.assert_called_once_with(schema)

    def test_deep_annotation_bypasses_rust_but_is_valid(self):
        value = "leaf"
        for _ in range(40):
            value = [value]
        schema = {"default": value}
        with (
            mock.patch.object(validation, "_TOOL_SCHEMA_VALIDATOR") as rust,
            mock.patch.object(
                validation.Draft202012Validator,
                "check_schema",
                wraps=self.python_check,
            ) as python,
        ):
            self.assert_matches_python(schema)
        rust.is_valid.assert_not_called()
        python.assert_called_once_with(schema)

    def test_cycle_in_schema_retains_recursion_error(self):
        schema = {"type": "object", "properties": {}}
        schema["properties"]["self"] = schema
        with mock.patch.object(validation, "_TOOL_SCHEMA_VALIDATOR") as rust:
            with self.assertRaises(RecursionError):
                self.python_check(schema)
            with self.assertRaises(RecursionError):
                validation.check_tool_parameters_schema(schema)
        rust.is_valid.assert_not_called()

    def test_deep_schema_retains_python_recursion_error(self):
        schema = {"type": "string"}
        for _ in range(sys.getrecursionlimit()):
            schema = {"not": schema}
        with mock.patch.object(validation, "_TOOL_SCHEMA_VALIDATOR") as rust:
            with self.assertRaises(RecursionError):
                self.python_check(schema)
            with self.assertRaises(RecursionError):
                validation.check_tool_parameters_schema(schema)
        rust.is_valid.assert_not_called()

    def test_cycle_in_unvalidated_annotation_keeps_python_acceptance(self):
        schema = {}
        schema["default"] = schema
        with mock.patch.object(validation, "_TOOL_SCHEMA_VALIDATOR") as rust:
            self.assert_matches_python(schema)
        rust.is_valid.assert_not_called()

    def test_unicode_and_large_integer_compatibility(self):
        cases = [
            _schema(**{"城市🌍": {"type": "string", "description": "中文 café"}}),
            {"description": "unpaired surrogate: \ud800"},
            {"type": "integer", "minimum": 10**100},
            {"type": "number", "minimum": None},
        ]
        for index, schema in enumerate(cases):
            with self.subTest(index=index):
                self.assert_matches_python(schema)

    def test_fixed_draft_and_python_regex_policy(self):
        cases = [
            True,
            False,
            {"pattern": "(?P<name>abc)"},
            {"pattern": "(?<name>abc)"},
            {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "items": [{"type": "string"}],
            },
            {"$schema": "https://example.invalid/meta", "type": "object"},
            {"$id": "not a uri spaces"},
        ]
        for schema in cases:
            with self.subTest(schema=schema):
                self.assert_matches_python(schema)

    def test_schema_check_does_not_resolve_instance_references(self):
        # These references are data being checked against the meta-schema,
        # not schemas to fetch/compile for validating a tool-call instance.
        cases = [
            {"$ref": "https://example.invalid/missing"},
            {"$ref": "file:///nonexistent/sglang-schema-test.json"},
            {"$ref": "#/$defs/missing"},
            {"$dynamicRef": "https://example.invalid/dynamic#node"},
        ]
        for schema in cases:
            with self.subTest(schema=schema):
                self.assert_matches_python(schema)


if __name__ == "__main__":
    unittest.main()
