from __future__ import annotations

import unittest

from rsi.core.errors import RegistryError, SchemaError
from rsi.core.registry import add, create, list_plugins, resolve
from rsi.core.schema import validate_json


class SchemaTests(unittest.TestCase):
    def test_object_required_and_types(self) -> None:
        schema = {
            "type": "object",
            "required": ["defects"],
            "properties": {
                "defects": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["capability", "severity"],
                        "properties": {
                            "capability": {"type": "string"},
                            "severity": {"type": "number", "minimum": 0, "maximum": 1},
                        },
                    },
                }
            },
        }
        self.assertEqual(validate_json({"defects": [{"capability": "x", "severity": 0.5}]}, schema), [])
        errors = validate_json({"defects": [{"capability": 3, "severity": 2}]}, schema)
        self.assertTrue(errors)
        self.assertTrue(any("expected string" in error for error in errors))
        self.assertTrue(any("above maximum" in error for error in errors))

    def test_missing_required(self) -> None:
        errors = validate_json({}, {"type": "object", "required": ["a"]})
        self.assertEqual(len(errors), 1)

    def test_enum_and_const(self) -> None:
        self.assertTrue(validate_json("x", {"enum": ["a", "b"]}))
        self.assertEqual(validate_json("a", {"const": "a"}), [])


class RegistryTests(unittest.TestCase):
    def test_builtin_plugins_registered(self) -> None:
        plugins = list_plugins()
        for category, name in [
            ("loop", "classic"),
            ("analyzer", "rule"),
            ("data_collector", "text_synthesis"),
            ("dataset_builder", "jsonl"),
            ("monitor", "entropy_collapse"),
            ("stop_policy", "target_score"),
        ]:
            self.assertIn(name, plugins[category])

    def test_duplicate_registration_rejected(self) -> None:
        class One:
            def __init__(self, runtime, cfg, **params):
                pass

        class Two:
            def __init__(self, runtime, cfg, **params):
                pass

        add("test_category", "dupe", One)
        with self.assertRaises(RegistryError):
            add("test_category", "dupe", Two)

    def test_resolve_import_path(self) -> None:
        cls, params = resolve("anything", {"type": "collections:OrderedDict"})
        self.assertEqual(cls.__name__, "OrderedDict")
        self.assertEqual(params, {})

    def test_unknown_plugin_lists_available(self) -> None:
        with self.assertRaises(RegistryError):
            create("loop", "does-not-exist", object(), object())


if __name__ == "__main__":
    unittest.main()
