from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from rsi.core.config import Config, ConfigError, apply_override, load_config


class ConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write(self, name: str, text: str) -> Path:
        path = self.root / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_include_override_and_types(self) -> None:
        self._write(
            "base.yaml",
            """
a:
  b: 1
  text: hello
  items: [1, 2]
""",
        )
        child = self._write(
            "child.yaml",
            """
include: [base.yaml]
a:
  c: true
""",
        )
        cfg = load_config(child, ["a.b=3", "a.text=world", "x.y=1.5"])
        self.assertEqual(cfg.get_int("a.b"), 3)
        self.assertEqual(cfg.get("a.text"), "world")
        self.assertTrue(cfg.get_bool("a.c"))
        self.assertEqual(cfg.get_float("x.y"), 1.5)
        self.assertEqual(cfg.get_list("a.items"), [1, 2])
        self.assertEqual(cfg.section("a").get("b"), 3)

    def test_missing_key_raises(self) -> None:
        cfg = Config({"a": {"b": 1}})
        with self.assertRaises(ConfigError):
            cfg.get("a.missing")
        self.assertEqual(cfg.get("a.missing", 7), 7)

    def test_optional_section_defaults_to_empty(self) -> None:
        cfg = Config({"a": {"b": 1}})
        self.assertEqual(cfg.section("deploy", required=False).as_dict(), {})
        with self.assertRaises(ConfigError):
            cfg.section("deploy")

    def test_env_expansion_and_missing_env(self) -> None:
        os.environ["RSI_TEST_VALUE"] = "secret"
        try:
            cfg = Config({"x": "${ENV:RSI_TEST_VALUE}", "y": "${ENV:RSI_TEST_MISSING:fallback}"})
            self.assertEqual(cfg.get("x"), "secret")
            self.assertEqual(cfg.get("y"), "fallback")
        finally:
            del os.environ["RSI_TEST_VALUE"]
        with self.assertRaises(ConfigError):
            Config({"x": "${ENV:RSI_TEST_MISSING}"}).get("x")

    def test_specs_list_and_named_map(self) -> None:
        cfg = Config(
            {
                "filters": {
                    "pipeline": [
                        "dedup",
                        {"type": "length", "params": {"min_prompt_chars": 4}},
                    ]
                },
                "benchmarks": {
                    "demo": {"type": "mock", "items": 2},
                    "other": "jsonl",
                },
            }
        )
        specs = cfg.specs("filters.pipeline")
        self.assertEqual(specs[0]["type"], "dedup")
        self.assertEqual(specs[1]["min_prompt_chars"], 4)
        benchmarks = cfg.specs("benchmarks")
        self.assertEqual({b["name"] for b in benchmarks}, {"demo", "other"})
        self.assertEqual(cfg.section("benchmarks").get("other"), "jsonl")

    def test_apply_override_rejects_bad_path(self) -> None:
        data = {"a": 1}
        with self.assertRaises(ConfigError):
            apply_override(data, "a.b=2")

    def test_child_inherits_dirs(self) -> None:
        cfg = Config({"a": 1}, config_dir=self.root, repo_root=self.root)
        child = cfg.child({"type": "x", "v": 2}, "child")
        self.assertEqual(child.get_int("v"), 2)
        self.assertEqual(child.repo_root, self.root)

    def test_omitted_config_uses_builtin_defaults(self) -> None:
        previous = os.getcwd()
        with tempfile.TemporaryDirectory() as td:
            os.chdir(td)
            try:
                cfg = load_config()
            finally:
                os.chdir(previous)
        self.assertEqual(cfg.get("trainer.type"), "mock")
        self.assertEqual(cfg.get("checkpoint.initial.model_path"), "mock://base-model")
        self.assertEqual(cfg.get("run.seed"), 7)
        self.assertTrue(cfg.get_list("stop.policies"))
        self.assertTrue(cfg.specs("analyze.methods"))
        self.assertTrue(Path(cfg.source).is_file())

    def test_partial_config_is_merged_over_defaults(self) -> None:
        path = self._write("partial.yaml", "trainer:\n  steps: 5\ndata:\n  max_generated: 8\n")
        cfg = load_config(path)
        self.assertEqual(cfg.get_int("trainer.steps"), 5)
        self.assertEqual(cfg.get_int("data.max_generated"), 8)
        self.assertEqual(cfg.get("trainer.type"), "mock")
        self.assertTrue(cfg.get_list("stop.policies"))
        self.assertEqual(cfg.plugin("loop")["type"], "classic")

    def test_overrides_apply_after_the_config_file(self) -> None:
        path = self._write("overrides.yaml", "trainer:\n  steps: 5\n")
        cfg = load_config(
            path,
            [
                "trainer.steps=9",
                "stop.policies=[{type: max_rounds, max_rounds: 1}]",
            ],
        )
        self.assertEqual(cfg.get_int("trainer.steps"), 9)
        self.assertEqual(cfg.get_list("stop.policies"), [{"type": "max_rounds", "max_rounds": 1}])

    def test_missing_config_file_still_fails(self) -> None:
        with self.assertRaises(ConfigError):
            load_config(self.root / "nope.yaml")


if __name__ == "__main__":
    unittest.main()
