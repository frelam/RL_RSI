from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from rsi.core.config import Config
from rsi.core.errors import ConfigError
from rsi.core.types import DataRequest, HarnessTask
from rsi.hubs.prompts import PromptHub
from rsi.hubs.skills import SkillHub
from rsi.modules.data_collectors import AgenticSynthesisCollector
from tests.helpers import REPO_ROOT, FakeRuntime


def cfg(data: dict) -> Config:
    return Config(data, config_dir=REPO_ROOT, repo_root=REPO_ROOT)


class HubTests(unittest.TestCase):
    def test_prompt_hub_render_and_missing_variable(self) -> None:
        hub = PromptHub(REPO_ROOT / "prompts")
        rendered = hub.render(
            "badcase_analysis",
            {
                "round": 3,
                "capability_table": [{"capability": "math", "score": 0.2}],
                "worst_cases": [{"item_id": "x", "score": 0.0}],
                "baseline": {},
            },
        )
        self.assertIn("Round 3", rendered.text)
        self.assertIn("math", rendered.text)
        self.assertEqual(rendered.version, 1)
        with self.assertRaises(ConfigError):
            hub.render("badcase_analysis", {"round": 1})

    def test_skill_hub_materialize(self) -> None:
        hub = SkillHub(REPO_ROOT / "skills")
        spec = hub.get("env-synthesis")
        self.assertIn("SKILL.md", spec.files)
        with tempfile.TemporaryDirectory() as td:
            destination = hub.materialize(["env-synthesis", "train-debug"], Path(td) / "skills")
            self.assertTrue((destination / "env-synthesis" / "SKILL.md").exists())
            self.assertTrue((destination / "train-debug" / "SKILL.md").exists())


class HarnessTests(unittest.TestCase):
    def test_mock_harness_schema_validation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            runtime = FakeRuntime(td)
            task = HarnessTask(
                name="env_synthesis",
                instruction="make a task",
                work_dir=Path(td) / "task",
                output_schema={
                    "type": "object",
                    "required": ["tasks"],
                    "properties": {"tasks": {"type": "array"}},
                },
                metadata={"round": 2},
            )
            result = runtime.harness("synth").run(task)
            self.assertTrue(result.ok)
            self.assertEqual(len(result.output["tasks"]), 1)
            self.assertTrue((task.work_dir / "mock_output.json").exists())

    def test_agentic_collector_writes_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            runtime = FakeRuntime(td)
            collector = AgenticSynthesisCollector(
                runtime,
                cfg(
                    {
                        "prompt": "env_synthesis",
                        "harness": "synth",
                        "skills": ["env-synthesis"],
                        "max_tasks": 2,
                    }
                ),
            )
            request = DataRequest(
                id="req-1",
                capability="agentic.tool_use",
                kind="agentic",
                count=2,
                constraints={"verifier": {"type": "harness", "name": "tests_pass"}},
            )
            context = SimpleNamespace(round=0, work_dir=Path(td) / "work")
            samples = collector.collect(request, context)
            self.assertEqual(len(samples), 1)
            sample = samples[0]
            self.assertEqual(sample.kind, "agentic")
            self.assertIsNotNone(sample.env_path)
            env = Path(sample.env_path)
            self.assertTrue((env / "main.py").exists())
            self.assertEqual(sample.verifier["name"], "tests_pass")
            self.assertIn("env-synthesis", sample.metadata["skills"])


if __name__ == "__main__":
    unittest.main()
