from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from rsi.cli import main
from rsi.core.jsonio import read_json
from tests.helpers import REPO_ROOT


def _write_config(td: str, name: str, extra: str) -> Path:
    """A config that extends configs/default.yaml (absolute include)."""
    path = Path(td) / name
    path.write_text(f"include: [{(REPO_ROOT / 'configs' / 'default.yaml').as_posix()}]\n{extra}", encoding="utf-8")
    return path


class LoopSmokeTests(unittest.TestCase):
    def test_demo_loop_reaches_target_and_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            config = _write_config(
                td,
                "short.yaml",
                "run:\n  id: smoke\nstop:\n  policies:\n    - {type: target_score, target: 0.60, min_rounds: 1}\n"
                "    - {type: max_rounds, max_rounds: 3}\n",
            )
            stdout, stderr = io.StringIO(), io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = main(
                    [
                        "run",
                        "--config",
                        str(config),
                        "--run-id",
                        "smoke",
                        "--runs-dir",
                        td,
                        "--quiet",
                    ]
                )
            self.assertEqual(code, 0, msg=stdout.getvalue() + stderr.getvalue())
            run_dir = Path(td) / "smoke"
            state = read_json(run_dir / "state.json")
            self.assertEqual(state["status"], "stopped")
            self.assertIn("target score", state["stop_reason"])
            self.assertGreaterEqual(state["round"], 2)
            self.assertGreaterEqual(state["best_score"], 0.60)
            self.assertEqual(len(state["history"]), state["round"])

            for index, record in enumerate(state["history"]):
                self.assertEqual(record["round"], index)
                self.assertIn("validation", record)
                if index == 0:
                    self.assertEqual(record["synthesize"]["replayed"], 0)
                else:
                    self.assertGreater(record["synthesize"]["replayed"], 0)
                self.assertIsNotNone(record["validation"]["score"])
                self.assertTrue(record["validation"]["accepted"])
                self.assertTrue(record["stop"]["should_stop"] if index == len(state["history"]) - 1 else True)

            registry = read_json(run_dir / "checkpoints" / "registry.json")
            self.assertEqual(registry["current"], state["best_checkpoint"])
            self.assertEqual(registry["checkpoints"][registry["current"]]["status"], "current")

            first = state["history"][0]
            dataset_dir = Path(first["synthesize"]["dataset_path"])
            self.assertTrue((dataset_dir / "train.jsonl").exists())
            self.assertTrue((dataset_dir / "val.jsonl").exists())
            self.assertTrue((dataset_dir / "reward_spec.json").exists())
            self.assertTrue((dataset_dir / "manifest.json").exists())
            manifest = read_json(dataset_dir / "manifest.json")
            self.assertEqual(manifest["dataset_id"], first["synthesize"]["dataset_id"])
            self.assertGreater(manifest["num_samples"], 0)
            self.assertGreater(len(manifest["stats"]["groups"]), 0)

            events = (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
            event_types = {json.loads(line)["type"] for line in events if line.strip()}
            self.assertIn("run_started", event_types)
            self.assertIn("round_completed", event_types)
            self.assertIn("run_stopped", event_types)
            self.assertTrue((run_dir / "commands.jsonl").exists())

            # Each round trains a candidate and promotes it.
            versions = [
                checkpoint["version"]
                for checkpoint in registry["checkpoints"].values()
                if checkpoint["status"] in {"current", "archived"}
            ]
            self.assertEqual(max(versions), state["round"])


    def test_judged_defect_pulls_training_rows_from_the_dataset_library(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            config = _write_config(
                td,
                "dataset.yaml",
                "stop:\n  policies:\n    - {type: max_rounds, max_rounds: 1}\n"
                "llm:\n  profiles:\n    judge:\n      fixtures:\n        badcase_analysis:\n"
                "          defects:\n"
                "            - {capability: math.reasoning, severity: 0.55, confidence: 0.8,\n"
                "               description: slips on final arithmetic, evidence: ['scored 0.0-0.25'],\n"
                "               suggested_data: {kind: dataset, seeds: [word problems], target_difficulty: [0.35, 0.8]}}\n",
            )
            stdout, stderr = io.StringIO(), io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = main(
                    ["run", "--config", str(config), "--run-id", "dataset-demo",
                     "--runs-dir", td, "--quiet"]
                )
            self.assertEqual(code, 0, msg=stdout.getvalue() + stderr.getvalue())
            state = read_json(Path(td) / "dataset-demo" / "state.json")
            self.assertEqual(state["round"], 1)
            first = state["history"][0]
            self.assertTrue(first["validation"]["accepted"])
            manifest = read_json(Path(first["synthesize"]["dataset_path"]) / "manifest.json")
            self.assertGreater(manifest["stats"]["sources"].get("dataset:math_word_problems", 0), 0)

    def test_scope_drops_out_of_scope_defects_before_planning(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            config = _write_config(
                td, "scope.yaml",
                "stop:\n  policies:\n    - {type: max_rounds, max_rounds: 1}\n"
                "data:\n  plan:\n    scope: [math.reasoning]\n",
            )
            stdout, stderr = io.StringIO(), io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = main(
                    ["run", "--config", str(config), "--run-id", "scope-demo",
                     "--runs-dir", td, "--quiet"]
                )
            self.assertEqual(code, 0, msg=stdout.getvalue() + stderr.getvalue())
            run_dir = Path(td) / "scope-demo"
            defects = read_json(run_dir / "rounds" / "round-0000" / "analyze" / "defects.json")
            self.assertEqual(defects["metadata"]["scope"], ["math.reasoning"])
            self.assertEqual({d["capability"] for d in defects["defects"]} - {"math.reasoning"}, set())
            first = read_json(run_dir / "state.json")["history"][0]
            self.assertLessEqual(set(first.get("validation", {}).get("targets", [])), {"math.reasoning"})


    def test_model_task_and_round_flags_drive_the_run(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            config = _write_config(
                td,
                "flags.yaml",
                "stop:\n  policies:\n    - {type: max_rounds, max_rounds: 9}\n",
            )
            stdout, stderr = io.StringIO(), io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = main(
                    ["run", "--config", str(config), "--run-id", "flags",
                     "--runs-dir", td, "--model", "mock://flag-model",
                     "--task", "agentic/reasoning", "--round", "2", "--quiet"]
                )
            self.assertEqual(code, 0, msg=stdout.getvalue() + stderr.getvalue())
            run_dir = Path(td) / "flags"
            state = read_json(run_dir / "state.json")
            self.assertEqual(state["round"], 2)
            self.assertIn("max_rounds", state["stop_reason"])
            registry = read_json(run_dir / "checkpoints" / "registry.json")
            self.assertEqual(registry["checkpoints"]["ckpt_v000"]["model_path"], "mock://flag-model")
            plan = read_json(run_dir / "rounds" / "round-0000" / "synthesize" / "plan.json")
            self.assertEqual(plan["metadata"]["dataset_strategy"]["task"], "agentic/reasoning")

    def test_unknown_task_preset_fails_before_running(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            config = _write_config(td, "unknown.yaml", "")
            stdout, stderr = io.StringIO(), io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = main(
                    ["run", "--config", str(config), "--run-id", "unknown",
                     "--runs-dir", td, "--task", "nope", "--quiet"]
                )
            self.assertEqual(code, 1)
            self.assertIn("Unknown task preset", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
