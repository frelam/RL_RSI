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


class LoopSmokeTests(unittest.TestCase):
    def test_demo_loop_reaches_target_and_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            config = REPO_ROOT / "configs" / "demo_mock.yaml"
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


if __name__ == "__main__":
    unittest.main()
