from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from rsi.cli import main
from rsi.core.jsonio import read_json
from tests.helpers import REPO_ROOT


class ResumeTests(unittest.TestCase):
    def _write_config(self, path: Path, max_rounds: int) -> None:
        default = (REPO_ROOT / "configs" / "default.yaml").as_posix()
        path.write_text(
            f"""
include: [{default}]
stop:
  policies:
    - type: max_rounds
      max_rounds: {max_rounds}
    - type: target_score
      target: 0.99
      min_rounds: 1
""",
            encoding="utf-8",
        )

    def test_resume_continues_from_last_round(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            config = Path(td) / "resume.yaml"
            self._write_config(config, 1)
            common = [
                "run",
                "--config",
                str(config),
                "--run-id",
                "resume",
                "--runs-dir",
                td,
                "--quiet",
            ]
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(main(common), 0)
            state = read_json(Path(td) / "resume" / "state.json")
            self.assertEqual(state["round"], 1)
            self.assertEqual(len(state["history"]), 1)

            self._write_config(config, 2)
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(main(common + ["--resume"]), 0)
            state = read_json(Path(td) / "resume" / "state.json")
            self.assertEqual(state["round"], 2)
            self.assertEqual([record["round"] for record in state["history"]], [0, 1])
            self.assertEqual(state["status"], "stopped")
            self.assertIn("max_rounds", state["stop_reason"])
            registry = read_json(Path(td) / "resume" / "checkpoints" / "registry.json")
            self.assertEqual(registry["checkpoints"][registry["current"]]["version"], 2)


if __name__ == "__main__":
    unittest.main()
