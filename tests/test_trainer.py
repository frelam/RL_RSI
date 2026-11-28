from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rsi.adapters.trainer import CommandTrainer
from rsi.core.config import Config
from rsi.core.errors import AdapterError
from tests.helpers import REPO_ROOT, FakeRuntime


def _trainer(runtime: FakeRuntime, **overrides) -> CommandTrainer:
    data = {"command": ["train"], **overrides}
    return CommandTrainer(
        runtime, Config(data, config_dir=REPO_ROOT, repo_root=REPO_ROOT)
    )


class CommandTrainerCheckpointTests(unittest.TestCase):
    def test_glob_picks_the_highest_step_directory(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            trainer = _trainer(FakeRuntime(td), checkpoint_glob="global_step_*")
            output = Path(td) / "checkpoints"
            for step in (5, 100, 20):
                (output / f"global_step_{step}").mkdir(parents=True)

            resolved = trainer._resolve_checkpoint(output, {"round": 1}, None)

        self.assertIsNotNone(resolved)
        self.assertEqual(resolved.name, "global_step_100")

    def test_subdir_and_best_step_template(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            runtime = FakeRuntime(td)
            output = Path(td) / "checkpoints"
            (output / "hf").mkdir(parents=True)
            (output / "global_step_30").mkdir(parents=True)

            subdir = _trainer(runtime, checkpoint_subdir="hf")
            template = _trainer(runtime, checkpoint_path="global_step_{best_step}")

            self.assertEqual(subdir._resolve_checkpoint(output, {}, None), output / "hf")
            self.assertEqual(
                template._resolve_checkpoint(output, {}, {"step": 30}),
                output / "global_step_30",
            )

    def test_export_requires_a_checkpoint_locator(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(AdapterError):
                _trainer(FakeRuntime(td), export={"command": ["merge"]})


if __name__ == "__main__":
    unittest.main()
