from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rsi.core.config import Config
from rsi.core.errors import RsiError
from rsi.core.runtime import CheckpointManager
from tests.helpers import REPO_ROOT


class CheckpointTests(unittest.TestCase):
    def test_reserve_promote_reject_and_persist(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "checkpoints"
            initial = Config(
                {"model_path": "mock://base", "base_model": "base"},
                config_dir=REPO_ROOT,
                repo_root=REPO_ROOT,
            )
            manager = CheckpointManager(root, initial, lambda path: Path(td) / path)
            self.assertEqual(manager.current().id, "ckpt_v000")
            first = manager.reserve(model_path="/models/v1", round_no=0, source_job="job-1")
            second = manager.reserve(model_path="/models/v2", round_no=1, parent_id=first.id)
            self.assertEqual(first.version, 1)
            self.assertEqual(second.version, 2)
            self.assertEqual(manager.current().id, "ckpt_v000")

            manager.promote(first.id)
            self.assertEqual(manager.current().id, first.id)
            self.assertEqual(manager.get(first.id).status, "current")
            manager.reject(second.id, "worse")
            self.assertEqual(manager.get(second.id).status, "rejected")
            self.assertIn("worse", manager.get(second.id).metrics["reject_reason"])

            reloaded = CheckpointManager(root, initial, lambda path: Path(td) / path)
            self.assertEqual(reloaded.current().id, first.id)
            self.assertEqual(len(reloaded.list()), 3)

    def test_reject_current_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            initial = Config({"model_path": "mock://base"}, config_dir=REPO_ROOT, repo_root=REPO_ROOT)
            manager = CheckpointManager(Path(td), initial, lambda path: path)
            with self.assertRaises(RsiError):
                manager.reject(manager.current().id)


if __name__ == "__main__":
    unittest.main()
