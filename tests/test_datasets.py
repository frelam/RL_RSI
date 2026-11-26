from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rsi.core.types import DataRequest, TaskSpec
from rsi.modules.datasets import (
    DatasetCatalog,
    DatasetSpec,
    merge_ability_priorities,
    preprocess_row,
    select_datasets,
)


class DatasetTests(unittest.TestCase):
    def test_greedy_selection_covers_high_priority_abilities(self) -> None:
        datasets = [
            DatasetSpec("math", "synthetic", abilities={"math": 1.0}, priority=1.0),
            DatasetSpec("code", "synthetic", abilities={"code": 1.0}, priority=0.8),
            DatasetSpec("general", "synthetic", abilities={"math": 0.5, "code": 0.5}, priority=0.7),
        ]
        selection = select_datasets(datasets, {"math": 1.0, "code": 0.8}, 2)
        self.assertEqual({item.name for item in selection.datasets}, {"math", "code"})
        self.assertAlmostEqual(sum(selection.ratios.values()), 1.0)
        self.assertEqual(selection.uncovered, [])

    def test_catalog_accepts_list_coverage_and_task_request_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "datasets.yaml"
            path.write_text(
                "datasets:\n  demo:\n    loader: synthetic\n    abilities: [math.reasoning]\n    rows: [{question: q, answer: a}]\n",
                encoding="utf-8",
            )
            catalog = DatasetCatalog(path)
            self.assertEqual(catalog.datasets[0].abilities, {"math.reasoning": 1.0})
        request = DataRequest("r", "math.reasoning", "dataset", 2)
        task = TaskSpec("task-1", {"math.reasoning": 0.9}, count=2)
        self.assertEqual(request.capability, task.ability_priorities.popitem()[0])

    def test_preprocess_normalizes_chat_and_drops_empty_rows(self) -> None:
        row = preprocess_row({"messages": [{"role": "user", "content": "  question  "}]}, ["normalize_chat", "drop_empty"])
        self.assertEqual(row["prompt"], "question")
        self.assertEqual(row["response"], "")
        self.assertIsNone(preprocess_row({"prompt": "  "}, ["normalize_chat", "drop_empty"]))

    def test_priority_merge_uses_available_sources_and_configured_weights(self) -> None:
        merged = merge_ability_priorities(
            {"math": 0.8, "code": 0.5},
            {"math": 0.2, "tool": 1.0},
            rule_weight=0.6,
            agent_weight=0.4,
        )
        self.assertAlmostEqual(merged["math"], 0.56)
        self.assertEqual(merged["code"], 0.5)
        self.assertEqual(merged["tool"], 1.0)


if __name__ == "__main__":
    unittest.main()
