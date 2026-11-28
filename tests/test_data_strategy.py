from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from rsi.cli import main
from rsi.core.jsonio import read_json
from rsi.modules.data_strategy import (
    adjust_target_distribution,
    choose_fill_datasets,
    refine_strategy,
    strategy_gap,
)
from rsi.modules.datasets import DatasetSpec, select_datasets
from tests.helpers import REPO_ROOT


def _dataset(name: str, abilities: dict[str, float], rows: int = 0) -> DatasetSpec:
    return DatasetSpec(
        name,
        "synthetic",
        abilities=abilities,
        priority=0.5,
        rows=[{"question": f"{name}-{index}", "answer": "a"} for index in range(rows)],
    )


class TargetAdjustmentTests(unittest.TestCase):
    def test_moves_share_from_the_largest_ability(self) -> None:
        plan = adjust_target_distribution(
            {"math": 0.6, "code": 0.4}, {"retrieval": 0.9}, step=0.1, min_share=0.0
        )

        self.assertAlmostEqual(plan.distribution["retrieval"], 0.1)
        self.assertAlmostEqual(plan.distribution["math"], 0.5)
        self.assertAlmostEqual(plan.distribution["code"], 0.4)
        self.assertEqual(plan.transfers, [{"from": "math", "to": "retrieval", "amount": 0.1}])
        self.assertEqual(plan.source, "task")

    def test_falls_back_to_the_second_largest_donor(self) -> None:
        plan = adjust_target_distribution(
            {"a": 0.5, "b": 0.5}, {"d": 0.9}, step=0.7, min_share=0.1, max_shift=1.0
        )

        self.assertAlmostEqual(plan.distribution["d"], 0.7)
        self.assertAlmostEqual(plan.distribution["a"], 0.1)
        self.assertAlmostEqual(plan.distribution["b"], 0.2)
        self.assertEqual([item["from"] for item in plan.transfers], ["a", "b"])

    def test_deficient_abilities_do_not_donate_and_max_shift_caps_the_move(self) -> None:
        plan = adjust_target_distribution(
            {"a": 0.5, "b": 0.5}, {"a": 0.9, "b": 0.9}, step=0.1, min_share=0.01
        )
        self.assertEqual(plan.distribution, {"a": 0.5, "b": 0.5})
        self.assertEqual(plan.transfers, [])

        limited = adjust_target_distribution(
            {"a": 0.8, "b": 0.2}, {"c": 0.9}, step=0.5, min_share=0.0, max_shift=0.2
        )
        self.assertAlmostEqual(limited.distribution["c"], 0.2)
        self.assertAlmostEqual(sum(limited.distribution.values()), 1.0)


class StrategyOptimizationTests(unittest.TestCase):
    def test_target_fit_is_exact_when_coverage_allows_it(self) -> None:
        datasets = [
            _dataset("head", {"head": 1.0}),
            _dataset("mixed", {"head": 0.5, "tail": 0.5}),
        ]

        selection = select_datasets(
            datasets, {"head": 0.8, "tail": 0.2}, 100, max_datasets=2, diversity_weight=0.0
        )

        self.assertAlmostEqual(selection.stats["actual_distribution"]["head"], 0.8)
        self.assertAlmostEqual(selection.stats["actual_distribution"]["tail"], 0.2)
        self.assertAlmostEqual(selection.stats["gap"], 0.0)
        self.assertEqual(selection.uncovered, [])

    def test_diversity_needs_a_positive_weight(self) -> None:
        datasets = [
            _dataset("head", {"head": 1.0}),
            _dataset("broad", {"head": 9.0, "extra": 1.0}),
        ]

        plain = select_datasets(
            datasets, {"head": 1.0}, 100, max_datasets=2, diversity_weight=0.0
        )
        diverse = select_datasets(
            datasets,
            {"head": 1.0},
            100,
            max_datasets=2,
            diversity_weight=0.1,
            diversity_floor=0.01,
        )

        self.assertEqual([dataset.name for dataset in plain.datasets], ["head"])
        self.assertIn("broad", {dataset.name for dataset in diverse.datasets})
        self.assertEqual(diverse.stats["diversity"]["covered"], ["extra"])
        self.assertGreater(diverse.stats["gap"], 0.0)

    def test_uncoverable_ability_is_reported_as_shortfall(self) -> None:
        selection = select_datasets(
            [_dataset("head", {"head": 1.0})],
            {"head": 0.9, "retrieval": 0.1},
            50,
            diversity_weight=0.0,
        )

        self.assertEqual(selection.uncovered, ["retrieval"])
        self.assertAlmostEqual(selection.stats["gap"], 0.1)
        self.assertAlmostEqual(selection.stats["longtail_gap"], 0.1)


class ReviewTests(unittest.TestCase):
    def test_review_adds_the_dataset_that_fills_the_missing_ability(self) -> None:
        head = _dataset("head", {"head": 1.0}, rows=20)
        tail = _dataset("tail", {"tail": 1.0}, rows=20)

        selection, review = refine_strategy(
            [head], {"head": 0.5, "tail": 0.5}, 20, pool=[head, tail]
        )

        self.assertTrue(review.accepted, review.as_dict())
        self.assertEqual(review.added, ["tail"])
        self.assertEqual({dataset.name for dataset in selection.datasets}, {"head", "tail"})
        self.assertAlmostEqual(selection.stats["gap"], 0.0)

    def test_missing_ability_without_a_dataset_stays_uncovered(self) -> None:
        head = _dataset("head", {"head": 1.0}, rows=20)

        selection, review = refine_strategy(
            [head], {"head": 0.5, "tail": 0.5}, 20, pool=[head]
        )

        self.assertFalse(review.accepted)
        self.assertIn("tail", selection.uncovered)
        self.assertIn("no unused dataset", review.reason)

    def test_fill_scores_prefer_the_dataset_with_less_collateral_coverage(self) -> None:
        focused = _dataset("focused", {"tail": 1.0})
        broad = _dataset("broad", {"tail": 0.5, "head": 0.5})
        selection = select_datasets([_dataset("head", {"head": 1.0})], {"head": 0.6, "tail": 0.4}, 10)
        metrics = strategy_gap(selection, {"head": 0.6, "tail": 0.4})

        additions = choose_fill_datasets(
            [focused, broad], [], {"head": 0.6, "tail": 0.4}, metrics, limit=1
        )

        self.assertEqual([dataset.name for dataset in additions], ["focused"])


class StrategyLoopTests(unittest.TestCase):
    def test_strategy_round_materializes_task_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            config = Path(td) / "strategy.yaml"
            config.write_text(
                f"include: [{(REPO_ROOT / 'configs' / 'default.yaml').as_posix()}]\n"
                "stop:\n  policies:\n    - {type: max_rounds, max_rounds: 1}\n"
                "data:\n  strategy:\n    enabled: true\n    count: 12\n"
                "    task:\n      id: reasoning\n"
                "      abilities: {math.reasoning: 0.6, code.debugging: 0.4}\n"
                "      datasets: [math_word_problems, code_debugging]\n",
                encoding="utf-8",
            )
            stdout, stderr = io.StringIO(), io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = main(
                    ["run", "--config", str(config), "--run-id", "strategy",
                     "--runs-dir", td, "--quiet"]
                )
            self.assertEqual(code, 0, msg=stdout.getvalue() + stderr.getvalue())
            state = read_json(Path(td) / "strategy" / "state.json")
            first = state["history"][0]
            selection = first["synthesize"]["dataset_selection"]
            self.assertIn("math_word_problems", selection["selected"])
            self.assertIn("strategy", first["synthesize"])
            self.assertEqual(
                first["synthesize"]["strategy"]["target"]["initial"],
                {"math.reasoning": 0.6, "code.debugging": 0.4},
            )
            manifest = read_json(Path(first["synthesize"]["dataset_path"]) / "manifest.json")
            self.assertGreater(
                manifest["stats"]["sources"].get("dataset:math_word_problems", 0), 0
            )
            self.assertIn("data_strategy", manifest["stats"])


if __name__ == "__main__":
    unittest.main()
