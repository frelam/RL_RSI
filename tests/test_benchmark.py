from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rsi.adapters.benchmark import JsonlBenchmark, aggregate_items
from rsi.core.config import Config
from rsi.core.jsonio import write_jsonl
from rsi.core.types import Checkpoint, ItemResult
from tests.helpers import REPO_ROOT, FakeRuntime


class BenchmarkTests(unittest.TestCase):
    def test_jsonl_benchmark_scores_through_inference(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bench.jsonl"
            write_jsonl(
                path,
                [
                    {
                        "id": "item-1",
                        "prompt": "The reference value 391 appears here. Repeat 391.",
                        "reference": "391",
                        "scorer": "contains",
                        "capability": "math.reasoning",
                    },
                    {
                        "id": "item-2",
                        "prompt": "The reference value 7 appears here. Repeat 7.",
                        "reference": "7",
                        "scorer": "contains",
                        "capability": "math.reasoning",
                    },
                ],
            )
            runtime = FakeRuntime(td)
            cfg = Config(
                {
                    "path": str(path),
                    "samples_per_item": 1,
                    "k": 1,
                    "scorer": "contains",
                    "store_responses": True,
                },
                config_dir=REPO_ROOT,
                repo_root=REPO_ROOT,
            )
            benchmark = JsonlBenchmark(runtime, cfg)
            report = benchmark.evaluate(Checkpoint(id="c0", version=0, model_path="mock"), 0)
            self.assertEqual(report.num_items, 2)
            self.assertAlmostEqual(report.aggregates["mean_score"], 1.0)
            self.assertAlmostEqual(report.capabilities["math.reasoning"], 1.0)
            self.assertTrue(all(item.passed for item in report.items))

    def test_aggregate_capabilities_and_subcapabilities(self) -> None:
        items = [
            ItemResult("a", "math", "algebra", score=1.0, passed=True, response="x"),
            ItemResult("b", "math", "algebra", score=0.0, passed=False, response="x"),
            ItemResult("c", "code", "", score=0.5, passed=True, response="x"),
        ]
        caps, subs, aggregates = aggregate_items(items)
        self.assertAlmostEqual(caps["math"], 0.5)
        self.assertAlmostEqual(subs["math.algebra"], 0.5)
        self.assertAlmostEqual(aggregates["mean_score"], 0.5)
        self.assertAlmostEqual(aggregates["pass_rate"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
