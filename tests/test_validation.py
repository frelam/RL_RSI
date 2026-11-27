from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from rsi.core.config import Config
from rsi.core.types import BenchmarkReport, Checkpoint, DataPlan, DataRequest, ItemResult
from rsi.loops.stages import ValidateStage
from tests.helpers import REPO_ROOT


class _Inference:
    def ensure(self, checkpoint):
        return None


class _Benchmark:
    def __init__(self, scores):
        self.scores = scores

    def evaluate(self, checkpoint, round_no):
        target, other = self.scores[checkpoint.id]
        return BenchmarkReport(
            benchmark="suite",
            checkpoint_id=checkpoint.id,
            round=round_no,
            items=[
                ItemResult("target-1", "target", score=target, passed=target >= 0.5),
                ItemResult("other-1", "other", score=other, passed=other >= 0.5),
            ],
        )


class _Artifacts:
    def write_json(self, round_no, stage, name, value):
        return None


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.config = Config(
            {
                "validate": {
                    "mode": "benchmark",
                    "metric": "mean_score",
                    "min_delta": 0.0,
                    "target_min_delta": 0.0,
                    "max_other_regression": 0.01,
                    "allow_regression": False,
                    "benchmark": {"names": []},
                },
                "collect": {"benchmark": {"enabled": True, "names": ["suite"]}},
            },
            config_dir=REPO_ROOT,
            repo_root=REPO_ROOT,
        )
        self.incumbent = Checkpoint(id="current", version=0, model_path="current")
        self.candidate = Checkpoint(id="candidate", version=1, model_path="candidate")
        self.runtime = SimpleNamespace(
            config=self.config,
            logger=logging.getLogger("test.validation"),
            catalog=None,
            artifacts=_Artifacts(),
            inference=lambda: _Inference(),
            benchmark=lambda name: _Benchmark(
                {
                    "current": (0.5, 0.8),
                    "candidate": (0.7, 0.795),
                }
            ),
        )
        self.stage = ValidateStage(self.runtime, self.config.section("validate"))
        baseline = _Benchmark({"current": (0.5, 0.8)}).evaluate(self.incumbent, 0)
        self.context = SimpleNamespace(
            round=0,
            memory={
                "checkpoint": self.incumbent,
                "collect": SimpleNamespace(reports=[baseline]),
                "data": SimpleNamespace(
                    plan=DataPlan(
                        round=0,
                        requests=[DataRequest(id="r", capability="target", kind="text", count=2)],
                    )
                ),
            },
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_validation_requires_target_improvement_and_allows_small_other_regression(self):
        result = self.stage.evaluate(self.context, self.candidate)
        self.assertTrue(result["accepted"])
        self.assertGreater(result["capability_deltas"]["target"], 0)
        self.assertGreaterEqual(result["capability_deltas"]["other"], -0.01)

    def test_validation_rejects_total_gain_when_target_does_not_improve(self):
        self.runtime.benchmark = lambda name: _Benchmark(
            {"current": (0.5, 0.8), "candidate": (0.5, 1.0)}
        )
        result = self.stage.evaluate(self.context, self.candidate)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["failed_targets"], ["target"])

    def test_validation_rejects_zero_score_when_no_baseline_metric_exists(self):
        result = self.stage.decision(0.0, None)
        self.assertFalse(result["accepted"])

    def test_validation_rejects_large_regression_and_incomplete_coverage(self):
        self.runtime.benchmark = lambda name: _Benchmark(
            {"current": (0.5, 0.8), "candidate": (0.7, 0.7)}
        )
        result = self.stage.evaluate(self.context, self.candidate)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["regressions"], ["other"])

        class IncompleteBenchmark:
            def evaluate(self, checkpoint, round_no):
                return BenchmarkReport(
                    benchmark="suite",
                    checkpoint_id=checkpoint.id,
                    round=round_no,
                    items=[ItemResult("target-1", "target", score=0.7)],
                )

        self.runtime.benchmark = lambda name: IncompleteBenchmark()
        result = self.stage.evaluate(self.context, self.candidate)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"], "benchmark coverage or results are incomplete")

    def test_partial_improvement_is_accepted_and_carried_to_the_next_round(self) -> None:
        class TwoTargets:
            def evaluate(self, checkpoint, round_no):
                target = 0.7 if checkpoint.id == "candidate" else 0.5
                return BenchmarkReport(
                    benchmark="suite", checkpoint_id=checkpoint.id, round=round_no,
                    items=[
                        ItemResult("target-1", "target", score=target, passed=True),
                        ItemResult("lag-1", "lag", score=0.5, passed=True),
                    ],
                )

        self.runtime.benchmark = lambda name: TwoTargets()
        self.context.memory["collect"] = SimpleNamespace(reports=[TwoTargets().evaluate(self.incumbent, 0)])
        self.context.memory["data"].plan.requests.append(
            DataRequest(id="r2", capability="lag", kind="text", count=2)
        )
        self.context.state = SimpleNamespace(pending_capabilities=["older"])
        self.runtime.checkpoints = SimpleNamespace(update_metrics=lambda *args, **kwargs: None)
        result = self.stage.evaluate(self.context, self.candidate)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["improved_targets"], ["target"])
        self.assertEqual(result["failed_targets"], ["lag"])
        self.assertIn("carried to the next round: lag", result["reason"])
        self.stage.record(self.context, self.candidate, result, "validation")
        self.assertEqual(self.context.state.pending_capabilities, ["lag", "older"])
        self.context.memory["data"].plan.requests.append(
            DataRequest(id="r3", capability="flat", kind="text", count=2)
        )
        result = self.stage.evaluate(self.context, self.candidate)
        self.assertTrue(result["accepted"])  # "flat" is not measured: reported, not a failure
        self.assertEqual(result["unmeasured_targets"], ["flat"])


if __name__ == "__main__":
    unittest.main()
