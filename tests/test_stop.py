from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from rsi.core.config import Config
from rsi.loops.stop import (
    BudgetPolicy,
    MaxRoundsPolicy,
    ManualPolicy,
    PlateauPolicy,
    StopEvaluator,
    TargetScorePolicy,
)
from tests.helpers import REPO_ROOT, FakeRuntime


def cfg(data: dict) -> Config:
    return Config(data, config_dir=REPO_ROOT, repo_root=REPO_ROOT)


def state(**kwargs) -> SimpleNamespace:
    values = {
        "round": 0,
        "best_score": -1.0,
        "best_checkpoint": None,
        "history": [],
        "budget": {"wall_hours": 0.0, "gpu_hours": 0.0},
    }
    values.update(kwargs)
    return SimpleNamespace(**values)


class StopPolicyTests(unittest.TestCase):
    def test_max_rounds(self) -> None:
        policy = MaxRoundsPolicy(SimpleNamespace(logger=__import__("logging").getLogger("t")), cfg({"max_rounds": 2}))
        self.assertIsNone(policy.evaluate(state(round=1), {}))
        decision = policy.evaluate(state(round=2), {})
        self.assertTrue(decision.should_stop)
        self.assertIn("max_rounds", decision.reason)

    def test_target_score_min_rounds(self) -> None:
        policy = TargetScorePolicy(
            SimpleNamespace(logger=__import__("logging").getLogger("t")),
            cfg({"target": 0.6, "min_rounds": 1}),
        )
        self.assertIsNone(policy.evaluate(state(round=0, best_score=0.9), {}))
        self.assertIsNone(policy.evaluate(state(round=1, best_score=0.5), {}))
        self.assertTrue(policy.evaluate(state(round=1, best_score=0.61), {}).should_stop)

    def test_plateau(self) -> None:
        policy = PlateauPolicy(
            SimpleNamespace(logger=__import__("logging").getLogger("t")),
            cfg({"patience": 2, "min_delta": 0.01}),
        )
        history = [
            {"validation": {"score": 0.5}},
            {"validation": {"score": 0.505}},
            {"validation": {"score": 0.503}},
        ]
        decision = policy.evaluate(state(round=3, history=history), {})
        self.assertIsNotNone(decision)
        self.assertIn("plateau", decision.reason)

    def test_budget(self) -> None:
        policy = BudgetPolicy(
            SimpleNamespace(logger=__import__("logging").getLogger("t")),
            cfg({"max_wall_hours": 1.0}),
        )
        self.assertIsNone(policy.evaluate(state(budget={"wall_hours": 0.5}), {}))
        self.assertTrue(policy.evaluate(state(budget={"wall_hours": 1.2}), {}).should_stop)

    def test_manual_flag(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            policy = ManualPolicy(
                SimpleNamespace(
                    logger=__import__("logging").getLogger("t"),
                    run_dir=Path(td),
                    path=lambda value: Path(td) / value,
                ),
                cfg({"file": "STOP"}),
            )
            self.assertIsNone(policy.evaluate(state(), {}))
            (Path(td) / "STOP").write_text("stop", encoding="utf-8")
            self.assertTrue(policy.evaluate(state(), {}).should_stop)

    def test_evaluator_combines_policies(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            runtime = FakeRuntime(td)
            evaluator = StopEvaluator(
                runtime,
                cfg(
                    {
                        "combine": "any",
                        "policies": [
                            {"type": "max_rounds", "max_rounds": 5},
                            {"type": "target_score", "target": 0.5, "min_rounds": 1},
                        ],
                    }
                ),
            )
            decision = evaluator.evaluate(state(round=1, best_score=0.6), {})
            self.assertTrue(decision.should_stop)
            self.assertIn("target score", decision.reason)


if __name__ == "__main__":
    unittest.main()
