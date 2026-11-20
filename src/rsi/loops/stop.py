from __future__ import annotations

import time
from typing import Any

from rsi.core.interfaces import StopPolicy
from rsi.core.registry import register
from rsi.core.types import StopDecision


@register("stop_policy", "max_rounds")
class MaxRoundsPolicy(StopPolicy):
    def __init__(self, runtime: Any, cfg: Any, **params: Any) -> None:
        super().__init__(runtime, cfg, **params)
        self.max_rounds = self.cfg.get_int("max_rounds", 5)

    def evaluate(self, state: Any, round_record: dict[str, Any]) -> StopDecision | None:
        if state.round >= self.max_rounds:
            return StopDecision(
                True,
                f"max_rounds reached ({state.round}/{self.max_rounds})",
                {"policy": "max_rounds", "rounds": state.round},
            )
        return None


@register("stop_policy", "target_score")
class TargetScorePolicy(StopPolicy):
    def __init__(self, runtime: Any, cfg: Any, **params: Any) -> None:
        super().__init__(runtime, cfg, **params)
        self.target = self.cfg.get_float("target", 0.8)
        self.min_rounds = self.cfg.get_int("min_rounds", 1)

    def evaluate(self, state: Any, round_record: dict[str, Any]) -> StopDecision | None:
        if state.round < self.min_rounds:
            return None
        if state.best_score >= self.target:
            return StopDecision(
                True,
                f"target score reached ({state.best_score:.4f} >= {self.target})",
                {"policy": "target_score", "score": state.best_score, "target": self.target},
            )
        return None


@register("stop_policy", "plateau")
class PlateauPolicy(StopPolicy):
    def __init__(self, runtime: Any, cfg: Any, **params: Any) -> None:
        super().__init__(runtime, cfg, **params)
        self.patience = self.cfg.get_int("patience", 2)
        self.min_delta = self.cfg.get_float("min_delta", 0.005)
        self.source = str(self.cfg.get("source", "candidate"))

    def evaluate(self, state: Any, round_record: dict[str, Any]) -> StopDecision | None:
        scores = []
        for record in state.history:
            validation = record.get("validation") or {}
            value = validation.get("score") if self.source == "candidate" else validation.get("best_score")
            if isinstance(value, (int, float)):
                scores.append(float(value))
        if len(scores) <= self.patience:
            return None
        recent = max(scores[-self.patience :])
        previous = max(scores[: -self.patience])
        if recent - previous < self.min_delta:
            return StopDecision(
                True,
                f"validation plateau: best {recent:.4f} vs previous {previous:.4f} "
                f"over {self.patience} round(s)",
                {
                    "policy": "plateau",
                    "recent": recent,
                    "previous": previous,
                    "patience": self.patience,
                },
            )
        return None


@register("stop_policy", "regression")
class RegressionPolicy(StopPolicy):
    def __init__(self, runtime: Any, cfg: Any, **params: Any) -> None:
        super().__init__(runtime, cfg, **params)
        self.drop = self.cfg.get_float("drop", 0.05)
        self.min_rounds = self.cfg.get_int("min_rounds", 2)

    def evaluate(self, state: Any, round_record: dict[str, Any]) -> StopDecision | None:
        if state.round < self.min_rounds or state.best_score < 0:
            return None
        validation = round_record.get("validation") or {}
        score = validation.get("score")
        if not isinstance(score, (int, float)):
            return None
        if float(score) < state.best_score - self.drop:
            return StopDecision(
                True,
                f"validation regression: {score:.4f} < best {state.best_score:.4f} - {self.drop}",
                {"policy": "regression", "score": score, "best": state.best_score},
            )
        return None


@register("stop_policy", "budget")
class BudgetPolicy(StopPolicy):
    def __init__(self, runtime: Any, cfg: Any, **params: Any) -> None:
        super().__init__(runtime, cfg, **params)
        self.max_wall_hours = self.cfg.get_float("max_wall_hours", 0.0)
        self.max_gpu_hours = self.cfg.get_float("max_gpu_hours", 0.0)

    def evaluate(self, state: Any, round_record: dict[str, Any]) -> StopDecision | None:
        budget = state.budget or {}
        wall = float(budget.get("wall_hours") or 0.0)
        gpu = float(budget.get("gpu_hours") or 0.0)
        if self.max_wall_hours > 0 and wall >= self.max_wall_hours:
            return StopDecision(
                True,
                f"wall-clock budget exceeded ({wall:.2f}h >= {self.max_wall_hours}h)",
                {"policy": "budget", "wall_hours": wall},
            )
        if self.max_gpu_hours > 0 and gpu >= self.max_gpu_hours:
            return StopDecision(
                True,
                f"GPU-hour budget exceeded ({gpu:.2f} >= {self.max_gpu_hours})",
                {"policy": "budget", "gpu_hours": gpu},
            )
        return None


@register("stop_policy", "manual")
class ManualPolicy(StopPolicy):
    def __init__(self, runtime: Any, cfg: Any, **params: Any) -> None:
        super().__init__(runtime, cfg, **params)
        self.filename = str(self.cfg.get("file", "STOP"))

    def evaluate(self, state: Any, round_record: dict[str, Any]) -> StopDecision | None:
        candidates = [self.runtime.run_dir / self.filename, self.runtime.path(self.filename)]
        for path in candidates:
            if path.exists():
                return StopDecision(
                    True,
                    f"manual stop flag found: {path}",
                    {"policy": "manual", "path": str(path)},
                )
        return None


class StopEvaluator:
    """Combines stop policies (OR by default) into one decision."""

    def __init__(self, runtime: Any, cfg: Any) -> None:
        self.runtime = runtime
        self.cfg = cfg
        self.combine = str(cfg.get("combine", "any"))
        specs = cfg.specs("policies")
        if not specs:
            specs = [{"type": "max_rounds", "max_rounds": 5, "name": "max_rounds"}]
            runtime.logger.warning("No stop policies configured; defaulting to max_rounds=5")
        self.policies: list[StopPolicy] = []
        for spec in specs:
            name = str(spec.get("name") or spec["type"])
            child = cfg.child(spec, name)
            self.policies.append(runtime.build("stop_policy", spec, child))

    def evaluate(self, state: Any, round_record: dict[str, Any]) -> StopDecision:
        decisions = []
        for policy in self.policies:
            try:
                decision = policy.evaluate(state, round_record)
            except Exception as exc:  # noqa: BLE001 - a broken policy must not stop the loop silently
                self.runtime.logger.warning("stop policy %s failed: %s", type(policy).__name__, exc)
                continue
            if decision is not None and decision.should_stop:
                decisions.append(decision)
        if not decisions:
            return StopDecision(False, "", {})
        if self.combine == "all":
            fired = {decision.details.get("policy") for decision in decisions}
            if len(fired) < len(self.policies):
                return StopDecision(False, "", {"partial": [d.reason for d in decisions]})
            return StopDecision(
                True,
                "; ".join(decision.reason for decision in decisions),
                {"policies": [decision.details for decision in decisions]},
            )
        return StopDecision(
            True,
            decisions[0].reason,
            {
                "policies": [decision.details for decision in decisions],
                "all_reasons": [decision.reason for decision in decisions],
            },
        )
