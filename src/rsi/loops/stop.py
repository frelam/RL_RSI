from __future__ import annotations

import math
import time
from typing import Any

from rsi.core.interfaces import StopPolicy
from rsi.core.registry import register
from rsi.core.types import StopDecision


def _history_with(state: Any, round_record: dict[str, Any]) -> list[dict[str, Any]]:
    """Prior round records plus the record under evaluation."""
    history = list(getattr(state, "history", None) or [])
    if round_record and (not history or history[-1] is not round_record):
        history.append(round_record)
    return history


def _scaling_projection(history: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Latest ``scaling_law_projection`` suggestion in the training records."""
    for record in reversed(history):
        train = record.get("train") if isinstance(record, dict) else None
        if not isinstance(train, dict):
            continue
        sources: list[Any] = [train.get("findings")]
        sources.extend(
            item.get("findings")
            for item in train.get("strategy") or []
            if isinstance(item, dict)
        )
        for findings in sources:
            if not isinstance(findings, list):
                continue
            for finding in findings:
                if not isinstance(finding, dict) or finding.get("code") != "scaling_law_projection":
                    continue
                suggestion = finding.get("suggestion")
                if isinstance(suggestion, dict):
                    return suggestion
    return None


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


@register("stop_policy", "data_reuse")
class DataReusePolicy(StopPolicy):
    """Stop when the data pool is empty or has been trained on for too many epochs.

    Every round's ``synthesize`` record reports the rows consumed and a
    ``data_pool`` summary with the distinct rows materialized so far. The
    epoch count is cumulative rows consumed divided by that unique pool, so a
    loop that keeps replaying the same rows stops instead of overfitting.
    """

    def __init__(self, runtime: Any, cfg: Any, **params: Any) -> None:
        super().__init__(runtime, cfg, **params)
        self.max_epochs = self.cfg.get_float("max_epochs", 2.0)
        self.min_rounds = self.cfg.get_int("min_rounds", 1)
        self.stop_on_exhausted = self.cfg.get_bool("stop_on_exhausted", True)
        if not math.isfinite(self.max_epochs) or self.max_epochs < 0:
            raise ValueError("data_reuse.max_epochs must be finite and non-negative")
        if self.min_rounds < 0:
            raise ValueError("data_reuse.min_rounds must be non-negative")

    def evaluate(self, state: Any, round_record: dict[str, Any]) -> StopDecision | None:
        if state.round < self.min_rounds:
            return None
        rounds = [
            record["synthesize"]
            for record in _history_with(state, round_record)
            if isinstance(record, dict)
            and isinstance(record.get("synthesize"), dict)
            and "selected" in record["synthesize"]
        ]
        if not rounds:
            return None
        consumed = sum(int(entry.get("selected") or 0) for entry in rounds)
        pool = int((rounds[-1].get("data_pool") or {}).get("pool_size") or 0)
        if pool <= 0:
            pool = sum(int((entry.get("data_pool") or {}).get("unique_new") or 0) for entry in rounds)
        last = rounds[-1]
        if (
            self.stop_on_exhausted
            and int(last.get("selected") or 0) == 0
            and int(last.get("replayed") or 0) == 0
        ):
            return StopDecision(
                True,
                "data exhausted: the latest round produced no training rows",
                {
                    "policy": "data_reuse",
                    "trigger": "exhausted",
                    "consumed": consumed,
                    "pool_size": pool,
                },
            )
        if self.max_epochs <= 0 or pool <= 0:
            return None
        epochs = consumed / pool
        if epochs < self.max_epochs:
            return None
        return StopDecision(
            True,
            f"data reused for {epochs:.2f} epochs (>= {self.max_epochs}); stop to avoid overfitting",
            {
                "policy": "data_reuse",
                "trigger": "max_epochs",
                "epochs": epochs,
                "consumed": consumed,
                "pool_size": pool,
                "max_epochs": self.max_epochs,
            },
        )


@register("stop_policy", "scaling_law")
class ScalingLawPolicy(StopPolicy):
    """Stop when the fitted ceiling is near, or reaching it is too expensive.

    It reads the latest ``scaling_law_projection`` finding from the training
    monitor. The peak trigger fires when the projected ceiling adds at most
    ``min_margin`` over the latest observed score. The budget trigger fires
    when the remaining compute to p90 exceeds ``max_remaining_compute`` while
    the projected gain stays within ``budget_margin``.
    """

    def __init__(self, runtime: Any, cfg: Any, **params: Any) -> None:
        super().__init__(runtime, cfg, **params)
        self.min_rounds = self.cfg.get_int("min_rounds", 1)
        self.min_margin = self.cfg.get_float("min_margin", 0.01)
        self.max_remaining_compute = self.cfg.get_float("max_remaining_compute", 0.0)
        self.budget_margin = self.cfg.get_float("budget_margin", 0.05)
        for name, value in (
            ("min_margin", self.min_margin),
            ("max_remaining_compute", self.max_remaining_compute),
            ("budget_margin", self.budget_margin),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"scaling_law.{name} must be finite and non-negative")

    def evaluate(self, state: Any, round_record: dict[str, Any]) -> StopDecision | None:
        if state.round < self.min_rounds:
            return None
        projection = _scaling_projection(_history_with(state, round_record))
        if projection is None:
            return None
        ceiling = projection.get("asymptotic_score")
        current = projection.get("current_score")
        if not all(
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            for value in (ceiling, current)
        ):
            return None
        ceiling, current = float(ceiling), float(current)
        margin = ceiling - current
        details = {
            "policy": "scaling_law",
            "ceiling": ceiling,
            "current": current,
            "margin": margin,
            "best_score": state.best_score,
        }
        if margin <= self.min_margin:
            return StopDecision(
                True,
                f"projected ceiling {ceiling:.3f} is within {self.min_margin} of the current score "
                f"{current:.3f}",
                {**details, "trigger": "theoretical_peak"},
            )
        if self.max_remaining_compute > 0:
            levels = projection.get("effect_levels") or {}
            remaining = (levels.get("p90") or {}).get("remaining_compute")
            if (
                isinstance(remaining, (int, float))
                and not isinstance(remaining, bool)
                and math.isfinite(remaining)
                and remaining > self.max_remaining_compute
                and margin <= self.budget_margin
            ):
                return StopDecision(
                    True,
                    f"p90 needs {remaining:.0f} more compute (budget "
                    f"{self.max_remaining_compute:.0f}) for a projected gain of {margin:.3f}",
                    {**details, "trigger": "compute_budget", "remaining_compute": remaining},
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
