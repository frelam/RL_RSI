from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from rsi.core.interfaces import MonitorAlgorithm
from rsi.core.registry import register
from rsi.core.types import TrainingResult


@dataclass
class Finding:
    code: str
    severity: str
    message: str
    step: int | None = None
    metric: str | None = None
    value: float | None = None
    suggestion: dict[str, Any] = field(default_factory=dict)


@dataclass
class SupervisorVerdict:
    use_checkpoint: bool
    retrain: bool
    reason: str
    adjustments: dict[str, Any] = field(default_factory=dict)


def _values(history: list[dict[str, Any]], metric: str, limit: int = 0) -> list[float]:
    values: list[float] = []
    for record in history:
        value = (record.get("metrics") or {}).get(metric)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(float(value))
    if limit > 0:
        values = values[-limit:]
    return values


def _is_bad_number(value: Any) -> bool:
    return isinstance(value, float) and (math.isnan(value) or math.isinf(value))


@register("monitor", "nan_guard")
class NaNGuard(MonitorAlgorithm):
    def observe(self, history: list[dict[str, Any]]) -> list[Finding]:
        if not history:
            return []
        record = history[-1]
        bad = [key for key, value in (record.get("metrics") or {}).items() if _is_bad_number(value)]
        if not bad:
            return []
        return [
            Finding(
                "nan_metric",
                "critical",
                f"non-finite values in metrics: {', '.join(bad)}",
                step=record.get("step"),
                metric=bad[0],
                suggestion={"action": "inspect data/reward scaling; revert to last stable checkpoint"},
            )
        ]


@register("monitor", "grad_spike")
class GradSpikeMonitor(MonitorAlgorithm):
    def __init__(self, runtime: Any, cfg: Any, **params: Any) -> None:
        super().__init__(runtime, cfg, **params)
        self.threshold = self.cfg.get_float("threshold", 10.0)
        self.ratio = self.cfg.get_float("ratio", 5.0)
        self.window = self.cfg.get_int("window", 10)

    def observe(self, history: list[dict[str, Any]]) -> list[Finding]:
        values = _values(history, "grad_norm")
        if not values:
            return []
        latest = values[-1]
        baseline = sorted(values[-self.window : -1] or values)[len(values[-self.window : -1] or values) // 2] if values else 0
        if latest <= self.threshold and not (baseline > 0 and latest > baseline * self.ratio):
            return []
        return [
            Finding(
                "grad_spike",
                "critical" if latest > self.threshold * 2 else "warning",
                f"grad_norm={latest:.2f} (threshold={self.threshold}, baseline≈{baseline:.2f})",
                step=history[-1].get("step"),
                metric="grad_norm",
                value=latest,
                suggestion={"clip_grad": True, "reduce_lr": 0.5},
            )
        ]


@register("monitor", "entropy_collapse")
class EntropyCollapseMonitor(MonitorAlgorithm):
    def __init__(self, runtime: Any, cfg: Any, **params: Any) -> None:
        super().__init__(runtime, cfg, **params)
        self.threshold = self.cfg.get_float("threshold", 0.15)
        self.critical_threshold = self.cfg.get_float("critical_threshold", 0.05)
        self.patience = self.cfg.get_int("patience", 3)

    def observe(self, history: list[dict[str, Any]]) -> list[Finding]:
        values = _values(history, "entropy", self.patience)
        if len(values) < self.patience or any(value >= self.threshold for value in values):
            return []
        latest = values[-1]
        return [
            Finding(
                "entropy_collapse",
                "critical" if latest < self.critical_threshold else "warning",
                f"entropy stayed below {self.threshold} for {self.patience} steps (latest {latest:.3f})",
                step=history[-1].get("step"),
                metric="entropy",
                value=latest,
                suggestion={"entropy_coeff": 0.002, "temperature": 0.1},
            )
        ]


@register("monitor", "reward_hacking")
class RewardHackingMonitor(MonitorAlgorithm):
    def __init__(self, runtime: Any, cfg: Any, **params: Any) -> None:
        super().__init__(runtime, cfg, **params)
        self.patience = self.cfg.get_int("patience", 5)
        self.reward_rise = self.cfg.get_float("reward_rise", 0.05)
        self.val_drop = self.cfg.get_float("val_drop", 0.0)

    def observe(self, history: list[dict[str, Any]]) -> list[Finding]:
        rewards = _values(history, "reward", self.patience)
        vals = _values(history, "val_score", self.patience)
        if len(rewards) < self.patience or len(vals) < self.patience:
            return []
        reward_delta = rewards[-1] - rewards[0]
        val_delta = vals[-1] - vals[0]
        if reward_delta < self.reward_rise or val_delta > self.val_drop:
            return []
        return [
            Finding(
                "reward_hacking",
                "warning",
                f"reward rose {reward_delta:+.3f} while val_score changed {val_delta:+.3f}",
                step=history[-1].get("step"),
                metric="reward",
                value=rewards[-1],
                suggestion={"audit_reward": True, "increase_verifier_weight": True},
            )
        ]


@register("monitor", "plateau")
class PlateauMonitor(MonitorAlgorithm):
    def __init__(self, runtime: Any, cfg: Any, **params: Any) -> None:
        super().__init__(runtime, cfg, **params)
        self.patience = self.cfg.get_int("patience", 6)
        self.min_delta = self.cfg.get_float("min_delta", 0.01)

    def observe(self, history: list[dict[str, Any]]) -> list[Finding]:
        values = _values(history, "val_score")
        if len(values) <= self.patience:
            return []
        recent = max(values[-self.patience :])
        previous = max(values[: -self.patience])
        if recent - previous >= self.min_delta:
            return []
        return [
            Finding(
                "val_plateau",
                "warning",
                f"val_score improved {recent - previous:+.4f} over the last {self.patience} evals",
                step=history[-1].get("step"),
                metric="val_score",
                value=values[-1],
                suggestion={"early_stop": True},
            )
        ]


@register("monitor", "saturation")
class SaturationMonitor(MonitorAlgorithm):
    def __init__(self, runtime: Any, cfg: Any, **params: Any) -> None:
        super().__init__(runtime, cfg, **params)
        self.threshold = self.cfg.get_float("threshold", 0.9)

    def observe(self, history: list[dict[str, Any]]) -> list[Finding]:
        values = _values(history, "val_score")
        if not values or values[-1] < self.threshold:
            return []
        return [
            Finding(
                "saturation",
                "info",
                f"val_score reached {values[-1]:.3f} (>= {self.threshold}); task may be saturated",
                step=history[-1].get("step"),
                metric="val_score",
                value=values[-1],
                suggestion={"early_stop": True, "raise_difficulty": True},
            )
        ]


@register("monitor", "response_length_spike")
class ResponseLengthSpikeMonitor(MonitorAlgorithm):
    """Flag abrupt length growth, which can indicate repetition or reward hacking."""

    def __init__(self, runtime: Any, cfg: Any, **params: Any) -> None:
        super().__init__(runtime, cfg, **params)
        self.window = max(2, self.cfg.get_int("window", 5))
        self.growth = self.cfg.get_float("growth", 1.5)
        self.minimum = self.cfg.get_float("minimum_length", 256.0)

    def observe(self, history: list[dict[str, Any]]) -> list[Finding]:
        values = _values(history, "response_length", self.window + 1)
        if len(values) < self.window + 1:
            return []
        baseline = sum(values[:-1]) / len(values[:-1])
        latest = values[-1]
        if baseline <= 0 or latest < self.minimum or latest < baseline * self.growth:
            return []
        return [Finding(
            "response_length_spike", "warning",
            f"response_length rose from {baseline:.1f} to {latest:.1f}",
            step=history[-1].get("step"), metric="response_length", value=latest,
            suggestion={"audit_repetition": True, "add_length_guard": True},
        )]


@register("monitor", "training_scale")
class TrainingScaleMonitor(MonitorAlgorithm):
    """Recommend larger rollout/batch settings after stable, high-reward progress plateaus."""

    def __init__(self, runtime: Any, cfg: Any, **params: Any) -> None:
        super().__init__(runtime, cfg, **params)
        self.window = max(3, self.cfg.get_int("window", 8))
        self.reward_floor = self.cfg.get_float("reward_floor", 0.7)
        self.val_delta = self.cfg.get_float("max_val_delta", 0.005)
        self.perplexity_ceiling = self.cfg.get_float("log_perplexity_ceiling", 1.0)
        self.rollout_multiplier = self.cfg.get_float("rollout_multiplier", 1.5)
        self.batch_multiplier = self.cfg.get_float("batch_multiplier", 1.5)
        self.lr_scaling = str(self.cfg.get("lr_scaling", "sqrt")).lower()
        if self.lr_scaling not in {"linear", "sqrt", "none"}:
            raise ValueError("training_scale.lr_scaling must be 'linear', 'sqrt', or 'none'")

    def observe(self, history: list[dict[str, Any]]) -> list[Finding]:
        if len(history) < self.window:
            return []
        window = history[-self.window:]
        rewards = [r for record in window if (r := _metric(record, "reward")) is not None]
        vals = [v for record in window if (v := _metric(record, "val_score")) is not None]
        perplexities = [p for record in window if (p := _metric(record, "log_perplexity")) is not None]
        if len(rewards) < 2 or len(vals) < 2 or not perplexities:
            return []
        if min(rewards) < self.reward_floor or max(vals) - min(vals) > self.val_delta:
            return []
        if max(perplexities) > self.perplexity_ceiling:
            return []
        suggestion = {
            "rollout_n_multiplier": self.rollout_multiplier,
            "batch_size_multiplier": self.batch_multiplier,
            "action": "increase_training_scale",
        }
        if self.lr_scaling == "linear":
            suggestion["learning_rate_multiplier"] = self.batch_multiplier
        elif self.lr_scaling == "sqrt":
            suggestion["learning_rate_multiplier"] = math.sqrt(self.batch_multiplier)
        return [Finding(
            "training_scale", "info",
            "reward is strong while validation is stable and log_perplexity is low",
            step=history[-1].get("step"), metric="reward", value=rewards[-1],
            suggestion=suggestion,
        )]


@register("monitor", "scaling_law")
class ScalingLawMonitor(MonitorAlgorithm):
    """Fit a bounded logistic learning curve after its slope has peaked."""

    def __init__(self, runtime: Any, cfg: Any, **params: Any) -> None:
        super().__init__(runtime, cfg, **params)
        self.enabled = self.cfg.get_bool("enabled", True)
        self.metric = str(self.cfg.get("metric", "val_score"))
        self.min_points = max(6, self.cfg.get_int("min_points", 7))
        self.curvature_drop = self.cfg.get_float("curvature_drop", 0.1)
        self.max_rmse = self.cfg.get_float("max_rmse", 0.03)
        self.batch_metric = str(self.cfg.get("batch_metric", "batch_size"))
        self.minibatch_metric = str(self.cfg.get("minibatch_metric", "mini_batch_size"))
        self.rollout_metric = str(self.cfg.get("rollout_metric", "rollout_n"))

        if not 0.0 <= self.curvature_drop < 1.0:
            raise ValueError("scaling_law.curvature_drop must be in [0, 1)")
        if self.max_rmse <= 0:
            raise ValueError("scaling_law.max_rmse must be positive")

    def observe(self, history: list[dict[str, Any]]) -> list[Finding]:
        if not self.enabled:
            return []
        points, current_step_compute = _compute_metric_points(
            history, self.metric, self.batch_metric, self.minibatch_metric, self.rollout_metric
        )
        if len(points) < self.min_points or not _has_observed_midpoint(points, self.curvature_drop):
            return []
        fit = _fit_logistic(points)
        if fit is None:
            return []
        ceiling, rate, intercept, rmse = fit
        if rmse > self.max_rmse:
            return []

        start_score = points[0][1]
        latest_compute = points[-1][0]
        effect_levels = {}
        for label, fraction in (("p50", 0.50), ("p75", 0.75), ("p90", 0.90), ("p100", 1.00)):
            target_compute = _logistic_target_compute(
                fraction, start_score, ceiling, rate, intercept
            )
            remaining_compute = (
                max(0.0, target_compute - latest_compute)
                if target_compute is not None else None
            )
            effect_levels[label] = {
                "score": start_score + fraction * (ceiling - start_score),
                "remaining_compute": remaining_compute,
                "equivalent_steps": (
                    math.ceil(remaining_compute / current_step_compute)
                    if remaining_compute is not None and current_step_compute > 0 else None
                ),
            }
        suggestion = {
            "asymptotic_score": ceiling,
            "current_score": points[-1][1],
            "current_compute": latest_compute,
            "current_step_compute": current_step_compute,
            "effect_levels": effect_levels,
            "fit_rmse": rmse,
            "action": "review_scaling_projection",
        }
        metrics = history[-1].setdefault("metrics", {})
        metrics.update({
            "scaling_law/current_score": suggestion["current_score"],
            "scaling_law/asymptotic_score": suggestion["asymptotic_score"],
            "scaling_law/current_compute": suggestion["current_compute"],
            "scaling_law/current_step_compute": suggestion["current_step_compute"],
            "scaling_law/fit_rmse": suggestion["fit_rmse"],
        })
        for label, level in effect_levels.items():
            metrics[f"scaling_law/{label}_score"] = level["score"]
            metrics[f"scaling_law/{label}_remaining_compute"] = level["remaining_compute"]
            metrics[f"scaling_law/{label}_equivalent_steps"] = level["equivalent_steps"]
        raw_step = history[-1].get("step", history[-1].get("global_step"))
        finding_step = int(raw_step) if isinstance(raw_step, (int, float)) and not isinstance(raw_step, bool) else None
        return [Finding(
            "scaling_law_projection", "info",
            f"{self.metric} logistic projection: ceiling≈{ceiling:.3f}, "
            + f"p50/p75/p90 compute remaining≈{effect_levels['p50']['remaining_compute']:.1f}/"
            f"{effect_levels['p75']['remaining_compute']:.1f}/"
            f"{effect_levels['p90']['remaining_compute']:.1f}",
            step=finding_step, metric=self.metric, value=points[-1][1],
            suggestion=suggestion,
        )]


def _record_number(record: dict[str, Any], name: str, default: float) -> float:
    value = (record.get("metrics") or {}).get(name, record.get(name, default))
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return default


def _compute_metric_points(
    history: list[dict[str, Any]], metric: str, batch_metric: str,
    minibatch_metric: str, rollout_metric: str,
) -> tuple[list[tuple[float, float]], float]:
    points: list[tuple[float, float]] = []
    cumulative_compute = 0.0
    current_step_compute = 1.0
    for record in history:
        batch = max(1.0, _record_number(record, batch_metric, 1.0))
        minibatch = max(1.0, _record_number(record, minibatch_metric, batch))
        rollout_n = max(1.0, _record_number(record, rollout_metric, 1.0))
        current_step_compute = batch * rollout_n * max(1.0, batch / minibatch)
        cumulative_compute += current_step_compute
        value = (record.get("metrics") or {}).get(metric)
        if (
            isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value)
            and 0 <= value <= 1
        ):
            points.append((cumulative_compute, float(value)))
    return points, current_step_compute


def _has_observed_midpoint(points: list[tuple[float, float]], drop: float) -> bool:
    slopes = [
        (right_y - left_y) / (right_x - left_x)
        for (left_x, left_y), (right_x, right_y) in zip(points, points[1:])
    ]
    if len(slopes) < 4 or max(slopes[:-1]) <= 0:
        return False
    peak_index = max(range(len(slopes) - 1), key=slopes.__getitem__)
    return peak_index > 0 and slopes[-1] <= slopes[peak_index] * (1.0 - drop)


def _fit_logistic(points: list[tuple[float, float]]) -> tuple[float, float, float, float] | None:
    """Search the bounded score ceiling and solve the linearized logistic fit."""
    start_score = points[0][1]
    observed_max = max(value for _, value in points)
    if observed_max <= start_score or observed_max >= 1.0:
        return None
    span = observed_max - start_score
    candidates = [observed_max + (1.0 - observed_max) * index / 512 for index in range(1, 513)]
    candidates.extend(
        min(1.0, observed_max + span * factor)
        for factor in (0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2)
    )
    best: tuple[float, float, float, float] | None = None
    best_sse = math.inf
    fit_points = points[1:]
    if len(fit_points) < 3:
        return None
    x_mean = sum(step for step, _ in fit_points) / len(fit_points)
    x_variance = sum((step - x_mean) ** 2 for step, _ in fit_points)
    if x_variance == 0:
        return None

    for ceiling in candidates:
        logits: list[float] = []
        valid = True
        for _, value in fit_points:
            fraction = (value - start_score) / (ceiling - start_score)
            if not 0.0 < fraction < 1.0:
                valid = False
                break
            logits.append(math.log(fraction / (1.0 - fraction)))
        if not valid:
            continue
        y_mean = sum(logits) / len(logits)
        rate = sum((step - x_mean) * (logit - y_mean)
                   for (step, _), logit in zip(fit_points, logits)) / x_variance
        if rate <= 0:
            continue
        intercept = y_mean - rate * x_mean
        predictions = [
            start_score + (ceiling - start_score) / (1.0 + math.exp(-max(-60.0, min(60.0, rate * step + intercept))))
            for step, _ in points
        ]
        sse = sum((prediction - value) ** 2
                  for prediction, (_, value) in zip(predictions, points))
        if sse < best_sse:
            best_sse = sse
            best = (ceiling, rate, intercept, math.sqrt(sse / len(points)))
    return best


def _logistic_target_compute(
    fraction: float, start_score: float, ceiling: float, rate: float, intercept: float
) -> float | None:
    if fraction >= 1.0:
        return None
    target = start_score + fraction * (ceiling - start_score)
    normalized = (target - start_score) / (ceiling - start_score)
    if normalized <= 0.0:
        return 0.0
    return (math.log(normalized / (1.0 - normalized)) - intercept) / rate


def _metric(record: dict[str, Any], name: str) -> float | None:
    value = (record.get("metrics") or {}).get(name)
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


@register("monitor", "zscore")
class ZScoreMonitor(MonitorAlgorithm):
    def __init__(self, runtime: Any, cfg: Any, **params: Any) -> None:
        super().__init__(runtime, cfg, **params)
        self.metric = str(self.cfg.get("metric", "loss"))
        self.window = self.cfg.get_int("window", 20)
        self.z = self.cfg.get_float("z", 4.0)
        self.severity = str(self.cfg.get("severity", "warning"))

    def observe(self, history: list[dict[str, Any]]) -> list[Finding]:
        values = _values(history, self.metric)
        if len(values) < max(5, self.window // 2):
            return []
        window = values[-self.window :]
        mean = sum(window) / len(window)
        variance = sum((value - mean) ** 2 for value in window) / len(window)
        std = math.sqrt(variance)
        if std == 0:
            return []
        z = (values[-1] - mean) / std
        if abs(z) < self.z:
            return []
        return [
            Finding(
                f"zscore.{self.metric}",
                self.severity,
                f"{self.metric} z-score {z:+.2f} (mean {mean:.3f}, std {std:.3f})",
                step=history[-1].get("step"),
                metric=self.metric,
                value=values[-1],
            )
        ]


class MonitorHub:
    """Runs monitor plugins on every metric record and accumulates findings."""

    def __init__(self, runtime: Any, cfg: Any) -> None:
        self.runtime = runtime
        self.cfg = cfg
        self.algorithms: list[MonitorAlgorithm] = []
        for spec in cfg.specs("algorithms"):
            name = str(spec.get("name") or spec["type"])
            child = cfg.child(spec, name)
            self.algorithms.append(runtime.build("monitor", spec, child))
        self.stop_on_critical = cfg.get_bool("stop_on_critical", True)
        self.stop_on_saturation = cfg.get_bool("stop_on_saturation", True)
        self.history: list[dict[str, Any]] = []
        self.findings: dict[str, Finding] = {}

    def observe(self, record: dict[str, Any]) -> str | None:
        self.history.append(record)
        critical = False
        saturation = False
        for algorithm in self.algorithms:
            try:
                findings = algorithm.observe(self.history)
            except Exception:  # noqa: BLE001 - monitoring must be best-effort
                continue
            for finding in findings:
                self.findings[finding.code] = finding
                if finding.severity == "critical":
                    critical = True
                if finding.code == "saturation" and finding.suggestion.get("early_stop"):
                    saturation = True
        if (critical and self.stop_on_critical) or (saturation and self.stop_on_saturation):
            return "stop"
        return None

    def snapshot(self) -> list[dict[str, Any]]:
        from rsi.core.jsonio import to_plain

        return [to_plain(finding) for finding in self.findings.values()]


class Supervisor:
    """Decides whether to use, retrain or discard a training run."""

    def __init__(self, runtime: Any, cfg: Any) -> None:
        self.runtime = runtime
        self.cfg = cfg
        self.enabled = cfg.get_bool("enabled", True)
        self.max_retries = cfg.get_int("max_retries", 1)
        self.retrain_on_critical = cfg.get_bool("retrain_on_critical", True)
        self.retrain_on_failure = cfg.get_bool("retrain_on_failure", True)
        self.discard_on_critical = cfg.get_bool("discard_on_critical", False)

    def verdict(
        self,
        result: TrainingResult,
        findings: list[dict[str, Any]],
        attempt: int,
    ) -> SupervisorVerdict:
        if not self.enabled:
            return SupervisorVerdict(
                use_checkpoint=bool(result.checkpoint_path),
                retrain=False,
                reason="supervisor disabled",
            )
        critical = [f for f in findings if f.get("severity") == "critical"]
        warning = [f for f in findings if f.get("severity") == "warning"]
        adjustments: dict[str, Any] = {}
        for finding in critical + warning:
            suggestion = finding.get("suggestion") or {}
            adjustments.update({k: v for k, v in suggestion.items() if k != "action"})

        if result.status in {"failed", "timeout"}:
            if self.retrain_on_failure and attempt < self.max_retries:
                return SupervisorVerdict(False, True, f"training {result.status}", adjustments)
            return SupervisorVerdict(False, False, f"training {result.status}; no retry left")
        if result.checkpoint_path is None:
            if attempt < self.max_retries:
                return SupervisorVerdict(False, True, "no checkpoint produced", adjustments)
            return SupervisorVerdict(False, False, "no checkpoint produced")
        if critical:
            codes = ", ".join(sorted({f.get("code", "?") for f in critical}))
            if self.retrain_on_critical and attempt < self.max_retries:
                return SupervisorVerdict(False, True, f"critical findings: {codes}", adjustments)
            if self.discard_on_critical:
                return SupervisorVerdict(False, False, f"critical findings: {codes}")
            return SupervisorVerdict(True, False, f"using best checkpoint despite: {codes}", adjustments)
        reason = f"accepted ({len(warning)} warning(s))" if warning else "accepted"
        return SupervisorVerdict(True, False, reason, adjustments)
