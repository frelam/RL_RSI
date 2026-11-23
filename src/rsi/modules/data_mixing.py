from __future__ import annotations

import math
from typing import Any

from rsi.core.interfaces import MixingStrategy
from rsi.core.jsonio import stable_hash
from rsi.core.registry import register
from rsi.core.types import DataPlan, DataSample


def group_key(sample: DataSample) -> str:
    return sample.group or sample.capability or "unknown"


@register("mixing", "static")
class StaticMixing(MixingStrategy):
    def __init__(self, runtime: Any, cfg: Any, **params: Any) -> None:
        super().__init__(runtime, cfg, **params)
        self.weights = {str(k): float(v) for k, v in self.cfg.get_dict("weights", {}).items()}
        self.last_info: dict[str, Any] = {}

    def plan(self, samples: list[DataSample], plan: DataPlan, context: Any) -> dict[str, float]:
        groups = sorted({group_key(sample) for sample in samples})
        if not self.weights:
            self.last_info = {"strategy": "static", "weights": {g: 1.0 for g in groups}}
            return {g: 1.0 for g in groups}
        weights = {g: max(0.0, self.weights.get(g, 0.0)) for g in groups}
        if sum(weights.values()) == 0:
            weights = {g: 1.0 for g in groups}
        self.last_info = {"strategy": "static", "weights": weights}
        return weights


@register("mixing", "capability_weighted")
class CapabilityWeightedMixing(MixingStrategy):
    """Weight groups by defect severity, with a configurable exploration floor."""

    def __init__(self, runtime: Any, cfg: Any, **params: Any) -> None:
        super().__init__(runtime, cfg, **params)
        self.floor = self.cfg.get_float("floor", 0.15)
        self.priority_bonus = self.cfg.get_float("priority_bonus", 0.5)
        self.last_info: dict[str, Any] = {}

    def plan(self, samples: list[DataSample], plan: DataPlan, context: Any) -> dict[str, float]:
        severity: dict[str, float] = {}
        if context.defects is not None:
            for defect in context.defects.defects:
                signal = defect.severity * defect.confidence
                severity[defect.capability] = max(severity.get(defect.capability, 0.0), signal)
        groups = sorted({group_key(sample) for sample in samples})
        weights: dict[str, float] = {}
        for group in groups:
            capability = group.split("::", 1)[0]
            signal = severity.get(capability, 0.0)
            priority = self.runtime.catalog.priority(capability)
            weights[group] = self.floor + signal + self.priority_bonus * priority * signal
        if sum(weights.values()) == 0:
            weights = {group: 1.0 for group in groups}
        self.last_info = {"strategy": "capability_weighted", "weights": weights, "severity": severity}
        return weights


@register("mixing", "history_bandit")
class HistoryBanditMixing(MixingStrategy):
    """Explore a small candidate set, then exploit the best validation delta.

    Candidate choice is recorded on the dataset artifact by DataPipeline; the
    strategy reads it back from prior round records, so the loop stays
    deterministic and auditable.
    """

    def __init__(self, runtime: Any, cfg: Any, **params: Any) -> None:
        super().__init__(runtime, cfg, **params)
        candidates = self.cfg.get_list("candidates", [])
        if not candidates:
            raise ValueError("history_bandit mixing requires a 'candidates' list")
        self.candidates: list[dict[str, float]] = [
            {str(k): float(v) for k, v in candidate.items()} for candidate in candidates if isinstance(candidate, dict)
        ]
        if not self.candidates:
            raise ValueError("history_bandit candidates must be mappings")
        self.explore_rounds = self.cfg.get_int("explore_rounds", len(self.candidates))
        self.last_info: dict[str, Any] = {}

    def plan(self, samples: list[DataSample], plan: DataPlan, context: Any) -> dict[str, float]:
        groups = sorted({group_key(sample) for sample in samples})
        choice = context.round % len(self.candidates)
        history = getattr(context.state, "history", []) if context.state is not None else []
        if context.round >= self.explore_rounds:
            scores: dict[int, list[float]] = {}
            for record in history:
                index = (record.get("synthesize") or record.get("data") or {}).get("mixing_candidate")
                delta = (record.get("validation") or {}).get("delta")
                try:
                    index, delta = int(index), float(delta)
                except (TypeError, ValueError):
                    continue
                if not 0 <= index < len(self.candidates) or not math.isfinite(delta):
                    continue
                scores.setdefault(index, []).append(delta)
            if scores:
                choice = max(
                    scores,
                    key=lambda index: (sum(scores[index]) / len(scores[index]), -index),
                )
        weights = {
            group: max(0.0, self.candidates[choice].get(group, 0.0))
            for group in groups
        }
        if sum(weights.values()) == 0:
            weights = {group: 1.0 for group in groups}
        self.last_info = {
            "strategy": "history_bandit",
            "candidate": choice,
            "candidates": len(self.candidates),
            "weights": weights,
        }
        return weights


class DiversitySampler:
    """Apply group quotas, difficulty band and deterministic shuffling."""

    def __init__(self, runtime: Any, cfg: Any) -> None:
        self.runtime = runtime
        self.cfg = cfg
        self.target_total = cfg.get_int("target_total", 0)
        self.min_per_group = cfg.get_int("min_per_group", 1)
        self.max_per_group = cfg.get_int("max_per_group", 0)
        self.band = [float(v) for v in cfg.get_list("target_band", [0.0, 1.0])]
        valid_band = (
            len(self.band) == 2
            and all(math.isfinite(value) for value in self.band)
            and 0 <= self.band[0] <= self.band[1] <= 1
        )
        if not valid_band:
            raise ValueError("mixing.target_band must be an ordered pair within [0, 1]")
        self.enforce_band = cfg.get_bool("enforce_band", True)
        self.shuffle = cfg.get_bool("shuffle", True)
        self.preserve_ratio = cfg.get_bool("preserve_ratio", False)
        self.relax_by = cfg.get_float("relax_by", 0.15)
        if not math.isfinite(self.relax_by) or self.relax_by < 0:
            raise ValueError("mixing.relax_by must be non-negative")
        self.last_stats: dict[str, Any] = {}

    def select(
        self, samples: list[DataSample], weights: dict[str, float], context: Any
    ) -> tuple[list[DataSample], dict[str, Any]]:
        low, high = self.band
        strict: dict[str, list[DataSample]] = {}
        relaxed: dict[str, list[DataSample]] = {}
        rng_seed = stable_hash(self.runtime.seed, context.round, "sampler")
        for sample in sorted(samples, key=lambda item: stable_hash(rng_seed, item.id)):
            group = group_key(sample)
            bounds = sample.metadata.get("target_difficulty", [0.0, 1.0])
            lower, upper = max(low, bounds[0]), min(high, bounds[1])
            if not self.enforce_band or lower <= sample.difficulty <= upper:
                strict.setdefault(group, []).append(sample)
            elif self.preserve_ratio and lower - self.relax_by <= sample.difficulty <= upper + self.relax_by:
                relaxed.setdefault(group, []).append(sample)

        groups = strict
        if self.preserve_ratio:
            expanded = {group: strict.get(group, []) + relaxed.get(group, [])
                        for group in sorted(set(strict) | set(relaxed) | set(weights))}
            quotas = self._ratio_quotas(strict, weights)
            relaxed_quotas = self._ratio_quotas(expanded, weights)
            if sum(relaxed_quotas.values()) > sum(quotas.values()):
                groups, quotas = expanded, relaxed_quotas
        elif self.target_total <= 0:
            quotas = {group: self._capacity(items) for group, items in groups.items()}
        else:
            quotas = {group: 0 for group in groups}
            for _ in range(self.target_total):
                available = [group for group, items in groups.items()
                             if quotas[group] < self._capacity(items)]
                if not available:
                    break
                group = max(available, key=lambda name: (
                    quotas[name] < self.min_per_group,
                    max(0.0, weights.get(name, 0.0)) / (quotas[name] + 1),
                    name,
                ))
                quotas[group] += 1
        selected = [sample for group, items in groups.items() for sample in items[:quotas.get(group, 0)]]
        if self.shuffle:
            selected.sort(key=lambda s: stable_hash(rng_seed, "final", s.id))
        counts: dict[str, int] = {}
        for sample in selected:
            counts[group_key(sample)] = counts.get(group_key(sample), 0) + 1
        relaxed_selected = sum(
            max(0, count - len(strict.get(group, []))) for group, count in counts.items()
        )
        stats = {
            "input": len(samples),
            "band_dropped": len(samples) - sum(map(len, strict.values())) - relaxed_selected,
            "selected": len(selected),
            "groups": counts,
            "target_total": self.target_total,
            "band": [low, high],
            "relaxed_selected": relaxed_selected,
        }
        self.last_stats = stats
        return selected, stats

    def _capacity(self, items: list[DataSample]) -> int:
        return min(len(items), self.max_per_group) if self.max_per_group > 0 else len(items)

    def _ratio_quotas(
        self, groups: dict[str, list[DataSample]], weights: dict[str, float]
    ) -> dict[str, int]:
        active = sorted(group for group, weight in weights.items() if weight > 0)
        capacities = {group: self._capacity(groups.get(group, [])) for group in active}
        if not active or any(capacities[group] < self.min_per_group for group in active):
            return {}
        total_weight = sum(weights[group] for group in active)
        if not math.isfinite(total_weight) or total_weight <= 0:
            return {}
        available = sum(capacities[group] for group in active)
        target = min(self.target_total, available) if self.target_total > 0 else available
        quotas = {group: 0 for group in active}
        minimums = {group: min(self.min_per_group, capacities[group]) for group in active}
        if target >= sum(minimums.values()):
            quotas.update(minimums)
        for _ in range(target - sum(quotas.values())):
            available_groups = [
                group for group in active if quotas[group] < capacities[group]
            ]
            if not available_groups:
                break
            group = max(
                available_groups,
                key=lambda name: (weights[name] / (quotas[name] + 1), name),
            )
            quotas[group] += 1
        return quotas
