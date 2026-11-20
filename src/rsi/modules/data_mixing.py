from __future__ import annotations

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
                index = (record.get("data") or {}).get("mixing_candidate")
                delta = (record.get("validation") or {}).get("delta")
                if index is None or delta is None:
                    continue
                scores.setdefault(int(index), []).append(float(delta))
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
        self.enforce_band = cfg.get_bool("enforce_band", True)
        self.shuffle = cfg.get_bool("shuffle", True)
        self.last_stats: dict[str, Any] = {}

    def select(
        self, samples: list[DataSample], weights: dict[str, float], context: Any
    ) -> tuple[list[DataSample], dict[str, Any]]:
        low, high = (self.band + [0.0, 1.0])[:2]
        banded: list[DataSample] = []
        band_dropped = 0
        for sample in samples:
            if self.enforce_band and not (low <= sample.difficulty <= high):
                band_dropped += 1
                continue
            banded.append(sample)
        groups: dict[str, list[DataSample]] = {}
        for sample in banded:
            groups.setdefault(group_key(sample), []).append(sample)
        rng_seed = stable_hash(self.runtime.seed, context.round, "sampler")
        for group in groups:
            groups[group].sort(key=lambda s: stable_hash(rng_seed, group, s.id))

        if self.target_total <= 0:
            selected = list(banded)
        else:
            total_weight = sum(max(0.0, weights.get(group, 0.0)) for group in groups) or 1.0
            quotas: dict[str, int] = {}
            for group in groups:
                share = max(0.0, weights.get(group, 0.0)) / total_weight
                quota = int(round(share * self.target_total))
                quota = max(self.min_per_group, quota) if groups[group] else quota
                if self.max_per_group > 0:
                    quota = min(quota, self.max_per_group)
                quotas[group] = min(quota, len(groups[group]))
            selected = []
            for group, quota in quotas.items():
                selected.extend(groups[group][:quota])
            if len(selected) < self.target_total:
                remaining = [
                    sample
                    for group in groups
                    for sample in groups[group][quotas.get(group, 0) :]
                ]
                remaining.sort(key=lambda s: stable_hash(rng_seed, "fill", s.id))
                selected.extend(remaining[: self.target_total - len(selected)])
        if self.shuffle:
            selected.sort(key=lambda s: stable_hash(rng_seed, "final", s.id))
        counts: dict[str, int] = {}
        for sample in selected:
            counts[group_key(sample)] = counts.get(group_key(sample), 0) + 1
        stats = {
            "input": len(samples),
            "band_dropped": band_dropped,
            "selected": len(selected),
            "groups": counts,
            "target_total": self.target_total,
            "band": [low, high],
        }
        self.last_stats = stats
        return selected, stats
