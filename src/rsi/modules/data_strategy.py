"""Chapter-2 data strategy: fit the target distribution, select datasets, and review."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from rsi.core.errors import ConfigError
from rsi.core.interfaces import Component
from rsi.core.types import DefectReport, TaskSpec
from rsi.modules.datasets import (
    DatasetCatalog,
    DatasetSelection,
    DatasetSpec,
    dataset_available,
    select_datasets,
)


@dataclass
class TargetDistributionPlan:
    """Step 1 output: the task preset moved toward the missing abilities."""

    initial: dict[str, float]
    distribution: dict[str, float]
    deficits: dict[str, float]
    transfers: list[dict[str, Any]] = field(default_factory=list)
    source: str = "defects"
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "initial": dict(self.initial),
            "distribution": dict(self.distribution),
            "deficits": dict(self.deficits),
            "transfers": list(self.transfers),
            "notes": list(self.notes),
        }


@dataclass
class StrategyReview:
    """Step 3 verdict: whether the optimized mix is close enough to the target."""

    accepted: bool
    gap: float
    longtail_gap: float
    gap_tolerance: float
    longtail_gap_tolerance: float
    deficits: dict[str, float] = field(default_factory=dict)
    iterations: list[dict[str, Any]] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "gap": self.gap,
            "longtail_gap": self.longtail_gap,
            "gap_tolerance": self.gap_tolerance,
            "longtail_gap_tolerance": self.longtail_gap_tolerance,
            "deficits": dict(self.deficits),
            "iterations": list(self.iterations),
            "added": list(self.added),
            "reason": self.reason,
        }


@dataclass
class DatasetStrategy:
    """Chapter 2 strategy handed to step 4 (materialize) and step 5 (train)."""

    target: TargetDistributionPlan
    candidates: list[str]
    selection: DatasetSelection
    review: StrategyReview
    count: int
    task_id: str = ""
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "task": self.task_id,
            "count": self.count,
            "target": self.target.as_dict(),
            "candidates": list(self.candidates),
            "selected": [dataset.name for dataset in self.selection.datasets],
            "ratios": dict(self.selection.ratios),
            "actual_distribution": dict(
                self.selection.stats.get("actual_distribution", {})
            ),
            "uncovered": list(self.selection.uncovered),
            "gap": self.review.gap,
            "longtail_gap": self.review.longtail_gap,
            "accepted": self.review.accepted,
            "review": self.review.as_dict(),
            "notes": list(self.notes),
        }


# --------------------------------------------------------------------- step 1
def _normalized(values: dict[str, float]) -> dict[str, float]:
    clean: dict[str, float] = {}
    for name, value in values.items():
        number = float(value)
        if math.isfinite(number) and number > 0:
            clean[str(name)] = number
    total = sum(clean.values())
    return {name: value / total for name, value in clean.items()} if total > 0 else {}


def adjust_target_distribution(
    preset: dict[str, float],
    deficits: dict[str, float],
    *,
    step: float = 0.05,
    min_share: float = 0.01,
    max_shift: float = 0.35,
) -> TargetDistributionPlan:
    """Move a fixed share from the largest preset abilities to the deficits.

    The preset is sorted by share. Every deficit receives ``step`` of mass; the
    largest non-deficient ability donates first, then the next largest, and so
    on, while every donor keeps at least ``min_share``. ``max_shift`` caps the
    total mass moved, so a long defect list cannot erase the task profile.
    """
    if not math.isfinite(step) or step <= 0:
        raise ValueError("target adjustment step must be positive")
    if not math.isfinite(min_share) or not 0 <= min_share < 1:
        raise ValueError("target adjustment min_share must be within [0, 1)")
    if not math.isfinite(max_shift) or max_shift <= 0:
        raise ValueError("target adjustment max_shift must be positive")

    initial = _normalized(preset)
    deficit_values = {
        str(name): max(0.0, min(1.0, float(value)))
        for name, value in deficits.items()
        if math.isfinite(float(value)) and float(value) > 0
    }
    if not initial:
        return TargetDistributionPlan(
            initial={},
            distribution=_normalized(deficit_values),
            deficits=deficit_values,
            source="defects",
            notes=["no task preset; analyzer deficits define the target"],
        )

    shares = dict(initial)
    donors = [name for name in shares if name not in deficit_values]
    transfers: list[dict[str, Any]] = []
    notes: list[str] = []
    remaining_shift = max_shift
    for ability, _severity in sorted(deficit_values.items(), key=lambda item: (-item[1], item[0])):
        if remaining_shift <= 1e-9:
            notes.append("max_shift reached before every deficit was served")
            break
        need = min(step, remaining_shift)
        for donor in sorted(donors, key=lambda name: (-shares[name], name)):
            if need <= 1e-9:
                break
            available = shares[donor] - min_share
            if available <= 1e-9:
                continue
            amount = min(available, need)
            shares[donor] -= amount
            shares[ability] = shares.get(ability, 0.0) + amount
            need -= amount
            remaining_shift -= amount
            transfers.append({"from": donor, "to": ability, "amount": amount})
        if need > 1e-9:
            notes.append(f"no donor share left for {ability}")
    if not donors:
        notes.append("every preset ability is deficient; the preset is kept as is")

    return TargetDistributionPlan(
        initial=initial,
        distribution=_normalized(shares),
        deficits=deficit_values,
        transfers=transfers,
        source="task",
        notes=notes,
    )


def preset_from_datasets(datasets: list[DatasetSpec]) -> dict[str, float]:
    """Fallback preset: aggregate coverage of the task's preset datasets."""
    preset: dict[str, float] = {}
    for dataset in datasets:
        for ability, coverage in dataset.abilities.items():
            if coverage > 0:
                preset[ability] = preset.get(ability, 0.0) + coverage
    return preset


def allocate_quotas(shares: dict[str, float], total: int) -> dict[str, int]:
    """Largest-remainder integer split of ``total`` by ``shares``."""
    if total <= 0:
        return {}
    positive = {name: value for name, value in shares.items() if value > 0}
    weight = sum(positive.values())
    if weight <= 0:
        return {}
    exact = {name: value / weight * total for name, value in positive.items()}
    quotas = {name: int(value) for name, value in exact.items()}
    remainder = total - sum(quotas.values())
    order = sorted(positive, key=lambda name: (exact[name] - quotas[name], name), reverse=True)
    for name in order[: max(0, remainder)]:
        quotas[name] += 1
    return quotas


# --------------------------------------------------------------------- step 3
def strategy_gap(
    selection: DatasetSelection,
    target: dict[str, float],
    *,
    longtail_threshold: float = 0.1,
) -> dict[str, Any]:
    """Target mass still missing, split into ordinary and long-tail abilities."""
    shares = _normalized(target)
    actual = selection.stats.get("actual_distribution", {})
    deficits = {
        name: max(0.0, share - float(actual.get(name, 0.0)))
        for name, share in shares.items()
    }
    longtail = [name for name, share in shares.items() if share <= longtail_threshold]
    return {
        "gap": float(sum(deficits.values())),
        "longtail_gap": float(sum(deficits[name] for name in longtail)),
        "deficits": deficits,
        "longtail": longtail,
    }


def choose_fill_datasets(
    pool: list[DatasetSpec],
    used: list[DatasetSpec],
    target: dict[str, float],
    metrics: dict[str, Any],
    *,
    impact_weight: float = 1.0,
    limit: int = 2,
) -> list[DatasetSpec]:
    """Pick unused datasets that cover the deficits with the least collateral coverage.

    ``fill`` is the dataset's composition on the deficient abilities;
    ``impact`` is its composition on the abilities the mix already satisfies.
    Datasets are ranked by ``fill - impact_weight * impact``.
    """
    if limit <= 0:
        return []
    deficits = {
        name: float(value)
        for name, value in (metrics.get("deficits") or {}).items()
        if float(value) > 1e-9
    }
    if not deficits:
        return []
    total_deficit = sum(deficits.values())
    targets = _normalized(target)
    satisfied = {name: share for name, share in targets.items() if name not in deficits}
    satisfied_weight = sum(satisfied.values()) or 1.0
    used_names = {dataset.name for dataset in used}
    scored: list[tuple[float, float, DatasetSpec]] = []
    for dataset in pool:
        if dataset.name in used_names or not dataset_available(dataset):
            continue
        total_coverage = sum(max(0.0, value) for value in dataset.abilities.values())
        if total_coverage <= 0:
            continue
        composition = {
            name: max(0.0, value) / total_coverage
            for name, value in dataset.abilities.items()
            if value > 0
        }
        fill = sum(composition.get(name, 0.0) * value for name, value in deficits.items()) / total_deficit
        if fill <= 0:
            continue
        impact = sum(composition.get(name, 0.0) * share for name, share in satisfied.items()) / satisfied_weight
        scored.append((fill - impact_weight * impact, fill, dataset))
    scored.sort(key=lambda item: (-item[0], -item[1], item[2].name))
    return [dataset for _score, _fill, dataset in scored[:limit]]


def refine_strategy(
    candidates: list[DatasetSpec],
    target: dict[str, float],
    count: int,
    *,
    pool: list[DatasetSpec] | None = None,
    max_datasets: int | None = 4,
    diversity_weight: float = 0.1,
    diversity_threshold: float = 0.05,
    diversity_floor: float = 0.01,
    under_weight: float = 4.0,
    over_weight: float = 1.0,
    other_over_weight: float = 1.0,
    longtail_threshold: float = 0.1,
    longtail_under_weight: float = 8.0,
    readiness_weight: float = 0.01,
    gap_tolerance: float = 0.05,
    longtail_gap_tolerance: float = 0.01,
    max_rework: int = 2,
    add_per_round: int = 2,
    impact_weight: float = 1.0,
    longtail_tolerance: float = 1e-6,
) -> tuple[DatasetSelection, StrategyReview]:
    """Optimize, review, add covering datasets, and re-optimize (step 2 <-> step 3)."""
    if max_rework < 0:
        raise ValueError("max_rework must be non-negative")
    if add_per_round <= 0:
        raise ValueError("add_per_round must be positive")
    if gap_tolerance < 0 or longtail_gap_tolerance < 0:
        raise ValueError("review tolerances must be non-negative")
    pool = list(pool or candidates)
    current = list(candidates)
    added: list[str] = []
    iterations: list[dict[str, Any]] = []

    def optimize(datasets: list[DatasetSpec]) -> DatasetSelection:
        return select_datasets(
            datasets,
            target,
            count,
            max_datasets=max_datasets,
            diversity_weight=diversity_weight,
            diversity_threshold=diversity_threshold,
            diversity_floor=diversity_floor,
            under_weight=under_weight,
            over_weight=over_weight,
            other_over_weight=other_over_weight,
            longtail_threshold=longtail_threshold,
            longtail_under_weight=longtail_under_weight,
            dataset_cost_weight=readiness_weight,
            longtail_tolerance=longtail_tolerance,
        )

    selection = optimize(current)
    for attempt in range(max_rework + 1):
        metrics = strategy_gap(selection, target, longtail_threshold=longtail_threshold)
        accepted = (
            metrics["gap"] <= gap_tolerance
            and metrics["longtail_gap"] <= longtail_gap_tolerance
        )
        record = {
            "attempt": attempt,
            "candidates": [dataset.name for dataset in current],
            "selected": [dataset.name for dataset in selection.datasets],
            "ratios": dict(selection.ratios),
            "gap": metrics["gap"],
            "longtail_gap": metrics["longtail_gap"],
            "accepted": accepted,
        }
        iterations.append(record)
        if accepted or attempt >= max_rework:
            break
        additions = choose_fill_datasets(
            pool,
            current,
            target,
            metrics,
            impact_weight=impact_weight,
            limit=add_per_round,
        )
        if not additions:
            record["reason"] = "no unused dataset covers the remaining deficit"
            break
        added.extend(dataset.name for dataset in additions)
        current.extend(additions)
        selection = optimize(current)
    final = strategy_gap(selection, target, longtail_threshold=longtail_threshold)
    accepted = (
        final["gap"] <= gap_tolerance
        and final["longtail_gap"] <= longtail_gap_tolerance
    )
    reason = "within tolerance" if accepted else iterations[-1].get("reason", "")
    if not accepted and not reason:
        reason = "max_rework reached with the gap above tolerance"
    return selection, StrategyReview(
        accepted=accepted,
        gap=final["gap"],
        longtail_gap=final["longtail_gap"],
        gap_tolerance=gap_tolerance,
        longtail_gap_tolerance=longtail_gap_tolerance,
        deficits=dict(final["deficits"]),
        iterations=iterations,
        added=added,
        reason=reason,
    )


# --------------------------------------------------------------------- planner
class DataStrategyPlanner(Component):
    """Build the chapter-2 data strategy from task presets and chapter-1 defects."""

    def __init__(self, runtime: Any, cfg: Any, **params: Any) -> None:
        super().__init__(runtime, cfg, **params)
        self.enabled = cfg.get_bool("enabled", False)
        self.count = cfg.get_int("count", 0)
        self.default_count = int(params.get("default_count", 0) or 0)
        target_cfg = cfg.section("target", required=False)
        self.step = target_cfg.get_float("step", 0.05)
        self.min_share = target_cfg.get_float("min_share", 0.01)
        self.max_shift = target_cfg.get_float("max_shift", 0.35)
        self.pending_priority = target_cfg.get_float("pending_priority", 0.25)
        optimize_cfg = cfg.section("optimize", required=False)
        self.max_datasets = optimize_cfg.get_int("max_datasets", 4)
        self.diversity_weight = optimize_cfg.get_float("diversity_weight", 0.1)
        self.diversity_threshold = optimize_cfg.get_float("diversity_threshold", 0.05)
        self.diversity_floor = optimize_cfg.get_float("diversity_floor", 0.01)
        self.under_weight = optimize_cfg.get_float("under_weight", 4.0)
        self.over_weight = optimize_cfg.get_float("over_weight", 1.0)
        self.other_over_weight = optimize_cfg.get_float("other_over_weight", 1.0)
        self.longtail_threshold = optimize_cfg.get_float("longtail_threshold", 0.1)
        self.longtail_under_weight = optimize_cfg.get_float("longtail_under_weight", 8.0)
        self.readiness_weight = optimize_cfg.get_float("readiness_weight", 0.01)
        review_cfg = cfg.section("review", required=False)
        self.gap_tolerance = review_cfg.get_float("gap_tolerance", 0.05)
        self.longtail_gap_tolerance = review_cfg.get_float("longtail_gap_tolerance", 0.01)
        self.max_rework = review_cfg.get_int("max_rework", 2)
        self.add_per_round = review_cfg.get_int("add_per_round", 2)
        self.impact_weight = review_cfg.get_float("impact_weight", 1.0)
        self.catalog_path = str(cfg.get("catalog", "configs/datasets.yaml"))
        self.task = self._task(cfg.get_dict("task", {}))
        self._datasets: list[DatasetSpec] | None = None
        self._validate_options()

    # ------------------------------------------------------------------ setup
    def _validate_options(self) -> None:
        if self.count < 0:
            raise ConfigError("data.strategy.count must be non-negative")
        if self.max_datasets <= 0:
            raise ConfigError("data.strategy.optimize.max_datasets must be positive")
        if self.step <= 0:
            raise ConfigError("data.strategy.target.step must be positive")
        if not 0 <= self.min_share < 1:
            raise ConfigError("data.strategy.target.min_share must be within [0, 1)")
        if not 0 < self.max_shift <= 1:
            raise ConfigError("data.strategy.target.max_shift must be within (0, 1]")
        if not 0 <= self.pending_priority <= 1:
            raise ConfigError("data.strategy.target.pending_priority must be within [0, 1]")
        if self.diversity_weight < 0:
            raise ConfigError("data.strategy.optimize.diversity_weight must be non-negative")
        if not 0 <= self.diversity_threshold <= 1:
            raise ConfigError("data.strategy.optimize.diversity_threshold must be within [0, 1]")
        if not 0 <= self.diversity_floor <= 1:
            raise ConfigError("data.strategy.optimize.diversity_floor must be within [0, 1]")
        if self.max_rework < 0:
            raise ConfigError("data.strategy.review.max_rework must be non-negative")
        if self.add_per_round <= 0:
            raise ConfigError("data.strategy.review.add_per_round must be positive")
        if self.gap_tolerance < 0 or self.longtail_gap_tolerance < 0:
            raise ConfigError("data.strategy.review tolerances must be non-negative")

    def _task(self, raw: dict[str, Any]) -> TaskSpec | None:
        if not raw:
            return None
        abilities = raw.get("abilities") or raw.get("ability_priorities") or {}
        if not isinstance(abilities, dict):
            raise ConfigError("data.strategy.task.abilities must be a mapping")
        priorities: dict[str, float] = {}
        for name, value in abilities.items():
            number = float(value)
            if not 0 <= number <= 1:
                raise ConfigError(f"data.strategy.task.abilities.{name} must be within [0, 1]")
            if number > 0:
                priorities[str(name)] = number
        datasets = [str(value) for value in (raw.get("datasets") or [])]
        return TaskSpec(
            id=str(raw.get("id") or "task"),
            ability_priorities=priorities,
            count=int(raw.get("count", 0) or 0),
            notes=str(raw.get("notes") or ""),
            datasets=datasets,
        )

    def _catalog_datasets(self) -> list[DatasetSpec]:
        if self._datasets is None:
            path = self.runtime.path(self.catalog_path)
            self._datasets = DatasetCatalog(path, self.runtime.repo_root).datasets
        return self._datasets

    def validate(self) -> None:
        if not self.enabled:
            return
        catalog = self._catalog_datasets()
        if not catalog:
            raise ConfigError(f"data.strategy.catalog has no datasets: {self.catalog_path}")
        if self.task is None:
            return
        known = {dataset.name for dataset in catalog}
        unknown = [name for name in self.task.datasets if name not in known]
        if unknown:
            raise ConfigError(
                f"data.strategy.task.datasets names unknown datasets: {', '.join(unknown)}"
            )

    # ------------------------------------------------------------------- plan
    def plan(self, defects: DefectReport, context: Any) -> DatasetStrategy:
        catalog = self._catalog_datasets()
        previous = self._previous_selection(getattr(context, "history", []) or [])
        preset_names = list(self.task.datasets) if self.task else []
        names = previous or preset_names
        candidates = self._resolve(names, catalog)
        notes: list[str] = []
        if not candidates:
            candidates = list(catalog)
            notes.append(
                "no previous or preset dataset; using the whole registry"
                if names
                else "using the whole registry"
            )
        preset = dict(self.task.ability_priorities) if self.task else {}
        if not preset:
            preset = preset_from_datasets(candidates)
        deficits = self._deficits(defects)
        pending = sorted(
            {str(name) for name in getattr(getattr(context, "state", None), "pending_capabilities", None) or []}
            & set(deficits)
        )
        for capability in pending:
            deficits[capability] = max(deficits[capability], self.pending_priority)
        if pending:
            notes.append(
                f"pending capabilities keep at least {self.pending_priority}: {', '.join(pending)}"
            )
        target = adjust_target_distribution(
            preset,
            deficits,
            step=self.step,
            min_share=self.min_share,
            max_shift=self.max_shift,
        )
        target.notes.extend(notes)
        count = self._count()
        if not target.distribution or count <= 0:
            selection = select_datasets([], target.distribution, 0)
            review = StrategyReview(
                accepted=False,
                gap=float(sum(target.distribution.values())),
                longtail_gap=0.0,
                gap_tolerance=self.gap_tolerance,
                longtail_gap_tolerance=self.longtail_gap_tolerance,
                deficits=dict(target.distribution),
                reason="no target distribution" if not target.distribution else "empty data budget",
            )
        else:
            selection, review = refine_strategy(
                candidates,
                target.distribution,
                count,
                pool=catalog,
                max_datasets=self.max_datasets,
                diversity_weight=self.diversity_weight,
                diversity_threshold=self.diversity_threshold,
                diversity_floor=self.diversity_floor,
                under_weight=self.under_weight,
                over_weight=self.over_weight,
                other_over_weight=self.other_over_weight,
                longtail_threshold=self.longtail_threshold,
                longtail_under_weight=self.longtail_under_weight,
                readiness_weight=self.readiness_weight,
                gap_tolerance=self.gap_tolerance,
                longtail_gap_tolerance=self.longtail_gap_tolerance,
                max_rework=self.max_rework,
                add_per_round=self.add_per_round,
                impact_weight=self.impact_weight,
            )
        self.logger.info(
            "data strategy: target %d abilities, %d candidate datasets, %d selected, gap %.4f",
            len(target.distribution),
            len(candidates),
            len(selection.datasets),
            review.gap,
        )
        return DatasetStrategy(
            target=target,
            candidates=[dataset.name for dataset in candidates],
            selection=selection,
            review=review,
            count=count,
            task_id=self.task.id if self.task else "",
            notes=notes,
        )

    # --------------------------------------------------------------- helpers
    def _count(self) -> int:
        if self.count > 0:
            return self.count
        if self.task is not None and self.task.count > 0:
            return self.task.count
        return self.default_count

    def _deficits(self, defects: DefectReport) -> dict[str, float]:
        merged = defects.metadata.get("ability_priorities") or {}
        output: dict[str, float] = {}
        for defect in defects.defects:
            raw = merged.get(defect.capability)
            try:
                value = max(0.0, min(1.0, float(raw))) if raw is not None else defect.severity * defect.confidence
            except (TypeError, ValueError):
                value = defect.severity * defect.confidence
            output[defect.capability] = max(output.get(defect.capability, 0.0), value)
        return output

    def _resolve(self, names: list[str], catalog: list[DatasetSpec]) -> list[DatasetSpec]:
        by_name = {dataset.name: dataset for dataset in catalog}
        missing = [name for name in names if name not in by_name]
        if missing:
            self.logger.warning("unknown candidate datasets ignored: %s", ", ".join(missing))
        resolved: list[DatasetSpec] = []
        seen: set[str] = set()
        for name in names:
            dataset = by_name.get(name)
            if dataset is None or dataset.name in seen:
                continue
            seen.add(dataset.name)
            resolved.append(dataset)
        return resolved

    @staticmethod
    def _previous_selection(history: list[dict[str, Any]]) -> list[str]:
        for record in reversed(history):
            selection = (record.get("synthesize") or {}).get("dataset_selection") or {}
            names = selection.get("selected") if isinstance(selection, dict) else None
            if names:
                return [str(name) for name in names]
        return []
