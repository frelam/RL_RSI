from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from rsi.core.errors import ConfigError
from rsi.core.interfaces import AnalyzerBackend
from rsi.core.jsonio import stable_hash
from rsi.core.registry import register
from rsi.core.types import CollectResult, Defect, DefectReport
from rsi.hubs.catalog import BenchmarkCatalog

Curve = list[tuple[float, float, float]]  # (params_b, mean, std)


def reference_at(curve: Curve, params_b: float) -> tuple[float, float]:
    """Linear interpolation of a reference curve, clamped at both ends."""
    if params_b <= curve[0][0]:
        mean, std = curve[0][1], curve[0][2]
    elif params_b >= curve[-1][0]:
        mean, std = curve[-1][1], curve[-1][2]
    else:
        for (x0, y0, s0), (x1, y1, s1) in zip(curve, curve[1:]):
            if params_b <= x1:
                ratio = 0.0 if x1 == x0 else (params_b - x0) / (x1 - x0)
                mean, std = y0 + (y1 - y0) * ratio, s0 + (s1 - s0) * ratio
                break
    return mean, std


def load_reference(path: Path) -> dict[str, Curve]:
    """Load ``benchmarks: {name[.subset]: {params_b: score | [mean, std]}}`` curves."""
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    entries = raw.get("benchmarks") if isinstance(raw, dict) else None
    if not isinstance(entries, dict):
        raise ConfigError(f"Benchmark reference must contain a 'benchmarks' mapping: {path}")
    curves: dict[str, Curve] = {}
    for name, points in entries.items():
        if not isinstance(points, dict) or not points:
            continue
        try:
            curves[str(name)] = sorted(
                (float(size), float(values[0]), float(values[1]) if len(values) > 1 else 0.0)
                for size, values in (
                    (size, value if isinstance(value, list) else [value]) for size, value in points.items()
                )
            )
        except (TypeError, ValueError, IndexError) as exc:
            raise ConfigError(
                f"Reference curve for {name!r} must map params_b to a score or [mean, std]: {path}"
            ) from exc
    return curves


def subset_scores(report: Any) -> dict[str, float]:
    """Mean item score per subset (``ItemResult.subcapability``), empty if unlabeled."""
    grouped: dict[str, list[float]] = {}
    for item in report.items:
        if item.subcapability:
            grouped.setdefault(item.subcapability, []).append(float(item.score))
    return {subset: sum(values) / len(values) for subset, values in grouped.items()}


@register("analyzer", "missing_capability")
class MissingCapabilityAnalyzer(AnalyzerBackend):
    """Infer missing capabilities from well-performing benchmarks.

    Algorithm:
      1. A benchmark "performs well" when its score is not much lower than the
         reference score (mean over open models) of the same parameter count.
         Subsets are judged one by one when the report labels items with a
         subset; otherwise the benchmark is judged as a whole.
      2. Every capability a well-performing benchmark or subset covers is marked.
      3. Every catalog capability that stayed unmarked is reported as missing.

    Config::

        type: missing_capability
        registry: configs/benchmarks.yaml
        reference: configs/benchmark_reference.yaml   # empty => no reference curves
        params_b: 7                                   # default: checkpoint.initial.params_b
        margin: 0.05                                  # "not much lower" tolerance
        sigma: 0.0                                    # extra tolerance: sigma * reference std
        unreferenced: skip                            # skip | absolute (use absolute_threshold)
    """

    def __init__(self, runtime: Any, cfg: Any, **params: Any) -> None:
        super().__init__(runtime, cfg, **params)
        self.registry = BenchmarkCatalog(
            self.runtime.path(self.cfg.get("registry", "configs/benchmarks.yaml"))
        )
        self.margin = self.cfg.get_float("margin", 0.05)
        self.sigma = self.cfg.get_float("sigma", 0.0)
        self.unreferenced = str(self.cfg.get("unreferenced", "skip"))
        if self.unreferenced not in {"skip", "absolute"}:
            raise ConfigError(f"unreferenced must be 'skip' or 'absolute', got {self.unreferenced!r}")
        self.absolute_threshold = self.cfg.get_float("absolute_threshold", 0.5)
        self.severity = self.cfg.get_float("severity", 0.6)
        self.confidence = self.cfg.get_float("confidence", 0.5)
        self.params_b = self.cfg.get("params_b", runtime.config.get("checkpoint.initial.params_b", None))
        self.reference: dict[str, Curve] = {}
        reference = self.cfg.get("reference", "configs/benchmark_reference.yaml")
        if reference:
            if self.params_b is None:
                raise ConfigError(
                    "missing_capability analyzer requires params_b or checkpoint.initial.params_b "
                    "to read the reference curves"
                )
            self.params_b = float(self.params_b)
            self.reference = load_reference(self.runtime.path(str(reference)))

    def _benchmark_score(self, report: Any) -> float:
        metric = report.primary_metric or "mean_score"
        return float(report.aggregates.get(metric, report.aggregates.get("mean_score", 0.0)))

    def _reference_for(self, key: str) -> tuple[float, float] | None:
        curve = self.reference.get(key)
        if curve is not None:
            return reference_at(curve, self.params_b)
        return None if self.unreferenced == "skip" else (self.absolute_threshold, 0.0)

    def analyze(
        self,
        collect: CollectResult,
        round_no: int,
        baseline: dict[str, float],
    ) -> DefectReport:
        marked: dict[str, list[str]] = {}
        weak: dict[str, list[str]] = {}
        judged: dict[str, dict[str, Any]] = {}
        skipped: list[str] = []
        for report in collect.reports:
            benchmark = report.benchmark
            observed = report.capability_coverage or sorted({item.capability for item in report.items})
            scores = subset_scores(report) or {None: self._benchmark_score(report)}
            for subset, score in scores.items():
                key = f"{benchmark}.{subset}" if subset else benchmark
                reference = self._reference_for(key)
                if reference is None:
                    skipped.append(key)
                    continue
                capabilities = self.registry.capabilities(benchmark, subset) or (
                    [c for c in observed if c and c != "general"] if subset is None else []
                )
                tolerance = max(self.margin, self.sigma * reference[1])
                good = score >= reference[0] - tolerance
                judged[key] = {"score": score, "reference": reference[0], "good": good}
                for capability in capabilities:
                    (marked if good else weak).setdefault(capability, []).append(key)
        if skipped:
            self.logger.warning("missing_capability: no reference score for %s; skipped", ", ".join(skipped))

        defects: list[Defect] = []
        if judged:
            for capability in sorted(entry["name"] for entry in self.runtime.catalog.all()):
                if capability in marked:
                    continue
                defects.append(self._defect(capability, weak.get(capability, []), judged, round_no))
            defects.sort(key=lambda defect: (defect.severity, defect.capability), reverse=True)
        marked_names = sorted(marked)
        missing = [defect.capability for defect in defects]
        num_good = sum(1 for row in judged.values() if row["good"])
        return DefectReport(
            round=round_no,
            checkpoint_id=collect.checkpoint_id,
            defects=defects,
            capabilities=collect.capabilities,
            summary=(
                f"{len(marked_names)} of {len(marked_names) + len(missing)} capabilities marked by "
                f"{num_good}/{len(judged)} well-performing benchmark(s)/subset(s); {len(missing)} missing"
                if judged
                else "missing_capability skipped: no benchmark could be judged"
            ),
            sources=["missing_capability"],
            metadata={
                "params_b": self.params_b,
                "benchmarks": judged,
                "skipped": skipped,
                "marked_capabilities": marked_names,
                "missing_capabilities": missing,
                "ability_priorities": {
                    defect.capability: min(1.0, self.runtime.catalog.priority(defect.capability) * defect.severity)
                    for defect in defects
                },
            },
        )

    def _defect(
        self,
        capability: str,
        covered_by: list[str],
        judged: dict[str, dict[str, Any]],
        round_no: int,
    ) -> Defect:
        info = self.runtime.catalog.get(capability)
        if covered_by:
            evidence = [
                f"{key}: score {judged[key]['score']:.2f} < reference "
                f"{judged[key]['reference']:.2f} (not well-performing)"
                for key in covered_by[:3]
            ]
            reason = "only covered by benchmarks or subsets that do not perform well"
        else:
            evidence = [f"no judged benchmark or subset covers {capability}"]
            reason = "not covered by any judged benchmark or subset"
        return Defect(
            id=stable_hash("missing", round_no, capability),
            capability=capability,
            description=f"{capability} is missing: {reason}",
            severity=min(0.98, self.severity * self.runtime.catalog.priority(capability)),
            confidence=self.confidence,
            source="missing_capability",
            evidence=evidence,
            suggested_data={
                "kind": info.get("kind", "text"),
                "seeds": info.get("seeds", []),
                "verifier": info.get("verifier", {}),
                "target_difficulty": self.cfg.get_list("target_difficulty", [0.2, 0.9]),
            },
            round=round_no,
        )
