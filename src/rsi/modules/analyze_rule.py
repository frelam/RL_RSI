from __future__ import annotations

import math
from typing import Any

from rsi.core.interfaces import AnalyzerBackend
from rsi.core.jsonio import stable_hash
from rsi.core.registry import register
from rsi.core.types import CapabilityScore, CollectResult, Defect, DefectReport, ItemResult
from rsi.modules.metrics import resolve_capability


@register("analyzer", "rule")
class RuleCapabilityAnalyzer(AnalyzerBackend):
    """Capability-deficit analysis driven by benchmark item scores.

    The README's algorithm is implemented literally:
      1. capabilities are the atomic vocabulary (catalog),
      2. each benchmark item maps to one capability,
      3. items below the baseline mark that capability negative,
      4. corroboration across benchmarks is combined with a noisy-OR bonus.
    """

    def __init__(self, runtime: Any, cfg: Any, **params: Any) -> None:
        super().__init__(runtime, cfg, **params)
        self.baseline_mode = str(self.cfg.get("baseline", "global"))
        self.absolute_threshold = self.cfg.get_float("absolute_threshold", 0.55)
        self.margin = self.cfg.get_float("margin", 0.08)
        self.min_gap = self.cfg.get_float("min_gap", 0.02)
        self.min_items = self.cfg.get_int("min_items", 2)
        self.min_severity = self.cfg.get_float("min_severity", 0.05)
        self.priority_weight = self.cfg.get_float("priority_weight", 0.5)
        self.include_subcapabilities = self.cfg.get_bool("include_subcapabilities", False)
        self.evidence_limit = self.cfg.get_int("evidence_limit", 3)

    def analyze(
        self,
        collect: CollectResult,
        round_no: int,
        baseline: dict[str, float],
    ) -> DefectReport:
        grouped: dict[str, list[tuple[ItemResult, float, str]]] = {}
        for report in collect.reports:
            for item in report.items:
                capability = resolve_capability(item, report.benchmark, self.runtime.catalog)
                grouped.setdefault(capability, []).append((item, item.score, report.benchmark))

        overall = collect.aggregate.get("mean_score", 0.0)
        threshold = self._threshold(overall)
        defects: list[Defect] = []
        for capability, rows in grouped.items():
            if len(rows) < self.min_items:
                continue
            score = sum(r[1] for r in rows) / len(rows)
            base = self._baseline_for(capability, baseline, threshold)
            gap = base - score
            if gap < self.min_gap:
                continue
            severity = min(
                0.98,
                gap * 2.5 * (1.0 + self.priority_weight * self.runtime.catalog.priority(capability)),
            )
            if severity < self.min_severity:
                continue
            confidence = min(0.95, 0.35 + 0.1 * len(rows))
            evidence = [
                f"{benchmark}:{item.item_id} score={score:.2f}"
                for item, score, benchmark in sorted(rows, key=lambda r: r[1])[: self.evidence_limit]
            ]
            info = self.runtime.catalog.get(capability)
            defects.append(
                Defect(
                    id=stable_hash("rule", round_no, capability, ""),
                    capability=capability,
                    subcapability="",
                    description=(
                        f"{capability} scores {score:.2f}, below baseline {base:.2f} "
                        f"(gap {gap:.2f}, {len(rows)} items)"
                    ),
                    severity=severity,
                    confidence=confidence,
                    source="rule",
                    evidence=evidence,
                    suggested_data={
                        "kind": info.get("kind", "text"),
                        "seeds": info.get("seeds", []),
                        "verifier": info.get("verifier", {}),
                        "target_difficulty": self.cfg.get_list("target_difficulty", [0.25, 0.8]),
                    },
                    round=round_no,
                )
            )

        if self.include_subcapabilities:
            defects.extend(self._subcapability_defects(collect, round_no, baseline, threshold))

        defects.sort(key=lambda d: d.severity, reverse=True)
        capabilities = {
            capability: CapabilityScore(
                capability=capability,
                score=sum(r[1] for r in rows) / len(rows),
                baseline=baseline.get(capability),
                delta=(sum(r[1] for r in rows) / len(rows) - baseline[capability])
                if capability in baseline
                else None,
                num_items=len(rows),
            )
            for capability, rows in grouped.items()
        }
        summary = (
            f"rule analysis found {len(defects)} defect(s); "
            f"threshold={threshold:.2f}, overall={overall:.2f}"
        )
        return DefectReport(
            round=round_no,
            checkpoint_id=collect.checkpoint_id,
            defects=defects,
            capabilities=capabilities,
            summary=summary,
            sources=["rule"],
            metadata={"baseline_mode": self.baseline_mode, "threshold": threshold},
        )

    def _threshold(self, overall: float) -> float:
        if self.baseline_mode == "absolute":
            return self.absolute_threshold
        return max(0.0, overall - self.margin)

    def _baseline_for(self, capability: str, baseline: dict[str, float], threshold: float) -> float:
        if self.baseline_mode == "previous" and capability in baseline:
            return max(threshold, baseline[capability] - self.margin)
        return threshold

    def _subcapability_defects(
        self,
        collect: CollectResult,
        round_no: int,
        baseline: dict[str, float],
        threshold: float,
    ) -> list[Defect]:
        grouped: dict[tuple[str, str], list[ItemResult]] = {}
        for report in collect.reports:
            for item in report.items:
                if not item.subcapability:
                    continue
                capability = resolve_capability(item, report.benchmark, self.runtime.catalog)
                grouped.setdefault((capability, item.subcapability), []).append(item)
        defects: list[Defect] = []
        for (capability, sub), items in grouped.items():
            if len(items) < self.min_items:
                continue
            score = sum(item.score for item in items) / len(items)
            gap = threshold - score
            if gap < self.min_gap:
                continue
            key = f"{capability}.{sub}"
            defects.append(
                Defect(
                    id=stable_hash("rule-sub", round_no, key),
                    capability=capability,
                    subcapability=sub,
                    description=f"{key} scores {score:.2f}, below {threshold:.2f}",
                    severity=min(0.98, gap * 2.0),
                    confidence=min(0.9, 0.3 + 0.1 * len(items)),
                    source="rule",
                    evidence=[f"{item.item_id} score={item.score:.2f}" for item in sorted(items, key=lambda i: i.score)[:3]],
                    round=round_no,
                )
            )
        return defects
