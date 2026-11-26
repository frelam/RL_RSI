from __future__ import annotations

from typing import Any

from rsi.core.jsonio import stable_hash
from rsi.core.types import CapabilityScore, Defect, DefectReport


class DefectMerger:
    """Combine rule / LLM-judge / harness findings without double counting.

    Severity is the strongest weighted signal plus a corroboration bonus;
    confidence uses noisy-OR. This keeps a single loud judge from dominating,
    while agreeing sources raise confidence.
    """

    def __init__(self, cfg: Any) -> None:
        self.weights: dict[str, float] = {
            str(k): float(v) for k, v in cfg.get_dict("weights", {}).items()
        }
        self.default_weight = cfg.get_float("default_weight", 1.0)
        self.min_severity = cfg.get_float("min_severity", 0.05)
        self.min_confidence = cfg.get_float("min_confidence", 0.2)
        self.max_defects = cfg.get_int("max_defects", 32)
        self.rule_profile_weight = cfg.get_float("rule_profile_weight", 0.6)
        self.agent_profile_weight = cfg.get_float("agent_profile_weight", 0.4)

    def merge(
        self,
        reports: list[DefectReport],
        *,
        round_no: int,
        checkpoint_id: str,
        capabilities: dict[str, CapabilityScore],
    ) -> DefectReport:
        buckets: dict[tuple[str, str], list[Defect]] = {}
        for report in reports:
            for defect in report.defects:
                buckets.setdefault(defect.key, []).append(defect)

        merged: list[Defect] = []
        for (capability, subcapability), items in buckets.items():
            sources = sorted({item.source for item in items})
            weighted = [
                item.severity * self.weights.get(item.source, self.default_weight) for item in items
            ]
            severity = max(weighted) if weighted else 0.0
            severity = min(0.99, severity + 0.08 * (len(sources) - 1))
            confidence = 1.0
            for item in items:
                weight = self.weights.get(item.source, self.default_weight)
                confidence *= 1.0 - min(0.98, item.confidence * weight)
            confidence = min(0.99, 1.0 - confidence)
            if severity < self.min_severity or confidence < self.min_confidence:
                continue
            descriptions = list(dict.fromkeys(item.description for item in items if item.description))
            evidence: list[str] = []
            suggested: dict[str, Any] = {}
            for item in items:
                evidence.extend(item.evidence)
                suggested.update(item.suggested_data)
            merged.append(
                Defect(
                    id=stable_hash("merged", round_no, capability, subcapability),
                    capability=capability,
                    subcapability=subcapability,
                    description=" | ".join(descriptions)[:1000],
                    severity=severity,
                    confidence=confidence,
                    source="+".join(sources),
                    evidence=list(dict.fromkeys(evidence))[:12],
                    suggested_data=suggested,
                    round=round_no,
                )
            )
        merged.sort(key=lambda defect: (defect.severity, defect.confidence), reverse=True)
        merged = merged[: self.max_defects]
        sources = sorted({source for report in reports for source in report.sources})
        profile_values: dict[str, list[tuple[float, float]]] = {}
        for report in reports:
            profile = report.metadata.get("ability_priorities") or {}
            if not isinstance(profile, dict):
                continue
            source_weight = (
                self.rule_profile_weight if "rule" in report.sources
                else self.agent_profile_weight if "trajectory_agent" in report.sources
                else self.default_weight
            )
            for capability, raw_priority in profile.items():
                try:
                    priority = max(0.0, min(1.0, float(raw_priority)))
                except (TypeError, ValueError):
                    continue
                profile_values.setdefault(str(capability), []).append((priority, source_weight))
        ability_priorities = {
            capability: sum(value * weight for value, weight in values) / sum(weight for _, weight in values)
            for capability, values in profile_values.items()
            if sum(weight for _, weight in values) > 0
        }
        summary = (
            f"merged {len(merged)} defect(s) across {len({d.capability for d in merged})} "
            f"capability(ies) from {', '.join(sources) or 'no source'}"
        )
        return DefectReport(
            round=round_no,
            checkpoint_id=checkpoint_id,
            defects=merged,
            capabilities=capabilities,
            summary=summary,
            sources=sources,
            metadata={
                "input_defects": sum(len(report.defects) for report in reports),
                "per_source": {
                    source: sum(1 for report in reports if source in report.sources) for source in sources
                },
                "ability_priorities": ability_priorities,
            },
        )
