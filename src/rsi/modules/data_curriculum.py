from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rsi.core.interfaces import Component
from rsi.core.types import Defect, DefectReport


@dataclass
class CurriculumDecision:
    defects: list[Defect]
    stage: str = ""
    eligible: list[str] = field(default_factory=list)
    excluded: list[dict[str, Any]] = field(default_factory=list)
    notes: str = ""


class CurriculumPolicy(Component):
    """Decide which capabilities to work on this round.

    Curriculum is deliberately separate from data generation: the planner asks
    for an ordered subset of defects, and the same policy can be swapped without
    touching collectors, filters or the loop.
    """

    def select(self, defects: list[Defect], context: Any) -> CurriculumDecision:
        raise NotImplementedError


class _Base(CurriculumPolicy):
    def __init__(self, runtime: Any, cfg: Any, **params: Any) -> None:
        super().__init__(runtime, cfg, **params)
        self.max_capabilities = self.cfg.get_int("max_capabilities", 4)

    def priority(self, defect: Defect) -> float:
        return defect.severity * defect.confidence * (1 + self.runtime.catalog.priority(defect.capability))

    def _limit(self, defects: list[Defect], stage: str, notes: str, excluded: list[dict[str, Any]] | None = None) -> CurriculumDecision:
        ordered = sorted(defects, key=self.priority, reverse=True)
        selected: list[Defect] = []
        seen: set[str] = set()
        for defect in ordered:
            if defect.capability in seen:
                continue
            seen.add(defect.capability)
            selected.append(defect)
            if len(selected) >= self.max_capabilities:
                break
        return CurriculumDecision(
            defects=selected,
            stage=stage,
            eligible=sorted(seen),
            excluded=excluded or [],
            notes=notes,
        )


class SeverityCurriculum(_Base):
    """Default: rank by severity, confidence and catalog priority."""

    def select(self, defects: list[Defect], context: Any) -> CurriculumDecision:
        return self._limit(defects, "severity", "ordered by severity, confidence and catalog priority")


class FixedOrderCurriculum(_Base):
    """Follow a fixed capability order, unlocking one prefix per round."""

    def __init__(self, runtime: Any, cfg: Any, **params: Any) -> None:
        super().__init__(runtime, cfg, **params)
        self.order = [str(v) for v in self.cfg.get_list("order", [])]
        if not self.order:
            raise ValueError("fixed curriculum requires a non-empty 'order'")
        self.unlock_per_round = self.cfg.get_int("unlock_per_round", 1)

    def select(self, defects: list[Defect], context: Any) -> CurriculumDecision:
        stage = min(len(self.order), 1 + (context.round // max(1, self.unlock_per_round)))
        eligible = set(self.order[:stage])
        chosen = [d for d in defects if d.capability in eligible]
        excluded = [
            {"capability": d.capability, "reason": "locked"}
            for d in defects
            if d.capability not in eligible
        ]
        return self._limit(
            chosen,
            f"stage-{stage}",
            f"eligible={sorted(eligible)}",
            excluded=excluded,
        )


class PrerequisiteCurriculum(_Base):
    """Only work on a capability once its prerequisites clear a score floor."""

    def __init__(self, runtime: Any, cfg: Any, **params: Any) -> None:
        super().__init__(runtime, cfg, **params)
        self.prerequisites = {
            str(k): [str(v) for v in values]
            for k, values in self.cfg.get_dict("prerequisites", {}).items()
            if isinstance(values, list)
        }
        self.score_floor = self.cfg.get_float("score_floor", 0.6)
        self.include_unmet = self.cfg.get_bool("include_unmet", False)

    def select(self, defects: list[Defect], context: Any) -> CurriculumDecision:
        scores = {}
        if context.collect_result is not None:
            scores = {
                name: score.score for name, score in context.collect_result.capabilities.items()
            }
        chosen: list[Defect] = []
        excluded: list[dict[str, Any]] = []
        for defect in defects:
            prereqs = self.prerequisites.get(defect.capability, [])
            unmet = [p for p in prereqs if scores.get(p, 0.0) < self.score_floor]
            if unmet and not self.include_unmet:
                excluded.append(
                    {
                        "capability": defect.capability,
                        "reason": "prerequisites unmet",
                        "unmet": unmet,
                        "scores": {p: scores.get(p) for p in unmet},
                    }
                )
                continue
            chosen.append(defect)
        notes = f"score_floor={self.score_floor}"
        if excluded:
            notes += f"; excluded {len(excluded)} locked capability(ies)"
        return self._limit(chosen, "prerequisite", notes, excluded=excluded)


from rsi.core.registry import register  # noqa: E402  (keeps base classes readable above)

register("curriculum", "severity")(SeverityCurriculum)
register("curriculum", "fixed")(FixedOrderCurriculum)
register("curriculum", "prerequisite")(PrerequisiteCurriculum)
