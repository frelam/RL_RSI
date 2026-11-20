from __future__ import annotations

import math
from typing import Any

from rsi.core.interfaces import AnalyzerBackend
from rsi.core.jsonio import stable_hash
from rsi.core.registry import register
from rsi.core.schema import validate_json
from rsi.core.types import (
    CollectResult,
    Defect,
    DefectReport,
    HarnessTask,
    LLMRequest,
)
from rsi.modules.metrics import worst_items


def _clamp(value: Any, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _defects_from_payload(
    payload: dict[str, Any] | None,
    *,
    source: str,
    round_no: int,
    catalog: Any,
    defaults: dict[str, Any] | None = None,
) -> list[Defect]:
    if not payload:
        return []
    raw_defects = payload.get("defects") or payload.get("findings") or []
    if not isinstance(raw_defects, list):
        return []
    defects: list[Defect] = []
    for index, raw in enumerate(raw_defects):
        if not isinstance(raw, dict):
            continue
        capability = str(raw.get("capability") or "").strip()
        if not capability:
            continue
        subcapability = str(raw.get("subcapability") or "")
        evidence = raw.get("evidence") or []
        if isinstance(evidence, str):
            evidence = [evidence]
        evidence = [str(item) for item in evidence if item is not None][:10]
        suggested = raw.get("suggested_data") if isinstance(raw.get("suggested_data"), dict) else {}
        merged_suggested = dict(defaults or {})
        merged_suggested.update(suggested)
        if "kind" not in merged_suggested:
            merged_suggested["kind"] = catalog.kind(capability)
        defects.append(
            Defect(
                id=stable_hash(source, round_no, capability, subcapability, index),
                capability=capability,
                subcapability=subcapability,
                description=str(raw.get("description") or raw.get("reason") or capability),
                severity=_clamp(raw.get("severity"), 0.5),
                confidence=_clamp(raw.get("confidence"), 0.5),
                source=source,
                evidence=evidence,
                suggested_data=merged_suggested,
                round=round_no,
            )
        )
    return defects


@register("analyzer", "llm_judge")
class LLMJudgeAnalyzer(AnalyzerBackend):
    """Analyze worst cases with one or more LLM judges.

    Voting supports multi-model judge, self-consistency and majority gating.
    The prompt and its JSON schema live in the Prompt Hub, so rubrics are
    versioned assets rather than code.
    """

    def __init__(self, runtime: Any, cfg: Any, **params: Any) -> None:
        super().__init__(runtime, cfg, **params)
        self.prompt_id = str(self.cfg.require("prompt").get("prompt"))
        profiles = self.cfg.get_list("profiles", [])
        if not profiles:
            profiles = [str(self.cfg.get("profile", runtime.config.get("llm.default_profile", "judge")))]
        self.profiles = [str(p) for p in profiles]
        self.votes = max(1, self.cfg.get_int("votes", 1))
        self.vote_policy = str(self.cfg.get("vote_policy", "majority"))
        self.min_vote_ratio = self.cfg.get_float("min_vote_ratio", 0.5)
        self.top_n = self.cfg.get_int("top_n", 8)
        self.max_score = self.cfg.get_float("max_score", 0.75)
        self.temperature = self.cfg.get_float("temperature", 0.0)
        self.max_tokens = self.cfg.get_int("max_tokens", 2048)
        self.min_confidence = self.cfg.get_float("min_confidence", 0.25)

    def analyze(
        self,
        collect: CollectResult,
        round_no: int,
        baseline: dict[str, float],
    ) -> DefectReport:
        cases = worst_items(collect.reports, self.runtime.catalog, self.top_n, self.max_score)
        capabilities = [
            {
                "capability": score.capability,
                "score": round(score.score, 4),
                "baseline": score.baseline,
                "delta": score.delta,
                "num_items": score.num_items,
            }
            for score in collect.capabilities.values()
        ]
        if not cases:
            return DefectReport(
                round=round_no,
                checkpoint_id=collect.checkpoint_id,
                capabilities=collect.capabilities,
                summary="llm judge skipped: no low-scoring cases",
                sources=["llm_judge"],
                metadata={"calls": 0},
            )
        rendered = self.runtime.prompt_hub.render(
            self.prompt_id,
            {
                "round": round_no,
                "capability_table": capabilities,
                "worst_cases": cases,
                "baseline": baseline,
                "num_cases": len(cases),
            },
        )
        votes: list[list[Defect]] = []
        errors: list[str] = []
        for vote in range(self.votes):
            profile = self.profiles[vote % len(self.profiles)]
            request = LLMRequest(
                prompt_id=rendered.key,
                prompt=rendered.text,
                system=rendered.system,
                schema=rendered.schema,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                metadata={
                    "round": round_no,
                    "vote": vote,
                    "profile": profile,
                    "checkpoint_id": collect.checkpoint_id,
                },
            )
            try:
                response = self.runtime.llm(profile).complete(request)
            except Exception as exc:  # noqa: BLE001 - judge failure is not fatal
                errors.append(f"{profile}: {exc}")
                self.logger.warning("llm judge %s failed: %s", profile, exc)
                continue
            payload = next((item for item in response.parsed if item is not None), None)
            if payload is None:
                errors.append(f"{profile}: response was not JSON")
                continue
            schema_errors = validate_json(payload, rendered.schema) if rendered.schema else []
            if schema_errors:
                errors.append(f"{profile}: {schema_errors[0]}")
                continue
            votes.append(
                _defects_from_payload(
                    payload,
                    source="llm_judge",
                    round_no=round_no,
                    catalog=self.runtime.catalog,
                )
            )
        defects = self._aggregate_votes(votes)
        defects = [d for d in defects if d.confidence >= self.min_confidence]
        return DefectReport(
            round=round_no,
            checkpoint_id=collect.checkpoint_id,
            defects=defects,
            capabilities=collect.capabilities,
            summary=f"llm judge found {len(defects)} defect(s) from {len(votes)}/{self.votes} valid votes",
            sources=["llm_judge"],
            metadata={
                "prompt": rendered.key,
                "profiles": self.profiles,
                "votes": self.votes,
                "valid_votes": len(votes),
                "errors": errors,
            },
        )

    def _aggregate_votes(self, votes: list[list[Defect]]) -> list[Defect]:
        if not votes:
            return []
        buckets: dict[tuple[str, str], list[Defect]] = {}
        for vote in votes:
            seen: set[tuple[str, str]] = set()
            for defect in vote:
                if defect.key in seen:
                    continue
                seen.add(defect.key)
                buckets.setdefault(defect.key, []).append(defect)
        threshold = 1
        if self.vote_policy == "majority":
            threshold = max(1, math.ceil(len(votes) * self.min_vote_ratio))
        merged: list[Defect] = []
        for (capability, subcapability), items in buckets.items():
            if len(items) < threshold:
                continue
            severity = sum(item.severity for item in items) / len(items)
            confidence = sum(item.confidence for item in items) / len(items)
            if self.vote_policy == "majority":
                confidence *= len(items) / len(votes)
            evidence: list[str] = []
            descriptions: list[str] = []
            suggested: dict[str, Any] = {}
            for item in items:
                descriptions.append(item.description)
                evidence.extend(item.evidence)
                suggested.update(item.suggested_data)
            merged.append(
                Defect(
                    id=stable_hash("llm", capability, subcapability, descriptions[:1]),
                    capability=capability,
                    subcapability=subcapability,
                    description=" | ".join(dict.fromkeys(descriptions))[:800],
                    severity=severity,
                    confidence=min(0.99, confidence),
                    source="llm_judge",
                    evidence=list(dict.fromkeys(evidence))[:10],
                    suggested_data=suggested,
                    round=items[0].round,
                )
            )
        merged.sort(key=lambda d: d.severity, reverse=True)
        return merged


@register("analyzer", "harness")
class HarnessDeepDiveAnalyzer(AnalyzerBackend):
    """Optional agentic deep-dive. The harness is advisory and schema-gated."""

    def __init__(self, runtime: Any, cfg: Any, **params: Any) -> None:
        super().__init__(runtime, cfg, **params)
        self.harness_name = str(self.cfg.get("harness", "analysis"))
        self.prompt_id = self.cfg.get("prompt")
        self.skills = [str(v) for v in self.cfg.get_list("skills", ["badcase-deepdive"])]
        self.top_n = self.cfg.get_int("top_n", 8)
        self.max_score = self.cfg.get_float("max_score", 0.75)
        self.timeout_s = self.cfg.get_int("timeout_s", 900)
        if not self.prompt_id:
            raise ValueError("harness analyzer requires analyze.prompt")

    def analyze(
        self,
        collect: CollectResult,
        round_no: int,
        baseline: dict[str, float],
    ) -> DefectReport:
        cases = worst_items(collect.reports, self.runtime.catalog, self.top_n, self.max_score)
        rendered = self.runtime.prompt_hub.render(
            self.prompt_id,
            {
                "round": round_no,
                "worst_cases": cases,
                "capability_table": [
                    {"capability": s.capability, "score": s.score, "num_items": s.num_items}
                    for s in collect.capabilities.values()
                ],
                "baseline": baseline,
            },
        )
        work_dir = self.runtime.artifacts.stage_dir(round_no, "analyze") / "harness"
        task = HarnessTask(
            name="badcase_deepdive",
            instruction=rendered.text,
            work_dir=work_dir,
            skills=self.skills,
            output_schema=rendered.schema,
            timeout_s=self.timeout_s,
            metadata={"round": round_no, "checkpoint_id": collect.checkpoint_id},
        )
        result = self.runtime.harness(self.harness_name).run(task)
        if not result.ok:
            self.logger.warning("harness deep-dive failed: %s", result.error)
        defects = _defects_from_payload(
            result.output,
            source="harness",
            round_no=round_no,
            catalog=self.runtime.catalog,
        )
        return DefectReport(
            round=round_no,
            checkpoint_id=collect.checkpoint_id,
            defects=defects,
            capabilities=collect.capabilities,
            summary=f"harness deep-dive produced {len(defects)} defect(s)",
            sources=["harness"],
            metadata={"ok": result.ok, "error": result.error, "artifacts": result.artifacts},
        )
