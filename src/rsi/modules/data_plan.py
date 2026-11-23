from __future__ import annotations

from typing import Any

from rsi.core.interfaces import Component
from rsi.core.jsonio import stable_hash
from rsi.core.schema import validate_json
from rsi.core.types import DataPlan, DataRequest, Defect, DefectReport


class DataPlanner(Component):
    """Turn a defect report into concrete, bounded data requests."""

    def __init__(self, runtime: Any, cfg: Any, **params: Any) -> None:
        super().__init__(runtime, cfg, **params)
        self.strategy = str(self.cfg.get("strategy", "rule"))
        self.count_per_defect = self.cfg.get_int("count_per_defect", 3)
        self.min_count = self.cfg.get_int("min_count", 2)
        self.max_count = self.cfg.get_int("max_count", 8)
        self.max_requests = self.cfg.get_int("max_requests", 12)
        self.min_severity = self.cfg.get_float("min_severity", 0.05)
        self.default_difficulty = [
            float(v) for v in self.cfg.get_list("target_difficulty", [0.25, 0.85])
        ]
        self.prompt_by_kind = {
            str(k): str(v) for k, v in self.cfg.get_dict("prompt_by_kind", {}).items()
        }
        self.fallback_kind = str(self.cfg.get("fallback_kind", "text"))
        self.llm_profile = str(self.cfg.get("profile", runtime.config.get("llm.default_profile", "judge")))
        self.llm_prompt = self.cfg.get("prompt")
        self.curriculum = runtime.build(
            "curriculum",
            self.cfg.plugin("curriculum", "severity"),
            self.cfg.section("curriculum", required=False),
        )

    def plan(self, defects: DefectReport, context: Any) -> DataPlan:
        candidates = [
            defect
            for defect in defects.defects
            if defect.severity >= self.min_severity
        ]
        if not candidates:
            return DataPlan(round=context.round, requests=[], strategy=self.strategy, notes="no defects")
        curriculum = self.curriculum.select(candidates, context)
        candidates = curriculum.defects[: self.max_requests]
        if not candidates:
            return DataPlan(
                round=context.round,
                requests=[],
                strategy=self.strategy,
                notes=f"curriculum '{curriculum.stage}' selected no capabilities",
                metadata={"curriculum": curriculum.notes, "excluded": curriculum.excluded},
            )
        if self.strategy == "llm" and self.llm_prompt:
            plan = self._llm_plan(candidates, context)
            if plan is not None:
                plan.metadata["curriculum"] = curriculum.notes
                plan.metadata["curriculum_stage"] = curriculum.stage
                return plan
            self.logger.warning("LLM data planning failed; falling back to rule planner")
        plan = self._rule_plan(candidates, context)
        plan.metadata["curriculum"] = curriculum.notes
        plan.metadata["curriculum_stage"] = curriculum.stage
        plan.metadata["curriculum_excluded"] = curriculum.excluded
        return plan

    def _rule_plan(self, defects: list[Defect], context: Any) -> DataPlan:
        requests: list[DataRequest] = []
        for defect in defects:
            info = self.runtime.catalog.get(defect.capability)
            kind = str(defect.suggested_data.get("kind") or info.get("kind") or self.fallback_kind)
            priority = self.runtime.catalog.priority(defect.capability)
            count = int(round(
                self.count_per_defect * (0.6 + defect.severity) * (0.75 + 0.5 * priority)
            ))
            count = max(self.min_count, min(self.max_count, count))
            seeds = list(defect.suggested_data.get("seeds") or info.get("seeds") or [])
            verifier = defect.suggested_data.get("verifier") or info.get("verifier") or {}
            requests.append(
                DataRequest(
                    id=stable_hash("request", context.round, defect.id, kind),
                    capability=defect.capability,
                    subcapability=defect.subcapability,
                    kind=kind,
                    count=count,
                    target_difficulty=list(
                        defect.suggested_data.get("target_difficulty") or self.default_difficulty
                    ),
                    constraints={
                        "min_severity": self.min_severity,
                        "verifier": verifier,
                        "source_round": context.round,
                    },
                    seeds=seeds,
                    prompt_id=self.prompt_by_kind.get(kind, ""),
                    reward_hint=self.runtime.catalog.reward(defect.capability),
                    defect_id=defect.id,
                    notes=defect.description[:500],
                )
            )
        return DataPlan(
            round=context.round,
            requests=requests,
            total_target=sum(r.count for r in requests),
            strategy="rule",
            notes=f"{len(requests)} requests from {len(defects)} defects",
        )

    def _llm_plan(self, defects: list[Defect], context: Any) -> DataPlan | None:
        try:
            rendered = self.runtime.prompt_hub.render(
                str(self.llm_prompt),
                {
                    "round": context.round,
                    "defects": [
                        {
                            "id": d.id,
                            "capability": d.capability,
                            "subcapability": d.subcapability,
                            "severity": d.severity,
                            "confidence": d.confidence,
                            "description": d.description,
                        }
                        for d in defects
                    ],
                    "capabilities": self.runtime.catalog.all(),
                    "max_requests": self.max_requests,
                    "count_per_request": self.count_per_defect,
                },
            )
            from rsi.core.types import LLMRequest

            response = self.runtime.llm(self.llm_profile).complete(
                LLMRequest(
                    prompt_id=rendered.key,
                    prompt=rendered.text,
                    system=rendered.system,
                    schema=rendered.schema,
                    metadata={"round": context.round},
                )
            )
            payload = next((item for item in response.parsed if item is not None), None)
            if payload is None:
                return None
            schema_errors = validate_json(payload, rendered.schema) if rendered.schema else []
            if schema_errors:
                self.logger.warning("LLM plan schema errors: %s", schema_errors[:3])
                return None
            raw_requests = payload.get("requests") or []
            if not isinstance(raw_requests, list):
                return None
            allowed = {defect.capability for defect in defects}
            requests: list[DataRequest] = []
            for index, raw in enumerate(raw_requests[: self.max_requests]):
                if not isinstance(raw, dict) or not raw.get("capability"):
                    continue
                capability = str(raw["capability"])
                if capability not in allowed:
                    continue
                kind = str(raw.get("kind") or self.runtime.catalog.kind(capability))
                count = max(1, min(self.max_count, int(raw.get("count", self.count_per_defect))))
                difficulty = raw.get("target_difficulty") or self.default_difficulty
                if not isinstance(difficulty, list) or len(difficulty) != 2:
                    difficulty = self.default_difficulty
                requests.append(
                    DataRequest(
                        id=stable_hash("llm-request", context.round, capability, index),
                        capability=capability,
                        subcapability=str(raw.get("subcapability") or ""),
                        kind=kind,
                        count=count,
                        target_difficulty=[float(difficulty[0]), float(difficulty[1])],
                        constraints={
                            "verifier": self.runtime.catalog.verifier(capability),
                            "source_round": context.round,
                        },
                        seeds=[str(s) for s in (raw.get("seeds") or self.runtime.catalog.seeds(capability))],
                        prompt_id=self.prompt_by_kind.get(kind, ""),
                        reward_hint=self.runtime.catalog.reward(capability),
                        defect_id=str(raw.get("defect_id") or ""),
                        notes=str(raw.get("notes") or ""),
                    )
                )
            if {request.capability for request in requests} != allowed:
                return None
            return DataPlan(
                round=context.round,
                requests=requests,
                total_target=sum(r.count for r in requests),
                strategy="llm",
                notes=f"{len(requests)} requests planned by LLM judge",
                metadata={"prompt": rendered.key},
            )
        except Exception as exc:  # noqa: BLE001 - planner fallback is intentional
            self.logger.warning("LLM planner error: %s", exc)
            return None
