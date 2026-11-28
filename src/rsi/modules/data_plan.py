from __future__ import annotations

import math
from typing import Any

from rsi.core.errors import ConfigError
from rsi.core.interfaces import Component
from rsi.core.jsonio import stable_hash
from rsi.core.schema import validate_json
from rsi.core.types import DataPlan, DataRequest, Defect, DefectReport, TaskSpec
from rsi.modules.data_strategy import DatasetStrategy, allocate_quotas
from rsi.modules.datasets import merge_ability_priorities


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
        distribution_cfg = self.cfg.section("target_distribution", required=False)
        self.target_distribution_enabled = distribution_cfg.get_bool("enabled", True)
        self.previous_weight = distribution_cfg.get_float("previous_weight", 0.3)
        self.pending_priority = distribution_cfg.get_float("pending_priority", 0.25)
        if not math.isfinite(self.previous_weight) or not 0 <= self.previous_weight < 1:
            raise ConfigError("data.plan.target_distribution.previous_weight must be within [0, 1)")
        if not math.isfinite(self.pending_priority) or not 0 <= self.pending_priority <= 1:
            raise ConfigError("data.plan.target_distribution.pending_priority must be within [0, 1]")
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
        distribution, distribution_info = self._target_distribution(candidates, defects, context)
        if self.strategy == "llm" and self.llm_prompt:
            plan = self._llm_plan(candidates, context)
            if plan is not None:
                plan.metadata["curriculum"] = curriculum.notes
                plan.metadata["curriculum_stage"] = curriculum.stage
                self._attach_target_distribution(plan, distribution, distribution_info)
                return plan
            self.logger.warning("LLM data planning failed; falling back to rule planner")
        plan = self._rule_plan(candidates, context)
        plan.metadata["curriculum"] = curriculum.notes
        plan.metadata["curriculum_stage"] = curriculum.stage
        plan.metadata["curriculum_excluded"] = curriculum.excluded
        self._attach_target_distribution(plan, distribution, distribution_info)
        return plan

    def plan_strategy(self, strategy: DatasetStrategy, defects: DefectReport, context: Any) -> DataPlan:
        """Declare the target abilities of a chapter-2 data strategy.

        The requests carry the target distribution, one request per target
        ability. Sampling itself is done from the strategy's selected datasets
        and ratios; the requests remain the round's declared capability targets
        for validation and training feedback.
        """
        target = dict(strategy.target.distribution)
        quotas = allocate_quotas(target, strategy.count)
        defect_by_capability = {defect.capability: defect for defect in defects.defects}
        requests: list[DataRequest] = []
        for capability in sorted(target):
            count = quotas.get(capability, 0)
            if count <= 0:
                continue
            defect = defect_by_capability.get(capability)
            info = self.runtime.catalog.get(capability)
            difficulty = list(self.default_difficulty)
            verifier = info.get("verifier") or {}
            seeds = list(info.get("seeds") or [])
            if defect is not None:
                difficulty = list(defect.suggested_data.get("target_difficulty") or difficulty)
                verifier = defect.suggested_data.get("verifier") or verifier
                seeds = list(defect.suggested_data.get("seeds") or seeds)
            requests.append(
                DataRequest(
                    id=stable_hash("strategy-request", context.round, capability),
                    capability=capability,
                    kind="dataset",
                    count=count,
                    target_difficulty=difficulty,
                    constraints={
                        "min_severity": self.min_severity,
                        "verifier": verifier,
                        "source_round": context.round,
                        "target_share": target[capability],
                        "strategy": "dataset_strategy",
                        "scheduled": False,
                    },
                    seeds=seeds,
                    reward_hint=self.runtime.catalog.reward(capability),
                    defect_id=defect.id if defect is not None else "",
                    notes=(defect.description if defect is not None else f"Target capability {capability}")[:500],
                )
            )
        plan = DataPlan(
            round=context.round,
            requests=requests,
            total_target=sum(request.count for request in requests),
            strategy="dataset_strategy",
            notes=f"{len(requests)} target abilities from the data strategy",
        )
        plan.metadata["target_distribution"] = target
        plan.metadata["target_distribution_info"] = strategy.target.as_dict()
        plan.metadata["dataset_strategy"] = strategy.summary()
        return plan

    def plan_task(self, task: TaskSpec, context: Any) -> DataPlan:
        """Create dataset-backed requests from an explicit task ability profile."""
        if task.count < 0:
            raise ValueError("task count must be non-negative")
        defects: list[Defect] = []
        for capability, priority in task.ability_priorities.items():
            priority = float(priority)
            if not 0 <= priority <= 1:
                raise ValueError(f"task priority for {capability!r} must be in [0, 1]")
            info = self.runtime.catalog.get(capability)
            defects.append(
                Defect(
                    id=stable_hash("task", task.id, capability),
                    capability=capability,
                    description=task.notes or f"Task {task.id} requires {capability}",
                    severity=priority,
                    confidence=1.0,
                    source="task",
                    suggested_data={"kind": "dataset", "seeds": info.get("seeds", [])},
                    round=context.round,
                )
            )
        report = DefectReport(
            round=context.round,
            checkpoint_id="task",
            defects=defects,
            sources=["task"],
            metadata={"ability_priorities": dict(task.ability_priorities)},
        )
        plan = self._rule_plan(defects[: self.max_requests], context)
        if task.count and plan.total_target > task.count:
            remaining = task.count
            for request in plan.requests:
                request.count = min(request.count, remaining)
                remaining -= request.count
            plan.requests = [request for request in plan.requests if request.count > 0]
            plan.total_target = sum(request.count for request in plan.requests)
        plan.strategy = "task_requirements"
        plan.notes = f"{len(plan.requests)} dataset requests for task {task.id}"
        plan.metadata["ability_priorities"] = dict(task.ability_priorities)
        return plan

    def _target_distribution(
        self, defects: list[Defect], report: DefectReport, context: Any
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """Mock pre-processing of the capability target distribution.

        The flow is: merged analyzer priorities + defect fallback + pending
        capabilities -> optional blend with the previous round -> normalize.
        Counts and the final matching algorithm can be refined later without
        changing the data contract (``plan.metadata["target_distribution"]``).
        """
        if not self.target_distribution_enabled:
            return {}, {"enabled": False}
        merged = report.metadata.get("ability_priorities") or {}
        current: dict[str, float] = {}
        for defect in defects:
            raw = merged.get(defect.capability)
            if raw is None:
                raw = defect.severity * defect.confidence
            try:
                value = max(0.0, min(1.0, float(raw)))
            except (TypeError, ValueError):
                value = max(0.0, min(1.0, defect.severity * defect.confidence))
            current[defect.capability] = max(current.get(defect.capability, 0.0), value)

        pending = {
            str(name)
            for name in getattr(getattr(context, "state", None), "pending_capabilities", None) or []
        }
        carried = sorted(pending & set(current))
        for capability in carried:
            current[capability] = max(current[capability], self.pending_priority)

        previous = self._previous_distribution(context)
        previous = {name: value for name, value in previous.items() if name in current}
        if previous and self.previous_weight > 0:
            blend = merge_ability_priorities(
                current, previous,
                rule_weight=1.0 - self.previous_weight,
                agent_weight=self.previous_weight,
            )
        else:
            blend = current
        total = sum(blend.values())
        if total <= 0:
            blend = dict.fromkeys(current, 1.0 / len(current)) if current else {}
            total = sum(blend.values())
        distribution = {name: value / total for name, value in sorted(blend.items())}
        info = {
            "enabled": True,
            "current": current,
            "previous": previous,
            "previous_weight": self.previous_weight,
            "pending": carried,
            "sources": ["defects"] + (["previous"] if previous else []),
        }
        return distribution, info

    @staticmethod
    def _previous_distribution(context: Any) -> dict[str, float]:
        history = getattr(getattr(context, "state", None), "history", None) or []
        for record in reversed(history):
            data = record.get("synthesize") or {}
            raw = data.get("target_distribution")
            if not isinstance(raw, dict) or not raw:
                continue
            return {
                str(name): max(0.0, min(1.0, float(value)))
                for name, value in raw.items()
                if isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > 0
            }
        return {}

    def _attach_target_distribution(
        self, plan: DataPlan, distribution: dict[str, float], info: dict[str, Any]
    ) -> None:
        plan.metadata["target_distribution"] = distribution
        plan.metadata["target_distribution_info"] = info
        for request in plan.requests:
            request.constraints["target_share"] = distribution.get(request.capability, 0.0)

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
