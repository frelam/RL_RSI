from __future__ import annotations

from typing import Any

from rsi.core.interfaces import AnalyzerBackend
from rsi.core.registry import register
from rsi.core.types import CollectResult, DefectReport, LLMRequest


@register("analyzer", "trajectory_agent")
class TrajectoryAbilityAnalyzer(AnalyzerBackend):
    """Estimate task ability requirements from collected interaction traces."""

    def __init__(self, runtime: Any, cfg: Any, **params: Any) -> None:
        super().__init__(runtime, cfg, **params)
        self.prompt_id = str(self.cfg.get("prompt", "trajectory_ability_analysis"))
        self.profile = str(self.cfg.get("profile", runtime.config.get("llm.default_profile", "judge")))
        self.max_trajectories = max(1, self.cfg.get_int("max_trajectories", 12))
        self.min_confidence = self.cfg.get_float("min_confidence", 0.3)

    def analyze(
        self, collect: CollectResult, round_no: int, baseline: dict[str, float]
    ) -> DefectReport:
        trajectories = collect.trajectories[: self.max_trajectories]
        if not trajectories:
            return DefectReport(
                round=round_no,
                checkpoint_id=collect.checkpoint_id,
                capabilities=collect.capabilities,
                summary="trajectory analysis skipped: no trajectories",
                sources=["trajectory_agent"],
                metadata={"ability_priorities": {}, "calls": 0},
            )
        rendered = self.runtime.prompt_hub.render(
            self.prompt_id,
            {
                "capabilities": self.runtime.catalog.all(),
                "trajectories": [
                    {
                        "prompt": item.prompt[:2000],
                        "answer": item.final_answer[:1000],
                        "reward": item.reward,
                        "success": item.success,
                        "metadata": item.metadata,
                    }
                    for item in trajectories
                ],
            },
        )
        response = self.runtime.llm(self.profile).complete(
            LLMRequest(
                prompt_id=rendered.key,
                prompt=rendered.text,
                system=rendered.system,
                schema=rendered.schema,
                temperature=0,
                max_tokens=1200,
                metadata={"round": round_no, "checkpoint_id": collect.checkpoint_id},
            )
        )
        payload = next((item for item in response.parsed if item is not None), None) or {}
        raw = payload.get("ability_priorities") or payload.get("abilities") or {}
        known = {item["name"] for item in self.runtime.catalog.all()}
        priorities: dict[str, float] = {}
        if isinstance(raw, list):
            raw = {item.get("ability"): item.get("priority") for item in raw if isinstance(item, dict)}
        if isinstance(raw, dict):
            for name, value in raw.items():
                try:
                    priority = max(0.0, min(1.0, float(value)))
                except (TypeError, ValueError):
                    continue
                if name in known and priority >= self.min_confidence:
                    priorities[str(name)] = priority
        return DefectReport(
            round=round_no,
            checkpoint_id=collect.checkpoint_id,
            capabilities=collect.capabilities,
            summary=f"trajectory analysis estimated {len(priorities)} ability priority/priorities",
            sources=["trajectory_agent"],
            metadata={
                "ability_priorities": priorities,
                "prompt": rendered.key,
                "num_trajectories": len(trajectories),
            },
        )
