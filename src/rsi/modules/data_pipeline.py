from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rsi.core.errors import ConfigError
from rsi.core.jsonio import atomic_write_json, to_plain
from rsi.core.types import DataPlan, DatasetArtifact, DefectReport
from rsi.modules.data_builder import JsonlDatasetBuilder  # noqa: F401 - registers plugin
from rsi.modules.data_collectors import (  # noqa: F401 - registers plugins
    AgenticSynthesisCollector,
    KnowledgeIngestCollector,
    RolloutReuseCollector,
    TextSynthesisCollector,
)
from rsi.modules.data_difficulty import (  # noqa: F401 - registers plugins
    CompositeDifficultyGrader,
    JudgeDifficultyGrader,
    PassRateDifficultyGrader,
    RuleDifficultyGrader,
)
from rsi.modules.data_filters import FilterPipeline
from rsi.modules.data_mixing import DiversitySampler
from rsi.modules.data_plan import DataPlanner


@dataclass
class DataContext:
    round: int
    runtime: Any
    state: Any
    collect_result: Any
    defects: DefectReport | None
    work_dir: Path
    plan: DataPlan | None = None
    rng: Any = None
    history: list[dict[str, Any]] = field(default_factory=list)
    mixing_info: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataPipelineResult:
    round: int
    plan: DataPlan
    dataset: DatasetArtifact | None
    generated: int
    filtered: int
    selected: int
    filter_stats: dict[str, Any] = field(default_factory=dict)
    mixing: dict[str, float] = field(default_factory=dict)
    mixing_info: dict[str, Any] = field(default_factory=dict)
    sampler_stats: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


class DataPipeline:
    """Plan -> generate -> filter -> grade -> mix -> build.

    One instance is created per loop so stateful filters (dedup) can persist
    across rounds. Every stage is a registry plugin; errors in one request do
    not abort the others.
    """

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self.cfg = runtime.config.section("data", required=False)
        self.enabled = self.cfg.get_bool("enabled", True)
        self.planner = DataPlanner(runtime, self.cfg.section("plan", required=False))
        self.collectors: dict[str, Any] = {}
        for spec in self.cfg.specs("collectors"):
            name = str(spec.get("name") or spec["type"])
            try:
                child = self.cfg.section(f"collectors.{name}")
            except ConfigError:
                child = self.cfg.child(spec, name)
            kind = str(spec.get("kind") or name)
            self.collectors[kind] = runtime.build("data_collector", spec, child)
        if not self.collectors:
            raise ConfigError("data.collectors is empty")
        self.default_collector = str(self.cfg.get("default_collector", ""))
        self.filter_pipeline = FilterPipeline(runtime, self.cfg.section("filter", required=False))
        self.grader = runtime.build(
            "difficulty",
            self.cfg.plugin("difficulty", "rule"),
            self.cfg.section("difficulty", required=False),
        )
        self.mixer = runtime.build(
            "mixing",
            self.cfg.plugin("mixing", "static"),
            self.cfg.section("mixing", required=False),
        )
        self.sampler = DiversitySampler(runtime, self.cfg.section("mixing", required=False))
        self.builder = runtime.build(
            "dataset_builder",
            self.cfg.plugin("build", "jsonl"),
            self.cfg.section("build", required=False),
        )
        self.max_generated = self.cfg.get_int("max_generated", 512)
        self.save_intermediate = self.cfg.get_bool("save_intermediate", False)

    def validate(self) -> None:
        """Check every prompt referenced by collectors, filters and graders."""
        for collector in self.collectors.values():
            prompt_id = getattr(collector, "prompt_id", None)
            if prompt_id:
                self.runtime.prompt_hub.get(str(prompt_id))
        for _, sample_filter in self.filter_pipeline.filters:
            prompt_id = getattr(sample_filter, "prompt_id", None)
            if prompt_id:
                self.runtime.prompt_hub.get(str(prompt_id))
        prompt_id = getattr(self.grader, "prompt_id", None)
        if prompt_id:
            self.runtime.prompt_hub.get(str(prompt_id))

    def run(
        self,
        round_no: int,
        defects: DefectReport,
        collect_result: Any,
        state: Any,
    ) -> DataPipelineResult:
        if not self.enabled:
            plan = DataPlan(round=round_no, strategy="disabled", notes="data.enabled=false")
            return DataPipelineResult(round_no, plan, None, 0, 0, 0)
        work_dir = self.runtime.artifacts.stage_dir(round_no, "synthesize")
        context = DataContext(
            round=round_no,
            runtime=self.runtime,
            state=state,
            collect_result=collect_result,
            defects=defects,
            work_dir=work_dir,
            rng=self.runtime.rng("data", round_no),
            history=list(getattr(state, "history", []) or []),
            mixing_info={},
        )
        plan = self.planner.plan(defects, context)
        context.plan = plan
        atomic_write_json(work_dir / "plan.json", plan)
        if not plan.requests:
            result = DataPipelineResult(
                round=round_no,
                plan=plan,
                dataset=None,
                generated=0,
                filtered=0,
                selected=0,
                stats={"skipped": "no data requests"},
            )
            atomic_write_json(work_dir / "pipeline.json", result)
            return result

        samples = []
        errors: list[str] = []
        generated_per_request: dict[str, int] = {}
        for request in plan.requests:
            if len(samples) >= self.max_generated:
                break
            collector = self.collectors.get(request.kind)
            if collector is None and self.default_collector:
                collector = self.collectors.get(self.default_collector)
            if collector is None:
                errors.append(f"no collector for kind {request.kind!r}")
                continue
            try:
                produced = collector.collect(request, context)
            except Exception as exc:  # noqa: BLE001 - keep the loop alive
                self.runtime.logger.warning(
                    "data collector %s failed for %s: %s", request.kind, request.id, exc
                )
                errors.append(f"{request.kind}/{request.id}: {exc}")
                continue
            produced = produced[: request.count]
            generated_per_request[request.id] = len(produced)
            samples.extend(produced)
        samples = samples[: self.max_generated]
        if self.save_intermediate:
            from rsi.core.jsonio import write_jsonl

            write_jsonl(work_dir / "generated.jsonl", samples)

        kept, decisions, filter_stats = self.filter_pipeline.run(samples, context)
        for sample in kept:
            sample.difficulty = self.grader.grade(sample, context)
        weights = self.mixer.plan(kept, plan, context)
        context.mixing_info = dict(getattr(self.mixer, "last_info", {}) or {})
        selected, sampler_stats = self.sampler.select(kept, weights, context)
        if self.save_intermediate:
            from rsi.core.jsonio import write_jsonl

            write_jsonl(work_dir / "filtered.jsonl", kept)
            write_jsonl(work_dir / "selected.jsonl", selected)
        dataset = self.builder.build(selected, context, plan, weights)
        if dataset is not None:
            dataset.mixing = weights
            dataset.stats.update(
                {
                    "mixing_info": context.mixing_info,
                    "sampler": sampler_stats,
                    "filter": filter_stats,
                    "generated_per_request": generated_per_request,
                    "errors": errors,
                }
            )
        result = DataPipelineResult(
            round=round_no,
            plan=plan,
            dataset=dataset,
            generated=len(samples),
            filtered=len(kept),
            selected=len(selected),
            filter_stats=filter_stats,
            mixing=weights,
            mixing_info=context.mixing_info,
            sampler_stats=sampler_stats,
            errors=errors,
            stats={
                "generated_per_request": generated_per_request,
                "requested": plan.total_target,
                "generated": len(samples),
                "filtered": len(kept),
                "selected": len(selected),
                "dataset_id": dataset.dataset_id if dataset else None,
            },
        )
        atomic_write_json(work_dir / "pipeline.json", result)
        if self.save_intermediate:
            from rsi.core.jsonio import write_jsonl

            write_jsonl(work_dir / "filter_decisions.jsonl", decisions)
        return result
