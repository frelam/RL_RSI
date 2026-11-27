from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rsi.core.errors import ConfigError
from rsi.core.jsonio import atomic_write_json, read_jsonl, stable_hash
from rsi.core.types import ChatMessage, Checkpoint, DataPlan, DataSample, DatasetArtifact, DefectReport, TaskSpec
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
from rsi.modules.data_mixing import DiversitySampler, group_key
from rsi.modules.data_plan import DataPlanner


@dataclass
class DataContext:
    round: int
    runtime: Any
    state: Any
    collect_result: Any
    defects: DefectReport | None
    work_dir: Path
    checkpoint: Checkpoint | None = None
    plan: DataPlan | None = None
    rng: Any = None
    history: list[dict[str, Any]] = field(default_factory=list)
    target_distribution: dict[str, float] = field(default_factory=dict)
    dataset_selection: Any = None
    mixing_info: dict[str, Any] = field(default_factory=dict)
    mixing_groups: set[str] | None = None
    sampler_stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataPipelineResult:
    round: int
    plan: DataPlan
    dataset: DatasetArtifact | None
    generated: int
    filtered: int
    selected: int
    replayed: int = 0
    filter_stats: dict[str, Any] = field(default_factory=dict)
    mixing: dict[str, float] = field(default_factory=dict)
    mixing_info: dict[str, Any] = field(default_factory=dict)
    sampler_stats: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


def _capability_counts(samples: list[DataSample]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in samples:
        counts[sample.capability] = counts.get(sample.capability, 0) + 1
    return counts


def _replay_quotas(shares: dict[str, float], total: int) -> dict[str, int]:
    exact = {name: share * total for name, share in shares.items()}
    quotas = {name: int(value) for name, value in exact.items()}
    remainder = total - sum(quotas.values())
    order = sorted(shares, key=lambda name: (exact[name] - quotas[name], name), reverse=True)
    for name in order[: max(0, remainder)]:
        quotas[name] += 1
    return quotas


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
        self.replay_cfg = self.cfg.section("replay", required=False)
        self.replay_fraction = self.replay_cfg.get_float("fraction", 0.0)
        self.replay_rounds = max(0, self.replay_cfg.get_int("recent_rounds", 1))
        self.max_replay = max(0, self.replay_cfg.get_int("max_samples", 128))
        self.replay_strategy = str(self.replay_cfg.get("strategy", "uniform"))
        if not math.isfinite(self.replay_fraction) or not 0 <= self.replay_fraction < 1:
            raise ConfigError("data.replay.fraction must be within [0, 1)")
        if self.replay_strategy not in {"uniform", "target"}:
            raise ConfigError("data.replay.strategy must be 'uniform' or 'target'")
        self._last_replay_info: dict[str, Any] = {}

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

    def plan_task(self, task: TaskSpec, round_no: int = 0) -> DataPlan:
        """Expose task-to-dataset planning without starting a training run."""
        return self.planner.plan_task(task, type("TaskContext", (), {"round": round_no})())

    def collect_task(self, task: TaskSpec, round_no: int = 0) -> list[DataSample]:
        """Collect task-targeted rows from the configured dataset library."""
        plan = self.plan_task(task, round_no)
        collector = self.collectors.get("dataset")
        if collector is None:
            raise ConfigError("collect_task requires a data collector with kind 'dataset'")
        context = DataContext(
            round=round_no,
            runtime=self.runtime,
            state=None,
            collect_result=None,
            defects=None,
            work_dir=self.runtime.artifacts.stage_dir(round_no, "task-data"),
            plan=plan,
            rng=self.runtime.rng("task-data", task.id, round_no),
        )
        samples: list[DataSample] = []
        for request in plan.requests:
            samples.extend(collector.collect(request, context)[:request.count])
        return samples[:task.count]

    def run(
        self,
        round_no: int,
        defects: DefectReport,
        collect_result: Any,
        state: Any,
        *,
        checkpoint: Checkpoint | None = None,
        difficulty_shift: float | dict[str, float] = 0.0,
        reward_multiplier: float = 1.0,
        attempt: int = 0,
    ) -> DataPipelineResult:
        if not self.enabled:
            plan = DataPlan(round=round_no, strategy="disabled", notes="data.enabled=false")
            return DataPipelineResult(round_no, plan, None, 0, 0, 0)
        work_dir = self.runtime.artifacts.stage_dir(round_no, "synthesize")
        if attempt:
            work_dir = work_dir / f"attempt-{attempt}"
        context = DataContext(
            round=round_no,
            runtime=self.runtime,
            state=state,
            collect_result=collect_result,
            defects=defects,
            work_dir=work_dir,
            checkpoint=checkpoint,
            rng=self.runtime.rng("data", round_no, attempt),
            history=list(getattr(state, "history", []) or []),
            mixing_info={},
        )
        plan = self.planner.plan(defects, context)
        context.plan = plan
        context.target_distribution = dict(plan.metadata.get("target_distribution") or {})
        plan.metadata.update(
            attempt=attempt,
            difficulty_shift=difficulty_shift,
            reward_multiplier=reward_multiplier,
        )
        shifts = difficulty_shift if isinstance(difficulty_shift, dict) else {}
        for request in plan.requests:
            shift = (
                float(shifts.get(request.capability, 0.0))
                if isinstance(difficulty_shift, dict)
                else float(difficulty_shift)
            )
            if len(request.target_difficulty) != 2:
                raise ConfigError(f"invalid target_difficulty for {request.id}")
            low, high = map(float, request.target_difficulty)
            if not all(math.isfinite(value) for value in (low, high)) or not 0 <= low <= high <= 1:
                raise ConfigError(f"target_difficulty must be within [0, 1] for {request.id}")
            request.target_difficulty = [min(1.0, low + shift), min(1.0, high + shift)]
            if request.target_difficulty[0] > request.target_difficulty[1]:
                request.target_difficulty[0] = request.target_difficulty[1]
            if attempt:
                from rsi.core.jsonio import stable_hash

                request.id = stable_hash(request.id, attempt)
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

        new_samples: list[DataSample] = []
        errors: list[str] = []
        generated_per_request: dict[str, int] = {}
        for request in plan.requests:
            if len(new_samples) >= self.max_generated:
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
            for sample in produced:
                sample.metadata["target_difficulty"] = request.target_difficulty
                reward_hint = dict(request.reward_hint)
                reward_hint["weight"] = float(reward_hint.get("weight", 1.0)) * reward_multiplier
                sample.metadata["reward_hint"] = reward_hint
            generated_per_request[request.id] = len(produced)
            new_samples.extend(produced)
        new_samples = new_samples[: self.max_generated]
        replay_samples = self._load_replay(
            context.history,
            len(new_samples),
            round_no,
            target_distribution=context.target_distribution,
        )
        replay_info = dict(self._last_replay_info)
        plan.metadata["replayed"] = len(replay_samples)
        samples = replay_samples + new_samples
        if self.save_intermediate:
            from rsi.core.jsonio import write_jsonl

            write_jsonl(work_dir / "generated.jsonl", samples)

        if self.sampler.preserve_ratio:
            # A group with zero surviving samples still contributes to the target.
            context.mixing_groups = {group_key(sample) for sample in samples}
        kept, decisions, filter_stats = self.filter_pipeline.run(samples, context)
        for sample in kept:
            sample.difficulty = self.grader.grade(sample, context)
        weights = self.mixer.plan(kept, plan, context)
        context.mixing_info = dict(getattr(self.mixer, "last_info", {}) or {})
        selected, sampler_stats = self.sampler.select(kept, weights, context)
        context.sampler_stats = sampler_stats
        if self.save_intermediate:
            from rsi.core.jsonio import write_jsonl

            write_jsonl(work_dir / "filtered.jsonl", kept)
            write_jsonl(work_dir / "selected.jsonl", selected)
        dataset = self.builder.build(selected, context, plan, weights)
        merge_info = {
            "target_distribution": context.target_distribution,
            "new_by_capability": _capability_counts(new_samples),
            "replay_by_capability": _capability_counts(replay_samples),
            "selected_by_capability": _capability_counts(selected),
        }
        if dataset is not None:
            dataset.mixing = weights
            dataset.stats.update(
                {
                    "mixing_info": context.mixing_info,
                    "sampler": sampler_stats,
                    "filter": filter_stats,
                    "generated_per_request": generated_per_request,
                    "errors": errors,
                    "replay": replay_info,
                    "merge": merge_info,
                }
            )
        result = DataPipelineResult(
            round=round_no,
            plan=plan,
            dataset=dataset,
            generated=len(new_samples),
            filtered=len(kept),
            selected=len(selected),
            replayed=len(replay_samples),
            filter_stats=filter_stats,
            mixing=weights,
            mixing_info=context.mixing_info,
            sampler_stats=sampler_stats,
            errors=errors,
            stats={
                "generated_per_request": generated_per_request,
                "requested": plan.total_target,
                "generated": len(new_samples),
                "replayed": len(replay_samples),
                "filtered": len(kept),
                "selected": len(selected),
                "dataset_id": dataset.dataset_id if dataset else None,
                "target_distribution": context.target_distribution,
                "replay": replay_info,
                "merge": merge_info,
            },
        )
        atomic_write_json(work_dir / "pipeline.json", result)
        if self.save_intermediate:
            from rsi.core.jsonio import write_jsonl

            write_jsonl(work_dir / "filter_decisions.jsonl", decisions)
        return result

    def _load_replay(
        self,
        history: list[dict[str, Any]],
        new_count: int,
        round_no: int,
        target_distribution: dict[str, float] | None = None,
    ) -> list[DataSample]:
        self._last_replay_info = {}
        if (
            not self.replay_cfg.get_bool("enabled", False)
            or self.replay_fraction <= 0
            or new_count <= 0
        ):
            return []
        target = round(new_count * self.replay_fraction / (1 - self.replay_fraction))
        target = min(target, self.max_replay)
        if target <= 0:
            return []
        candidates: list[DataSample] = []
        seen: set[str] = set()
        used_rounds = 0
        for record in reversed(history):
            if used_rounds >= self.replay_rounds:
                break
            dataset_path = (record.get("synthesize") or {}).get("dataset_path")
            if not dataset_path:
                continue
            train_path = Path(dataset_path) / "train.jsonl"
            if not train_path.exists():
                continue
            source_round = int(record.get("round", -1))
            for row in read_jsonl(train_path):
                sample = self._sample_from_replay_row(row, source_round)
                if sample is None:
                    continue
                key = stable_hash(sample.prompt)
                if key not in seen:
                    seen.add(key)
                    candidates.append(sample)
            used_rounds += 1

        strategy = getattr(self, "replay_strategy", "uniform")
        distribution = {
            str(name): float(value)
            for name, value in (target_distribution or {}).items()
            if isinstance(value, (int, float)) and float(value) > 0
        }
        quotas: dict[str, int] = {}
        if strategy == "target" and distribution and candidates:
            replay, quotas = self._target_replay(candidates, distribution, target, round_no)
            used_strategy = "target"
        else:
            replay = sorted(
                candidates, key=lambda sample: stable_hash(self.runtime.seed, round_no, "replay", sample.id)
            )[:target]
            used_strategy = "uniform"
        self._last_replay_info = {
            "strategy": used_strategy,
            "target": target,
            "loaded": len(candidates),
            "selected": len(replay),
            "by_capability": _capability_counts(replay),
            "target_by_capability": quotas,
            "rounds": used_rounds,
        }
        return replay

    def _target_replay(
        self,
        candidates: list[DataSample],
        distribution: dict[str, float],
        target: int,
        round_no: int,
    ) -> tuple[list[DataSample], dict[str, int]]:
        """Mock merge: allocate the replay budget by the target distribution."""
        by_capability: dict[str, list[DataSample]] = {}
        for sample in candidates:
            by_capability.setdefault(sample.capability, []).append(sample)
        shares = {name: distribution.get(name, 0.0) for name in by_capability}
        total = sum(shares.values())
        if total <= 0:
            return [], {}
        shares = {name: value / total for name, value in shares.items()}
        quotas = _replay_quotas(shares, target)
        order = lambda sample: stable_hash(self.runtime.seed, round_no, "replay", sample.id)
        replay: list[DataSample] = []
        for capability in sorted(by_capability):
            replay.extend(sorted(by_capability[capability], key=order)[: quotas.get(capability, 0)])
        limit = min(target, len(candidates))
        if len(replay) < limit:
            taken = {sample.id for sample in replay}
            leftovers = sorted(
                (sample for sample in candidates if sample.id not in taken), key=order
            )
            replay.extend(leftovers[: limit - len(replay)])
        replay.sort(key=order)
        return replay[:target], quotas

    def _sample_from_replay_row(
        self, row: dict[str, Any], source_round: int
    ) -> DataSample | None:
        extra = row.get("extra_info") if isinstance(row.get("extra_info"), dict) else {}
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        raw_messages = row.get("messages")
        prompt_value = row.get("prompt", "")
        if isinstance(prompt_value, list):
            raw_messages = prompt_value
            prompt_value = ""
        messages = [
            ChatMessage(role=str(item.get("role", "user")), content=str(item.get("content", "")))
            for item in raw_messages or []
            if isinstance(item, dict)
        ]
        prompt = str(prompt_value or "")
        if not prompt:
            prompt = next((message.content for message in messages if message.role == "user"), "")
        response = str(row.get("response") or "")
        if not response:
            response = next(
                (message.content for message in reversed(messages) if message.role == "assistant"),
                "",
            )
        if not prompt:
            return None
        if not messages:
            messages = [ChatMessage(role="user", content=prompt)]
        if response and not any(message.role == "assistant" for message in messages):
            messages.append(ChatMessage(role="assistant", content=response))

        reward_model = row.get("reward_model") if isinstance(row.get("reward_model"), dict) else {}
        reference = row.get("reference", reward_model.get("ground_truth"))
        capability = str(row.get("capability") or row.get("ability") or "general")
        verifier = row.get("verifier") or extra.get("verifier")
        if verifier is None and reward_model:
            style = str(reward_model.get("style") or "contains")
            verifier = {"type": "rule", "name": style}
        sample_metadata = dict(metadata)
        sample_metadata.update(replay=True, replayed_from_round=source_round)
        sample_id = row.get("id") or extra.get("id")
        if not sample_id:
            sample_id = stable_hash("replay", source_round, prompt, response)
        try:
            difficulty = float(row.get("difficulty", extra.get("difficulty", 0.0)) or 0.0)
            quality = float(row.get("quality", extra.get("quality", 1.0)) or 0.0)
        except (TypeError, ValueError):
            difficulty, quality = 0.0, 1.0
        return DataSample(
            id=str(sample_id),
            kind=str(row.get("kind") or extra.get("kind") or "text"),
            capability=capability,
            subcapability=str(row.get("subcapability") or extra.get("subcapability") or ""),
            prompt=prompt,
            response=response,
            reference=None if reference is None else str(reference),
            messages=messages,
            verifier=verifier if isinstance(verifier, dict) else None,
            source=f"replay:{row.get('source') or extra.get('source') or source_round}",
            difficulty=max(0.0, min(1.0, difficulty)),
            quality=max(0.0, min(1.0, quality)),
            reward=row.get("reward"),
            group=str(row.get("group") or capability),
            round=source_round,
            env_path=row.get("env_path") or extra.get("env_path"),
            metadata=sample_metadata,
        )
