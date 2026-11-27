from __future__ import annotations

import math
import time
from typing import Any

from rsi.core.jsonio import atomic_write_json, stable_hash, to_plain, write_jsonl
from rsi.core.types import (
    CapabilityScore,
    DefectReport,
    StageResult,
    TrainingSpec,
)
from rsi.loops.base import LoopContext, Stage
from rsi.modules.analyze_merge import DefectMerger
from rsi.modules.collect import CollectService
from rsi.modules.data_pipeline import DataPipeline
from rsi.modules.metrics import capability_table, overall_metrics
from rsi.modules.train_monitor import MonitorHub, Supervisor


def _cap_scores(table: dict[str, CapabilityScore]) -> dict[str, float]:
    return {name: round(score.score, 6) for name, score in table.items()}


def _state_baseline(state: Any) -> dict[str, float]:
    for record in reversed(getattr(state, "history", []) or []):
        capabilities = (record.get("collect") or {}).get("capabilities")
        if isinstance(capabilities, dict) and capabilities:
            return {str(k): float(v) for k, v in capabilities.items()}
    return {}


class SyncStage(Stage):
    name = "sync"

    def run(self, ctx: LoopContext) -> StageResult:
        started = time.time()
        checkpoint = self.runtime.checkpoints.current()
        endpoint = self.runtime.inference().ensure(checkpoint)
        ctx.memory["checkpoint"] = checkpoint
        return StageResult(
            "ok",
            f"serving {checkpoint.id} v{checkpoint.version}",
            {
                "checkpoint_id": checkpoint.id,
                "version": checkpoint.version,
                "model_path": checkpoint.model_path,
                "endpoint": to_plain(endpoint),
            },
            duration_s=time.time() - started,
        )


class CollectStage(Stage):
    name = "collect"

    def __init__(self, runtime: Any, cfg: Any) -> None:
        super().__init__(runtime, cfg)
        self.service = CollectService(runtime, runtime.config.section("collect"))
        self.save_items = cfg.get_bool("save_items", True)
        self.save_trajectories = cfg.get_bool("save_trajectories", True)

    def validate(self) -> None:
        for name in self.cfg.get_list("benchmark.names", []):
            self.runtime.benchmark(str(name))

    def run(self, ctx: LoopContext) -> StageResult:
        started = time.time()
        checkpoint = ctx.memory["checkpoint"]
        baseline = _state_baseline(ctx.state)
        result = self.service.run(checkpoint, ctx.round, baseline)
        ctx.memory["collect"] = result
        for report in result.reports:
            payload = report if self.save_items else {
                "benchmark": report.benchmark,
                "checkpoint_id": report.checkpoint_id,
                "round": report.round,
                "capabilities": report.capabilities,
                "subcapabilities": report.subcapabilities,
                "aggregates": report.aggregates,
                "num_items": report.num_items,
            }
            self.runtime.artifacts.write_json(
                ctx.round, "collect", f"benchmark-{stable_hash(report.benchmark, length=8)}.json", payload
            )
        if self.save_trajectories and result.trajectories:
            self.runtime.artifacts.write_jsonl(
                ctx.round, "collect", "trajectories.jsonl", result.trajectories
            )
        self.runtime.artifacts.write_json(
            ctx.round,
            "collect",
            "collect.json",
            {
                "round": result.round,
                "checkpoint_id": result.checkpoint_id,
                "capabilities": _cap_scores(result.capabilities),
                "aggregate": result.aggregate,
                "num_reports": len(result.reports),
                "num_trajectories": len(result.trajectories),
            },
        )
        return StageResult(
            "ok",
            f"{len(result.reports)} benchmark(s), score {result.aggregate.get('mean_score', 0):.4f}",
            {
                "score": result.aggregate.get("mean_score", 0.0),
                "capabilities": _cap_scores(result.capabilities),
                "aggregate": result.aggregate,
                "num_reports": len(result.reports),
                "num_trajectories": len(result.trajectories),
                "checkpoint_id": result.checkpoint_id,
            },
            duration_s=time.time() - started,
        )


class AnalyzeStage(Stage):
    name = "analyze"

    def __init__(self, runtime: Any, cfg: Any) -> None:
        super().__init__(runtime, cfg)
        self.analyzers: list[tuple[str, Any]] = []
        for spec in cfg.specs("methods"):
            name = str(spec.get("name") or spec["type"])
            child = cfg.child(spec, name)
            self.analyzers.append((name, runtime.build("analyzer", spec, child)))
        self.merger = DefectMerger(cfg.section("merge", required=False))
        self.scope = {str(v) for v in runtime.config.get_list("data.plan.scope", [])}

    def validate(self) -> None:
        for name, analyzer in self.analyzers:
            prompt_id = getattr(analyzer, "prompt_id", None)
            if prompt_id:
                self.runtime.prompt_hub.get(str(prompt_id))
            harness_name = getattr(analyzer, "harness_name", None)
            if harness_name:
                self.runtime.harness(str(harness_name))

    def run(self, ctx: LoopContext) -> StageResult:
        started = time.time()
        collect = ctx.memory["collect"]
        baseline = collect.baseline_capabilities
        reports = []
        errors: list[str] = []
        for name, analyzer in self.analyzers:
            try:
                report = analyzer.analyze(collect, ctx.round, baseline)
                reports.append(report)
                self.runtime.artifacts.write_json(
                    ctx.round, "analyze", f"defects-{stable_hash(name, length=8)}.json", report
                )
            except Exception as exc:  # noqa: BLE001 - one analyzer must not kill the loop
                self.logger.warning("analyzer %s failed: %s", name, exc)
                errors.append(f"{name}: {exc}")
        dropped = 0
        if self.scope:
            for report in reports:
                dropped += sum(1 for defect in report.defects if defect.capability not in self.scope)
                report.defects = [d for d in report.defects if d.capability in self.scope]
        if not reports:
            merged = DefectReport(
                round=ctx.round,
                checkpoint_id=collect.checkpoint_id,
                capabilities=collect.capabilities,
                summary="no analyzer produced a report",
                metadata={"errors": errors},
            )
        else:
            merged = self.merger.merge(
                reports,
                round_no=ctx.round,
                checkpoint_id=collect.checkpoint_id,
                capabilities=collect.capabilities,
            )
            merged.metadata["errors"] = errors
        merged.metadata["scope"] = sorted(self.scope)
        merged.metadata["out_of_scope_defects"] = dropped
        ctx.memory["defects"] = merged
        self.runtime.artifacts.write_json(ctx.round, "analyze", "defects.json", merged)
        top = [
            {
                "capability": defect.capability,
                "subcapability": defect.subcapability,
                "severity": round(defect.severity, 4),
                "confidence": round(defect.confidence, 4),
                "source": defect.source,
            }
            for defect in merged.defects[:10]
        ]
        if self.cfg.get_bool("require_defects", False) and not merged.defects:
            return StageResult(
                "failed",
                "no defects found but analyze.require_defects=true",
                {"num_defects": 0, "errors": errors},
                error="no defects",
            )
        return StageResult(
            "ok",
            f"{len(merged.defects)} defect(s) from {', '.join(merged.sources) or 'none'}",
            {
                "num_defects": len(merged.defects),
                "top": top,
                "sources": merged.sources,
                "summary": merged.summary,
                "errors": errors,
            },
            duration_s=time.time() - started,
        )


class DataStage(Stage):
    name = "synthesize"

    def __init__(self, runtime: Any, cfg: Any) -> None:
        super().__init__(runtime, cfg)
        self.pipeline = DataPipeline(runtime)

    def validate(self) -> None:
        self.pipeline.validate()

    def run(self, ctx: LoopContext) -> StageResult:
        started = time.time()
        if not self.cfg.get_bool("enabled", True):
            return StageResult("skipped", "data.enabled=false", duration_s=0.0)
        defects = ctx.memory.get("defects")
        collect = ctx.memory["collect"]
        if defects is None:
            return StageResult("skipped", "no defect report", duration_s=0.0)
        feedback = ctx.state.history[-1].get("train", {}).get("strategy_feedback", {}) if ctx.state.history else {}
        result = self.pipeline.run(
            ctx.round, defects, collect, ctx.state, checkpoint=ctx.memory["checkpoint"],
            difficulty_shift=feedback.get("difficulty_shifts", {}),
            reward_multiplier=float(feedback.get("reward_multiplier", 1.0)),
        )
        ctx.memory["data"] = result
        ctx.memory["data_pipeline"] = self.pipeline
        ctx.memory["dataset"] = result.dataset
        dataset = result.dataset
        summary = (
            f"generated {result.generated}, kept {result.filtered}, selected {result.selected}"
            + (f", dataset {dataset.dataset_id}" if dataset else ", no dataset")
        )
        return StageResult(
            "ok",
            summary,
            {
                "dataset_id": dataset.dataset_id if dataset else None,
                "dataset_path": dataset.path if dataset else None,
                "num_samples": dataset.num_samples if dataset else 0,
                "generated": result.generated,
                "filtered": result.filtered,
                "selected": result.selected,
                "requested": result.plan.total_target,
                "replayed": result.replayed,
                "capabilities": sorted({request.capability for request in result.plan.requests}),
                "target_distribution": result.stats.get("target_distribution", {}),
                "replay": result.stats.get("replay", {}),
                "merge": result.stats.get("merge", {}),
                "mixing": result.mixing,
                "mixing_candidate": result.mixing_info.get("candidate"),
                "errors": result.errors,
            },
            duration_s=time.time() - started,
        )


class TrainStage(Stage):
    name = "train"

    def __init__(self, runtime: Any, cfg: Any) -> None:
        super().__init__(runtime, cfg)
        self.training_cfg = cfg
        self.monitor_cfg = cfg.section("monitor", required=False)
        self.supervisor_cfg = cfg.section("supervisor", required=False)
        self.debug_cfg = cfg.section("debug_agent", required=False)
        self.feedback_cfg = cfg.section("feedback", required=False)

    def validate(self) -> None:
        self.runtime.trainer()
        MonitorHub(self.runtime, self.monitor_cfg)
        Supervisor(self.runtime, self.supervisor_cfg)
        if self.debug_cfg.get_bool("enabled", False):
            prompt_id = self.debug_cfg.get("prompt")
            if prompt_id:
                self.runtime.prompt_hub.get(str(prompt_id))
            self.runtime.harness(str(self.debug_cfg.get("harness", "debug")))

    def run(self, ctx: LoopContext) -> StageResult:
        started = time.time()
        dataset = ctx.memory.get("dataset")
        if dataset is None or dataset.num_train == 0:
            return StageResult("skipped", "no training samples", duration_s=0.0)
        checkpoint = ctx.memory["checkpoint"]
        supervisor = Supervisor(self.runtime, self.supervisor_cfg)
        trainer = self.runtime.trainer()
        adjustments: dict[str, Any] = {}
        max_retries = self.supervisor_cfg.get_int("max_retries", 1)
        max_replans = max(0, self.feedback_cfg.get_int("max_replans", 1))
        feedback = {"difficulty_shifts": {}, "reward_multiplier": 1.0, "replan": False}
        strategy_history: list[dict[str, Any]] = []
        ctx.memory["gpu_hours"] = 0.0
        result = None
        findings: list[dict[str, Any]] = []
        verdict = None
        for strategy_attempt in range(max_replans + 1):
            if strategy_attempt:
                pipeline = ctx.memory.get("data_pipeline")
                if pipeline is None:
                    break
                revised = pipeline.run(
                    ctx.round, ctx.memory["defects"], ctx.memory["collect"], ctx.state,
                    checkpoint=checkpoint,
                    difficulty_shift=feedback["difficulty_shifts"],
                    reward_multiplier=feedback["reward_multiplier"],
                    attempt=strategy_attempt,
                )
                if revised.dataset is None or revised.dataset.num_train == 0:
                    strategy_history.append({
                        "attempt": strategy_attempt,
                        "status": "skipped",
                        "reason": "no training samples after strategy adjustment",
                    })
                    break
                dataset = revised.dataset
                ctx.memory["data"] = revised
                ctx.memory["dataset"] = dataset
            for attempt in range(max_retries + 1):
                monitor = MonitorHub(self.runtime, self.monitor_cfg)
                work_dir = ctx.stage_dir("train") / f"strategy-{strategy_attempt}" / f"attempt-{attempt}"
                work_dir.mkdir(parents=True, exist_ok=True)
                spec = TrainingSpec(
                    round=ctx.round,
                    dataset=dataset,
                    checkpoint=checkpoint,
                    work_dir=work_dir,
                    seed=self.runtime.seed + ctx.round + strategy_attempt * 1000003 + attempt * 1009,
                    params={"attempt": attempt, "strategy_attempt": strategy_attempt,
                            "overrides": dict(adjustments)},
                )
                result = trainer.train(spec, observer=monitor.observe)
                ctx.memory["gpu_hours"] += float(result.metadata.get("gpu_hours") or 0)
                findings = monitor.snapshot()
                verdict = supervisor.verdict(result, findings, attempt)
                atomic_write_json(
                    work_dir / "training.json",
                    {"result": to_plain(result), "findings": findings,
                     "verdict": to_plain(verdict), "adjustments": adjustments},
                )
                if result.metrics:
                    write_jsonl(work_dir / "metrics.jsonl", result.metrics)
                if not verdict.retrain or attempt >= max_retries:
                    break
                self.logger.warning(
                    "training attempt %d rejected (%s); retrying with %s",
                    attempt, verdict.reason, verdict.adjustments,
                )
                adjustments.update(verdict.adjustments)
            strategy_history.append({
                "attempt": strategy_attempt,
                "dataset_id": dataset.dataset_id,
                "status": result.status,
                "findings": findings,
                "verdict": to_plain(verdict),
            })
            if verdict is None or not verdict.use_checkpoint:
                break
            feedback = self._strategy_feedback(findings, ctx.memory.get("data"), ctx.memory["collect"])
            strategy_history[-1]["feedback"] = feedback
            if not feedback["replan"] or strategy_attempt >= max_replans:
                break

        ctx.memory["training"] = result
        ctx.memory["train_findings"] = findings
        ctx.memory["train_verdict"] = verdict
        self.runtime.artifacts.write_json(
            ctx.round, "train", "train.json",
            {"result": to_plain(result), "findings": findings,
             "verdict": to_plain(verdict), "strategy": strategy_history},
        )
        if verdict is not None and not verdict.use_checkpoint:
            if self.debug_cfg.get_bool("enabled", False):
                self._debug_with_harness(ctx, result, findings)
            return StageResult(
                "ok", f"no usable checkpoint: {verdict.reason}",
                {"status": result.status, "candidate": None, "findings": len(findings),
                 "reason": verdict.reason, "strategy": strategy_history},
                duration_s=time.time() - started,
            )
        if result is None or not result.checkpoint_path:
            return StageResult(
                "failed", "trainer produced no checkpoint path",
                {"status": result.status if result else "unknown", "findings": len(findings)},
                error="missing checkpoint", duration_s=time.time() - started,
            )
        metrics: dict[str, Any] = {
            "train_status": result.status, "train_steps": result.num_steps,
            "best_step": result.best_step, "dataset_id": dataset.dataset_id,
            "findings": len(findings),
        }
        if result.best_metric:
            metrics["train_best"] = result.best_metric.get("metrics", {})
        candidate = self.runtime.checkpoints.reserve(
            model_path=result.checkpoint_path, round_no=ctx.round, metrics=metrics,
            source_job=result.job_id, parent_id=checkpoint.id,
            tags=["trained", f"round-{ctx.round}"],
        )
        ctx.memory["candidate"] = candidate
        if self.debug_cfg.get_bool("enabled", False) and (
            result.status in {"failed", "timeout"}
            or any(f.get("severity") == "critical" for f in findings)
        ):
            self._debug_with_harness(ctx, result, findings)
        return StageResult(
            "ok", f"trained {result.num_steps} steps, candidate {candidate.id}",
            {"status": result.status, "candidate": candidate.id,
             "checkpoint_path": candidate.model_path, "steps": result.num_steps,
             "best_step": result.best_step, "findings": findings,
             "reason": verdict.reason if verdict else "", "strategy": strategy_history,
             "strategy_feedback": feedback},
            duration_s=time.time() - started,
        )

    def _strategy_feedback(self, findings: list[dict[str, Any]], data: Any, collect: Any) -> dict[str, Any]:
        codes = {finding.get("code") for finding in findings}
        targets = sorted({request.capability for request in data.plan.requests}) if data else []
        threshold = self.feedback_cfg.get_float("high_pass_rate", 0.95)
        high_pass = [
            name for name in targets
            if getattr(collect.capabilities.get(name), "pass_rate", 0.0) >= threshold
        ]
        shift_all = bool(codes & {"saturation", "val_plateau"})
        shift = self.feedback_cfg.get_float("difficulty_step", 0.1)
        max_shift = self.feedback_cfg.get_float("max_difficulty_shift", 0.3)
        existing = getattr(self, "_difficulty_shifts", {})
        shifts = dict(existing)
        for name in (high_pass or targets if shift_all else high_pass):
            shifts[name] = min(max_shift, shifts.get(name, 0.0) + shift)
        reward_multiplier = min(
            self.feedback_cfg.get_float("max_reward_multiplier", 3.0),
            getattr(self, "_reward_multiplier", 1.0) * self.feedback_cfg.get_float(
                "reward_hacking_weight_multiplier", 1.5
            ),
        ) if "reward_hacking" in codes else getattr(self, "_reward_multiplier", 1.0)
        self._difficulty_shifts = shifts
        self._reward_multiplier = reward_multiplier
        return {
            "difficulty_shifts": shifts,
            "reward_multiplier": reward_multiplier,
            "replan": bool(shifts != existing or reward_multiplier > 1.0),
        }

    def _debug_with_harness(self, ctx: LoopContext, result: Any, findings: list[dict[str, Any]]) -> None:
        prompt_id = self.debug_cfg.get("prompt")
        if not prompt_id:
            self.logger.warning("train.debug_agent.enabled=true but train.debug_agent.prompt is missing")
            return
        try:
            rendered = self.runtime.prompt_hub.render(
                str(prompt_id),
                {
                    "round": ctx.round,
                    "status": result.status,
                    "error": result.error or "",
                    "metrics": (result.metrics[-20:] if result.metrics else []),
                    "findings": findings,
                    "dataset": to_plain(result.metadata),
                },
            )
            from rsi.core.types import HarnessTask

            work_dir = ctx.stage_dir("train") / "debug"
            task = HarnessTask(
                name="train_debug",
                instruction=rendered.text,
                work_dir=work_dir,
                skills=[str(v) for v in self.debug_cfg.get_list("skills", ["train-debug"])],
                output_schema=rendered.schema,
                metadata={"round": ctx.round, "status": result.status},
            )
            debug = self.runtime.harness(str(self.debug_cfg.get("harness", "debug"))).run(task)
            self.runtime.artifacts.write_json(
                ctx.round, "train", "debug_advice.json", debug
            )
        except Exception as exc:  # noqa: BLE001 - advisory only
            self.logger.warning("train debug harness failed: %s", exc)


class ValidateStage(Stage):
    name = "validation"

    def __init__(self, runtime: Any, cfg: Any) -> None:
        super().__init__(runtime, cfg)
        self.mode = str(cfg.get("mode", "auto"))
        self.metric = str(cfg.get("metric", "mean_score"))
        self.min_delta = cfg.get_float("min_delta", 0.0)
        self.allow_regression = cfg.get_bool("allow_regression", False)
        self.target_min_delta = cfg.get_float("target_min_delta", 0.0)
        self.max_other_regression = cfg.get_float("max_other_regression", 0.01)
        self.names = [str(v) for v in cfg.get_list("benchmark.names", [])]
        if not self.names and runtime.config.get_bool("collect.benchmark.enabled", True):
            self.names = [str(v) for v in runtime.config.get_list("collect.benchmark.names", [])]

    def validate(self) -> None:
        if self.mode not in {"auto", "benchmark", "metrics"}:
            raise ValueError(f"unknown validate.mode: {self.mode}")
        if self.mode == "benchmark" and not self.names:
            raise ValueError("benchmark validation requires benchmark names")
        for name in self.names:
            self.runtime.benchmark(name)

    def run(self, ctx: LoopContext) -> StageResult:
        candidate = ctx.memory.get("candidate")
        if candidate is None:
            return StageResult("skipped", "no candidate checkpoint")
        use_benchmark = self.mode == "benchmark" or (
            self.mode == "auto" and bool(self.names) and (
                self.runtime.inference().supports_checkpoint_switch()
                or self.cfg.get_bool("force_benchmark", False)
            )
        )
        if use_benchmark:
            result = self.evaluate(ctx, candidate)
        else:
            training = ctx.memory["training"]
            best = (training.best_metric or {}).get("metrics", {})
            score = best.get(self.metric, best.get("val_score"))
            incumbent = ctx.memory["checkpoint"]
            previous = incumbent.metrics.get("validation_score")
            baseline = previous if (
                incumbent.metrics.get("validation_source") == "trainer_metrics"
                and incumbent.metrics.get("validation_metric") == self.metric
            ) else None
            result = self.decision(score, baseline)
            result.update(source="trainer_metrics", capabilities={}, aggregate={"val_score": score})
        self.record(ctx, candidate, result, self.name)
        ctx.memory["validation"] = result
        return StageResult("ok", result["reason"], result)

    def evaluate(
        self, ctx: LoopContext, candidate: Any, *, stage: str = "validation",
        baseline_checkpoint: Any = None,
    ) -> dict[str, Any]:
        if not self.names:
            raise ValueError("benchmark validation requires benchmark names")
        incumbent = baseline_checkpoint or ctx.memory["checkpoint"]
        reports = {report.benchmark: report for report in ctx.memory["collect"].reports}
        if all(reports.get(name) and reports[name].checkpoint_id == incumbent.id for name in self.names):
            baseline_reports = [reports[name] for name in self.names]
        else:
            self.runtime.inference().ensure(incumbent)
            baseline_reports = [
                self.runtime.benchmark(name).evaluate(incumbent, ctx.round) for name in self.names
            ]
        self.runtime.inference().ensure(candidate)
        candidate_reports = []
        baseline_by_name = dict(zip(self.names, baseline_reports))
        for name in self.names:
            report = self.runtime.benchmark(name).evaluate(candidate, ctx.round)
            candidate_reports.append(report)
            suffix = stable_hash(name, length=8)
            self.runtime.artifacts.write_json(ctx.round, stage, f"benchmark-{suffix}.json", report)
            self.runtime.artifacts.write_json(
                ctx.round, stage, f"baseline-{suffix}.json", baseline_by_name[name]
            )

        base_table = capability_table(baseline_reports, catalog=self.runtime.catalog)
        table = capability_table(candidate_reports, catalog=self.runtime.catalog)
        baseline = overall_metrics(baseline_reports, base_table)
        aggregate = overall_metrics(candidate_reports, table)
        result = self.decision(aggregate.get(self.metric), baseline.get(self.metric))
        data = ctx.memory.get("data")
        targets = sorted({request.capability for request in data.plan.requests}) if data else []
        deltas = {name: rec.score - base_table[name].score for name, rec in table.items()
                  if name in base_table}
        unmeasured_targets = [name for name in targets if name not in deltas]
        failed_targets = [
            name for name in targets
            if name in deltas and (deltas[name] <= 0 or deltas[name] < self.target_min_delta)
        ]
        improved_targets = [name for name in targets if name not in failed_targets and name not in unmeasured_targets]
        unresolved = failed_targets + unmeasured_targets
        regressions = [name for name in sorted(set(base_table) - set(targets))
                       if name not in deltas or deltas[name] < -self.max_other_regression]
        comparable = all(
            old.checkpoint_id == incumbent.id
            and new.checkpoint_id == candidate.id
            and bool(old.items)
            and sorted(
                (item.item_id, item.capability, item.subcapability) for item in old.items
            )
            == sorted(
                (item.item_id, item.capability, item.subcapability) for item in new.items
            )
            and len(old.items) == len(new.items)
            and all(
                not item.error and math.isfinite(item.score) and 0 <= item.score <= 1
                for item in old.items + new.items
            )
            for old, new in zip(baseline_reports, candidate_reports)
        )
        if not comparable:
            result.update(accepted=False, reason="benchmark coverage or results are incomplete")
        elif not improved_targets:
            result.update(accepted=False, reason="no target capability improved: " + ", ".join(unresolved))
        elif regressions and not self.allow_regression:
            result.update(accepted=False, reason="other capabilities regressed: " + ", ".join(regressions))
        elif unresolved:
            result["reason"] += "; carried to the next round: " + ", ".join(unresolved)
        result.update(source="benchmark", capabilities=_cap_scores(table), aggregate=aggregate,
                      targets=targets, capability_deltas=deltas, failed_targets=failed_targets,
                      unmeasured_targets=unmeasured_targets, improved_targets=improved_targets,
                      regressions=regressions)
        return result

    def decision(self, score: float | None, baseline: float | None) -> dict[str, Any]:
        if (
            not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not math.isfinite(score)
            or (
                baseline is not None
                and (
                    not isinstance(baseline, (int, float))
                    or isinstance(baseline, bool)
                    or not math.isfinite(baseline)
                )
            )
        ):
            return {"score": None, "baseline": baseline, "delta": None,
                    "accepted": False, "reason": f"missing or invalid metric {self.metric}"}
        delta = score - baseline if baseline is not None else None
        accepted = score > 0 and (
            delta is None or delta >= self.min_delta or self.allow_regression
        )
        return {"score": score, "baseline": baseline, "delta": delta, "accepted": accepted,
                "reason": f"{self.metric} {score:.4f}: {'accepted' if accepted else 'insufficient improvement'}"}

    def record(self, ctx: LoopContext, candidate: Any, result: dict[str, Any], stage: str) -> None:
        result.update(metric=self.metric, checkpoint_id=candidate.id)
        self.runtime.checkpoints.update_metrics(candidate.id, {
            "validation_score": result["score"], "validation_source": result["source"],
            "validation_metric": self.metric, "accepted": result["accepted"],
        })
        if result.get("source") == "benchmark":
            carried = [
                name for name in getattr(ctx.state, "pending_capabilities", None) or []
                if name not in result["improved_targets"]
            ]
            ctx.state.pending_capabilities = sorted(
                set(carried) | set(result["failed_targets"]) | set(result["unmeasured_targets"])
            )
        self.runtime.artifacts.write_json(ctx.round, stage, "validation.json", result)


class DistillStage(Stage):
    name = "opd"

    def __init__(self, runtime: Any, cfg: Any, validation: ValidateStage) -> None:
        super().__init__(runtime, cfg)
        self.validation = validation
        self.enabled = cfg.get_bool("enabled", False)
        spec = cfg.plugin("trainer", "mock") if self.enabled else None
        self.trainer = runtime.build(
            "trainer", spec, cfg.section("trainer", required=False)
        ) if spec else None
        self.trainer_type = str(spec["type"]) if spec else ""

    def validate(self) -> None:
        if not self.enabled:
            return
        if not self.validation.names or self.validation.mode == "metrics":
            raise ValueError("OPD requires benchmark validation")
        if self.trainer_type not in {"command", "verl", "slime"}:
            raise ValueError("OPD requires a command trainer configured for teacher-model distillation")
        command = self.cfg.get("trainer.command", [])
        rendered = " ".join(map(str, command)) if isinstance(command, list) else str(command)
        if "{teacher_model_path}" not in rendered:
            raise ValueError("OPD trainer.command must pass {teacher_model_path} to its distillation entry point")

    def run(self, ctx: LoopContext) -> StageResult:
        teacher = ctx.memory.get("candidate")
        validation = ctx.memory.get("validation") or {}
        if not self.enabled or not validation.get("accepted"):
            return StageResult("skipped", "OPD requires an accepted candidate and opd.enabled")
        if validation["source"] != "benchmark":
            raise ValueError("OPD requires benchmark validation; set validate.mode=benchmark")
        checkpoint = ctx.memory["checkpoint"]
        result = self.trainer.train(TrainingSpec(
            round=ctx.round, dataset=ctx.memory["dataset"], checkpoint=checkpoint,
            work_dir=ctx.stage_dir(self.name), seed=self.runtime.seed + ctx.round,
            teacher=teacher,
        ))
        self.runtime.artifacts.write_json(ctx.round, self.name, "training.json", result)
        ctx.memory["gpu_hours"] = ctx.memory.get("gpu_hours", 0.0) + float(result.metadata.get("gpu_hours") or 0)
        if result.status != "completed" or not result.checkpoint_path:
            reason = "OPD produced no usable checkpoint; retaining the validated candidate"
            return StageResult("skipped", reason, {"candidate": teacher.id, "reason": reason})
        merged = self.runtime.checkpoints.reserve(
            model_path=result.checkpoint_path, round_no=ctx.round, parent_id=checkpoint.id,
            source_job=result.job_id, tags=["opd"], metrics={"teacher_id": teacher.id},
        )
        merged_validation = self.validation.evaluate(
            ctx, merged, stage=self.name, baseline_checkpoint=teacher
        )
        self.validation.record(ctx, merged, merged_validation, self.name)
        if merged_validation["accepted"]:
            self.runtime.checkpoints.archive_candidate(teacher.id)
            ctx.memory["candidate"] = merged
            ctx.memory["validation"] = merged_validation
            return StageResult("ok", merged_validation["reason"], merged_validation)
        self.runtime.checkpoints.reject(merged.id, merged_validation["reason"])
        reason = f"OPD rejected: {merged_validation['reason']}; retaining the validated candidate"
        ctx.memory["candidate"] = teacher
        ctx.memory["validation"] = validation
        return StageResult("ok", reason, {
            **merged_validation, "accepted": True, "checkpoint_id": teacher.id,
            "opd_candidate": merged.id, "reason": reason,
        })

    def close(self) -> None:
        if self.trainer is not None:
            self.trainer.close()


class PromoteStage(Stage):
    name = "promote"

    def run(self, ctx: LoopContext) -> StageResult:
        started = time.time()
        candidate = ctx.memory.get("candidate")
        validation = ctx.memory.get("validation")
        if candidate is None or validation is None:
            return StageResult("skipped", "nothing to promote", duration_s=0.0)
        promoted = False
        if validation.get("accepted"):
            self.runtime.checkpoints.promote(candidate.id)
            ctx.state.best_checkpoint = candidate.id
            ctx.state.best_score = max(ctx.state.best_score, float(validation["score"]))
            promoted = True
            reason = "accepted"
        else:
            self.runtime.checkpoints.reject(candidate.id, validation["reason"])
            reason = "rejected"
        payload = {
            "promoted": promoted,
            "checkpoint": candidate.id if promoted else None,
            "candidate": candidate.id,
            "best_checkpoint": ctx.state.best_checkpoint,
            "best_score": ctx.state.best_score,
            "reason": reason,
            "validation_reason": validation["reason"],
        }
        ctx.memory["checkpoint_out"] = candidate.id if promoted else None
        self.runtime.artifacts.write_json(ctx.round, "promote", "promotion.json", payload)
        return StageResult(
            "ok",
            f"{reason} {candidate.id}; best={ctx.state.best_checkpoint} ({ctx.state.best_score:.4f})",
            payload,
            duration_s=time.time() - started,
        )
