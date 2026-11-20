from __future__ import annotations

import time
from typing import Any

from rsi.core.jsonio import atomic_write_json, now_iso, stable_hash, to_plain, write_jsonl
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
        result = self.pipeline.run(ctx.round, defects, collect, ctx.state)
        ctx.memory["data"] = result
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
        if dataset is None or getattr(dataset, "num_samples", 0) == 0:
            return StageResult("skipped", "no training samples", duration_s=0.0)
        checkpoint = ctx.memory["checkpoint"]
        supervisor = Supervisor(self.runtime, self.supervisor_cfg)
        trainer = self.runtime.trainer()
        adjustments: dict[str, Any] = {}
        attempt = 0
        max_retries = self.supervisor_cfg.get_int("max_retries", 1)
        result = None
        findings: list[dict[str, Any]] = []
        verdict = None
        while True:
            monitor = MonitorHub(self.runtime, self.monitor_cfg)
            work_dir = ctx.stage_dir("train") / f"attempt-{attempt}"
            spec = TrainingSpec(
                round=ctx.round,
                dataset=dataset,
                checkpoint=checkpoint,
                work_dir=work_dir,
                seed=self.runtime.seed + ctx.round + attempt * 1009,
                params={"attempt": attempt, "overrides": dict(adjustments)},
            )
            result = trainer.train(spec, observer=monitor.observe)
            findings = monitor.snapshot()
            verdict = supervisor.verdict(result, findings, attempt)
            atomic_write_json(
                work_dir / "training.json",
                {
                    "result": to_plain(result),
                    "findings": findings,
                    "verdict": to_plain(verdict),
                    "adjustments": adjustments,
                },
            )
            if result.metrics:
                write_jsonl(work_dir / "metrics.jsonl", result.metrics)
            if not verdict.retrain or attempt >= max_retries:
                break
            self.logger.warning(
                "training attempt %d rejected (%s); retrying with %s",
                attempt,
                verdict.reason,
                verdict.adjustments,
            )
            adjustments.update(verdict.adjustments)
            attempt += 1
        ctx.memory["training"] = result
        ctx.memory["train_findings"] = findings
        ctx.memory["train_verdict"] = verdict
        self.runtime.artifacts.write_json(
            ctx.round,
            "train",
            "train.json",
            {"result": to_plain(result), "findings": findings, "verdict": to_plain(verdict)},
        )

        if verdict is not None and not verdict.use_checkpoint:
            if self.debug_cfg.get_bool("enabled", False):
                self._debug_with_harness(ctx, result, findings)
            return StageResult(
                "ok",
                f"no usable checkpoint: {verdict.reason}",
                {
                    "status": result.status,
                    "candidate": None,
                    "findings": len(findings),
                    "reason": verdict.reason,
                    "attempts": attempt + 1,
                },
                duration_s=time.time() - started,
            )
        if result is None or not result.checkpoint_path:
            return StageResult(
                "failed",
                "trainer produced no checkpoint path",
                {"status": result.status if result else "unknown", "findings": len(findings)},
                error="missing checkpoint",
                duration_s=time.time() - started,
            )
        metrics: dict[str, Any] = {
            "train_status": result.status,
            "train_steps": result.num_steps,
            "best_step": result.best_step,
            "dataset_id": dataset.dataset_id,
            "findings": len(findings),
        }
        if result.best_metric:
            metrics["train_best"] = result.best_metric.get("metrics", {})
        candidate = self.runtime.checkpoints.reserve(
            model_path=result.checkpoint_path,
            round_no=ctx.round,
            metrics=metrics,
            source_job=result.job_id,
            parent_id=checkpoint.id,
            tags=["trained", f"round-{ctx.round}"],
        )
        ctx.memory["candidate"] = candidate
        if self.debug_cfg.get_bool("enabled", False) and (
            result.status in {"failed", "timeout"} or any(f.get("severity") == "critical" for f in findings)
        ):
            self._debug_with_harness(ctx, result, findings)
        return StageResult(
            "ok",
            f"trained {result.num_steps} steps, candidate {candidate.id} ({verdict.reason if verdict else ''})",
            {
                "status": result.status,
                "candidate": candidate.id,
                "checkpoint_path": candidate.model_path,
                "steps": result.num_steps,
                "best_step": result.best_step,
                "findings": findings,
                "reason": verdict.reason if verdict else "",
                "attempts": attempt + 1,
            },
            duration_s=time.time() - started,
        )

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

    def validate(self) -> None:
        for name in self.cfg.get_list("benchmark.names", []):
            self.runtime.benchmark(str(name))

    def run(self, ctx: LoopContext) -> StageResult:
        started = time.time()
        candidate = ctx.memory.get("candidate")
        if candidate is None:
            return StageResult("skipped", "no candidate checkpoint", duration_s=0.0)
        training = ctx.memory.get("training")
        collect = ctx.memory["collect"]
        use_benchmark = self.mode == "benchmark"
        if self.mode == "auto":
            use_benchmark = self.runtime.inference().supports_checkpoint_switch() or self.cfg.get_bool(
                "force_benchmark", False
            )
        source = "metrics"
        reports = []
        table = {}
        aggregate: dict[str, float] = {}
        score = 0.0
        if use_benchmark:
            names = [str(v) for v in self.cfg.get_list("benchmark.names", [])] or list(
                collect.metadata.get("benchmark_names", [])
            )
            if names:
                self.runtime.inference().ensure(candidate)
                for name in names:
                    reports.append(self.runtime.benchmark(name).evaluate(candidate, ctx.round))
                table = capability_table(reports, collect.baseline_capabilities, self.runtime.catalog)
                aggregate = overall_metrics(reports, table)
                score = float(aggregate.get(self.metric, aggregate.get("mean_score", 0.0)))
                source = "benchmark"
                for report in reports:
                    self.runtime.artifacts.write_json(
                        ctx.round, self.name, f"benchmark-{stable_hash(report.benchmark, length=8)}.json", report
                    )
        if source == "metrics":
            best = training.best_metric if training is not None else None
            if best and isinstance(best.get("metrics"), dict):
                score = float(best["metrics"].get(self.metric, best["metrics"].get("val_score", 0.0)))
            source = "trainer_metrics"
            aggregate = {"val_score": score}
        baseline = ctx.state.best_score
        if baseline < 0:
            baseline = float(collect.aggregate.get("mean_score", 0.0))
        delta = score - baseline
        accepted = delta >= self.min_delta
        if self.allow_regression:
            accepted = accepted or score > 0
        accepted = bool(accepted and score > 0)
        validation = {
            "source": source,
            "score": score,
            "baseline": baseline,
            "delta": delta,
            "accepted": accepted,
            "metric": self.metric,
            "capabilities": _cap_scores(table),
            "aggregate": aggregate,
            "checkpoint_id": candidate.id,
        }
        self.runtime.checkpoints.update_metrics(
            candidate.id,
            {"validation_score": score, "validation_source": source, "accepted": accepted},
        )
        ctx.memory["validation"] = validation
        self.runtime.artifacts.write_json(ctx.round, self.name, "validation.json", validation)
        return StageResult(
            "ok",
            f"{source} score {score:.4f} ({delta:+.4f} vs {baseline:.4f}) -> "
            f"{'accept' if accepted else 'reject'}",
            validation,
            duration_s=time.time() - started,
        )


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
            self.runtime.checkpoints.reject(candidate.id, "validation rejected")
            reason = "rejected"
        payload = {
            "promoted": promoted,
            "checkpoint": candidate.id if promoted else None,
            "candidate": candidate.id,
            "best_checkpoint": ctx.state.best_checkpoint,
            "best_score": ctx.state.best_score,
            "reason": reason,
        }
        ctx.memory["checkpoint_out"] = candidate.id if promoted else None
        self.runtime.artifacts.write_json(ctx.round, "promote", "promotion.json", payload)
        return StageResult(
            "ok",
            f"{reason} {candidate.id}; best={ctx.state.best_checkpoint} ({ctx.state.best_score:.4f})",
            payload,
            duration_s=time.time() - started,
        )
