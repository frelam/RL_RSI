from __future__ import annotations

import time
from typing import Any

from rsi.core.jsonio import now_iso, to_plain
from rsi.core.registry import register
from rsi.core.types import LoopResult, StageResult
from rsi.loops.base import Loop, LoopContext, Stage
from rsi.loops.stages import (
    AnalyzeStage,
    CollectStage,
    DataStage,
    DistillStage,
    PromoteStage,
    SyncStage,
    TrainStage,
    ValidateStage,
)
from rsi.loops.state import LoopState, load_state, new_state, save_state, state_path
from rsi.loops.stop import StopEvaluator


@register("loop", "classic")
class ClassicLoop(Loop):
    """The README loop: collect -> analyze -> synthesize -> train -> validate -> promote."""

    name = "classic"

    def __init__(self, runtime: Any, cfg: Any, **params: Any) -> None:
        super().__init__(runtime, cfg, **params)
        self.stops = StopEvaluator(runtime, runtime.config.section("stop"))
        validation = ValidateStage(runtime, runtime.config.section("validate"))
        opd_cfg = runtime.config.child(runtime.config.get_dict("opd", {}), "opd")
        self.distill = DistillStage(runtime, opd_cfg, validation)
        self.stages: list[Stage] = [
            SyncStage(runtime, cfg),
            CollectStage(runtime, runtime.config.section("collect")),
            AnalyzeStage(runtime, runtime.config.section("analyze")),
            DataStage(runtime, runtime.config.section("data")),
            TrainStage(runtime, runtime.config.section("train")),
            validation,
            self.distill,
            PromoteStage(runtime, runtime.config.section("checkpoint")),
        ]
        self.on_stage_error = str(cfg.get("on_stage_error", "abort"))
        self.max_stage_retries = cfg.get_int("max_stage_retries", 1)
        self.overwrite = cfg.get_bool("overwrite", False)

    def validate(self) -> None:
        self.runtime.inference()
        self.runtime.trainer()
        for stage in self.stages:
            stage.validate()

    def run(self, *, resume: bool = False) -> LoopResult:
        try:
            return self._run(resume=resume)
        finally:
            self.distill.close()

    def _run(self, *, resume: bool) -> LoopResult:
        path = state_path(self.runtime)
        if path.exists() and resume:
            state = load_state(path)
            if state.status == "stopped":
                self.runtime.logger.info(
                    "resuming a stopped run: %s", state.stop_reason
                )
            state.status = "running"
            self.runtime.logger.info(
                "resuming run at round %d (best %s @ %.4f)",
                state.round,
                state.best_checkpoint,
                state.best_score,
            )
        elif path.exists() and not self.overwrite:
            raise RuntimeError(
                f"Run directory already contains state: {self.runtime.run_dir}. "
                "Use --resume or a new --run-id."
            )
        else:
            state = new_state(self.runtime.run_id, self.name, self.runtime.seed)
        save_state(self.runtime, state)
        started = time.monotonic()
        previous_wall_hours = float(state.budget.get("wall_hours") or 0.0)

        pre = self.stops.evaluate(state, {})
        if pre.should_stop and state.round > 0:
            return self._finalize(state, pre.reason, pre.details)

        while True:
            round_no = state.round
            round_dir = self.runtime.artifacts.round_dir(round_no)
            ctx = LoopContext(self.runtime, state, round_no, round_dir)
            record: dict[str, Any] = {
                "round": round_no,
                "started_at": now_iso(),
                "stages": {},
                "checkpoint_in": None,
            }
            for stage in self.stages:
                result = self._run_stage(stage, ctx)
                record["stages"][stage.name] = {
                    "status": result.status,
                    "summary": result.summary,
                    "duration_s": round(result.duration_s, 3),
                    "error": result.error,
                }
                if result.data:
                    record[stage.name] = to_plain(result.data)
                if stage.name == "sync" and result.status == "ok":
                    record["checkpoint_in"] = result.data["checkpoint_id"]
                if result.status == "failed":
                    self.runtime.events.emit(
                        "stage_failed",
                        round=round_no,
                        stage=stage.name,
                        error=result.error,
                    )
                    if self.on_stage_error == "skip":
                        self.runtime.logger.warning(
                            "stage %s failed; continuing because loop.on_stage_error=skip", stage.name
                        )
                        continue
                    state.status = "failed"
                    state.stop_reason = f"stage {stage.name} failed: {result.error or result.summary}"
                    save_state(self.runtime, state)
                    self.runtime.artifacts.write_json(round_no, "loop", "failed_round.json", record)
                    return self._result(state, "failed")
            record["finished_at"] = now_iso()
            record["checkpoint_out"] = ctx.memory.get("checkpoint_out")
            state.round += 1
            state.budget["wall_hours"] = previous_wall_hours + (time.monotonic() - started) / 3600.0
            state.budget["gpu_hours"] = float(state.budget.get("gpu_hours") or 0.0) + ctx.memory.get(
                "gpu_hours", 0.0
            )
            decision = self.stops.evaluate(state, record)
            record["stop"] = to_plain(decision)
            state.history.append(record)
            save_state(self.runtime, state)
            self.runtime.events.emit(
                "round_completed",
                round=round_no,
                checkpoint_in=record.get("checkpoint_in"),
                checkpoint_out=record.get("checkpoint_out"),
                best_score=state.best_score,
                stop=decision.should_stop,
                stop_reason=decision.reason,
            )
            if decision.should_stop:
                return self._finalize(state, decision.reason, decision.details)

    def _run_stage(self, stage: Stage, ctx: LoopContext) -> StageResult:
        last: StageResult | None = None
        for attempt in range(self.max_stage_retries + 1):
            started = time.monotonic()
            try:
                result = stage.run(ctx)
            except Exception as exc:  # noqa: BLE001 - stage boundary
                self.runtime.logger.exception("stage %s crashed", stage.name)
                result = StageResult(
                    "failed",
                    f"{type(exc).__name__}: {exc}",
                    error=str(exc),
                    duration_s=time.monotonic() - started,
                )
            if result.duration_s == 0:
                result.duration_s = time.monotonic() - started
            if result.status != "failed":
                return result
            last = result
            if attempt < self.max_stage_retries:
                self.runtime.logger.warning(
                    "stage %s failed (attempt %d/%d): %s",
                    stage.name,
                    attempt + 1,
                    self.max_stage_retries + 1,
                    result.summary,
                )
        return last or StageResult("failed", "stage produced no result")

    def _finalize(self, state: LoopState, reason: str, details: dict[str, Any]) -> LoopResult:
        state.status = "stopped"
        state.stop_reason = reason
        save_state(self.runtime, state)
        self.runtime.events.emit("run_stopped", reason=reason, details=details, best_score=state.best_score)
        return self._result(state, "stopped")

    def _result(self, state: LoopState, status: str) -> LoopResult:
        return LoopResult(
            status=status,
            stop_reason=state.stop_reason,
            rounds=state.round,
            best_checkpoint=state.best_checkpoint,
            best_score=state.best_score,
            run_dir=str(self.runtime.run_dir),
            metadata={"history": state.history},
        )
