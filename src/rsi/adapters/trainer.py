from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Callable

from rsi.adapters.templating import format_template
from rsi.core.errors import AdapterError
from rsi.core.interfaces import TrainerBackend
from rsi.core.jsonio import atomic_write_json, now_iso, read_jsonl, write_jsonl
from rsi.core.registry import register
from rsi.core.types import TrainingResult, TrainingSpec


def _notify(observer: Callable[[dict[str, Any]], Any] | None, record: dict[str, Any]) -> str | None:
    if observer is None:
        return None
    try:
        action = observer(record)
    except Exception:  # noqa: BLE001 - monitoring must not kill training
        return None
    return str(action) if action is not None else None


@register("trainer", "mock")
class MockTrainer(TrainerBackend):
    """Deterministic trainer with controllable anomalies for testing monitors."""

    def __init__(self, runtime: Any, cfg: Any, **params: Any) -> None:
        super().__init__(runtime, cfg, **params)
        self.steps = self.cfg.get_int("steps", 40)
        self.eval_every = max(1, self.cfg.get_int("eval_every", 10))
        self.sleep_s = self.cfg.get_float("sleep_s", 0.0)
        self.noise = self.cfg.get_float("noise", 0.01)
        self.val_base = self.cfg.get_float("val_base", 0.42)
        self.val_gain = self.cfg.get_float("val_gain", 0.08)
        self.entropy_start = self.cfg.get_float("entropy_start", 1.3)
        self.entropy_decay = self.cfg.get_float("entropy_decay", 0.01)
        self.best_metric = str(self.cfg.get("best_metric", "val_score"))
        self.gpu_hours_per_step = self.cfg.get_float("gpu_hours_per_step", 0.0)
        self.inject = self.cfg.get_dict("inject", {})

    def train(self, spec: TrainingSpec, observer: Callable[[dict[str, Any]], Any] | None = None) -> TrainingResult:
        started = time.time()
        rng = self.runtime.rng("trainer", spec.round, spec.checkpoint.id)
        metrics: list[dict[str, Any]] = []
        best: dict[str, Any] | None = None
        best_step: int | None = None
        status = "completed"
        val_score = self.val_base + self.val_gain * (spec.round + 1)
        entropy_collapse = int(self.inject.get("entropy_collapse_step", 0) or 0)
        grad_spike = int(self.inject.get("grad_spike_step", 0) or 0)
        nan_step = int(self.inject.get("nan_step", 0) or 0)

        for step in range(1, self.steps + 1):
            progress = step / max(1, self.steps)
            entropy = max(0.03, self.entropy_start - self.entropy_decay * step + self.noise * rng.uniform(-1, 1))
            if entropy_collapse and step >= entropy_collapse:
                entropy = min(entropy, 0.05)
            loss = max(0.01, 1.4 - 1.0 * progress + self.noise * rng.uniform(-1, 1))
            grad_norm = max(0.01, 0.9 + 0.3 * rng.random())
            reward = 0.15 + 0.5 * progress + self.noise * rng.uniform(-1, 1)
            if step % self.eval_every == 0 or step == 1:
                val_score = self.val_base + self.val_gain * (spec.round + 1) + 0.02 * rng.uniform(-1, 1)
            if grad_spike and step >= grad_spike:
                grad_norm = 40.0
            if nan_step and step >= nan_step:
                loss = float("nan")
            record = {
                "step": step,
                "metrics": {
                    "loss": loss,
                    "grad_norm": grad_norm,
                    "entropy": entropy,
                    "kl": 0.02 + 0.01 * rng.random(),
                    "reward": reward,
                    "val_score": val_score,
                    "response_length": 400 + 60 * rng.random(),
                    "lr": 1e-6,
                },
            }
            if self.gpu_hours_per_step:
                record["metrics"]["gpu_hours"] = step * self.gpu_hours_per_step
            metrics.append(record)
            value = record["metrics"].get(self.best_metric)
            if value is not None and (best is None or float(value) > float(best["metrics"].get(self.best_metric, -math.inf))):
                best = record
                best_step = step
            if _notify(observer, record) == "stop":
                status = "stopped_by_monitor"
                break
            if self.sleep_s:
                time.sleep(self.sleep_s)

        checkpoint_dir = spec.work_dir / "checkpoint"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            checkpoint_dir / "model.json",
            {
                "format": "mock-checkpoint",
                "source": spec.checkpoint.model_path,
                "source_id": spec.checkpoint.id,
                "round": spec.round,
                "steps": len(metrics),
                "seed": spec.seed,
                "created_at": now_iso(),
            },
        )
        metrics_path = spec.work_dir / "metrics.jsonl"
        write_jsonl(metrics_path, metrics)
        duration = time.time() - started
        if self.gpu_hours_per_step and metrics:
            duration_gpu_hours = len(metrics) * self.gpu_hours_per_step
        else:
            duration_gpu_hours = 0.0
        return TrainingResult(
            job_id=f"mock-train-r{spec.round:04d}",
            status=status,
            checkpoint_path=str(checkpoint_dir.resolve()),
            metrics=metrics,
            best_metric=best,
            best_step=best_step,
            num_steps=len(metrics),
            duration_s=duration,
            log_path=str(metrics_path),
            metadata={
                "dataset_id": spec.dataset.dataset_id,
                "checkpoint_in": spec.checkpoint.id,
                "gpu_hours": duration_gpu_hours,
                "trainer": "mock",
            },
        )


@register("trainer", "command")
class CommandTrainer(TrainerBackend):
    """Run a real RL trainer via command template and stream a JSONL metrics file.

    Config::

        trainer:
          type: verl
          command: [python, -m, verl.trainer.main_ppo, ...]
          output_dir: "{work_dir}/checkpoint"
          metrics_path: "{work_dir}/metrics.jsonl"
          poll_interval_s: 5
          timeout_s: 86400

    Placeholders: ``{dataset_dir}``, ``{train_path}``, ``{val_path}``,
    ``{reward_spec}``, ``{model_path}``, ``{checkpoint_id}``, ``{output_dir}``,
    ``{metrics_path}``, ``{round}``, ``{run_dir}``, ``{work_dir}``, ``{seed}``.
    The metrics file must contain one JSON object per line with a ``step`` and
    either top-level metrics or a nested ``metrics`` object.
    """

    def __init__(self, runtime: Any, cfg: Any, **params: Any) -> None:
        super().__init__(runtime, cfg, **params)
        self.command = self.cfg.get("command")
        if not self.command:
            raise AdapterError("command trainer requires 'command'")
        self.output_dir_template = str(self.cfg.get("output_dir", "{work_dir}/checkpoint"))
        self.metrics_path_template = str(self.cfg.get("metrics_path", "{work_dir}/metrics.jsonl"))
        self.poll_interval_s = self.cfg.get_float("poll_interval_s", 5.0)
        self.timeout_s = self.cfg.get_float("timeout_s", 86400.0)
        self.best_metric = str(self.cfg.get("best_metric", "val_score"))
        self.gpu_count = self.cfg.get_int("gpu_count", 0)
        self.env = self.cfg.get_dict("env", {})
        self._proc = None

    def train(self, spec: TrainingSpec, observer: Callable[[dict[str, Any]], Any] | None = None) -> TrainingResult:
        started = time.time()
        variables = {
            "dataset_dir": str(Path(spec.dataset.path).parent),
            "train_path": spec.dataset.train_path,
            "val_path": spec.dataset.val_path,
            "reward_spec": spec.dataset.reward_spec_path,
            "model_path": spec.checkpoint.model_path,
            "checkpoint_id": spec.checkpoint.id,
            "round": spec.round,
            "run_dir": str(self.runtime.run_dir),
            "work_dir": str(spec.work_dir),
            "seed": spec.seed,
        }
        command = format_template(self.command, variables)
        output_dir = Path(format_template(self.output_dir_template, variables))
        metrics_path = Path(format_template(self.metrics_path_template, variables))
        if not output_dir.is_absolute():
            output_dir = spec.work_dir / output_dir
        if not metrics_path.is_absolute():
            metrics_path = spec.work_dir / metrics_path
        variables.update({"output_dir": str(output_dir), "metrics_path": str(metrics_path)})
        # Re-format once so output paths can be referenced in the command.
        command = format_template(self.command, variables)
        log_path = spec.work_dir / "train.log"
        proc, log_handle = self.runtime.executor.start(
            command,
            cwd=spec.work_dir,
            env=self.env,
            log_path=log_path,
            label="trainer.run",
        )
        self._proc = proc
        offset = 0
        tailed: list[dict[str, Any]] = []
        status = "completed"
        error: str | None = None
        deadline = started + self.timeout_s if self.timeout_s > 0 else None
        try:
            while True:
                lines, offset = _read_appended_jsonl(metrics_path, offset)
                for record in lines:
                    tailed.append(record)
                    if _notify(observer, record) == "stop":
                        status = "stopped_by_monitor"
                        _terminate(proc)
                        break
                if status == "stopped_by_monitor":
                    break
                if proc.poll() is not None:
                    break
                if deadline is not None and time.time() > deadline:
                    status = "timeout"
                    error = f"training exceeded {self.timeout_s}s"
                    _terminate(proc)
                    break
                time.sleep(self.poll_interval_s)
        finally:
            log_handle.close()
            self._proc = None
        # Drain anything written between the last poll and exit.
        lines, offset = _read_appended_jsonl(metrics_path, offset)
        tailed.extend(lines)
        duration = time.time() - started
        if proc.returncode not in (0, None) and status == "completed":
            status = "failed"
            error = f"trainer exited with code {proc.returncode}"
        best = _best_record(tailed, self.best_metric)
        gpu_hours = (duration / 3600.0) * self.gpu_count if self.gpu_count else 0.0
        checkpoint_path = str(output_dir.resolve()) if output_dir.exists() else None
        return TrainingResult(
            job_id=f"train-r{spec.round:04d}-{int(started)}",
            status=status,
            checkpoint_path=checkpoint_path,
            metrics=tailed,
            best_metric=best,
            best_step=int(best["step"]) if best and "step" in best else None,
            num_steps=len(tailed),
            duration_s=duration,
            log_path=str(log_path),
            error=error,
            metadata={
                "dataset_id": spec.dataset.dataset_id,
                "checkpoint_in": spec.checkpoint.id,
                "gpu_hours": gpu_hours,
                "command": command if isinstance(command, str) else " ".join(command),
            },
        )

    def close(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            _terminate(self._proc)
        self._proc = None


@register("trainer", "verl")
class VerlTrainer(CommandTrainer):
    """Verl preset. Behavior is fully defined by the command template."""


@register("trainer", "slime")
class SlimeTrainer(CommandTrainer):
    """Slime preset. Behavior is fully defined by the command template."""


def _read_appended_jsonl(path: Path, offset: int) -> tuple[list[dict[str, Any]], int]:
    if not path.exists():
        return [], offset
    with path.open("rb") as fh:
        fh.seek(offset)
        chunk = fh.read()
    if not chunk:
        return [], offset
    last_newline = chunk.rfind(b"\n")
    if last_newline < 0:
        return [], offset
    complete = chunk[: last_newline + 1]
    records: list[dict[str, Any]] = []
    for line in complete.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(raw, dict):
            continue
        metrics = raw.get("metrics") if isinstance(raw.get("metrics"), dict) else raw
        record = {"step": raw.get("step", raw.get("global_step", len(records) + 1)), "metrics": dict(metrics)}
        records.append(record)
    return records, offset + len(complete)


def _best_record(records: list[dict[str, Any]], metric: str) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    best_value = -math.inf
    for record in records:
        value = (record.get("metrics") or {}).get(metric)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and float(value) > best_value:
            best, best_value = record, float(value)
    return best


def _terminate(proc: Any) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except Exception:  # noqa: BLE001
        proc.kill()
