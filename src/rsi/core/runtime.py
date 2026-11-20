from __future__ import annotations

import logging
import os
import random
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any

from rsi.core.config import Config
from rsi.core.errors import AdapterError, RsiError
from rsi.core.jsonio import (
    append_jsonl,
    atomic_write_json,
    atomic_write_text,
    now_iso,
    read_json,
    stable_hash,
    to_plain,
    write_jsonl,
)
from rsi.core.registry import create
from rsi.core.types import Checkpoint
from rsi.hubs.catalog import CapabilityCatalog
from rsi.hubs.prompts import PromptHub
from rsi.hubs.skills import SkillHub


# --------------------------------------------------------------------- logging
def _build_logger(run_dir: Path, run_id: str, quiet: bool) -> logging.Logger:
    logger = logging.getLogger(f"rsi.run.{stable_hash(str(run_dir), length=10)}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    stream = logging.StreamHandler(sys.stderr)
    stream.setLevel(logging.WARNING if quiet else logging.INFO)
    stream.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S"))
    logger.addHandler(stream)
    run_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(run_dir / "rsi.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    )
    logger.addHandler(file_handler)
    logger.info("logging initialized for run %s", run_id)
    return logger


# --------------------------------------------------------------------- event log
class EventLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def emit(self, event_type: str, **fields: Any) -> None:
        record = {"ts": now_iso(), "type": event_type}
        record.update(fields)
        with self._lock:
            append_jsonl(self.path, record)


# --------------------------------------------------------------------- artifacts
class ArtifactStore:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir

    def round_dir(self, round_no: int) -> Path:
        path = self.run_dir / "rounds" / f"round-{round_no:04d}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def stage_dir(self, round_no: int, stage: str) -> Path:
        path = self.round_dir(round_no) / stage
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_json(self, round_no: int, stage: str, name: str, value: Any) -> Path:
        return atomic_write_json(self.stage_dir(round_no, stage) / name, value)

    def write_jsonl(self, round_no: int, stage: str, name: str, rows: Any) -> Path:
        return write_jsonl(self.stage_dir(round_no, stage) / name, rows)

    def write_text(self, round_no: int, stage: str, name: str, text: str) -> Path:
        return atomic_write_text(self.stage_dir(round_no, stage) / name, text)

    def rel(self, path: str | Path) -> str:
        try:
            return str(Path(path).resolve().relative_to(self.run_dir.resolve()))
        except ValueError:
            return str(path)


# --------------------------------------------------------------------- commands
@dataclass
class CommandResult:
    command: str
    returncode: int
    stdout: str
    stderr: str
    duration_s: float
    dry_run: bool = False


class CommandExecutor:
    """The only place that shells out.

    Every external tool invocation is recorded in ``commands.jsonl`` so a run
    can be audited and replayed. Adapters must use this instead of calling
    subprocess directly.
    """

    def __init__(self, run_dir: Path, dry_run: bool = False) -> None:
        self.run_dir = run_dir
        self.dry_run = dry_run
        self.record_path = run_dir / "commands.jsonl"
        self._lock = threading.Lock()

    def run(
        self,
        argv: str | list[str],
        *,
        cwd: str | Path | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
        label: str = "",
        check: bool = False,
    ) -> CommandResult:
        display = argv if isinstance(argv, str) else " ".join(str(a) for a in argv)
        cwd_path = Path(cwd) if cwd else self.run_dir
        started = time.time()
        if self.dry_run:
            self._record(label, display, cwd_path, 0, 0.0)
            return CommandResult(display, 0, "", "", 0.0, dry_run=True)
        merged_env = os.environ.copy()
        if env:
            merged_env.update({str(k): str(v) for k, v in env.items()})
        try:
            completed = subprocess.run(
                argv,
                cwd=str(cwd_path),
                env=merged_env,
                shell=isinstance(argv, str),
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.time() - started
            self._record(label, display, cwd_path, 124, duration)
            raise AdapterError(f"Command timed out after {timeout}s: {display}") from exc
        except OSError as exc:
            duration = time.time() - started
            self._record(label, display, cwd_path, 127, duration)
            raise AdapterError(f"Cannot start command {display!r}: {exc}") from exc
        duration = time.time() - started
        self._record(label, display, cwd_path, completed.returncode, duration)
        result = CommandResult(
            command=display,
            returncode=completed.returncode,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
            duration_s=duration,
        )
        if check and result.returncode != 0:
            tail = (result.stderr or result.stdout).strip()[-2000:]
            raise AdapterError(f"Command failed ({result.returncode}): {display}\n{tail}")
        return result

    def start(
        self,
        argv: list[str] | str,
        *,
        cwd: str | Path | None = None,
        env: dict[str, str] | None = None,
        log_path: str | Path | None = None,
        label: str = "",
    ) -> tuple[subprocess.Popen[str], Any]:
        """Start a long-running process, streaming stdout/stderr to a log file."""
        display = argv if isinstance(argv, str) else " ".join(str(a) for a in argv)
        cwd_path = Path(cwd) if cwd else self.run_dir
        log_path = Path(log_path) if log_path else self.run_dir / "process.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if self.dry_run:
            raise AdapterError("Cannot start a process in dry-run mode")
        handle = log_path.open("w", encoding="utf-8")
        merged_env = os.environ.copy()
        if env:
            merged_env.update({str(k): str(v) for k, v in env.items()})
        try:
            proc = subprocess.Popen(
                argv,
                cwd=str(cwd_path),
                env=merged_env,
                shell=isinstance(argv, str),
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except OSError as exc:
            handle.close()
            raise AdapterError(f"Cannot start command {display!r}: {exc}") from exc
        self._record(label, display, cwd_path, None, 0.0, pid=proc.pid)
        return proc, handle

    def _record(
        self,
        label: str,
        command: str,
        cwd: Path,
        returncode: int | None,
        duration: float,
        **extra: Any,
    ) -> None:
        record = {
            "ts": now_iso(),
            "label": label,
            "command": command,
            "cwd": str(cwd),
            "returncode": returncode,
            "duration_s": round(duration, 3),
        }
        record.update(extra)
        with self._lock:
            append_jsonl(self.record_path, record)


# --------------------------------------------------------------------- checkpoints
class CheckpointManager:
    """Single source of truth for checkpoint versioning and promotion.

    Stages never pass checkpoint objects around by hand beyond the current
    round: they read ``current()`` at the start and ``promote()`` at the end.
    """

    def __init__(self, root: Path, initial_cfg: Config, resolve_path: Any) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.registry_path = root / "registry.json"
        self._resolve_path = resolve_path
        self._lock = threading.RLock()
        if self.registry_path.exists():
            self._data = read_json(self.registry_path)
        else:
            self._data = self._bootstrap(initial_cfg)
            self._save()

    def _bootstrap(self, cfg: Config) -> dict[str, Any]:
        model_path = str(cfg.get("model_path", "mock://base"))
        if "://" not in model_path:
            model_path = str(self._resolve_path(model_path))
        checkpoint = Checkpoint(
            id="ckpt_v000",
            version=0,
            model_path=model_path,
            base_model=str(cfg.get("base_model", "")),
            status="current",
            created_round=0,
            source_job=None,
            metrics={},
            tags=["initial"],
            created_at=now_iso(),
        )
        return {
            "current": checkpoint.id,
            "next_version": 1,
            "checkpoints": {checkpoint.id: to_plain(checkpoint)},
        }

    def _save(self) -> None:
        atomic_write_json(self.registry_path, self._data)

    def current_id(self) -> str:
        return str(self._data["current"])

    def current(self) -> Checkpoint:
        return self.get(self.current_id())

    def get(self, checkpoint_id: str) -> Checkpoint:
        with self._lock:
            raw = self._data["checkpoints"].get(checkpoint_id)
        if raw is None:
            raise RsiError(f"Unknown checkpoint {checkpoint_id!r}")
        return Checkpoint(**raw)

    def list(self) -> list[Checkpoint]:
        with self._lock:
            return [Checkpoint(**raw) for raw in self._data["checkpoints"].values()]

    def reserve(
        self,
        *,
        model_path: str,
        round_no: int,
        metrics: dict[str, Any] | None = None,
        source_job: str | None = None,
        parent_id: str | None = None,
        tags: list[str] | None = None,
    ) -> Checkpoint:
        with self._lock:
            version = int(self._data.get("next_version", 1))
            checkpoint_id = f"ckpt_v{version:03d}"
            while checkpoint_id in self._data["checkpoints"]:
                version += 1
                checkpoint_id = f"ckpt_v{version:03d}"
            self._data["next_version"] = version + 1
            checkpoint = Checkpoint(
                id=checkpoint_id,
                version=version,
                model_path=str(model_path),
                base_model=self.current().base_model,
                status="candidate",
                created_round=round_no,
                parent_id=parent_id or self.current_id(),
                source_job=source_job,
                metrics=dict(metrics or {}),
                tags=list(tags or []),
                created_at=now_iso(),
            )
            self._data["checkpoints"][checkpoint_id] = to_plain(checkpoint)
            self._save()
            return checkpoint

    def promote(self, checkpoint_id: str) -> Checkpoint:
        with self._lock:
            if checkpoint_id not in self._data["checkpoints"]:
                raise RsiError(f"Unknown checkpoint {checkpoint_id!r}")
            for raw in self._data["checkpoints"].values():
                if raw.get("status") == "current":
                    raw["status"] = "archived"
            raw = self._data["checkpoints"][checkpoint_id]
            raw["status"] = "current"
            self._data["current"] = checkpoint_id
            self._save()
            return Checkpoint(**raw)

    def reject(self, checkpoint_id: str, reason: str = "") -> None:
        with self._lock:
            raw = self._data["checkpoints"].get(checkpoint_id)
            if raw is None:
                raise RsiError(f"Unknown checkpoint {checkpoint_id!r}")
            if raw.get("status") == "current":
                raise RsiError("Refusing to reject the current checkpoint")
            raw["status"] = "rejected"
            raw.setdefault("metrics", {})["reject_reason"] = reason
            self._save()

    def update_metrics(self, checkpoint_id: str, metrics: dict[str, Any]) -> Checkpoint:
        with self._lock:
            raw = self._data["checkpoints"].get(checkpoint_id)
            if raw is None:
                raise RsiError(f"Unknown checkpoint {checkpoint_id!r}")
            raw.setdefault("metrics", {}).update(to_plain(metrics))
            self._save()
            return Checkpoint(**raw)


# --------------------------------------------------------------------- runtime
class Runtime:
    """Shared services passed to every component."""

    def __init__(
        self,
        config: Config,
        *,
        run_id: str,
        runs_dir: str | Path | None = None,
        seed: int | None = None,
        quiet: bool = False,
        dry_run: bool = False,
    ) -> None:
        self.config = config
        self.run_id = run_id
        configured_root = str(config.get("paths.repo_root", "") or "")
        if configured_root:
            root = Path(configured_root).expanduser()
            self.repo_root = root.resolve() if root.is_absolute() else (config.config_dir / root).resolve()
        else:
            self.repo_root = config.repo_root
        self.seed = int(seed if seed is not None else config.get("run.seed", 7))
        runs_base = Path(runs_dir) if runs_dir else self.path(config.get("paths.runs_dir", "runs"))
        if not runs_base.is_absolute():
            runs_base = self.repo_root / runs_base
        self.run_dir = runs_base / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.dry_run = dry_run
        self.logger = _build_logger(self.run_dir, run_id, quiet)
        self.events = EventLog(self.run_dir / "events.jsonl")
        self.artifacts = ArtifactStore(self.run_dir)
        self.executor = CommandExecutor(self.run_dir, dry_run=dry_run)
        self.executor.record_path.touch(exist_ok=True)
        self.prompt_hub = PromptHub(self.path(config.get("prompts.dir", "prompts")))
        self.skill_hub = SkillHub(self.path(config.get("skills.dir", "skills")))
        self.catalog = CapabilityCatalog(self.path(config.get("capabilities.path", "configs/capabilities.yaml")))
        self.checkpoints = CheckpointManager(
            self.run_dir / "checkpoints",
            config.section("checkpoint.initial", required=False),
            self.path,
        )
        self._cache: dict[tuple[str, str], Any] = {}
        self._write_run_metadata()
        self.events.emit(
            "run_started",
            run_id=run_id,
            seed=self.seed,
            config_source=config.source,
            repo_root=str(self.repo_root),
        )

    # ------------------------------------------------------------------ paths
    def path(self, value: str | Path) -> Path:
        path = Path(str(value)).expanduser()
        return path if path.is_absolute() else (self.repo_root / path)

    def rng(self, *parts: Any) -> random.Random:
        digest = stable_hash(self.seed, self.run_id, *parts)
        return random.Random(int(digest, 16))

    # ------------------------------------------------------------------ plugins
    def build(self, category: str, spec: str | dict[str, Any], cfg: Config | None = None) -> Any:
        return create(category, spec, self, cfg or self.config)

    def _cached(self, key: tuple[str, str], factory: Any) -> Any:
        if key not in self._cache:
            self._cache[key] = factory()
        return self._cache[key]

    def llm(self, profile: str | None = None) -> Any:
        profile = profile or str(self.config.get("llm.default_profile", "judge"))
        key = ("llm", profile)

        def factory() -> Any:
            cfg = self.config.section(f"llm.profiles.{profile}")
            spec = cfg.as_dict()
            return create("llm", spec, self, cfg)

        return self._cached(key, factory)

    def inference(self) -> Any:
        key = ("inference", "default")

        def factory() -> Any:
            cfg = self.config.section("infer")
            return create("inference", cfg.as_dict(), self, cfg)

        return self._cached(key, factory)

    def trainer(self) -> Any:
        key = ("trainer", "default")

        def factory() -> Any:
            cfg = self.config.section("trainer")
            return create("trainer", cfg.as_dict(), self, cfg)

        return self._cached(key, factory)

    def benchmark(self, name: str) -> Any:
        key = ("benchmark", name)

        def factory() -> Any:
            cfg = self.config.section(f"benchmark.benchmarks.{name}")
            return create("benchmark", cfg.as_dict(), self, cfg)

        return self._cached(key, factory)

    def harness(self, name: str) -> Any:
        key = ("harness", name)

        def factory() -> Any:
            cfg = self.config.section(f"harness.{name}")
            return create("harness", cfg.as_dict(), self, cfg)

        return self._cached(key, factory)

    # ------------------------------------------------------------------ misc
    def close(self) -> None:
        for component in reversed(list(self._cache.values())):
            close = getattr(component, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001 - best effort cleanup
                    self.logger.exception("Failed to close %s", type(component).__name__)
        self._cache.clear()
        for handler in list(self.logger.handlers):
            try:
                handler.flush()
                handler.close()
            finally:
                self.logger.removeHandler(handler)

    def __enter__(self) -> "Runtime":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def _write_run_metadata(self) -> None:
        path = self.run_dir / "run.json"
        existing = read_json(path, default={}) if path.exists() else {}
        payload = {
            "run_id": self.run_id,
            "created_at": existing.get("created_at") or now_iso(),
            "updated_at": now_iso(),
            "seed": self.seed,
            "config_source": self.config.source,
            "repo_root": str(self.repo_root),
            "python": sys.version.split()[0],
            "platform": sys.platform,
        }
        atomic_write_json(path, payload)
