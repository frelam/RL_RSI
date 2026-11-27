from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rsi.core.jsonio import atomic_write_json, now_iso, read_json


@dataclass
class LoopState:
    run_id: str
    loop: str
    status: str = "running"
    round: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)
    best_checkpoint: str | None = None
    best_score: float = -1.0
    stop_reason: str = ""
    started_at: str = ""
    updated_at: str = ""
    seed: int = 0
    budget: dict[str, Any] = field(default_factory=dict)
    # Target capabilities a finished round did not improve; the next round plans their data first.
    pending_capabilities: list[str] = field(default_factory=list)


def new_state(run_id: str, loop: str, seed: int) -> LoopState:
    timestamp = now_iso()
    return LoopState(
        run_id=run_id,
        loop=loop,
        started_at=timestamp,
        updated_at=timestamp,
        seed=seed,
        budget={"wall_hours": 0.0, "gpu_hours": 0.0, "rounds": 0},
    )


def load_state(path: str | Path) -> LoopState:
    raw = read_json(path)
    if not isinstance(raw, dict):
        raise ValueError(f"State file is not a mapping: {path}")
    names = {field.name for field in dataclasses.fields(LoopState)}
    return LoopState(**{key: value for key, value in raw.items() if key in names})


def save_state(runtime: Any, state: LoopState) -> Path:
    state.updated_at = now_iso()
    state.budget["rounds"] = state.round
    return atomic_write_json(runtime.run_dir / "state.json", state)


def state_path(runtime: Any) -> Path:
    return runtime.run_dir / "state.json"
