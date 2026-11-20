from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rsi.core.interfaces import Component
from rsi.core.types import LoopResult, StageResult


@dataclass
class LoopContext:
    runtime: Any
    state: Any
    round: int
    round_dir: Path
    memory: dict[str, Any] = field(default_factory=dict)

    def stage_dir(self, name: str) -> Path:
        path = self.round_dir / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def log(self, message: str, **kwargs: Any) -> None:
        self.runtime.logger.info("[round %d] " + message, self.round, **kwargs)


class Stage(ABC):
    name: str = "stage"

    def __init__(self, runtime: Any, cfg: Any) -> None:
        self.runtime = runtime
        self.cfg = cfg
        self.logger = runtime.logger.getChild(self.name)

    @abstractmethod
    def run(self, ctx: LoopContext) -> StageResult:
        """Execute the stage. Must be side-effect-complete for this round."""

    def validate(self) -> None:
        """Construct every configured plugin without running it."""


class Loop(Component, ABC):
    """A loop is an orchestration strategy composed of stages.

    Add a new orchestration by subclassing Loop in a new file; reuse the stage
    implementations or write new ones. Nothing in the modules knows which loop
    is running.
    """

    category = "loop"
    name = "loop"

    def __init__(self, runtime: Any, cfg: Any, **params: Any) -> None:
        super().__init__(runtime, cfg, **params)

    @abstractmethod
    def run(self, *, resume: bool = False) -> LoopResult:
        """Run until a stop policy fires."""

    @abstractmethod
    def validate(self) -> None:
        """Build the full component tree from config; no external calls."""
