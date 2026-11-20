from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from .types import (
    BenchmarkReport,
    Checkpoint,
    CollectResult,
    DataPlan,
    DataRequest,
    DataSample,
    DatasetArtifact,
    DefectReport,
    FilterBatch,
    HarnessResult,
    HarnessTask,
    LLMRequest,
    LLMResponse,
    StopDecision,
    TrainingResult,
    TrainingSpec,
)


class Component:
    """Base class for every pluggable component.

    Components receive the runtime (paths, logger, adapters, RNG) and their
    own configuration subtree. Constructor params from the plugin spec land in
    ``self.params`` so adapters can stay config-driven.
    """

    category = "component"

    def __init__(self, runtime: Any, cfg: Any, **params: Any) -> None:
        self.runtime = runtime
        self.cfg = cfg
        self.params = params
        name = params.get("instance_name") or self.__class__.__name__
        self.logger: logging.Logger = runtime.logger.getChild(str(name))

    def close(self) -> None:
        """Release external resources (servers, subprocesses)."""


# --------------------------------------------------------------------- adapters
class InferenceBackend(Component, ABC):
    category = "inference"

    @abstractmethod
    def ensure(self, checkpoint: Checkpoint) -> Any:
        """Make sure this checkpoint is being served and return EndpointInfo."""

    @abstractmethod
    def chat(self, request: Any) -> Any:
        """Run a chat completion against the served policy."""

    def supports_checkpoint_switch(self) -> bool:
        return False


class LLMBackend(Component, ABC):
    category = "llm"

    @abstractmethod
    def complete(self, request: LLMRequest) -> LLMResponse:
        """Text / structured completion used by judges and data generators."""


class BenchmarkBackend(Component, ABC):
    category = "benchmark"

    @abstractmethod
    def evaluate(self, checkpoint: Checkpoint, round_no: int) -> BenchmarkReport:
        """Evaluate a checkpoint and return item-level results."""


class HarnessBackend(Component, ABC):
    category = "harness"

    @abstractmethod
    def run(self, task: HarnessTask) -> HarnessResult:
        """Run an external agent harness and return a structured envelope."""


class TrainerBackend(Component, ABC):
    category = "trainer"

    @abstractmethod
    def train(self, spec: TrainingSpec, observer: Any = None) -> TrainingResult:
        """Train from a dataset artifact. ``observer(metrics_record)`` may return
        the string ``"stop"`` to request a graceful early stop."""


# --------------------------------------------------------------------- modules
class AnalyzerBackend(Component, ABC):
    category = "analyzer"

    @abstractmethod
    def analyze(self, collect: CollectResult, round_no: int, baseline: dict[str, float]) -> DefectReport:
        """Turn benchmark evidence / trajectories into capability defects."""


class DataCollector(Component, ABC):
    category = "data_collector"

    kind = "generic"

    @abstractmethod
    def collect(self, request: DataRequest, context: Any) -> list[DataSample]:
        """Produce samples for one data request."""


class SampleFilter(Component, ABC):
    category = "data_filter"

    @abstractmethod
    def filter(self, samples: list[DataSample], context: Any) -> FilterBatch:
        """Filter a batch; stateful filters (dedup) may use the whole batch."""


class DifficultyGrader(Component, ABC):
    category = "difficulty"

    @abstractmethod
    def grade(self, sample: DataSample, context: Any) -> float:
        """Return a difficulty in [0, 1]."""


class MixingStrategy(Component, ABC):
    category = "mixing"

    @abstractmethod
    def plan(self, samples: list[DataSample], plan: DataPlan, context: Any) -> dict[str, float]:
        """Return group -> relative weight."""


class DatasetBuilder(Component, ABC):
    category = "dataset_builder"

    @abstractmethod
    def build(
        self,
        samples: list[DataSample],
        context: Any,
        plan: DataPlan,
        mixing: dict[str, float],
    ) -> DatasetArtifact:
        """Materialize samples into a trainable dataset artifact."""


# --------------------------------------------------------------------- training / stop
class MonitorAlgorithm(Component, ABC):
    category = "monitor"

    @abstractmethod
    def observe(self, history: list[dict[str, Any]]) -> list[Any]:
        """Inspect the metric history and return zero or more findings."""


class StopPolicy(Component, ABC):
    category = "stop_policy"

    @abstractmethod
    def evaluate(self, state: Any, round_record: dict[str, Any]) -> StopDecision | None:
        """Return a stop decision when this policy fires."""
