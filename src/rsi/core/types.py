from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------- chat / LLM
@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class ChatRequest:
    messages: list[ChatMessage]
    temperature: float = 0.0
    max_tokens: int = 1024
    n: int = 1
    stop: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatResponse:
    texts: list[str]
    model: str
    usage: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] | None = None
    latency_s: float = 0.0


@dataclass
class EndpointInfo:
    base_url: str
    model: str
    checkpoint_id: str = ""
    managed: bool = False
    pid: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMRequest:
    prompt_id: str
    prompt: str
    system: str = ""
    schema: dict[str, Any] | None = None
    temperature: float = 0.0
    max_tokens: int = 2048
    n: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    texts: list[str]
    parsed: list[dict[str, Any] | None]
    model: str
    usage: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] | None = None


# --------------------------------------------------------------------- trajectory
@dataclass
class Trajectory:
    id: str
    prompt: str
    messages: list[ChatMessage]
    final_answer: str
    reward: float | None = None
    success: bool | None = None
    num_turns: int = 0
    tool_calls: int = 0
    response_length: int = 0
    checkpoint_id: str = ""
    round: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------- benchmark
@dataclass
class BenchmarkItem:
    id: str
    prompt: str
    capability: str
    subcapability: str = ""
    reference: str | None = None
    scorer: str = "contains"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ItemResult:
    item_id: str
    capability: str
    subcapability: str = ""
    score: float = 0.0
    passed: bool = False
    response: str = ""
    expected: str | None = None
    num_samples: int = 1
    latency_s: float = 0.0
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkReport:
    benchmark: str
    checkpoint_id: str
    round: int
    items: list[ItemResult] = field(default_factory=list)
    capabilities: dict[str, float] = field(default_factory=dict)
    subcapabilities: dict[str, float] = field(default_factory=dict)
    aggregates: dict[str, float] = field(default_factory=dict)
    num_items: int = 0
    primary_metric: str = "mean_score"
    artifact: str = ""
    started_at: str = ""
    finished_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CapabilityScore:
    capability: str
    score: float
    baseline: float | None = None
    delta: float | None = None
    num_items: int = 0
    pass_rate: float = 0.0


@dataclass
class CollectResult:
    round: int
    checkpoint_id: str
    reports: list[BenchmarkReport] = field(default_factory=list)
    capabilities: dict[str, CapabilityScore] = field(default_factory=dict)
    trajectories: list[Trajectory] = field(default_factory=list)
    aggregate: dict[str, float] = field(default_factory=dict)
    baseline_capabilities: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------- defects
@dataclass
class Defect:
    id: str
    capability: str
    description: str
    severity: float
    confidence: float
    source: str
    subcapability: str = ""
    evidence: list[str] = field(default_factory=list)
    suggested_data: dict[str, Any] = field(default_factory=dict)
    round: int = 0

    @property
    def key(self) -> tuple[str, str]:
        return (self.capability, self.subcapability)


@dataclass
class DefectReport:
    round: int
    checkpoint_id: str
    defects: list[Defect] = field(default_factory=list)
    capabilities: dict[str, CapabilityScore] = field(default_factory=dict)
    summary: str = ""
    sources: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------- data
@dataclass
class DataRequest:
    id: str
    capability: str
    kind: str
    count: int
    subcapability: str = ""
    target_difficulty: list[float] = field(default_factory=lambda: [0.2, 0.9])
    constraints: dict[str, Any] = field(default_factory=dict)
    seeds: list[str] = field(default_factory=list)
    prompt_id: str = ""
    reward_hint: dict[str, Any] = field(default_factory=dict)
    defect_id: str = ""
    notes: str = ""


@dataclass
class DataPlan:
    round: int
    requests: list[DataRequest] = field(default_factory=list)
    total_target: int = 0
    strategy: str = ""
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataSample:
    id: str
    kind: str
    capability: str
    prompt: str
    response: str = ""
    subcapability: str = ""
    reference: str | None = None
    messages: list[ChatMessage] = field(default_factory=list)
    verifier: dict[str, Any] | None = None
    source: str = ""
    difficulty: float = 0.0
    quality: float = 1.0
    reward: float | None = None
    group: str = ""
    round: int = 0
    env_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FilterDecision:
    sample_id: str
    keep: bool
    filter: str
    reason: str = ""
    score: float | None = None


@dataclass
class FilterBatch:
    kept: list[DataSample] = field(default_factory=list)
    decisions: list[FilterDecision] = field(default_factory=list)
    dropped: list[DataSample] = field(default_factory=list)


@dataclass
class DatasetArtifact:
    dataset_id: str
    path: str
    train_path: str
    val_path: str
    reward_spec_path: str
    manifest_path: str
    num_samples: int
    num_train: int
    num_val: int
    round: int
    schema: str = "prompt_response"
    groups: dict[str, int] = field(default_factory=dict)
    sources: dict[str, int] = field(default_factory=dict)
    difficulty: dict[str, float] = field(default_factory=dict)
    mixing: dict[str, float] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------- training
@dataclass
class TrainingSpec:
    round: int
    dataset: DatasetArtifact
    checkpoint: "Checkpoint"
    work_dir: Path
    seed: int = 0
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingResult:
    job_id: str
    status: str
    checkpoint_path: str | None = None
    metrics: list[dict[str, Any]] = field(default_factory=list)
    best_metric: dict[str, Any] | None = None
    best_step: int | None = None
    num_steps: int = 0
    duration_s: float = 0.0
    log_path: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Checkpoint:
    id: str
    version: int
    model_path: str
    base_model: str = ""
    status: str = "candidate"
    created_round: int = 0
    parent_id: str | None = None
    source_job: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    created_at: str = ""


# --------------------------------------------------------------------- loop
@dataclass
class StageResult:
    status: str
    summary: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    duration_s: float = 0.0


@dataclass
class StopDecision:
    should_stop: bool
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class LoopResult:
    status: str
    stop_reason: str
    rounds: int
    best_checkpoint: str | None
    best_score: float
    run_dir: str
    metadata: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------- harness
@dataclass
class HarnessTask:
    name: str
    instruction: str
    work_dir: Path
    skills: list[str] = field(default_factory=list)
    input_files: list[str] = field(default_factory=list)
    output_schema: dict[str, Any] | None = None
    timeout_s: int = 1800
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HarnessResult:
    ok: bool
    output: dict[str, Any] | None = None
    raw_text: str = ""
    duration_s: float = 0.0
    error: str | None = None
    command: str = ""
    artifacts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
