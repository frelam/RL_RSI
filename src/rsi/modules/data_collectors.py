from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from rsi.core.errors import AdapterError, ConfigError
from rsi.core.interfaces import DataCollector
from rsi.core.jsonio import stable_hash
from rsi.core.registry import register
from rsi.core.types import (
    ChatMessage,
    DataRequest,
    DataSample,
    HarnessTask,
    LLMRequest,
)
from rsi.modules.datasets import (
    DatasetCatalog,
    DatasetSpec,
    fetch_dataset_rows,
    fill_dataset_quotas,
    rows_to_samples,
)

_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe_component(value: str) -> str:
    return _SAFE_NAME.sub("-", value)[:80] or "item"


def _message_list(prompt: str, response: str) -> list[ChatMessage]:
    return [
        ChatMessage(role="user", content=prompt),
        ChatMessage(role="assistant", content=response),
    ]


def _sample_from_payload(
    raw: dict[str, Any],
    *,
    request: DataRequest,
    round_no: int,
    source: str,
    verifier: dict[str, Any] | None,
    seed_topic: str,
) -> DataSample | None:
    prompt = str(raw.get("prompt") or raw.get("question") or "")
    response = str(raw.get("response") or raw.get("answer") or raw.get("solution") or "")
    messages = raw.get("messages")
    if not prompt and isinstance(messages, list) and messages:
        prompt = str(messages[0].get("content", ""))
    if not prompt:
        return None
    if not messages:
        messages = [m.__dict__ for m in _message_list(prompt, response)] if response else [
            {"role": "user", "content": prompt}
        ]
    reference = raw.get("reference", raw.get("ground_truth"))
    return DataSample(
        id=stable_hash("sample", request.id, prompt, response, len(messages)),
        kind=request.kind,
        capability=request.capability,
        subcapability=request.subcapability,
        prompt=prompt,
        response=response,
        reference=None if reference is None else str(reference),
        messages=[
            ChatMessage(role=str(m.get("role", "user")), content=str(m.get("content", "")))
            for m in messages
            if isinstance(m, dict)
        ],
        verifier=raw.get("verifier") if isinstance(raw.get("verifier"), dict) else verifier,
        source=source,
        reward=raw.get("reward") if isinstance(raw.get("reward"), (int, float)) else None,
        group=request.capability,
        round=round_no,
        metadata={
            "request_id": request.id,
            "defect_id": request.defect_id,
            "seed_topic": seed_topic,
            "round": round_no,
            "synthetic": True,
            "license": raw.get("license", "synthetic"),
            **dict(raw.get("metadata") or {}),
        },
    )


@register("data_collector", "text_synthesis")
class TextSynthesisCollector(DataCollector):
    """Generated text data for a capability deficit (teacher-model synthesis)."""

    kind = "text"

    def __init__(self, runtime: Any, cfg: Any, **params: Any) -> None:
        super().__init__(runtime, cfg, **params)
        self.prompt_id = str(self.cfg.require("prompt").get("prompt"))
        self.profile = str(self.cfg.get("profile", runtime.config.get("llm.default_profile", "judge")))
        self.samples_per_call = max(1, self.cfg.get_int("samples_per_call", 3))
        self.temperature = self.cfg.get_float("temperature", 0.8)
        self.max_tokens = self.cfg.get_int("max_tokens", 2048)
        self.max_calls = self.cfg.get_int("max_calls", 6)

    def collect(self, request: DataRequest, context: Any) -> list[DataSample]:
        seed_topic = request.seeds[0] if request.seeds else request.capability
        samples: list[DataSample] = []
        seen: set[str] = set()
        calls = min(self.max_calls, math.ceil(request.count / self.samples_per_call))
        for call in range(calls):
            rendered = self.runtime.prompt_hub.render(
                request.prompt_id or self.prompt_id,
                {
                    "request_id": request.id,
                    "round": context.round,
                    "capability": request.capability,
                    "subcapability": request.subcapability,
                    "count": self.samples_per_call,
                    "seeds": request.seeds,
                    "seed_topic": seed_topic,
                    "target_difficulty": request.target_difficulty,
                    "constraints": request.constraints,
                    "notes": request.notes,
                },
            )
            response = self.runtime.llm(self.profile).complete(
                LLMRequest(
                    prompt_id=rendered.key,
                    prompt=rendered.text,
                    system=rendered.system,
                    schema=rendered.schema,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    metadata={
                        "round": context.round,
                        "request_id": request.id,
                        "capability": request.capability,
                        "seed_topic": seed_topic,
                        "call_index": call,
                    },
                )
            )
            payload = next((item for item in response.parsed if item is not None), None)
            if payload is None:
                self.logger.warning("text synthesis returned non-JSON for %s", request.id)
                continue
            raw_samples = payload.get("samples") or payload.get("items") or []
            if not isinstance(raw_samples, list):
                continue
            for raw in raw_samples:
                if not isinstance(raw, dict):
                    continue
                sample = _sample_from_payload(
                    raw,
                    request=request,
                    round_no=context.round,
                    source=f"llm:{self.profile}:{rendered.key}",
                    verifier=request.constraints.get("verifier"),
                    seed_topic=seed_topic,
                )
                if sample is None or sample.id in seen:
                    continue
                seen.add(sample.id)
                samples.append(sample)
                if len(samples) >= request.count:
                    return samples
        return samples


@register("data_collector", "agentic_synthesis")
class AgenticSynthesisCollector(DataCollector):
    """Use an agent harness to synthesize tasks / environments.

    The harness writes task files into an isolated directory. Everything it
    returns is schema-checked before becoming a sample, and paths are confined
    to the request work directory.
    """

    kind = "agentic"

    def __init__(self, runtime: Any, cfg: Any, **params: Any) -> None:
        super().__init__(runtime, cfg, **params)
        self.harness_name = str(self.cfg.get("harness", "synth"))
        self.prompt_id = str(self.cfg.require("prompt").get("prompt"))
        self.skills = [str(v) for v in self.cfg.get_list("skills", ["env-synthesis"])]
        self.max_tasks = self.cfg.get_int("max_tasks", 8)
        self.timeout_s = self.cfg.get_int("timeout_s", 1200)

    def collect(self, request: DataRequest, context: Any) -> list[DataSample]:
        seed_topic = request.seeds[0] if request.seeds else request.capability
        rendered = self.runtime.prompt_hub.render(
            request.prompt_id or self.prompt_id,
            {
                "request_id": request.id,
                "round": context.round,
                "capability": request.capability,
                "subcapability": request.subcapability,
                "count": request.count,
                "seeds": request.seeds,
                "seed_topic": seed_topic,
                "target_difficulty": request.target_difficulty,
                "constraints": request.constraints,
                "notes": request.notes,
            },
        )
        work_dir = context.work_dir / "agentic" / _safe_component(request.id)
        task = HarnessTask(
            name="env_synthesis",
            instruction=rendered.text,
            work_dir=work_dir,
            skills=self.skills,
            output_schema=rendered.schema,
            timeout_s=self.timeout_s,
            metadata={
                "round": context.round,
                "request_id": request.id,
                "capability": request.capability,
                "seed_topic": seed_topic,
            },
        )
        result = self.runtime.harness(self.harness_name).run(task)
        if not result.ok:
            raise AdapterError(f"agentic synthesis failed: {result.error}")
        raw_tasks = (result.output or {}).get("tasks") or []
        if not isinstance(raw_tasks, list):
            raise AdapterError("agentic synthesis output must contain a 'tasks' list")
        samples: list[DataSample] = []
        for index, raw in enumerate(raw_tasks[: self.max_tasks]):
            if not isinstance(raw, dict):
                continue
            instruction = str(raw.get("instruction") or raw.get("prompt") or "")
            if not instruction:
                continue
            task_dir = work_dir / f"task-{index:03d}"
            _materialize_task_files(task_dir, raw.get("files") or {})
            verifier = raw.get("verifier") if isinstance(raw.get("verifier"), dict) else request.constraints.get("verifier")
            messages = [ChatMessage(role="user", content=instruction)]
            samples.append(
                DataSample(
                    id=stable_hash("agentic", request.id, index, instruction),
                    kind="agentic",
                    capability=request.capability,
                    subcapability=request.subcapability,
                    prompt=instruction,
                    response=str(raw.get("solution") or ""),
                    reference=raw.get("reference"),
                    messages=messages,
                    verifier=verifier,
                    source=f"harness:{self.harness_name}",
                    difficulty=float(raw.get("difficulty", 0.5)),
                    group=request.capability,
                    round=context.round,
                    env_path=str(task_dir),
                    metadata={
                        "request_id": request.id,
                        "defect_id": request.defect_id,
                        "round": context.round,
                        "synthetic": True,
                        "task": raw.get("name") or f"task-{index}",
                        "seed_topic": seed_topic,
                        "skills": self.skills,
                    },
                )
            )
        return samples


@register("data_collector", "rollout_reuse")
class RolloutReuseCollector(DataCollector):
    """Turn successful / high-reward collected trajectories into training data."""

    kind = "rollout"

    def __init__(self, runtime: Any, cfg: Any, **params: Any) -> None:
        super().__init__(runtime, cfg, **params)
        self.only_success = self.cfg.get_bool("only_success", True)
        self.min_reward = self.cfg.get_float("min_reward", 0.0)
        self.include_unlabeled = self.cfg.get_bool("include_unlabeled", False)

    def collect(self, request: DataRequest, context: Any) -> list[DataSample]:
        if context.collect_result is None:
            return []
        samples: list[DataSample] = []
        for trajectory in context.collect_result.trajectories:
            if self.only_success:
                if trajectory.success is True:
                    pass
                elif trajectory.success is None and trajectory.reward is not None:
                    if float(trajectory.reward) < self.min_reward:
                        continue
                elif not self.include_unlabeled:
                    continue
            if not trajectory.final_answer.strip():
                continue
            samples.append(
                DataSample(
                    id=stable_hash("rollout-sample", trajectory.id),
                    kind="rollout",
                    capability=str(trajectory.metadata.get("capability") or request.capability),
                    prompt=trajectory.prompt,
                    response=trajectory.final_answer,
                    messages=trajectory.messages,
                    source="rollout_reuse",
                    reward=trajectory.reward,
                    group=request.capability,
                    round=context.round,
                    metadata={
                        "trajectory_id": trajectory.id,
                        "checkpoint_id": trajectory.checkpoint_id,
                        "round": context.round,
                        "success": trajectory.success,
                    },
                )
            )
            if len(samples) >= request.count:
                break
        return samples


@register("data_collector", "knowledge_ingest")
class KnowledgeIngestCollector(DataCollector):
    """Chunk local knowledge sources and synthesize QA pairs with provenance."""

    kind = "knowledge"

    def __init__(self, runtime: Any, cfg: Any, **params: Any) -> None:
        super().__init__(runtime, cfg, **params)
        self.paths = [str(p) for p in self.cfg.get_list("paths", [])]
        self.prompt_id = str(self.cfg.get("prompt", "knowledge_qa"))
        self.profile = str(self.cfg.get("profile", runtime.config.get("llm.default_profile", "judge")))
        self.chunk_chars = self.cfg.get_int("chunk_chars", 3000)
        self.overlap_chars = self.cfg.get_int("overlap_chars", 300)
        self.max_chunks = self.cfg.get_int("max_chunks", 24)
        self.max_file_bytes = self.cfg.get_int("max_file_bytes", 2_000_000)
        self.extensions = {str(e) for e in self.cfg.get_list("extensions", [".md", ".txt", ".json", ".jsonl"])}
        self.license = str(self.cfg.get("license", "local-corpus"))

    def collect(self, request: DataRequest, context: Any) -> list[DataSample]:
        chunks = self._load_chunks()
        samples: list[DataSample] = []
        for path, chunk_index, chunk in chunks:
            if len(samples) >= request.count:
                break
            rendered = self.runtime.prompt_hub.render(
                request.prompt_id or self.prompt_id,
                {
                    "request_id": request.id,
                    "round": context.round,
                    "capability": request.capability,
                    "context": chunk,
                    "source": str(path),
                    "count": request.count,
                },
            )
            response = self.runtime.llm(self.profile).complete(
                LLMRequest(
                    prompt_id=rendered.key,
                    prompt=rendered.text,
                    system=rendered.system,
                    schema=rendered.schema,
                    temperature=0.3,
                    metadata={"round": context.round, "request_id": request.id, "source": str(path)},
                )
            )
            payload = next((item for item in response.parsed if item is not None), None)
            if payload is None:
                continue
            for raw in payload.get("samples") or []:
                if not isinstance(raw, dict):
                    continue
                sample = _sample_from_payload(
                    raw,
                    request=request,
                    round_no=context.round,
                    source=f"knowledge:{path.name}",
                    verifier=request.constraints.get("verifier"),
                    seed_topic=path.stem,
                )
                if sample is None:
                    continue
                sample.metadata.update(
                    {
                        "provenance": str(path),
                        "chunk_index": chunk_index,
                        "license": self.license,
                    }
                )
                samples.append(sample)
                if len(samples) >= request.count:
                    break
        return samples

    def _load_chunks(self) -> list[tuple[Path, int, str]]:
        roots = [self.runtime.path(p) for p in self.paths]
        files: list[Path] = []
        for root in roots:
            if root.is_file():
                files.append(root)
            elif root.is_dir():
                files.extend(
                    path
                    for path in sorted(root.rglob("*"))
                    if path.is_file() and path.suffix.lower() in self.extensions
                )
        chunks: list[tuple[Path, int, str]] = []
        for path in files:
            if path.stat().st_size > self.max_file_bytes:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for index, chunk in enumerate(
                _chunk_text(text, self.chunk_chars, self.overlap_chars)
            ):
                chunks.append((path, index, chunk))
                if len(chunks) >= self.max_chunks:
                    return chunks
        return chunks


@register("data_collector", "dataset_library")
class DatasetLibraryCollector(DataCollector):
    """Load curated external datasets that cover the requested capability."""

    kind = "dataset"

    def __init__(self, runtime: Any, cfg: Any, **params: Any) -> None:
        super().__init__(runtime, cfg, **params)
        catalog_path = runtime.path(self.cfg.get("catalog", "configs/datasets.yaml"))
        self.catalog = DatasetCatalog(catalog_path, runtime.repo_root)
        self.max_datasets = max(1, self.cfg.get_int("max_datasets", 4))

    def collect(self, request: DataRequest, context: Any) -> list[DataSample]:
        priority = self.runtime.catalog.priority(request.capability)
        selection = self.catalog.select(
            {request.capability: priority}, request.count, max_datasets=self.max_datasets
        )
        if not selection.datasets:
            self.logger.info("no dataset catalog entries cover %s", request.capability)
            return []
        quotas = _dataset_quotas(selection.ratios, request.count)
        sources: list[tuple[DatasetSpec, list[DataSample]]] = []
        for dataset in selection.datasets:
            try:
                rows = fetch_dataset_rows(dataset, self.runtime.repo_root)
            except Exception as exc:  # noqa: BLE001 - other sources may still serve the request
                self.logger.warning("dataset %s failed: %s", dataset.name, exc)
                continue
            sources.append((dataset, rows_to_samples(dataset, rows, request, len(rows))))
        samples, source_stats = fill_dataset_quotas(sources, request.count, quotas)
        samples.sort(key=lambda sample: (sample.metadata.get("dataset", ""), sample.id))
        samples = samples[: request.count]
        for sample in samples:
            sample.metadata["dataset_selection"] = {
                "ratios": selection.ratios,
                "uncovered": selection.uncovered,
                "source_counts": source_stats,
                "optimization": selection.stats,
            }
        return samples


def _dataset_quotas(ratios: dict[str, float], total: int) -> dict[str, int]:
    if not ratios or total <= 0:
        return {}
    exact = {name: ratio * total for name, ratio in ratios.items()}
    quotas = {name: int(value) for name, value in exact.items()}
    remainder = total - sum(quotas.values())
    order = sorted(ratios, key=lambda name: (exact[name] - quotas[name], name), reverse=True)
    for name in order[:remainder]:
        quotas[name] += 1
    return quotas


def _chunk_text(text: str, size: int, overlap: int) -> list[str]:
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    step = max(1, size - overlap)
    return [text[start : start + size] for start in range(0, len(text), step) if text[start : start + size].strip()]


def _materialize_task_files(root: Path, files: dict[str, Any]) -> None:
    if not files:
        return
    root.mkdir(parents=True, exist_ok=True)
    base = root.resolve()
    for relative, content in files.items():
        relative_path = Path(str(relative))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise AdapterError(f"Unsafe task file path: {relative!r}")
        target = (root / relative_path).resolve()
        if base != target and base not in target.parents:
            raise AdapterError(f"Task file escapes work dir: {relative!r}")
        target.parent.mkdir(parents=True, exist_ok=True)
        text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, indent=2)
        target.write_text(text, encoding="utf-8")
