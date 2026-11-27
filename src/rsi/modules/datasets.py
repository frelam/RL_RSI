from __future__ import annotations

import json
import math
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from rsi.adapters.verl_agentic_rl import agentic_row_to_row, is_agentic_row
from rsi.adapters.verl_reasoning_rl import is_verl_row, verl_row_to_row
from rsi.core.errors import AdapterError, ConfigError
from rsi.core.jsonio import read_jsonl, stable_hash
from rsi.core.types import ChatMessage, DataRequest, DataSample


@dataclass
class DatasetSpec:
    """A source dataset and the capabilities it can train."""

    name: str
    loader: str
    url: str = ""
    #: Local rows, already preprocessed; a file-backed dataset without one is never selected.
    path: str = ""
    abilities: dict[str, float] = field(default_factory=dict)
    priority: float = 0.5
    reward: dict[str, Any] = field(default_factory=dict)
    #: ``path::function`` of the reward implementation; documentation only.
    reward_code: str = ""
    split: str = "train"
    config: str | None = None
    #: Load-time steps run by ``preprocess_row``.
    preprocess: list[str] = field(default_factory=list)
    #: Script(s) that turned the raw download into the rows at ``path``; documentation only.
    pipeline: str = ""
    #: Keep only rows of this dataset when ``path`` mixes several (matches ``subcapability``).
    source: str = ""
    rows: list[dict[str, Any]] = field(default_factory=list)
    license: str = ""


@dataclass
class DatasetSelection:
    datasets: list[DatasetSpec]
    ratios: dict[str, float]
    uncovered: list[str]


class DatasetCatalog:
    """Load dataset metadata from a user-owned YAML catalog."""

    def __init__(self, path: str | Path, root: str | Path | None = None) -> None:
        self.path = Path(path)
        self.root = Path(root) if root else self.path.parent
        self.datasets: list[DatasetSpec] = []
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream) or {}
        if not isinstance(raw, dict) or not isinstance(raw.get("datasets", {}), dict):
            raise ConfigError(f"Dataset catalog must contain a datasets mapping: {self.path}")
        for name, value in raw.get("datasets", {}).items():
            if not isinstance(value, dict):
                raise ConfigError(f"Dataset {name!r} must be a mapping")
            abilities = value.get("abilities") or value.get("capabilities") or {}
            if isinstance(abilities, list):
                abilities = {str(item): 1.0 for item in abilities}
            if not isinstance(abilities, dict):
                raise ConfigError(f"Dataset {name!r} abilities must be a mapping or list")
            # `preprocess` is a list of load-time steps, or a mapping that also
            # names the script(s) which produced the rows (`pipeline`).
            preprocess = value.get("preprocess") or {}
            if isinstance(preprocess, dict):
                pipeline, steps = str(preprocess.get("pipeline") or ""), preprocess.get("steps") or []
            else:
                pipeline, steps = "", preprocess
            if not isinstance(steps, list):
                raise ConfigError(f"Dataset {name!r} preprocess steps must be a list")
            self.datasets.append(
                DatasetSpec(
                    name=str(name),
                    loader=str(value.get("loader", "jsonl")),
                    url=str(value.get("url") or ""),
                    path=str(value.get("path") or ""),
                    abilities={str(k): _unit(v, f"{name}.abilities.{k}") for k, v in abilities.items()},
                    priority=_unit(value.get("priority", 0.5), f"{name}.priority"),
                    reward=dict(value.get("reward") or {}),
                    reward_code=str(value.get("reward_code") or ""),
                    split=str(value.get("split", "train")),
                    config=str(value["config"]) if value.get("config") is not None else None,
                    preprocess=[str(v) for v in steps],
                    pipeline=pipeline,
                    source=str(value.get("source") or ""),
                    rows=[row for row in value.get("rows", []) if isinstance(row, dict)],
                    license=str(value.get("license") or ""),
                )
            )

    def select(self, priorities: dict[str, float], count: int) -> DatasetSelection:
        return select_datasets(self.datasets, priorities, count)


#: Loaders that read the rows from `path`.
_FILE_LOADERS = frozenset({"jsonl", "parquet"})


def _unit(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be in [0, 1]") from exc
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ConfigError(f"{name} must be in [0, 1]")
    return number


def select_datasets(
    datasets: list[DatasetSpec], priorities: dict[str, float], count: int
) -> DatasetSelection:
    """Greedily maximize uncovered ability priority, then derive mix ratios."""
    remaining = {key: _unit(value, f"priority.{key}") for key, value in priorities.items()}
    # A file-backed dataset without local rows is declared only; it cannot serve a request.
    candidates = [item for item in datasets if item.path or item.loader not in _FILE_LOADERS]
    selected: list[DatasetSpec] = []
    contribution: dict[str, float] = {}
    while candidates and remaining and len(selected) < max(0, count):
        ranked = []
        for dataset in candidates:
            gain = sum(priority * dataset.abilities.get(ability, 0.0) for ability, priority in remaining.items())
            ranked.append((gain * dataset.priority, gain, dataset.name, dataset))
        _, gain, _, winner = max(ranked, key=lambda item: item[:3])
        if gain <= 0:
            break
        selected.append(winner)
        contribution[winner.name] = gain * winner.priority
        candidates.remove(winner)
        remaining = {
            ability: priority * (1.0 - winner.abilities.get(ability, 0.0))
            for ability, priority in remaining.items()
            if priority * (1.0 - winner.abilities.get(ability, 0.0)) > 1e-9
        }
    total = sum(contribution.values())
    ratios = {name: value / total for name, value in contribution.items()} if total else {}
    return DatasetSelection(
        datasets=selected,
        ratios=ratios,
        uncovered=sorted(ability for ability, priority in remaining.items() if priority > 1e-6),
    )


def merge_ability_priorities(
    rule: dict[str, float],
    agent: dict[str, float],
    *,
    rule_weight: float = 0.6,
    agent_weight: float = 0.4,
) -> dict[str, float]:
    """Combine rule-derived task needs and agent trajectory estimates."""
    if rule_weight < 0 or agent_weight < 0 or rule_weight + agent_weight <= 0:
        raise ValueError("ability source weights must be non-negative with a positive sum")
    output: dict[str, float] = {}
    for ability in sorted(set(rule) | set(agent)):
        rule_value = _unit(rule.get(ability, 0.0), f"rule.{ability}")
        agent_value = _unit(agent.get(ability, 0.0), f"agent.{ability}")
        weight = (rule_weight if ability in rule else 0.0) + (agent_weight if ability in agent else 0.0)
        output[ability] = (
            rule_weight * rule_value * (ability in rule)
            + agent_weight * agent_value * (ability in agent)
        ) / weight
    return output


def fetch_dataset_rows(spec: DatasetSpec, root: str | Path) -> list[dict[str, Any]]:
    """Fetch rows through a source-specific adapter with optional dependencies."""
    if spec.loader == "synthetic":
        rows = list(spec.rows)
    elif spec.loader == "jsonl":
        path = _resolve_path(spec, root)
        rows = read_jsonl(path)
    elif spec.loader == "parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise AdapterError("Parquet datasets require the optional 'pyarrow' package") from exc
        rows = pq.read_table(_resolve_path(spec, root)).to_pylist()
    elif spec.loader == "web":
        if not spec.url:
            raise ConfigError(f"Web dataset {spec.name!r} requires url")
        with urllib.request.urlopen(spec.url, timeout=30) as response:
            payload = response.read().decode("utf-8")
        if spec.url.lower().split("?", 1)[0].endswith(".json"):
            value = json.loads(payload)
            rows = value if isinstance(value, list) else value.get("data", [])
        else:
            rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
    elif spec.loader == "huggingface":
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise AdapterError("Hugging Face datasets require the optional 'datasets' package") from exc
        dataset = load_dataset(spec.url, spec.config, split=spec.split)
        rows = [dict(row) for row in dataset]
    else:
        raise ConfigError(f"Unknown dataset loader {spec.loader!r} for {spec.name!r}")
    return [_adapt_verl_row(row) for row in rows if isinstance(row, dict)]


def _adapt_verl_row(row: dict[str, Any]) -> dict[str, Any]:
    """Map verl rows (agentic tasks first, then reasoning_rl) to the generic shape."""
    if is_agentic_row(row):
        return agentic_row_to_row(row)
    return verl_row_to_row(row) if is_verl_row(row) else row


def _resolve_path(spec: DatasetSpec, root: str | Path) -> Path:
    path = Path(spec.path).expanduser()
    return path if path.is_absolute() else Path(root) / path


def preprocess_row(row: dict[str, Any], methods: list[str]) -> dict[str, Any] | None:
    """Apply common, deterministic normalization steps to a source row."""
    result = dict(row)
    for method in methods:
        if method == "normalize_chat":
            if "prompt" not in result and isinstance(result.get("messages"), list):
                result["prompt"] = next(
                    (str(item.get("content", "")) for item in result["messages"] if item.get("role") == "user"), ""
                )
            result["prompt"] = str(result.get("prompt") or result.get("question") or "").strip()
            result["response"] = str(result.get("response") or result.get("answer") or result.get("solution") or "").strip()
        elif method == "strip_text":
            for key in ("prompt", "question", "response", "answer", "solution"):
                if isinstance(result.get(key), str):
                    result[key] = result[key].strip()
        elif method == "drop_empty":
            if not any(str(result.get(key) or "").strip() for key in ("prompt", "question")):
                return None
        elif method == "deduplicate":
            pass
        else:
            raise ConfigError(f"Unknown dataset preprocess method {method!r}")
    return result


def rows_to_samples(
    spec: DatasetSpec, rows: list[dict[str, Any]], request: DataRequest, limit: int
) -> list[DataSample]:
    if limit <= 0:
        return []
    samples: list[DataSample] = []
    seen: set[str] = set()
    ability = request.capability
    for index, original in enumerate(rows):
        row = preprocess_row(original, spec.preprocess)
        if row is None or (spec.source and row.get("subcapability") != spec.source):
            continue
        capability = str(row.get("capability") or row.get("ability") or ability)
        if capability != ability and spec.abilities.get(capability, 0.0) <= 0:
            continue
        prompt = str(row.get("prompt") or row.get("question") or "").strip()
        response = str(row.get("response") or row.get("answer") or row.get("solution") or "").strip()
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            messages = [{"role": "user", "content": prompt}]
            if response:
                messages.append({"role": "assistant", "content": response})
        fingerprint = stable_hash(prompt, response)
        if not prompt or fingerprint in seen:
            continue
        seen.add(fingerprint)
        samples.append(
            DataSample(
                id=str(row.get("id") or stable_hash(spec.name, index, fingerprint)),
                kind=str(row.get("kind") or "text"),
                capability=capability,
                subcapability=str(row.get("subcapability") or ""),
                prompt=prompt,
                response=response,
                reference=row.get("reference", row.get("ground_truth")),
                messages=[ChatMessage(str(message.get("role", "user")), str(message.get("content", "")))
                          for message in messages if isinstance(message, dict)],
                verifier=dict(row.get("verifier") or request.constraints.get("verifier") or {}),
                source=f"dataset:{spec.name}",
                reward=row.get("reward") if isinstance(row.get("reward"), (int, float)) else None,
                group=capability,
                round=0,
                metadata={"dataset": spec.name, "url": spec.url, "license": spec.license,
                          "reward_hint": spec.reward, **dict(row.get("metadata") or {})},
            )
        )
        if len(samples) >= limit:
            break
    return samples
