from __future__ import annotations

import csv
import importlib
import io
import json
import math
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml
from scipy.optimize import Bounds, LinearConstraint, milp

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
    #: Git ref used by the ``github`` loader; defaults to ``main``.
    ref: str = ""
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
    stats: dict[str, Any] = field(default_factory=dict)


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
                    ref=str(value.get("ref") or ""),
                    preprocess=[str(v) for v in steps],
                    pipeline=pipeline,
                    source=str(value.get("source") or ""),
                    rows=[row for row in value.get("rows", []) if isinstance(row, dict)],
                    license=str(value.get("license") or ""),
                )
            )

    def select(
        self,
        priorities: dict[str, float],
        count: int,
        *,
        max_datasets: int | None = None,
        **options: Any,
    ) -> DatasetSelection:
        return select_datasets(
            self.datasets, priorities, count, max_datasets=max_datasets, **options
        )


#: Loaders that read the rows from `path`.
_FILE_LOADERS = frozenset({"jsonl", "parquet"})


def dataset_available(spec: DatasetSpec) -> bool:
    """A file-backed dataset without local rows is declared only; it cannot be used."""
    return bool(spec.path) or spec.loader not in _FILE_LOADERS


def _unit(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be in [0, 1]") from exc
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ConfigError(f"{name} must be in [0, 1]")
    return number


def select_datasets(
    datasets: list[DatasetSpec],
    priorities: dict[str, float],
    count: int,
    *,
    max_datasets: int | None = None,
    under_weight: float = 4.0,
    over_weight: float = 1.0,
    other_over_weight: float = 1.0,
    longtail_threshold: float = 0.1,
    longtail_under_weight: float = 8.0,
    dataset_cost_weight: float = 0.01,
    longtail_tolerance: float = 1e-6,
    diversity_weight: float = 0.0,
    diversity_threshold: float = 0.05,
    diversity_floor: float = 0.01,
) -> DatasetSelection:
    """Fit dataset proportions to a target ability distribution with LP/MILP.

    Coverage scores estimate each dataset's composition over the requested
    abilities plus one aggregate ``__other__`` dimension for all unrequested
    abilities. The first stage minimizes the worst relative long-tail shortfall.
    The second minimizes asymmetric relative L1 error, with larger penalties for
    under-covering long-tail abilities and for displacing targets with ``__other__``.
    ``count`` is the required sample total; ``max_datasets`` adds binary variables.
    With a positive ``diversity_weight``, catalog abilities outside the target
    that reach ``diversity_floor`` of the batch also enter the objective.
    """
    if count < 0:
        raise ValueError("dataset sample count must be non-negative")
    if max_datasets is not None and max_datasets <= 0:
        raise ValueError("max_datasets must be positive")
    weights = (under_weight, over_weight, other_over_weight, longtail_under_weight, dataset_cost_weight)
    if not all(math.isfinite(value) and value >= 0 for value in weights):
        raise ValueError("dataset selection weights must be finite and non-negative")
    if under_weight == 0 or longtail_under_weight < 1:
        raise ValueError("under_weight must be positive and longtail_under_weight must be at least 1")
    if not 0 <= longtail_threshold <= 1:
        raise ValueError("longtail_threshold must be within [0, 1]")
    if not math.isfinite(longtail_tolerance) or longtail_tolerance < 0:
        raise ValueError("longtail_tolerance must be finite and non-negative")
    if not math.isfinite(diversity_weight) or diversity_weight < 0:
        raise ValueError("diversity_weight must be finite and non-negative")
    if not 0 <= diversity_threshold <= 1:
        raise ValueError("diversity_threshold must be within [0, 1]")
    if not 0 <= diversity_floor <= 1:
        raise ValueError("diversity_floor must be within [0, 1]")

    requested = {
        key: value
        for key, value in (
            (key, _unit(value, f"priority.{key}")) for key, value in priorities.items()
        )
        if value > 0
    }
    # A file-backed dataset without local rows is declared only; it cannot serve a request.
    available = [item for item in datasets if dataset_available(item)]
    candidates = [
        item for item in available if any(item.abilities.get(ability, 0.0) > 0 for ability in requested)
    ]
    if count == 0 or not requested or not candidates:
        target_shares = _normalized_shares(requested)
        return DatasetSelection(
            datasets=[],
            ratios={},
            uncovered=sorted(requested),
            stats={
                "target_distribution": target_shares,
                "actual_distribution": dict.fromkeys(requested, 0.0),
                "gap": float(sum(target_shares.values())),
                "longtail_gap": 0.0,
                "diversity": {"score": 0.0, "covered": [], "available": []},
                "composition_estimated_from_coverage": True,
            },
        )

    capabilities = sorted(requested) + ["__other__"]
    target = np.array([requested[name] for name in capabilities[:-1]] + [0.0], dtype=float)
    target /= target.sum()
    coverage = np.zeros((len(candidates), len(capabilities)))
    dataset_totals: list[float] = []
    for dataset_index, item in enumerate(candidates):
        total_coverage = sum(max(0.0, value) for value in item.abilities.values())
        dataset_totals.append(total_coverage)
        for capability_index, name in enumerate(capabilities[:-1]):
            coverage[dataset_index, capability_index] = max(0.0, item.abilities.get(name, 0.0)) / total_coverage
        coverage[dataset_index, -1] = max(0.0, 1.0 - coverage[dataset_index, :-1].sum())
    other_abilities = [] if diversity_weight <= 0 else sorted(
        {
            ability
            for item in candidates
            for ability, value in item.abilities.items()
            if ability not in requested and value >= max(diversity_threshold, 1e-12)
        }
    )
    diversity_enabled = diversity_weight > 0 and bool(other_abilities)
    is_longtail = (target > 0) & (target <= longtail_threshold)
    scales = np.maximum(count * target, 1.0)
    scales[-1] = float(count)

    dataset_count = len(candidates)
    capability_count = len(capabilities)
    z_index = dataset_count + 2 * capability_count
    binary_start = z_index + 1
    use_binaries = max_datasets is not None and max_datasets < dataset_count
    diversity_start = binary_start + (dataset_count if use_binaries else 0)
    diversity_count = len(other_abilities) if diversity_enabled else 0
    variable_count = diversity_start + diversity_count

    rows: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []

    def add_constraint(coefficients: dict[int, float], low: float, high: float) -> None:
        row = np.zeros(variable_count)
        for index, value in coefficients.items():
            row[index] = value
        rows.append(row)
        lower.append(low)
        upper.append(high)

    add_constraint(dict(enumerate(np.ones(dataset_count))), count, count)
    for capability_index in range(capability_count):
        coefficients = {
            dataset_index: coverage[dataset_index, capability_index]
            for dataset_index in range(dataset_count)
        }
        coefficients[dataset_count + capability_index] = 1.0
        coefficients[dataset_count + capability_count + capability_index] = -1.0
        add_constraint(coefficients, count * target[capability_index], count * target[capability_index])
    for capability_index in np.flatnonzero(is_longtail):
        add_constraint(
            {
                dataset_count + int(capability_index): 1.0 / scales[capability_index],
                z_index: -1.0,
            },
            -np.inf,
            0.0,
        )
    if use_binaries:
        for dataset_index in range(dataset_count):
            add_constraint(
                {dataset_index: 1.0, binary_start + dataset_index: -float(count)}, -np.inf, 0.0
            )
        add_constraint(
            {binary_start + index: 1.0 for index in range(dataset_count)},
            -np.inf,
            float(max_datasets),
        )
    if diversity_enabled:
        floor = max(1e-9, diversity_floor * count)
        for diversity_index, ability in enumerate(other_abilities):
            coefficients = {
                dataset_index: max(0.0, item.abilities.get(ability, 0.0)) / dataset_totals[dataset_index]
                for dataset_index, item in enumerate(candidates)
                if dataset_totals[dataset_index] > 0
            }
            coefficients[diversity_start + diversity_index] = -floor
            add_constraint(coefficients, 0.0, np.inf)

    constraints = LinearConstraint(np.vstack(rows), np.array(lower), np.array(upper))
    variable_lower = np.zeros(variable_count)
    variable_upper = np.full(variable_count, np.inf)
    variable_upper[:dataset_count] = count
    if use_binaries:
        variable_upper[binary_start : binary_start + dataset_count] = 1.0
    if diversity_enabled:
        variable_upper[diversity_start:] = 1.0
    bounds = Bounds(variable_lower, variable_upper)
    integrality = np.zeros(variable_count, dtype=int)
    if use_binaries:
        integrality[binary_start : binary_start + dataset_count] = 1
    if diversity_enabled:
        integrality[diversity_start:] = 1

    tail_limit = 0.0
    if is_longtail.any():
        stage_one_objective = np.zeros(variable_count)
        stage_one_objective[z_index] = 1.0
        stage_one = milp(
            stage_one_objective, integrality=integrality, bounds=bounds, constraints=constraints
        )
        if not stage_one.success:
            raise RuntimeError(f"long-tail dataset optimization failed: {stage_one.message}")
        tail_limit = float(stage_one.x[z_index]) + longtail_tolerance

    stage_two_bounds = Bounds(variable_lower, variable_upper.copy())
    stage_two_bounds.ub[z_index] = tail_limit
    objective = np.zeros(variable_count)
    objective[:dataset_count] = np.array(
        [dataset_cost_weight * (1.0 - item.priority) / count for item in candidates]
    )
    for capability_index in range(capability_count):
        tail_multiplier = longtail_under_weight if is_longtail[capability_index] else 1.0
        objective[dataset_count + capability_index] = (
            under_weight * tail_multiplier / scales[capability_index]
        )
        excess_weight = other_over_weight if capability_index == capability_count - 1 else over_weight
        objective[dataset_count + capability_count + capability_index] = (
            excess_weight / scales[capability_index]
        )
    if diversity_enabled:
        objective[diversity_start:] = -diversity_weight / diversity_count
    result = milp(objective, integrality=integrality, bounds=stage_two_bounds, constraints=constraints)
    if not result.success:
        raise RuntimeError(f"dataset mixture optimization failed: {result.message}")

    ratios = {
        item.name: max(0.0, float(result.x[index]) / count)
        for index, item in enumerate(candidates)
        if result.x[index] > 1e-8
    }
    ratio_vector = np.array([max(0.0, result.x[index]) / count for index in range(dataset_count)])
    actual_vector = ratio_vector @ coverage
    actual = {name: float(actual_vector[index]) for index, name in enumerate(capabilities)}
    shortfall = np.maximum(target - actual_vector, 0.0)
    excess = np.maximum(actual_vector - target, 0.0)
    gap = float(sum(shortfall[index] for index in range(len(requested))))
    longtail_gap = float(sum(shortfall[index] for index in np.flatnonzero(is_longtail)))
    covered = [
        ability
        for index, ability in enumerate(other_abilities)
        if diversity_enabled and result.x[diversity_start + index] > 0.5
    ]
    selected = [item for index, item in enumerate(candidates) if ratio_vector[index] > 1e-8]
    return DatasetSelection(
        datasets=selected,
        ratios=ratios,
        uncovered=[
            name for index, name in enumerate(capabilities[:-1]) if actual_vector[index] <= 1e-8
        ],
        stats={
            "solver": "scipy.optimize.milp",
            "objective": float(result.fun),
            "sample_total": count,
            "target_distribution": {name: float(target[index]) for index, name in enumerate(capabilities)},
            "actual_distribution": actual,
            "shortfall": {name: float(shortfall[index]) for index, name in enumerate(capabilities)},
            "excess": {name: float(excess[index]) for index, name in enumerate(capabilities)},
            "gap": gap,
            "longtail_gap": longtail_gap,
            "diversity": {
                "score": len(covered) / len(other_abilities) if other_abilities else 0.0,
                "weight": diversity_weight,
                "covered": covered,
                "available": other_abilities,
            },
            "longtail_capabilities": [name for index, name in enumerate(capabilities) if is_longtail[index]],
            "longtail_max_relative_shortfall": float(
                max((count * shortfall[index] / scales[index] for index in np.flatnonzero(is_longtail)), default=0.0)
            ),
            "composition_estimated_from_coverage": True,
        },
    )


def _normalized_shares(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()} if total else dict.fromkeys(weights, 0.0)


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


#: Built-in and user-defined row loaders, keyed by the registry ``loader`` value.
_DATASET_LOADERS: dict[str, Callable[[DatasetSpec, Path], list[dict[str, Any]]]] = {}


def register_dataset_loader(name: str):
    """Decorator that registers a dataset row loader.

    A private dataset can name a registered loader or a dotted callable
    (``module:function``) in its registry entry, without editing this module.
    """

    def decorator(func: Callable[[DatasetSpec, Path], list[dict[str, Any]]]):
        _DATASET_LOADERS[name] = func
        return func

    return decorator


def dataset_loader_names() -> list[str]:
    return sorted(_DATASET_LOADERS)


def _import_dataset_loader(path: str) -> Callable[[DatasetSpec, Path], list[dict[str, Any]]]:
    module_name, _, attr = path.partition(":")
    if not attr:
        module_name, _, attr = path.rpartition(".")
    if not module_name or not attr:
        raise ConfigError(f"Invalid dataset loader path: {path!r}")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ConfigError(f"Cannot import dataset loader module {module_name!r}: {exc}") from exc
    func = getattr(module, attr, None)
    if not callable(func):
        raise ConfigError(f"Dataset loader {path!r} is not a callable")
    return func


def fetch_dataset_rows(spec: DatasetSpec, root: str | Path) -> list[dict[str, Any]]:
    """Fetch rows through a registered or dotted-path loader.

    Built-in loaders cover ``jsonl``, ``parquet``, ``web``, ``huggingface``,
    ``github`` and ``synthetic``. A registry entry may also name
    ``package.module:function``; the callable receives ``(spec, root)`` and
    returns row mappings.
    """
    loader = _DATASET_LOADERS.get(spec.loader)
    if loader is None:
        if ":" not in spec.loader and "." not in spec.loader:
            available = ", ".join(dataset_loader_names())
            raise ConfigError(
                f"Unknown dataset loader {spec.loader!r} for {spec.name!r}; available: {available}"
            )
        loader = _import_dataset_loader(spec.loader)
    rows = loader(spec, Path(root))
    return [_adapt_verl_row(row) for row in rows if isinstance(row, dict)]


@register_dataset_loader("synthetic")
def _load_synthetic_rows(spec: DatasetSpec, root: Path) -> list[dict[str, Any]]:
    return list(spec.rows)


@register_dataset_loader("jsonl")
def _load_jsonl_rows(spec: DatasetSpec, root: Path) -> list[dict[str, Any]]:
    return read_jsonl(_resolve_path(spec, root))


@register_dataset_loader("parquet")
def _load_parquet_rows(spec: DatasetSpec, root: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise AdapterError("Parquet datasets require the optional 'pyarrow' package") from exc
    return pq.read_table(_resolve_path(spec, root)).to_pylist()


@register_dataset_loader("web")
def _load_web_rows(spec: DatasetSpec, root: Path) -> list[dict[str, Any]]:
    if not spec.url:
        raise ConfigError(f"Web dataset {spec.name!r} requires url")
    with urllib.request.urlopen(spec.url, timeout=30) as response:
        payload = response.read().decode("utf-8")
    if spec.url.lower().split("?", 1)[0].endswith(".json"):
        value = json.loads(payload)
        return value if isinstance(value, list) else value.get("data", [])
    return [json.loads(line) for line in payload.splitlines() if line.strip()]


@register_dataset_loader("huggingface")
def _load_huggingface_rows(spec: DatasetSpec, root: Path) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise AdapterError("Hugging Face datasets require the optional 'datasets' package") from exc
    dataset = load_dataset(spec.url, spec.config, split=spec.split)
    return [dict(row) for row in dataset]


_GITHUB_RAW = "https://raw.githubusercontent.com/"
_GITHUB_ROW_SUFFIXES = (".json", ".jsonl", ".ndjson", ".csv")
_GITHUB_MAX_FILES = 32


def _github_parts(spec: DatasetSpec) -> tuple[str, str, str, str, str | None]:
    """Resolve a GitHub spec to (owner, repo, ref, path, raw file url)."""
    url = spec.url.strip().rstrip("/")
    if not url:
        raise ConfigError(f"GitHub dataset {spec.name!r} requires url")
    if url.startswith(_GITHUB_RAW):
        parts = url[len(_GITHUB_RAW) :].split("/", 3)
        if len(parts) < 4:
            raise ConfigError(f"Invalid raw GitHub URL: {spec.url}")
        owner, repo, ref, path = parts
        return owner, repo, ref, path, url
    if "github.com/" in url:
        parts = url.split("github.com/", 1)[1].split("/", 3)
        if len(parts) < 2 or not all(parts[:2]):
            raise ConfigError(f"Invalid GitHub URL: {spec.url}")
        owner, repo = parts[0], parts[1]
        rest = "/".join(parts[2:])
        if rest.startswith("blob/"):
            ref, _, path = rest[len("blob/") :].partition("/")
            raw = f"{_GITHUB_RAW}{owner}/{repo}/{ref}/{urllib.parse.quote(path)}"
            return owner, repo, ref, path, raw
        if rest.startswith("tree/"):
            ref, _, path = rest[len("tree/") :].partition("/")
            return owner, repo, ref, path, None
        return owner, repo, spec.ref or "main", spec.path, None
    parts = url.split("/")
    if len(parts) == 2 and all(parts):
        return parts[0], parts[1], spec.ref or "main", spec.path, None
    raise ConfigError(
        f"Unsupported GitHub URL {spec.url!r}; use owner/repo, a github.com blob/tree URL, "
        "or a raw.githubusercontent.com URL"
    )


def _github_get(url: str) -> bytes:
    headers = {"User-Agent": "RL-RSI", "Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("RSI_GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def _rows_from_text(text: str, suffix: str, name: str) -> list[dict[str, Any]]:
    suffix = suffix.lower()
    if suffix == ".csv":
        return [dict(row) for row in csv.DictReader(io.StringIO(text))]
    if suffix in {".jsonl", ".ndjson"}:
        rows = []
        for line in text.splitlines():
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
        return rows
    value = json.loads(text)
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if isinstance(value, dict):
        for key in ("data", "rows", "examples"):
            if isinstance(value.get(key), list):
                return [row for row in value[key] if isinstance(row, dict)]
    raise AdapterError(f"GitHub file {name!r} does not contain a row list")


def _github_rows_from_file(download_url: str, path: str) -> list[dict[str, Any]]:
    suffix = Path(path).suffix
    if suffix.lower() not in _GITHUB_ROW_SUFFIXES:
        raise AdapterError(
            f"Unsupported GitHub dataset file {path!r}; supported: {', '.join(_GITHUB_ROW_SUFFIXES)}"
        )
    return _rows_from_text(_github_get(download_url).decode("utf-8"), suffix, path)


def _github_list_files(owner: str, repo: str, ref: str, path: str, depth: int = 0) -> list[dict[str, Any]]:
    """List row files under a repo path through the contents API."""
    api = (
        f"https://api.github.com/repos/{owner}/{repo}/contents/"
        f"{urllib.parse.quote(path)}?ref={urllib.parse.quote(ref)}"
    )
    value = json.loads(_github_get(api).decode("utf-8"))
    if isinstance(value, dict):
        if value.get("type") == "file":
            return [value]
        raise AdapterError(f"GitHub path {path!r} is not a file or directory listing")
    if not isinstance(value, list):
        raise AdapterError(f"Unexpected GitHub contents response for {path!r}")
    files: list[dict[str, Any]] = []
    for entry in value:
        if not isinstance(entry, dict):
            continue
        if entry.get("type") == "file" and Path(str(entry.get("path", ""))).suffix.lower() in _GITHUB_ROW_SUFFIXES:
            files.append(entry)
        elif entry.get("type") == "dir" and depth < 3:
            files.extend(
                _github_list_files(owner, repo, ref, str(entry.get("path", "")), depth + 1)
            )
        if len(files) >= _GITHUB_MAX_FILES:
            return files[:_GITHUB_MAX_FILES]
    return files


@register_dataset_loader("github")
def _load_github_rows(spec: DatasetSpec, root: Path) -> list[dict[str, Any]]:
    """Read JSON / JSONL / CSV rows from a GitHub repository.

    ``url`` accepts ``owner/repo``, a ``github.com`` blob or tree URL, or a
    ``raw.githubusercontent.com`` URL. With ``owner/repo`` (or a tree URL),
    ``path`` is the file or directory inside the repo and ``ref`` the branch or
    tag. Directory listings use the GitHub contents API and stop after
    ``_GITHUB_MAX_FILES`` row files. Set ``GITHUB_TOKEN`` for higher rate limits.
    """
    owner, repo, ref, path, raw_url = _github_parts(spec)
    if raw_url:
        return _github_rows_from_file(raw_url, path)
    if Path(path).suffix.lower() in _GITHUB_ROW_SUFFIXES:
        raw = f"{_GITHUB_RAW}{owner}/{repo}/{ref}/{urllib.parse.quote(path)}"
        return _github_rows_from_file(raw, path)
    rows: list[dict[str, Any]] = []
    for entry in _github_list_files(owner, repo, ref, path):
        download_url = entry.get("download_url")
        if not download_url:
            continue
        rows.extend(_github_rows_from_file(str(download_url), str(entry.get("path", ""))))
    return rows


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
                          "prompt_only": not response, "reward_hint": spec.reward,
                          **dict(row.get("metadata") or {})},
            )
        )
        if len(samples) >= limit:
            break
    return samples


def fill_dataset_quotas(
    sources: list[tuple[DatasetSpec, list[DataSample]]], request_count: int, quotas: dict[str, int]
) -> tuple[list[DataSample], dict[str, int]]:
    """Take each dataset's quota, then fill a shortfall from the rows the datasets have left.

    ``sources`` holds each selected dataset with all of its usable samples, in selection
    order. A dataset that yields less than its quota leaves a gap; the gap is filled from
    the spare rows of the other datasets, best-contributing source first.
    """
    chosen: list[DataSample] = []
    counts: dict[str, int] = {}
    spare: list[tuple[str, list[DataSample]]] = []
    for spec, usable in sources:
        quota = quotas.get(spec.name, 0)
        taken = usable[:quota]
        chosen.extend(taken)
        counts[spec.name] = len(taken)
        if len(usable) > quota:
            spare.append((spec.name, usable[quota:]))
    deficit = request_count - len(chosen)
    for name, rest in spare:
        if deficit <= 0:
            break
        extra = rest[:deficit]
        chosen.extend(extra)
        counts[name] += len(extra)
        deficit -= len(extra)
    return chosen, counts
