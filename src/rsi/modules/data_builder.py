from __future__ import annotations

import hashlib
import math
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any

from rsi.core.interfaces import DatasetBuilder
from rsi.core.jsonio import atomic_write_json, now_iso, stable_hash, write_jsonl
from rsi.core.registry import register
from rsi.core.types import DataPlan, DataSample, DatasetArtifact
from rsi.modules.data_mixing import group_key
from rsi.modules.data_quotas import target_shares


@register("dataset_builder", "jsonl")
class JsonlDatasetBuilder(DatasetBuilder):
    """Write train/val JSONL, reward spec, manifest and statistics.

    ``schema`` selects the row view:
      - ``prompt_response``: flat, framework-neutral
      - ``messages``: chat messages
      - ``verl_jsonl``: a readable verl adapter view (parquet conversion is an
        external, version-specific adapter boundary)
    """

    default_schema = "prompt_response"

    def __init__(self, runtime: Any, cfg: Any, **params: Any) -> None:
        super().__init__(runtime, cfg, **params)
        self.schema = str(self.cfg.get("schema", self.default_schema))
        self.val_ratio = self.cfg.get_float("val_ratio", 0.05)
        self.val_min = self.cfg.get_int("val_min", 1)
        self.val_max = self.cfg.get_int("val_max", 64)
        self.include_metadata = self.cfg.get_bool("include_metadata", True)

    def build(
        self,
        samples: list[DataSample],
        context: Any,
        plan: DataPlan,
        mixing: dict[str, float],
    ) -> DatasetArtifact:
        dataset_dir = Path(context.work_dir) / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        dataset_id = f"ds-r{context.round:04d}-{stable_hash(context.round, [s.id for s in samples], length=8)}"
        seed = stable_hash(self.runtime.seed, context.round, "dataset")
        ordered = sorted(samples, key=lambda sample: stable_hash(seed, sample.id))
        sampler_stats = getattr(context, "sampler_stats", {})
        shares = target_shares(mixing) if sampler_stats.get("ratio_feasible") else {}
        tolerance = Fraction(str(sampler_stats.get("ratio_tolerance", 0.0)))
        train_samples, val_samples = self._split_by_capability(ordered, seed, shares, tolerance)
        rows = [self._row(sample) for sample in ordered]
        train_path = write_jsonl(dataset_dir / "train.jsonl", [self._row(s) for s in train_samples])
        val_path = write_jsonl(dataset_dir / "val.jsonl", [self._row(s) for s in val_samples])
        all_path = write_jsonl(dataset_dir / "samples.jsonl", rows)
        reward_spec = self._reward_spec(samples, dataset_id)
        reward_path = atomic_write_json(dataset_dir / "reward_spec.json", reward_spec)
        stats = self._stats(samples, mixing)
        train_groups = Counter(map(group_key, train_samples))
        stats["train_group_ratio"] = {
            group: count / len(train_samples) for group, count in sorted(train_groups.items())
        }
        if "ratio_feasible" in sampler_stats:
            stats["sampler"] = sampler_stats
        manifest = {
            "dataset_id": dataset_id,
            "round": context.round,
            "seed": self.runtime.seed,
            "schema": self.schema,
            "created_at": now_iso(),
            "num_samples": len(samples),
            "num_train": len(train_samples),
            "num_val": len(val_samples),
            "mixing": mixing,
            "mixing_candidate": getattr(context, "mixing_info", {}).get("candidate"),
            "plan": {
                "strategy": plan.strategy,
                "requests": len(plan.requests),
                "total_target": plan.total_target,
            },
            "stats": stats,
            "builder": self.cfg.as_dict(),
            "files": {
                "train.jsonl": _sha256(train_path),
                "val.jsonl": _sha256(val_path),
                "reward_spec.json": _sha256(reward_path),
            },
            "all_samples": str(all_path.name),
        }
        manifest_path = atomic_write_json(dataset_dir / "manifest.json", manifest)
        stats_path = atomic_write_json(dataset_dir / "stats.json", stats)
        return DatasetArtifact(
            dataset_id=dataset_id,
            path=str(dataset_dir),
            train_path=str(train_path),
            val_path=str(val_path),
            reward_spec_path=str(reward_path),
            manifest_path=str(manifest_path),
            num_samples=len(samples),
            num_train=len(train_samples),
            num_val=len(val_samples),
            round=context.round,
            schema=self.schema,
            groups=stats["groups"],
            sources=stats["sources"],
            difficulty=stats["difficulty"],
            mixing=dict(mixing),
            stats={**stats, "stats_path": str(stats_path), "dataset_id": dataset_id},
        )

    def _val_count(self, total: int) -> int:
        if total <= 1:
            return 0
        count = int(round(total * self.val_ratio))
        count = max(self.val_min, count) if total > self.val_min else 0
        return max(0, min(count, self.val_max, total - 1))

    def _split_by_capability(
        self,
        samples: list[DataSample],
        seed: str,
        shares: dict[str, Fraction],
        tolerance: Fraction,
    ) -> tuple[list[DataSample], list[DataSample]]:
        groups: dict[str, list[DataSample]] = {}
        for sample in samples:
            groups.setdefault(sample.capability, []).append(sample)
        for name, items in groups.items():
            items.sort(key=lambda sample: stable_hash(seed, "split", name, sample.id))
        available = sum(max(0, len(items) - 1) for items in groups.values())
        target = min(self._val_count(len(samples)), available)
        counts = Counter(map(group_key, samples)) if shares else Counter()
        while True:
            quotas = self._validation_quotas(groups, len(samples), target)
            if target == 0 or not shares:
                break
            val_counts = Counter(
                group_key(sample) for name, items in groups.items() for sample in items[:quotas[name]]
            )
            if all(
                abs(Fraction(counts[group] - val_counts[group], len(samples) - target) - share)
                <= tolerance
                for group, share in shares.items()
            ):
                break
            target -= 1

        train: list[DataSample] = []
        validation: list[DataSample] = []
        for name, items in groups.items():
            validation.extend(items[:quotas[name]])
            train.extend(items[quotas[name]:])
        order = lambda sample: stable_hash(seed, "output", sample.id)
        return sorted(train, key=order), sorted(validation, key=order)

    def _validation_quotas(
        self, groups: dict[str, list[DataSample]], total: int, target: int
    ) -> dict[str, int]:
        ideals = {name: target * len(items) / total for name, items in groups.items()}
        quotas = {
            name: min(max(0, math.floor(ideals[name])), len(items) - 1)
            for name, items in groups.items()
        }
        remaining = target - sum(quotas.values())
        while remaining:
            available = [name for name, items in groups.items() if quotas[name] < len(items) - 1]
            if not available:
                break
            name = max(available, key=lambda group: (ideals[group] - quotas[group], group))
            quotas[name] += 1
            remaining -= 1
        return quotas

    def _row(self, sample: DataSample) -> dict[str, Any]:
        messages = [{"role": m.role, "content": m.content} for m in sample.messages]
        if not messages:
            messages = [
                {"role": "user", "content": sample.prompt},
                *([{"role": "assistant", "content": sample.response}] if sample.response else []),
            ]
        if self.schema == "verl_jsonl":
            row: dict[str, Any] = {
                "data_source": sample.source or f"rsi.{sample.kind}",
                "prompt": [{"role": "user", "content": sample.prompt}],
                "response": sample.response,
                "ability": sample.capability,
                "reward_model": {
                    "style": (sample.verifier or {}).get("type", "rule"),
                    "ground_truth": sample.reference or "",
                },
                "extra_info": {
                    "id": sample.id,
                    "kind": sample.kind,
                    "subcapability": sample.subcapability,
                    "difficulty": sample.difficulty,
                    "quality": sample.quality,
                    "verifier": sample.verifier,
                    "env_path": sample.env_path,
                    "round": sample.round,
                    "source": sample.source,
                },
            }
            return row
        row = {
            "id": sample.id,
            "kind": sample.kind,
            "capability": sample.capability,
            "subcapability": sample.subcapability,
            "prompt": sample.prompt,
            "response": sample.response,
            "reference": sample.reference,
            "verifier": sample.verifier,
            "source": sample.source,
            "group": sample.group,
            "difficulty": round(sample.difficulty, 4),
            "quality": round(sample.quality, 4),
            "reward": sample.reward,
            "round": sample.round,
            "env_path": sample.env_path,
        }
        if self.schema == "messages":
            row["messages"] = messages
        if self.include_metadata:
            row["metadata"] = sample.metadata
        return row

    def _reward_spec(self, samples: list[DataSample], dataset_id: str) -> dict[str, Any]:
        reward_cfg = self.cfg.get_dict("reward", {})
        default = reward_cfg.get("default") or {"type": "rule", "name": "contains", "weight": 1.0}
        overrides = reward_cfg.get("by_capability") or {}
        by_capability: dict[str, Any] = {}
        hints = {sample.capability: sample.metadata["reward_hint"]
                 for sample in samples if sample.metadata.get("reward_hint")}
        for capability in sorted({sample.capability for sample in samples}):
            rule = dict(self.runtime.catalog.reward(capability) or {})
            rule = {**default, **rule, **hints.get(capability, {})}
            if capability in overrides and isinstance(overrides[capability], dict):
                rule.update(overrides[capability])
            by_capability[capability] = rule
        return {
            "version": 1,
            "dataset_id": dataset_id,
            "default": default,
            "by_capability": by_capability,
        }

    def _stats(self, samples: list[DataSample], mixing: dict[str, float]) -> dict[str, Any]:
        groups: dict[str, int] = {}
        kinds: dict[str, int] = {}
        sources: dict[str, int] = {}
        buckets = {"easy": 0, "medium": 0, "hard": 0}
        for sample in samples:
            groups[sample.capability] = groups.get(sample.capability, 0) + 1
            kinds[sample.kind] = kinds.get(sample.kind, 0) + 1
            sources[sample.source] = sources.get(sample.source, 0) + 1
            if sample.difficulty < 0.33:
                buckets["easy"] += 1
            elif sample.difficulty < 0.66:
                buckets["medium"] += 1
            else:
                buckets["hard"] += 1
        difficulties = [sample.difficulty for sample in samples]
        qualities = [sample.quality for sample in samples]
        return {
            "groups": groups,
            "kinds": kinds,
            "sources": sources,
            "difficulty": {
                "mean": sum(difficulties) / len(difficulties) if difficulties else 0.0,
                "min": min(difficulties) if difficulties else 0.0,
                "max": max(difficulties) if difficulties else 0.0,
                "buckets": buckets,
            },
            "quality_mean": sum(qualities) / len(qualities) if qualities else 0.0,
            "mixing": mixing,
        }


@register("dataset_builder", "verl_jsonl")
class VerlJsonlDatasetBuilder(JsonlDatasetBuilder):
    default_schema = "verl_jsonl"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
