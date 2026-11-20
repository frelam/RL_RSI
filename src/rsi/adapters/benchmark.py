from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Any

from rsi.adapters.templating import format_template
from rsi.core.errors import AdapterError
from rsi.core.interfaces import BenchmarkBackend
from rsi.core.jsonio import now_iso, read_json, read_jsonl
from rsi.core.registry import register
from rsi.core.scoring import pass_at_k, score_text
from rsi.core.types import (
    BenchmarkItem,
    BenchmarkReport,
    ChatMessage,
    ChatRequest,
    Checkpoint,
    ItemResult,
)


def aggregate_items(
    items: list[ItemResult],
    *,
    samples_per_item: int = 1,
    pass_k: int = 1,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    capabilities: dict[str, list[float]] = {}
    subcapabilities: dict[str, list[float]] = {}
    for item in items:
        capabilities.setdefault(item.capability, []).append(item.score)
        if item.subcapability:
            key = f"{item.capability}.{item.subcapability}"
            subcapabilities.setdefault(key, []).append(item.score)
    cap_scores = {key: sum(vals) / len(vals) for key, vals in capabilities.items()}
    sub_scores = {key: sum(vals) / len(vals) for key, vals in subcapabilities.items()}
    mean_score = sum(item.score for item in items) / len(items) if items else 0.0
    pass_rate = sum(1 for item in items if item.passed) / len(items) if items else 0.0
    avg_len = sum(len(item.response) for item in items) / len(items) if items else 0.0
    aggregates = {
        "mean_score": mean_score,
        "pass_rate": pass_rate,
        "pass@1": pass_rate,
        "avg_response_len": avg_len,
        "num_items": float(len(items)),
    }
    if samples_per_item > 1:
        k = max(1, min(pass_k, samples_per_item))
        aggregates[f"pass@{k}"] = (
            sum(pass_at_k(samples_per_item, int(round(item.score * samples_per_item)), k) for item in items)
            / len(items)
            if items
            else 0.0
        )
    return cap_scores, sub_scores, aggregates


@register("benchmark", "mock")
class MockBenchmark(BenchmarkBackend):
    """Deterministic score curve keyed by checkpoint version.

    Config::

        type: mock
        items_per_capability: 6
        pass_threshold: 0.5
        max_gain_versions: 4
        capabilities:
          math.reasoning: {base: 0.42, gain: 0.08}
    """

    def __init__(self, runtime: Any, cfg: Any, **params: Any) -> None:
        super().__init__(runtime, cfg, **params)
        self.name = str(self.params.get("instance_name") or self.cfg.get("name", "mock"))
        self.items_per_capability = self.cfg.get_int("items_per_capability", 6)
        self.pass_threshold = self.cfg.get_float("pass_threshold", 0.5)
        self.max_gain_versions = self.cfg.get_int("max_gain_versions", 4)
        self.noise = self.cfg.get_float("noise", 0.005)
        self.capabilities_cfg = self.cfg.get_dict("capabilities", {})
        self.subcapabilities = [str(v) for v in self.cfg.get_list("subcapabilities", [])]
        self.primary_metric = str(self.cfg.get("primary_metric", "mean_score"))

    def evaluate(self, checkpoint: Checkpoint, round_no: int) -> BenchmarkReport:
        if not self.capabilities_cfg:
            raise AdapterError(f"MockBenchmark {self.name!r} has no capabilities configured")
        rng = self.runtime.rng("benchmark", self.name, checkpoint.id)
        items: list[ItemResult] = []
        for capability, raw in self.capabilities_cfg.items():
            spec = raw if isinstance(raw, dict) else {"base": float(raw)}
            base = float(spec.get("base", 0.4))
            gain = float(spec.get("gain", 0.08))
            version_gain = gain * min(checkpoint.version, self.max_gain_versions)
            for index in range(self.items_per_capability):
                noise = self.noise * (rng.random() * 2 - 1)
                score = max(0.0, min(1.0, base + version_gain + noise))
                sub = self.subcapabilities[index % len(self.subcapabilities)] if self.subcapabilities else ""
                items.append(
                    ItemResult(
                        item_id=f"{capability}#{index}",
                        capability=capability,
                        subcapability=sub,
                        score=score,
                        passed=score >= self.pass_threshold,
                        response="mock response " + ("x" * int(80 + rng.random() * 120)),
                        num_samples=1,
                        latency_s=0.01,
                        metadata={"difficulty": float(spec.get("difficulty", 0.5))},
                    )
                )
        cap_scores, sub_scores, aggregates = aggregate_items(items)
        aggregates[f"pass@{1}"] = aggregates["pass_rate"]
        return BenchmarkReport(
            benchmark=self.name,
            checkpoint_id=checkpoint.id,
            round=round_no,
            items=items,
            capabilities=cap_scores,
            subcapabilities=sub_scores,
            aggregates=aggregates,
            num_items=len(items),
            primary_metric=self.primary_metric,
            started_at=now_iso(),
            finished_at=now_iso(),
            metadata={"mock": True, "checkpoint_version": checkpoint.version},
        )


@register("benchmark", "jsonl")
class JsonlBenchmark(BenchmarkBackend):
    """Evaluate a local JSONL benchmark through the policy endpoint."""

    def __init__(self, runtime: Any, cfg: Any, **params: Any) -> None:
        super().__init__(runtime, cfg, **params)
        self.name = str(self.params.get("instance_name") or self.cfg.get("name", "jsonl"))
        self.path = self.runtime.path(self.cfg.require("path").get("path"))
        self.max_items = self.cfg.get_int("max_items", 0)
        self.samples_per_item = self.cfg.get_int("samples_per_item", 1)
        self.k = self.cfg.get_int("k", self.samples_per_item)
        self.temperature = self.cfg.get_float("temperature", 0.7)
        self.max_tokens = self.cfg.get_int("max_tokens", 1024)
        self.default_scorer = str(self.cfg.get("scorer", "contains"))
        self.pass_threshold = self.cfg.get_float("pass_threshold", 0.5)
        self.primary_metric = str(self.cfg.get("primary_metric", "mean_score"))
        self.store_responses = self.cfg.get_bool("store_responses", False)

    def _load_items(self) -> list[BenchmarkItem]:
        rows = read_jsonl(self.path)
        if self.max_items > 0:
            rows = rows[: self.max_items]
        items: list[BenchmarkItem] = []
        for index, row in enumerate(rows):
            prompt = str(row.get("prompt") or "")
            messages = row.get("messages") or []
            if not prompt and messages:
                prompt = str(messages[-1].get("content", ""))
            items.append(
                BenchmarkItem(
                    id=str(row.get("id", f"item-{index}")),
                    prompt=prompt,
                    capability=str(row.get("capability", "general")),
                    subcapability=str(row.get("subcapability", "")),
                    reference=row.get("reference"),
                    scorer=str(row.get("scorer", self.default_scorer)),
                    metadata=dict(row.get("metadata") or {}),
                )
            )
        return items

    def evaluate(self, checkpoint: Checkpoint, round_no: int) -> BenchmarkReport:
        inference = self.runtime.inference()
        inference.ensure(checkpoint)
        items = self._load_items()
        results: list[ItemResult] = []
        started = now_iso()
        for index, item in enumerate(items):
            request = ChatRequest(
                messages=[ChatMessage(role="user", content=item.prompt)],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                n=max(1, self.samples_per_item),
            )
            try:
                response = inference.chat(request)
                scores = [
                    score_text(text, item.reference, item.scorer, item.metadata) for text in response.texts
                ]
                score = sum(scores) / len(scores) if scores else 0.0
                best = max(scores) if scores else 0.0
                results.append(
                    ItemResult(
                        item_id=item.id,
                        capability=item.capability,
                        subcapability=item.subcapability,
                        score=score,
                        passed=best >= self.pass_threshold,
                        response=(response.texts[0] if response.texts else "")[: 4000 if self.store_responses else 500],
                        expected=item.reference,
                        num_samples=len(response.texts),
                        latency_s=response.latency_s,
                        metadata=dict(item.metadata),
                    )
                )
            except Exception as exc:  # noqa: BLE001 - one bad item should not kill the eval
                results.append(
                    ItemResult(
                        item_id=item.id,
                        capability=item.capability,
                        subcapability=item.subcapability,
                        score=0.0,
                        passed=False,
                        expected=item.reference,
                        error=str(exc)[:1000],
                        metadata=dict(item.metadata),
                    )
                )
            if (index + 1) % 20 == 0:
                self.logger.info("benchmark %s: %d/%d items", self.name, index + 1, len(items))
        cap_scores, sub_scores, aggregates = aggregate_items(
            results, samples_per_item=max(1, self.samples_per_item), pass_k=self.k
        )
        return BenchmarkReport(
            benchmark=self.name,
            checkpoint_id=checkpoint.id,
            round=round_no,
            items=results,
            capabilities=cap_scores,
            subcapabilities=sub_scores,
            aggregates=aggregates,
            num_items=len(results),
            primary_metric=self.primary_metric,
            started_at=started,
            finished_at=now_iso(),
            metadata={"path": str(self.path)},
        )


@register("benchmark", "evalscope")
class EvalScopeBenchmark(BenchmarkBackend):
    """Adapter boundary for EvalScope or any CLI evaluator producing JSON.

    The command may use ``{base_url}``, ``{model}``, ``{checkpoint_path}``,
    ``{output_dir}``, ``{round}`` and ``{seed}``. The JSON output is read from
    ``output_path`` and flattened into per-benchmark scores.
    """

    def __init__(self, runtime: Any, cfg: Any, **params: Any) -> None:
        super().__init__(runtime, cfg, **params)
        self.name = str(self.params.get("instance_name") or self.cfg.get("name", "evalscope"))
        self.command = self.cfg.get("command")
        if not self.command:
            raise AdapterError("evalscope benchmark requires 'command'")
        self.output_path_template = str(self.cfg.get("output_path", "{output_dir}/results.json"))
        self.timeout_s = self.cfg.get_float("timeout_s", 7200.0)
        self.capability_map = self.cfg.get_dict("capability_map", {})
        self.pass_threshold = self.cfg.get_float("pass_threshold", 0.5)
        self.primary_metric = str(self.cfg.get("primary_metric", "mean_score"))
        self.env = self.cfg.get_dict("env", {})

    def evaluate(self, checkpoint: Checkpoint, round_no: int) -> BenchmarkReport:
        endpoint = self.runtime.inference().ensure(checkpoint)
        output_dir = self.runtime.artifacts.stage_dir(round_no, "collect") / f"bench-{self.name}"
        output_dir.mkdir(parents=True, exist_ok=True)
        variables = {
            "base_url": endpoint.base_url,
            "model": endpoint.model,
            "checkpoint_path": checkpoint.model_path,
            "output_dir": str(output_dir),
            "round": round_no,
            "seed": self.runtime.seed,
        }
        command = format_template(self.command, variables)
        output_path = Path(format_template(self.output_path_template, variables))
        self.runtime.executor.run(
            command,
            cwd=self.runtime.run_dir,
            env=self.env,
            timeout=self.timeout_s,
            label=f"benchmark.{self.name}",
            check=True,
        )
        if not output_path.exists():
            raise AdapterError(f"EvalScope did not write results to {output_path}")
        raw = read_json(output_path)
        metrics = raw.get("results") or raw.get("metrics") or raw
        if not isinstance(metrics, dict):
            raise AdapterError("EvalScope output must contain a results mapping")
        items: list[ItemResult] = []
        for metric_name, value in metrics.items():
            score = _extract_score(value)
            if score is None:
                continue
            if score > 1.5:
                score /= 100.0
            capability = str(self.capability_map.get(metric_name, metric_name))
            items.append(
                ItemResult(
                    item_id=str(metric_name),
                    capability=capability,
                    score=max(0.0, min(1.0, score)),
                    passed=score >= self.pass_threshold,
                    num_samples=1,
                )
            )
        if not items:
            raise AdapterError("No numeric scores found in EvalScope output")
        cap_scores, sub_scores, aggregates = aggregate_items(items)
        return BenchmarkReport(
            benchmark=self.name,
            checkpoint_id=checkpoint.id,
            round=round_no,
            items=items,
            capabilities=cap_scores,
            subcapabilities=sub_scores,
            aggregates=aggregates,
            num_items=len(items),
            primary_metric=self.primary_metric,
            started_at=now_iso(),
            finished_at=now_iso(),
            metadata={"results_path": str(output_path)},
        )


def _extract_score(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, dict):
        for key in ("score", "mean_score", "accuracy", "acc", "pass@1", "pass_rate", "avg"):
            if key in value and isinstance(value[key], (int, float)):
                return float(value[key])
        for sub in value.values():
            extracted = _extract_score(sub)
            if extracted is not None:
                return extracted
    return None
