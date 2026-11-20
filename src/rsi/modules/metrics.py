from __future__ import annotations

import re
from typing import Any

from rsi.core.types import BenchmarkReport, CapabilityScore, ItemResult


def resolve_capability(item: ItemResult, benchmark: str, catalog: Any) -> str:
    """Map an unlabeled benchmark item to a capability via the catalog.

    Benchmarks should label items directly. For legacy suites, the catalog can
    declare ``benchmark: [{benchmark: math500, match: "algebra.*"}]`` rules.
    """
    if item.capability and item.capability != "general":
        return item.capability
    haystack = f"{item.item_id} {item.subcapability}"
    for capability in catalog.all():
        for rule in capability.get("benchmark", []):
            if not isinstance(rule, dict):
                continue
            target = rule.get("benchmark")
            if target and str(target) != benchmark:
                continue
            pattern = rule.get("match")
            if pattern and re.search(str(pattern), haystack, flags=re.IGNORECASE):
                return str(capability["name"])
    return item.capability or "general"


def capability_table(
    reports: list[BenchmarkReport],
    baseline: dict[str, float] | None = None,
    catalog: Any = None,
) -> dict[str, CapabilityScore]:
    baseline = baseline or {}
    scores: dict[str, list[float]] = {}
    passes: dict[str, list[bool]] = {}
    for report in reports:
        for item in report.items:
            capability = resolve_capability(item, report.benchmark, catalog) if catalog else item.capability
            scores.setdefault(capability, []).append(float(item.score))
            passes.setdefault(capability, []).append(bool(item.passed))
    table: dict[str, CapabilityScore] = {}
    for capability, values in scores.items():
        score = sum(values) / len(values)
        base = baseline.get(capability)
        table[capability] = CapabilityScore(
            capability=capability,
            score=score,
            baseline=base,
            delta=(score - base) if base is not None else None,
            num_items=len(values),
            pass_rate=sum(passes.get(capability, [])) / max(1, len(passes.get(capability, []))),
        )
    return table


def overall_metrics(reports: list[BenchmarkReport], table: dict[str, CapabilityScore]) -> dict[str, float]:
    items = [item for report in reports for item in report.items]
    total = len(items)
    mean_score = sum(item.score for item in items) / total if total else 0.0
    pass_rate = sum(1 for item in items if item.passed) / total if total else 0.0
    avg_len = sum(len(item.response) for item in items) / total if total else 0.0
    return {
        "mean_score": mean_score,
        "pass_rate": pass_rate,
        "avg_response_len": avg_len,
        "num_items": float(total),
        "num_capabilities": float(len(table)),
        "num_reports": float(len(reports)),
    }


def worst_items(
    reports: list[BenchmarkReport],
    catalog: Any,
    limit: int = 8,
    max_score: float = 0.75,
) -> list[dict[str, Any]]:
    rows: list[tuple[float, dict[str, Any]]] = []
    for report in reports:
        for item in report.items:
            if item.score > max_score:
                continue
            capability = resolve_capability(item, report.benchmark, catalog)
            rows.append(
                (
                    item.score,
                    {
                        "benchmark": report.benchmark,
                        "item_id": item.item_id,
                        "capability": capability,
                        "subcapability": item.subcapability,
                        "score": item.score,
                        "passed": item.passed,
                        "prompt": report.metadata.get("prompts", {}).get(item.item_id, ""),
                        "response": item.response[:1500],
                        "expected": item.expected,
                        "error": item.error,
                    },
                )
            )
    rows.sort(key=lambda pair: pair[0])
    return [row for _, row in rows[:limit]]
