from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from rsi.core.interfaces import SampleFilter
from rsi.core.jsonio import stable_hash
from rsi.core.registry import register
from rsi.core.scoring import normalize_text, score_text
from rsi.core.types import DataSample, FilterBatch, FilterDecision

_DEFAULT_PII = [
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    r"\b(?:\+?\d{1,3}[- ]?)?(?:1[3-9]\d{9}|\d{3}[-.]\d{3}[-.]\d{4})\b",
    r"\b(?:sk|pk)-[A-Za-z0-9]{16,}\b",
    r"\bAKIA[0-9A-Z]{16}\b",
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
]


def _empty_batch(samples: list[DataSample]) -> FilterBatch:
    return FilterBatch(
        kept=list(samples),
        decisions=[
            FilterDecision(sample_id=s.id, keep=True, filter="identity", reason="no-op")
            for s in samples
        ],
    )


@register("data_filter", "pii")
class PIIFilter(SampleFilter):
    def __init__(self, runtime: Any, cfg: Any, **params: Any) -> None:
        super().__init__(runtime, cfg, **params)
        patterns = self.cfg.get_list("patterns", _DEFAULT_PII)
        self.patterns = [re.compile(str(p), re.IGNORECASE) for p in patterns]
        self.action = str(self.cfg.get("action", "drop"))
        self.fields = [str(v) for v in self.cfg.get_list("fields", ["prompt", "response"])]
        if self.action not in {"drop", "redact"}:
            raise ValueError("pii.action must be 'drop' or 'redact'")

    def filter(self, samples: list[DataSample], context: Any) -> FilterBatch:
        batch = FilterBatch()
        for sample in samples:
            matched: list[str] = []
            for field in self.fields:
                value = getattr(sample, field, "") or ""
                for pattern in self.patterns:
                    if pattern.search(value):
                        matched.append(pattern.pattern)
                        if self.action == "redact":
                            setattr(sample, field, pattern.sub("[REDACTED]", value))
                            value = getattr(sample, field)
            if matched and self.action == "drop":
                batch.dropped.append(sample)
                batch.decisions.append(
                    FilterDecision(sample.id, False, "pii", f"matched: {matched[0][:80]}")
                )
            else:
                batch.kept.append(sample)
                reason = "redacted" if matched else ""
                batch.decisions.append(
                    FilterDecision(sample.id, True, "pii", reason, score=0.5 if matched else 1.0)
                )
        return batch


@register("data_filter", "length")
class LengthFilter(SampleFilter):
    def __init__(self, runtime: Any, cfg: Any, **params: Any) -> None:
        super().__init__(runtime, cfg, **params)
        self.min_prompt = self.cfg.get_int("min_prompt_chars", 8)
        self.max_prompt = self.cfg.get_int("max_prompt_chars", 20000)
        self.min_response = self.cfg.get_int("min_response_chars", 1)
        self.max_response = self.cfg.get_int("max_response_chars", 32000)

    def filter(self, samples: list[DataSample], context: Any) -> FilterBatch:
        batch = FilterBatch()
        for sample in samples:
            reasons: list[str] = []
            if len(sample.prompt) < self.min_prompt:
                reasons.append("prompt too short")
            if len(sample.prompt) > self.max_prompt:
                reasons.append("prompt too long")
            if sample.kind not in {"agentic"}:
                if len(sample.response) < self.min_response:
                    reasons.append("response too short")
                if len(sample.response) > self.max_response:
                    reasons.append("response too long")
            if reasons:
                batch.dropped.append(sample)
                batch.decisions.append(FilterDecision(sample.id, False, "length", "; ".join(reasons)))
            else:
                batch.kept.append(sample)
                batch.decisions.append(FilterDecision(sample.id, True, "length"))
        return batch


@register("data_filter", "format")
class FormatFilter(SampleFilter):
    def __init__(self, runtime: Any, cfg: Any, **params: Any) -> None:
        super().__init__(runtime, cfg, **params)
        self.require_response = self.cfg.get_bool("require_response", True)
        self.require_reference = self.cfg.get_bool("require_reference", False)
        self.reject_identical = self.cfg.get_bool("reject_identical", True)
        self.require_json = self.cfg.get_bool("require_json", False)
        self.allowed_kinds = [str(v) for v in self.cfg.get_list("allowed_kinds", [])]

    def filter(self, samples: list[DataSample], context: Any) -> FilterBatch:
        batch = FilterBatch()
        for sample in samples:
            reason = ""
            if self.allowed_kinds and sample.kind not in self.allowed_kinds:
                reason = f"kind {sample.kind} not allowed"
            elif not sample.prompt.strip():
                reason = "empty prompt"
            elif sample.kind not in {"agentic"} and self.require_response and not sample.response.strip():
                reason = "empty response"
            elif self.require_reference and not sample.reference:
                reason = "missing reference"
            elif self.reject_identical and sample.response.strip() == sample.prompt.strip():
                reason = "response equals prompt"
            elif self.require_json:
                try:
                    json.loads(sample.response)
                except json.JSONDecodeError:
                    reason = "response is not JSON"
            if reason:
                batch.dropped.append(sample)
                batch.decisions.append(FilterDecision(sample.id, False, "format", reason))
            else:
                batch.kept.append(sample)
                batch.decisions.append(FilterDecision(sample.id, True, "format"))
        return batch


@register("data_filter", "dedup")
class DedupFilter(SampleFilter):
    """Exact + n-gram near-duplicate filter; persists across rounds by default."""

    def __init__(self, runtime: Any, cfg: Any, **params: Any) -> None:
        super().__init__(runtime, cfg, **params)
        self.cross_round = self.cfg.get_bool("cross_round", True)
        self.ngram = self.cfg.get_int("ngram", 3)
        self.threshold = self.cfg.get_float("threshold", 0.9)
        self.max_index = self.cfg.get_int("max_index", 20000)
        self._seen: set[str] = set()
        self._grams: list[tuple[str, set[tuple[str, ...]]]] = []

    def filter(self, samples: list[DataSample], context: Any) -> FilterBatch:
        batch = FilterBatch()
        batch_seen: set[str] = set()
        for sample in samples:
            key = stable_hash("dedup", normalize_text(sample.prompt), length=24)
            if key in batch_seen or (self.cross_round and key in self._seen):
                batch.dropped.append(sample)
                batch.decisions.append(FilterDecision(sample.id, False, "dedup", "exact duplicate"))
                continue
            grams = _ngrams(normalize_text(sample.prompt), self.ngram)
            duplicate = False
            for other_id, other in self._grams:
                if _jaccard(grams, other) >= self.threshold:
                    duplicate = True
                    batch.decisions.append(
                        FilterDecision(sample.id, False, "dedup", f"near duplicate of {other_id}")
                    )
                    break
            if duplicate:
                batch.dropped.append(sample)
                continue
            batch_seen.add(key)
            if self.cross_round:
                self._seen.add(key)
            self._grams.append((sample.id, grams))
            if len(self._grams) > self.max_index:
                self._grams.pop(0)
            batch.kept.append(sample)
            batch.decisions.append(FilterDecision(sample.id, True, "dedup"))
        return batch


@register("data_filter", "judge_quality")
class JudgeQualityFilter(SampleFilter):
    def __init__(self, runtime: Any, cfg: Any, **params: Any) -> None:
        super().__init__(runtime, cfg, **params)
        self.prompt_id = str(self.cfg.require("prompt").get("prompt"))
        self.profile = str(self.cfg.get("profile", runtime.config.get("llm.default_profile", "judge")))
        self.min_score = self.cfg.get_float("min_score", 0.5)
        self.on_error = str(self.cfg.get("on_error", "keep"))
        self.max_chars = self.cfg.get_int("max_chars", 4000)
        self._cache: dict[str, float] = {}

    def filter(self, samples: list[DataSample], context: Any) -> FilterBatch:
        batch = FilterBatch()
        for sample in samples:
            key = stable_hash("quality", sample.prompt[: self.max_chars], sample.response[: self.max_chars])
            if key in self._cache:
                score = self._cache[key]
            else:
                try:
                    rendered = self.runtime.prompt_hub.render(
                        self.prompt_id,
                        {
                            "prompt": sample.prompt[: self.max_chars],
                            "response": sample.response[: self.max_chars],
                            "reference": sample.reference or "",
                            "capability": sample.capability,
                            "kind": sample.kind,
                            "round": context.round,
                        },
                    )
                    from rsi.core.types import LLMRequest

                    response = self.runtime.llm(self.profile).complete(
                        LLMRequest(
                            prompt_id=rendered.key,
                            prompt=rendered.text,
                            system=rendered.system,
                            schema=rendered.schema,
                            temperature=0.0,
                            max_tokens=512,
                            metadata={"round": context.round, "capability": sample.capability},
                        )
                    )
                    payload = next((item for item in response.parsed if item is not None), None)
                    if payload is None:
                        raise ValueError("judge returned non-JSON")
                    score = float(payload.get("score", 0.0))
                    if payload.get("keep") is False:
                        score = min(score, self.min_score - 0.01)
                except Exception as exc:  # noqa: BLE001
                    self.logger.warning("quality judge failed: %s", exc)
                    if self.on_error == "drop":
                        batch.dropped.append(sample)
                        batch.decisions.append(FilterDecision(sample.id, False, "judge_quality", "judge error"))
                    else:
                        batch.kept.append(sample)
                        batch.decisions.append(FilterDecision(sample.id, True, "judge_quality", "judge error"))
                    continue
                self._cache[key] = score
            sample.quality = max(0.0, min(1.0, score))
            if sample.quality >= self.min_score:
                batch.kept.append(sample)
                batch.decisions.append(FilterDecision(sample.id, True, "judge_quality", score=sample.quality))
            else:
                batch.dropped.append(sample)
                batch.decisions.append(
                    FilterDecision(sample.id, False, "judge_quality", f"score {sample.quality:.2f}", score=sample.quality)
                )
        return batch


@register("data_filter", "verifier")
class VerifierFilter(SampleFilter):
    def __init__(self, runtime: Any, cfg: Any, **params: Any) -> None:
        super().__init__(runtime, cfg, **params)
        self.min_score = self.cfg.get_float("min_score", 0.99)
        self.require_reference = self.cfg.get_bool("require_reference", False)
        self.defer_llm = self.cfg.get_bool("defer_llm", True)

    def filter(self, samples: list[DataSample], context: Any) -> FilterBatch:
        batch = FilterBatch()
        for sample in samples:
            verifier = sample.verifier or {}
            name = str(verifier.get("name") or verifier.get("type") or "contains")
            if not sample.reference:
                if self.require_reference:
                    batch.dropped.append(sample)
                    batch.decisions.append(FilterDecision(sample.id, False, "verifier", "missing reference"))
                else:
                    batch.kept.append(sample)
                    batch.decisions.append(FilterDecision(sample.id, True, "verifier", "unverifiable"))
                continue
            if name not in {"exact", "contains", "regex", "numeric"}:
                batch.kept.append(sample)
                batch.decisions.append(FilterDecision(sample.id, True, "verifier", f"deferred:{name}"))
                continue
            score = score_text(sample.response, sample.reference, name, verifier)
            sample.metadata["verified_score"] = score
            if score >= self.min_score:
                batch.kept.append(sample)
                batch.decisions.append(FilterDecision(sample.id, True, "verifier", score=score))
            else:
                batch.dropped.append(sample)
                batch.decisions.append(FilterDecision(sample.id, False, "verifier", f"score {score:.2f}", score=score))
        return batch


@register("data_filter", "pattern")
class PatternFilter(SampleFilter):
    def __init__(self, runtime: Any, cfg: Any, **params: Any) -> None:
        super().__init__(runtime, cfg, **params)
        self.patterns = [re.compile(str(p), re.IGNORECASE) for p in self.cfg.get_list("patterns", [])]
        self.target = str(self.cfg.get("target", "both"))

    def filter(self, samples: list[DataSample], context: Any) -> FilterBatch:
        batch = FilterBatch()
        for sample in samples:
            text = {
                "prompt": sample.prompt,
                "response": sample.response,
                "both": sample.prompt + "\n" + sample.response,
            }.get(self.target, sample.prompt + "\n" + sample.response)
            match = next((p.pattern for p in self.patterns if p.search(text)), None)
            if match:
                batch.dropped.append(sample)
                batch.decisions.append(FilterDecision(sample.id, False, "pattern", f"matched {match[:80]}"))
            else:
                batch.kept.append(sample)
                batch.decisions.append(FilterDecision(sample.id, True, "pattern"))
        return batch


class FilterPipeline:
    """Sequential filter chain driven by ``data.filter.pipeline``."""

    def __init__(self, runtime: Any, cfg: Any) -> None:
        self.runtime = runtime
        self.cfg = cfg
        self.filters: list[tuple[str, SampleFilter]] = []
        for spec in cfg.specs("pipeline"):
            name = str(spec.get("name") or spec["type"])
            child = cfg.child(spec, name)
            instance = runtime.build("data_filter", spec, child)
            self.filters.append((name, instance))

    def run(
        self, samples: list[DataSample], context: Any
    ) -> tuple[list[DataSample], list[FilterDecision], dict[str, Any]]:
        current = list(samples)
        decisions: list[FilterDecision] = []
        stats: dict[str, Any] = {}
        for name, instance in self.filters:
            before = len(current)
            batch = instance.filter(current, context)
            current = batch.kept
            decisions.extend(batch.decisions)
            reasons = Counter(d.reason for d in batch.decisions if not d.keep)
            stats[name] = {
                "input": before,
                "kept": len(current),
                "dropped": before - len(current),
                "reasons": dict(reasons.most_common(8)),
            }
        stats["total"] = {
            "input": len(samples),
            "kept": len(current),
            "dropped": len(samples) - len(current),
        }
        return current, decisions, stats


def _ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    tokens = re.findall(r"\w+", text)
    if len(tokens) <= n:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)}


def _jaccard(left: set[tuple[str, ...]], right: set[tuple[str, ...]]) -> float:
    if not left or not right:
        return 0.0
    union = len(left | right)
    return len(left & right) / union if union else 0.0
