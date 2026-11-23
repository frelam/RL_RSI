from __future__ import annotations

import math
import re
from typing import Any

from rsi.core.interfaces import DifficultyGrader
from rsi.core.scoring import score_text
from rsi.core.registry import register
from rsi.core.types import ChatMessage, ChatRequest, DataSample, LLMRequest

_REASONING_MARKERS = [
    "because",
    "therefore",
    "however",
    "step by step",
    "first",
    "second",
    "finally",
    "证明",
    "因此",
    "首先",
    "然后",
    "最后",
]
_NUMBERED_LINE = re.compile(r"^\s*(?:\d+[.)]|[-*])\s+", re.MULTILINE)


def _rule_difficulty(sample: DataSample, cfg: Any) -> float:
    text = sample.prompt + "\n" + sample.response
    max_chars = max(100, cfg.get_int("max_chars", 6000))
    length = min(1.0, math.log1p(len(text)) / math.log1p(max_chars))
    lowered = text.lower()
    marker_count = sum(1 for marker in _REASONING_MARKERS if marker in lowered)
    reasoning = min(1.0, marker_count / 5.0)
    steps = min(1.0, len(_NUMBERED_LINE.findall(text)) / 8.0)
    tool_calls = float(sample.metadata.get("tool_calls") or 0)
    tools = min(1.0, tool_calls / 10.0)
    code = 1.0 if "```" in text or sample.metadata.get("has_code") else 0.0
    kind = 1.0 if sample.kind == "agentic" else 0.0
    weights = cfg.get_dict(
        "weights",
        {"length": 0.35, "reasoning": 0.25, "steps": 0.2, "tools": 0.1, "code": 0.05, "kind": 0.05},
    )
    total_weight = sum(float(v) for v in weights.values()) or 1.0
    score = (
        float(weights.get("length", 0)) * length
        + float(weights.get("reasoning", 0)) * reasoning
        + float(weights.get("steps", 0)) * steps
        + float(weights.get("tools", 0)) * tools
        + float(weights.get("code", 0)) * code
        + float(weights.get("kind", 0)) * kind
    ) / total_weight
    return max(0.0, min(1.0, score))


@register("difficulty", "rule")
class RuleDifficultyGrader(DifficultyGrader):
    def __init__(self, runtime: Any, cfg: Any, **params: Any) -> None:
        super().__init__(runtime, cfg, **params)
        self.offset = self.cfg.get_float("offset", 0.0)
        self.scale = self.cfg.get_float("scale", 1.0)

    def grade(self, sample: DataSample, context: Any) -> float:
        value = _rule_difficulty(sample, self.cfg) * self.scale + self.offset
        return max(0.0, min(1.0, value))


@register("difficulty", "judge")
class JudgeDifficultyGrader(DifficultyGrader):
    def __init__(self, runtime: Any, cfg: Any, **params: Any) -> None:
        super().__init__(runtime, cfg, **params)
        self.prompt_id = str(self.cfg.require("prompt").get("prompt"))
        self.profile = str(self.cfg.get("profile", runtime.config.get("llm.default_profile", "judge")))
        self._cache: dict[str, float] = {}

    def grade(self, sample: DataSample, context: Any) -> float:
        key = sample.id
        if key in self._cache:
            return self._cache[key]
        try:
            rendered = self.runtime.prompt_hub.render(
                self.prompt_id,
                {
                    "prompt": sample.prompt[:3000],
                    "response": sample.response[:3000],
                    "capability": sample.capability,
                    "kind": sample.kind,
                    "round": context.round,
                },
            )
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
                raise ValueError("difficulty judge returned non-JSON")
            value = max(0.0, min(1.0, float(payload.get("difficulty", 0.5))))
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("difficulty judge failed, using rule fallback: %s", exc)
            value = _rule_difficulty(sample, self.cfg)
        self._cache[key] = value
        return value


@register("difficulty", "pass_rate")
class PassRateDifficultyGrader(DifficultyGrader):
    """Model-relative difficulty: 1 - current capability score."""

    def __init__(self, runtime: Any, cfg: Any, **params: Any) -> None:
        super().__init__(runtime, cfg, **params)
        self.default = self.cfg.get_float("default", 0.5)
        self.blend_rule = self.cfg.get_float("blend_rule", 0.0)

    def grade(self, sample: DataSample, context: Any) -> float:
        value = self.default
        if context.collect_result is not None:
            score = context.collect_result.capabilities.get(sample.capability)
            if score is not None:
                value = 1.0 - score.score
                sample.metadata["difficulty_basis"] = {
                    "capability": sample.capability,
                    "score": score.score,
                }
                if self.blend_rule > 0:
                    value = (1 - self.blend_rule) * value + self.blend_rule * _rule_difficulty(sample, self.cfg)
        return max(0.0, min(1.0, value))


@register("difficulty", "rollout")
class RolloutDifficultyGrader(PassRateDifficultyGrader):
    """Probe verifiable prompts with batches of increasing cumulative size."""

    def __init__(self, runtime: Any, cfg: Any, **params: Any) -> None:
        super().__init__(runtime, cfg, **params)
        self.batches = [int(n) for n in cfg.get_list("samples", [4, 8, 16])]
        self.max_tokens = cfg.get_int("max_tokens", 512)
        self.probe_missing = self.cfg.get_bool("probe_missing", False)
        if not self.batches or sorted(set(self.batches)) != self.batches or self.batches[0] < 1:
            raise ValueError("difficulty.samples must contain increasing positive rollout sizes")
        if self.max_tokens < 1:
            raise ValueError("difficulty.max_tokens must be positive")

    def grade(self, sample: DataSample, context: Any) -> float:
        verifier = sample.verifier or {}
        if not sample.reference or verifier.get("type", "rule") != "rule":
            return super().grade(sample, context)
        rollout = sample.metadata.get("adaptive_rollout") or {}
        if rollout.get("scored"):
            value = 1.0 - float(rollout["correct"]) / float(rollout["scored"])
            if self.blend_rule > 0:
                value = (1 - self.blend_rule) * value + self.blend_rule * _rule_difficulty(sample, self.cfg)
            return max(0.0, min(1.0, value))
        if not self.probe_missing or getattr(context, "checkpoint", None) is None:
            return super().grade(sample, context)
        inference = self.runtime.inference()
        inference.ensure(context.checkpoint)
        successes = count = 0
        low, high = sample.metadata.get("target_difficulty", [0.0, 1.0])
        for total in self.batches:
            requested = total - count
            response = inference.chat(ChatRequest(
                messages=[ChatMessage(role="user", content=sample.prompt)],
                n=requested,
                max_tokens=self.max_tokens,
                temperature=0.7,
            ))
            texts = response.texts[:requested]
            successes += sum(
                score_text(text, sample.reference, verifier.get("name", "contains"), verifier) >= 1.0
                for text in texts
            )
            count += len(texts)
            if count >= total and 0 < successes < count and low <= 1 - successes / count <= high:
                break
        sample.metadata["difficulty_rollout"] = {"samples": count, "successes": successes}
        return 1 - successes / count if count else super().grade(sample, context)


@register("difficulty", "composite")
class CompositeDifficultyGrader(DifficultyGrader):
    def __init__(self, runtime: Any, cfg: Any, **params: Any) -> None:
        super().__init__(runtime, cfg, **params)
        specs = cfg.specs("graders")
        if not specs:
            raise ValueError("composite difficulty requires a 'graders' list")
        self.graders: list[tuple[DifficultyGrader, float]] = []
        for spec in specs:
            name = str(spec.get("name") or spec["type"])
            child = cfg.child(spec, name)
            self.graders.append((runtime.build("difficulty", spec, child), float(spec.get("weight", 1.0))))
        total = sum(weight for _, weight in self.graders) or 1.0
        self.graders = [(grader, weight / total) for grader, weight in self.graders]

    def grade(self, sample: DataSample, context: Any) -> float:
        return max(
            0.0,
            min(1.0, sum(grader.grade(sample, context) * weight for grader, weight in self.graders)),
        )
