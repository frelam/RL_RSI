from __future__ import annotations

import json
from typing import Any

from rsi.adapters.openai_http import api_key_from_config, openai_chat, parse_json_text
from rsi.core.errors import AdapterError
from rsi.core.interfaces import LLMBackend
from rsi.core.registry import register
from rsi.core.types import LLMRequest, LLMResponse


def _substitute(value: Any, variables: dict[str, Any]) -> Any:
    if isinstance(value, str):
        for key, replacement in variables.items():
            value = value.replace("{{" + key + "}}", str(replacement))
        return value
    if isinstance(value, list):
        return [_substitute(item, variables) for item in value]
    if isinstance(value, dict):
        return {key: _substitute(item, variables) for key, item in value.items()}
    return value


@register("llm", "mock")
class MockLLM(LLMBackend):
    """Deterministic LLM used for offline loops and tests.

    Fixtures live under ``fixtures.<prompt_id>``. A fixture may be an object
    (returned as JSON), a string, or a list (cycled per call). Templates may
    reference ``{{round}}``, ``{{call_index}}`` and any request metadata.
    """

    def __init__(self, runtime: Any, cfg: Any, **params: Any) -> None:
        super().__init__(runtime, cfg, **params)
        self.fixtures: dict[str, Any] = self.cfg.get_dict("fixtures", {})
        self._calls: dict[str, int] = {}

    def complete(self, request: LLMRequest) -> LLMResponse:
        candidates = [request.prompt_id]
        if "@" in request.prompt_id:
            candidates.append(request.prompt_id.split("@", 1)[0])
        candidates.append("*")
        fixture: Any = None
        for key in candidates:
            if key in self.fixtures:
                fixture = self.fixtures[key]
                break
        if fixture is None:
            raise AdapterError(
                f"MockLLM has no fixture for prompt {request.prompt_id!r}. "
                f"Available: {sorted(self.fixtures)}"
            )
        texts: list[str] = []
        parsed: list[dict[str, Any] | None] = []
        for _ in range(max(1, request.n)):
            index = self._calls.get(request.prompt_id, 0)
            self._calls[request.prompt_id] = index + 1
            item = fixture[index % len(fixture)] if isinstance(fixture, list) else fixture
            variables = dict(request.metadata)
            variables.update(
                {
                    "call_index": index,
                    "prompt_id": request.prompt_id,
                    "model": str(self.cfg.get("model", "mock-llm")),
                }
            )
            item = _substitute(item, variables)
            if isinstance(item, str):
                text = item
            elif isinstance(item, dict) and "__text__" in item:
                text = str(item["__text__"])
            else:
                text = json.dumps(item, ensure_ascii=False)
            texts.append(text)
            parsed.append(parse_json_text(text))
        prompt_chars = len(request.prompt) + sum(len(t) for t in texts)
        return LLMResponse(
            texts=texts,
            parsed=parsed,
            model=str(self.cfg.get("model", "mock-llm")),
            usage={"prompt_tokens": len(request.prompt) // 4, "completion_tokens": prompt_chars // 4},
            raw={"mock": True},
        )


@register("llm", "openai")
class OpenAICompatibleLLM(LLMBackend):
    """OpenAI-compatible endpoint; works with vLLM/SGLang judge servers too."""

    def __init__(self, runtime: Any, cfg: Any, **params: Any) -> None:
        super().__init__(runtime, cfg, **params)
        self.base_url = str(self.cfg.require("base_url").get("base_url"))
        self.model = str(self.cfg.get("model", "default"))
        self.api_key = api_key_from_config(self.cfg)
        self.timeout_s = self.cfg.get_float("timeout_s", 120.0)
        self.max_retries = self.cfg.get_int("max_retries", 3)
        self.retry_backoff_s = self.cfg.get_float("retry_backoff_s", 1.0)
        self.json_mode = self.cfg.get_bool("json_mode", True)
        self.extra_headers = self.cfg.get_dict("extra_headers", {})
        self.extra_body = self.cfg.get_dict("extra_body", {})

    def complete(self, request: LLMRequest) -> LLMResponse:
        messages: list[dict[str, Any]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": request.prompt})
        raw = openai_chat(
            base_url=self.base_url,
            model=self.model,
            messages=messages,
            api_key=self.api_key,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            n=request.n,
            stop=request.stop if hasattr(request, "stop") else None,
            schema=request.schema,
            timeout_s=self.timeout_s,
            max_retries=self.max_retries,
            retry_backoff_s=self.retry_backoff_s,
            extra_headers=self.extra_headers,
            extra_body=self.extra_body,
            json_mode=self.json_mode,
        )
        choices = raw.get("choices") or []
        texts = [str((choice.get("message") or {}).get("content") or "") for choice in choices]
        if not texts:
            raise AdapterError(f"LLM returned no choices: {str(raw)[:1000]}")
        parsed = [parse_json_text(text) if request.schema else None for text in texts]
        return LLMResponse(
            texts=texts,
            parsed=parsed,
            model=str(raw.get("model") or self.model),
            usage=dict(raw.get("usage") or {}),
            raw=raw,
        )


# Alias kept for readability in configs.
register("llm", "openai_compatible")(OpenAICompatibleLLM)
