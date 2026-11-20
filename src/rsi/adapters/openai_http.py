from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from rsi.core.errors import AdapterError


def normalize_base_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    parsed = urllib.parse.urlparse(base_url)
    if parsed.path in ("", "/"):
        base_url += "/v1"
    return base_url


def openai_chat(
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    api_key: str = "",
    temperature: float = 0.0,
    max_tokens: int = 1024,
    n: int = 1,
    stop: list[str] | None = None,
    schema: dict[str, Any] | None = None,
    timeout_s: float = 120.0,
    max_retries: int = 3,
    retry_backoff_s: float = 1.0,
    extra_headers: dict[str, str] | None = None,
    extra_body: dict[str, Any] | None = None,
    json_mode: bool = True,
) -> dict[str, Any]:
    """Minimal OpenAI-compatible chat completion client (stdlib only)."""
    url = normalize_base_url(base_url) + "/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "n": n,
    }
    if stop:
        payload["stop"] = stop
    if schema and json_mode:
        payload["response_format"] = {"type": "json_object"}
    if extra_body:
        payload.update(extra_body)
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if extra_headers:
        headers.update(extra_headers)
    body = json.dumps(payload).encode("utf-8")

    last_error: str = ""
    for attempt in range(max_retries + 1):
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            last_error = f"HTTP {exc.code} from {url}: {raw[:2000]}"
            if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                raise AdapterError(last_error) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = f"Cannot reach {url}: {exc}"
        if attempt < max_retries:
            time.sleep(retry_backoff_s * (2**attempt))
    raise AdapterError(last_error or f"Chat completion failed: {url}")


def parse_json_text(text: str) -> dict[str, Any] | None:
    """Parse JSON, tolerating a markdown code fence."""
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start >= 0 and end > start:
            try:
                value = json.loads(candidate[start : end + 1])
            except json.JSONDecodeError:
                return None
        else:
            return None
    return value if isinstance(value, dict) else None


def api_key_from_config(cfg: Any, default_env: str = "OPENAI_API_KEY") -> str:
    literal = cfg.get("api_key", "") if hasattr(cfg, "get") else ""
    if literal:
        return str(literal)
    env_name = str(cfg.get("api_key_env", default_env)) if hasattr(cfg, "get") else default_env
    return os.environ.get(env_name, "")
