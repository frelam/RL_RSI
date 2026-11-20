"""Prompt Hub: versioned prompt assets with declared variables and output schema."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from rsi.core.errors import ConfigError

_VAR_RE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


@dataclass
class PromptSpec:
    prompt_id: str
    version: int
    template: str
    system: str = ""
    variables: list[str] = field(default_factory=list)
    optional_variables: list[str] = field(default_factory=list)
    schema: dict[str, Any] | None = None
    description: str = ""
    path: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RenderedPrompt:
    prompt_id: str
    version: int
    text: str
    system: str
    schema: dict[str, Any] | None
    spec: PromptSpec

    @property
    def key(self) -> str:
        return f"{self.prompt_id}@{self.version}"


class PromptHub:
    """Loads ``prompts/*.yaml``. IDs are referenced as ``name`` or ``name@3``."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self._specs: dict[str, PromptSpec] = {}
        self._latest: dict[str, int] = {}
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.directory.exists():
            return
        for path in sorted(self.directory.glob("*.y*ml")):
            with path.open("r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh) or {}
            if not isinstance(raw, dict):
                raise ConfigError(f"Prompt file must be a mapping: {path}")
            prompt_id = str(raw.get("id") or path.stem)
            version = int(raw.get("version", 1))
            template = raw.get("template")
            if not isinstance(template, str) or not template.strip():
                raise ConfigError(f"Prompt {path} has no template")
            declared = list(raw.get("variables") or [])
            declared += [v for v in _VAR_RE.findall(template) if v not in declared]
            optional = list(raw.get("optional_variables") or [])
            spec = PromptSpec(
                prompt_id=prompt_id,
                version=version,
                template=template,
                system=str(raw.get("system") or ""),
                variables=declared,
                optional_variables=optional,
                schema=raw.get("schema"),
                description=str(raw.get("description") or ""),
                path=str(path),
                metadata=dict(raw.get("metadata") or {}),
            )
            key = f"{prompt_id}@{version}"
            if key in self._specs:
                raise ConfigError(f"Duplicate prompt {key} at {path}")
            self._specs[key] = spec
            self._latest[prompt_id] = max(self._latest.get(prompt_id, 0), version)

    def get(self, prompt_id: str) -> PromptSpec:
        self._load()
        if "@" in prompt_id:
            key = prompt_id
        else:
            if prompt_id not in self._latest:
                available = ", ".join(sorted(self._latest)) or "<none>"
                raise ConfigError(f"Unknown prompt {prompt_id!r}. Available: {available}")
            key = f"{prompt_id}@{self._latest[prompt_id]}"
        if key not in self._specs:
            raise ConfigError(f"Unknown prompt version {key!r}")
        return self._specs[key]

    def render(self, prompt_id: str, variables: dict[str, Any]) -> RenderedPrompt:
        spec = self.get(prompt_id)
        required = [v for v in spec.variables if v not in spec.optional_variables]
        missing = [v for v in required if v not in variables]
        if missing:
            raise ConfigError(f"Prompt {spec.prompt_id}@{spec.version} missing variables: {missing}")

        def substitute(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in variables:
                return match.group(0)
            value = variables[name]
            if isinstance(value, (dict, list, tuple)):
                return json.dumps(value, ensure_ascii=False, indent=2, default=str)
            return str(value)

        return RenderedPrompt(
            prompt_id=spec.prompt_id,
            version=spec.version,
            text=_VAR_RE.sub(substitute, spec.template),
            system=spec.system,
            schema=spec.schema,
            spec=spec,
        )

    def list(self) -> list[dict[str, Any]]:
        self._load()
        return [
            {"id": s.prompt_id, "version": s.version, "description": s.description, "path": s.path}
            for s in sorted(self._specs.values(), key=lambda s: (s.prompt_id, s.version))
        ]
