"""Capability catalog: the shared vocabulary between benchmarks, analysis and data."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from rsi.core.errors import ConfigError


class CapabilityCatalog:
    """Maps capability IDs to planning metadata (kind, reward, seeds, priority).

    Unknown capabilities get a conservative fallback derived from the name so a
    new benchmark capability does not break the loop. Add entries to make the
    data planner smarter.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._defaults: dict[str, Any] = {
            "kind": "text",
            "priority": 0.5,
            "reward": {"type": "rule", "name": "contains"},
            "verifier": {"type": "rule", "name": "contains"},
            "seeds": [],
            "description": "",
        }
        self._caps: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        if not isinstance(raw, dict):
            raise ConfigError(f"Capability catalog must be a mapping: {self.path}")
        defaults = raw.get("defaults") or {}
        if isinstance(defaults, dict):
            self._defaults.update(defaults)
        caps = raw.get("capabilities") or {}
        if not isinstance(caps, dict):
            raise ConfigError("capabilities must be a mapping")
        for name, spec in caps.items():
            if not isinstance(spec, dict):
                raise ConfigError(f"Capability {name!r} must be a mapping")
            merged = copy.deepcopy(self._defaults)
            merged.update(spec)
            merged["name"] = str(name)
            self._caps[str(name)] = merged

    def get(self, capability: str) -> dict[str, Any]:
        if capability in self._caps:
            return copy.deepcopy(self._caps[capability])
        fallback = copy.deepcopy(self._defaults)
        fallback["name"] = capability
        fallback["description"] = capability.replace(".", " ")
        # Coarse inference from the capability namespace.
        if capability.startswith("code") or "coding" in capability:
            fallback["kind"] = "text"
        if capability.startswith("agentic") or "tool" in capability:
            fallback["kind"] = "agentic"
        if capability.startswith("knowledge"):
            fallback["kind"] = "knowledge"
        return fallback

    def all(self) -> list[dict[str, Any]]:
        return [self.get(name) for name in self._caps]

    def kind(self, capability: str) -> str:
        return str(self.get(capability).get("kind") or "text")

    def priority(self, capability: str) -> float:
        try:
            return float(self.get(capability).get("priority", 0.5))
        except (TypeError, ValueError):
            return 0.5

    def seeds(self, capability: str) -> list[str]:
        value = self.get(capability).get("seeds") or []
        return [str(v) for v in value] if isinstance(value, list) else []

    def verifier(self, capability: str) -> dict[str, Any]:
        value = self.get(capability).get("verifier") or {}
        return value if isinstance(value, dict) else {}

    def reward(self, capability: str) -> dict[str, Any]:
        value = self.get(capability).get("reward") or {}
        return value if isinstance(value, dict) else {}

    def benchmark_filters(self, capability: str) -> list[dict[str, Any]]:
        value = self.get(capability).get("benchmark") or []
        return [v for v in value if isinstance(v, dict)] if isinstance(value, list) else []
