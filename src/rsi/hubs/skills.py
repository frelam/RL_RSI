"""Skill Hub: directories of agent skills (SKILL.md + assets) used by harnesses."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from rsi.core.errors import ConfigError


@dataclass
class SkillSpec:
    name: str
    description: str
    version: str
    path: str
    files: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class SkillHub:
    """Scans ``skills/<name>/SKILL.md``. Skills are versioned by directory."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self._skills: dict[str, SkillSpec] | None = None

    def _load(self) -> dict[str, SkillSpec]:
        if self._skills is not None:
            return self._skills
        skills: dict[str, SkillSpec] = {}
        if self.directory.exists():
            for path in sorted(self.directory.iterdir()):
                skill_file = path / "SKILL.md"
                if not path.is_dir() or not skill_file.exists():
                    continue
                meta = _parse_frontmatter(skill_file.read_text(encoding="utf-8"))
                name = str(meta.get("name") or path.name)
                if name != path.name:
                    raise ConfigError(
                        f"Skill directory {path.name!r} does not match SKILL.md name {name!r}"
                    )
                skills[name] = SkillSpec(
                    name=name,
                    description=str(meta.get("description") or ""),
                    version=str(meta.get("version") or "1"),
                    path=str(path),
                    files=sorted(
                        str(p.relative_to(path)) for p in path.rglob("*") if p.is_file()
                    ),
                    metadata=meta,
                )
        self._skills = skills
        return skills

    def get(self, name: str) -> SkillSpec:
        skills = self._load()
        if name not in skills:
            available = ", ".join(sorted(skills)) or "<none>"
            raise ConfigError(f"Unknown skill {name!r}. Available: {available}")
        return skills[name]

    def list(self) -> list[SkillSpec]:
        return [self._load()[name] for name in sorted(self._load())]

    def materialize(self, names: list[str], destination: str | Path) -> Path:
        """Copy selected skills into a harness working directory."""
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        for name in names:
            spec = self.get(name)
            target = destination / name
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(spec.path, target)
        return destination


def _parse_frontmatter(text: str) -> dict[str, Any]:
    if not text.startswith("---"):
        return {}
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}
    try:
        data = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid SKILL.md frontmatter: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError("SKILL.md frontmatter must be a mapping")
    return data
