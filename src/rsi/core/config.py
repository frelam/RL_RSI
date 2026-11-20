from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any, Iterable

import yaml

from .errors import ConfigError

_MISSING = object()
_ENV_RE = re.compile(r"\$\{ENV:([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\}")
_KNOWN_VAR_RE = re.compile(r"\$\{(config_dir|repo_root)\}")


class Config:
    """Hierarchical configuration with dotted access.

    A config file is a plain YAML mapping. Every component receives its own
    subtree, so keys are relative inside a component:

        analyze:
          methods:
            - type: rule
              threshold: 0.4

    CLI overrides use the full dotted path: ``analyze.methods.0.threshold=0.5``
    is not supported (lists are replaced wholesale); use ``analyze.rule.threshold``
    style component keys for tunables.
    """

    def __init__(
        self,
        data: dict[str, Any],
        *,
        source: str | None = None,
        config_dir: str | Path | None = None,
        repo_root: str | Path | None = None,
        prefix: tuple[str, ...] = (),
    ) -> None:
        if not isinstance(data, dict):
            raise ConfigError(f"Configuration node must be a mapping, got {type(data).__name__}")
        self._data = data
        self.source = source
        self.config_dir = Path(config_dir) if config_dir else Path.cwd()
        self.repo_root = Path(repo_root) if repo_root else self.config_dir
        self._prefix = tuple(prefix)

    # ------------------------------------------------------------------ reading
    def get(self, path: str | None = None, default: Any = _MISSING) -> Any:
        if path in (None, "", "."):
            node: Any = self._data
        else:
            node = self._data
            for part in self._split(path):
                if not isinstance(node, dict) or part not in node:
                    if default is _MISSING:
                        raise ConfigError(f"Missing config key: {self.full_path(path)}")
                    return default
                node = node[part]
        return self._resolve(node)

    def require(self, *paths: str) -> "Config":
        for path in paths:
            self.get(path)
        return self

    def section(self, path: str, required: bool = True) -> "Config":
        value = self.get(path, _MISSING)
        if value is _MISSING:
            if required:
                raise ConfigError(f"Missing config section: {self.full_path(path)}")
            value = {}
        if not isinstance(value, dict):
            raise ConfigError(f"Config section {self.full_path(path)} must be a mapping")
        return Config(
            value,
            source=self.source,
            config_dir=self.config_dir,
            repo_root=self.repo_root,
            prefix=self._prefix + self._split(path),
        )

    def child(self, data: dict[str, Any], name: str = "") -> "Config":
        """Create a config node for an inline plugin spec."""
        if not isinstance(data, dict):
            raise ConfigError(f"Child config {name!r} must be a mapping")
        return Config(
            data,
            source=self.source,
            config_dir=self.config_dir,
            repo_root=self.repo_root,
            prefix=self._prefix + ((name,) if name else ()),
        )

    def get_bool(self, path: str, default: bool | None = None) -> bool:
        value = self.get(path, default if default is not None else _MISSING)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        raise ConfigError(f"{self.full_path(path)} must be a boolean, got {value!r}")

    def get_int(self, path: str, default: int | None = None) -> int:
        value = self.get(path, default if default is not None else _MISSING)
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{self.full_path(path)} must be an integer, got {value!r}") from exc

    def get_float(self, path: str, default: float | None = None) -> float:
        value = self.get(path, default if default is not None else _MISSING)
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{self.full_path(path)} must be a number, got {value!r}") from exc

    def get_list(self, path: str, default: Any = _MISSING) -> list[Any]:
        value = self.get(path, default)
        if value is None:
            return []
        if not isinstance(value, list):
            raise ConfigError(f"{self.full_path(path)} must be a list, got {type(value).__name__}")
        return value

    def get_dict(self, path: str, default: Any = _MISSING) -> dict[str, Any]:
        value = self.get(path, default)
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ConfigError(f"{self.full_path(path)} must be a mapping, got {type(value).__name__}")
        return value

    def specs(self, path: str) -> list[dict[str, Any]]:
        """Return a normalized list of plugin specs.

        Accepted forms:
          - string: ``- mock``
          - list of mappings: ``- {type: mock, threshold: 0.4}``
          - single mapping with ``type``
          - named mapping: ``{demo: {type: mock}, other: {type: jsonl}}``
        A nested ``params`` mapping is flattened into the spec.
        """
        value = self.get(path, None)
        if value is None:
            return []
        items: list[Any]
        if isinstance(value, dict):
            if "type" in value:
                items = [value]
            else:
                items = []
                for name, item in value.items():
                    if isinstance(item, dict):
                        spec = dict(item)
                        spec.setdefault("name", name)
                    elif isinstance(item, str):
                        spec = {"type": item, "name": name}
                    else:
                        raise ConfigError(f"{self.full_path(path)}.{name} is not a plugin spec")
                    items.append(spec)
        elif isinstance(value, list):
            items = value
        elif isinstance(value, str):
            items = [value]
        else:
            raise ConfigError(f"{self.full_path(path)} must be a plugin list or mapping")

        normalized: list[dict[str, Any]] = []
        for item in items:
            if isinstance(item, str):
                spec: dict[str, Any] = {"type": item}
            elif isinstance(item, dict):
                spec = copy.deepcopy(item)
            else:
                raise ConfigError(f"{self.full_path(path)} contains a non-plugin item: {item!r}")
            params = spec.pop("params", None)
            if params is not None:
                if not isinstance(params, dict):
                    raise ConfigError(f"{self.full_path(path)} plugin params must be a mapping")
                spec.update(params)
            if not spec.get("type"):
                raise ConfigError(f"{self.full_path(path)} plugin is missing 'type': {item!r}")
            normalized.append(spec)
        return normalized

    def plugin(self, path: str, default_type: str | None = None) -> dict[str, Any]:
        specs = self.specs(path) if self.get(path, None) is not None else []
        if specs:
            return specs[0]
        if default_type is None:
            raise ConfigError(f"Missing plugin config at {self.full_path(path)}")
        return {"type": default_type}

    # ------------------------------------------------------------------ helpers
    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._resolve(self._data))

    def full_path(self, path: str | None = None) -> str:
        parts = list(self._prefix)
        if path:
            parts.extend(self._split(path))
        return ".".join(parts) or "<root>"

    def keys(self) -> list[str]:
        return list(self._data.keys())

    def __contains__(self, path: str) -> bool:
        return self.get(path, _MISSING) is not _MISSING

    def __repr__(self) -> str:
        return f"Config(prefix={'.'.join(self._prefix) or '<root>'}, keys={list(self._data)[:8]})"

    @staticmethod
    def _split(path: str) -> tuple[str, ...]:
        return tuple(part for part in path.split(".") if part)

    def _resolve(self, value: Any) -> Any:
        if isinstance(value, str):
            return self._resolve_string(value)
        if isinstance(value, dict):
            return {k: self._resolve(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._resolve(v) for v in value]
        return value

    def _resolve_string(self, value: str) -> str:
        def env(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            if name in os.environ:
                return os.environ[name]
            if default is not None:
                return default
            raise ConfigError(f"Environment variable {name} is required by config {self.source or ''}")

        def known(match: re.Match[str]) -> str:
            return str(self.config_dir if match.group(1) == "config_dir" else self.repo_root)

        return _KNOWN_VAR_RE.sub(known, _ENV_RE.sub(env, value))


# ---------------------------------------------------------------------- loading
def load_config(path: str | Path, overrides: Iterable[str] = ()) -> Config:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    sources: list[Path] = []
    data = _load_yaml_with_includes(path, seen=set(), sources=sources)
    for expr in overrides:
        apply_override(data, expr)
    repo_root = _resolve_repo_root(path, sources, data)
    return Config(data, source=str(path), config_dir=path.parent, repo_root=repo_root)


def _resolve_repo_root(path: Path, sources: list[Path], data: dict[str, Any]) -> Path:
    explicit = (data.get("paths") or {}).get("repo_root") if isinstance(data.get("paths"), dict) else None
    if explicit:
        root = Path(str(explicit)).expanduser()
        return root.resolve() if root.is_absolute() else (path.parent / root).resolve()
    for source in [path, *sources]:
        root = _infer_repo_root(source.parent)
        if root is not None:
            return root
    return path.parent.parent


def _load_yaml_with_includes(path: Path, seen: set[Path], sources: list[Path]) -> dict[str, Any]:
    path = path.resolve()
    if path in seen:
        raise ConfigError(f"Circular config include: {path}")
    if not path.exists():
        raise ConfigError(f"Included config not found: {path}")
    seen.add(path)
    sources.append(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"Config root must be a mapping: {path}")
    includes = raw.pop("include", [])
    if isinstance(includes, str):
        includes = [includes]
    merged: dict[str, Any] = {}
    for include in includes:
        include_path = Path(include)
        if not include_path.is_absolute():
            include_path = path.parent / include_path
        merged = _deep_merge(merged, _load_yaml_with_includes(include_path, seen, sources))
    merged = _deep_merge(merged, raw)
    seen.discard(path)
    return merged


def _deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        out = dict(base)
        for key, value in override.items():
            out[key] = _deep_merge(out[key], value) if key in out else value
        return out
    return override


def _infer_repo_root(start: Path) -> Path | None:
    current = start.resolve()
    for _ in range(6):
        if (current / ".git").exists() or (current / "pyproject.toml").exists():
            return current
        if current.parent == current:
            break
        current = current.parent
    return None


def apply_override(data: dict[str, Any], expression: str) -> None:
    """Apply ``dot.path=value`` to a dict. Values are parsed as YAML scalars."""
    if "=" not in expression:
        raise ConfigError(f"Override must look like key.path=value, got {expression!r}")
    key, raw = expression.split("=", 1)
    key = key.strip()
    if not key:
        raise ConfigError(f"Override has an empty key: {expression!r}")
    value = yaml.safe_load(raw) if raw.strip() else ""
    node = data
    parts = key.split(".")
    for part in parts[:-1]:
        if part not in node:
            node[part] = {}
        if not isinstance(node[part], dict):
            raise ConfigError(f"Cannot override {key!r}: {part!r} is not a mapping")
        node = node[part]
    node[parts[-1]] = value


def dump_config(config: Config) -> str:
    return yaml.safe_dump(config.as_dict(), sort_keys=False, allow_unicode=True)
