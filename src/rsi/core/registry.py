from __future__ import annotations

import importlib
import inspect
from typing import Any

from .errors import RegistryError

_REGISTRY: dict[str, dict[str, type]] = {}


def register(category: str, name: str):
    """Class decorator: ``@register("analyzer", "rule")``."""

    def decorator(cls: type) -> type:
        add(category, name, cls)
        return cls

    return decorator


def add(category: str, name: str, cls: type) -> None:
    bucket = _REGISTRY.setdefault(category, {})
    existing = bucket.get(name)
    if existing is not None and existing is not cls:
        raise RegistryError(f"Plugin {category}.{name} already registered by {existing}")
    bucket[name] = cls


def get(category: str, name: str) -> type:
    try:
        return _REGISTRY[category][name]
    except KeyError as exc:
        available = ", ".join(sorted(_REGISTRY.get(category, {}))) or "<none>"
        raise RegistryError(f"Unknown plugin {category}.{name}. Available: {available}") from exc


def list_plugins() -> dict[str, list[str]]:
    return {category: sorted(entries) for category, entries in sorted(_REGISTRY.items())}


def _import_class(path: str) -> type:
    if ":" in path:
        module_name, attr = path.split(":", 1)
    else:
        module_name, _, attr = path.rpartition(".")
    if not module_name or not attr:
        raise RegistryError(f"Invalid plugin import path: {path!r}")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise RegistryError(f"Cannot import plugin module {module_name!r}: {exc}") from exc
    try:
        obj = getattr(module, attr)
    except AttributeError as exc:
        raise RegistryError(f"Module {module_name!r} has no attribute {attr!r}") from exc
    if not inspect.isclass(obj):
        raise RegistryError(f"Plugin path {path!r} does not resolve to a class")
    return obj


def resolve(category: str, spec: str | dict[str, Any] | type) -> tuple[type, dict[str, Any]]:
    if isinstance(spec, type):
        return spec, {}
    if isinstance(spec, str):
        spec = {"type": spec}
    if not isinstance(spec, dict):
        raise RegistryError(f"Invalid plugin spec: {spec!r}")

    spec = dict(spec)
    plugin_type = spec.pop("type", None)
    name = spec.pop("name", None)
    if not plugin_type:
        raise RegistryError(f"Plugin spec has no 'type': {spec!r}")
    if isinstance(plugin_type, type):
        cls = plugin_type
    else:
        try:
            cls = get(category, str(plugin_type))
        except RegistryError:
            cls = _import_class(str(plugin_type))
    if name is not None:
        spec.setdefault("instance_name", name)
    return cls, spec


def create(
    category: str,
    spec: str | dict[str, Any] | type,
    runtime: Any,
    cfg: Any,
    **extra: Any,
) -> Any:
    cls, params = resolve(category, spec)
    params.update(extra)
    try:
        return cls(runtime, cfg, **params)
    except TypeError as exc:
        raise RegistryError(
            f"Failed to construct {category}.{cls.__name__}: {exc}. "
            "Component constructors must accept (runtime, cfg, **params)."
        ) from exc
