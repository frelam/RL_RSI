from __future__ import annotations

from typing import Any


class _KeepMissing(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def format_template(value: Any, variables: dict[str, Any]) -> Any:
    """Format ``{placeholders}`` while leaving unknown placeholders untouched.

    Works recursively for lists (command argv) and mappings. This keeps shell
    snippets such as ``${HOME}`` or JSON braces from raising KeyError.
    """
    if isinstance(value, str):
        try:
            return value.format_map(_KeepMissing({k: v for k, v in variables.items()}))
        except (ValueError, IndexError):
            return value
    if isinstance(value, list):
        return [format_template(item, variables) for item in value]
    if isinstance(value, dict):
        return {key: format_template(item, variables) for key, item in value.items()}
    return value
