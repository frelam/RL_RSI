"""Explicit registration order for all built-in plugins and loops."""

from rsi import adapters  # noqa: F401
from rsi import modules  # noqa: F401
from rsi import loops  # noqa: F401


def bootstrap() -> None:
    """Idempotent: imports above performed registration."""
