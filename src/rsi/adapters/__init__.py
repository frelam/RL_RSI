"""Adapters are the only place that talks to the outside world.

Importing this package registers every built-in adapter.
"""

from rsi.adapters import benchmark, harness, inference, llm, trainer  # noqa: F401
