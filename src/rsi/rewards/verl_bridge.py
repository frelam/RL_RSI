"""verl custom-reward bridge for RSI-materialized rows.

Mount it in a verl trainer command without touching verl source (``rsi`` must
be importable, e.g. ``pip install -e .``)::

    reward.custom_reward_function.path=src/rsi/rewards/verl_bridge.py
    reward.custom_reward_function.name=compute_score

Every row written by the ``verl_jsonl`` / ``verl_parquet`` builder carries its
verifier in ``extra_info.verifier``. Built-in rule scorers
(``exact``/``contains``/``regex``/``numeric``) and the registered adapters
(``reasoning_rl``, ``agentic_rl``) are dispatched from that verifier, so one
reward function serves every capability a round mixes.
"""

from __future__ import annotations

from typing import Any

# Importing the verl adapters registers the ``reasoning_rl`` / ``agentic_rl``
# scorers. The verl process imports this file directly, so it cannot rely on
# ``rsi.bootstrap`` having run.
from rsi.adapters import verl_agentic_rl, verl_reasoning_rl  # noqa: F401
from rsi.core.scoring import score_text


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any = None,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, float]:
    """Score one verl rollout with the verifier recorded on its dataset row."""
    info = dict(extra_info or {})
    verifier = info.get("verifier") if isinstance(info.get("verifier"), dict) else {}
    name = str(verifier.get("name") or verifier.get("type") or "exact")
    metadata: dict[str, Any] = {
        key: value for key, value in verifier.items() if key != "extra_info"
    }
    metadata.update({key: value for key, value in info.items() if key != "verifier"})
    metadata["data_source"] = verifier.get("data_source") or data_source
    verifier_extra = verifier.get("extra_info")
    if isinstance(verifier_extra, dict):
        metadata["extra_info"] = verifier_extra
    if ground_truth is None or not str(ground_truth).strip():
        return {"score": 0.0}
    try:
        score = score_text(str(solution_str), str(ground_truth), name, metadata)
    except (KeyError, TypeError, ValueError):
        score = score_text(str(solution_str), str(ground_truth), "exact", metadata)
    return {"score": float(score)}
