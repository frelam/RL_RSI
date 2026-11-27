"""Adapter for verl ``examples/reasoning_rl`` datasets and their rule reward.

verl rows (``{domain}/train_*.parquet``) follow one contract::

    data_source   "{domain}_{dataset}", e.g. ``math_dapo``; selects the reward
    prompt        [{"role": "user", "content": ...}]
    ability       math / code / logic / stem / if
    reward_model  {"style": "rule", "ground_truth": ...}
    extra_info    split, index, task_id, domain, source, difficulty,
                  prior_solve_rate, seed (reasoning_gym needs the seed)

The reward is the built-in ``rsi.rewards.reasoning.compute_score`` (ported from
verl ``examples/reasoning_rl/reward``), which dispatches on the ``data_source``
prefix. This adapter maps rows to the RSI catalog shape and exposes that reward
as the ``reasoning_rl`` scorer.
"""

from __future__ import annotations

from typing import Any, Callable

from rsi.core.errors import AdapterError
from rsi.core.jsonio import stable_hash
from rsi.core.scoring import register_scorer

SCORER = "reasoning_rl"

#: verl ability -> RSI capability.
ABILITY_TO_CAPABILITY = {
    "math": "math.reasoning",
    "code": "code.problem_solving",
    "logic": "logic.reasoning",
    "stem": "stem.reasoning",
    "if": "instruction.following",
}

#: data_source -> upstream dataset and ground-truth shape read by the reward.
SOURCES: dict[str, tuple[str, str, str]] = {
    "math_bigmath": ("SynthLabsAI/Big-Math-RL-Verified", "answer string", "math_verify, then math_dapo"),
    "math_dapo": ("BytedTsinghua-SIA/DAPO-Math-17k", "answer string", "math_verify, then math_dapo"),
    "code_deepcoder": (
        "agentica-org/DeepCoder-Preview-Dataset",
        '{"inputs","outputs"[,"fn_name"]}',
        "all test cases must pass (sandbox, local prime_code as fallback)",
    ),
    "code_codecontests": (
        "deepmind/code_contests",
        '{"inputs","outputs"}',
        "all test cases must pass (sandbox, local prime_code as fallback)",
    ),
    "code_apps": (
        "codeparrot/apps",
        '{"inputs","outputs"}',
        "all test cases must pass (sandbox, local prime_code as fallback)",
    ),
    "logic_synlogic": ("MiniMaxAI/SynLogic", '{"answer","task"}', "normalised answer match"),
    "logic_enigmata": (
        "BytedTsinghua-SIA/Enigmata-Data",
        '{"answer","task","meta"}',
        "task-aware verifier, then normalised answer match",
    ),
    "logic_reasoning_gym": (
        "reasoning_gym (procedural, needs extra_info.seed)",
        '{"answer","task"}',
        "answer match, then the library verifier seeded by extra_info.seed",
    ),
    "stem_drsci": ("MiniByte-666/Dr_SCI (verifiable split)", "answer string", "same as math"),
    "if_nemotron": (
        "nvidia/Nemotron-RL-instruction_following",
        '{"constraints":[{"id","kwargs"}]}',
        "1.0 only when every constraint passes",
    ),
}

_compute_score: Callable[..., Any] | None = None


def load_compute_score() -> Callable[..., Any]:
    """The built-in ``compute_score(data_source, solution_str, ground_truth, extra_info, ...)``."""
    global _compute_score
    if _compute_score is None:
        from rsi.rewards.reasoning.compute_score import compute_score

        _compute_score = compute_score
    return _compute_score


def is_verl_row(row: dict[str, Any]) -> bool:
    """True for a verl parquet/JSONL row (``data_source`` plus ``reward_model``)."""
    return bool(row.get("data_source")) and isinstance(row.get("reward_model"), dict)


def verl_row_to_row(row: dict[str, Any]) -> dict[str, Any]:
    """Map one verl row to the generic row shape used by ``rows_to_samples``."""
    data_source = str(row.get("data_source") or "")
    reward_model = row.get("reward_model")
    if not data_source or not isinstance(reward_model, dict):
        raise AdapterError(f"not a verl reasoning_rl row: data_source={data_source!r}")
    ability = str(row.get("ability") or "")
    if ability not in ABILITY_TO_CAPABILITY:
        raise AdapterError(f"unknown verl ability {ability!r} in {data_source}")
    prompt = row.get("prompt")
    if isinstance(prompt, str):
        prompt = [{"role": "user", "content": prompt}]
    messages = [
        {"role": str(item.get("role", "user")), "content": str(item.get("content", ""))}
        for item in prompt or []
        if isinstance(item, dict)
    ]
    text = str(row.get("question") or next((m["content"] for m in messages if m["role"] == "user"), "")).strip()
    reference = reward_model.get("ground_truth")
    if reference is not None and not isinstance(reference, str):
        reference = str(reference)
    extra_info = row.get("extra_info") if isinstance(row.get("extra_info"), dict) else {}
    sample_id = str(extra_info.get("task_id") or extra_info.get("id") or stable_hash(data_source, text))
    return {
        "id": sample_id,
        "prompt": text,
        "messages": messages,
        "capability": ABILITY_TO_CAPABILITY[ability],
        "subcapability": data_source,
        "reference": reference,
        "verifier": {"type": "rule", "name": SCORER, "data_source": data_source, "extra_info": extra_info},
        "metadata": {"verl": {"data_source": data_source, "ability": ability, "extra_info": extra_info}},
    }


def reasoning_rl_score(prediction: str, reference: str | None, metadata: dict[str, Any] | None) -> float:
    """Score one response with the built-in ``compute_score``.

    ``metadata`` carries ``data_source`` and the verl ``extra_info`` (reasoning_gym
    regenerates the puzzle from its seed).
    """
    metadata = metadata or {}
    data_source = str(metadata.get("data_source") or "")
    if not data_source:
        raise AdapterError("reasoning_rl scorer needs metadata.data_source")
    result = load_compute_score()(data_source, prediction, reference, metadata.get("extra_info"))
    return float(result["score"])


register_scorer(SCORER, reasoning_rl_score)
