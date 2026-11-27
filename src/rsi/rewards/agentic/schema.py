# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Derived from verl-toolRL agentic_rl_data/schema.py (the subset the rewards read).
"""Frozen reward contract shared by the agentic rewards (CONTRACTS C4).

Kept as its own module so the reward range and the reward payload validation
have exactly one definition, as in the upstream package.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

REWARD_KEYS = frozenset({"score", "info"})


#: Frozen reward range for every task type (plan §5): 1/0 or 0.8+0.1+0.1 style,
#: never negative (the old tool_rl reward allowed negatives; this plan does not).
SCORE_RANGE: tuple[float, float] = (0.0, 1.0)

#: The frozen task types. Every reward registry must cover them exactly; an
#: unknown task type is a data/schema bug and must raise, never score 0.
TASK_TYPES = frozenset(
    {"mt_fc", "st_fc", "irrelevance", "search_qa", "math_ci", "code_exec", "abstain"}
)


class SchemaError(ValueError):
    """Raised when a task / trajectory / reward payload breaks the contract."""


def _is_mapping(value: Any) -> bool:
    return isinstance(value, Mapping)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_reward_result(
    result: Any,
    *,
    min_score: float = SCORE_RANGE[0],
    max_score: float = SCORE_RANGE[1],
) -> list[str]:
    """Return contract violations for the ``{"score": float, "info": dict}`` payload."""
    errors: list[str] = []
    if not _is_mapping(result):
        return [f"reward result must be a mapping, got {type(result).__name__}"]
    for key in REWARD_KEYS:
        if key not in result:
            errors.append(f"missing required key {key!r}")
    score = result.get("score")
    if not _is_number(score):
        errors.append("score must be a finite number")
    elif isinstance(score, float) and not math.isfinite(score):
        errors.append("score must be finite")
    elif not (min_score <= float(score) <= max_score):
        errors.append(f"score {score!r} outside frozen range [{min_score}, {max_score}]")
    info = result.get("info")
    if not _is_mapping(info):
        errors.append("info must be a mapping (never null)")
    return errors


def gt_call_groups(task: Mapping) -> list[list[dict[str, Any]]]:
    """Normalised per-turn ground-truth call groups (``[]`` = no tools needed)."""
    reference = task.get("reference") if _is_mapping(task.get("reference")) else {}
    groups: list[list[dict[str, Any]]] = []
    for group in reference.get("gt_calls") or []:
        if not isinstance(group, list):
            continue
        calls = []
        for call in group:
            if not _is_mapping(call):
                continue
            calls.append(
                {
                    "name": call.get("name"),
                    "arguments": dict(call.get("arguments") or {}),
                }
            )
        groups.append(calls)
    return groups


def flatten_gt_calls(task: Mapping) -> list[dict[str, Any]]:
    """All ground-truth calls in turn order (for single-turn checks / replay)."""
    return [call for group in gt_call_groups(task) for call in group]
