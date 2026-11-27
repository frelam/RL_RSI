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
# Derived from verl-toolRL agentic_rl_data/rewards/irrelevance.py.
"""``irrelevance`` reward - hammer-style "no tool needed" negatives (plan §5).

``score = 1.0`` when the policy emitted **no tool call at all**, else ``0.0``.
This is the literal plan rule ("未调用任何工具 = 1，否则 0"): a *decision*
reward about whether to invoke a tool, not a reply-quality reward, so a direct
answer and a refusal score identically here.

Boundary vs ``abstain`` (plan §1.5, §5): ``irrelevance`` (hammer) asks "should
a tool be called?" and accepts any no-call reply; ``abstain`` (When2Call)
additionally asks "does the reply correctly ask for missing information /
declare no suitable tool instead of fabricating an answer?" - see
``abstain.py``.  An abstain sample scored by ``irrelevance`` would get 1.0 for
a fabricated direct answer; the converse is not true (a correct question
scores 1.0 under both).

Reads only ``trajectory.turns[*].assistant.tool_calls`` (CONTRACTS C4).
"""

from __future__ import annotations

from . import base

TASK_TYPE = "irrelevance"
MAX_SCORE = 1.0


def reward(task: dict, trajectory: dict) -> dict:
    """Pure reward for one ``irrelevance`` task/trajectory pair."""
    calls = base.all_calls(trajectory)
    score = 0.0 if calls else MAX_SCORE
    info = {
        "n_output_calls": len(calls),
        "called_calls": [base.call_repr(call) for call in calls[:10]],
        "empty_trajectory": base.is_empty_trajectory(trajectory),
        "rule": "no tool call = 1, any tool call = 0 (plan §5)",
        "boundary": (
            "decision-only reward: direct answer and refusal score the same; "
            "see abstain for reply-quality scoring of 'information is missing -> ask'"
        ),
    }
    return base.make_result(score, info, task_type=TASK_TYPE)
