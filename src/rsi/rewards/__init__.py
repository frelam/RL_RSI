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

"""Built-in rewards.

Two families live here, both ported from verl with their scoring logic intact:

- :mod:`rsi.rewards.reasoning` -- reasoning RL (math, code, logic, stem,
  instruction following, bbh). Entry point ``compute_score(data_source,
  solution_str, ground_truth, extra_info, ...) -> {"score": float}``.
- :mod:`rsi.rewards.agentic` -- agentic tool-use RL (single/multi-turn tool
  calls, abstention, irrelevance, search QA, math with code interpreter, code
  execution). Entry point ``dispatch_reward(task, trajectory) -> {"score":
  float, "info": dict}``; a task type with no registered reward is an error.

They share :mod:`rsi.rewards.common` (answer extraction, answer-text
normalisation, structured-answer comparison, repetition penalty, score
construction).

Optional dependencies are imported lazily or behind a guarded import, so
importing this package never requires them: ``math-verify`` and verl's
``math_dapo`` for math, verl's ``prime_code`` / ``sandbox_fusion`` for code,
``langdetect`` for instruction following, ``reasoning_gym`` for its domain. A
verifier that cannot be imported fails closed to score 0 (logged), it never
raises.
"""

from rsi.rewards.agentic import REWARDS as AGENTIC_REWARDS
from rsi.rewards.agentic import dispatch_reward
from rsi.rewards.reasoning.compute_score import compute_score

#: Every built-in reward, keyed by ``"<family>.<task_type>"``.
BUILTIN_REWARDS = {
    "reasoning_rl": compute_score,
    **{f"agentic_rl.{task_type}": reward for task_type, reward in AGENTIC_REWARDS.items()},
}


def list_rewards() -> list[str]:
    """Names of the built-in rewards."""
    return sorted(BUILTIN_REWARDS)


def get_reward(name: str):
    """Return the built-in reward called ``name``; ``KeyError`` lists the options."""
    try:
        return BUILTIN_REWARDS[name]
    except KeyError:
        raise KeyError(f"unknown built-in reward {name!r}; known: {', '.join(list_rewards())}") from None


__all__ = ["BUILTIN_REWARDS", "compute_score", "dispatch_reward", "get_reward", "list_rewards"]
