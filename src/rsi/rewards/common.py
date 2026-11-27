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
# Derived from verl examples/reasoning_rl/reward/compute_score.py and verl-toolRL
# agentic_rl_data/rewards/base.py.
"""Primitives shared by the built-in reward families.

Everything here is reward-agnostic: answer extraction, answer-text
normalisation, structured-answer comparison, the repetition penalty and score
construction. Domain verifiers (:mod:`rsi.rewards.reasoning`,
:mod:`rsi.rewards.agentic`) import from this module instead of keeping a copy.

Invariants that must survive any edit:

* only the visible (post-``</think>``) response is graded, never the
  reasoning; a string without a think block is all visible answer;
* a bool is never equal to a number, and a huge literal that ``float()``
  refuses falls through to text comparison instead of killing the reward
  worker;
* an answer nested deeper than ``MAX_STRUCTURE_DEPTH`` is refused at parse
  time so every downstream comparison stays stack-safe;
* the repetition penalty is applied to every returned score, including
  format-gate and verifier-crash zeros.
"""

from __future__ import annotations

import ast
import json
import math
import re
from collections import Counter
from collections.abc import Mapping
from typing import Any

ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
BOXED_RE = re.compile(r"\\boxed\s*\{")
# SynLogic task prompts use heterogeneous final-answer conventions: besides
# "Final Answer: ...", many tasks (web_of_lies, cryptarithm, word_sorting_mistake)
# instruct 'The answer is $YOUR_ANSWER' — often wrapped in markdown bold.
FINAL_ANSWER_RE = re.compile(r"(?:final answer\s*[::]|the answer is)\s*[::]?\s*(.+)$", re.IGNORECASE | re.MULTILINE)
FENCED_BLOCK_RE = re.compile(r"```[a-zA-Z0-9_+-]*\s*\n(.*?)```", re.DOTALL)
# Enigmata string/grid tasks (campsite, star_battle, ...) ask for the board
# wrapped in <begin_board>...</begin_board> inside the final answer.
BOARD_RE = re.compile(r"<begin_board>(.*?)<end_board>", re.DOTALL | re.IGNORECASE)


# ---------------------------------------------------------------------------
# answer extraction


def extract_boxed(text: str) -> str | None:
    """Return the content of the last \\boxed{...} with balanced braces."""
    starts = [m.end() for m in BOXED_RE.finditer(text)]
    if not starts:
        return None
    i = starts[-1]
    depth = 1
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i:j]
    return None


def extract_all_boxed(text: str) -> list[str]:
    """Contents of every \\boxed{...} with balanced braces, in order."""
    contents = []
    for m in BOXED_RE.finditer(text):
        i = m.end()
        depth = 1
        for j in range(i, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    contents.append(text[i:j])
                    break
    return contents


def strip_fences(s: str) -> str:
    """Unwrap a markdown fenced code block (```lang\\n...```) around an answer.

    Several SynLogic tasks (skyscraper_puzzle, zebra_puzzle) require the final
    answer inside a ```python / ```json block; models also wrap <answer>-tag
    content in fences on their own. The fence is container, not content.
    """
    s = s.strip()
    m = FENCED_BLOCK_RE.fullmatch(s)
    return m.group(1).strip() if m else s


def final_response(solution_str: str) -> str:
    """Return the user-facing response body (everything after </think>).

    The stage-1 format gate guarantees a single closed think block, so there
    the partition always succeeds.  A string WITHOUT a think block -- e.g. the
    bare ``\\boxed{...}`` the hallucination dispatcher's ``math_match``
    synthesizes from an already-extracted answer -- is all visible answer, so
    the whole string is returned; returning an empty body silently disabled
    every last-resort matcher that parses the response (``_gt_segment_match``,
    ``_or_branch_match``, ``_conditional_match``, ``_evaluated_gt_match``) on
    that path.  Constraints apply to the visible answer only, not to internal
    reasoning.
    """
    head, sep, body = solution_str.partition("</think>")
    return (body if sep else head).strip()


# ---------------------------------------------------------------------------
# answer-text normalisation


# Font/text wrappers carry no math content; the command name must go BEFORE
# braces are stripped, otherwise ``\text{negative number}`` normalises to
# ``\textnegativenumber`` and never matches a plain-text ground truth.
TEXT_WRAP_RE = re.compile(
    r"\\(?:text|textbf|textit|textsf|texttt|mathbf|mathrm|mathit|mathsf|mathtt|operatorname)\s*\{([^{}]*)\}"
)
WS_RE = re.compile(r"\s+")


def normalise_text(s: str) -> str:
    s = s.strip()
    # Strip markdown emphasis/backticks — prompts such as web_of_lies ask for
    # 'The answer is **yes, no**' and the stars are presentation, not content.
    s = s.replace("**", "").replace("__", "").replace("`", "")
    s = s.strip().strip('"').strip("'")
    # LaTeX font/text wrappers are presentation too (the same normalisation the
    # math domain applies): a logic answer boxed as \text{cucumber, ...} must
    # compare equal to the bare ground truth.
    for _ in range(4):  # unwrap a few nesting levels, innermost first
        unwrapped = TEXT_WRAP_RE.sub(r"\1", s)
        if unwrapped == s:
            break
        s = unwrapped
    # Inline LaTeX spacing: \ , \, \; \: are spaces (\! a negative space) --
    # unlike the math domain, whitespace stays significant here, so the
    # spacing commands map to a plain space instead of vanishing.
    s = re.sub(r"\\[,;: ]", " ", s)
    s = s.replace("\\!", "")
    s = re.sub(r"\s+", " ", s)
    return s.casefold()


def loose_token_form(s: str) -> str:
    """Canonical form insensitive to separator spacing and inner quotes.

    Fixes two systematic false negatives against SynLogic ground truths:
      - gt "A,C,D,E" vs natural model output "A, C, D, E" (boolean_expressions,
        calcudoko row separators);
      - gt "[[WORD]]" vs prompt-example-compliant "[['WORD']]" (cipher): the
        quoted form parses, the bare-word ground truth never does, so text
        comparison is the only path and must ignore the quotes.
    Whitespace *between* tokens is preserved ("1 2 3" != "123").
    """
    s = normalise_text(s)
    s = re.sub(r"\s*([,\[\]\(\)\{\}:;])\s*", r"\1", s)
    return s.replace('"', "").replace("'", "")


# ---------------------------------------------------------------------------
# structured answers (grids, lists, dicts)


# An answer nested hundreds of levels deep (a policy emitting a long run of "[")
# parses fine but then blows the C stack inside the recursive comparison and
# repr() paths below. Refusing it at parse time keeps every downstream consumer
# iterative-safe; the flat text comparison still gets its say, so identical
# deep answers still match.
MAX_STRUCTURE_DEPTH = 64


def too_deep(obj, limit: int = MAX_STRUCTURE_DEPTH) -> bool:
    """True when ``obj`` nests deeper than ``limit`` (iterative, no recursion)."""
    stack = [(obj, 1)]
    while stack:
        value, depth = stack.pop()
        if depth > limit:
            return True
        if isinstance(value, list | tuple):
            stack.extend((item, depth + 1) for item in value)
        elif isinstance(value, dict):
            stack.extend((item, depth + 1) for item in value.values())
    return False


def parse_structured(s: str):
    """Best-effort parse of an answer into a Python object (grids, tuples,
    lists, numbers). Returns None when it only parses to a plain string —
    that case is covered by text comparison instead."""
    s = s.strip()
    for parser in (ast.literal_eval, json.loads):
        try:
            obj = parser(s)
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            continue
        if not isinstance(obj, str):
            return None if too_deep(obj) else obj
    return None


def structured_equal(a, b, tol: float = 1e-6) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, int | float) and isinstance(b, int | float):
        try:
            return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)
        except (OverflowError, ValueError):
            # ``float()`` refuses an integer beyond ~1e308 (OverflowError). Such a
            # literal is not "un-equal", it is simply unordered by ``isclose``:
            # fall through to the text comparison so two identical huge integers
            # still match instead of killing the reward worker.
            pass
    if isinstance(a, list | tuple) and isinstance(b, list | tuple):
        return len(a) == len(b) and all(structured_equal(x, y, tol) for x, y in zip(a, b, strict=True))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(structured_equal(a[k], b[k], tol) for k in a)
    return loose_token_form(str(a)) == loose_token_form(str(b))


# ---------------------------------------------------------------------------
# repetition penalty


# Degenerate loops (the same block pasted over and over) can still carry the
# right answer -- e.g. a repeated <answer> fence whose last copy is correct --
# and would otherwise collect full credit.  A single word n-gram occurring
# more than ``rep_threshold(len)`` times marks the trajectory as a death loop;
# every ``REP_STEP_PER`` occurrences past the threshold cost ``REP_STEP``,
# penalties from several looped n-grams add up, and ``repetition_adjusted``
# floors the final score at ``REP_FLOOR``.  Applied to every returned score,
# including format-gate and verifier-crash zeros: a degenerate sample deserves
# the penalty regardless of correctness.
#
# The threshold is length-scaled: a fixed count cannot separate a death loop
# from normal discourse once the response-length curriculum reaches 16k-24k
# tokens -- connective reasoning phrases ("if the first person is", "the
# answer is") naturally occur 20-40 times across ~10k words, so a flat 15
# mis-scored verbose-but-legitimate responses negative.  The threshold is
# therefore ``max(REP_MIN_COUNT, total // REP_RATIO)``: short texts keep
# the original absolute floor (calibrated on death loops, not prose), long
# texts only get charged when one n-gram occupies >1/REP_RATIO of all
# n-gram positions -- the signature of a loop, not of verbose reasoning.
REP_NGRAM = 5
REP_MIN_COUNT = 15
REP_RATIO = 50
REP_STEP = 0.1
REP_STEP_PER = 10
REP_FLOOR = -1.0


def rep_threshold(total_ngrams: int) -> int:
    return max(REP_MIN_COUNT, total_ngrams // REP_RATIO)


def repetition_penalty(text: str) -> float:
    """Death-loop penalty in points: excess occurrences of any word n-gram past
    ``rep_threshold`` are summed across all looped n-grams, and every full
    ``REP_STEP_PER`` of them costs ``REP_STEP``."""
    words = text.split()
    if len(words) < REP_NGRAM + REP_MIN_COUNT:  # too short to loop past the threshold
        return 0.0
    counts = Counter(zip(*(words[i:] for i in range(REP_NGRAM))))
    threshold = rep_threshold(len(words) - REP_NGRAM + 1)
    excess = sum(count - threshold for count in counts.values() if count > threshold)
    return REP_STEP * (excess // REP_STEP_PER)


def repetition_adjusted(solution_str: str, score: float) -> float:
    return max(score - repetition_penalty(solution_str), REP_FLOOR)


# ---------------------------------------------------------------------------
# score construction


def clamp_score(score: Any, score_min: float = 0.0, score_max: float = 1.0) -> float:
    """Clamp ``score`` into ``[score_min, score_max]``; a non-numeric or
    non-finite score becomes ``score_min`` (never raises). The default range
    is the agentic contract's frozen ``schema.SCORE_RANGE`` (CONTRACTS C4)."""
    try:
        value = float(score)
    except (TypeError, ValueError):
        return score_min
    if not math.isfinite(value):
        return score_min
    return max(score_min, min(score_max, value))


def weighted_sum(parts: list[tuple[float, float]]) -> float:
    """``sum(weight * value)`` via ``math.fsum`` for stable 0.8+0.1+0.1 arithmetic."""
    return math.fsum(float(weight) * float(value) for weight, value in parts)


def jsonable(value: Any) -> Any:
    """Recursively convert ``value`` into plain JSON-serialisable data."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [jsonable(item) for item in sorted(value, key=lambda item: repr(item))]
    return repr(value)
