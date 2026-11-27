"""Adapter for the ``agentic_rl_data`` tasks (verl-toolRL) and their rewards.

Rows exported by ``agentic_rl_data/scripts/export_verl_parquet.py`` follow one
contract::

    data_source   "agentic_rl"
    prompt        [{"role": "system"|"user", "content": ...}]
    tools         [{"name", "description", "parameters"}]  (parameters is a JSON string)
    reward_model  {"style": "rule", "ground_truth": str}  (label for logging only)
    extra_info    index, task_id, source, task_type, agentic_task, max_turns, ...

``extra_info["agentic_task"]`` is the canonical task JSON (``id / source /
task_type / messages / tools / env / reference / meta``). Rewards are pure
functions ``reward(task, trajectory) -> {"score", "info"``} dispatched by
``task_type`` (CONTRACTS C4):

    st_fc, irrelevance, abstain, search_qa, math_ci   scored from one response
    mt_fc, code_exec                                   need ``trajectory.env_state``,
                                                       i.e. a multi-turn rollout

This adapter maps rows to the RSI catalog shape and exposes the built-in
``dispatch_reward`` (``rsi.rewards.agentic``, ported from the package) as the
``agentic_rl`` scorer.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from rsi.core.errors import AdapterError
from rsi.core.jsonio import stable_hash
from rsi.core.scoring import register_scorer

SCORER = "agentic_rl"

DATA_SOURCE = "agentic_rl"

#: Fine-grained capabilities, one per dataset focus. They mirror
#: ``configs/capabilities.yaml`` (``agentic.*``).
SINGLE_TURN_TOOL_CALL = "agentic.single_turn_tool_call"
MULTI_TURN_STATEFUL = "agentic.multi_turn_stateful_tool_use"
INTERLEAVED_USER_TURNS = "agentic.interleaved_user_turns"
CLARIFY = "agentic.clarify_when_under_specified"
TOOL_IRRELEVANCE = "agentic.tool_irrelevance"
PARALLEL_TOOL_CALLS = "agentic.parallel_tool_calls"
MULTI_DOMAIN_STATE_OPS = "agentic.multi_domain_state_ops"
SEARCH_QA = "agentic.search_qa"
MATH_CODE_REASONING = "agentic.math_code_reasoning"
CODE_FROM_TESTS = "agentic.code_from_tests"

#: task_type -> capability for sources whose focus is the task type itself.
#: hammer and when2call emit two task types and are split by it per row.
CAPABILITY_BY_TASK_TYPE = {
    "st_fc": SINGLE_TURN_TOOL_CALL,
    "irrelevance": TOOL_IRRELEVANCE,
    "abstain": CLARIFY,
    "search_qa": SEARCH_QA,
    "math_ci": MATH_CODE_REASONING,
    "mt_fc": MULTI_TURN_STATEFUL,
    "code_exec": CODE_FROM_TESTS,
}
#: source -> capability for the datasets that train a skill of their own.
FOCUSED_SOURCES = {
    "interleaved_mt": INTERLEAVED_USER_TURNS,  # 3-6 user turns that revise earlier goals
    "stateful_abstain": CLARIFY,  # miss_func, miss_param, inferable in a stateful dialogue
    "parallel_mt": PARALLEL_TOOL_CALLS,  # independent calls in one turn
    "vehicle_travel_mt": MULTI_DOMAIN_STATE_OPS,  # vehicle, travel, trading
    "social_ops_mt": MULTI_DOMAIN_STATE_OPS,  # microblog, messaging, ticketing
}


def capability_for(source: str, task_type: str) -> str:
    """The capability a task of ``source`` with ``task_type`` trains."""
    if source in FOCUSED_SOURCES:
        return FOCUSED_SOURCES[source]
    try:
        return CAPABILITY_BY_TASK_TYPE[task_type]
    except KeyError:
        known = sorted(CAPABILITY_BY_TASK_TYPE)
        raise AdapterError(f"unknown agentic task_type {task_type!r}; expected one of {known}") from None

#: task_type -> (what the reward reads, whether one response is enough).
TASK_TYPES: dict[str, tuple[str, str]] = {
    "st_fc": ("calls vs reference.gt_calls", "single-turn"),
    "irrelevance": ("whether any tool was called", "single-turn"),
    "abstain": ("final reply text + whether a tool was called", "single-turn"),
    "search_qa": ("final_answer vs reference.final_answers (normalised EM/F1)", "single-turn"),
    "math_ci": ("final_answer vs reference.final_answer (equivalence)", "single-turn"),
    "mt_fc": ("env_state.goal_satisfied / replay_ok, per-turn match, format", "needs env_state"),
    "code_exec": ("env_state.code_tests.all_passed", "needs env_state"),
}
ENV_STATE_TASK_TYPES = frozenset(name for name, (_, mode) in TASK_TYPES.items() if mode != "single-turn")

#: source -> upstream dataset and task type.
#: source -> upstream raw data, task_type, env.type. Upstream ids are as written
#: in the package loaders (several are marked UNVERIFIED there).
SOURCES: dict[str, tuple[str, str, str]] = {
    "apigen": ("HF Salesforce/xlam-function-calling-60k", "st_fc", "none"),
    "sealtools": ("GitHub fairyshine/Seal-Tools (tool.jsonl, train.jsonl)", "st_fc", "none"),
    "toolace": ("HF Team-ACE/ToolACE", "st_fc", "none"),
    "apibank": ("GitHub AlibabaResearch/DAMO-ConvAI api-bank", "st_fc", "none"),
    "hammer": ("HF MadeAgents/xlam-irrelevance-7.5k", "irrelevance|st_fc", "none"),
    "procedural_st": ("generated: generators/procedural_st_tasks.py", "st_fc", "none"),
    "when2call": ("HF nvidia/When2Call (train split)", "st_fc|abstain", "none"),
    "search_r1": ("HF RUC-NLPIR/FlashRAG_datasets", "search_qa", "search"),
    "asearcher": ("HF inclusionAI/ASearcher-train-data", "search_qa", "search"),
    "a2search": ("GitHub zfj1998/A2Search parquet", "search_qa", "search"),
    "retool": ("HF JoeYing/ReTool-SFT", "math_ci", "python"),
    "reasoning_math": ("any verl-style math parquet (DAPO-Math-17k, DeepScaleR, ...)", "math_ci", "python"),
    "procedural_mt": ("generated: generators/stateful_tasks.py", "mt_fc", "mock_fc"),
    "interleaved_mt": ("generated: generators/interleaved_tasks.py", "mt_fc", "mock_fc"),
    "stateful_abstain": ("generated: generators/abstain_tasks.py", "mt_fc", "mock_fc"),
    "vehicle_travel_mt": ("generated: generators/vehicle_travel_tasks.py", "mt_fc", "mock_fc"),
    "social_ops_mt": ("generated: generators/social_ops_tasks.py", "mt_fc", "mock_fc"),
    "parallel_mt": ("generated: generators/parallel_mt_tasks.py", "mt_fc", "mock_fc"),
    "funreason_mt": ("HF Bingguang/FunReason-MT", "mt_fc", "mock_fc (replay)"),
    "apigen_mt": ("HF Salesforce/APIGen-MT-5k", "mt_fc", "mock_fc (replay)"),
    "toolmind": ("HF Nanbeige/ToolMind", "mt_fc", "mock_fc (replay)"),
    "toucan": ("HF Agent-Ark/Toucan-1.5M", "mt_fc", "mock_fc (replay)"),
    "code_exec": ("HF codeparrot/apps, BAAI/TACO, deepmind/code_contests", "code_exec", "code_tests"),
}
SEARCH_SOURCES = frozenset(name for name, (_, _, env) in SOURCES.items() if env.startswith("search"))

_dispatch_reward: Callable[[dict, dict], dict] | None = None
_THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.S)
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def load_dispatch_reward() -> Callable[[dict, dict], dict]:
    """The built-in ``dispatch_reward(task, trajectory)`` (``rsi.rewards.agentic``)."""
    global _dispatch_reward
    if _dispatch_reward is None:
        from rsi.rewards.agentic import dispatch_reward

        _dispatch_reward = dispatch_reward
    return _dispatch_reward


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Parse Qwen/Hermes ``<tool_call>{"name", "arguments"}</tool_call>`` blocks."""
    calls: list[dict[str, Any]] = []
    for match in _TOOL_CALL_RE.finditer(_THINK_RE.sub("", text or "")):
        try:
            call = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if not isinstance(call, dict) or not isinstance(call.get("name"), str) or not call["name"].strip():
            continue
        arguments = call.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        calls.append({"name": call["name"].strip(), "arguments": arguments if isinstance(arguments, dict) else {}})
    return calls


def single_turn_trajectory(text: str) -> dict[str, Any]:
    """The trajectory a single response forms (CONTRACTS C3, no env_state)."""
    return {
        "turns": [{"step": 1, "assistant": {"content": text or "", "tool_calls": parse_tool_calls(text)}, "tool_results": []}],
        "final_answer": text or None,
        "num_turns": 1,
        "done": True,
        "env_state": None,
    }


def tools_for_model(tools: Any) -> list[dict[str, Any]]:
    """Row ``tools`` with ``parameters`` parsed from its JSON string."""
    parsed = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        parameters = tool.get("parameters")
        if isinstance(parameters, str):
            try:
                parameters = json.loads(parameters)
            except json.JSONDecodeError:
                parameters = {}
        parsed.append({
            "name": str(tool.get("name") or ""),
            "description": str(tool.get("description") or ""),
            "parameters": parameters if isinstance(parameters, dict) else {},
        })
    return parsed


def tools_for_verl(tools: Any) -> list[dict[str, str]]:
    """``tools`` in the on-disk verl shape (``parameters`` as a JSON string)."""
    return [
        {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": json.dumps(tool.get("parameters") or {}, ensure_ascii=False, sort_keys=True),
        }
        for tool in tools_for_model(tools)
    ]


def parse_task(raw: Any) -> dict[str, Any] | None:
    """Decode ``extra_info.agentic_task`` (JSON string or mapping); None when invalid."""
    if isinstance(raw, str) and raw.strip():
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None
    return dict(raw) if isinstance(raw, dict) else None


def is_agentic_row(row: dict[str, Any]) -> bool:
    """True for a verl row exported from ``agentic_rl_data``.

    Only a non-empty ``agentic_task`` counts: mixed parquet files give rows of
    other sources a null ``agentic_task`` key.
    """
    extra_info = row.get("extra_info")
    return (isinstance(extra_info, dict) and bool(extra_info.get("agentic_task"))) or row.get("data_source") == DATA_SOURCE


def agentic_row_to_row(row: dict[str, Any]) -> dict[str, Any]:
    """Map one verl row to the generic row shape used by ``rows_to_samples``."""
    extra_info = row.get("extra_info") if isinstance(row.get("extra_info"), dict) else {}
    task = parse_task(extra_info.get("agentic_task"))
    if task is None:
        raise AdapterError("not a verl agentic_rl row: extra_info.agentic_task is missing")
    task_type = str(task.get("task_type") or extra_info.get("task_type") or "")
    source = str(extra_info.get("source") or task.get("source") or "")
    capability = capability_for(source, task_type)
    prompt = row.get("prompt")
    if isinstance(prompt, str):
        prompt = [{"role": "user", "content": prompt}]
    messages = [
        {"role": str(item.get("role", "user")), "content": str(item.get("content", ""))}
        for item in prompt or task.get("messages") or []
        if isinstance(item, dict)
    ]
    text = next((m["content"] for m in messages if m["role"] == "user"), "").strip()
    reward_model = row.get("reward_model") if isinstance(row.get("reward_model"), dict) else {}
    meta = task.get("meta") if isinstance(task.get("meta"), dict) else {}
    tools = tools_for_model(row.get("tools")) or tools_for_model(task.get("tools"))
    env = task.get("env") if isinstance(task.get("env"), dict) else {}
    return {
        "id": str(extra_info.get("task_id") or task.get("id") or stable_hash(task)),
        "prompt": text,
        "messages": messages,
        # The capability names what this dataset trains; the source is the sub-capability.
        "capability": capability,
        "kind": "agentic",
        "subcapability": source,
        # Upstream keeps the label for logging only; the reward reads the task.
        # Never None: a missing reference short-circuits ``score_text`` to 0.
        "reference": str(reward_model.get("ground_truth") or "") or "no reference answer; scored by verifier.task",
        "verifier": {"type": "rule", "name": SCORER, "task_type": task_type, "task": task},
        "metadata": {
            "agentic": {
                "source": source,
                "task_type": task_type,
                "env_type": str(env.get("type") or "none"),
                "tools": tools,
                "max_turns": meta.get("max_turns"),
            },
            "verl": {"data_source": DATA_SOURCE, "ability": capability, "extra_info": extra_info},
        },
    }


def agentic_rl_score(prediction: str, reference: str | None, metadata: dict[str, Any] | None) -> float:
    """Score one response (or a finished multi-turn trajectory) with the built-in reward.

    ``metadata`` is the sample verifier: ``task`` (or ``extra_info.agentic_task``)
    and, for ``mt_fc`` / ``code_exec``, the ``trajectory`` with ``env_state``.
    """
    metadata = metadata or {}
    extra_info = metadata.get("extra_info") if isinstance(metadata.get("extra_info"), dict) else {}
    task = parse_task(metadata.get("task") or extra_info.get("agentic_task"))
    if task is None:
        raise AdapterError("agentic_rl scorer needs metadata.task (extra_info.agentic_task)")
    trajectory = metadata.get("trajectory")
    if isinstance(trajectory, str):
        try:
            trajectory = json.loads(trajectory)
        except json.JSONDecodeError:
            trajectory = None
    if not isinstance(trajectory, dict):
        if task.get("task_type") in ENV_STATE_TASK_TYPES:
            raise AdapterError(f"task_type {task.get('task_type')!r} needs metadata.trajectory with env_state")
        trajectory = single_turn_trajectory(prediction)
    return float(load_dispatch_reward()(task, trajectory)["score"])


register_scorer(SCORER, agentic_rl_score)
