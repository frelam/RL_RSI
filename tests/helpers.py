from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

from rsi.bootstrap import bootstrap
from rsi.core.config import Config
from rsi.core.jsonio import stable_hash
from rsi.core.registry import create
from rsi.hubs.catalog import CapabilityCatalog
from rsi.hubs.prompts import PromptHub
from rsi.hubs.skills import SkillHub

REPO_ROOT = Path(__file__).resolve().parents[1]

bootstrap()

_LLM_FIXTURES = {
    "sample_quality": {"score": 0.9, "keep": True, "reason": "specific and verifiable"},
    "sample_difficulty": {"difficulty": 0.55, "reason": "multi-step"},
    "data_text_synthesis": {
        "samples": [
            {"prompt": "synthetic prompt one", "response": "answer one", "reference": "answer one"},
            {"prompt": "synthetic prompt two", "response": "answer two", "reference": "answer two"},
        ]
    },
    "knowledge_qa": {
        "samples": [{"prompt": "question", "response": "answer", "reference": "answer"}]
    },
    "data_plan": {
        "requests": [
            {"capability": "code.debugging", "kind": "text", "count": 2, "seeds": ["debugging"]}
        ]
    },
    "badcase_analysis": {
        "defects": [
            {
                "capability": "long_context.retrieval",
                "severity": 0.6,
                "confidence": 0.8,
                "description": "multi-hop retrieval fails",
                "evidence": ["case-1"],
            }
        ]
    },
    "trajectory_ability_analysis": {
        "ability_priorities": {"code.debugging": 0.8, "unknown.ability": 0.9}
    },
}

_HARNESS_FIXTURES = {
    "env_synthesis": {
        "tasks": [
            {
                "name": "task-1",
                "instruction": "fix the failing test",
                "files": {"main.py": "def add(a,b): return a-b\n"},
                "verifier": {"type": "harness", "name": "tests_pass"},
                "difficulty": 0.6,
            }
        ]
    }
}


class FakeRuntime:
    """Small runtime double for unit tests; the loop smoke test uses the real one."""

    def __init__(self, tmp: str | Path, config: Config | None = None) -> None:
        self.logger = logging.getLogger(f"test.{id(self)}")
        self.logger.addHandler(logging.NullHandler())
        self.config = config or Config(
            {"llm": {"default_profile": "judge"}, "run": {"seed": 7}},
            config_dir=REPO_ROOT,
            repo_root=REPO_ROOT,
        )
        self.repo_root = REPO_ROOT
        self.run_dir = Path(tmp)
        self.run_id = "test"
        self.seed = 7
        self.prompt_hub = PromptHub(REPO_ROOT / "prompts")
        self.skill_hub = SkillHub(REPO_ROOT / "skills")
        self.catalog = CapabilityCatalog(REPO_ROOT / "configs" / "capabilities.yaml")
        self._inference = None
        self._harness = None
        self._llm = None

    def path(self, value: str | Path) -> Path:
        path = Path(str(value)).expanduser()
        return path if path.is_absolute() else (self.repo_root / path)

    def rng(self, *parts: Any) -> random.Random:
        return random.Random(int(stable_hash(self.seed, *parts), 16))

    def build(self, category: str, spec: Any, cfg: Any = None) -> Any:
        return create(category, spec, self, cfg or self.config)

    def inference(self) -> Any:
        if self._inference is None:
            from rsi.adapters.inference import MockInference

            cfg = Config(
                {"answer_template": "{{prompt}}"},
                config_dir=REPO_ROOT,
                repo_root=REPO_ROOT,
            )
            self._inference = MockInference(self, cfg)
        return self._inference

    def llm(self, profile: str | None = None) -> Any:
        if self._llm is None:
            from rsi.adapters.llm import MockLLM

            cfg = Config(
                {"fixtures": _LLM_FIXTURES},
                config_dir=REPO_ROOT,
                repo_root=REPO_ROOT,
            )
            self._llm = MockLLM(self, cfg)
        return self._llm

    def harness(self, name: str) -> Any:
        if self._harness is None:
            from rsi.adapters.harness import MockHarness

            cfg = Config(
                {"fixtures": _HARNESS_FIXTURES},
                config_dir=REPO_ROOT,
                repo_root=REPO_ROOT,
            )
            self._harness = MockHarness(self, cfg)
        return self._harness
