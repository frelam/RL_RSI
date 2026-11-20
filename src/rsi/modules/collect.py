from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rsi.core.errors import ConfigError
from rsi.core.interfaces import Component
from rsi.core.jsonio import read_jsonl, stable_hash
from rsi.core.types import (
    BenchmarkReport,
    ChatMessage,
    ChatRequest,
    Checkpoint,
    CollectResult,
    Trajectory,
)
from rsi.modules.metrics import capability_table, overall_metrics


class RolloutCollector(Component):
    """Collect single-turn trajectories from the deployed policy endpoint."""

    def __init__(self, runtime: Any, cfg: Any, **params: Any) -> None:
        super().__init__(runtime, cfg, **params)
        self.enabled = self.cfg.get_bool("enabled", False)
        self.prompts_path = self.cfg.get("prompts_path")
        self.limit = self.cfg.get_int("limit", 64)
        self.samples_per_prompt = self.cfg.get_int("samples_per_prompt", 1)
        self.max_tokens = self.cfg.get_int("max_tokens", 1024)
        self.temperature = self.cfg.get_float("temperature", 0.7)
        self.max_prompt_chars = self.cfg.get_int("max_prompt_chars", 8000)

    def run(self, checkpoint: Checkpoint, round_no: int) -> list[Trajectory]:
        if not self.enabled:
            return []
        if not self.prompts_path:
            raise ConfigError("collect.rollout.enabled=true requires collect.rollout.prompts_path")
        path = self.runtime.path(self.prompts_path)
        rows = read_jsonl(path)
        rows = rows[: self.limit]
        inference = self.runtime.inference()
        inference.ensure(checkpoint)
        trajectories: list[Trajectory] = []
        for index, row in enumerate(rows):
            messages = [
                ChatMessage(role=str(m.get("role", "user")), content=str(m.get("content", "")))
                for m in (row.get("messages") or [])
            ]
            prompt = str(row.get("prompt") or "")
            if not messages and prompt:
                messages = [ChatMessage(role="user", content=prompt[: self.max_prompt_chars])]
            if not messages:
                continue
            prompt_id = str(row.get("id", f"prompt-{index}"))
            try:
                response = inference.chat(
                    ChatRequest(
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        n=max(1, self.samples_per_prompt),
                    )
                )
            except Exception as exc:  # noqa: BLE001 - one prompt should not stop collection
                self.logger.warning("rollout failed for %s: %s", prompt_id, exc)
                continue
            for sample_index, text in enumerate(response.texts):
                trajectories.append(
                    Trajectory(
                        id=stable_hash("traj", prompt_id, sample_index, checkpoint.id),
                        prompt=prompt or messages[-1].content,
                        messages=messages + [ChatMessage(role="assistant", content=text)],
                        final_answer=text,
                        reward=row.get("reward"),
                        success=row.get("success"),
                        num_turns=1,
                        response_length=len(text),
                        checkpoint_id=checkpoint.id,
                        round=round_no,
                        metadata={
                            "prompt_id": prompt_id,
                            "capability": row.get("capability", ""),
                            "subcapability": row.get("subcapability", ""),
                            "sample_index": sample_index,
                            "source": row.get("source", "rollout"),
                            **dict(row.get("metadata") or {}),
                        },
                    )
                )
        self.logger.info("collected %d trajectories from %d prompts", len(trajectories), len(rows))
        return trajectories


class CollectService(Component):
    """Runs the configured benchmarks and rollout collector, then summarizes."""

    def __init__(self, runtime: Any, cfg: Any, **params: Any) -> None:
        super().__init__(runtime, cfg, **params)
        self.benchmarks_enabled = self.cfg.get_bool("benchmark.enabled", True)
        self.benchmark_names = [str(v) for v in self.cfg.get_list("benchmark.names", [])]
        self.rollout = RolloutCollector(
            runtime, self.cfg.section("rollout", required=False)
        )

    def run(
        self,
        checkpoint: Checkpoint,
        round_no: int,
        baseline: dict[str, float] | None = None,
    ) -> CollectResult:
        reports: list[BenchmarkReport] = []
        if self.benchmarks_enabled:
            if not self.benchmark_names:
                raise ConfigError("collect.benchmark.enabled=true but collect.benchmark.names is empty")
            for name in self.benchmark_names:
                self.logger.info("running benchmark %s on %s", name, checkpoint.id)
                reports.append(self.runtime.benchmark(name).evaluate(checkpoint, round_no))
        table = capability_table(reports, baseline, self.runtime.catalog)
        aggregate = overall_metrics(reports, table)
        trajectories = self.rollout.run(checkpoint, round_no)
        self.logger.info(
            "collect round %d: %d reports, %d capabilities, %d trajectories",
            round_no,
            len(reports),
            len(table),
            len(trajectories),
        )
        return CollectResult(
            round=round_no,
            checkpoint_id=checkpoint.id,
            reports=reports,
            capabilities=table,
            trajectories=trajectories,
            aggregate=aggregate,
            baseline_capabilities=dict(baseline or {}),
            metadata={"benchmark_names": self.benchmark_names},
        )
