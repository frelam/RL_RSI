from __future__ import annotations

import json
from typing import Any

from rsi.adapters.llm import _substitute
from rsi.adapters.templating import format_template
from rsi.core.errors import AdapterError
from rsi.core.interfaces import HarnessBackend
from rsi.core.jsonio import atomic_write_json, read_json
from rsi.core.registry import register
from rsi.core.schema import validate_json
from rsi.core.types import HarnessResult, HarnessTask


@register("harness", "mock")
class MockHarness(HarnessBackend):
    """Fixture-driven harness for offline loops and tests.

    Fixtures are keyed by ``task.name`` (or ``*``) and may contain
    ``{{round}}`` placeholders from ``task.metadata``.
    """

    def __init__(self, runtime: Any, cfg: Any, **params: Any) -> None:
        super().__init__(runtime, cfg, **params)
        self.fixtures: dict[str, Any] = self.cfg.get_dict("fixtures", {})
        self._calls: dict[str, int] = {}

    def run(self, task: HarnessTask) -> HarnessResult:
        fixture = self.fixtures.get(task.name, self.fixtures.get("*"))
        if fixture is None:
            raise AdapterError(
                f"MockHarness has no fixture for task {task.name!r}. Available: {sorted(self.fixtures)}"
            )
        index = self._calls.get(task.name, 0)
        self._calls[task.name] = index + 1
        item = fixture[index % len(fixture)] if isinstance(fixture, list) else fixture
        variables = dict(task.metadata)
        variables.update({"round": task.metadata.get("round", 0), "task_name": task.name, "call_index": index})
        item = _substitute(item, variables)
        output = item if isinstance(item, dict) else {"text": str(item)}
        errors = validate_json(output, task.output_schema) if task.output_schema else []
        task.work_dir.mkdir(parents=True, exist_ok=True)
        output_path = task.work_dir / "mock_output.json"
        atomic_write_json(output_path, output)
        return HarnessResult(
            ok=not errors,
            output=output,
            raw_text=json.dumps(output, ensure_ascii=False)[:20000],
            duration_s=0.0,
            error="; ".join(errors) if errors else None,
            command="mock",
            artifacts=[str(output_path)],
            metadata={"mock": True, "skills": task.skills},
        )


@register("harness", "command")
class CommandHarness(HarnessBackend):
    """Run any external agent harness as a bounded subprocess.

    Contract: the harness reads ``{task_file}`` and writes a JSON object to
    ``{output_file}``. Skills are copied into ``{skills_dir}``. Output is
    schema-validated before it can influence the loop.
    """

    def __init__(self, runtime: Any, cfg: Any, **params: Any) -> None:
        super().__init__(runtime, cfg, **params)
        self.command = self.cfg.get("command")
        if not self.command:
            raise AdapterError("command harness requires 'command'")
        self.default_timeout_s = self.cfg.get_int("timeout_s", 1800)
        self.env = self.cfg.get_dict("env", {})
        self.output_name = str(self.cfg.get("output_name", ".rsi_output.json"))
        self.task_name = str(self.cfg.get("task_name", ".rsi_task.json"))

    def run(self, task: HarnessTask) -> HarnessResult:
        task.work_dir.mkdir(parents=True, exist_ok=True)
        task_file = task.work_dir / self.task_name
        output_file = task.work_dir / self.output_name
        skills_dir = task.work_dir / ".rsi_skills"
        materialized: list[str] = []
        if task.skills:
            self.runtime.skill_hub.materialize(task.skills, skills_dir)
            materialized = list(task.skills)
        payload = {
            "name": task.name,
            "instruction": task.instruction,
            "skills": task.skills,
            "input_files": task.input_files,
            "output_schema": task.output_schema,
            "metadata": task.metadata,
        }
        atomic_write_json(task_file, payload)
        variables = {
            "task_file": str(task_file),
            "output_file": str(output_file),
            "work_dir": str(task.work_dir),
            "skills_dir": str(skills_dir),
            "task_name": task.name,
            "round": task.metadata.get("round", 0),
        }
        command = format_template(self.command, variables)
        timeout = task.timeout_s or self.default_timeout_s
        started_error: str | None = None
        stdout = ""
        try:
            completed = self.runtime.executor.run(
                command,
                cwd=task.work_dir,
                env=self.env,
                timeout=timeout,
                label=f"harness.{task.name}",
            )
            stdout = (completed.stdout or "") + (completed.stderr or "")
            if completed.returncode != 0:
                started_error = f"harness exited {completed.returncode}"
        except AdapterError as exc:
            started_error = str(exc)
        output: dict[str, Any] | None = None
        if output_file.exists():
            try:
                loaded = read_json(output_file)
                output = loaded if isinstance(loaded, dict) else {"value": loaded}
            except (OSError, json.JSONDecodeError) as exc:
                started_error = started_error or f"invalid harness output: {exc}"
        errors = validate_json(output, task.output_schema) if (task.output_schema and output is not None) else []
        if output is None and not errors:
            errors = ["harness produced no output file"]
        ok = started_error is None and not errors
        return HarnessResult(
            ok=ok,
            output=output,
            raw_text=stdout[-20000:],
            error="; ".join([e for e in [started_error, *errors] if e]) or None,
            command=command if isinstance(command, str) else " ".join(command),
            artifacts=[str(output_file)] if output_file.exists() else [],
            metadata={"skills_materialized": materialized},
        )
