from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import yaml

from rsi import __version__
from rsi.bootstrap import bootstrap
from rsi.core.config import Config, dump_config, load_config
from rsi.core.errors import ConfigError, RsiError
from rsi.core.jsonio import read_json, read_jsonl
from rsi.core.registry import list_plugins
from rsi.core.runtime import Runtime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rsi",
        description="Code-driven recursive self-improvement loop for RL post-training.",
    )
    parser.add_argument("--version", action="version", version=f"rsi {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a loop")
    run.add_argument("--config", default="configs/default.yaml")
    run.add_argument("--model", default=None, help="initial checkpoint path")
    run.add_argument(
        "--task",
        default=None,
        help="task preset(s) from tasks.yaml, e.g. agentic/reasoning",
    )
    run.add_argument("--round", dest="rounds", type=int, default=None, help="maximum loop rounds")
    run.add_argument("--run-id", default=None)
    run.add_argument("--runs-dir", default=None)
    run.add_argument("--seed", type=int, default=None)
    run.add_argument("--resume", action="store_true", help="continue an existing run")
    run.add_argument("--dry-run", action="store_true", help="validate the full component tree only")
    run.add_argument("--quiet", action="store_true")
    run.add_argument("--debug", action="store_true")
    run.add_argument(
        "overrides",
        nargs="*",
        metavar="KEY=VALUE",
        help="dotted config overrides, e.g. stop.combine=all analyze.merge.min_confidence=0.3",
    )

    validate = sub.add_parser("validate", help="construct every configured plugin without running")
    validate.add_argument("--config", default="configs/default.yaml")
    validate.add_argument("--dump-config", action="store_true")
    validate.add_argument("--quiet", action="store_true")
    validate.add_argument("overrides", nargs="*", metavar="KEY=VALUE")

    plugins = sub.add_parser("plugins", help="list registered plugins")
    plugins.add_argument("--category", default=None)

    inspect = sub.add_parser("inspect", help="print the state of a run directory")
    inspect.add_argument("run_dir")
    inspect.add_argument("--history", action="store_true", help="include round history")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        bootstrap()
        if args.command == "plugins":
            return _plugins(args.category)
        if args.command == "inspect":
            return _inspect(args.run_dir, args.history)
        if args.command == "run":
            return _run(args)
        if args.command == "validate":
            return _validate(args)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        if getattr(args, "debug", False):
            traceback.print_exc()
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 1
    return 1


def _run(args: argparse.Namespace) -> int:
    config = load_config(args.config, args.overrides)
    if args.model:
        _check_model_path(args.model)
        config.set("checkpoint.initial.model_path", args.model)
    if args.task:
        abilities = _apply_task_preset(config, args.task)
        _warn_unmeasured_abilities(config, abilities)
    if args.rounds is not None:
        _set_max_rounds(config, args.rounds)
    run_id = args.run_id or str(config.get("run.id", "")) or time.strftime("run-%Y%m%d-%H%M%S")
    runtime = Runtime(
        config,
        run_id=run_id,
        runs_dir=args.runs_dir,
        seed=args.seed,
        quiet=args.quiet,
        dry_run=args.dry_run,
    )
    try:
        loop = runtime.build(
            "loop",
            config.plugin("loop", "classic"),
            config.section("loop", required=False),
        )
        if args.dry_run:
            loop.validate()
            stages = [stage.name for stage in getattr(loop, "stages", [])]
            print(f"dry-run OK: loop={getattr(loop, 'name', '?')} run_dir={runtime.run_dir}")
            print(f"stages: {' -> '.join(stages)}")
            return 0
        result = loop.run(resume=args.resume)
        print(f"run {result.status}: {result.stop_reason}")
        print(f"rounds: {result.rounds}")
        print(
            f"best checkpoint: {result.best_checkpoint or '<none>'}"
            f" (score {result.best_score:.4f})"
        )
        print(f"artifacts: {result.run_dir}")
        return 0 if result.status == "stopped" else 1
    finally:
        runtime.close()


def _check_model_path(value: str) -> None:
    """Fail early on a local checkpoint path that does not exist.

    URI-style values (``mock://``, ``hf://``) and bare hub ids are passed
    through; only paths the call site clearly treats as local are checked.
    """
    if "://" in value:
        return
    path = Path(value).expanduser()
    if (path.is_absolute() or value.startswith((".", "~"))) and not path.exists():
        raise ConfigError(
            f"Model checkpoint not found: {path}. Pass an existing local path, "
            "a hub id such as Qwen/Qwen3-8B, or a URI such as mock://base-model."
        )


def _apply_task_preset(config: Config, spec: str) -> dict[str, float]:
    """Merge the named presets into the chapter-2 data strategy task."""
    names = [name.strip() for name in re.split(r"[/,]", spec) if name.strip()]
    if not names:
        raise ConfigError("--task must name at least one preset")
    path = config.config_dir / "tasks.yaml"
    if not path.exists():
        path = config.repo_root / "configs" / "tasks.yaml"
    if not path.exists():
        raise ConfigError(f"Task preset file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    presets = raw.get("tasks") or {}
    abilities: dict[str, float] = {}
    datasets: list[str] = []
    for name in names:
        preset = presets.get(name)
        if not isinstance(preset, dict):
            available = ", ".join(sorted(presets)) or "none"
            raise ConfigError(f"Unknown task preset {name!r}; available: {available}")
        for ability, weight in (preset.get("abilities") or {}).items():
            key = str(ability)
            abilities[key] = max(abilities.get(key, 0.0), float(weight))
        for dataset in preset.get("datasets") or []:
            if str(dataset) not in datasets:
                datasets.append(str(dataset))
    config.set("data.strategy.enabled", True)
    config.set(
        "data.strategy.task",
        {"id": spec, "abilities": abilities, "datasets": datasets},
    )
    return abilities


def _benchmark_capabilities(config: Config, name: str) -> set[str]:
    """Capabilities a configured benchmark can measure, best effort."""
    section = config.section(f"benchmark.benchmarks.{name}", required=False)
    kind = str(section.get("type", ""))
    if kind == "mock":
        return set(section.get_dict("capabilities", {}))
    if kind in {"evalscope", "eval_scope"}:
        return {str(value) for value in section.get_dict("capability_map", {}).values()}
    if kind == "jsonl":
        raw = str(section.get("path", "") or "")
        if not raw:
            return set()
        path = Path(raw)
        if not path.is_absolute():
            path = config.repo_root / path
        if not path.is_file():
            return set()
        try:
            rows = read_jsonl(path)
        except (OSError, ValueError):
            return set()
        return {
            str(row["capability"])
            for row in rows
            if isinstance(row, dict) and row.get("capability")
        }
    return set()


def _warn_unmeasured_abilities(config: Config, abilities: dict[str, float]) -> None:
    """Tell the user when the configured benchmarks cannot check the task targets."""
    names: list[str] = []
    for path in ("collect.benchmark.names", "validate.benchmark.names"):
        for value in config.get_list(path, []):
            if str(value) not in names:
                names.append(str(value))
    if not names or not abilities:
        return
    covered: set[str] = set()
    unresolved: list[str] = []
    for name in names:
        found = _benchmark_capabilities(config, name)
        if found:
            covered |= found
        else:
            unresolved.append(name)
    missing = sorted(set(abilities) - covered)
    if not missing:
        return
    detail = f"; coverage unknown for {', '.join(unresolved)}" if unresolved else ""
    print(
        f"warning: {len(missing)}/{len(abilities)} task abilities are not measured by "
        f"benchmarks {names}{detail}: {', '.join(missing)}",
        file=sys.stderr,
    )


def _set_max_rounds(config: Config, rounds: int) -> None:
    if rounds <= 0:
        raise ConfigError("--round must be positive")
    policies = [dict(policy) for policy in config.get_list("stop.policies", [])]
    for policy in policies:
        if str(policy.get("type")) == "max_rounds":
            policy["max_rounds"] = rounds
            break
    else:
        policies.append({"type": "max_rounds", "max_rounds": rounds})
    config.set("stop.policies", policies)


def _validate(args: argparse.Namespace) -> int:
    config = load_config(args.config, args.overrides)
    runtime = Runtime(
        config,
        run_id="validate",
        runs_dir=Path(config.get("paths.runs_dir", "runs")) / "_validate",
        quiet=args.quiet,
        dry_run=True,
    )
    try:
        loop = runtime.build(
            "loop",
            config.plugin("loop", "classic"),
            config.section("loop", required=False),
        )
        loop.validate()
        stages = [stage.name for stage in getattr(loop, "stages", [])]
        print(f"config OK: {config.source}")
        print(f"loop: {getattr(loop, 'name', '?')} ({' -> '.join(stages)})")
        if args.dump_config:
            print(dump_config(config))
        return 0
    finally:
        runtime.close()


def _plugins(category: str | None) -> int:
    plugins = list_plugins()
    if category:
        names = plugins.get(category)
        if names is None:
            print(f"unknown category {category!r}; available: {', '.join(sorted(plugins))}")
            return 1
        for name in names:
            print(name)
        return 0
    for name, entries in plugins.items():
        print(f"{name}:")
        for entry in entries:
            print(f"  - {entry}")
    return 0


def _inspect(run_dir: str, history: bool) -> int:
    path = Path(run_dir) / "state.json"
    if not path.exists():
        print(f"no state.json in {run_dir}", file=sys.stderr)
        return 1
    state = read_json(path)
    summary = {
        "run_id": state.get("run_id"),
        "status": state.get("status"),
        "rounds": state.get("round"),
        "best_checkpoint": state.get("best_checkpoint"),
        "best_score": state.get("best_score"),
        "stop_reason": state.get("stop_reason"),
        "budget": state.get("budget"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if history:
        for record in state.get("history", []):
            validation = record.get("validation") or {}
            print(
                f"round {record.get('round')}: "
                f"{(record.get('collect') or {}).get('score')} -> "
                f"{validation.get('score')} "
                f"({'accepted' if validation.get('accepted') else 'rejected'})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
