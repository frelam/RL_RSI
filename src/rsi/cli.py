from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from rsi import __version__
from rsi.bootstrap import bootstrap
from rsi.core.config import Config, dump_config, load_config
from rsi.core.errors import RsiError
from rsi.core.jsonio import read_json
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
