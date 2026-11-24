# RL-RSI Architecture

> Status: runnable modular skeleton (v0.1). The default configuration uses a deterministic offline mock stack. Real vLLM, verl, EvalScope, and agent harness integrations are provided through adapters.

## Purpose

RL-RSI orchestrates a recursive improvement cycle around existing training and evaluation systems. It coordinates collection, analysis, data synthesis, training, validation, checkpoint promotion, and stopping. It is not a training engine. Its core responsibilities are data contracts, checkpoint lineage, evaluation consistency, reproducibility, and auditability.

The design follows three principles:

1. Python owns deterministic control flow. External harnesses provide bounded, schema-validated suggestions.
2. Components communicate through narrow interfaces and data objects in `src/rsi/core/types.py`.
3. Plugins and configuration select implementations without coupling loops to specific adapters.

## Package boundaries

- `src/rsi/core/` contains types, interfaces, configuration, registry, runtime services, schema checks, and scoring.
- `src/rsi/adapters/` integrates inference, language models, benchmarks, trainers, harnesses, and command execution.
- `src/rsi/hubs/` stores versioned prompts, skills, and capability definitions.
- `src/rsi/modules/` implements collection, analysis, data planning, filtering, mixing, building, monitoring, and feedback.
- `src/rsi/loops/` composes reusable stages into the classical loop and persists loop state.

Dependency direction is `loops -> modules -> core` and `adapters -> core`. Modules do not import loop implementations. Loops resolve plugins through the runtime registry.

## Classical loop

The classical loop synchronizes the current checkpoint, collects benchmark evidence and trajectories, analyzes capability gaps, plans and builds data, trains a candidate, validates it, optionally runs teacher-aware distillation, and promotes or rejects the candidate. A stop policy evaluates the completed round.

Each completed round is persisted in `state.json`. Stage artifacts, dataset manifests, command logs, and the checkpoint registry are stored under `runs/<id>/`. Checkpoint registry state is authoritative. Candidate checkpoints become current only after validation accepts them.

## Training and monitoring

Trainer implementations call an observer with normalized metric records. `MonitorHub` runs the configured monitor plugins and stores findings. The trainer adapter owns backend-specific parsing and field mapping. The command trainer supports JSONL records and configurable metric aliases. See [training strategy and monitoring theory](docs/training_strategy_monitoring.md) for schedule rationale, signal interpretation, and plugin lifecycle.

Critical findings can stop training. The supervisor can retry failed or unstable attempts with suggested overrides. Data feedback can rebuild a dataset with adjusted difficulty or reward weights. Recommendations remain attached to findings and run artifacts for audit.

## Configuration and extension

Configuration supports nested YAML, recursive includes, environment substitution, and dotted CLI overrides. Plugin specs accept registered names or import paths. Components receive their own configuration subtree and implement the interface in `src/rsi/core/interfaces.py`.

To add a trainer, implement `TrainerBackend.train()` and invoke its observer with normalized records. To add a metric parser, normalize backend records in the adapter before notifying the observer. To add an observe-action rule, implement `MonitorAlgorithm.observe(history)`, register it under the `monitor` category, and include it in `train.monitor.algorithms`. Rules can return `Finding` objects with severity, evidence, and suggested actions.

## Validation and tests

```bash
PYTHONPATH=src python -m rsi validate --config configs/default.yaml
PYTHONPATH=src python -m rsi run --config configs/demo_mock.yaml --run-id demo --runs-dir runs
PYTHONPATH=src python -m rsi inspect runs/demo --history
PYTHONPATH=src python -m unittest discover -s tests -t .
```

The mock stack supports offline testing. Real trainer commands are recorded in `commands.jsonl`. External data, model, and benchmark versions should be pinned by deployments that require strict reproducibility.
