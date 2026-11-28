# Capability analysis and validation

This document covers the three places where the loop reasons about capabilities: the catalog and the analyzers that estimate what the model lacks, the scope that limits a run, and the validation that decides whether a candidate checkpoint is promoted. See [capability analysis and validation theory](capability_decision_theory.md) for the statistical reading of the rules and their error rates.

## Rollout evidence

`collect.rollout` sends prompts from `prompts_path` to the serving checkpoint and stores the trajectories for the trajectory analyzer and rollout reuse. With `collect.rollout.fail_cases.enabled`, it also reads a JSONL fail-case database of real user failures. Each row needs a `prompt` or `messages` and may carry `capability`, `subcapability`, `success` and `reward`. The collected trajectory keeps `fail_case: true` in its metadata, so a review can separate user-reported cases from synthetic prompts.

## Capability catalog and analyzers

`configs/capabilities.yaml` is the shared vocabulary between benchmarks, analysis, and data planning. It defines ability names, priorities, reward and verifier settings, and benchmark mappings. Benchmark adapters may also declare a URL and description. Reports include the capabilities observed in their evaluated items.

Two optional analyzers add to the rule and LLM-judge findings:

- `trajectory_agent` estimates task ability priorities from collected rollout traces.
- `missing_capability` infers missing capabilities from benchmark results (next section).

The analysis merger combines the defects of all analyzers and their ability priority estimates using configurable source weights under `analyze.merge`. Defect severity uses the strongest weighted signal plus a corroboration bonus, and confidence uses noisy-OR. A priority estimate is weighted by `rule_profile_weight` for the rule analyzer, `agent_profile_weight` for `trajectory_agent`, and `default_weight` for any other source.

## Missing capability analyzer

Enable the `missing_capability` analyzer under `analyze.methods` to infer missing capabilities from benchmark results:

1. A benchmark performs well when its score is not much lower than the reference score of an open-weight model of the same parameter count.
2. Every capability that a well-performing benchmark or subset covers is marked as present.
3. Every catalog capability that stays unmarked is reported as a missing capability.

### Inputs

- `configs/benchmarks.yaml` maps each benchmark to the capabilities it covers. An optional `subsets` entry lists only the subsets that differ from the benchmark-level capabilities.
- `configs/benchmark_reference.yaml` gives the mean and standard deviation of open-model scores per benchmark, keyed by parameter count. A `benchmark.subset` entry holds the curve of one subset. The reference is interpolated linearly between neighbouring sizes and clamped outside the table.
- The model size comes from `params_b`, or from `checkpoint.initial.params_b`. The analyzer fails validation when a reference file is set and neither is given.

### Judging a report

A report that labels its items with a subset is judged subset by subset; otherwise the benchmark is judged as a whole, using its primary metric. A benchmark or subset performs well when:

```
score >= reference_mean - max(margin, sigma * reference_std)
```

A benchmark or subset without a reference curve is skipped, or judged against `absolute_threshold` when `unreferenced: absolute`. When every collected benchmark is skipped, the analyzer reports no defects.

### Output

- A missing capability gets severity `min(0.98, severity * catalog_priority)` and a suggested data request that uses the catalog entry's kind, seeds, and verifier.
- A capability covered only by benchmarks or subsets that do not perform well is listed with their scores as evidence.
- The report metadata records `params_b`, the per-benchmark judgement (`score`, `reference`, `good`), `skipped`, `marked_capabilities`, and `missing_capabilities`.

### Configuration

| Option under the `missing_capability` entry | Default | Effect |
| --- | --- | --- |
| `registry` | `configs/benchmarks.yaml` | Benchmark to capability mapping. |
| `reference` | `configs/benchmark_reference.yaml` | Reference curves. An empty value disables them. |
| `params_b` | `checkpoint.initial.params_b` | Model size in billions. |
| `margin` | `0.05` | Absolute tolerance below the reference. |
| `sigma` | `0.0` | Extra tolerance of this many reference standard deviations. |
| `unreferenced` | `skip` | `skip` or `absolute`. |
| `absolute_threshold` | `0.5` | Reference used when `unreferenced: absolute`. |
| `severity` | `0.6` | Base severity, scaled by the catalog priority. |
| `confidence` | `0.5` | Defect confidence. |
| `target_difficulty` | `[0.2, 0.9]` | Difficulty band of the suggested data. |

## Capability scope

`data.plan.scope` limits the run to a target capability scope. Defects outside the scope are dropped before planning, and the merged report records `scope` and `out_of_scope_defects`. The benchmarks should cover every capability in the scope, otherwise the analyzer reports it as missing or the validation cannot measure its improvement. An empty scope applies no restriction.

## Validation

Validation compares the candidate with its parent on the round's target capabilities (the capabilities requested by the data plan) and on the other measured capabilities. The result is written to `validation.json` and to the checkpoint metrics.

### Modes

`validate.mode` selects the evidence:

- `benchmark` runs the benchmarks in `validate.benchmark.names`, or `collect.benchmark.names` when empty, on both the parent and the candidate. Benchmark reports collected for the parent in the same round are reused.
- `metrics` reads the trainer's best metric and needs no inference endpoint.
- `auto` uses `benchmark` when benchmark names are set and the inference backend can switch checkpoints (or `force_benchmark` is enabled), otherwise `metrics`.

### Decision

The overall metric (`validate.metric`, `mean_score` by default) must be a finite score above zero, and its delta against the parent must reach `min_delta`.

Benchmark validation adds three conditions, checked in order:

- The parent and candidate reports must cover the same items without errors or invalid scores.
- At least one target capability must improve by more than zero and by at least `target_min_delta`. Otherwise the candidate is rejected.
- No other measured capability may regress by more than `max_other_regression`, unless `allow_regression` is enabled.

### Carried targets

Targets that did not improve (`failed_targets`) and targets that no benchmark measures (`unmeasured_targets`) are listed in the result. They are saved as `pending_capabilities` in `state.json`, together with earlier pending capabilities that did not improve in this round. The next round plans their data first, before other defects.

### Configuration

| Option under `validate` | Default config | Effect |
| --- | --- | --- |
| `mode` | `auto` | `auto`, `benchmark`, or `metrics`. |
| `metric` | `mean_score` | Overall metric compared against `min_delta`. |
| `min_delta` | `0.0` | Minimum overall improvement. |
| `target_min_delta` | `0.0` | Minimum improvement of a target capability. |
| `max_other_regression` | `0.01` | Allowed drop of a non-target capability. |
| `allow_regression` | `false` | Accept other regressions and a missing overall delta. |
| `force_benchmark` | `false` | Use benchmark validation in `auto` mode even when the backend cannot switch checkpoints. |
| `benchmark.names` | `[]` | Benchmarks to run. Empty reuses `collect.benchmark.names`. |
