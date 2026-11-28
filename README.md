# RL_RSI

A recursive self-improvement loop: measure a model's capability gaps, build the data and reward to close them, RL-train the next checkpoint, and promote it only if the target improves with no regressions.

- **Capability-driven.** Each round starts from measured gaps, not a fixed dataset.
- **Algorithms first, LLM fallback.** Analysis, data selection, filtering and monitoring are deterministic; an LLM agent fills only what rules can't judge.
- **Plugin-based.** Benchmarks, datasets, rewards, trainers and analyzers are swappable plugins for verl, vLLM, EvalScope or your own harness.

## Install

Requires Python 3.10+.

```bash
git clone https://github.com/frelam/RL_RSI.git && cd RL_RSI
python -m pip install -e .
```

Runs offline with the built-in mock stack. [configs/real.yaml](configs/real.yaml) wires vLLM, verl, real benchmarks and OpenAI-compatible judges.

## Usage

Take a model, pick a task preset, and let the loop run:

```bash
# Improve a checkpoint on agentic + reasoning, at most 20 loop rounds
rsi run --model /path/to/checkpoint --task agentic/reasoning --round 20
```

Presets live in [configs/tasks.yaml](configs/tasks.yaml) (`/` or `,` combines them); all other settings come from the config file.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the design and [configs/datasets.yaml](configs/datasets.yaml) for the dataset registry.

**A classical Loop** (May support other loop Orchestration):

## 1. Collect trajectories/benchmarks and analyze missing capabilities

- Start from the serving checkpoint and the capabilities carried over from the last round.
- evidence source:
  - The deployment endpoint: benchmark runs and policy rollouts.
  - Benchmarks: the existing suite, plus new ones added to the registry.
  - The capability catalog: abilities and their priorities. See [capability catalog](docs/capability_analysis.md#capability-catalog-and-analyzers).

- **Capability gap analysis**
  - **Collect evidence.** Run the configured benchmarks; collect rollouts only inside the deployment you control.
    - A failed rollout is logged and skipped.
    - Trajectories feed the trajectory analyzer and, in chapter 2, the rollout collector.
  - **Map coverage.** Map benchmarks to the capabilities they cover.
  - **Detect failures.** Algorithmic analyzers first:
    - `rule`: low-scoring capabilities below the baseline.
    - `missing_capability`: capabilities that no passing benchmark covers, judged against same-size open-model references. See [missing capability analyzer](docs/capability_analysis.md#missing-capability-analyzer).
    - LLM analyzers only for what rules cannot judge: bad-case review, trajectory priorities, and an optional deep dive.
  - **Merge and prioritize.** Merge findings, drop out-of-scope capabilities, and order by catalog priority. See [merging findings](docs/capability_decision_theory.md#merging-of-analyzer-findings) and [planning order](docs/capability_decision_theory.md#planning-order).

## 2. Determine a new data strategy to fix the model
- Data source:
  - Private datasets. See [the dataset registry and collector](docs/dataset_library.md).
  - from web
  - from hf or github
  - Synthetic data

- Auto dataset generation
  - **Adjust the target distribution.** Base on the missing capabilities, adjust target capabilities distribution. See [target adjustment](docs/dataset_selection_theory.md#target-distribution-adjustment).
  - **Optimize dataset selection and ratios.** Use algorithum to determine what datasets to use and their mixing ratio to fit target distribution. See [selection theory](docs/dataset_selection_theory.md#two-stage-distribution-optimization).
  - **Review and rework.** Compare the estimate with the target; the long-tail gap has a tighter tolerance. If not meet the requirement:
    - use algorithum to add a new dataset in the candidate datasets group, and go to 'optimize dataset selection' for a second time. See [review and rework](docs/dataset_selection_theory.md#review-and-rework).
    - if have llm agent, use them to synthetic data.
  - **Materialize the dataset.** If every things go well, excute the dataset plan.  Download through the registry loaders, take each ratio's rows, then run the existing filters, difficulty grading, replay, mixing and builder to produce one dataset version.
    - Have some build-in data process technology
      - Deduplication (using hashing, etc.)
      - Prevent benchmark data leakage
      - Replay data. See [replay share](docs/dataset_selection_theory.md#replay-share).

- Enhance the reward associated base on the new dataset and the problem from step 1.
  - select build-in reward base on new dataset.
  - If have llm agent, write code to imporve the reward. 

## 3. use RL framework to train by model it-self.(via verl, slime, etc.)
- Filter the data with difficulty level from data pool.
  - Efficient filtering algorithm. rollout n increase 4->8->16->32 gradually, can reduce half of rollout time. Under independent samples, at 10% accuracy the all-wrong probability falls from 18.53% at 16 rollouts to 3.43% at 32. Expected work rises from 10.07 to 12.19 completions per candidate (+21%). The 32-rollout extension is enabled when the filter is enabled and can be turned off. See [the feature theory and configuration](docs/adaptive_rollout_filter.md).
  - Preserve data mixing ratio after filtering data.
    - With `data.mixing.preserve_ratio` enabled, select the largest dataset whose integer group quotas fit available samples, group bounds, the total cap, and the configured absolute share tolerance (`ratio_tolerance`, 0.10 by default). See [the data mixing theory and configuration](docs/data_mixing_ratio.md).
    - Start with the strict difficulty band and widen it by `relax_by` for up to `relax_max_steps` steps. Use the smallest widening that reaches the largest feasible mix, and fill quotas with strict samples first. All candidates must pass the upstream filter pipeline. If no feasible mix exists, training skips that dataset.
    - The JSONL builder may reduce the validation holdout to keep the training mix within tolerance. The dataset manifest records the target, selected, and training shares.
- Run a short smoke check before a formal run. The offline mock example is in [the smoke-run guide](docs/smoke_run.md). For a real trainer, validate the config and run only a few optimizer steps to check reward examples, data selection, metric health, and basic infrastructure. Change one setting at a time; use the observed training mix before deciding whether to adjust its target ratio.
- Formal Run.
- Use a staged training schedule.
  - Start with conservative batch and rollout sizes to validate reward behavior and learn a useful reward landscape. Increase scale only when metrics indicate stable learning and additional gradient averaging is valuable. See [training strategy](docs/training_strategy.md).
  - When increasing batch size, the `training_scale` monitor can suggest a learning-rate increase using linear or square-root scaling. Configure `lr_scaling` and review the assumptions in [batch-size and learning-rate scaling](docs/training_strategy.md#learning-rate-scaling-with-batch-size).
- Monitor metric and take action.
  - Metric like validation, entropy, reward, gradient norms, response length, and other backend metrics through a normalized metric record.
  - Trainer adapters can map backend-specific field names using `trainer.metric_aliases`.
  - Rules are independently registered plugins selected through `train.monitor.algorithms`, and findings carry suggested actions for audit or retry handling. Built-in rules cover non-finite metrics, gradient spikes, entropy collapse, reward/validation divergence, validation plateaus, saturation, abrupt response-length growth, and stable high-reward plateaus where a larger training scale may be worth testing. See [metric-driven monitoring](docs/training_monitoring.md).
  - Build-in rules:
    - Use reward/validation divergence as evidence to audit reward and verification, then refilter toward examples the policy solves inconsistently.
    - Falling entropy alongside flat validation can indicate premature policy concentration, but should be checked against reward, KL, and validation signals before increasing difficulty.
    - Large response-length jumps merit repetition analysis and length-aware reward guards.
    - Domain-level reward imbalance should be monitored with domain-tagged metrics and addressed through targeted data quotas. These are diagnostic suggestions, not universal prescriptions.
- Use RL scaling law to predict max performance and the required step
  - Refer to https://natolambert.com/slides/2026/cornell-post-training/talk#33
  - The opt-in `scaling_law` monitor fits a bounded logistic curve after validation accuracy shows a slope peak and subsequent decline. It refreshes on each metric record, tracks compute using the current batch size, mini-batch size, and rollout count, and reports remaining compute plus current-rate step equivalents for p50, p75, p90, and the asymptotic p100 ceiling. See [RL scaling projection theory and configuration](docs/rl_scaling_law.md) for assumptions and limitations.

- Go back to filter data step and continue training.
- If run out of data, jump out the Loop. ( more than 2 epoch, to avoid overfit)
- Finally get a new version of checkpoint which is good at validation score.


## 4. Check new checkpoint
- run benchmark check if desired domain has been improved.
  - The candidate is promoted only when a target capability improves and no other measured capability regresses. Targets that did not improve are carried to the next round. See [validation](docs/capability_analysis.md#validation) and [validation as a joint test](docs/capability_decision_theory.md#validation-as-a-joint-test).
- if ok, run OPD to merge new domain abilities to previous checkpoint.
- if not ok, return to step 2
