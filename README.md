# RL_RSI

A Self-improvment RL Loop to automatically improve a model performance via reinforcement learning.

> **Implementation status (v0.1)**: the repository includes a runnable modular skeleton, a deterministic offline mock stack, and end-to-end tests. See [ARCHITECTURE.md](ARCHITECTURE.md) for the system design.
>
> ```bash
> # Validate configuration and plugins without calling external systems
> PYTHONPATH=src python -m rsi validate --config configs/default.yaml
>
> # Run the classical loop offline
> PYTHONPATH=src python -m rsi run --config configs/default.yaml --run-id demo
>
> # Inspect collection and validation scores and promotion decisions
> PYTHONPATH=src python -m rsi inspect runs/demo --history
>
> # Run the full test suite
> PYTHONPATH=src python -m unittest discover -s tests -t .
> ```
>
> Real vLLM / verl / EvalScope / agent harness integrations use adapters. Example settings are commented in `configs/default.yaml`. External commands are recorded in `runs/<id>/commands.jsonl`, and checkpoints are managed in `runs/<id>/checkpoints/registry.json`.
>
> The example datasets and benchmarks are hand-written examples:
>
> - `examples/datasets/*.jsonl` are local training sets registered in `configs/datasets.yaml`. When a judged defect asks for `kind: dataset`, the loop loads these rows instead of synthesizing them.
> - `examples/benchmark_*.jsonl` are file-backed JSONL benchmarks. They need a real inference endpoint; with the mock endpoint every item scores 0.
> - `configs/datasets.yaml` is the dataset registry, one entry per dataset (open-source or self-defined). Each entry gives the download `url`, the `abilities` it covers, the `reward` and `reward_code` that score a response, and the `preprocess` pipeline that turns the download into local rows.
> - Datasets from verl `examples/reasoning_rl` (math, code, logic, stem, instruction following) keep their rows as is: the `jsonl` and `parquet` loaders map them to the capabilities. Each registered dataset holds one example row. Scoring uses the built-in reward in `src/rsi/rewards/reasoning/`.
> - Datasets from verl-toolRL `agentic_rl_data` (tool-use agents) are filed under ten fine-grained `agentic.*` capabilities, one per dataset focus: single-turn tool call, stateful multi-turn tool use, interleaved user turns, clarify when under-specified, tool irrelevance, parallel tool calls, multi-domain state operations, search QA, math with code reasoning and code from tests. The source is the sub-capability. Each row keeps its tool schemas and task, and each registered dataset holds one example row built with the package's own loaders. Scoring uses the built-in reward in `src/rsi/rewards/agentic/`. `mt_fc` and `code_exec` scores need the env state of a multi-turn rollout.

loop for: trajectory collect(rollout or from inference) -> analyze bad case -> task synthesis(data collect\env synthesis) -> RL train -> update checkpoint ->  traj collect

This feature will act like glue, combining Verl, VLLM, Harness, evalscope and certain algorithms into an RSI Loop.

**A classical Loop** (May support other loop Orchestration):

## 1. Collect trajectory/benchmark and analysis failures

- collect trajectory from deploy endpoint during a period of time.(Due to data security and privacy concerns, we are unable to collect users' environmental and data information.)
- collect newest benchmark and merge into the former benchmark.
- Collect failure traces and run benchmark to test model.(via evalscope, etc.)(pass@k\mean@k\response_length)
- Defined a set of foundation capabilities.
  - The catalog is `configs/capabilities.yaml`: ability names, priorities, and reward and verifier settings. See [capability catalog](docs/capability_analysis.md#capability-catalog-and-analyzers).
- Define a task-capabilities map. User can custom its own task.
  - The optional `trajectory_agent` analyzer estimates ability priorities from collected traces.
- Define a task-benchmark map. To cover required capabilities.
- Define a benchmark-capabilities map. To digest which capabilities not cover
  - `configs/benchmarks.yaml` maps each benchmark and subset to the capabilities it covers.
- Summarize which capabilities are missing.(use benchmark maps or LLM judge)
  - Three analyzers: low scores against a baseline (`rule`), LLM-judged bad cases (`llm_judge`), and `missing_capability`, which compares scores with same-size open-model references. See [missing capability analyzer](docs/capability_analysis.md#missing-capability-analyzer) and [its reference rule](docs/capability_decision_theory.md#reference-rule-as-a-one-sided-test).
- Prioritize the missing capabilities based on their foundational nature and importance.
  - Severity scales with the catalog priority, and findings from all analyzers are merged. See [merging findings](docs/capability_decision_theory.md#merging-of-analyzer-findings) and [planning order](docs/capability_decision_theory.md#planning-order).
- deliver the missing-capability map and priorities that chapter 2 folds into the task's target capability distribution.
  - `data.plan.scope` limits the run to a target capability scope. See [capability scope](docs/capability_analysis.md#capability-scope).

## 2. Determine a new data strategy to fix the model
- Datasource:<br />
              1. Private datasets (adapters can be defined to specify data retrieval interfaces)<br />
              2. from web<br />
              3. from hf or github<br />
              4. Synthetic data<br />
  - Private datasets are registered in `configs/datasets.yaml`, one entry per dataset, and loaded by the `dataset_library` collector. See [the dataset registry and collector](docs/dataset_library.md).

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
