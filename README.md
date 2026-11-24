# RL_RSI

A Self-improvment RL Loop to automatically improve a model performance via reinforcement learning.

> **Implementation status (v0.1)**: the repository includes a runnable modular skeleton, a deterministic offline mock stack, and end-to-end tests. See [ARCHITECTURE.md](ARCHITECTURE.md) for the system design.
>
> ```bash
> # Validate configuration and plugins without calling external systems
> PYTHONPATH=src python -m rsi validate --config configs/default.yaml
>
> # Run the classical loop offline
> PYTHONPATH=src python -m rsi run --config configs/demo_mock.yaml --run-id demo
>
> # Inspect collection and validation scores and promotion decisions
> PYTHONPATH=src python -m rsi inspect runs/demo --history
>
> # Run the full test suite
> PYTHONPATH=src python -m unittest discover -s tests -t .
> ```
>
> Real vLLM / verl / EvalScope / agent harness integrations use adapters. Example settings are commented in `configs/default.yaml`. External commands are recorded in `runs/<id>/commands.jsonl`, and checkpoints are managed in `runs/<id>/checkpoints/registry.json`.

loop for: trajectory collect(rollout or from inference) -> analyze bad case -> task synthesis(data collect\env synthesis) -> RL train -> update checkpoint ->  traj collect

This feature will act like glue, combining Verl, VLLM, Harness, evalscope and certain algorithms into an RSI Loop.

**A classical Loop** (May support other loop Orchestration):

## 1. Collect trajectory/benchmark and analysis failures

- collect trajectory from deploy endpoint during a period of time.(Due to data security and privacy concerns, we are unable to collect users' environmental and data information.)
- collect newest benchmark and merge into the former benchmark.
- Collect failure traces and run benchmark to test model.(via evalscope, etc.)(pass@k\mean@k\response_length)
- Defined a set of foundation capabilities.
- Summarize which capabilities are missing.(use benchmark maps or LLM judge)
- Prioritize the missing capabilities based on their foundational nature and importance.
- deliver missing capabilities map and their priorities.

## 2. Determine a new data strategy to fix the model
- Need some smart algorithm to produce data strategy. 
  e.g.:<br />
      1.Greedily cover abilities <br />
      2.Assign greater data weight to more important and fundamental capabilities.<br />
      3.Ensure that all affected domains have corresponding data coverage.<br />
      4.Ensure that data strategies for other dimensions are not significantly affected.<br />
- Datasource:<br />
              1. Private datasets (adapters can be defined to specify data retrieval interfaces)<br />
              2. from web<br />
              3. from hf or github<br />
              4. Synthetic data<br />
- Combine recent train.jsonl data with newly sampled data to synthesize a dataset and reduce forgetting. Validation samples are excluded from replay.
- Deliver a new dataset and the desired mix ratio.
- Enhance the reward associated with the data

## 4. use RL framework to train by model it-self.(via verl, slime, etc.)
- Filter the data with difficulty level from data pool.
  - Efficient filtering algorithm. rollout n increase 4->8->16->32 gradually, can reduce half of rollout time. Under independent samples, at 10% accuracy the all-wrong probability falls from 18.53% at 16 rollouts to 3.43% at 32. Expected work rises from 10.07 to 12.19 completions per candidate (+21%). The 32-rollout extension is enabled when the filter is enabled and can be turned off. See [the feature theory and configuration](docs/adaptive_rollout_filter.md).
  - Preserve data mixing ratio after filtering data.
    - With `data.mixing.preserve_ratio` enabled, select the largest dataset whose integer group quotas fit available samples, group bounds, the total cap, and the configured absolute share tolerance (`ratio_tolerance`, 0.10 by default). See [the data mixing theory and configuration](docs/data_mixing_ratio.md).
    - Start with the strict difficulty band and widen it by `relax_by` for up to `relax_max_steps` steps. Use the smallest widening that reaches the largest feasible mix, and fill quotas with strict samples first. All candidates must pass the upstream filter pipeline. If no feasible mix exists, training skips that dataset.
    - The JSONL builder may reduce the validation holdout to keep the training mix within tolerance. The dataset manifest records the target, selected, and training shares.
- Smoke run.<br /> 1. Ensure reward no hacking. 
            <br /> 2. Ensure data strategy work. 
            <br /> 3. finetune Training hyper parameter.
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

- Go back to filter data step and continue training.
- Finally get a new version of checkpoint which is good at validation score.


## 5. Check new checkpoint
- run benchmark check if desired domain has been improved.
- if ok, run OPD to merge new domain abilities to previous checkpoint.
- if not ok, return to step 2
