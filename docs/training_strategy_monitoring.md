# Training Strategy and Metric-Driven Monitoring

## Training scale

Batch size and rollout count control different sources of training noise. Larger batches average more policy updates and reduce the variance of gradient estimates, while larger rollout groups improve estimates of outcome variance and expose examples with mixed success. Both increase compute and can hide reward or data problems when enabled too early.

A practical schedule starts with conservative values while validating the reward implementation, data filters, and end-to-end trainer path. Increase one scale dimension at a time after reward and validation behavior are coherent. This makes regressions easier to attribute. Retain examples with mixed outcomes as the policy improves, since uniformly easy examples provide little learning signal and uniformly impossible examples may provide weak or misleading advantage estimates.

The exact schedule depends on the algorithm, reward estimator, sequence length, and hardware. RSI therefore reports scale suggestions instead of imposing universal batch or rollout values. The `training_scale` rule looks for high reward, stable validation, and low `log_perplexity`, then suggests multipliers for a later run. A trainer command can expose tunable values with `{override_<name>}` placeholders. Retry findings can also provide overrides to the command adapter.

## Metric normalization

Trainer backends publish different names and record shapes. The command trainer accepts JSONL records with a top-level `step` or `global_step` and either a nested `metrics` object or flat metric fields. `trainer.metric_aliases` maps backend names onto canonical RSI names. For example:

```yaml
trainer:
  metric_aliases:
    val_score: [validation/score, eval/reward]
    log_perplexity: [actor/log_perplexity]
```

Monitor rules consume normalized records with `step` and `metrics`. A custom backend can implement its own parser before invoking the observer. A custom metric rule can be registered as a plugin and selected in `train.monitor.algorithms`.

## Interpreting signals

Metrics are evidence about a training process, not direct diagnoses. Reward rising while held-out validation stays flat can indicate reward exploitation, train/validation mismatch, or a lagging validation metric. Audit the verifier and inspect representative examples before changing data. Low entropy with flat validation can indicate premature policy concentration, but entropy is algorithm-dependent and should be read alongside KL, reward, and validation.

Abrupt response-length growth can reflect legitimate task complexity or repetition. Inspect sampled generations before adding length penalties, because a blunt penalty can suppress valid reasoning. Domain-level reward gaps require domain-tagged metrics and should be interpreted with sample counts and difficulty mix. Targeted refiltering or quota changes should follow evidence that a domain is overrepresented or saturated.

When reward is strong, validation changes little, and log perplexity remains low, the policy may benefit from more diverse rollouts or a larger batch. It may also simply need more optimization steps. Run a controlled follow-up and change one scale parameter at a time. Plateau detection alone is not a reason to increase scale.

## Observe-action lifecycle

`MonitorHub` runs only the configured plugins, retains the latest finding for each finding code, and can stop a run on critical findings or saturation. A finding records severity, metric evidence, and a suggestion. `Supervisor` translates critical and warning suggestions into retry overrides when a retry occurs. The data feedback loop separately handles recognized findings such as reward/validation divergence and plateaus. Informational recommendations are recorded for review. Custom plugins can follow the same contract without changing the trainer or loop.

Rules should use windows and minimum sample counts to avoid reacting to single noisy observations. Thresholds belong in configuration. Prefer advisory findings for ambiguous signals, and reserve automatic stopping for failures such as non-finite metrics or severe instability.
