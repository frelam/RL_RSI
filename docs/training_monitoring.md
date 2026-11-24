# Training Monitoring

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

The `training_scale` rule can recommend a larger batch and rollout count when reward is high, validation is stable, and log perplexity remains low. It also reports an LR multiplier based on the configured linear or square-root rule. A plateau alone is not a reason to increase scale. See [training strategy and LR scaling](training_strategy.md) for the reasoning and configuration.

## Observe-action lifecycle

`MonitorHub` runs only the configured plugins, retains the latest finding for each finding code, and can stop a run on critical findings or saturation. A finding records severity, metric evidence, and a suggestion. `Supervisor` translates critical and warning suggestions into retry overrides when a retry occurs. Informational recommendations are recorded for review. The data feedback loop separately handles recognized findings such as reward/validation divergence and plateaus. Custom plugins can follow the same contract without changing the trainer or loop.

Rules should use windows and minimum sample counts to avoid reacting to single noisy observations. Thresholds belong in configuration. Prefer advisory findings for ambiguous signals, and reserve automatic stopping for failures such as non-finite metrics or severe instability.
