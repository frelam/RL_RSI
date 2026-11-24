# Training Strategy

## Staged scaling

Start with conservative batch and rollout sizes while validating the reward implementation, data filters, and trainer path. Increase one scale dimension at a time after reward and validation behavior are coherent. Retain examples with mixed outcomes as the policy improves, since uniformly easy examples provide little learning signal and uniformly impossible examples may provide weak or misleading advantage estimates.

Batch size and rollout count affect different sources of noise. A larger batch averages more sample-level gradient contributions. More rollouts per prompt improve outcome estimates and expose prompts with mixed success. Both increase compute, and either can obscure reward or data problems when increased too early.

## Learning-rate scaling with batch size

When increasing batch size by a factor $k$, RSI's `training_scale` monitor can recommend increasing the learning rate by either the same factor (linear scaling) or its square root. Configure `lr_scaling` as `linear` or `sqrt`. The default is `sqrt`. The recommendation is exposed as `learning_rate_multiplier` alongside `batch_size_multiplier`; a command trainer can consume it through a `{override_<name>}` placeholder when a retry or later run applies the finding.

The linear rule uses $\eta' = k\eta$. It approximately preserves the expected parameter displacement per processed example when the larger batch reduces the number of optimizer updates by $k$. This is most defensible when gradients are sufficiently aligned and the optimizer remains in a stable region. It can become too aggressive when curvature, clipping, momentum, adaptive optimizer state, or policy constraints dominate.

The square-root rule uses $\eta' = \sqrt{k}\eta$. Under a noise-dominated approximation, averaging $k$ times as many independent samples reduces gradient standard deviation by about $\sqrt{k}$. Scaling the step by $\sqrt{k}$ can then maintain a similar signal-to-noise ratio while limiting the step increase. This is a more conservative heuristic, not a guarantee of equivalent optimization.

These rules are starting points rather than universal laws. RL gradients are correlated, advantage normalization and clipping alter the effective update, and rollout count may change the data distribution as well as estimator variance. Check KL, gradient norms, reward, entropy, and held-out validation after changing batch and learning rate. Use a controlled follow-up and reduce the rate if updates become unstable. The monitor reports an advisory multiplier and does not change trainer settings by itself.

## Configuration

The `training_scale` monitor requires a window with strong reward, stable validation, and low `log_perplexity`. Its batch multiplier determines both the suggested batch-size increase and the LR multiplier formula.

```yaml
train:
  monitor:
    algorithms:
      - type: training_scale
        batch_multiplier: 1.5
        rollout_multiplier: 1.5
        lr_scaling: sqrt # linear | sqrt | none
```

Set `lr_scaling: none` to report no learning-rate multiplier. Trainer commands may make suggested values configurable with `{override_learning_rate}` or a backend-specific placeholder corresponding to the LR value used by that command. See [metric-driven monitoring](training_monitoring.md) for metric normalization and findings.
