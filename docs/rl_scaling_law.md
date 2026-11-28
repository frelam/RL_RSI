# RL performance scaling projection

The `scaling_law` monitor estimates a bounded learning curve from validation accuracy recorded during one training run. It is an advisory forecast for planning additional compute and training steps. It does not control the trainer, change a run's step budget, or guarantee future performance. Once enough evidence exists, the finding is refreshed on every metric record.

## Model and inference

The monitor uses a three-parameter logistic curve with the first observed score fixed as its baseline:

$$
p(c)=p_0 + (p_{\infty}-p_0)\,\sigma(kc+b),\qquad
\sigma(z)=\frac{1}{1+e^{-z}}.
$$

Here $c$ is cumulative relative compute, $p_0$ is the first observed validation score, $p_{\infty}\leq 1$ is the fitted asymptotic score, and $k>0$ and $b$ describe the rate and location of the transition. The inflection point is $c_*=-b/k$. The first observed score anchors the baseline because the monitor does not have evidence about the score before training began.

For a candidate ceiling $p_{\infty}$, the observations can be linearized:

$$
\operatorname{logit}\left(\frac{p(c)-p_0}{p_{\infty}-p_0}\right)=kc+b.
$$

The implementation searches candidate ceilings in $(\max(p),1]$, solves the linear regression for $k$ and $b$, and selects the candidate with the lowest score-space RMSE. This avoids an extra numerical-optimization dependency. It emits a projection only when the observed finite-difference slope has peaked and then declined, there are enough observations, the fitted rate is positive, and the fit RMSE is below the configured limit.

Each record contributes a relative compute increment

$$
\Delta c = B\times N\times\max(1,B/M),
$$

where $B$ is `batch_size`, $M$ is `mini_batch_size`, and $N$ is `rollout_n`. The monitor reads these values from that record, so a mid-run change in batch size, mini-batch size, or rollout count changes the cumulative compute axis immediately. The formula is a planning proxy that treats generated rollouts and optimizer mini-batches as multiplicative costs. It is not a hardware FLOP counter.

The monitor reports four fractions of the fitted improvement from $p_0$ to $p_{\infty}$. `p50`, `p75`, and `p90` are finite targets at 50%, 75%, and 90% of that improvement. `p100` is the asymptotic ceiling. For a target $p_t<p_{\infty}$, the projected cumulative compute is

$$
c_t=\frac{\operatorname{logit}\left(\frac{p_t-p_0}{p_{\infty}-p_0}\right)-b}{k}.
$$

The finding reports the fitted ceiling, current score, cumulative compute, latest step compute, and remaining compute for each level. It also divides remaining compute by the latest step cost to provide an equivalent step count. The same values are written into the current record under names such as `scaling_law/p50_remaining_compute` and `scaling_law/p50_equivalent_steps`, so a trainer stream can display them directly. This count is a current-rate conversion, not a promise about future steps. `p100` has no finite exact compute under a logistic curve and is reported with null remaining compute and step count. A sequence of runs can use the projected remaining compute as a budget estimate and reevaluate after each run with the newly observed history. The estimate changes when the data mix, policy, optimizer, or compute scale changes.

## Feasibility and limitations

A logistic learning curve is a useful low-parameter approximation when held-out accuracy improves smoothly through an acceleration phase and then decelerates toward a ceiling. It captures an S-shaped transition and gives an interpretable midpoint. The second derivative is positive before the inflection, zero at it, and negative afterward. A drop in the second derivative alone does not identify the midpoint: noise, benchmark variance, or a change in training conditions can produce the same observation. The monitor therefore checks the empirical slope peak and decline, then fits the full observed curve.

The model is not identifiable from one post-midpoint sample. It needs multiple observations spanning the transition, a baseline, and an assumed score bound. Fixing the baseline and bounding accuracy to $[0,1]$ make this implementation suitable for accuracy-like metrics, not unbounded reward, loss, or arbitrary benchmark scores. Its fit quality is only an in-sample check, and no statistical confidence interval is implied. Validation noise, distribution shifts, curriculum changes, optimizer changes, or catastrophic forgetting can invalidate the forecast. Compare projections with later checkpoints and use controlled runs before allocating substantial compute.

## Loop stop policy

The `scaling_law` stop policy turns the projection into a loop decision. Enable it under `stop.policies` together with the monitor:

```yaml
stop:
  policies:
    - type: scaling_law
      min_rounds: 1
      min_margin: 0.01          # stop when ceiling - current <= margin
      max_remaining_compute: 0  # > 0 enables the budget trigger
      budget_margin: 0.05
```

The peak trigger fires when the fitted ceiling adds at most `min_margin` over the latest observed score. The budget trigger fires when p90 needs more than `max_remaining_compute` and the projected gain is at most `budget_margin`. The policy reads the latest `scaling_law_projection` finding from the training record and does nothing when no fit was produced. Because the projection is a single-run forecast, treat the stop as advisory evidence alongside reward, validation, and remaining data.

## Configuration

Enable it as a monitor and map the backend's validation accuracy to `val_score` through `trainer.metric_aliases` when needed:

```yaml
train:
  monitor:
    algorithms:
      - type: scaling_law
        enabled: true
        metric: val_score
        min_points: 7
        curvature_drop: 0.10
        max_rmse: 0.03
        batch_metric: batch_size
        minibatch_metric: mini_batch_size
        rollout_metric: rollout_n

trainer:
  metric_aliases:
    val_score: [validation/accuracy, eval/accuracy]
```

The monitor is opt-in. It accepts finite scores in $[0,1]$. `min_points` is clamped to at least six observations, `curvature_drop` sets the required decline from the largest prior observed slope, and `max_rmse` controls the maximum in-sample score RMSE for reporting a fit. Findings are included with the normal training monitor artifacts.
