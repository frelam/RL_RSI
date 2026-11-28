# Dataset selection theory

The `dataset_library` collector picks registered datasets for a data request and splits the request count between them. This document gives the distribution-matching model and the rounding rules that follow. See [the dataset registry and collector](dataset_library.md) for the fields and loaders.

## Target distribution adjustment

A task starts with preset ability shares $\pi^{(0)}_a$ (`TaskSpec.ability_priorities`, normalized). Merged chapter-1 findings give deficits $d$ with severity $s_d$. The planner sorts the preset abilities by share and the deficits by severity, then moves mass to each deficit:

- The donor order is the preset share order. A donor keeps at least `min_share`.
- Each deficit receives at most `step`, taken from the largest donor, then the next largest, until the step is filled.
- Deficits do not donate. `max_shift` caps the total moved mass, so a long defect list cannot erase the task profile.

The result is normalized to a distribution. Transfers are recorded for audit. One distribution is produced per round; passing different `step` values produces different target distributions for step 2.

Candidate datasets follow the round chain: round 0 uses `TaskSpec.datasets`, later rounds use the datasets selected in the previous round, and a task without presets uses the whole registry.

## Coverage and target distributions

Let $A$ be the requested abilities, with nonnegative priorities $p_a$, and let dataset $i$ cover ability $a$ with $c_{ia} \in [0,1]$. The target share is $\pi_a = p_a / \sum_b p_b$. Coverage scores are not observed row frequencies, so the selector explicitly estimates each dataset's composition by normalizing its scores over all registered abilities:

$$
r_{ia} = \frac{c_{ia}}{\sum_b c_{ib}}.
$$

The catalog may contain thousands of abilities while a request names only a few. The optimizer therefore keeps only the requested columns and aggregates every unrequested ability into one `__other__` column, with $r_{i,other}=1-\sum_{a\in A}r_{ia}$ and $\pi_{other}=0$. Individual unspecified abilities remain unconstrained, but their aggregate excess is penalized because it displaces the requested distribution. This aggregation keeps the model at $|A|+1$ ability columns rather than expanding the full catalog.

The estimate can be replaced by sample-level ability statistics without changing the optimization model. When a plan contains several dataset requests, the collector solves one joint selection over the plan's `target_distribution` and reuses those ratios per request, keeping only the selected sources that cover that request's capability. A single dataset request falls back to its own capability priority.

## Two-stage distribution optimization

For a required sample total $N$, let $x_i \ge 0$ be the planned rows from dataset $i$, with $\sum_i x_i = N$. Using exactly $N$ rows is sufficient because a larger solution can be scaled back to $N$ without changing its distribution. The resulting ability mass is $y_a = \sum_i r_{ia}x_i$. Nonnegative variables $u_a$ and $o_a$ measure shortfall and excess:

$$
y_a + u_a - o_a = N\pi_a.
$$

An explicitly requested ability is long-tail when $\pi_a \le 0.1$. Its relative shortfall scale is $s_a = \max(N\pi_a, 1)$; `__other__` is never long-tail and uses $s_{other}=N$. The first LP/MILP minimizes $z$ subject to $u_a/s_a \le z$ for every long-tail ability. This protects the worst long-tail ability before considering the aggregate fit.

The second stage preserves the optimal $z$ up to a $10^{-6}$ numerical tolerance and minimizes

$$
\sum_a \frac{\alpha_a u_a + \beta_a o_a}{s_a}
+ 0.01\sum_i \frac{(1-q_i)x_i}{N},
$$

where $q_i$ is dataset readiness. By default $\alpha_a=4$ and $\beta_a=1$; for long-tail abilities $\alpha_a=32$. Thus covering a missing long-tail ability is worth substantially more than avoiding excess head data. A small readiness cost resolves otherwise equivalent mixtures.

`max_datasets` adds a binary variable for each candidate, links its sample variable to that choice, and bounds the number of selected sources. Continuous proportions use LP; this cardinality constraint makes the problem a MILP. Both are solved by HiGHS through `scipy.optimize.milp`.

A target is exactly feasible without a dataset limit only when its distribution lies in the convex hull of the available dataset composition vectors. Otherwise the optimizer returns the closest solution under the asymmetric loss and reports the remaining shortfall rather than silently changing the target.

## Diversity and the strategy objective

The chapter-2 strategy uses the same composition model with one addition. Let $O$ be the catalog abilities outside the target that at least one candidate covers with a declared score of at least `diversity_threshold`. For each $a\in O$, a binary $d_a$ is 1 when the mixture gives it at least `diversity_floor` of the batch:

$$
y_a \ge \text{diversity\_floor}\cdot N d_a .
$$

The second-stage objective becomes

$$
\text{distribution error} + \gamma\left(1 - \frac{1}{|O|}\sum_{a\in O}d_a\right),
$$

where the distribution error is the asymmetric under/over-coverage term from the previous section and $\gamma$ is `diversity_weight`. $\gamma=0$ removes the diversity term. The aggregate `__other__` excess is still penalized, so diversity trades against covering target abilities rather than overwhelming them.

## Review and rework

After the optimization, the review computes

$$
\text{gap}=\sum_a\max(0,\pi_a-\hat\pi_a),
\qquad
\text{longtail-gap}=\sum_{a:\pi_a\le\theta}\max(0,\pi_a-\hat\pi_a),
$$

where $\hat\pi_a$ is the mixture's estimated share and $\theta$ is `longtail_threshold`. The mix is accepted when both are within `gap_tolerance` and `longtail_gap_tolerance`. Long-tail abilities also receive the larger under-coverage weight in the optimizer.

Otherwise the reviewer adds registry datasets that cover the missing abilities. For an unused dataset $i$ with composition $r_{ia}$,

$$
\text{fill}_i=\frac{\sum_{a\in D}r_{ia}\delta_a}{\sum_{a\in D}\delta_a},
\qquad
\text{impact}_i=\frac{\sum_{a\notin D}r_{ia}\pi_a}{\sum_{a\notin D}\pi_a},
\qquad
\text{score}_i=\text{fill}_i-w_{\text{impact}}\,\text{impact}_i,
$$

where $\delta_a$ is the remaining deficit and $D$ the deficient set. The best `add_per_round` datasets join the candidate set, and the optimizer runs again, up to `max_rework` times.

When no unused dataset covers a remaining deficit, the reviewer records the deficit and lets the materializer fill it. If `data.strategy.review.synthesize` is on, the configured collector (default `text`) generates at most `synthesis_max` rows for the deficient abilities, and those rows join the dataset mix. The round total is kept at the strategy count by dropping over-covered dataset rows first. If synthesis is off or yields no rows, the round is marked rejected and no dataset is materialized, so a mix that cannot reach its target is not trained on silently.

## Integer quotas

The request count $N$ is split by largest-remainder rounding: each dataset gets $\lfloor N r_i \rfloor$ samples, and the remaining units go to the datasets with the largest fractional parts, breaking ties by name. The result satisfies

- $\sum_i n_i = N$ exactly.
- $|n_i - N r_i| < 1$ for every dataset, so no dataset receives less than its floor or more than its ceiling.

This is Hamilton's apportionment method. It is house-monotone, meaning that a larger $N$ never lowers a dataset's quota. It is not ratio-monotone: a small change in the ratios can move a unit between two datasets neither of whose ratio rose. With at most four sources the effect is at most one sample.

A dataset that fails to load or has fewer usable rows than its quota leaves a shortfall. The collector fills the shortfall with the rows the other selected datasets have left, in selection order, and still returns fewer than $N$ samples only when the selected datasets hold too few usable rows in total. The fill keeps each quota within $|n_i - N r_i| < 1$ when the quotas can be met, and moves the actual shares away from the ratios only when they cannot. Each sample's `dataset_selection.source_counts` records the resulting per-source counts.

## Request counts in `plan_task`

`plan_task` turns each capability of a `TaskSpec` into one dataset request with `kind: dataset`. The count of a request follows the rule planner:

$$
\text{count} = \mathrm{round}\bigl(\texttt{count\_per\_defect} \cdot (0.6 + s) \cdot (0.75 + 0.5\pi)\bigr)
$$

where $s$ is the task priority of the capability and $\pi$ is its catalog priority, clamped to $[\texttt{min\_count}, \texttt{max\_count}]$. With the default config the unclamped range is $1.8$ to $8.0$ and the clamp to $[2, 6]$ applies at both ends. When the total exceeds `task.count`, the plan trims requests in the order of `ability_priorities`: earlier requests keep their count and later ones are reduced or dropped.

## Replay share

`data.replay` mixes rows from the `train.jsonl` of recent rounds into the new samples. For $n$ new samples and fraction $f$, it draws

$$
\text{target} = \mathrm{round}\Bigl(\frac{n f}{1-f}\Bigr)
$$

rows, capped by `max_samples` and by the rows found in the last `recent_rounds` datasets. Without the cap, replayed rows make up exactly $f$ of the combined set: with $f = 0.2$, one replayed row is added for every four new ones. With $n f/(1-f) > \texttt{max\_samples}$ the share is below $f$ and shrinks as $n$ grows. $f$ must be in $[0, 1)$. Replay rows are deduplicated among themselves by prompt.

`data.replay.strategy` chooses the rows. `uniform` shuffles all candidate rows and takes the target; `target` apportions the replay target across capabilities by the current target distribution, then fills any shortage from the remaining rows. The per-capability targets and the selected counts are recorded in `pipeline.json` and the dataset statistics.

## Configuration

| Option | Default config | Role in the model |
| --- | --- | --- |
| `priority` (per dataset) | `0.3` to `0.8` | Readiness $q_i$, used as a small mixture tie-breaking cost. |
| `abilities` (per dataset) | `0.3` to `1.0` | Coverage scores $c_{ia}$, normalized into estimated composition. |
| `max_datasets` | `4` | Maximum number of sources in the MILP. |
| `data.strategy.target.step`, `min_share`, `max_shift` | `0.05`, `0.01`, `0.35` | Target adjustment: share per deficit, donor floor, total moved mass. |
| `data.strategy.optimize.diversity_weight`, `diversity_threshold`, `diversity_floor` | `0.1`, `0.05`, `0.01` | Diversity trade-off $\gamma$, coverage needed to count an ability, minimum batch share per covered ability. |
| `data.strategy.review.gap_tolerance`, `longtail_gap_tolerance` | `0.05`, `0.01` | Accepted target shortfall, ordinary and long-tail. |
| `data.strategy.review.max_rework`, `add_per_round`, `impact_weight` | `2`, `2`, `1.0` | Add-and-reoptimize cycles, datasets added per cycle, collateral-coverage penalty. |
| `data.strategy.review.synthesize`, `collector`, `synthesis_max` | `true`, `text`, `128` | LLM fallback for uncovered deficits, its collector, and its row cap. |
| `data.plan.count_per_defect`, `min_count`, `max_count` | `4`, `2`, `6` | Request count before the `task.count` cap. |
| `data.replay.fraction`, `max_samples`, `recent_rounds` | `0.2`, `128`, `1` | Replay target and its caps. |
| `data.replay.strategy` | `target` | `uniform` shuffle or `target` capability allocation. |
