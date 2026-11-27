# Dataset selection theory

The `dataset_library` collector picks registered datasets for a data request and splits the request count between them. This document gives the distribution-matching model and the rounding rules that follow. See [the dataset registry and collector](dataset_library.md) for the fields and loaders.

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
| `data.plan.count_per_defect`, `min_count`, `max_count` | `4`, `2`, `6` | Request count before the `task.count` cap. |
| `data.replay.fraction`, `max_samples`, `recent_rounds` | `0.2`, `128`, `1` | Replay target and its caps. |
| `data.replay.strategy` | `target` | `uniform` shuffle or `target` capability allocation. |
