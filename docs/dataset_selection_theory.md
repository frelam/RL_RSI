# Dataset selection theory

The `dataset_library` collector picks registered datasets for a data request and splits the request count between them. This document gives the coverage model behind the selection, the guarantee of the greedy search, and the rounding rules that follow. See [the dataset registry and collector](dataset_library.md) for the fields and loaders.

## Coverage model

Let $A$ be the abilities a request needs, with priority $p_a \ge 0$ for ability $a$. Let dataset $i$ cover ability $a$ with $c_{ia} \in [0,1]$ (`abilities`) and have readiness $q_i \in [0,1]$ (`priority`). The collector is called once per request, and a request carries one capability, so in practice $|A| = 1$ and $p_a$ is the catalog priority of that capability.

Treating $c_{ia}$ as the probability that dataset $i$ trains ability $a$, the priority-weighted coverage of a set $S$ of datasets is

$$
F(S) = \sum_{a \in A} p_a \Bigl(1 - \prod_{i \in S}(1 - c_{ia})\Bigr)
$$

$F$ is monotone and submodular. Its marginal gain from adding dataset $i$ is

$$
\Delta_i(S) = \sum_{a \in A} p_a \prod_{j \in S}(1 - c_{ja})\, c_{ia}
$$

The remaining priority that the selector keeps after each pick is $p_a \prod_{j \in S}(1 - c_{ja})$, so the `gain` it computes for a candidate is exactly this marginal gain. An ability is reported as `uncovered` when its remaining priority is still above $10^{-6}$ when the search stops.

## Greedy selection guarantee

The selector repeatedly takes the dataset with the largest $q_i \Delta_i(S)$, for at most $k$ = `max_datasets` picks. With equal readiness this is the standard greedy algorithm for monotone submodular maximization under a cardinality constraint, which returns

$$
F(S_k) \ge \Bigl(1 - \bigl(1 - \tfrac{1}{k}\bigr)^k\Bigr) F(\mathrm{OPT}_k)
$$

For $k = 4$ the factor is $0.6836$, and it approaches $1 - 1/e \approx 0.6321$ from above as $k$ grows.

Readiness scales the score of each candidate, so a dataset with low readiness can lose to one with a smaller gain. Writing $\rho = \min_i q_i / \max_i q_i$ over the candidates, the chosen gain is at least $\rho$ times the best available gain, and the same telescoping argument gives

$$
F(S_k) \ge \Bigl(1 - \bigl(1 - \tfrac{\rho}{k}\bigr)^k\Bigr) F(\mathrm{OPT}_k)
$$

The bound degrades as the readiness spread grows. With the default registry values $q \in \{0.3, 0.7, 0.8\}$ and $k = 4$, $\rho = 0.375$ and the factor is $0.3255$. This is a worst-case bound on the coverage objective only. The readiness term is a deliberate preference for rows that are ready to use, and the trade-off is intended when the readiness values differ.

When the search stops because no candidate has a positive gain, every ability is covered as far as the available datasets allow, and $F(S) = F(\mathrm{OPT})$.

## Sample ratios

The ratio of dataset $i$ is its score at the time it was picked, divided by the total score:

$$
r_i = \frac{q_i \Delta_i(S_{i-1})}{\sum_j q_j \Delta_j(S_{j-1})}
$$

The ratios sum to one. With equal readiness the unweighted gains telescope, $\sum_i \Delta_i(S_{i-1}) = F(S)$, so $r_i$ is the share of the covered priority that dataset $i$ contributed. Unequal readiness moves share toward datasets with more ready rows.

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

## Configuration

| Option | Default config | Role in the model |
| --- | --- | --- |
| `priority` (per dataset) | `0.3` to `0.8` | Readiness $q_i$. Equal values give the $1 - (1-1/k)^k$ guarantee. |
| `abilities` (per dataset) | `0.3` to `1.0` | Coverage $c_{ia}$. |
| `max_datasets` | `4` | Number of picks $k$. |
| `data.plan.count_per_defect`, `min_count`, `max_count` | `4`, `2`, `6` | Request count before the `task.count` cap. |
| `data.replay.fraction`, `max_samples`, `recent_rounds` | `0.2`, `128`, `1` | Replay target and its caps. |
