# Preserving data mixing ratios after filtering

RL-RSI can enforce a target mixture while retaining as many eligible samples as possible. The mixing strategy supplies relative group weights, the sampler turns those weights into integer quotas, and the JSONL builder checks the training shares after validation holdout. The guarantees apply to sample counts. Token counts and the distribution of optimizer updates also depend on sequence lengths and trainer behavior.

## Integer quota model

For each group $i$ with positive weight $w_i$, the target share is $p_i = w_i / \sum_j w_j$. Groups use each sample's `group`, falling back to its `capability`. The built-in strategies that derive their group set from samples use the groups observed before filtering, so a group can have a positive target and zero surviving capacity. Static weights and history-bandit candidates can explicitly name groups with no generated samples.

At a given difficulty band, let $c_i$ be the number of eligible samples in group $i$, capped by `max_per_group` when positive. Let $m$ be `min_per_group` and $\varepsilon$ be `ratio_tolerance`, measured in absolute share. For an integer total $N > 0$, a group's quota $q_i$ must satisfy the following bounds.

$$
\begin{aligned}
L_i(N) &= \max\bigl(m,\lceil N\max(0,p_i-\varepsilon)\rceil\bigr) \\
U_i(N) &= \min\bigl(c_i,\lfloor N\min(1,p_i+\varepsilon)\rfloor\bigr)
\end{aligned}
$$

A total is feasible exactly when every $L_i(N) \le U_i(N)$ and $\sum_i L_i(N) \le N \le \sum_i U_i(N)$. The bounds are necessary by definition. They are sufficient because assigning every lower bound leaves $N-\sum_i L_i(N)$ samples to allocate, and the groups have at least that many units of integer headroom in aggregate.

## Finding the largest feasible total

Anchoring the calculation on each group's filtered volume captures the right bottleneck in the continuous, exact-ratio case. Each group imposes $N \le c_i / p_i$, so the smallest such bound limits the total. With tolerance, a group with $p_i > \varepsilon$ imposes $N \le \lfloor c_i/(p_i-\varepsilon)\rfloor$. The solver combines these bounds with the sum of capacities and the configured total cap.

The solver checks integer totals in descending order from that upper bound. Rounding may make adjacent totals differ in feasibility, so the first feasible total gives the maximum. With zero tolerance, $N$ must be a multiple of the least common multiple of the reduced denominators of the target shares.

For a 2:1 target with capacities of 10 and 2 and an overall cap of 8, exact shares allow 6 samples with quotas of 4 and 2. With a tolerance of 0.10, quotas of 6 and 2 are feasible at the cap. The actual shares are 75% and 25%, each about 8.33 percentage points from its target.

For a feasible total, the solver starts with the lower bounds. It allocates remaining slots first to groups that can fill them with strict-band samples and then prefers the group furthest below its ideal count $Np_i$. Group names break ties. The first priority minimizes the number of relaxed samples because each allocation uses either one available strict sample or one relaxed sample, and the lower bounds already account for mandatory slots.

For $G$ active groups, $K$ totals checked and $R$ slots allocated above the lower bounds, one quota solve uses $O(KG + R\log(G+1))$ arithmetic and heap operations and $O(G)$ auxiliary space. The solver uses rational arithmetic for the share bounds. Sample ordering and eligibility scans are handled by the sampler.

## Bounded difficulty relaxation

The strict band is the intersection of the global `target_band` and the sample's request-specific band. At step $k$, the sampler moves the lower endpoint down and the upper endpoint up by $k$ times `relax_by`. All candidates come from the pool that passed upstream filters, including any enabled rollout filter.

Eligible capacities can only increase as the band widens. The sampler evaluates progressively wider bands up to `relax_max_steps` and keeps the first band that reaches the largest feasible total. It can stop early when it reaches the total cap. Inside each group it fills the quota from strict samples first.

For example, at an exact 2:1 target, strict capacities of 4 and 1 yield quotas of 2 and 1. If one widening step makes another sample from the smaller group eligible, the quotas increase to 4 and 2. Only one relaxed sample is needed. A group with zero candidates after upstream filtering gains no capacity from widening. If a positive total remains infeasible at every configured step, the sampler returns an empty selection and training skips the dataset.

## Validation holdout

Removing $v_i$ samples from each group changes its training share to $(q_i-v_i)/(N-\sum_j v_j)$. An exact 1:1 selection with two samples per group becomes 1:2 or 2:1 if one sample is held out.

The JSONL builder computes a stratified holdout by capability, keeping at least one training sample per represented capability. It checks the resulting training shares by mixing group. If the split exceeds the share tolerance, the builder decreases the holdout size until its stratified split passes, possibly leaving zero validation rows. This protects the training ratio using the existing stratified allocator. The holdout size is the largest one that passes among the splits that allocator produces for the sizes it checks.

## Configuration and diagnostics

| Option under `data.mixing` | Default config | Effect |
| --- | --- | --- |
| `preserve_ratio` | `true` | Enforce share bounds when selecting quotas. |
| `ratio_tolerance` | `0.10` | Allow 10 percentage points of deviation per group. Set to `0` for exact shares. |
| `min_per_group` | `1` | Require at least one sample from each positive-weight group. |
| `max_per_group` | `0` | Use observed capacity without a separate group cap. |
| `target_total` | `0` | Use the largest feasible total without a separate total cap. |
| `enforce_band` | `true` | Apply the difficulty band during sampling. |
| `relax_by` | `0.15` | Widen each endpoint by this amount per step. |
| `relax_max_steps` | `1` | Allow one widening step. Set to `0` to use the strict band only. |

`pipeline.json` records `ratio_feasible`, `target_ratio`, `actual_ratio`, the selected group counts, and the amount of relaxation used. The dataset manifest includes the sampler stats and `stats.train_group_ratio`, which measures the rows in `train.jsonl` after the holdout.

The quota solver, eligibility sampler, and dataset splitter each have a dedicated module. [Integer quotas](../src/rsi/modules/data_quotas.py) · [Mixing and eligibility](../src/rsi/modules/data_mixing.py) · [Dataset splitting](../src/rsi/modules/data_builder.py)
