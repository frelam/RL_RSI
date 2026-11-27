# Capability analysis and validation theory

The analysis and validation stages decide what the model lacks and whether a candidate checkpoint is promoted. Both decisions read noisy benchmark scores. This document gives the statistical reading of the rules and the error rates that follow from them. See [capability analysis](capability_analysis.md) for the inputs, outputs, and configuration.

## Reference rule as a one-sided test

The `missing_capability` analyzer judges a benchmark or subset by

$$
\text{good} \iff s \ge \mu - t, \qquad t = \max(\texttt{margin},\ \texttt{sigma} \cdot \sigma_{\text{ref}})
$$

where $s$ is the observed score and $(\mu, \sigma_{\text{ref}})$ is the reference at the model size.

Assume the model has the average ability of its size class, so that $\mathbb{E}[s] = \mu$. The observed score deviates from $\mu$ by the spread between models and by sampling noise:

$$
s \sim \mathcal{N}(\mu, \tau^2), \qquad \tau^2 = \sigma_{\text{ref}}^2 + \mathrm{SE}^2, \qquad \mathrm{SE} \le \frac{0.5}{\sqrt{n}}
$$

The bound on the standard error holds for any score in $[0,1]$ averaged over $n$ items: it is $0.05$ for $n = 100$, $0.0158$ for $n = 1000$, and $0.0071$ for $n = 5000$. Under this model the probability that an average model passes is

$$
\Pr[\text{good}] = \Phi\Bigl(\frac{t}{\tau}\Bigr)
$$

so the rule is a one-sided test with significance level $1 - \Phi(t/\tau)$. Two cases make the choice of tolerance concrete:

- When `sigma` dominates, $\Pr[\text{good}] = \Phi(\texttt{sigma})$ whatever the reference spread. `sigma` of $0.5$, $1.0$, $1.645$, and $2.0$ pass an average model with probability $0.6915$, $0.8413$, $0.95$, and $0.9772$. Set `sigma` to the normal quantile of the desired pass rate.
- When `margin` dominates, $\Pr[\text{good}] = \Phi(\texttt{margin}/\tau)$. A margin fixed in score units passes more easily on benchmarks with a small spread between models.

For a model whose true score is $d$ below the reference mean, the pass probability drops to $\Phi((t - d)/\tau)$. The reference standard deviation is computed from a few self-reported models per size class, so $\sigma_{\text{ref}}$ is itself an estimate. The probabilities are planning estimates, not calibration guarantees.

## Error structure of the missing list

Let $J$ be the judged benchmarks and subsets, $G \subseteq J$ the well-performing ones, and $M(b)$ the capabilities that $b$ covers in the registry. The analyzer marks and reports

$$
\text{marked} = \bigcup_{b \in G} M(b), \qquad \text{missing} = \text{catalog} \setminus \text{marked}
$$

Three properties follow.

- **Union rule.** One well-performing benchmark marks a capability as present, even when other benchmarks that cover it perform weakly. The rule is optimistic about the model.
- **Two causes of a missing capability.** The evidence of a defect separates them. A capability covered only by weak benchmarks or subsets is a signal about the model. A capability covered by no judged benchmark is a coverage gap: it is reported whenever no collected, registered, referenced benchmark measures it, whatever the model does. Read the second class as unmeasured, and close it by adding a benchmark or a reference curve.
- **Skipped benchmarks do not mark.** With `unreferenced: skip`, a benchmark without a reference curve cannot mark its capabilities, which enlarges the second class. When every benchmark is skipped, the analyzer reports no defects rather than the whole catalog.

The severity of a missing capability is

$$
\text{severity} = \min(0.98,\ \texttt{severity} \cdot \pi_a)
$$

with $\pi_a$ the catalog priority, so it is linear in the priority and bounded by `severity` ($0.6$ by default).

## Merging of analyzer findings

The merger groups defects by capability and subcapability. For findings from $m$ sources with severity $s_i$, confidence $c_i$, and source weight $w_i$:

$$
\text{severity} = \min\Bigl(0.99,\ \max_i w_i s_i + 0.08\,(m - 1)\Bigr), \qquad
\text{confidence} = \min\Bigl(0.99,\ 1 - \prod_i \bigl(1 - \min(0.98, w_i c_i)\bigr)\Bigr)
$$

- Severity is the strongest weighted signal plus a fixed corroboration bonus of $0.08$ per additional source.
- Confidence is a noisy-OR. It is monotone in every input, never exceeds the sum of the inputs, and no single source can raise it above $1 - (1 - 0.98) = 0.98$. A source weight above one multiplies a source's confidence before the cap.
- Noisy-OR assumes conditionally independent sources. The rule and `missing_capability` analyzers both read benchmark scores, so their agreement raises confidence more than independent evidence would.

A merged defect is dropped when its severity is below `analyze.merge.min_severity` ($0.05$) or its confidence is below `min_confidence` ($0.2$). With weight $1.0$ and `severity` $0.6$, a missing capability therefore survives only if its catalog priority is at least $0.05 / 0.6 \approx 0.083$.

## Planning order

The default curriculum orders defects by

$$
\text{score} = \text{severity} \cdot \text{confidence} \cdot (1 + \pi_a)
$$

and keeps one defect per capability, up to `max_capabilities`. Capabilities carried from the previous round sort before all others. The factor $1 + \pi_a$ lies in $[1, 2]$, so the catalog priority can at most double the rank score of an equally severe and confident defect. with the change strictly positive. A carried capability is planned again only when the current defect report still lists it with a severity of at least `data.plan.min_severity`.

## Validation as a joint test

Let $d_c$ be the change in the mean item score of capability $c$, candidate minus parent, with $T$ targets and $O$ other measured capabilities. and the target change is strictly positive. Benchmark validation accepts the candidate when

$$
\max_{c \in \text{targets}} d_c \ge \texttt{target\_min\_delta} \quad\wedge\quad
\min_{c \in \text{others}} d_c \ge -\texttt{max\_other\_regression} \quad\wedge\quad
\delta \ge \texttt{min\_delta}
$$

and the two reports cover the same items without errors.

Consider a candidate that is equal to its parent, so every $d_c$ is pure measurement noise. Let $\alpha = \Pr[d_c \ge \texttt{target\_min\_delta}]$ and $\beta = \Pr[d_c < -\texttt{max\_other\_regression}]$ under that null. If the capabilities are independent, the probability of a false acceptance is

$$
\Pr[\text{accept} \mid \text{null}] = \bigl(1 - (1 - \alpha)^T\bigr)(1 - \beta)^O \,\Pr[\delta \ge \texttt{min\_delta} \mid \text{null}]
$$

The two capability conditions pull in opposite directions. More targets make it easier for one of them to clear `target_min_delta` by chance. More measured capabilities make it likelier that one of them shows a spurious regression.

### Sizing the regression threshold

For binary item scores and $n$ items per capability on each side, the unpaired standard deviation of $d_c$ is

$$
\mathrm{SD}[d_c] = \sqrt{\frac{p(1-p)}{n} + \frac{p'(1-p')}{n}} \le \frac{0.707}{\sqrt{n}}
$$

With the default `max_other_regression` of $0.01$:

| Items per capability $n$ | $\mathrm{SD}[d_c]$ (upper bound) | $\beta$ per capability | $(1-\beta)^5$ |
| --- | --- | --- | --- |
| 100 | 0.0707 | 0.4438 | 0.0532 |
| 1000 | 0.0224 | 0.3274 | 0.1377 |
| 2500 | 0.0141 | 0.2398 | 0.2540 |
| 10000 | 0.0071 | 0.0786 | 0.6639 |

The default threshold is below one standard deviation of noise until a capability has about $10{,}000$ items, so an unchanged candidate is likely to fail the regression check on a small benchmark. Validation compares the same items on both sides, and the paired deviation $\sqrt{\mathrm{Var}[X - Y]/n}$ is smaller when the two models agree on most items. The table is an upper bound.

The rule does not weight a capability by its item count, and the overall metric is the mean over all items, so a large benchmark dominates `min_delta`. To size the thresholds from data, run validation twice on one checkpoint to measure $\mathrm{SD}[d_c]$ and set `max_other_regression` and `target_min_delta` to a multiple of it. Two standard deviations give a $2.3\%$ rate per capability under normal noise.

### Carried targets

After a benchmark validation, the loop updates the carried set:

$$
\text{pending}_{r+1} = (\text{pending}_r \setminus \text{improved}) \cup \text{failed\_targets} \cup \text{unmeasured\_targets}
$$

A capability leaves the set only when it is a target that improves. Because the curriculum plans carried capabilities first, a target that fails or cannot be measured is retried before new defects, and it cannot be silently dropped by a successful round on other capabilities.
