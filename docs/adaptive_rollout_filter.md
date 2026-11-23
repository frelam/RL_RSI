# Adaptive rollout data filter

`adaptive_rollout` estimates whether the current policy can solve a verifiable text sample before that sample enters the training dataset. It uses the sample's reference and configured text verifier (`exact`, `contains`, `regex`, or `numeric`) to score independent policy completions. Put it after inexpensive privacy, format, length, deduplication, static verifier, and pattern checks so invalid samples do not consume inference calls. It measures task difficulty on the current policy's own completions, independently of the data generator's answer.

## Decision rule

For each eligible prompt, request 4 completions. If all 4 receive the same binary result, add 4 more and evaluate the cumulative 8. If all 8 remain homogeneous, add 8 and evaluate 16. Stop as soon as both correct and incorrect completions appear. Mixed-result prompts are kept as learning examples. Prompts that remain all-correct or all-wrong after 16 are dropped as too easy or too difficult for the current policy.

An optional all-wrong extension takes the cumulative count to 32. It only runs when all 16 initial completions are incorrect. It requests 8 completions to reach 24, then 8 more only if all 24 remain incorrect. Any correct answer ends the extension. The intermediate 24 check can save 8 completions when the extra batch finds a correct answer. Configure `extend_all_wrong_to_32: true` to enable it.

Samples without a usable textual reference, agentic samples, and verifiers outside the supported text scorers are preserved without inference. Inference errors keep samples by default. Set `on_error: drop` to reject these samples instead. The filter records outcomes, request counts, and per-sample probe evidence in filter statistics and sample metadata. Earlier filters run first to avoid rollout cost on invalid and duplicate samples.

## Probability model

Assume each rollout independently solves a prompt with the same probability `p`, and the verifier returns a correct or incorrect result. The probability of seeing no correct completion in `n` attempts is

```text
P(all wrong | p, n) = (1 - p)^n
```

At `p = 0.1`, the chance that every completion is wrong is 18.53% after 16 attempts and 3.43% after 32. The extension reduces this false-hard probability by 15.10 percentage points, or 81.47% relative to stopping at 16. A prompt with nonzero probability of success can still fail all 32 attempts, so 32 is evidence of difficulty rather than proof of impossibility.

The expected number of completions per eligible prompt for the 4→8→16 policy is

```text
E[rollouts] = 4 + 4(p^4 + (1-p)^4) + 8(p^8 + (1-p)^8)
```

With the all-wrong extension, add the chance-weighted work of reaching 24 and then 32:

```text
E[rollouts with extension] = E[rollouts] + 8(1-p)^16 + 8(1-p)^24
```

At `p = 0.1`, expected work increases from 10.07 to 12.19 completions per prompt, a 21.06% increase. The increase is 2.12 completions on average because only all-wrong prefixes receive the extra sampling. For comparison, at `p = 0.2` it increases from 6.99 to 7.25 completions, or 3.77%. The extension trades extra inference for fewer false-hard classifications. The filter itself is opt-in, and the extension can be disabled in its configuration.

These estimates describe independent Bernoulli samples with constant accuracy. Real completions may be correlated, and verifier mistakes, prompt variation, and sampling temperature can change the observed rates. The probabilities are planning estimates rather than calibration guarantees.

## Configuration

The default configuration includes the filter disabled. Enable it after the deterministic filters for datasets with textual references and an inference endpoint that serves the policy being evaluated. Configure the rollout difficulty grader in data.difficulty.graders to use stored outcomes for difficulty-band sampling. When enabled, the 32-completion all-wrong extension is on by default. Set extend_all_wrong_to_32 to false to cap sampling at the configured final target. Historical replay samples bypass this policy resampling by default. Enable it after the deterministic filters for datasets with textual references and an inference endpoint that serves the policy being evaluated. When enabled, the 32-completion all-wrong extension is on by default. Set `extend_all_wrong_to_32: false` to cap sampling at the configured final target.

```yaml
data:
  filter:
    pipeline:
      - type: adaptive_rollout
        enabled: true
        samples: [4, 8, 16]
        preserve_replay: true
        temperature: 0.7
        max_tokens: 1024
        extend_all_wrong_to_32: true
        on_error: keep
```

`samples` defaults to `[4, 8, 16]` and must contain strictly increasing positive cumulative counts. The 32-completion extension is available when the largest configured count is below 32.

The runtime-generated text verifier is binary, so the filter deliberately classifies only completions scored exactly 1 as correct. This keeps the probability model aligned with the existing exact, contains, regex, and numeric scoring functions.
