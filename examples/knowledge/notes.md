# RSI Loop Notes

The loop is code-driven. Deterministic code owns orchestration, checkpoint
promotion and stopping. Harnesses are bounded tools: they receive a workspace
and return schema-validated JSON.

## Observations

- Long-context retrieval failures concentrate on multi-hop evidence, where the
  answer requires combining two distant passages.
- Code debugging samples are useful only when a failing test or traceback is
  included; otherwise the task is under-specified.
- Reward hacking often shows up as longer answers that repeat the format
  constraint without adding evidence.

## Policy

A checkpoint is promoted only after validation against the frozen benchmark,
and only when the improvement exceeds the configured minimum delta.
