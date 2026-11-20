---
name: badcase-deepdive
description: Investigate worst-case benchmark items and report evidence-backed defects.
version: 1
---

# Badcase Deep-Dive Skill

This is an advisory analysis skill. The deterministic loop already has rule and
LLM-judge analyzers; this skill exists for cases where an agent needs to inspect
trajectories, files or tool logs before proposing a capability defect.

## Procedure

1. Read the worst cases and capability table from the instruction.
2. Cluster failures by mechanism, not by surface wording. Examples: evidence
   not retrieved, arithmetic slip, constraint dropped, tool call malformed.
3. For each cluster, name the capability that should be trained and estimate
   severity and confidence.
4. Cite the specific case IDs/scores that support the claim.
5. Emit at most five defects. Prefer one strong defect over many weak ones.

## Output

Write JSON to the harness output file:

```json
{
  "defects": [
    {
      "capability": "long_context.retrieval",
      "subcapability": "multi_hop",
      "severity": 0.6,
      "confidence": 0.75,
      "description": "Fails to combine evidence from passages 2 and 5.",
      "evidence": ["case-17 score=0.0", "case-22 score=0.25"],
      "suggested_data": {"kind": "text", "seeds": ["multi-hop retrieval"]}
    }
  ]
}
```

## Guardrails

- Do not modify any files except the declared output file.
- Do not invent item IDs or scores.
- The output is schema-validated and merged with lower-weight advisory status;
  it cannot on its own promote a checkpoint.
