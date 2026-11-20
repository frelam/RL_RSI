---
name: env-synthesis
description: Design small, offline, verifiable agentic tasks from a capability request.
version: 1
---

# Environment Synthesis Skill

You are invoked by the `agentic_synthesis` data collector through a harness.
Your job is to turn one capability request into a handful of small, runnable,
verifiable tasks.

## Inputs

A task JSON file is provided by the harness (`{task_file}` read-only). It
contains:

- `instruction`: the rendered request, including capability, seeds, count and
  difficulty band;
- `metadata`: round, request id, checkpoint id;
- `output_schema`: the JSON schema your output must satisfy;
- `skills`: the skill names mounted under `{skills_dir}`.

## Procedure

1. Read the request and identify the underlying sub-capability (not just the
   topic).
2. Design tasks with a single clear success condition.
3. Keep workspaces self-contained: only relative paths, no network, no writes
   outside `{work_dir}`.
4. Include a verifier description. Prefer deterministic checks (tests, exact
   outputs, assertions) over subjective ones.
5. Estimate difficulty in `[0, 1]`; avoid duplicating the same task shape.
6. Write the result to `{output_file}` as JSON matching `output_schema`.

## Output contract

```json
{
  "tasks": [
    {
      "name": "short-id",
      "instruction": "...",
      "files": {"main.py": "...", "test_main.py": "..."},
      "verifier": {"type": "harness", "name": "tests_pass"},
      "difficulty": 0.55
    }
  ]
}
```

## Guardrails

- If you cannot produce a verifiable task, return fewer tasks rather than
  inventing a weak verifier.
- Never embed secrets, credentials, network calls or destructive commands.
- Never write outside the declared work directory.
- The orchestrator validates your output against the schema; invalid output is
  discarded, not repaired.
