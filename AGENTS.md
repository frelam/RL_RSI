# Agent guidance

## README writing

- Write the README for users who need a clear, concise introduction to the project.
- Use plain, direct language and keep explanations brief. Lead with what the project does and how users can get started.
- Group information under clear headings. Use nested lists to show steps, options, and related details.
- Keep the main README focused on the project overview and common usage. Link to focused documentation for implementation details and extended explanations.
- Prefer consistent terminology and concise, scannable examples. Remove repetition and wording that does not help users understand or use the project.
- Keep statements accurate and aligned with the current implementation. Update examples and links when related code or documentation changes.

### Chapter format

Each README chapter covers one loop stage. Follow chapter 2 as the template:

- Start from the previous chapter's output and list the datasource options.
- Name the main procedure in bold (for example, **Auto dataset generation**) and list its steps in execution order.
- Write one short sentence per step. Keep to ideas and flow: leave out config keys, file paths, parameters, and implementation details, and link to the focused doc instead.
- Under each step, use nested short bullets only for branch conditions and fallbacks.
- Give the algorithmic path first, with one short line per algorithmic step. Add an LLM-agent path only where the deterministic algorithm cannot meet the requirement.
- Do not close with a handoff paragraph or section.

### Procedure design

- Make the algorithmic stages explicit and testable: target adjustment, dataset selection with mixing ratios, review and rework. Keep the LLM agent as a fallback, not the default path.
- Tie downstream work to the new dataset and to the capabilities found missing in the previous chapter, including reward selection and enhancement.
- Describe built-in data processing (deduplication, benchmark leakage prevention, replay) as part of materialization, not as separate chapters.
- Keep the strategy opt-in: when disabled, the previous default path stays unchanged.
