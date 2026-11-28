# Dataset registry and the dataset library collector

The dataset library loads curated rows from registered datasets instead of synthesizing them. It serves every data request whose kind is `dataset`, for example a defect whose catalog entry has `kind: dataset`. This document covers the registry, the loaders, the source selection, and the task entry points. See [dataset selection theory](dataset_selection_theory.md) for the coverage model and the guarantee of the selection.

## Registry

`configs/datasets.yaml` holds one entry per dataset, open-source or self-defined. `path` is resolved relative to the repository root.

| Field | Effect |
| --- | --- |
| `url` | Where the raw data is downloaded from. For the `huggingface` loader it is the repository id. |
| `path` | Local rows, already preprocessed. |
| `loader` | A built-in name (`jsonl`, `parquet`, `web`, `huggingface`, `synthetic`) or a registered loader name. |
| `abilities` | Capability to how well the dataset covers it (0 to 1). |
| `priority` | 0 to 1, default 0.5. How ready the local rows are to be used. |
| `reward` | `{type, name, weight}`. `name` is the scorer that scores a response. |
| `reward_code` | Where that reward is implemented (`path::function`). Documentation only. |
| `preprocess` | A list of load-time steps, or `{pipeline, steps}` where `pipeline` names the script that turns the download into `path`. |
| `license`, `source` | Optional. `source` keeps only the rows of this dataset when `path` mixes several. |
| `config`, `split` | Optional, for the `huggingface` loader. |
| `rows` | Optional, inline rows for the `synthetic` loader. |

An entry with local rows is active. A `jsonl` or `parquet` entry without `path` is declared only and is skipped when datasets are selected.

## Loaders

| `loader` | Reads rows from | Requires |
| --- | --- | --- |
| `jsonl` | The file at `path`. | Nothing. |
| `parquet` | The file at `path`. | Optional `pyarrow` package. |
| `web` | An HTTP(S) `url` that returns JSON or JSONL. It does not crawl. | Network access. |
| `huggingface` | `load_dataset(url, config, split)`. | Optional `datasets` package. |
| `synthetic` | The inline `rows`. | Nothing. |

verl-style rows (verl `examples/reasoning_rl`, verl-toolRL `agentic_rl_data`) are mapped to the common row shape when loaded, by `src/rsi/adapters/verl_*.py`.

### Custom loaders

Define a retrieval adapter without editing the loader switch:

```python
from rsi.modules.datasets import register_dataset_loader

@register_dataset_loader("internal_api")
def load_internal(spec, root):
    return fetch_rows_from_internal_api(spec.url)
```

A registry entry can then use `loader: internal_api`. The value may also be a dotted callable (`my_package.loaders:load_internal`), which is imported on first use. A loader receives `(DatasetSpec, root)` and returns row mappings.

### Load-time preprocessing

`preprocess.steps` run on each row when it is loaded:

- `normalize_chat` fills `prompt` from the first user message, then from `question`, and `response` from `answer` or `solution`.
- `strip_text` strips whitespace from the text fields.
- `drop_empty` drops rows without a prompt.
- `deduplicate` is accepted and does nothing. Deduplication happens when rows become samples.

## Source selection

When `data.strategy.enabled` is false, `DataPlanner` builds the round's target distribution from the merged analyzer priorities (with a defect severity fallback), a floor for `pending_capabilities`, and an optional blend with the previous round's distribution. The result is written to `plan.metadata["target_distribution"]` and to `pipeline.json`. `data.plan.target_distribution` controls the blend weight and the pending floor; setting `enabled: false` skips the pre-processing.

When the chapter-2 strategy is enabled, the target distribution comes from the task preset plus the defect adjustment instead. The strategy selects datasets and ratios across the whole round, and the pipeline materializes that mix through the collector's `load_dataset` before the normal filter, mixing and builder stages. If the review cannot close a deficit and `data.strategy.review.synthesize` is on, the configured collector generates the missing rows; otherwise the round materializes no dataset. See [the data strategy](dataset_selection_theory.md#target-distribution-adjustment).

The selector normalizes the requested ability priorities into a target distribution and each dataset's coverage scores into an estimated composition. When the plan has more than one dataset request, the collector runs one joint selection over the plan's `target_distribution`, then keeps the selected sources that cover each request's capability and renormalizes their ratios for that request. A single dataset request uses its own capability priority. Thousands of unrequested abilities are aggregated into one `__other__` dimension: their individual shares are free, but their aggregate excess is penalized. It then solves a two-stage LP/MILP for the per-source sample ratios:

1. Minimize the worst relative shortfall among long-tail abilities (target share at most 0.1).
2. Subject to that long-tail bound, minimize relative distribution error. Under-coverage costs 4 times over-coverage, and long-tail under-coverage costs another 8 times as much. Dataset readiness is a small tie-breaking cost.

The request count is the required sample total. `max_datasets` is an independent upper bound and introduces binary dataset-selection variables. Abilities with no coverage in the resulting mixture are reported as `uncovered`; the selection stats also record target and actual distributions, shortfalls, excess, and the long-tail diagnostic.

## From rows to samples

- Each dataset gets an integer quota from the request count and the ratios, using largest-remainder rounding.
- Rows are filtered by `source`, preprocessed, and deduplicated by a hash of prompt and response. A row whose capability the dataset does not cover is dropped.
- The `benchmark_leakage` filter drops samples that match a configured benchmark prompt, by normalized hash or n-gram similarity. It reads explicit `paths` and, by default, every file-backed entry under `benchmark.benchmarks`.
- A dataset that fails to load is logged and skipped. The other sources still serve the request.
- When the selected datasets hand out fewer samples than the quota asks for, the gap is filled from the rows they have left, in selection order.
- Each sample records `dataset`, `url`, `license`, and `reward_hint` in its metadata, plus a `dataset_selection` summary with the ratios, uncovered abilities, and per-source counts.

In the loop, these samples then pass through the existing filter, difficulty, mixing, and builder stages like any other sample.

## Reward selection and enhancement

The builder writes `reward_spec.json` next to the rows. For each capability, the rule merges the builder default, the capability catalog reward, the dataset `reward` hint, and `data.build.reward.by_capability`. The chapter-2 strategy therefore selects built-in rewards from the datasets it materializes.

- `data.build.reward.defect_boost` above 1 raises the rule weight for capabilities present in the chapter-1 defect report, scaled by `severity x confidence`.
- With `data.build.reward.codegen.enabled`, an LLM agent writes `<dataset>/rewards/<capability>.py` for up to `max_capabilities` defects. The rule records `code_path`, `code_entrypoint`, and `code_language`. The module is persisted for the trainer; the builder does not execute it.

## Task entry points

`TaskSpec(id, ability_priorities, count=32, notes="", datasets=[])` describes a task by the capabilities it needs and their priorities (0 to 1), plus the datasets that seed the first round of the data strategy.

- `DataPipeline.plan_task(task, round_no=0)` returns a `DataPlan` with one `dataset` request per capability. The request count scales with the task priority and the catalog priority, between `data.plan.min_count` and `max_count`, and the total is capped at `task.count`. The strategy is `task_requirements`.
- `DataPipeline.collect_task(task, round_no=0)` plans the task and loads its rows through the dataset library. It returns up to `task.count` samples and does not run the filter, difficulty, mixing, or builder stages. It requires a collector with kind `dataset`.
- With `data.strategy.enabled: true`, the loop uses `task.datasets` as the first-round candidate set, adjusts `task.ability_priorities` with the chapter-1 defects, and materializes the strategy's selected datasets and ratios. Later rounds reuse the previous round's selected datasets as candidates.

## Configuration

| Option under the collector entry in `data.collectors` | Default config | Effect |
| --- | --- | --- |
| `type` | `dataset_library` | Registers the collector for requests with kind `dataset`. |
| `catalog` | `configs/datasets.yaml` | Registry file. |
| `max_datasets` | `4` | Maximum number of sources selected per request. |
| `data.strategy.enabled` | `false` | Use the task strategy instead of defect requests. |
| `data.strategy.catalog` | `configs/datasets.yaml` | Registry the strategy selects from. |
| `data.strategy.review.synthesize`, `collector`, `synthesis_max` | `true`, `text`, `128` | LLM fallback for uncovered deficits. |
| `data.build.reward.defect_boost` | `1.0` | Weight multiplier for chapter-1 defects. |
| `data.build.reward.codegen.enabled`, `prompt`, `max_capabilities` | `false`, `reward_code`, `2` | LLM-written reward modules per defect. |
| `benchmark_leakage.paths`, `benchmarks`, `threshold` | `[]`, `[]`, `0.9` | Extra benchmark sources, benchmark names, and n-gram similarity limit. |
