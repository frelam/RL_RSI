# Dataset registry and the dataset library collector

The dataset library loads curated rows from registered datasets instead of synthesizing them. It serves every data request whose kind is `dataset`, for example a defect whose catalog entry has `kind: dataset`. This document covers the registry, the loaders, the source selection, and the task entry points. See [dataset selection theory](dataset_selection_theory.md) for the coverage model and the guarantee of the selection.

## Registry

`configs/datasets.yaml` holds one entry per dataset, open-source or self-defined. `path` is resolved relative to the repository root.

| Field | Effect |
| --- | --- |
| `url` | Where the raw data is downloaded from. For the `huggingface` loader it is the repository id. |
| `path` | Local rows, already preprocessed. |
| `loader` | `jsonl` (default), `parquet`, `web`, `huggingface`, or `synthetic`. |
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

### Load-time preprocessing

`preprocess.steps` run on each row when it is loaded:

- `normalize_chat` fills `prompt` from the first user message, then from `question`, and `response` from `answer` or `solution`.
- `strip_text` strips whitespace from the text fields.
- `drop_empty` drops rows without a prompt.
- `deduplicate` is accepted and does nothing. Deduplication happens when rows become samples.

## Source selection

For a request, the `dataset_library` collector selects at most `max_datasets` active datasets with a greedy weighted-coverage search:

1. Each ability has a remaining priority. It starts at the catalog priority of the requested capability.
2. Each candidate's gain is the sum of `remaining priority * coverage` over the remaining abilities. The candidate with the largest `gain * priority` wins.
3. The winner is selected, and each remaining priority is scaled down by `1 - coverage`.
4. The search stops when no candidate has a positive gain, or the dataset limit is reached.

The per-source sample ratio is the winner's contribution divided by the total contribution. Abilities that keep a positive remaining priority are reported as `uncovered`.

## From rows to samples

- Each dataset gets an integer quota from the request count and the ratios, using largest-remainder rounding.
- Rows are filtered by `source`, preprocessed, and deduplicated by a hash of prompt and response. A row whose capability the dataset does not cover is dropped.
- A dataset that fails to load is logged and skipped. The other sources still serve the request.
- When the selected datasets hand out fewer samples than the quota asks for, the gap is filled from the rows they have left, in selection order.
- Each sample records `dataset`, `url`, `license`, and `reward_hint` in its metadata, plus a `dataset_selection` summary with the ratios, uncovered abilities, and per-source counts.

In the loop, these samples then pass through the existing filter, difficulty, mixing, and builder stages like any other sample.

## Task entry points

`TaskSpec(id, ability_priorities, count=32, notes="")` describes a task by the capabilities it needs and their priorities (0 to 1).

- `DataPipeline.plan_task(task, round_no=0)` returns a `DataPlan` with one `dataset` request per capability. The request count scales with the task priority and the catalog priority, between `data.plan.min_count` and `max_count`, and the total is capped at `task.count`. The strategy is `task_requirements`.
- `DataPipeline.collect_task(task, round_no=0)` plans the task and loads its rows through the dataset library. It returns up to `task.count` samples and does not run the filter, difficulty, mixing, or builder stages. It requires a collector with kind `dataset`.

## Configuration

| Option under the collector entry in `data.collectors` | Default config | Effect |
| --- | --- | --- |
| `type` | `dataset_library` | Registers the collector for requests with kind `dataset`. |
| `catalog` | `configs/datasets.yaml` | Registry file. |
| `max_datasets` | `4` | Maximum number of sources selected per request. |
