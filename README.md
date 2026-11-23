# RL_RSI

A Self-improvment RL Loop to automatically improve a model performance via reinforcement learning.

> **实现状态（v0.1）**：仓库已包含可运行的模块化骨架、默认离线 mock 栈、端到端测试。设计说明见 [ARCHITECTURE.md](ARCHITECTURE.md)。
>
> ```bash
> # 只校验配置与插件树，不调用外部系统
> PYTHONPATH=src python -m rsi validate --config configs/default.yaml
>
> # 离线跑通经典循环（3 轮左右达到目标分）
> PYTHONPATH=src python -m rsi run --config configs/demo_mock.yaml --run-id demo
>
> # 查看每轮采集分、验证分与接受/拒绝
> PYTHONPATH=src python -m rsi inspect runs/demo --history
>
> # 全量测试
> PYTHONPATH=src python -m unittest discover -s tests -t .
> ```
>
> 真实 vLLM / verl / EvalScope / agent harness 通过 adapter 接入，示例配置在 `configs/default.yaml` 的注释里。所有外部命令记录到 `runs/<id>/commands.jsonl`，checkpoint 由 `runs/<id>/checkpoints/registry.json` 统一管理。

loop for: trajectory collect(rollout or from inference) -> analyze bad case -> task synthesis(data collect\env synthesis) -> RL train -> update checkpoint ->  traj collect

This feature will act like glue, combining Verl, VLLM, Harness, evalscope and certain algorithms into an RSI Loop.

**A classical Loop** (May support other loop Orchestration):

## 1. Collect trajectory/benchmark and analysis failures

- collect trajectory from deploy endpoint during a period of time.(Due to data security and privacy concerns, we are unable to collect users' environmental and data information.)
- collect newest benchmark and merge into the former benchmark.
- Collect failure traces and run benchmark to test model.(via evalscope, etc.)(pass@k\mean@k\response_length)
- Defined a set of foundation capabilities.
- Summarize which capabilities are missing.(use benchmark maps or LLM judge)
- Prioritize the missing capabilities based on their foundational nature and importance.
- deliver missing capabilities map and their priorities.

## 2. Determine a new data strategy to fix the model
- Need some smart algorithm to produce data strategy. 
  e.g.:<br />
      1.Greedily cover abilities <br />
      2.Assign greater data weight to more important and fundamental capabilities.<br />
      3.Ensure that all affected domains have corresponding data coverage.<br />
      4.Ensure that data strategies for other dimensions are not significantly affected.<br />
- Datasource:<br />
              1. Private datasets (adapters can be defined to specify data retrieval interfaces)<br />
              2. from web<br />
              3. from hf or github<br />
              4. Synthetic data<br />
- Combine recent train.jsonl data with newly sampled data to synthesize a dataset and reduce forgetting. Validation samples are excluded from replay.
- Deliver a new dataset and the desired mix ratio.
- Enhance the reward associated with the data

## 4. use RL framework to train by model it-self.(via verl, slime, etc.)
- Filter the data with difficulty level from data pool.
- Efficient rollout-based data filtering keeps examples that the current policy solves inconsistently. Under independent samples, at 10% accuracy the all-wrong probability falls from 18.53% at 16 rollouts to 3.43% at 32. Expected work rises from 10.07 to 12.19 completions per candidate (+21%). The 32-rollout extension is enabled when the filter is enabled and can be turned off. See [the feature theory and configuration](docs/adaptive_rollout_filter.md).
- With `data.mixing.preserve_ratio` enabled, select the largest dataset whose integer group quotas fit available samples, group bounds, the total cap, and the configured absolute share tolerance (`ratio_tolerance`, 0.10 by default). See [the data mixing theory and configuration](docs/data_mixing_ratio.md).
- Start with the strict difficulty band and widen it by `relax_by` for up to `relax_max_steps` steps. Use the smallest widening that reaches the largest feasible mix, and fill quotas with strict samples first. All candidates must pass the upstream filter pipeline. If no feasible mix exists, training skips that dataset.
- The JSONL builder may reduce the validation holdout to keep the training mix within tolerance. The dataset manifest records the target, selected, and training shares.
- Smoke run.<br /> 1. Ensure reward no hacking. 
            <br /> 2. Ensure data strategy work. 
            <br /> 3. finetune Training hyper parameter.
- Formal Run.
- train strategy required(small batchsize and rollout n firstly,then increase them)
- Monitoring the metric.(Validation\entropy and so on. Need a adapter to parse log for different training backend）
- Strategy to action on observe. 
eg. 1. Check if some task is saturation. Decide need early stop
    2. collect harder data if pass rate too high.
- Go back to filter data step and continue training.
- Finally get a new version of checkpoint which is good at validation score.


## 5. Check new checkpoint
- run benchmark check if desired domain has been improved.
- if ok, run OPD to merge new domain abilities to previous checkpoint.
- if not ok, return to step 2
