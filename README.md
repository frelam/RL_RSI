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
- Combine existing datasets with newly sampled data to synthesize a new dataset (replaying old data to prevent forgetting).
- Deliver a new dataset and the desired mix ratio.
- Enhance the reward associated with the data

## 4. use RL framework to train by model it-self.(via verl, slime, etc.)
- Filter the data with difficulty level from data pool. Need some algorithum.
  e.g. : <br />
 <br /> 1. Iterate through each domain, using its filtered data volume as a benchmark. Calculate the required data volume for other domains based on the proportions. If the criteria are not met, eliminate the proposed solution. Finally, retain the solution with the largest total data volume.
 <br /> 2. If the ratio requirements are not met after filtering, compensate by relaxing the filtering criteria.
- Smoke run.<br /> 1. Ensure reward no hacking. 
            <br /> 2. Ensure data strategy work. 
            <br /> 3. finetune Training hyper parameter.
- Formal Run.
- train strategy required(small batchsize and rollout n firstly,then increase them)
- Monitoring the metric.(Validation\entropy and so on. Check if some task is saturation. Decide need early stop)
- Strategy to action on observe. eg. collect harder data if pass rate too high.
- Go back to filter data step and continue training.
- Finally get a new version of checkpoint which is good at validation score.


## 5. Check new checkpoint
- run benchmark check if desired domain has been improved.
- if ok, run OPD to merge new domain abilities to previous checkpoint.
- if not ok, return to step 2
