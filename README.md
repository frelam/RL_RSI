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

## 1. trajectory collect and benchmark.  
- Deploy model with current checkpoint as a inference endpoint. (via vllm\sglang, etc.)
- collect trajectory from deploy endpoint during a period of time.(Due to data security and privacy concerns, we are unable to collect users' environmental and data information.)
- run benchmark with current checkpoint (via evalscope, etc.)(pass@k\mean@k\response_length and so on)
- collect benchmark result and trajectory

## 2. analyze bad case(base on P!=NP, verify is easier than generation)
- Rule-base benchmark analyze. (pattern of low-scoring benchmark combination)
- About rule-base analyze algorithm： 1. Defining hundreds of basic atomic capabilities. 2. Maintain a mapping between each sub-item of a benchmark and multiple capabilities. 3. If some benchmark sub-item is lower than average, mark the corresponse capabilities as nagative. 4. Use OR operations to synthesize the capability deficiencies reported by various benchmarks. 
- Using LLM judge to analyze bad case.（Or maybe human judge\multi model judge and vote\teacher judge\self judge）
- About LLM judge analyze algorithm: Good skills, good prompt, and use some scoring rules to reduce scoring bias.
- Summarize What capabilities are missing in the model.

## 3. collect more data for model addressing the shortcomings of the model
- Deploy the current version of model into a harness as an agent.
- Base on the defect found in step2, use agent to collect data.
- Need some smart algorithm of data collection. eg: 1.Break down the missing capabilities into sub-capabilities to find sub-problems. 2. Ensure data diversity. 3. design a  curriculum learning strategy， decide which ability first to learn and which the next round.
- For agentic task: write code to synthesis enviroment or task.
- For generate task: collect or synthesis data.
- Collect new Knowledge: collect some new world data. May be world new knowledge, new problem, or use camara to collect real-world events
- Consider and decide on the data mixing ratio. (Understanding the conflicts and synergies between datasets category. If confict, find Pareto value; If synergies, find best ratio.) Use experiment to find the best ratio.
- Filter the data according to difficulty level.
- Build a new dataset and reward.

## 4. use RL framework to train by model it-self.(via verl, slime, etc.)
- Deploy the current version of model into a harness as an agent.
- Use agent to debug training scripts based on the new reward, data, and receipe.
- Train the model for some while.
- Analyze the metrics(if training goes in the desired direction). If not, identify the problem (reward\penalty) and adjust. Descide continuous training or re-training from the begining.
- Monitoring the metric.(Validation\entropy and so on. Check if some task is saturation. Decide need early stop)
- Need metrics observe-action strategy： for example，If domain pass rate differ too much，update domain data， to avoid gradient direction not correct
- Finally get a new version of checkpoint which is good at validation score.


## 5. return to step 1 and as a loop.
