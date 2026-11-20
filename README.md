# RL_RSI

A Self-improvment RL Loop to automatically improve a model performance via reinforcement learning.

loop for: trajectory collect(rollout or from inference) -> analyze bad case -> task synthesis(data collect\env synthesis) -> RL train -> update checkpoint ->  traj collect

This feature will act like glue, combining Verl, VLLM, Harness, evalscope and certain algorithms into an RSI Loop.

## 1. trajectory collect.  
- Deploy model with current checkpoint as a inference endpoint. (via vllm\sglang, etc.)
- collect trajectory from deploy endpoint during a period of time.(Due to data security and privacy concerns, we are unable to collect users' environmental and data information.)
- run benchmark with current checkpoint (via evalscope, etc.)(pass@k\mean@k\response_length)
- collect benchmark result and trajectory

## 2. analyze bad case(base on P!=NP, verify is easier than generation)
- Rule-base benchmark analyze. (pattern of low-scoring benchmark combination)
- About rule-base analyze algorithm： 1. Defining hundreds of basic atomic capabilities. 2. Maintain a mapping between each sub-item of a benchmark and multiple capabilities. 3. If some benchmark sub-item is lower than average, mark the corresponse capabilities as nagative. 4. Use OR operations to synthesize the capability deficiencies reported by various benchmarks. 
- Using LLM judge to analyze bad case.
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
- Monitoring the metric.(Validation\entropy and so on. decide early stop or so on)
- Analyze the metrics(if training goes in the desired direction). If not, identify the problem (reward\penalty) and adjust. Descide continuous training or re-training from the begining.
- Finally get a new version of checkpoint which is good at validation score.

## 5. return to step 1 and as a loop.
