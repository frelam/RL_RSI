# RL_RSI

A Self-improvment RL Loop to automatically improve a model performance via reinforcement learning.

loop for: trajectory collect(rollout or from inference) -> analyze bad case -> task synthesis(data collect\env synthesis) -> RL train -> update checkpoint ->  traj collect

This feature will act like glue, combining Verl, VLLM, Harness, and certain algorithms into an RSI Loop.

## 1. trajectory collect.  
- Deploy model with current checkpoint as a inference endpoint. (via vllm\sglang, etc.)
- collect trajectory from deploy endpoint during a period of time.(Due to data security and privacy concerns, we are unable to collect users' environmental and data information.)
- run benchmark with current checkpoint (via evalscope, etc.)
- collect benchmark result and trajectory

## 2. analyze bad case(base on P!=NP, verify is easier than generation)
- Rule-base benchmark analyze. (pattern of low-scoring benchmark combination)
- About rule-base analyze algorithm： 1. Defining hundreds of basic atomic capabilities. 2. Maintain a mapping between each sub-item of a benchmark and multiple capabilities. 3. If some benchmark sub-item is lower than average, mark the corresponse capabilities as nagative. 4. Use OR operations to synthesize the capability deficiencies reported by various benchmarks. 
- Using LLM judge to analyze bad case. 
- Summarize what abilities are lack in Model.

## 3. collect more data for model addressing the shortcomings of the model
- Deploy the self model into a harness as an agent.
- Base on the defect found in step2, use agent to collect data.
- Need some smart policy of data collection. eg: 1.Break down the missing capabilities into sub-capabilities to find sub-problems.
- write code to synthesis enviroment or task for agentic data.
- synthesis data for other task.
- collect some new world data. May be world new knowledge, new problem, or use camara to collect real-world events
- Consider and decide on the data mixing ratio.
- Filter the data according to difficulty level
- build a new dataset and reward

## 4. use RL framework to train by model it-self.(via verl, slime, etc.)
- Debug training scripts based on the new reward, data, and receipe.
- using the new dataset to train the model until converge.
- dynamic indicator monitoring(early stop, auto recover)
- run benchmark, if not good, than analyze and retrain(may be go back to step3)
- finally get a new version of checkpoint which is better than before.

## 5. return to step 1 and as a loop.
