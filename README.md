# RL_RSI

A Self-distillation Loop to automatically improve a model performance via reinforcement learning.

loop for: trajectory collect(rollout or from inference) -> analyze bad case -> task synthesis(data collect\env synthesis) -> RL train -> update checkpoint ->  traj collect

This feature will be available as a plugin, compatible with various harnesses, such as DeepSeek Harness and Hermes.

## 1. trajectory collect.  
- rl train validation trajectory collect
- collect trajectory from deploy endpoint

## 2. analyze bad case(base on P!=NP, verify is easier than generation)
- Using benchmarks.
- Using LLM judge analyze bad case. 
- Summarize what abilities are lack in Model.

## 3. collect more data for model addressing the shortcomings of the model
- Deploy the self model into a harness, and use search tools to collect data.
- collect data policy: 1.Break down the missing capabilities into sub-capabilities to find sub-problems.
- write code to synthesis enviroment or task for agentic data.
- synthesis data for other task.
- collect some new world data. May be world new knowledge, new problem, or use camara to collect real-world events
- Consider and decide on the data mixing ratio.
- Filter the data according to difficulty level
- build a new dataset and reward

## 4. use RL framework to train it-self(verl, slime, etc.)
- Debug training scripts based on the new reward, data, and receipe.
- using the new dataset train the model until converge.
- dynamic indicator monitoring(early stop, auto recover)

## 5. deploy a new checkpoint and return to step 1 and that is a loop.
