# RL_RSI

A Self-distillation Loop to automatically improve a model performance via reinforcement learning.

loop for: trajectory collect(rollout or from inference) -> analyze bad case -> task synthesis(data collect\env synthesis) -> RL train -> update checkpoint ->  traj collect

## 1. traj collect.  
### rl train validation trajectory collect
### collect trajectory from deploy endpoint

## 2. analyze bad case(base on P!=NP, verify is easier than generation)
### Using benchmarks.
### Using LLM judge analyze bad case. 
### Summarize what ability is lack of Model.

## 3. collect more data for model addressing the shortcomings of the model
### Deploy the self model into a harness, and use search tools to collect data.
#### collect data policy: 1.Break down the missing capabilities into sub-capabilities to find sub-problems.
### write code to sync enviroment or task for agentic data.
### sync data for other task.
### collect some new world data. May be world new knowledge, new problem, or use camara to collect real-world events

## 4. use RL framework to train it-self(verl, slime, etc.)
### using new collected data.
### Filter the data according to difficulty level
### train the model until convert

## 5. return to step 1 and that is a loop.
