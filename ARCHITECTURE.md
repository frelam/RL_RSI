# RL-RSI 架构设计

> 状态：可运行骨架（v0.1）。默认配置是离线、确定性 mock 栈；真实 vLLM / verl / EvalScope / agent harness 通过 adapter 接入。
> 代码入口：`src/rsi/`；示例配置：`configs/default.yaml`；离线演示：`configs/demo_mock.yaml`。

## 1. 这个系统是什么

RL-RSI 是一个**代码驱动的递归自提升编排层**：

```text
trajectory collect + benchmark
        ↓
bad-case / capability analysis
        ↓
data & environment synthesis
        ↓
filter → difficulty → mixing → dataset + reward
        ↓
RL training (verl / slime / ...)
        ↓
validation → checkpoint promotion
        ↓
新 checkpoint 回到 collect
```

它不是训练框架，也不是 harness。它负责**不变量**：数据契约、checkpoint 版本、评估口径、停止条件、审计与可复现；vLLM、verl、harness 都是被编排的工具。

三个设计原则：

1. **代码拥有控制流，harness 只拥有开放性任务。** 评估、比较、晋升、停止、数据配额全部是确定性 Python；harness 的输出必须经过 JSON Schema 校验，且默认只有建议权。
2. **模块之间只通过数据对象通信。** 每个组件实现一个窄接口，输入输出是 `core/types.py` 里的 dataclass；替换实现不需要改编排。
3. **每一项选择都是配置。** 插件用注册表解析，配置按 `aaa.bbb.ccc` 分组，CLI 可直接覆盖叶子节点。

## 2. 业界形态与设计取舍

公开资料中，2025–2026 年的自改进/RSI 工作大致分三条线：

- **自博弈与零数据训练**：模型同时扮演出题者与解题者，用验证器筛选轨迹（Absolute Zero、R-Zero 一类；ICLR 2026 已有专门的 RSI workshop 和大量 self-play 论文）。
- **验证器门控的合成数据**：ReST/STaR 一类的 rejection sampling、可验证奖励、代码执行器、环境测试；核心瓶颈从“生成”转向“可验证性”。
- **Agentic self-improvement**：harness 驱动环境合成、工具使用、训练脚本自调试；但 harness 的开放性与不确定性也最高。

基础设施侧，verl（HybridFlow）、slime、SkyRL、AReaL 等已经把分布式 rollout 与训练解耦；ProRL Agent 一类工作把 rollout 做成独立 HTTP 服务。**缺少的不是训练引擎，而是一个把 collect → analyze → synthesize → train → re-evaluate 串起来、且能审计/复现/停止的编排契约。**

基于这个判断，本项目做如下取舍：

| 趋势 | 架构响应 |
| --- | --- |
| rollout 与训练解耦 | `InferenceBackend` / `TrainerBackend` 分离；trainer 可接 rollout-as-a-service |
| 验证器成为瓶颈 | benchmark item 级分数是硬证据；rule analyzer 先于 LLM judge；data filter 有 verifier 门 |
| agentic 环境合成增长 | `AgenticSynthesisCollector` + skill hub + harness adapter；输出 schema 校验、工作目录隔离 |
| 长周期/持续训练 | `LoopState` + checkpoint registry + resume；停止策略插件化 |
| 审计与回放需求 | 每次外部命令写 `commands.jsonl`；每阶段产物落盘；run manifest 记录种子与配置 |

对未来需求的判断：RSI 会从“单次流水线”走向“持续、多源、可验证、可回滚”的闭环；因此本设计把**数据血缘、checkpoint 谱系、评估口径、停止/预算**放在核心，而不是把某个具体算法写死在 loop 里。

## 3. 总体分层

```text
┌──────────────────────────────────────────────────────────────────────┐
│ CLI / experiment scripts                                             │
│   rsi run | validate | inspect | plugins                             │
└───────────────────────────────┬──────────────────────────────────────┘
                                │ Config (YAML + dotted overrides)
┌───────────────────────────────▼──────────────────────────────────────┐
│ Orchestration: rsi.loops                                             │
│   Loop (abstract) → ClassicLoop → Stage[]                            │
│   LoopState · StopEvaluator · stage retry/failure policy             │
└───────┬───────────────────────┬───────────────────────┬──────────────┘
        │                       │                       │
┌───────▼──────────┐   ┌────────▼─────────┐   ┌─────────▼────────────┐
│ Collection       │   │ Analysis         │   │ Data                 │
│  CollectService  │   │  rule analyzer   │   │  planner/curriculum  │
│  RolloutCollector│   │  LLM judge       │   │  collectors          │
│  benchmarks      │   │  harness deep-dive│  │  filter/grade/mix    │
│                  │   │  DefectMerger     │   │  dataset builder     │
└───────┬──────────┘   └────────┬─────────┘   └─────────┬────────────┘
        │                       │                       │
┌───────▼───────────────────────▼───────────────────────▼──────────────┐
│ Training & checkpoint                                                │
│   Trainer · MonitorHub · Supervisor · validation · promotion          │
│   CheckpointManager (candidate → current / rejected / archived)      │
└───────┬──────────────────────────────────────────────────────────────┘
        │
┌───────▼──────────────────────────────────────────────────────────────┐
│ Adapters (only place that touches the outside world)                  │
│   inference: mock | openai | vllm | sglang                           │
│   benchmark: mock | jsonl | evalscope                                │
│   trainer:   mock | command | verl | slime                           │
│   harness:   mock | command                                          │
│   llm:       mock | openai-compatible                                │
└──────────────────────────────────────────────────────────────────────┘
        │
┌───────▼──────────────────────────────────────────────────────────────┐
│ Cross-cutting: Runtime · Registry · Config · ArtifactStore           │
│                EventLog · CommandExecutor · PromptHub · SkillHub      │
└──────────────────────────────────────────────────────────────────────┘
```

依赖方向是单向的：`loops → modules → core`，`adapters → core`。模块不 import loop，loop 不 import 具体 adapter 类，只通过 `Runtime` 的注册表解析。

## 4. 目录结构

```text
src/rsi/
├── core/                     # 不依赖外部工具的契约层
│   ├── types.py              # 所有跨模块数据对象
│   ├── interfaces.py         # Component + 窄接口 ABC
│   ├── config.py             # YAML/include/覆盖/分组读取
│   ├── registry.py           # 类别 + 名称 → 插件类
│   ├── runtime.py            # 路径/日志/事件/命令/checkpoint/hub
│   ├── schema.py             # 轻量 JSON Schema 校验
│   └── scoring.py            # pass@k 与文本 scorer
├── adapters/                 # 外部系统边界
│   ├── inference.py          # mock / OpenAI-compatible / vLLM 本地拉起
│   ├── benchmark.py          # mock / JSONL / EvalScope 命令
│   ├── trainer.py            # mock / 通用命令 / verl / slime 预设
│   ├── harness.py            # mock / command（任务文件 + 输出文件契约）
│   ├── llm.py                # mock / OpenAI-compatible
│   └── openai_http.py        # 零依赖 HTTP 客户端
├── hubs/
│   ├── prompts.py            # Prompt Hub：版本、变量、输出 schema
│   ├── skills.py             # Skill Hub：SKILL.md + 资产，可物化到 sandbox
│   └── catalog.py            # Capability Catalog：能力词表与数据/奖励提示
├── modules/
│   ├── collect.py            # benchmark + rollout 汇总
│   ├── metrics.py            # item → capability 聚合与能力映射
│   ├── analyze_rule.py       # 规则版能力缺陷
│   ├── analyze_llm.py        # LLM judge / harness 深挖
│   ├── analyze_merge.py      # 多来源缺陷融合（noisy-OR）
│   ├── data_plan.py          # defect → DataRequest（rule / LLM）
│   ├── data_curriculum.py    # severity / fixed / prerequisite 课程策略
│   ├── data_collectors.py    # text / agentic / rollout / knowledge
│   ├── data_filters.py       # PII / 长度 / 格式 / judge / dedup / verifier
│   ├── data_difficulty.py    # rule / judge / pass-rate / composite
│   ├── data_mixing.py        # 静态 / 能力加权 / history bandit + 配额采样
│   ├── data_quotas.py        # 有容量与比例约束的整数配额求解
│   ├── data_builder.py       # JSONL / verl view / reward spec / manifest
│   ├── data_pipeline.py      # plan → generate → filter → grade → mix → build
│   ├── train_monitor.py      # 监控算法、MonitorHub、Supervisor
│   └── ...
├── loops/
│   ├── base.py               # Loop / Stage / LoopContext
│   ├── stages.py             # sync/collect/analyze/synthesize/train/validation/promote
│   ├── classic.py            # README 中的经典循环
│   ├── state.py              # LoopState 持久化
│   └── stop.py               # 停止策略插件
├── hubs / configs / prompts / skills / examples / tests
```

## 5. 核心数据契约

所有跨模块数据都在 `core/types.py`。主要对象：

| 对象 | 生产者 | 消费者 | 关键字段 |
| --- | --- | --- | --- |
| `Trajectory` | `RolloutCollector` | 分析、rollout 复用 | messages, final_answer, reward, checkpoint_id |
| `BenchmarkReport` | benchmark adapter | collect/metrics/analyzer | item 级 score、capability、aggregates |
| `CapabilityScore` | `metrics.capability_table` | analyzer/planner/difficulty | score, baseline, delta, num_items |
| `DefectReport` | analyzer + merger | planner | defects[], severity, confidence, evidence |
| `DataPlan` | `DataPlanner` | collectors | requests[], strategy, curriculum |
| `DataSample` | collectors | filter/grade/mix/builder | prompt, response, verifier, difficulty, provenance |
| `DatasetArtifact` | builder | trainer | train/val/reward/manifest 路径、mixing、stats |
| `TrainingResult` | trainer | supervisor/validation | metrics, best_metric, checkpoint_path, status |
| `Checkpoint` | `CheckpointManager` | sync/inference/validate | id, version, model_path, status, lineage |
| `LoopState` | loop | loop/CLI | round, history, best_score, budget, stop_reason |

示例（截断）：

```json
{
  "capability": "long_context.retrieval",
  "severity": 0.685,
  "confidence": 0.99,
  "source": "rule+llm_judge",
  "evidence": ["demo:long_context.retrieval#3 score=0.22"]
}
```

```json
{
  "dataset_id": "ds-r0000-c5db38ae",
  "train_path": ".../train.jsonl",
  "val_path": ".../val.jsonl",
  "reward_spec_path": ".../reward_spec.json",
  "mixing": {"code.debugging": 0.41, "long_context.retrieval": 1.15},
  "num_samples": 8
}
```

`DatasetArtifact.stats` 带 filter 统计、mixing candidate、生成错误等；`manifest.json` 带文件哈希、配置快照和数据集血缘，便于复现。

## 6. 编排模型

### 6.1 Classic Loop

`ClassicLoop` 由 8 个 Stage 顺序组成，每个 Stage 的输入来自 `LoopContext.memory` 或 run state：

```text
sync       读取 current checkpoint，确保 inference endpoint 可用
collect    benchmark + rollout → CollectResult
analyze    rule + LLM judge (+ 可选 harness) → 融合后的 DefectReport
synthesize plan → collectors → filters → difficulty → mixing → dataset
train      trainer 运行；MonitorHub 实时观察；Supervisor 判定重试/使用/丢弃
validation 在候选 checkpoint 上跑 benchmark（或退回 trainer 指标）
opd        可选 teacher-aware command adapter；以已验证 checkpoint 为基准评估
promote    晋升已接受 checkpoint；否则拒绝并保留现有版本
stop policy 在整轮结束后评估，不属于 Stage
```

每一轮记录写入 `state.json.history`，包含 `checkpoint_in`、各阶段摘要、`validation.score/delta/accepted`、`stop`。失败阶段不会写入历史；`state.status=failed`，`--resume` 可重试。

### 6.2 新增一种编排

新增一个文件即可，不需要改模块：

```python
@register("loop", "two_phase")
class TwoPhaseLoop(Loop):
    name = "two_phase"

    def run(self, *, resume: bool = False) -> LoopResult:
        # 复用 SyncStage / CollectStage / AnalyzeStage / DataStage ...
        ...
```

`Loop.validate()` 负责在不动外部系统的前提下构造完整组件树，`rsi validate` 直接调用它。Stage 的失败策略是配置项：`loop.on_stage_error=abort|skip`、`loop.max_stage_retries`。

### 6.3 Run 目录与 checkpoint 放置

```text
runs/<run_id>/
├── run.json                 # run id / seed / repo root / python
├── state.json               # 唯一循环状态（原子写）
├── events.jsonl             # run_started / round_completed / stage_failed ...
├── commands.jsonl           # 每一次外部命令（审计/回放）
├── rsi.log
├── checkpoints/
│   └── registry.json        # current + 全部 checkpoint + next_version
└── rounds/round-0000/
    ├── collect/             # benchmark 报告、trajectories
    ├── analyze/             # 分方法 defects + 合并结果
    ├── synthesize/          # plan、pipeline 统计、dataset/
    ├── train/strategy-0/    # metrics, findings and checkpoint per strategy attempt
    ├── opd/                 # optional distillation output and teacher comparison
    ├── validation/          # candidate and incumbent benchmarks, validation.json
    └── promote/             # promotion.json
```

checkpoint 的**权威状态只在 `checkpoints/registry.json`**：`candidate → current / rejected / archived`。Stage 每轮通过 `SyncStage` 读取 `current()`，因此数据、评估、训练看到的是同一个版本；测试和审计不会拿着过期对象继续跑。

## 7. 配置系统

- YAML 嵌套 + `include` 合并（列表整体替换，映射递归合并）。
- 点路径覆盖：`rsi run ... analyze.merge.min_confidence=0.35 stop.combine=all`。
- `${ENV:NAME}` / `${ENV:NAME:default}` 展开；`${config_dir}` / `${repo_root}` 可用。
- 每个组件收到自己的子树，例如 `data.filter.pipeline[0]` 是一个独立 `Config`。
- 插件列表的四种写法：

```yaml
methods: [rule, {type: llm_judge, prompt: badcase_analysis}]
collectors:
  text: {type: text_synthesis, prompt: data_text_synthesis}
benchmarks:
  demo: {type: mock, items_per_capability: 8}
```

- `paths.repo_root` 可显式指定；未指定时从 config/include 的位置向上找 `.git`/`pyproject.toml`。
- 密钥只通过环境变量注入，不写进配置文件；`manifest` 不包含完整 config 快照中的敏感字段（builder 只记录自己的子树）。

常用分组：

```text
run.seed                          # 全局种子
loop.*                            # 编排与失败策略
infer.*                           # policy 推理端点
collect.*                         # benchmark / rollout 采集
analyze.methods / analyze.merge.* # 缺陷分析
data.plan.* / data.collectors.*   # 规划与数据源
data.filter.pipeline.*            # 过滤链
data.difficulty.*                 # 难度
data.mixing.*                     # 配额与混合
data.build.*                      # 数据集与 reward
train.monitor.* / train.supervisor.*
trainer.*                         # RL 引擎
validate.*                        # 候选评估
stop.policies.*                   # 停止条件
llm.profiles.*                    # judge/generator 模型
harness.<name>.*                  # 每种 harness
checkpoint.*                      # 初始与保留策略
```

## 8. 组件、插件与 adapter

注册表类别：`loop / analyzer / data_collector / data_filter / difficulty / curriculum / mixing / dataset_builder / monitor / stop_policy / trainer / inference / benchmark / harness / llm`。

插件解析顺序：

1. 注册名（`mock`、`verl`、`text_synthesis`）；
2. 完整 import 路径（`my_pkg.my_module:MyClass`）。

Adapter 边界与“真实接入点”：

| Adapter | 真实系统 | 现状 | 接入方式 |
| --- | --- | --- | --- |
| inference | vLLM / SGLang / 任意 OpenAI API | mock + OpenAI-compatible + vLLM 本地拉起 | `infer.type=vllm`，配置 `launch.command` |
| benchmark | EvalScope / lm-eval / 自研 | mock + 本地 JSONL + 命令解析 | `type: evalscope`，命令模板 + 结果 JSON |
| trainer | verl / slime / 自研 | mock + 通用命令 + 预设 | `type: verl`，命令模板 + JSONL metrics |
| harness | SWE-agent / OpenHands / 自研 | mock + 通用命令 | `type: command`，任务/输出文件契约 |
| llm | 任意 OpenAI-compatible judge/generator | mock + HTTP | `llm.profiles.<name>` |

**Adapter 的职责只有翻译**：参数翻译、输出解析、把异常变成 `AdapterError`。它不决定数据是否合格、不决定 checkpoint 是否晋升。

### Metric 约定

真实 trainer 需要产出每行一个 JSON 对象的 metrics 文件：

```json
{"step": 10, "metrics": {"loss": 0.8, "grad_norm": 1.2, "entropy": 0.9, "reward": 0.4, "val_score": 0.52}}
```

vLLM/verl 的日志格式各异，因此“文本日志 → 上述 JSONL”的解析属于 adapter 的一部分；这是接入真实 verl 时唯一需要按版本定制的代码。

## 9. 数据闭环

```text
DefectReport
   │ curriculum: severity / fixed / prerequisite
   ▼
DataPlanner (rule | LLM) → DataRequest[]
   │ kind = text | agentic | knowledge | rollout
   ▼
Collectors
   ├── text_synthesis      teacher LLM，prompt hub + JSON schema
   ├── agentic_synthesis   harness + skill hub → 任务目录 + verifier
   ├── rollout_reuse       当前/历史成功轨迹
   └── knowledge_ingest    本地语料分块 → 有 provenance 的 QA
   ▼
FilterPipeline
   format → PII → length → judge_quality → dedup → verifier → pattern → optional adaptive_rollout
   ▼
DifficultyGrader
   rule / judge / pass-rate（1 - 当前能力分）/ rollout / composite
   ▼
MixingStrategy + DiversitySampler
   static / capability_weighted / history_bandit
   目标难度带、每组最小/最大配额、确定性打散
   ▼
DatasetBuilder
   samples.jsonl / train.jsonl / val.jsonl
   reward_spec.json / stats.json / manifest.json
```

要点：

- **课程策略**单独成模块：`severity` 按 `severity×confidence`，`fixed` 按能力顺序逐轮解锁，`prerequisite` 在前置能力达到阈值前锁定后续能力。
- **历史 replay** 从近期轮次的 train.jsonl 按比例抽样，排除 val.jsonl，并在配比采样时保留其原能力分组。
- **去重是跨轮次的**（`DedupFilter` 实例常驻），避免每轮重复合成同一批数据。
- **自适应 rollout** 是可选数据过滤器。当前策略按 4、8、16 个回答逐步估计可验证题目难度，保留结果有对有错的题目，可将全错题扩展到 32 个回答。
- **训练反馈** 可提高目标难度、调整 reward 权重，并在有限次策略尝试内重建数据集。
- **目标能力验收** 对候选和现有版本比较同一批 benchmark 项。目标能力必须提高，其他能力退步幅度由 validate.max_other_regression 限制。
- **OPD** 通过独立配置的 teacher-aware trainer 命令执行，并用 benchmark 对比蒸馏产物与已验证 teacher。
- **比例采样**以筛选前的分组或策略配置确定目标，使用筛选后的容量求最大整数配额。难度带可逐档放宽，同等总量优先采用更严格的档位。JSONL 构建器可缩小验证集，以便切分后的训练占比仍在容差内。
- **mixing 实验**：`history_bandit` 先用探索轮次，再根据历史 `validation.delta` 选择候选比例；选择记录在 dataset artifact 和 round record 中，可审计。
- **reward spec** 按能力生成：catalog 的 `reward` 规则 + 配置覆盖；每条样本的 `verifier` 随数据落盘。
- **provenance**：source、request_id、defect_id、round、license、文件路径都写入样本 metadata。

## 10. 分析闭环

### 规则分析

现有 benchmark 结果本身就是“验证器已经做过一次”的证据，优先使用：

1. catalog 把 item 的 capability/subcapability 映射到原子能力；
2. 按能力聚合分数，与 `global` / `previous` / `absolute` 基线比较；
3. 低于阈值的能力产生 defect，`severity ∝ gap × priority`；
4. 多 benchmark 的同一能力用“最强信号 + 佐证加成”合并；
5. 可选的 subcapability 粒度输出。

### LLM judge

- Prompt 与输出 schema 都在 Prompt Hub，版本化、可审查。
- 支持多 profile / 多次投票：`votes`、`vote_policy=majority|mean`、`min_vote_ratio`。
- 只分析最差 cases（`top_n`、`max_score`），控制成本。
- 输出经过 JSON Schema 校验；非法输出被丢弃，不会进入 planner。
- 默认与规则结果融合：`severity = max(weighted) + corroboration`，`confidence = noisy-OR`。单一 judge 不能凭一票制造高置信缺陷。

### Harness 深挖

可选的 `harness` analyzer 把 worst cases 交给 agent 检查，输出同样是 `defects` schema；它在 merge 中权重与 rule/judge 同级但默认不参与晋升决策。

## 11. 训练稳定性与 checkpoint

### 监控算法

`MonitorHub` 在训练过程中逐条消费 metrics，运行配置的算法：

| 算法 | 信号 | 默认动作 |
| --- | --- | --- |
| `nan_guard` | loss/grad 非有限 | critical → 停止训练 |
| `grad_spike` | grad_norm 绝对值或相对基线突增 | critical/warning |
| `entropy_collapse` | entropy 连续低于阈值 | warning/critical |
| `reward_hacking` | reward 上升但 val_score 不升 | warning，建议审计 reward |
| `plateau` | val_score 在窗口内不再提升 | warning，建议 early stop |
| `saturation` | val_score 达到饱和阈值 | info，建议提高难度 |
| `zscore` | 任意指标的滚动 z-score | 可配置 |

训练器通过 observer 回调接收每条 metric；observer 返回 `"stop"` 时，`CommandTrainer` 会终止进程，`MockTrainer` 直接跳出循环。

### Supervisor

- `retrain_on_critical`：critical 且还有重试次数 → 重新训练，携带 finding 里的 `adjustments`（供 adapter 翻译为 CLI override）。
- `retrain_on_failure`：进程失败/超时 → 重试。
- `discard_on_critical`：严重不稳定 → 丢弃候选。
- 否则接受 best checkpoint（best metric 来自 trainer 的 `best_metric` 或 validation）。

### Checkpoint 谱系与同步

- 初始 checkpoint 由 `checkpoint.initial` 注册为 `ckpt_v000`。
- 训练产出路径 → `reserve()` 得到 `candidate`（version 自增）。
- validation 分数写回候选 metrics。
- `promote()` 把旧的 current 标为 archived，新候选标为 current；`reject()` 保留候选并记录原因。
- 每轮开头 `SyncStage` 重新读取 `current()`，并调用 `inference.ensure(checkpoint)`；因此下一轮的 collect/train 不会误用旧版本。
- `--resume` 从 `state.json` 继续，不重放已完成轮次；失败轮次不会写入历史，可重试。

### 接受/拒绝

`validation` 在 inference 支持切换 checkpoint 时，用同一组 benchmark 项配对比较候选和现有版本；否则使用 trainer 指标。

```text
score >= baseline + validate.min_delta
每个目标能力都必须改善；非目标能力退步不得超过 validate.max_other_regression。
```

validate.target_min_delta 和 validate.max_other_regression 控制目标能力改善与其他能力的回退上限。拒绝的候选保留在 registry 供审计，但不会成为 current。OPD 产物未通过 benchmark 对比时保留已验证的 teacher。

## 12. 停止条件

停止策略是独立插件，默认“任一触发”：

- `max_rounds`：完成轮次上限；
- `target_score`：best score 达到目标（可设 `min_rounds`）；
- `plateau`：最近 `patience` 轮的验证提升小于 `min_delta`；
- `regression`：最新候选比 best 低超过 `drop`；
- `budget`：wall-clock / GPU-hour 预算；
- `manual`：run 目录或仓库根出现 STOP 文件。

`stop.combine=all` 可以要求多个条件同时满足。策略评估发生在每轮结束后，也用于 resume 前的预检查。

## 13. 确定性设计：代码驱动，harness 是工具

1. **唯一命令出口**：所有外部命令走 `CommandExecutor`，记录 command、cwd、returncode、耗时到 `commands.jsonl`。`CommandTrainer` 的进程也通过它启动。
2. **种子**：`Runtime.rng(scope...)` 用 `sha256(seed, run_id, scope)` 生成独立随机流；mock benchmark、采样、打散都可复现。
3. **schema 门**：harness 输出、LLM judge 输出、data synthesis 输出都过 `core/schema.py`；不合格的产物被丢弃而不是“尽力修复”。
4. **权限边界**：harness 工作目录隔离；skill 物化到 `.rsi_skills`；任务文件路径禁止绝对路径和 `..`；输出只能写 `{output_file}`。
5. **建议权**：analysis harness 只能产生 defect 建议；训练 debug harness 只能产生 diagnosis/actions；二者都不能直接修改 checkpoint 或配置。
6. **可审计**：每阶段产物、round record、checkpoint registry、dataset manifest 都落盘；`rsi inspect` 可回看每轮分数与接受/拒绝。
7. **真实系统边界**：vLLM/verl/EvalScope 的不确定性被封装在 adapter；一旦 parse 失败或超时，抛 `AdapterError`，由 stage 失败策略决定 abort/retry，而不是静默继续。

## 14. 扩展指南

### 新增 analyzer

```python
@register("analyzer", "my_method")
class MyAnalyzer(AnalyzerBackend):
    def analyze(self, collect, round_no, baseline):
        return DefectReport(...)
```

配置：`analyze.methods: [{type: my_method, ...}]`。不需要改 merge 或 loop。

### 新增数据源

```python
@register("data_collector", "my_source")
class MyCollector(DataCollector):
    kind = "my_kind"
    def collect(self, request, context): ...
```

配置 `data.collectors.<name>`，planner 只要产出 `kind: my_kind` 就会路由到它。

### 新增 harness

优先使用 `type: command`，遵守任务/输出文件契约；如果需要特殊权限或协议，再继承 `HarnessBackend`。把 skill 放进 `skills/<name>/SKILL.md`，配置 `skills: [name]`。

### 新增 trainer

优先使用 `type: command`（或 `verl`/`slime` 预设），只写命令模板和 metrics 解析。若框架需要强类型 API，再实现 `TrainerBackend.train()`，保持 observer 回调语义。

### 新增 loop

新建 `loops/<name>.py`，组合现有 Stage 或写新 Stage，注册 `loop.<name>`；配置 `loop.type=<name>`。模块层完全不需要知道。

## 15. 运行

```bash
# 校验完整组件树（不调用外部系统）
PYTHONPATH=src python -m rsi validate --config configs/default.yaml

# 离线跑通经典循环
PYTHONPATH=src python -m rsi run \
  --config configs/demo_mock.yaml \
  --run-id demo --runs-dir runs

# 查看状态
PYTHONPATH=src python -m rsi inspect runs/demo --history

# 覆盖任意叶子配置
PYTHONPATH=src python -m rsi run --config configs/default.yaml \
  stop.combine=all analyze.merge.min_confidence=0.35

# 测试
PYTHONPATH=src python -m unittest discover -s tests -t .
```

## 16. 已知边界与后续路线

- **真实 verl 接入**：命令模板和 JSONL metrics 解析已就绪；不同 verl 版本的日志字段需要 adapter 内做映射；parquet 数据集转换目前是外部步骤（`verl_jsonl` 只是可读视图）。
- **rollout-as-a-service**：`InferenceBackend` 已抽象，但高并发轨迹收集、超时重试、限流还需要真实服务侧实现。
- **在线/隐私数据**：README 明确不采集用户环境数据；当前 rollout 来自本地 prompt 语料。若接线上流量，需要额外的脱敏、采样与合规门。
- **分布式 loop**：当前是单进程编排、串行 Stage；多机训练通过 trainer 命令外扩。未来可把 Stage 变成可恢复的 DAG/任务队列。
- **评估污染**：benchmark 需要冻结并版本化；manifest 已记录文件哈希，后续应增加 benchmark 版本号与去重检查。
- **成本**：LLM judge、多票、harness 深挖都可能很贵；配置里所有调用都有 `top_n`/`votes`/`max_tasks` 上限，默认 mock 不产生费用。
- **安全**：训练 debug harness 默认关闭；启用时只输出建议，不自动应用。真实环境合成需要额外沙箱（容器/网络隔离）。

## 17. 参考资料

公开资料用于判断形态，不作为实现正确性的依据：

- ICLR 2026 Workshop on AI with Recursive Self-Improvement: <https://iclr.cc/virtual/2026/workshop/10000796>
- Workshop accepted papers: <https://recursive-workshop.github.io/papers.html>
- RSI research map: <https://github.com/lobehub/awesome-rsi>
- “From Bounded Self-Refinement to Autonomous Research Loops”: <https://arxiv.org/html/2607.07663v1>
- RSI survey preprint: <https://www.preprints.org/manuscript/202609.2680>
- Evidence review (2026): <https://www.pelles.ai/blog/recursive-self-improvement-state-of-the-evidence>
- Morph, “Self-Improving AI: What Actually Works in 2026”: <https://www.morphllm.com/self-improving-ai>
- Aman primer on RSI: <https://aman.ai/primers/recursive-self-improvement>
- verl / HybridFlow: <https://github.com/verl-project/verl>
- verl Agentic RL docs: <https://verl.readthedocs.io/en/latest/start/agentic_rl.html>
- ProRL Agent (rollout-as-a-service): <https://arxiv.org/html/2603.18815v1>
- Search Self-Play: <https://openreview.net/forum?id=ZmGirmNJqE>
