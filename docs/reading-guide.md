# 源码阅读指南：先读控制流，再读边界细节

不需要按目录顺序读完全部代码。下列路径都相对 `src/arlab/`，链接指向本包真实源码；按函数名导航，不使用容易随排版变化失效的固定行号。

## 第一遍：五个核心问题

| 顺序 | 入口 | 读到这里应能回答 |
|---|---|---|
| 1 | [`Harness.run()`](../src/arlab/core/harness.py) | 模型调用怎样组成持续任务？中断时先恢复什么？ |
| 2 | [`ContextBuilder.build()`](../src/arlab/core/context.py) | 当前上下文来自哪些事实？被移出视图的内容如何查回？ |
| 3 | [`enabled_tools()` / `decision_schema()` / `authorize_action()`](../src/arlab/core/capabilities.py)，再看 `Harness._execute()` | 模型可以选择什么？真正的权限检查在哪里？ |
| 4 | [`TrialService.run()`](../src/arlab/core/experiments.py) | 一次实验怎样建立身份、选择恢复点、计费和独立评价？ |
| 5 | [`Store`](../src/arlab/core/store.py) | 计划、实验事实、反馈、笔记与预算分别怎样保存？ |

`Harness.run()` 的注释按“恢复—决策—执行—保存—结束”标识；`TrialService.run()` 按原有七段执行顺序标识。只是导航，没有新加工作流或改变异常处理。

第一遍可暂时跳过 SQL 表细节、HTTP 传输、容器参数和 IQL 数学代码。不是删掉这些能力，而是先理解它们为何被调用。

## 第二遍：追一次有状态实验

从 `run` 或 `continue_trial` 出发：

```text
Harness._execute
 → TrialService.continue_trial（仅续训工具）
 → TrialService.run
 → core/checkpoints：training_binding / series_binding
 → TrialService._worker
 → tasks/iql/trainer 或 algorithms/runtime
 → 独立评价 worker
 → TrialService 提交 TrialResult
 → Harness 保存 Feedback
 → 下一轮 ContextBuilder.build
```

对照[术语表](glossary.md)追踪 branch、candidate、series、trial 和 attempt。不要把同一轨迹的多个端点当成独立训练，也不要把有检查点等同于无需恢复预算。

## 第三遍：读完整研究和最终验证

| 文件 | 关键函数／对象 | 关注边界 |
|---|---|---|
| [`classroom/session.py`](../src/arlab/classroom/session.py) | `budget_preview`、`run_study` | 用户授权、动态开发、程序化最终确认 |
| [`tasks/iql/study.py`](../src/arlab/tasks/iql/study.py) | `apply_nine_task_protocol`、`require_nine_task_protocol` | 三任务池、九任务名单、固定确认要求 |
| [`core/panels.py`](../src/arlab/core/panels.py) | `PanelIndex` | 局部可比较与合同完整不是同一概念 |
| [`audit.py`](../src/arlab/audit.py) | `_protocol`、`audit_campaign` | 冻结方法、名义预算与有限恢复池 |
| [`classroom/reporting.py`](../src/arlab/classroom/reporting.py) | `study_report` | 开发复验、未用于搜索的任务与结论范围 |

## 按教材机制定位

| 教材主题 | 对应位置 |
|---|---|
| 规划：分解与反馈式调整 | `core/schema.py` 的计划／假设动作；`Harness._execute` 与 `AgentConfig` |
| 记忆：构建、管理与访问 | `core/store.py`、`remember`／`recall`；`ContextBuilder` 的笔记选择 |
| 工具：定义、选择、执行 | `core/tool_catalog.py`、`schema.py`、`capabilities.py`、`Harness._execute` |
| 反馈：来源、使用、终止 | `Feedback`、`core/diagnostics.py`、`panels.py`、`Harness.run`／`finish` |
| 上下文：选择、按需装配、压缩 | `core/context.py`、`reading.py`、`paging.py`、`research/service.py` |
| 驾驭：循环、权限、持久化 | `Harness.run`、`capabilities.py`、`Store`、`checkpoints.py` |
| 评测：轨迹和最终状态分开 | `audit.py`、`classroom/reporting.py`、`tests/` |

教材讨论自我提升和 RSI，不意味着这个具体范例必须实现它们。本版本刻意不包含 RSI。

## 最后再读领域与运行设施

固定 IQL 在 [`tasks/iql/trainer.py`](../src/arlab/tasks/iql/trainer.py)；开放候选的协议、运行检查和策略执行在 `algorithms/`。调研传输与资料提取在 `research/`。安装、下载、容器和学生入口在 `deployment/` 与 `classroom/`。

当前源码保留原函数名、参数、控制流和文件布局。阅读整理不增加新服务、不改变模型工具、不改网络与 in-sample 权限。代码“如何运行”与文档“建议按什么顺序阅读”是不同层次。
