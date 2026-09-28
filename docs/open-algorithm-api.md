# 开放算法 API：限制更新的数据来源，不规定研究结论

## 一个候选的组成

`CodeRecipe` 保存 `kind=python`、名称、源文件字典、入口、配置和每 tick 的优化器步数上限。
源文件是不可变实验工件。源码、配置、数据、运行实现与训练种子共同决定实验身份。
注册候选时只解析 AST 和 schema，不导入、不执行它的顶层代码。
执行需要合同中的 `CodeExecutionPolicy`，不能在 recipe 中自行加入权限字段。

入口 `candidate.py:build` 收到 `BuildSpec`：观察维数、动作维数、总日程 tick、设备、
候选配置与是否为推理实例。它返回实现以下接口的对象；继承 `Algorithm` 只是便利方式。

| 对象/方法 | 职责 |
|---|---|
| `value_modules` | 登记 Q、V、target 与价值侧共享编码器。网络结构与输出形状不限于原始 IQL |
| `policy_modules` | 登记策略侧模块。不要求高斯、对数似然或单一 actor |
| `action_value_names` | 完整登记接受 `(observations, actions, ...)` 的动作价值根模块 |
| `optimizers`、`schedulers` | 以名称登记 PyTorch 对象，用于预算、归属检查和恢复 |
| `schedule(tick)` | 返回任意合法的、带名称的 value/policy 阶段序列 |
| `value_step(ctx, name)` | 自行计算价值目标与损失，调用 `ctx.optimize` |
| `policy_step(ctx, name)` | 自行计算策略目标；允许动作采样和冻结 critic 指导 |
| `act(observations)` | 返回 `[batch, action_dim]` 的标准化动作，范围由评分端再次裁剪 |
| `extra_state_dict` / `load_extra_state_dict` | 保存并恢复控制器、计数器等额外状态 |

模块名称在价值组与策略组间不能重复；参数共享可以存在。
共享参数归价值组，策略更新不得改变它们。梯度可以穿过冻结价值网络传到动作。
策略可缓存自己的推理状态，并可实现 `reset(seed)` 在新 episode 开始时重置。

接口不要求必须有两个 Q、不要求存在 V，也不要求网络输出标量。
如确实没有动作价值网络，`action_value_names` 可以为空；禁止隐藏实际存在的 Q。
网络结构在候选初始化时登记；需要改变结构时提交新候选，不在运行中绕开登记表换网络。

## 自适应端点与额外状态

在当前自适应 Campaign 中，`BuildSpec.total_ticks` 是开始前固定的 `schedule_steps`（默认百万步），不是本次 5,000 或 20,000 步的观察端点。网络、优化器、调度器、RNG 与算法显式导出的额外状态随检查点恢复。修改源码/配置、数据或 seed 后是新轨迹。

`run`／`continue_trial` 的 `steps` 表示累计目标；同端点只增加评价次数可复用训练工件。额外 Python 状态必须通过已有状态接口保存，未登记的任意全局状态不在恢复保证之内。独立 `train` 的日程默认与自适应 Campaign 不同，详见[训练协议](adaptive-research.md)。

## 目标与更新组织

下面是一个接口片段，而不是完整算法：

```python
from arlab.algorithms.protocol import Update

def schedule(self, tick):
    result = [Update("value", "representation"), Update("value", "critic")]
    if tick >= self.warmup and tick % self.policy_every == 0:
        result.append(Update("policy", "extraction"))
    return result

def value_step(self, ctx, name):
    batch = ctx.batch
    multi = ctx.data.n_step(batch, horizon=3, discount=0.99)
    # 自定义目标、分布式表示或额外 in-sample 正则项。
    # 所有 Q 调用仍必须使用登记的数据状态—动作配对。
    loss = self.compute_objective(batch, multi, name)
    ctx.optimize(loss, "critic", gradient_clip=10)
    return {"loss": float(loss.detach())}
```

目标可以是有符号的标量，不要求逐样本非负。需有限、可微，且对所选优化器有梯度。
优化器可自行选择，不固定为 Adam。直接调用已登记优化器的 `step()` 也受归属与计数 hook 检查，
但推荐用 `ctx.optimize`，它还检查损失、梯度和有限张量来源。

每 tick 的实际优化器调用数受 recipe 声明上限与 owner 上限双重约束。
不同 schedule 在相同 tick 下可能成本不同；比较时记录实际优化器步数与时间。
不得利用未登记优化器规避预算，也不能把手工参数更新伪装成免费操作。

## 数据与动作来源

`ctx.batch` 是新抽取且有来源记录的离线批次。
`ctx.data.sample(n)`、`take(indices)` 登记新的真实批次；完整配对会被独立克隆，
之后原地修改候选拿到的张量不会改写它的来源证据。

`ctx.data.successors(batch)` 返回有真实后继记录的位置与后继批次。
`n_step(batch, horizon, discount)` 沿显式后继索引累积奖励、保留 bootstrap 状态与折扣。
真 terminal 的 bootstrap 为零；数据或 time-limit 截断从真实 next observation 的 V bootstrap。
旧数据没有链接时不会猜测时间连续性。状态相似、索引相邻都不足以证明轨迹相连。

value 阶段对 action-value 根模块的查询必须匹配一个已登记批次的状态和动作。
支持位置参数与关键字参数。对子批次、重复抽样或后继样本调用 `take`，不要手动伪造配对。

policy 阶段可查询生成动作的 Q，价值参数、已有梯度和 buffers 由边界保护并在结束时核对。
这允许扩散、混合策略、梯度引导和多候选动作选择，不仅是 AWR。

## 完整依赖与检查局限

除查询时检查之外，`PolicyOnlyValue` 为生成动作估值附加 Tensor 来源标记。
常规张量运算和 `detach()/clone()` 保留标记；带标记的目标不能进入 `ctx.optimize` 的 value 阶段。
这能拦截常见的“上一轮策略 Q 缓存 → detach → 下一轮价值目标”误用。

它不是完整信息流证明。经 Python 数值、NumPy、自定义扩展或显式去除子类可能洗掉标记；
隐藏网络或绕过受检 API 也会破坏检查前提。测试说明哪些路径被验证，审查负责检查声明是否完整。
科学报告应称“登记接口上的运行时不变量通过”，而非“任意代码已被证明 in-sample”。

## 通用策略工件

导出包括源码及配置、所有登记模块的 safetensors、额外状态和标准化参数。
评价端只验证结构和内容哈希，不导入候选 Python，也不读取训练 pickle。
候选推理 worker 通过 JSON 接收标准化 observation，返回 action；奖励、环境状态和正式评分留在父进程。
源码没有一条合法路径可以“返回一个分数”取代真实 rollout。

本版使用 uv 私有环境中的本地推理进程，不是宿主文件权限或网络隔离，也不启用 Docker。
候选按协议只接收 observation，但恶意代码可绕开协议访问当前用户有权访问的文件；必须审阅。
动作返回需满足形状和有限值要求；请求具有启动/逐动作超时与消息大小上限。
候选的真实推理开销单独记录。
