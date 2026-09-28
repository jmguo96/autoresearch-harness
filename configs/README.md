# 配置：生成的研究合同与类型默认分开

学生通过 `bash start.sh` 使用按真实数据生成的合同，不需要复制 JSON 或逐项设置任务和种子。

| 文件 | 用途 |
|---|---|
| `agent.default.json` | 稳定研究者指令和上下文配置；不含密钥，不是模型可修改资产 |
| `decision.schema.json` | 完整持久化动作协议；运行时按权限生成可用子集，不等于所有动作始终开放 |
| `search-iql.action.json` | 用户调试 `arlab research CAMPAIGN --action FILE` 的一个例子 |
| `iql.baseline.json` | 固定 IQL 对照 |
| 其它配方与源码模板 JSON | 可作为候选起点，不代表算法收益 |

当前工具目录含 21 类动作。默认在线、源码授权的学生协议关闭 `search_web`，因此实际目录少于完整 schema；离线或不同源码权限会进一步改变可用集合。菜单、模型 schema 与执行授权都来自 `core/capabilities.py`。本次未修改任何 JSON 的内容。

`ResearchAccess` 类型默认为不开网络；学生主入口会按预设启用免额外搜索 Key 的论文与公开代码调研。底层 `arlab` 新合同可通过 `--research-online` 明确授权，额外域名需用户批准。源码执行是另一项权限。

v0.6 学生合同由 `tasks/iql/study.py` 与 `core/presets.py` 生成：三任务开发池、自适应累计端点、高预算、固定九任务 final。不要把 `Budget` 类的通用默认值当成学生主预设，也不要复制假的已验证数据哈希。没有 RSI 配置。

全部数字和研究条件见[研究协议](../docs/nine-task-study.md)与[自适应说明](../docs/adaptive-research.md)。
