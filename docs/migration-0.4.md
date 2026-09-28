> 历史记录：本页保存所标版本的内容，不是本次阅读整理版的操作或验收说明。当前入口见[文档目录](README.md)，本次变更见[整理核对](review-readability.md)。

# 0.3 → 0.4

这是行为与合同升级，不是无缝续跑旧数据库的补丁。请解压到新目录并建立新 campaign / 部署 root。
旧源码、实验和审计保留为历史；不能把旧 completed 状态升级成新完成门禁的通过记录。

已删除 `evolution/`、SystemProgram、evolve、rsi-demo、compare-improvers 及相关配置。
普通研究配置变为 `AgentConfig`，从 `configs/agent.default.json` 读取，保存为 `agent.json`。
报告内历史兼容字段 `system` 现在是 `agent-*` 配置身份；`rsi_enabled=false`。
算法源码修改、IQL baseline、D4RL 与 in-sample value-learning 检查保留。

旧算法 recipe 可以在新运行中重新注册；旧训练结果不因 ID 相似直接复用。
旧 NPZ 仍按既有数据协议检查，多步候选继续要求 next_indices；本次未改数据转换语义。
新增 research_access、check 预算及工具 action schema 已提供示例。默认不联网，也不自动授权本机 Python。

v0.3 research review 中普通研究问题 R01/R02/R03/R07/R08 已增加修复与回归；
R04/R05/R06/R09 的 RSI 代码路径被删除，不描述为仍存在的 RSI 功能已修好。
历史部署 review 只保留历史范围；v0.4 的实际证据以 `evidence/validation.json` 和新测试为准。
