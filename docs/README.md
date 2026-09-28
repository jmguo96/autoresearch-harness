# 当前安装版：v0.6.3 uv-local

先读 [快速开始](quickstart.md)、[项目内安装合同](project-local-install.md)和[本轮 Review](review-uv-local.md)。
历史文档中的 Docker / apt 命令不是本版安装方式。

# 文档目录：当前协议与历史记录分开

研究逻辑保留 v0.6；当前包为 **v0.6.3 Kimi＋uv 本地安装版**。第一次阅读不需要从部署器或全部算法文件开始。

## 当前文档

| 读者要做的事 | 文档 |
|---|---|
| 只填 Key、检查 Kimi 连接 | [Kimi API 说明](kimi-api.md) |
| 启动、看结果、恢复同一研究 | [快速开始](quickstart.md) |
| 按教材理解运行全过程 | [驱动范例](driver-example.md) |
| 找到关键函数与事实来源 | [源码阅读指南](reading-guide.md) |
| 对齐分支／修订／轨迹／端点等名词 | [术语表](glossary.md) |
| 看模块边界、状态与权限 | [架构](architecture.md) |
| 看模型输入与按需读取 | [上下文](context-design.md) |
| 看三任务开发、冻结和九任务确认 | [研究协议](nine-task-study.md) |
| 看低成本探索、真正续训与高预算 | [自适应研究](adaptive-research.md) |
| 看论文与参考代码工具 | [调研工具](research-tools.md) |
| 看固定 IQL 与 in-sample 定义 | [算法说明](iql.md) |
| 编写开放候选算法 | [开放算法 API](open-algorithm-api.md) |
| 管理底层 uv 安装、依赖与数据 | [部署参考](deployment.md)、[D4RL 参考](d4rl.md) |
| 检查本次接入改动与旧运行边界 | [uv-local Review](review-uv-local.md)、[迁移说明](migration-uv-local.md) |
| 查原先引用的论文与接口 | [参考来源](references.md) |

本文档以源码中的合同、动作定义与执行条件为依据。资料中的预算示例不是要求花完的调度计划，合成演示不是 D4RL 结果。

## 历史材料（不作为当前操作说明）

`README-*-history.md`、`review-v*.md`、`deployment-review.md`、`migration-0.*.md` 保存当时的记录。标题或正文中的版本、测试数、任务池和限制属于相应版本，不应覆盖当前协议；这些文件顶部有历史提示。

[原 v0.6 Review](review-v0.6.md)记录本次整理前的代码修复；[迁移到 v0.6](migration-0.6.md)描述旧功能版本之间的变化。本地安装版没有再次修改这些研究逻辑。旧阅读整理与等价证明属于历史材料。

新增：[原生库无 root 补齐](native-libraries.md)、[本轮修复审查](review-native-local.md)。
