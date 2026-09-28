> 历史记录：本页保存所标版本的内容，不是本次阅读整理版的操作或验收说明。当前入口见[文档目录](README.md)，本次变更见[整理核对](review-readability.md)。

# 迁移到 v0.6

请解压到新源码目录，并使用新的部署 root / study 名称。不能用 v0.5.1 的 completed 或 frozen 标签跳过新研究协议；旧执行不被升级成新证据。

旧版总训练量参与学习率日程；新版自适应实验的暂停端点与固定日程分开。旧 checkpoint 没有经过验证的 series identity，不作为新流程的续训来源。权重可作人工研究材料，但不能静默作为原实验续训。

`max_candidates` 现在是活跃假设分支上限，`max_revisions` 是不可变版本上限；`propose(new_branch=false)` 和默认源码编辑保留原方向。首次从 baseline 修改仍是新方向，baseline 不可归档。

`continue_trial.steps` 是累计步数，非增量；`run` 自动使用最高兼容端点。正式最终审计始终独立重训，不能继承开发权重。

历史 evidence 保存在 history-v0.5.1，新结果以当前 validation.json、测试日志与代码哈希为准。学生接口及免额外密钥的调研能力不变。
