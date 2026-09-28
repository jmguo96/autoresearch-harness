> 历史记录：本页保存所标版本的内容，不是本次阅读整理版的操作或验收说明。当前入口见[文档目录](README.md)，本次变更见[整理核对](review-readability.md)。

# 0.5 → 0.5.1：三种数据分布共同探索

默认开发面板从单个 hopper-medium-v2 改为 halfcheetah-medium-v2、hopper-medium-replay-v2、walker2d-medium-expert-v2。最终仍为完整九项，用新种子重新训练；报告从 1+8 改为 3+6，并单列两组统计。

这是研究协议变化，不是给旧运行补两条记录。源码、合同和数据身份受绑定保护；不要原地覆盖旧文件后继续旧 Campaign。保留旧代码和运行目录，在新解压目录运行 `bash start.sh`。若曾用 --root 指定路径，应换新路径，例如：

```bash
bash start.sh --root "$HOME/arlab-three-distributions"
```

原始下载的 D4RL 数据可以按既有下载器的校验规则在新部署中复用，但不能重用旧实验分数来冒充新的开发或最终面板。模型服务地址和名称可以重新填写，密钥仍不落盘。旧模型提示、训练结果与最终成绩不在本次补丁中迁移。

部署配置复用会比较 case.split；单任务或全 medium 的旧配置与本协议冲突时拒绝覆盖，并保留建议配置供 owner 查看。课堂入口显式要求固定三分布面板，普通底层 Harness 仍支持其它明确建立的合同；这不是重新限制候选算法源码。

RSI 仍移除，IQL 仍为唯一 baseline；没有改变 in-sample 的训练边界，没有新增搜索密钥要求。旧版验收仅在 evidence/history-v0.5 归档，本版结果见 evidence/validation.json。
