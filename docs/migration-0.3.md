> 历史记录：本页保存所标版本的内容，不是本次阅读整理版的操作或验收说明。当前入口见[文档目录](README.md)，本次变更见[整理核对](review-readability.md)。

# 0.2 → 0.3

本次新增部署与下载，不改变固定 IQL、离线数据转换语义、开放算法接口或 RSI 目标。
代码版本参与实验身份，因此用新的部署目录和 campaign，不在旧 campaign 中混入新协议。

## 行为变化

- `deploy.sh` 默认 D4RL，只有 `--profile smoke` 使用合成任务。
- `prepare-d4rl` 保留参数，新增转换回执和严格已有文件校验；损坏或错配时不静默覆盖。
- `download-d4rl` / `scripts/download_d4rl.py` 批量准备 12 任务；`init-d4rl --suite12` 仍只是配置。
- `doctor --d4rl-env` 现在实际执行 reset 和 step。
- `init-d4rl` 新增 `--device`，CPU 模式不再把 container_gpu 标为 true。
- 进程传递受控 MuJoCo/Docker 变量，并改为先合作式退出再强制终止。
- 已有 source candidate 与 in-sample 学习边界保持；安装权限没有开放给研究者或改进器。

## 旧数据

旧 NPZ 不被一键部署无条件信任或覆盖。可在新的 data 目录从原始 HDF5 重新导入；
新回执需要匹配原始内容、任务身份、转换语义和轨迹链接。可用 `--raw-dir` 复用已取得的文件。
已失去原始 HDF5 的情况不能凭空补上来源证据；原有手工训练接口仍可使用符合其合同的数据。

## 安装包

一键部署入口在源码分发包中，使用完整 ZIP，而不是仅安装 Wheel。
Wheel 包含 Python 运行时与下载命令，但不包含所有根目录脚本和构建上下文。
