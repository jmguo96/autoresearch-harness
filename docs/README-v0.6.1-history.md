> 历史材料：原 v0.6.1 README。链接因归档位置调整，安装命令不适用于当前无 Docker 版。

# Autoresearch Harness v0.6.1：Kimi 一键接入版

一个独立运行的自动研究智能体：模型选择研究行动，稳定的 Harness（驾驭系统）负责授权、执行、反馈和持久状态。研究以固定 IQL 为唯一 baseline；候选算法源码开放，价值更新保持 in-sample。不依赖外部编码 Agent，不包含 RSI。

**以 v0.6 教材阅读整理版为基础，预置实验室 New API 地址，默认使用 Kimi。学生只填写管理员分发的 API Key。** 研究逻辑、三分布开发池、九任务确认、续训与高预算保持不变。模型接入说明见 [Kimi 接入](kimi-api.md)；旧运行不要覆盖，见[迁移说明](migration-kimi.md)。

> **HTTP 提醒：**原说明提供的是明文 HTTP，Key 和研究内容不受 TLS 加密。客户端仅对该准确实验室地址开放例外；请在管理员认可的受保护网络/加密隧道内使用，建议管理员提供 HTTPS。

## 开始使用

在教师准备的 Linux x86_64／对应 WSL2 环境中，进入源码目录：

```bash
bash start.sh
```

默认路径准备完整九份 D4RL 数据，展示预算并取得授权，只提示隐藏输入实验室 API Key。程序通过 `/v1/models` 选择当前 Key 可见的 Kimi，并在研究开始前固定准确 ID；不猜型号、不回退其他模型。密钥隐藏输入，不写入配置。宿主需 Bash、Python 3.10+，默认源码执行需管理员预装 Docker；GPU 驱动也由管理员准备。目标环境是否就绪以实际检查结果为准，不保证所有干净机器都能安装成功。

```bash
bash start.sh check-api    # 可选：安装前只验证模型列表；不下载数据、不调用推理
bash start.sh report       # 查看状态与报告，不调用模型
bash start.sh              # 同一源码版本中继续同一研究，不重置预算
```

论文和公开仓库调研不要求额外搜索密钥；通用网页搜索默认关闭。网络失败不会替换成虚构资料，D4RL 失败不会退回合成环境。[学生操作说明](quickstart.md)给出教师准备、离线资料和恢复方式。

## 研究主线

```text
固定任务合同与权限
  → 模型选择调研、代码修改或低成本实验
  → 根据真实反馈选择续训、扩任务、加种子或停止方向
  → IQL 与候选完成三任务开发确认，冻结一个方法
  → 程序独立完成九任务复验，生成有明确范围的结果
```

开发池是 `halfcheetah-medium-v2`、`hopper-medium-replay-v2`、`walker2d-medium-expert-v2`。新方向可先做一个任务、一个种子、约 5,000 步和 3 个评价 episodes；这些是起步建议，不是晋级阈值。模型可以选择后续投入，但不能提前使用另外六项选方法。

最终条件不缩减：开发冻结要求三任务、确认种子 0/1/2、百万步和 10 个评价 episodes；九任务确认使用五个新种子，每个任务重新训练固定 IQL 与冻结方法。高预算是授权上限，不是必须花完的额度。完整参数与成本口径只在[研究协议](nine-task-study.md)和[自适应实验说明](adaptive-research.md)展开。

## 第一遍只读这条路径

| 阅读材料 | 要回答的问题 |
|---|---|
| [教材驱动范例](driver-example.md) | 规划、记忆、工具和反馈如何接成一次研究？ |
| [源码阅读指南](reading-guide.md) | 应从哪几个函数进入，而不是顺读全部工程文件？ |
| [术语表](glossary.md) | 分支、修订、训练轨迹和实验端点有什么区别？ |
| [架构与边界](architecture.md) | 哪些判断交给模型，哪些由运行程序执行？ |
| [上下文组织](context-design.md) | 模型这一轮看见什么，详细证据如何查回？ |

保留阅读整理版的核心注释和阅读顺序；Kimi 连接代码单独位于 `classroom/kimi.py` 与 `core/endpoints.py`，不进入实验策略或算法实现。所有当前文档及历史材料入口见[文档目录](README.md)。

## 本地机制演示

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,research]'
python examples/adaptive_workflow.py --output runs/adaptive-example
pytest -q
```

示例使用**显式脚本控制器和合成连续控制环境**。训练、续训与评价真实执行，行动顺序预先安排；它用于理解机制，不是大模型自主研究轨迹，也不是 D4RL 成绩。正式研究仍从 `start.sh` 进入。

## 本接入版如何验收

[接入 Review](review-kimi.md)和[当前验收记录](../evidence/history-v0.6.1/validation.json)记录本轮实际测试；[研究逻辑完整性对比](../evidence/history-v0.6.1/research-integrity.json)列出未改变的实现。原阅读整理版记录存于 `evidence/history-readable/`。

本次没有真实实验室 Key，没有进行网关鉴权和 Kimi 实际推理，也没有新的 D4RL、MuJoCo、CUDA、Docker 或在线资料服务验收。协议测试与合成实验不是算法改善证据。`check-api` 只检查模型列表，不保证推理渠道可用或额度充足。
