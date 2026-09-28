> 历史记录：本页保存所标版本的内容，不是本次阅读整理版的操作或验收说明。当前入口见[文档目录](README.md)，本次变更见[整理核对](review-readability.md)。

# Autoresearch Lab 0.4：先调研、再实验，保留证据而不是堆叠上下文

独立的计算研究智能体：**稳定 Harness + 规划 / 工具 / 反馈 / 记忆 + 开放算法源码研究**。
唯一交付的领域任务包是 offline RL：固定 IQL baseline，D4RL 为正式研究环境。
不依赖 Codex、Claude Code 或其他编码智能体运行时；**已移除 RSI 模块、命令和自修改发布流程**。

## 这次改变了什么

- 当前上下文是数据库的有限视图，不再重复发送完整 trial 账本、候选源码、baseline 配方、配方 schema 与事件历史。
- 默认 28,000 字符状态视图；稳定指令和动作 schema 另外发送一次，总模型文本另有 70,000 字符检查。
  **字符不是 Token**；不承诺某个模型的实际输入 Token 数。
- 新增论文检索、公共网页搜索、HTML / 文本 / PDF 文本获取、正文分页定位、GitHub 仓库发现与 commit 固定的代码阅读。
- 新增独立候选自检、分页实验查询、合同覆盖率比较。自检有真实训练与评价，但不能冒充正式研究成绩。
- 修复对照缺失仍能结束、相同缺失面板被视为完整、只看最短训练排序、审计优化器预算漏配等普通研究逻辑。
- 网络结构、价值目标、策略分布、训练顺序与完整候选源码仍可修改；**没有恢复固定 IQL 骨架的限制**。

## 安装与轻量运行

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,research]'

# 教学决策器 + 真实 IQL / 开放候选训练与独立合成环境评价。
arlab open-demo --output runs/open --steps 12 --candidates 3

# 明确的离线 HTTP 样本 + 真实训练，展示调研到实验的完整工具链。
python examples/research_workflow.py --output runs/reading --steps 4 --offline-fixture

# 查看下一轮上下文，不调用模型。
arlab context runs/open/campaign
pytest -q
```

`open-demo` 和 walkthrough 的决策顺序都是脚本化的，不是大模型自主发现。
`--offline-fixture` 的论文与仓库回应也明确是样本，**不是下载的论文或 CORL 源码**。
训练与 rollout 真正执行，但合成环境不是 D4RL，小运行不用于性能结论。
walkthrough 不加 `--offline-fixture` 才会尝试真实网络；失败不会退回假结果。

## 一键准备 D4RL 与在线调研

主安装路径为 Linux x86_64 的 Ubuntu/Debian/相应 WSL2，需要 Bash 和 Python 3.10+。
脚本创建私有 Python 3.10 研究环境，保留 legacy D4RL 协议，不静默切换到 Minari。

```bash
# 不下载、不安装、不授权执行模型生成的代码。
bash deploy.sh --dry-run --research-online

# 单个 Hopper medium-v2：数据、依赖、配置与小规模自检。
bash deploy.sh --yes --research-online

# 12 个任务 + 容器内的开放源码研究。Docker / 驱动需用户先准备好。
bash deploy.sh --yes --suite12 --research-online --code-mode docker \
  --root "$HOME/arlab-v04"
```

安装不会调用大模型，也不会自动启动大规模研究。

```bash
export ARLAB_API_KEY='你的模型服务密钥'
# 以下两项按需配置，不是论文搜索必需的模型密钥。
export BRAVE_SEARCH_API_KEY='你的 Brave Search API 密钥'   # 使用 search_web 时需要
export ARLAB_GITHUB_TOKEN='只读公开资源用途的 GitHub token' # 可选

bash "$HOME/arlab-v04/start-research.sh" --dry-run
bash "$HOME/arlab-v04/start-research.sh" --confirm-budget \
  --model '你的模型名称' --base-url 'https://你的模型服务地址/v1'
```

在线调研授权与候选代码执行授权分开。默认不联网；`--research-online` 授权受限公共 HTTPS。
新增出版商域名由用户通过 `--research-host exact.example.org` 明确批准，不由资料正文或模型自行扩权。
`trusted_local` 只用于已审阅代码，不是文件系统沙箱；模型生成代码应使用另行配置的 Docker 环境。
具体限制和目标环境验收范围见 [部署说明](deployment.md)。

已有合同也能交给真实模型；创建新运行，不覆盖旧状态：

```bash
arlab run runs/open/campaign/contract.json --output runs/llm-research \
  --provider compatible --model '你的模型名称' --base-url 'https://你的服务地址/v1' \
  --research-online --agent-config configs/agent.default.json \
  --code-mode docker --code-image '你的已构建 worker 镜像 ID'
```

## 工具是研究行动，不是无限权限的 Shell

| 类别 | 已实现工具 |
|---|---|
| 获取知识 | `search_literature`, `search_web`, `fetch_source`, `read_source`, `find_source` |
| 查阅实现 | `search_repositories`, `read_repository`, `inspect` |
| 建立与修改候选 | `propose`, `edit_algorithm`, `check_candidate` |
| 实验与分析 | `run`, `query_experiments`, `compare` |
| 组织与积累 | `plan`, `recall`, `remember`, `finish` |

搜索返回不等于读过全文。正文读取产生 `span-*`，关联 `src-*`、URL、内容哈希、行位置、读取范围；
GitHub 文件还记录 commit。记忆与假设可引用这些 ID，也可引用真实 trial ID。
来源存在不是语义正确性证明；“作者声称”和“本项目已复验”需要分别表达。
工具协议、参数示例与服务配置见 [调研工具](research-tools.md)。

## 上下文与代码阅读顺序

```text
core/config.py / schema.py     稳定研究指令、用户配置、动作协议
core/context.py               有限状态视图、去重、按需读取与显式裁剪
core/panels.py / stats.py      实验覆盖率、同档位排名与配对比较
core/reading.py / paging.py    长文件、日志、反馈和实验的分页
core/harness.py                决策 → 授权工具 → 真实反馈 → 持久状态
research/service.py           调研流程与不可变证据
research/network.py           公共 HTTPS、域名 / 地址检查、额度和超时
core/experiments.py            自检与正式实验隔离、训练 / 评价和恢复
algorithms/                   开放算法协议与 value / policy 边界
 tasks/iql/                   固定 baseline、D4RL 与独立评价
 audit.py                     用户调用的封存审计，不向研究者开放
```

[教材驱动范例](driver-example.md) · [上下文设计](context-design.md) ·
[开放算法 API](open-algorithm-api.md) · [本轮 Review](review-v0.4.md) ·
[0.3 → 0.4 迁移](migration-0.4.md)

## 验证范围

本次自动测试 **228 passed、1 skipped**。新增调研测试使用可标识的 HTTP fixtures、DNS/TLS
测试替身及错误注入；完整流程另外实际执行了 CPU IQL / 候选训练与合成环境 rollout。
容器中的真实 arXiv / Crossref / GitHub 请求均因 DNS 失败，见 `evidence/live-research-probe.json`。
**没有把这些测试算成真实网络服务、模型 API 或 D4RL 性能验收。**

未运行：干净机器联网部署、真实 D4RL / MuJoCo、CUDA、Docker、真实模型自主研究。
不存在本项目“优于 IQL”的成绩声明。来源解析不包含视觉读图、OCR、付费墙绕过或 JavaScript 浏览器。
运行限制与已知边界见 [架构](architecture.md) 和 `evidence/validation.json`。
