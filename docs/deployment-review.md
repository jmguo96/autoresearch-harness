> 历史记录：本页保存所标版本的内容，不是本次阅读整理版的操作或验收说明。当前入口见[文档目录](README.md)，本次变更见[整理核对](review-readability.md)。

# v0.3 一键部署：代码 Review 与验收范围

## 结论与方法

最终验收：**172 项通过、1 项因缺少 legacy D4RL 依赖跳过，0 失败**。

本次在 v0.2.0 源码上实现部署功能，并对新增模块及受影响的进程、配置、下载路径进行
同轮自审。不是第三方审计，也不是对任意 Python 或供应链的形式验证。

Review 依据为实际源码、故障注入测试、原有回归测试、真实的合成任务训练/评价，以及
独立安装包检查。测试数量、解释器、依赖版本和源码哈希以 `../evidence/validation.json`
为准；不把历史 evidence 当作本版本结果。

**核心限制：本环境不能联网安装依赖，没有 legacy D4RL/MuJoCo、CUDA 或 Docker。
因此未验收干净 Linux 机器上的完整联网安装，也未取得真实 D4RL 基准结果。**
已经提供 owner 手动触发的 CI 工作流，但未将“有 CI 文件”写成“CI 已通过”。

## 范围

- `deploy.sh`、`scripts/deploy.py`、`scripts/download_d4rl.py`。
- `src/arlab/deployment/` 的安装编排、HTTPS 下载、tar 解包、状态、配置和启动门禁。
- `cli.py` 的 prepare/download/init/doctor 接口。
- `core/jobs.py`、`algorithms/policy.py`、worker 的环境与退出路径。
- `requirements/`、Docker 构建、部署文档与新测试。
- 原有 IQL、开放算法、记忆、规划和 RSI 由全套测试做回归；本次没有重新设计科研方法。

## 已发现并修复的问题

这里的 P1/P2 是本次代码 Review 的优先级，不是外部漏洞评级。

| 编号 | 优先级 | 问题与影响 | 修复位置与验收 |
|---|---|---|---|
| R01 | P1 | 初始化合同固定 CUDA，CPU 部署可能到正式训练才失败 | `manager.preflight` 选择设备；`tasks.check` 真正运行张量；`tasks.make_contract` 与 CLI 显式写入 device/container_gpu。CPU 合同回归通过。 |
| R02 | P1 | MuJoCo 无头变量在子进程过滤中丢失，主进程检查通过不代表 worker 可执行 | `LocalBackend.run` 传递受控 MuJoCo/D4RL 变量，仍过滤模型密钥。对应子进程测试通过。 |
| R03 | P1 | 策略子进程重置 HOME 后丢失 rootless/自定义 Docker 上下文 | `policy_process_environment` 保留 owner Docker 客户端配置；不将这些变量通过 `-e` 传入候选容器。本机模式不继承 Docker 连接变量。单元测试通过；真实 daemon 未验收。 |
| R04 | P1 | 下载中断没有恢复，错误追加可能把两个实体拼接 | `fetch_https` 仅在有实体校验条件时续传，严格检查 Content-Range/ETag；200 重写、416 重启、暂时错误有限重试。注入 HTTP 故障测试通过。 |
| R05 | P1 | 已有数据可能损坏、任务不匹配或转换语义不同 | 原始/转换哈希、数据维度、environment、conversion 和显式轨迹链接校验；错配报错，不覆盖科研数据或切换合成数据。离线 HDF5 样本测试通过，样本不冒充官方数据。 |
| R06 | P1 | 自动解包可能路径穿越，安装中断可能留下无回执目录 | `extract_mujoco` 手动验证成员、大小、路径及链接；完整目录与安装回执一并原子发布。恶意 tar 与合法内部链接测试通过。 |
| R07 | P1 | 为方便运行而默认开放本机 Python 或顺便启动昂贵研究 | 默认 `code_mode=disabled`；本机需额外确认，Docker 不可用不降级；`launch` 检查 ready、源码/数据哈希、模型、凭据及 `--confirm-budget`。实际启动拒绝测试通过。 |
| R08 | P2 | 重复部署覆盖用户预算、模型或数据配置 | 保留相容的 owner 修改；冲突保存 `research.proposed.json` 后失败，不改现有合同。重复部署测试通过。 |
| R09 | P2 | git 分支重试漂移或依赖手工升级被静默接受 | 首次记录确切 commit，后续不 pull；检查 tracked 变化及未跟踪可执行代码；重用时核对主要依赖版本。完整跨机器锁定仍有边界，见下文。 |
| R10 | P2 | 直接 SIGKILL 可能使策略清理 finally 不执行 | 分层执行使用 SIGTERM、宽限等待、最后才 SIGKILL；合作式 worker 可清理策略子进程；Docker 保留强制删除逻辑。真实本地超时清理测试通过，不保证任意敌意程序的清理。 |
| R11 | P2 | 跳过训练检查仍可能被当成部署成功 | `--skip-smoke` 写 `prepared_unverified` 而非 ready；启动器拒绝。失败/中断写明步骤，不把安装成功当成环境评价成功。 |
| R12 | P2 | 预览命令也可能写 Python 字节码 | shell 使用 `python -B`，入口设置 `dont_write_bytecode`；dry-run 不执行子命令、不创建部署目录、不下载、不 sudo。 |

## 验收分层

### 已实际运行

完整测试包含原有 Harness/算法/RSI 回归，以及新增的下载、安装编排和权限测试。
独立运行记录包含真实执行的 `deploy.sh --profile smoke --reuse-runtime ...`、生成 CLI 的
`doctor`、研究预算预览和缺少预算确认时的拒绝。轻量部署确实运行 IQL、开放候选训练、
策略导出和独立 rollout；没有调用远程大模型。

`--reuse-runtime` 明确复用已有 Python/依赖，报告写明
`skipped_owner_supplied_runtime`。它证明部署控制流和工件连接能够运行，
**不证明本次执行了 pip/uv/Git/MuJoCo 下载，也不证明 Python 3.10 的兼容依赖组合已装成功**。

下载测试用可注入的 HTTP 响应覆盖断连、范围错配、实体变化、404/503、校验失败、
缓存损坏、超限和并发锁。它们证明相应代码行为，不证明官方服务器当前可达。

### 尚未运行，不能算通过

| 项目 | 原因 / 应如何补验 |
|---|---|
| 干净 Linux 上完整 D4RL 一键安装 | 当前环境无联网安装能力；在目标主机执行 `bash deploy.sh --yes --device cpu`，保留所有状态和日志。 |
| 官方 HDF5 在线下载 | 没有实际请求并下载完整官方数据；需要检查服务可达、文件身份与完整转换。 |
| legacy MuJoCo 编译与真实 D4RL rollout | 当前未安装；单元测试不能替代真实环境 reset/step、训练和评价。 |
| CUDA 与 GPU 容器 | 无 GPU / Docker；需要正确驱动、NVIDIA Container Toolkit 并实际验收。 |
| Python 3.10 / Torch 2.5.1 固定组合 | 本次运行用现有 Python 3.13 / CPU Torch；目标兼容组合只完成静态核对。 |
| 真实模型接口、算法性能提升、RSI 效率提升 | 本次部署自检不访问 LLM，也不开展有效规模的算法/演化对照实验。 |

## 仍保留的风险和设计边界

1. **完整环境可复现性尚非完全锁定。** 主依赖采用固定版本，但没有所有传递依赖的
   多平台带哈希 lockfile。默认分支只在首次部署时解析并保存，两个新部署可能解析到不同
   commit。正式复现应传入经审核的完整 refs，并保存 freeze、inspect、源码和数据哈希。
2. **供应链真实性不等于内容一致性。** TLS、批准域名和回执能够减少意外变化，
   但首次计算出的哈希不是独立认证。可以传入可信数据/二进制哈希；没有的部分如实标记。
   pip/Git 构建属于 owner 批准的第三方代码执行，并非不受信任代码沙箱。
3. **操作系统支持有边界。** 主路径限定 Linux x86_64 的 Ubuntu/Debian apt 环境；
   其他平台明确失败或要求 owner 预装系统库，不自动更换 benchmark。
4. **不自动安装驱动或 Docker daemon。** 这些改变超出项目局部环境；脚本不开放 socket，
   不修改用户组，不静默退回 `trusted_local`。
5. **legacy 依赖有维护风险。** 固定旧版本的目的在于保留 benchmark API，
   不是宣称这些依赖没有安全问题。正式服务应隔离部署，并单独开展依赖漏洞审计。
6. **权限不等于算法正确性。** in-sample 的运行检查仍依赖注册接口和可观察数据流；
   任意源码研究需要审阅。Docker 隔离也不能证明数学目标或统计结论正确。
7. **不移植旧运行状态。** 实验身份绑定源码与协议；升级后新建部署目录和 campaign。
   不能把更换版本之后的结果混入旧对照。

## 教材中的呈现方式

部署代码是驾驭系统的 owner 操作入口，不应伪装为智能体自动取得的安装权限。
教材可以分别展示部署状态 `running → ready/failed/interrupted`，研究循环中的
规划—工具—反馈—记忆，以及 RSI 的候选与发布边界。它们协作，但有不同的权限和验证条件。

建议读者首先查看 `--dry-run`，然后阅读 `manager.run` 的少量阶段调用，再分别进入下载、
配置与检查模块。目录划分服务于可读性，不表示需要部署多个服务或用更多 agent。
