# Autoresearch Harness 0.6.3：实验室 New API，项目内 uv 安装

**不使用 Docker、不执行 sudo、不向系统安装包、不改系统 Python 或 Shell 配置。**
缺少 GL/OSMesa/OpenMP 库时，自动使用独立 APT 索引解析同发行版 deb，下载后仅解包到项目。
APT 只运行私有索引更新、依赖模拟与元数据/下载地址查询；不运行真正的安装、删除或升级。
下载、私有 Python、虚拟环境、缓存、临时文件和研究结果默认全部放在这个源码目录的
`.arlab/d4rl9/` 下。`--root` 只能选择源码目录内部的子目录，不能指向另一个项目。

## 学生使用

在解压后的目录运行：

```bash
bash start.sh --dry-run   # 可选预览：不写入、不下载、不安装
bash start.sh
```

安装和研究支出分别确认。新部署默认从实验室 New API 的 `/v1/models` 选择并固定
Kimi；已有研究按原合同固定模型继续。学生只填管理员分配的 **API Key**，隐藏输入、不落盘。

```bash
bash start.sh setup       # 只安装、下载九份数据并做小规模自检，不调用 Kimi
bash start.sh check-api   # 检查当前配置模型是否在 Key 的模型列表中，不安装、不训练
bash start.sh report      # 查看进度和报告
bash start.sh             # 继续原研究；不会因重启增加预算
```

**实验室默认网关使用明文 HTTP。** Key 和研究内容的这一段传输没有 TLS 保护；应使用
管理员认可的加密隧道或 HTTPS 服务。不落盘不等于传输安全。详见 [Kimi 接入](docs/kimi-api.md)。

## 安装方式与边界

安装器把固定版 uv 放到项目目录，然后实际调用：

```text
uv python install --no-bin 3.10.19
uv venv --no-project --python <项目私有 Python> <项目目录>/.arlab/d4rl9/venv
uv pip install --python <项目 venv/bin/python> <固定依赖>
```

机器上已有匹配版本的 uv 时只复制其可执行文件；没有则从固定版本的 PyPI wheel
提取 uv，校验注册表提供的 SHA256。不运行在线 Shell 安装脚本，不写 `~/.local/bin`。

项目预先配置 HOME、XDG、uv/pip、PyTorch、CUDA、Matplotlib、Hugging Face 的缓存与临时目录。
环境设置只作用于项目进程及子进程，不修改 `.bashrc`、`.zshrc`、系统 PATH 或其他项目环境。
当前终端已经激活的 venv/Conda 不作为安装目标；每条安装命令都明确指定本项目 venv。

**系统前提：Linux x86_64（包括相应 WSL2）、Bash、启动用 Python 3.10+、Git 和可用的
C/C++ 编译器。** 缺少 GL/OSMesa/OpenMP 运行库不再立即停止：支持 Debian 11/12/13 与
Ubuntu 20.04/22.04/24.04 的匹配发行版 deb。使用宿主已有 apt-get、apt-cache、dpkg-deb
和可信发行版密钥环，索引、deb、解包库、日志全部放在项目。不会更新 `/var/lib/dpkg/status`，
不执行维护脚本或 ldconfig，不把下载的 glibc/动态加载器加入进程搜索路径。
编译器本体缺失、不支持的系统、联网/签名失败或 ABI 验证失败仍会明确报错；不是任何裸系统
的通用引导器。细节见 [无 root 原生库补齐](docs/native-libraries.md)。`patchelf` 仍由 uv 安装。

## 重要：uv 虚拟环境不是安全沙箱

源码候选使用 `trusted_local` 子进程执行。程序会说明风险，并在研究预算确认中要求
用户同意本地代码执行。**不能声称任意模型生成代码都无法读取或写入其他项目，
也不能声称候选无法联网。** 缓存重定向和虚拟环境是协作式约定，不是操作系统权限隔离。
请审阅生成代码，不以 root 运行，不用含敏感资料的账号执行未经审阅的候选。
确需强隔离时，这个无容器版本本身不提供相应保证。

## 研究内容不变

固定 IQL 为唯一 baseline；无 RSI。论文、指定网页和公开代码调研仍免额外搜索密钥，
通用网页搜索默认关闭。候选保持源码级开放，Q/V 更新不使用数据外动作估值。

开发池为 `halfcheetah-medium-v2`、`hopper-medium-replay-v2`、`walker2d-medium-expert-v2`。
可以先做单任务/单种子/短训练，再由模型根据证据续训、扩任务或加种子。冻结前必须完成
规定的三任务开发对照，最终仍是九任务 × 五个新种子的独立确认，完整预算没有缩减。

## 目录与阅读

```text
.arlab/d4rl9/
├── tools/uv/bin/uv     项目专用 uv
├── python/            uv 下载的 Python
├── venv/              uv 创建并安装依赖的虚拟环境
├── downloads/         uv 元数据/wheel、MuJoCo 等下载
├── sources/           固定 commit 的 D4RL/MJRL
├── mujoco/            MuJoCo 二进制
├── data/              九份原始 HDF5、转换数据和回执
├── cache/             uv、pip、调研限流及框架缓存
├── home/, config/, share/, state/, tmp/   项目进程私有目录
├── native/            自动 deb 前缀、私有 APT 索引与手动组件
├── configs/, attempts/, checks/          配置、安装日志和检查结果
└── runs/study/        研究、最终评价与报告
```

使用项目生成的 `bin/arlab` 包装器，而不是在日常终端中 `source env.sh`。
不要把另一项目的缓存/venv 用符号链接挂到这些管理路径，安装器会拒绝。

- [快速开始](docs/quickstart.md)与[本地安装说明](docs/project-local-install.md)
- [部署细节](docs/deployment.md)与[D4RL 前提](docs/d4rl.md)
- [教材驱动范例](docs/driver-example.md)与[源码阅读路径](docs/reading-guide.md)
- [自适应研究](docs/adaptive-research.md)、[九任务协议](docs/nine-task-study.md)
- [本轮 Review](docs/review-native-local.md)、[验收记录](evidence/validation.json)

本版修改安装和本地执行环境，不声称已经完成真实 Kimi、完整 D4RL 或算法性能验收。
旧版 Docker/系统安装说明属于历史；旧研究目录不要原地转换为本版。
