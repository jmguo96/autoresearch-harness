# 学生快速开始：Kimi＋项目内 uv

在教师准备的 Linux x86_64／WSL2 机器上，解压并进入项目目录：

```bash
bash start.sh --dry-run   # 可选：预览，无下载或写入
bash start.sh
```

不用 Docker，不用 sudo，不用手动激活 venv。安装器把 uv、Python、依赖、数据和缓存
放到当前源码目录的 `.arlab/d4rl9`。默认 Kimi，学生只需输入实验室管理员分发的
API Key；地址预置，准确模型 ID 从 `/v1/models` 查询后固定。

第一次启动确认本地安装，再确认完整研究支出及本地执行候选 Python 的风险。
Key 隐藏输入，仅传入主研究进程，不写配置，不向安装/训练/策略进程传递。
网关当前是明文 HTTP，需管理员认可的网络/加密隧道或 HTTPS，详见 [API 说明](kimi-api.md)。

```bash
bash start.sh setup       # 教师预装、自检，不消耗模型额度
bash start.sh check-api   # 单独查询模型列表
bash start.sh report      # 查看结果
bash start.sh             # 中断后继续，不增加预算
```

已有同名研究和参数不能被新版本覆盖。不要把已经创建环境或开始研究的旧版运行原地改成新版。
自定义目录只能在当前项目内，例如 `--root "$PWD/.arlab/course-a"`。

## 环境前提

需要 Bash、启动用 Python 3.10+、Git 和可用 C/C++ 编译器。缺少 OpenMP/GL/OSMesa
库时，v0.6.3 会在支持的 Debian/Ubuntu amd64 上自动使用私有 APT 索引下载匹配 deb 并
解包到项目；需要宿主已有 apt-get、apt-cache、dpkg-deb 和可信发行版 keyring。
不修改系统包状态，不使用 sudo，详见[原生库机制与支持范围](native-libraries.md)。
Python 3.10 与依赖仍由项目内 uv 创建。


**uv 环境不是沙箱。** 自动候选 Python 仍有当前用户权限；请审阅代码、不以 root 运行。
不能因为缓存放在项目中就保证任意生成代码无法访问其他项目。

## 研究没有缩小

三分布开发池可低成本起步、续训、加种子。固定方法前要求三任务完整对照；随后自动完成
九任务×五种子最终确认。最终作业及探索资源仍需明确授权，不要求为安装简便而降成单任务。
报告位于 `.arlab/d4rl9/runs/study/report.md`，缺少必要实验仍标记不完整。
文献与公开仓库调研无需额外搜索 Key，通用网页搜索关闭，无 RSI。

合成自检为独立入口 `bash start.sh demo`，不是 D4RL 失败后的替代流程。
完整真实 D4RL、远程 Kimi 和新机器安装的实际验收范围请查看当前 evidence/validation.json。

## 遇到 GL/OSMesa 缺失

v0.6.3 默认自动下载同发行版 deb 并解包到项目，不再要求管理员安装这些库。
重复 `bash start.sh setup` 即可；只核查原生层可用 `python3 scripts/prepare_native.py`。
这不会调用模型或启动研究。完整规则见 [原生依赖说明](native-libraries.md)。
