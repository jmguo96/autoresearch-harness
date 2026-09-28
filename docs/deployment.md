# v0.6.3 部署：本地 uv，无 sudo、无 Docker

当前规范以[项目内安装合同](project-local-install.md)为准。本文只列常用命令，
旧的 apt/Docker 部署描述仅适用于历史版本。

## 教材入口

```bash
bash start.sh --dry-run
bash start.sh setup
bash start.sh
```

默认部署根是源码下 `.arlab/d4rl9`。服务预设为实验室 Kimi，只填 Key。
不用系统包安装，也不构建镜像。`trusted_local` 是本版的默认候选模式，预算确认会说明
本地代码权限；`--code-mode disabled` 可明确关闭候选源码执行，但不是完整源码研究模式。
`--code-mode docker` 在本版安装入口无效。

## 底层安装入口

```bash
bash deploy.sh --yes --suite9 --research-online \
  --root "$PWD/.arlab/d4rl9" --code-mode trusted_local
```

`--skip-system-deps` 为旧脚本兼容的无操作参数；不提供时也不会安装系统依赖。
可显式 `--python /path/to/python3.10` 使用已有解释器建立项目 venv，解释器只读，
安装始终使用 `uv pip install --python <root>/venv/bin/python`。
可显式 `--uv /path/to/uv` 提供固定 0.10.0 可执行文件，复制到项目后再使用，不改原文件。
没有 uv 时自动下载固定 PyPI wheel 并验证其注册表哈希，不运行 curl|sh。

原生依赖先预检，缺库时从匹配发行版签名仓库下载 deb，仅在项目内解包并复验；不修复或更改系统。数据支持 --raw-dir / --offline-data
以复用已有原始 HDF5；默认下载到 root/data，MuJoCo 到 root/downloads。既有源文件作为
输入不被改写，输出必须在项目内。所有常规缓存与临时目录统一重定向。

## 状态与继续

`deployment.json` 只有真实自检通过才为 ready。失败包含阶段和日志；重复原命令继续。
`--skip-smoke` 只得到 prepared_unverified，不能绕过研究启动门禁。
`--reuse-runtime` 仅供显式合成测试，不能用它宣称私有依赖已安装或 D4RL 已验收。
源码、依赖环境和合同身份保持原有严格检查，旧版需保留原目录，不手动改哈希。

## 额外边界

不要手动 source env.sh 到日常 Shell；`start.sh` 和生成的 bin/arlab 包装器自行设置环境。
不要直接用 venv 的裸 console script 绕过包装器后再要求其采用所有项目缓存设置。
下载/安装目录合同适用于这些支持的入口，不是对任意第三方代码的 OS 写入约束。

本轮真实执行与未验收内容见 [Review](review-native-local.md) 和 [验收](../evidence/validation.json)。

## 原生库补齐

无需额外学生操作，默认 `--native-deps auto`。只检查使用 `--native-deps check`；
已有锁和 deb 的无网络重试使用 `--native-offline`。见 [具体机制](native-libraries.md)。
