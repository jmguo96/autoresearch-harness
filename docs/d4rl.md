# D4RL 安装与数据：uv 项目隔离版

保持 legacy D4RL、MuJoCo 2.1.0、mujoco-py 2.1.2.14 和固定 IQL，不改换 Gymnasium/Minari。
教材主入口下载九份数据；开发三项和最终九项协议见 [研究协议](nine-task-study.md)。

```bash
bash start.sh setup
```

使用 uv 在 `.arlab/d4rl9/venv` 安装；Python 下载、Git 源码、MuJoCo、数据、缓存和临时
文件都放在源码目录下。没有 Docker、sudo 或系统包安装；APT 仅用于项目内签名索引与依赖模拟。

## 不能由虚拟环境代替的依赖

宿主需已有 Git 与 C/C++ 编译器。安装器先在项目 tmp 中编译、运行探针；缺少
GL/OSMesa/OpenMP 头文件或库时，自动使用匹配 Debian/Ubuntu 发行版的签名索引，
下载依赖 deb 并仅解包到 `native/auto`，随后复验。APT/密钥环等前提与支持范围见
[原生库说明](native-libraries.md)。教师也可把 ABI 兼容组件提供到
`.arlab/d4rl9/native/{bin,include,lib,lib64}`；不会替换宿主 glibc 或安装系统包。
动态库路径和头文件路径只写入项目启动环境，不修改 .bashrc。`patchelf` 由 uv 装入 venv。

CUDA 模式只使用宿主已有驱动；不安装或切换驱动。无头 MuJoCo 的 CPU 构建与 PyTorch
使用 GPU 训练是两个概念，不因 `MUJOCO_PY_FORCE_CPU=1` 就取消 GPU 训练。

## 单独下载／导入

部署就绪后使用带本地环境的包装器：

```bash
.arlab/d4rl9/bin/arlab download-d4rl --suite9 --data-root "$PWD/.arlab/d4rl9/data"
.arlab/d4rl9/bin/arlab prepare-d4rl --env hopper-medium-v2 \
  --hdf5 /read/only/official.hdf5 --output "$PWD/.arlab/d4rl9/data/hopper-medium-v2.npz"
```

省略 HDF5 时会请求固定来源。原始文件和转换文件有 SHA256 与回执；哈希一致不等于
首次下载的独立认证。数据处理、timeout 语义、next_indices 及 in-sample 检查未改动。
缺失数据、限流、下载失败、MuJoCo 编译失败都不自动改用合成任务。

`bash start.sh demo` 是另行选择的合成环境测试；不能用它说明真实 D4RL 已装成功。
详见[项目安装合同](project-local-install.md)及当前验收记录。
