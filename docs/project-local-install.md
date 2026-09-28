# 项目内安装合同（v0.6.3）

## 用户可依赖的行为

学生仍使用 `bash start.sh`，只填写实验室 Key。安装不需要 Docker，安装器不会调用
sudo、doas、系统安装/升级/删除或容器命令。普通 Runner 仍拒绝直接的 apt/dpkg 安装入口；
新的 NativeApt 仅用隔离配置执行索引更新、模拟依赖和元数据查询，真正 dpkg 执行被指向
`/bin/false`。dpkg-deb 只输出包的数据归档供受检解包，不执行包维护脚本。
该检查不等于拦截任意第三方 Python/build hook 的系统调用。

“本目录”是解压得到的源码目录，不是从任意位置调用脚本时的工作目录。管理根默认
`.arlab/d4rl9`，自定义 `--root` 必须在源码下。路径穿越和已有目录符号链接会被拒绝。
已有 HDF5 / Python / uv 可以作为外部只读输入；它们不被安装器改写。默认 Python
由 uv 下载到项目，不使用外部解释器作为安装目标。

## 安装顺序

1. 验证目录、平台、磁盘及本地库；缺库时自动在项目内下载/解包匹配 deb，再编译和运行探针。
2. 本地已有固定版本 uv 则复制；否则下载固定 PyPI 元数据/wheel，校验并只提取可执行文件。
3. `uv python install --no-bin` 安装项目私有 Python，不向用户全局 bin 添加链接。
4. `uv venv` 创建 `<root>/venv`，不包含 system-site-packages。
5. 所有依赖、项目、D4RL 和 MJRL 安装都通过 `uv pip install --python <root>/venv/bin/python`。
6. MuJoCo 下载到 `<root>/downloads`、解包到 `<root>/mujoco`；Git 源码在 `<root>/sources`。
7. 数据下载、配置与真实自检沿用原研究流程；只有自检通过才写 ready。

`pip` 被作为一个固定的环境内工具安装，以读取兼容的依赖来源报告；安装过程本身
不调用 `python -m pip install`。这不是完整的传递依赖哈希锁定，仍记录实际版本与
Git commit。uv wheel 的注册表哈希验证也不是独立数字签名。

## 进程环境

`deployment/local.py` 是唯一的项目缓存目录表。HOME、XDG_CONFIG/CACHE/DATA/STATE/RUNTIME、
uv Python/tool/cache、pip、Python userbase、临时文件、PyTorch、Triton、CUDA、Numba、
Matplotlib、Hugging Face 及 W&B 目录都位于 root。

`UV_NO_CONFIG=1`、`PIP_CONFIG_FILE=/dev/null` 防止读取另一项目或用户的安装配置；
安装不继承活动 venv/Conda 的目标、不采用用户配置的 PIP_TARGET/UV_SYSTEM_PYTHON。
安装进程不继承模型密钥。Git 不读取用户全局配置或 credential helper；仅访问公开仓库。
私有镜像、用户代理证书或 Git 高级配置未必自动适用，不会为适配它们关闭 TLS 校验。

包装器设置同一环境再运行主程序；worker 的密钥过滤之后恢复这些项目路径。
策略推理临时目录显式放在 root/tmp。缓存重定向不改变核心算法的训练方法。

## 原生依赖也在项目内补齐

uv 继续管理 Python 与虚拟环境；原生库由独立模块提供，不把 `.deb` 当作 Python 包。

- 先使用现有宿主/用户提供库完成真实编译与 OSMesa 上下文测试，已满足则不下载。
- 失败后，按 `/etc/os-release` 精确选择 Debian/Ubuntu 同版本仓库与宿主可信密钥环。
- 使用 `native/apt/` 下的私有配置、空 status、缓存、索引和日志解析整个依赖闭包。
  不读取宿主 APT 配置片段与钩子，不修改宿主包数据库；`APT_CONFIG` 在 APT 启动前设置。
- 对已认证索引里的精确版本和 SHA256 建立 `native/packages.lock.json`。实际 `.deb`
  放在 `downloads/native/debs/`；恢复优先复用锁与校验合格的缓存。
- 只提取数据归档，拒绝路径穿越、越界链接、设备文件与异常大成员。
- 私有 sysroot 提供 GL/KHR/OpenMP 头文件及共享库；不导出 libc、ELF loader、系统配置或程序。
- 临时前缀编译/运行通过后原子发布到 `native/auto/`，使用稳定路径再次验证。
- `native/include`、`native/lib`、`native/lib64` 仍保留给教师提供的本地组件；不覆盖它们。

详情、局限及不联网重试参见 [native-libraries.md](native-libraries.md)。源码编译并不需要
sudo（可以使用本地 prefix），但本版实现的是匹配 deb 的自动补齐，不包含通用 Mesa/LLVM/GCC
全栈源码构建器。不支持的发行版不猜测 Ubuntu 包，也不拿新版 glibc 覆盖宿主。

## 不能保证的事情

- venv 不限制当前用户的文件、网络、进程或设备权限。模型生成 Python 仍可能越出目录。
- 更改 HOME、工作目录与缓存只是常规软件约定；硬编码绝对路径或恶意依赖能够绕开它。
- CPU、GPU、磁盘空间和 I/O 仍与其他项目共享。无全局依赖修改不等于无资源竞争。
- 调研限流文件现在只在项目内共享；多个项目/账号同时运行时需教师协调总频率。
- 缓存可删除不意味着运行目录可删除；检查点、模型来源与研究数据是证据，应先备份。

因此，本版能实现的是**项目管理的安装、下载与常规缓存不污染其他项目**，不是对
任意源码的“绝不影响外部文件”保证。请不要以 root 执行研究，不把秘密放到研究账号。

## 使用与清理

```bash
bash start.sh --dry-run
bash start.sh setup
bash start.sh
```

自定义位置：`bash start.sh --root "$PWD/.arlab/course-a"`。
外部磁盘空间不够时，将**整个项目源码目录**移动到目标磁盘，再新建部署，不能把
venv 或 python 目录随意挪走后宣称旧路径可直接恢复。

停止研究、备份 runs 和数据后，可删除项目的 `.arlab/`；无需卸载系统包或清理 Docker。
不要复制示例中的删除命令盲删已有科研记录，因此这里不提供自动递归删除入口。

## 外部依据

以下为本轮查阅的实现文档，不提供本项目的安装成功或性能证据：

- uv 环境变量：https://docs.astral.sh/uv/reference/environment/
- uv 虚拟环境及显式 Python 安装目标：https://docs.astral.sh/uv/pip/environments/
- uv 0.10.0 发布：https://pypi.org/project/uv/0.10.0/
- patchelf Python 分发：https://pypi.org/project/patchelf/0.17.2.4/
- mujoco-py CPU 构建依赖：https://github.com/openai/mujoco-py/blob/master/mujoco_py/builder.py

核心定位：local.py（目录/环境）、uv_bootstrap.py（固定 uv）、native.py（检查与本地补齐）、native_apt.py（私有仓库解析）、native_archive.py（受检解包）、
manager.py（编排）、common.py（受控命令）、classroom/bootstrap.py（学生入口）。
