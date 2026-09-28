# 原生库也能不使用 root：v0.6.3

## 这次修正了什么

`sudo apt install` 不是取得系统库文件的唯一方式。旧版把“不允许系统安装”错误地等同为
“只准使用预装 GL/OSMesa”；本版在缺库后自动下载匹配发行版 deb 并解包到项目。
Python 和 Python 包仍由本地 uv 管理，不使用 Docker/Conda，不扩展研究者权限。

```bash
bash start.sh setup       # 正常安装，自动补齐原生库
bash start.sh             # 原来的学生入口
python3 scripts/prepare_native.py  # 可选：只准备原生库，不安装 Python、不下载 D4RL、不调用 Kimi
```

## 实现路径

1. 编译并运行 `native-probe.c`，检查 OpenMP、GL、OSMesa context 与渲染调用。
2. 已满足时零下载。缺库时检测 Debian/Ubuntu 的准确 codename 与 amd64。
3. 生成私有 APT 配置，使用宿主发行版的可信 archive keyring。
4. 在项目内获取签名索引；在空的私有包状态上模拟依赖解析，不忽略 LLVM/C++ 等运行时依赖。
5. 保存精确版本、下载地址、大小、SHA256 和依赖说明到 `packages.lock.json`。
6. 下载全部所选 deb 到项目，按已认证索引的 SHA256 校验。
7. 用 `dpkg-deb --fsys-tarfile` 读取数据区，由受检 Python 解包，不用 `dpkg -i`，不执行 preinst/postinst/triggers。
8. 保留原私有 sysroot；只向构建/加载器导出必要头文件和共享库。不导出 glibc 与 ELF loader。
9. 在暂存前缀编译、运行通过后原子发布，使用最终路径再次验证并记录回执。

APT 被用作索引验证器和依赖求解器，不是系统安装器。`install` 只会与 `--simulate`
组合用于求解；实际获取文件使用项目 HTTPS 下载器。普通 Runner 仍拒绝任意 apt/dpkg 调用。
私有配置还把 `Dir::Bin::dpkg` 指向 `/bin/false`，并关闭宿主配置、维护钩子和不安全仓库选项。

```text
.arlab/d4rl9/
├── downloads/native/debs/          原始 deb 与下载回执
├── native/apt/                     私有索引、配置、keyring 副本、缓存和日志
├── native/packages.lock.json       精确依赖锁；重试不会重新选择版本
├── native/auto/sysroot/             提取的数据文件
├── native/auto/include/            仅 GL/KHR/必要的 OpenMP 头文件
├── native/auto/lib/                受控共享库，不含下载的 libc/ld-linux
├── native/auto/.arlab-native.json   完整文件清单与校验回执
├── native/logs/                    解包错误日志
├── native/include,lib,lib64,bin/    保留给用户提供的本地组件，不覆盖
├── tmp/                           编译检查的临时文件
└── native-check.json               最终检查结果
```

所有目录都在源码树中。环境变量只作用于启动进程及其子进程，不修改 shell 配置、
`/etc/apt`、`/var/lib/apt`、`/var/lib/dpkg/status`、`/usr` 或系统动态链接器配置。
无全局 `ldconfig`、无驱动变更。软件仍共享 CPU/GPU，uv 不是安全沙箱。

## 支持边界

自动路径支持 Debian 11/12/13 与 Ubuntu 20.04/22.04/24.04 的 amd64；每次只使用宿主自己的
发行版仓库。具体宿主仍须实际验收。需要已有 Bash、Python 3.10+、Git、可用 C/C++
编译器、apt-get、apt-cache、dpkg-deb 与可信发行版 archive keyring。它们不被更改。
已有 GCC 缺少 OpenMP 头文件时，尝试选择其主版本对应的 `libgcc-N-dev`，只导出 omp.h。
非 GCC 工具链和缺失完整编译器，不保证自动解决。

不跨发行版下载 glibc，不把另一版本 loader/libc 注入 Python。依赖解析中列出的基础系统
包会记录为使用宿主提供项。真实编译、加载与 context 创建失败仍会停止，而不是禁用校验。
这不等于全部软件 ABI 都得到形式证明，完整 MuJoCo/D4RL 自检仍在后续安装环节执行。

从源码编译同样可以使用 `--prefix` 或 Meson 的 `--prefix` 安装到项目；这是有效的替代
方式。但本轮没有实现通用 Mesa/LLVM/GCC 源码引导器。对不支持系统，应由教师提供经过
审核的本地前缀，或者明确扩展对应构建配置，不能拿其他发行版的包碰运气。

## 网络、中断与缓存

`bash start.sh setup --native-deps check` 仅检查不下载。
`bash start.sh setup --native-offline` 只复用已经存在的锁与对应 deb，没有缓存就明确报错。
签名失败、包校验失败、未知发行版、链接越界、文件损坏不静默降级。
重复部署复用原锁；在线仓库日后移除某个版本时，需要保留缓存或显式创建并审核新锁。

当前自动解析使用官方 HTTPS 仓库。宿主没有网络时，可以从同发行版、同架构课程机器
拷贝 `native/packages.lock.json` 和 `downloads/native/debs/`，再使用 offline 模式；
信任拷贝来源是用户职责，不把一个任意 JSON 声称为独立签名。

## 对本次失败目录的修复

用户贴出的错误发生在旧版 preflight，尚未创建 Python 运行环境。可用配套补丁脚本在
这个精确前提下升级安装文件并继续。补丁检查原文件哈希、已有运行环境回执和研究状态；
若已创建 venv 或开始研究，拒绝原地升级，避免更换代码后混用旧实验身份。

## 依据

- dpkg-deb（解包不是系统安装）：https://manpages.debian.org/bookworm/dpkg/dpkg-deb.1.en.html
- apt-get（simulate、download、print-uris）：https://manpages.debian.org/bookworm/apt/apt-get.8.en.html
- APT_CONFIG 与私有 Dir 配置：https://manpages.debian.org/bookworm/apt/apt.conf.5.en.html
- mujoco-py CPU 依赖：https://github.com/openai/mujoco-py/blob/master/mujoco_py/builder.py

这些文档说明接口，不提供本项目端到端 D4RL 成功的证据。验收范围以本轮记录为准。
