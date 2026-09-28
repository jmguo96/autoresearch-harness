# v0.6.3：项目内原生库补齐与 Review

## 修正的问题

v0.6.2 把“禁止系统安装”错误地等同为“原生头文件和库必须已经预装”。
用户在 OpenMP/GL/OSMesa 的 preflight 编译检查中被阻断。本版允许缺库后自动下载
匹配发行版的 deb、验证后提取数据文件，并仅向本项目进程暴露必要的头文件和共享库。
Python 继续由项目 uv 管理；没有 Docker、sudo、全局包安装或研究协议改变。

## 安装路径与审查重点

| 项目 | 处理与验证 |
|---|---|
| 下载 deb 与系统安装混淆 | APT 只在私有目录校验索引和模拟求解；实际包由受限 HTTPS 下载器获取，`Dir::Bin::dpkg=/bin/false` |
| 主机 hooks 或 package 状态影响 | 通过 APT_CONFIG 提前指定私有目录，不读取宿主配置片段；空私有 status；真实签名测试仓库检查宿主状态哈希不变 |
| 随便下载一个发行版的包 | 精确匹配 Ubuntu focal/jammy/noble、Debian bullseye/bookworm/trixie、amd64；未知系统拒绝猜测 |
| 只补头文件忽略 LLVM 等间接依赖 | 从匹配签名索引解析依赖闭包；记录版本、URL、大小、SHA256、依赖与宿主提供基础项 |
| 解包执行 pre/postinst | 只读取 `dpkg-deb --fsys-tarfile` 数据流；测试中维护脚本标记未执行 |
| 路径穿越与恶意软链接 | 流式受检解包，只保留 include/lib 数据；安全重定位内部链接；不允许设备文件、越界和文件冲突 |
| 私有 glibc 污染 Python | 不下载/导出系统 libc/loader；自动 lib 目录只包含受检 ELF 库；不是把完整 sysroot 加入 LD_LIBRARY_PATH |
| omp.h 与 GCC 版本不匹配 | 已有 OpenMP 检查失败时查 GCC 主版本，解析对应 libgcc-N-dev；仅导出 omp.h 及所需非 libc 库 |
| 文件“存在”却不能加载 | 在 staging 和原子发布后的真实前缀都做 C 编译、链接、OpenMP 执行、OSMesa context/渲染调用检查 |
| 重试后悄悄换版本 | 保留 packages.lock.json 和校验回执；损坏或宿主身份变化拒绝复用 |
| 手动 native/ 被覆盖 | 自动结果在 native/auto，保留用户的 native/include、lib、lib64、bin |
| 用户在预检失败后被要求全部重来 | 单独补丁检查原始文件和运行状态，只允许尚未创建运行环境/研究的目录更新；备份并保留失败日志 |

## 可读性与源码范围

`native.py` 组织检查与补齐；`native_apt.py` 只处理索引、精确解析和下载锁；
`native_archive.py` 只处理安全解包与运行前缀。`scripts/prepare_native.py` 是可选的
owner 诊断入口，不作为研究者工具；常规 `start.sh setup` 自动调用同一逻辑。

63 个既有非安装 Python 文件与原 v0.6.2 逐字节相同，包括研究循环、模型/Kimi
调用、预算/面板/续训、IQL、开放算法边界和调研能力。变化集中在安装器、进程
环境、学生参数透传、版本标识和新测试。见 evidence/research-integrity.json。

## 本轮实际测试

最终完整回归 **418 passed、1 skipped、0 failed**，跳过项仍为缺少 legacy D4RL。
新增 37 项原生库测试又以 **UID 1000 普通用户全部通过**，它们是全套测试的子集，
不能额外加到 418。测试使用真实 APT、GPG、dpkg-deb 和 C 编译器；仓库签名、
deb 和 GL/OSMesa ABI 库均是明确生成的测试夹具，不是官方下载文件或真实 Mesa。

实际测试覆盖私有签名仓库、依赖解析、索引篡改拒绝、数据区解包、维护脚本不执行、
宿主 dpkg 状态不变、原生编译/加载、缓存复用与破坏、路径/软链接、接口透传。
原有研究与合成环境回归一起重跑。具体版本、记录在 evidence/validation.json。

非 root 第一轮夹具继承 setgid 目录使 dpkg-deb 拒绝，已显式设置测试 DEBIAN
目录权限；另一次测试临时路径太长导致 GnuPG 签名键测试 socket 无法创建，最后
使用较短目录重跑全部 37 项。没有通过放宽运行安全检查来修复这些测试问题。

## 尚未验证与保留边界

当前运行环境 DNS 不可用，没有成功下载官方原生库包，没有完成用户目标机器、
真实 Mesa/LLVM、legacy MuJoCo、D4RL、GPU 或 Kimi 推理的全流程验收。
不能把签名本地仓库和测试 ABI 的通过说成目标服务器部署成功。

自动 deb 路径依赖现有 apt-get/apt-cache/dpkg-deb、可信发行版 keyring、Git 和
C/C++ 编译器；不会引导一整套 GCC/Mesa/LLVM 源码编译。源码前缀安装本身可行，
但不在本次已实现的自动路径内。未知系统、缺失编译器、ABI 错配或上游不可达会
明确失败，不覆盖 glibc、不安装系统包。项目局部依赖不等于操作系统级沙箱。

源码身份已变化，已有正式研究不可原地混用。配套修复器只允许像本次用户日志
那样停在旧 preflight 的 v0.6.2 目录；其余情况用独立新源码和运行目录。
