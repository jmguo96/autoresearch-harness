> 历史记录：本页保存所标版本的内容，不是本次阅读整理版的操作或验收说明。当前入口见[文档目录](README.md)，本次变更见[整理核对](review-readability.md)。

# v0.4 上下文与调研能力：实现 Review

日期：2026-09-20。本轮实际修改 v0.3 源码，删除 RSI，保留固定 IQL 和开放算法研究。
这是同轮自审、故障注入与回归验证，不是独立第三方安全审计，不包含真实 D4RL 性能比较。

## 一、研究逻辑修正

| 原研究 Review | 当前处理 | 验收 |
|---|---|---|
| R01：没有 baseline 也可完成 | finish 必须同时满足 baseline 与 selected 最终合同面板 | baseline 缺失拒绝，完整负结果允许 |
| R02：重放全部 trial 导致上下文耗尽 | bounded ContextBuilder + 程序摘要 + 按需读取 | 216 / 1,080 条账本，100,000 字符反馈，超长单行 |
| R03：两侧同样缺数据却 complete | 比较显式 expected_cells，单独报告匹配和合同覆盖 | 4/18 单元不能产生完整结论 |
| R07：审计 optimizer 预算漏配 | 唯一冻结方法成本上界预检、owner cap、dry-run | 上界较大候选正确估算，不足时零作业 |
| R08：只用最短 rung 排名 | 最新共同完整 rung，不能混合不同训练量 | 中间档反转改变默认排序 |
| R04/R05/R06/R09：演化异常/发布/计费/deadline | 删除 RSI 模块、CLI 与运行依赖 | 无 evolution 导入，无递归命令 |

删除代码路径不等于验证了这些旧 RSI 算法；本版本不提供 RSI 功能。
原 IQL trainer、networks、data 及开放算法 protocol/runtime/guard/context/provenance 与 v0.3 逐文件核对不变。

## 二、新能力的边界 Review

| 边界 | 处理与测试 |
|---|---|
| 搜索片段被当作全文 | 区分 metadata/abstract/search_snippet 与正文 scope；读取得到精确 span |
| 查无结果被当作原创性 | 输出与稳定指令明确空搜索不是 novelty 证明；API 错误不能作为空成功 |
| 新增工具把长正文再次塞满上下文 | 每次读取分页，结果截断标记和可检索反馈引用；常驻仅来源索引 |
| 单行很长导致分页丢字 | 额外字符 offset，测试完整拼接恢复英文 / 中文 / 换行 |
| 研究笔记被大量自动实验笔记挤出 | 默认先筛选类型再检索，不让 episode 记录占满工作集 |
| 源码引用随分支变化漂移 | GitHub ref 首次解析为 commit；文件快照带 hash / path / license 元信息 |
| 接口未配置返回伪结果 | Brave 缺 key、网络失败、HTTP 非成功均明确报错，无虚构 fallback |
| SSRF / 私网 / 身份泄露 | 公共 IP + 精确域名 + TLS 固定连接；重定向去认证头并重新检查 |
| arXiv 连接频率 | 同机同用户共享锁与 3.1 秒间隔；不承诺多机限流 |
| 恶意/异常资料解析 | XML 禁 DTD；PDF 子进程有限资源；格式错误转换为工具失败反馈 |
| 缺失或不支持的查询条件 | 对聚合面板的 case/status 错用明确拒绝，不返回误导性的空列表 |
| 低成本 check 被混作成绩 | purpose=check，独立 quota，仍计费，排除所有正式排名和完成条件 |
| 上下文计量 | 字符计数字段与实际序列化一致，整个模型输入含一次 schema；不冒充 Token 精确预算 |
| 网络权限变成任意源码执行 | research_access 与 code_execution 独立；下载源码工具不执行代码 |

本轮新测试文件为 `tests/test_context_research.py` 与 `tests/test_research_tools.py`。
旧模型适配器测试改为比较严格 Pydantic 标准化输出，因为新分页动作具有默认参数；没有放松响应验证。

## 三、真实执行与测试替身分开

最终完整测试为 **228 passed、1 skipped、0 failed**，共 229 项。日志与 XML 见 evidence/tests.log 和 evidence/tests.xml。
调研客户端通过注入 HTTP 响应测试 Atom / Crossref / Brave / GitHub、分页、认证和故障路径。
DNS/TLS 连接目标使用测试替身核对；不是实际连接外部服务。一个空白 PDF 实际经 pypdf 子进程解析，
只用于验证无文本页被明确标记；不代表科学论文表格或图像解析质量。

实际研究 walkthrough 使用显式 `--offline-fixture`，论文内容和 GitHub commit 是标记过的测试资料；
真实训练、策略导出与独立合成环境 rollout 是实际数值执行。它检验工具链，不证明大模型自主研究能力。

在相同环境用实际客户端请求 arXiv / Crossref / GitHub 时，DNS 失败，均保存为 unavailable。
没有用网页阅读工具能访问文档这一事实，替代容器中应用的网络验收。
完整 D4RL / MuJoCo、Docker/CUDA、在线模型、全新机器依赖安装和性能提升均未验收。

## 四、还不能声称解决的事项

网络代码的限制不能证明对任意攻击都安全；同步 DNS 不能硬性抢占，单机锁不覆盖多机。
正文没有 JS 浏览器、OCR 或视觉图表核验；PDF 提取范围有限，网页编码简化处理。
来源与引用存在性不等于语义支持，外部数据仍可能错误、恶意或与任务无关。
固定字符上限降低输入增长，但未做真实模型上的能力、Token 成本或研究效率对照。
内层研究仍单决策流同步执行，没有自动并行 GPU 调度，数据库索引并非超大规模系统。
候选执行继续保留 trusted_local / Docker 的既有威胁模型，未新增对任意 Python 的形式验证。
