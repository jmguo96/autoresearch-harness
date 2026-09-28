# v0.6 调研工具：发现、取得、定位、引用

网络工具是稳定 Harness 的受控外部能力，不运行递归改进。不依赖任何专有编码 Agent。
学生 `start.sh` 生成的研究默认启用免额外搜索密钥的论文/公开代码调研，通用 `search_web` 关闭。底层 `ResearchAccess` 类型的默认值是 `enabled=false`；直接使用底层合同的用户须在新研究前通过 `--research-online` 授权。入口预设与类型默认不是同一层，不能混淆。

## 一、已实现能力与限制

| 工具 | 主要输入 | 输出 / 语义 |
|---|---|---|
| search_literature | query, provider=arxiv/crossref, offset, limit | 论文元数据与可能的摘要，来源 ID，下一页位置 |
| search_web（默认关闭） | query, offset, limit | 可选 Brave 后端，学生无需配置 |
| fetch_source | url, refresh | HTML / UTF-8 文本 / PDF 文本快照、哈希、scope、链接 |
| read_source | source, start_line, offset, max_lines, max_characters | 正文片段，稳定 span ID，next 游标 |
| find_source | source, query, offset, limit | 字面、忽略大小写的匹配位置；不是语义搜索 |
| search_repositories | query, page, limit | GitHub 公开仓库索引，提供方不完整标记 |
| read_repository | repository, ref, path, offset, limit | 固定 commit 的目录索引或代码源快照，许可证元信息 |

arXiv query 使用官方查询表达式，例如 `ti:"Implicit Q-Learning"` 或 `all:"offline reinforcement learning"`。
Crossref 使用 bibliographic query，不应照搬 arXiv 语法。GitHub 搜索结果不能自动证明作者身份或代码正确性。
接口按照供应商公开文档实现；本环境没有成功完成在线请求，具体边界见验收记录。

## 二、服务配置

学生 Kimi 预设使用模型密钥 `NEW_API_KEY`；底层通用服务仍可使用 `ARLAB_API_KEY`。模型与调研密钥分开。arXiv / Crossref 适配器不要求项目密钥。
GitHub 支持未鉴权公开资源访问；可选 `ARLAB_GITHUB_TOKEN`，私有库在工具层被拒绝。
`search_web` 默认关闭且不向模型提供。仅 owner 显式设置 web_search_enabled=true 后才需要 Brave Key；论文/公开代码调研不需要它。
不附带搜索服务账户、付费额度或任何密钥。

```bash
export ARLAB_API_KEY='模型服务密钥'

arlab run configs/my-d4rl.json --output runs/research \
  --provider compatible --model '你的模型' --base-url 'https://你的服务/v1' \
  --research-online
```

上述是已有合同的底层调用方式；学生通常只运行 `bash start.sh`。无需填写 Brave Key 或 GitHub token。

启用在线调研并不授权候选源码在宿主执行。本地源码执行在研究预算确认中单独披露风险；本版不使用 Docker，venv 不提供文件/网络沙箱。

可选高级设置：GitHub token 仅用于公开资源的已鉴权访问；`search_web` 只有用户显式开启时才需要 Brave Key。缺密钥不会偷偷换搜索服务。

## 三、手动检查一个研究工具

在已经建立的 campaign 上，用户可调用相同的受控调研入口。这不是任意文件/网络工具：

```bash
arlab research runs/research --action configs/search-iql.action.json
```

action 文件是单个动作对象，例如：

```json
{"kind":"search_literature","provider":"arxiv","query":"ti:\"Implicit Q-Learning\"","limit":5}
```

返回 source ID 后，使用 `read_source` 读取搜索元数据；取得真正论文 URL 后 `fetch_source`，再读取正文。
仓库例子：

```json
{"kind":"read_repository","repository":"corl-team/CORL","ref":"HEAD","path":"algorithms/offline/iql.py"}
```

第一次把 ref 解析成 40 位 commit 并持久绑定；读取后续文件可显式提供该 commit。
空 path 返回目录树。只读取和保存代码，不 git clone、不 import、不 pip install、不直接执行。
自动许可元信息不足时应由人检查 LICENSE 及再使用条件。

## 四、证据链

```text
查询参数 + 原始响应哈希
→ src-*：某一时刻、某 URL / commit 的内容快照
→ span-*：确切读取的行、偏移与正文片段
→ remember / hypothesis.evidence_ids
→ 候选修改与真实实验
```

搜索范围明确分为 metadata、abstract、search_snippet；正文有 html_text / pdf_text 等 scope；
仓库文件是 repository_file，不会因为内容里写了“最优”就成为科学验证。
刷新同一 URL 创建新快照，不覆盖已被引用的原文。来源 ID 和实验 ID 都是证据引用，二者不能互相替代。

论文读取不自动生成“已验证结论”。模型可以形成有证据指向的笔记，但 `verified_claim=false`。
空搜索不能证明原创性；获取论文不能证明已读完；读取部分文本不能证明图表、公式或结论理解正确。

## 五、页面和 PDF

HTML 提取正文、标题和链接，移除 script/style；不执行 JavaScript。UTF-8 解码可能替换无法解码字符。
PDF 使用可选 pypdf，在有限 CPU / 地址空间的子进程内提取文本。上限为 100 页和 500,000 字符；
上限、无文本页及提取范围会显式记录。默认单响应网络大小 8 MB。
没有 OCR、视觉模型读图、表格数值复核或付费墙绕过。扫描页没有文字时报告 unread，而不是伪造正文。
限制子进程不等于已形式证明可以安全解析任意恶意 PDF；不可信材料宜运行在隔离的研究宿主。

## 六、网络和资料的信任边界

仅允许用户批准的精确公共 HTTPS 主机及端口 443；拒绝凭据嵌入、IP 私网、localhost、链接本地地址。
DNS 解析后连接固定公共 IP，并保留原域名 TLS SNI / 证书校验；重定向重新检查且不转发提供方认证头。
不读取代理环境变量，不让模型设置 HTTP 头。源文件、网页和代码注释都是不可信数据，不是新指令。
这些约束减小暴露面，不是一般性提示注入防御或任意候选源码的安全证明。

请求、下载字节、响应上限、重试次数、超时都有记录。arXiv 两个主机共用同一项目内的文件锁，
请求开始间隔至少 3.1 秒并限制一条连接；多项目、多用户或多机器联合使用需由教师协调共享限流，不能声称项目内锁覆盖全部用量。
标准库 DNS 解析不能被这段同步代码硬性抢占，解析后仍会重新检查截止时间。

默认授权含 arXiv、Crossref、GitHub、Brave、PMLR、OpenReview、PyTorch / GitHub 文档的列明域名。
默认列表之外的出版商或项目站点，即使出现在搜索结果中，也必须由 owner 用 `--research-host` 批准。
批量爬取、私人内部资料和身份认证网页不在本轮实现范围。

## 七、离线资料

用户可以显式导入本地 UTF-8 文本：

```bash
arlab import-source runs/research /path/to/notes.txt --title '用户提供的研究笔记'
```

此命令是 owner 操作，不向智能体开放任意宿主文件读取；拒绝符号链接和过大的文件。
没有网络时模型可读取已有快照或用户导入内容，但不能把它们说成实时搜索结果。

## 八、研究不是按固定步骤表执行

默认提示鼓励调研、可证伪假设、低成本检查和匹配实验，但不强迫每次都搜索固定篇数，
也不把查阅文献本身视为进展指标。模型可按信息缺口选择下一步，稳定 Harness 管理证据、权限和资源。
