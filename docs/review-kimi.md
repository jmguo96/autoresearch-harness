# v0.6.1 Kimi 接入 Review

这次以 v0.6 教材阅读整理版为基础，新增实验室 New API 的单 Key 使用路径。
这是接入功能修改，不是纯排版，也不是科研算法升级。完整回归为 **350 通过、1 项因缺少 legacy D4RL 依赖跳过、0 失败**，包含 54 项新增接入测试。详情见 [validation.json](../evidence/validation.json)。

## 与用户说明的对应关系

| 用户说明 | 实现 |
|---|---|
| 使用管理员分发的 New API Key，而非上游账号 | 默认读取 NEW_API_KEY 或终端隐藏输入；旧 ARLAB_API_KEY 只作兼容回退 |
| Base URL 为指定实验室 `/v1` | 原地址预置；学生不再填写 URL |
| 模型名称必须以 `/models` 返回为准 | 自动发现可见 Kimi；auto-kimi 不作为模型名发出；新研究前固定真实 ID |
| 普通兼容模型使用 Chat Completions | 继续使用原 JSON Decision 适配器；显式 stream=false |
| Codex 有单独 Responses/stream=true 规则 | 不调用该渠道、不安装 Codex、不引入 Responses 依赖 |
| 代理可能导致超时 | 发现与推理客户端均不读取环境代理；不更改全局代理变量 |
| 503 model_not_found 应检查渠道 | 模型列表失败即停止；单次推理不对明确的 model_not_found 做内部重试 |
| 不要泄露 Key | 不保存 Key；错误信息不回显提供方正文；安装与候选 worker 的环境不继承 NEW_API_KEY |

## 测试与审查项

- 默认启动在未配置型号时只要求 Key，不询问 URL 或型号。
- 列表中没有 Kimi、指定型号消失、无鉴权权限、响应格式错误时失败关闭，没有切换其他模型。
- 仅在获得实际列表后选择模型；同时出现多个 Kimi 时采用文档明确的确定性规则，而不是虚构模型 ID。
- 恢复绑定原合同中的准确型号，不因列表出现较新型号而自动升级。
- 模型目录接口不跟随重定向，不向跳转位置转发 Bearer；测试还使用了真实本地 HTTP 服务。
- HTTP 例外要求配置允许且 Base URL 准确匹配给定实验室地址；不同主机、端口、路径仍拒绝。普通 HTTPS 及回环协议保持支持。
- Chat Completions 响应保持本地 schema 校验；推理说明字段不进入行动或研究证据。
- 隐藏输入/临时 Key 别名在成功和错误路径退出时清理；用户预先提供的环境变量由用户管理，不擅自删除。
- `check-api --dry-run` 不写入、不联网；`check-api` 本身不需 PyTorch/D4RL，只有标准库。
- 新老 Key 名称均不进入安装子进程、训练 worker 或策略进程环境。测试中的 Key 全是假值。

## 研究逻辑未改变

[research-integrity.json](../evidence/research-integrity.json)与原阅读版逐文件比较：

- 核心研究循环、上下文、预算、面板比较、分支、日志、检查点、审计与最终报告未改。
- 全部 IQL / 开放算法 / in-sample 检查 / 调研工具 / 研究提示 / 旧测试原样保留。
- `core/schema.py` 只新增 `ModelConfig.allow_lab_http`；其它协议类的 AST 不变。
- 改动局限于模型传输与启动接入、九任务安装预设的默认模型，以及版本/镜像标签。

模型单次输出上限在 Kimi 预设中设为 16,384、超时 300 秒；总研究预算没有因此自动扩大。
源码哈希、版本和包含 ModelConfig 的合同身份会变化，所以新版本不能覆盖旧运行。

## 测试过程

第一轮新增测试中，空计划不符合既有 Decision schema、测试调用漏传策略环境函数参数，导致两个测试失败。第二轮又发现测试断言没有计入 schema 填充的默认字段。修正的是测试夹具与期望，未放宽原验证规则；随后定向测试通过。早期日志保留在 evidence/kimi-tests-first.log、kimi-tests-second.log，定向最终结果在 kimi-tests-targeted.log。

完整回归在最终源码固定后执行。测试响应、本地 HTTP 模型列表以及所构造的最终状态均是明确标注的协议夹具，不是 Kimi 或 D4RL 成绩。

## 限制

没有真实实验室 Key，未执行给定网关的鉴权和 Kimi 推理；没有新执行官方数据下载、MuJoCo/CUDA/Docker 的完整验收。
API 文档未列出实际 Kimi ID，也未声明网关转发所有 Kimi 特性。/models 可见不等于推理路由、JSON Mode 或账户额度已验证。

HTTP 例外是依用户说明作出的部署选择；明文传输仍会暴露 Key 和研究内容。客户端不是隧道或 TLS 代理，不能以“不落盘”替代传输保护。正式课程建议教师配置 HTTPS。

API 别名背后模型映射的稳定性取决于管理员，软件固定的是公开 ID，不能证明上游权重不变。泛化、算法提高和教学效果仍需相应的独立实验。
