# 实验室 Kimi 接入：学生只填写 API Key

本版以用户提供的《实验室 New API 使用说明》（原文最后更新 2026-09-01）为连接依据。
模型地址和协议已配置，学生不需要注册上游账号，不需要安装 Codex/Claude Code，也不需要额外搜索密钥。

## 学生操作

```bash
bash start.sh
```

允许必要安装、确认原有研究预算后，终端只要求填写管理员分发的 API Key。输入不回显、不保存。
已安装好环境时，不再重复安装。地址与真实模型 ID 是非秘密配置，保存在部署目录的 `model.json`。

已有终端环境变量也可直接使用：

```bash
export NEW_API_KEY='管理员分发的Key'
bash start.sh
```

优先使用 `NEW_API_KEY`，为兼容前一版也接受 `ARLAB_API_KEY`；不要同时设置不同的 Key。
此例不要将真实 Key 提交到代码库或课件。隐藏输入比在共享机器的命令历史中留下 Key 更合适。

只检查 Key 和 Kimi 模型是否可见，不安装依赖、不下载数据、不调用推理：

```bash
bash start.sh check-api
```

`check-api` 仅需要已有的 Python 3.10+ 标准库。它没有证明推理接口可用、剩余额度充足或模型能完成研究。
默认 `start.sh` 自己会做模型列表检查，`check-api` 是可选的提前诊断，而不是学生必需的额外步骤。

## 预置协议

| 设置 | 默认值 |
|---|---|
| 模型家族 | Kimi |
| Base URL | `http://43.143.241.66:18319/v1` |
| 获取真实模型 ID | `GET /v1/models` |
| 推理接口 | `POST /v1/chat/completions` |
| 鉴权 | `Authorization: Bearer <NEW_API_KEY>` |
| 请求 | `messages`，非流式 `stream=false`，JSON 输出要求 |
| 单次输出上限 | 16,384 tokens；这是容量上限，不是承诺输出量 |
| 单次推理超时 | 300 秒，仍受原研究总体资源上限约束 |
| 代理 | 客户端不继承系统 HTTP(S) 代理 |
| 重定向 | 不跟随，不把 Bearer 头发送到另一个地址 |

这里使用 Kimi 的 Chat Completions，不使用说明中的 Codex `/responses`，也不需要其 `stream=true` 特例。
研究者仍然返回项目现有的 `Decision` JSON，由原 Harness 执行工具；没有切换成其他 Agent SDK。
返回正文经本地 schema 校验；`reasoning_content` 不当作行动 JSON，也不保存为研究证据。
本包不强行设置 temperature 或 thinking 参数，不假设实验室透传所有厂商专有选项。
`response_format=json_object` 默认开启；网关若拒绝此参数，教师可将非秘密 `model.json` 的 `json_mode` 改为 false 后创建新研究，本地校验仍然执行。

## 为什么不写死一个未经确认的 Kimi 名称？

用户说明只给出了通用示例，**没有列出 Kimi 的确切 ID**，并要求模型名以 `/v1/models` 返回为准。
所以 `auto-kimi` 只是客户端选择器，绝不会作为上游模型名发送：

```text
读取 Key → GET /models → 从当前 Key 可见的 Kimi ID 中选择 → 保存确切 ID → 启动研究
```

首次选择的确定性规则：优先准确别名 `kimi`，其次 `kimi-latest`，再其次是实验室管理员确认的短别名 `k3`；否则比较列表中 Kimi K 系列的数字版本，按数字降序及名称排序取一个。支持命名空间前缀；排除 base/audio/embedding/rerank 类型。只从服务器已经返回的 ID 中取值，不拼造名称，不回退 GPT/Claude，不声称此排序代表能力最优。

恢复同一次研究时只核对原来 ID 是否还可见，不自动升级。模型列表中没有 Kimi、Key 无效或列表请求返回 503 时明确停止。推理中的 model_not_found 不进行适配器内部重试，外层仍保留原有有限失败门禁。
除已确认的 `k3` 外，不符合可识别命名的内部别名需要管理员提供 `kimi` 映射或准确的 Kimi 型号；不会依据猜测把其他模型当作 Kimi。

**固定 ID 不保证固定上游权重。** `kimi`、`kimi-latest` 等别名的后台映射可能由管理员修改。正式跨组对照建议教师统一指定版本化 ID，并保持实验室渠道映射稳定。

## 教师设置（学生不需要执行）

教师可先看当前账号开放的 Kimi：

```bash
bash start.sh check-api
bash start.sh setup --root "$PWD/.arlab/course"
```

已经知道管理员开放的具体 Kimi 名称时，在准备好的目录中固定它：

```bash
bash start.sh configure --root "$PWD/.arlab/course" \
  --model '<模型列表中的准确Kimi名称>'
```

若管理员提供 HTTPS 地址，也可一次性设置 `--base-url 'https://课程网关/v1'`。学生之后仍只需填写 Key。
`configure` 不读取 Key、不查询模型、不启动研究；真正执行前仍须检查可见性。
其他供应商的高级接口保留，但不再作为学生默认配置路径。

## 明文 HTTP 的边界

**原说明提供的是 HTTP 公网地址，Key、提示和源码内容在这段传输中没有 TLS 加密。**
不落盘、不走代理、不跟随重定向只能减少部分泄露渠道，不能把明文传输变安全。
教师应在管理员认可的受保护网络或加密隧道内使用，或者让管理员提供 HTTPS；VPN 也只有在覆盖了完整风险链路时才保护该传输。

为了实现用户明确指定的地址，本版 `allow_lab_http=true` 只授权准确的 `http://43.143.241.66:18319/v1`。
改变主机、端口或路径不能继承这个例外。其他远程模型地址仍要求 HTTPS，论文调研的 HTTPS/公共地址检查完全不变。
HTTP 风险会在发送 Key 前显示；研究授权与安装授权继续分开，未主动访问实验室后台、用量管理或上游账号。

## 源码定位

- `classroom/kimi.py`：无第三方依赖的模型发现、Kimi 选择、预置连接参数。
- `core/endpoints.py`：模型连接验证和唯一 HTTP 例外；不是通用网络访问授权。
- `classroom/session.py`：隐藏输入、临时环境变量、模型固定与原研究流程连接。
- `core/models.py`：实际 Chat Completions 调用、严格结果校验和错误处理。
- `deployment/tasks.py`：九任务安装预设接入 Kimi；合成演示和低层通用预设保持原用途。

## 本次验收范围

测试使用明确标注的假 Key、协议响应和本地 HTTP 服务。验证了请求路径、鉴权头、JSON 行动、代理绕过、重定向拒绝、无 Kimi 时停止、临时 Key 清理、worker 不继承密钥以及模型固定。
没有用户的真实 Key，**没有进行实验室网关鉴权或 Kimi 实际推理验收**，也没有新的 D4RL 成绩。
模型列表可见不等于 Chat Completions 一定路由成功，最终还要以目标环境的实际响应为准。

## 参考材料

原始实验室说明保存在 [lab-new-api-original.md](lab-new-api-original.md)，其中的 Codex / Claude 示例仅为原始参考，不是本 Harness 的依赖。

外部核对（2026-09-21，仅核对厂商的接口，不证明实验室代理支持全部特性）：

- Kimi 官方 Chat Completions / JSON Mode / 响应字段：<https://platform.kimi.com/docs/api/chat>。
- 实验室端点、API Key 环境变量和模型名验证规则：以用户提供说明第 1、2、3、4 节为准。


## v0.6.2 安装范围

当前使用项目内 uv 虚拟环境；不使用 Docker、不执行 sudo，缓存与下载均由项目包装器放到
源码目录内部。Key 规则、Kimi 发现与 HTTP 风险不变。详见 [本地安装](project-local-install.md)。

当前 v0.6.3 仅增加原生库本地补齐，不改变本文的 Kimi 接口、模型选择或密钥规则。
