# 实验室 New API 使用说明

> 本文面向 **API 使用者**。管理员只会向你分发一个 New API Key；你不需要，也不应该获取任何上游 OpenAI / ChatGPT / Anthropic / Claude 账号、OAuth Token 或真实上游 API Key。

---

## 1. 你需要保存的三个信息

本平台当前地址：

```text
服务地址（Host）:
http://43.143.241.66:18319

OpenAI-compatible Base URL:
http://43.143.241.66:18319/v1

API Key:
由管理员单独分发，例如 sk-xxxxxxxx
```

建议在终端中设置：

```bash
export NEW_API_KEY='你的Key'
export NEW_API_HOST='http://43.143.241.66:18319'
export NEW_API_BASE_URL='http://43.143.241.66:18319/v1'
```

> 不要把 Key 发到群聊、提交到 Git、写进公开代码仓库或截图中。

---

## 2. 先检查 Key 是否可用

### 2.1 查看可用模型

```bash
curl "$NEW_API_BASE_URL/models" \
  -H "Authorization: Bearer $NEW_API_KEY"
```

模型名称必须以该接口实际返回的名称为准。

例如管理员可能开放：

```text
gpt-5.6-sol
claude-...
其他模型...
```

不要自行猜测模型名。

### 2.2 查询自己的 Key 用量

注意 `/api/usage/token/` 最后的 `/` 不要省略：

```bash
curl --noproxy 43.143.241.66 "$NEW_API_HOST/api/usage/token/" \
  -H "Authorization: Bearer $NEW_API_KEY"
```
查询时记得关闭proxy，否则会出现time out
该接口显示的是 **New API 为你的 Key 记录的额度与使用量**，不是上游 ChatGPT/Claude 账号的原始账单页面。

---

# 3. 最重要的接口规则

不同模型可能使用不同协议，不要把所有模型都当成 `/v1/chat/completions`。

| 场景 | 接口 | 关键要求 |
|---|---|---|
| 普通 OpenAI-compatible Chat 模型 | `/v1/chat/completions` | `messages` 格式 |
| Codex / 本平台 `gpt-5.6-sol` | `/v1/responses` | **本平台统一要求 `stream=true`** |
| Claude 原生协议 | `/v1/messages` | Anthropic Messages 格式 |
| 查询模型 | `/v1/models` | 使用自己的 New API Key |

---

# 4. 普通 OpenAI-compatible 模型

如果模型走标准 Chat Completions，可以使用 OpenAI SDK。

安装：

```bash
pip install -U openai
```

Python：

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["NEW_API_KEY"],
    base_url="http://43.143.241.66:18319/v1",
)

response = client.chat.completions.create(
    model="<MODEL_NAME>",
    messages=[
        {
            "role": "user",
            "content": "Hello!"
        }
    ],
)

print(response.choices[0].message.content)
```

将 `<MODEL_NAME>` 替换为 `/v1/models` 中实际可用的模型名。

---

# 5. Codex / `gpt-5.6-sol` 直接 API 调用

## 5.1 必须使用 Responses API

本平台当前 Codex 接入规范：

```text
POST /v1/responses
stream = true
input = list
```

不要使用：

```text
/v1/chat/completions
```

否则可能看到：

```text
codex channel: /v1/chat/completions endpoint not supported
```

## 5.2 Python SDK 示例

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["NEW_API_KEY"],
    base_url="http://43.143.241.66:18319/v1",
)

stream = client.responses.create(
    model="gpt-5.6-sol",
    input=[
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Please inspect this bug and suggest a fix."
                }
            ]
        }
    ],
    stream=True,
)

for event in stream:
    if event.type == "response.output_text.delta":
        print(event.delta, end="", flush=True)
```

### 注意

不要写成：

```python
input="Hello"
```

本平台当前 Codex 上游可能返回：

```text
Input must be a list
```

请使用上面示例中的 list 形式。

## 5.3 curl 示例

`-N` 用于关闭 curl 输出缓冲，方便直接查看 SSE 流：

```bash
curl -N "$NEW_API_BASE_URL/responses" \
  -H "Authorization: Bearer $NEW_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-5.6-sol",
    "input": [
      {
        "role": "user",
        "content": [
          {
            "type": "input_text",
            "text": "Hello, please explain what this repository does."
          }
        ]
      }
    ],
    "stream": true
  }'
```

正常返回是流式 SSE 数据，不应该直接把整个 HTTP Body 当成普通 JSON 调用 `response.json()`。

---

# 6. Codex CLI 接入

Codex CLI 可以直接连接本平台，不需要登录管理员提供的任何 ChatGPT 账号。

## 6.1 安装 Codex CLI

Mac / Linux：

```bash
curl -fsSL https://chatgpt.com/codex/install.sh | sh
```

或者：

```bash
npm install -g @openai/codex
```

检查：

```bash
codex --version
```

## 6.2 配置 Key

```bash
export NEW_API_KEY='你的Key'
```

不要把 Key 明文写进 `config.toml`。

## 6.3 配置 `~/.codex/config.toml`

创建或编辑：

```bash
mkdir -p ~/.codex
nano ~/.codex/config.toml
```

写入：

```toml
model = "gpt-5.6-sol"
model_provider = "lab_newapi"

# 可选：如果当前模型支持 reasoning effort
model_reasoning_effort = "high"

[model_providers.lab_newapi]
name = "Lab New API"
base_url = "http://43.143.241.66:18319/v1"
env_key = "NEW_API_KEY"
wire_api = "responses"
```

然后：

```bash
codex
```

Codex CLI 自己会使用 Responses 流式协议，因此在 CLI 中不需要再手工写 `stream=true`。但如果你自己写 HTTP / Python 请求调用 Codex，则仍按本文约定显式设置 `stream=true`。

## 6.4 不要把 `high` 随便拼到模型名后面

例如管理员开放的是：

```text
gpt-5.6-sol
```

不要自行改成：

```text
gpt-5.6-sol-high
```

除非 `/v1/models` 确实返回了这个模型名。

推理强度通常通过参数或 Codex 配置控制，例如：

```toml
model_reasoning_effort = "high"
```

模型名称始终以平台实际返回为准。

---

# 7. Claude 原生 API

New API 同时支持 Anthropic Messages 风格接口。

```bash
curl "$NEW_API_HOST/v1/messages" \
  -H "x-api-key: $NEW_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "content-type: application/json" \
  -d '{
    "model": "<CLAUDE_MODEL>",
    "max_tokens": 1024,
    "messages": [
      {
        "role": "user",
        "content": "Hello!"
      }
    ]
  }'
```

其中 `<CLAUDE_MODEL>` 必须替换成管理员实际开放的 Claude 模型名。

---

# 8. Claude Code 接入

Claude Code 可以把本平台当成 Anthropic-compatible LLM Gateway 使用。

## 8.1 安装 Claude Code

```bash
npm install -g @anthropic-ai/claude-code
```

检查：

```bash
claude --version
```

## 8.2 推荐环境变量配置

Claude Code 的 `ANTHROPIC_BASE_URL` 应指向 **Host 根地址**：

```bash
export ANTHROPIC_BASE_URL="http://43.143.241.66:18319"
```

这里不要写：

```text
http://43.143.241.66:18319/v1
```

因为 Claude Code 会自行请求 Anthropic 风格的 `/v1/messages` 等路径。

推荐使用 Bearer Token 方式：

```bash
export ANTHROPIC_AUTH_TOKEN="$NEW_API_KEY"
```

如果当前 shell 以前登录或配置过其他 Anthropic Key，建议避免凭据冲突：

```bash
unset ANTHROPIC_API_KEY
```

## 8.3 指定 Claude 模型

```bash
export ANTHROPIC_MODEL="<CLAUDE_MODEL>"
```

然后：

```bash
claude
```

也可以：

```bash
claude --model "<CLAUDE_MODEL>"
```

## 8.4 如果平台只给你开放一个 Claude 模型

Claude Code 除主模型外，还可能使用 `sonnet` / `opus` / `haiku` 等模型别名或后台模型。

如果你的 Key 实际只开放一个模型，建议将多个默认模型都指向同一个可用模型：

```bash
export ANTHROPIC_MODEL="<CLAUDE_MODEL>"
export ANTHROPIC_DEFAULT_SONNET_MODEL="<CLAUDE_MODEL>"
export ANTHROPIC_DEFAULT_OPUS_MODEL="<CLAUDE_MODEL>"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="<CLAUDE_MODEL>"
```

如果管理员分别开放了 Sonnet / Opus / Haiku，则分别填写对应的真实模型名：

```bash
export ANTHROPIC_DEFAULT_SONNET_MODEL="<SONNET_MODEL>"
export ANTHROPIC_DEFAULT_OPUS_MODEL="<OPUS_MODEL>"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="<HAIKU_MODEL>"
```

## 8.5 Claude Code 使用 `x-api-key` 的备选方式

如果管理员明确要求使用 Anthropic `x-api-key` 鉴权，也可以：

```bash
unset ANTHROPIC_AUTH_TOKEN
export ANTHROPIC_API_KEY="$NEW_API_KEY"
export ANTHROPIC_BASE_URL="http://43.143.241.66:18319"
```

通常网关场景优先使用前面的 `ANTHROPIC_AUTH_TOKEN`，避免同时设置两套凭据。

---

# 9. 推荐的本地配置方式

## 临时使用

只在当前终端生效：

```bash
export NEW_API_KEY='你的Key'
```

## 长期使用

可以写入 `~/.bashrc` 或 macOS 默认 zsh 的 `~/.zshrc`：

```bash
export NEW_API_KEY='你的Key'
export NEW_API_HOST='http://43.143.241.66:18319'
export NEW_API_BASE_URL='http://43.143.241.66:18319/v1'
```

加载：

```bash
source ~/.bashrc
```

或：

```bash
source ~/.zshrc
```

### 安全建议

如果机器多人共用，不建议把 Key 直接写入 shell 配置。

可以单独创建：

```bash
nano ~/.newapi_env
chmod 600 ~/.newapi_env
```

内容：

```bash
export NEW_API_KEY='你的Key'
export NEW_API_HOST='http://43.143.241.66:18319'
export NEW_API_BASE_URL='http://43.143.241.66:18319/v1'
```

需要时：

```bash
source ~/.newapi_env
```

---

# 10. 常见报错

## 10.1 `401 Unauthorized`

通常表示 Key 错误、Key 已禁用或 Key 无效。

检查：

```bash
curl "$NEW_API_BASE_URL/models" \
  -H "Authorization: Bearer $NEW_API_KEY"
```

## 10.2 `503 model_not_found`

例如：

```text
No available channel for model xxx under group xxx
```

含义：

```text
你的 Key 当前所属分组下，没有可用的对应模型渠道
```

可能原因：模型名写错、管理员没有给该分组开放该模型、当前上游渠道被禁用或模型池暂时不可用。

## 10.3 `codex channel: /v1/chat/completions endpoint not supported`

表示你拿 Codex 渠道调用了 `/v1/chat/completions`。

请改为：

```text
/v1/responses
```

并使用：

```json
"stream": true
```

## 10.4 `Input must be a list`

错误示例：

```json
{
  "input": "Hello"
}
```

本平台推荐：

```json
{
  "input": [
    {
      "role": "user",
      "content": [
        {
          "type": "input_text",
          "text": "Hello"
        }
      ]
    }
  ]
}
```

## 10.5 HTTP 200，但 Python 报 `Expecting value: line 1 column 1`

通常是因为 Codex 返回的是 SSE 流，而代码执行了：

```python
response.json()
```

流式请求应该使用：

```python
requests.post(..., stream=True)
```

然后逐行读取 SSE，或者直接使用 OpenAI SDK：

```python
client.responses.create(..., stream=True)
```

## 10.6 Claude Code 提示模型不存在

先检查：

```bash
curl "$NEW_API_BASE_URL/models" \
  -H "Authorization: Bearer $NEW_API_KEY"
```

确保 `ANTHROPIC_MODEL` 使用平台实际存在的 Claude 模型。如果 Claude Code 后台尝试调用其他别名，再配置 `ANTHROPIC_DEFAULT_SONNET_MODEL`、`ANTHROPIC_DEFAULT_OPUS_MODEL`、`ANTHROPIC_DEFAULT_HAIKU_MODEL`。

## 10.7 Claude Code 仍然走自己的 Claude 订阅

检查：

```bash
echo "$ANTHROPIC_BASE_URL"
echo "$ANTHROPIC_AUTH_TOKEN"
```

修改环境变量后，重新启动 Claude Code 会话。

---

# 11. Prompt Cache / Cached Tokens

部分模型支持 Prompt Cache。

如果响应的 usage 中出现：

```json
{
  "input_tokens_details": {
    "cached_tokens": 12345
  }
}
```

说明该请求存在缓存读取。

需要注意：

- 短 prompt 通常不会出现明显缓存；
- 长、重复且前缀稳定的上下文更容易命中；
- 不要以 TTFT 单独判断是否命中缓存；
- 是否命中应以 `cached_tokens` 为准。

---

# 12. 使用规范

1. 每个人只使用管理员分配给自己的 Key，不共享 Key。
2. 不要尝试获取或使用后台真实上游账号凭据。
3. 不要把 Key 提交到 GitHub / GitLab 等代码仓库。
4. 批量实验请控制并发，避免一个人瞬间占满整个上游池。
5. 模型名称以 `/v1/models` 返回为准。
6. Codex 统一使用 `/v1/responses`，本平台当前要求 `stream=true`。
7. Claude Code 使用 `ANTHROPIC_BASE_URL` 指向本平台 Host，而不是 Anthropic 官方地址。
8. 遇到 `503 model_not_found` 优先联系管理员检查分组与渠道，不要反复高速重试。
9. 如果 Key 泄露，请立即联系管理员废弃并重新生成。

---

# 13. 快速配置速查

## 普通 OpenAI SDK

```text
API Key  = 你的 New API Key
Base URL = http://43.143.241.66:18319/v1
```

## Codex API

```text
Endpoint = http://43.143.241.66:18319/v1/responses
Model    = gpt-5.6-sol
Input    = list
Stream   = true
```

## Codex CLI

`~/.codex/config.toml`：

```toml
model = "gpt-5.6-sol"
model_provider = "lab_newapi"

[model_providers.lab_newapi]
name = "Lab New API"
base_url = "http://43.143.241.66:18319/v1"
env_key = "NEW_API_KEY"
wire_api = "responses"
```

终端：

```bash
export NEW_API_KEY='你的Key'
codex
```

## Claude Code

```bash
export NEW_API_KEY='你的Key'
export ANTHROPIC_BASE_URL='http://43.143.241.66:18319'
export ANTHROPIC_AUTH_TOKEN="$NEW_API_KEY"
export ANTHROPIC_MODEL='<CLAUDE_MODEL>'

claude
```

---

# 14. 参考文档

- New API 官方 API 使用说明  
  https://docs.newapi.pro/en/docs/guide/feature-guide/user/api

- New API 应用接入  
  https://docs.newapi.pro/en/docs/apps

- OpenAI Codex  
  https://developers.openai.com/codex

- OpenAI Codex GitHub  
  https://github.com/openai/codex

- Claude Code 环境变量  
  https://code.claude.com/docs/en/env-vars

- Claude Code 模型配置  
  https://code.claude.com/docs/en/model-config

---

## 最后更新

2026-09-01

> 本 README 针对当前实验室 New API 部署约定编写。若管理员后续调整服务地址、模型名、分组或协议映射，应以管理员最新通知为准。
