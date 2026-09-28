"""One-key Kimi setup against the owner's New API gateway.

Discovery is standard-library-only so `start.sh check-api` works before installing
PyTorch, Docker or D4RL. /models validates visibility, NOT inference capability.
No secret, provider error body, or hidden reasoning is written to a file.
"""
from __future__ import annotations

import getpass
import json
import os
import re
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from arlab.core.endpoints import AUTO_KIMI, LAB_NEW_API_BASE_URL, is_lab_gateway, validate_model_endpoint

KEY_ENV = "NEW_API_KEY"
MAX_MODELS_BYTES = 1_000_000
HTTP_NOTICE = (
    "注意：实验室网关使用明文 HTTP，Key 和研究内容未受 TLS 加密保护。"
    "请在管理员认可的受保护网络/加密隧道内使用；建议管理员提供 HTTPS。"
)


def default_model_settings() -> dict[str, Any]:
    """Deployment defaults, not an assertion that a guessed model ID exists."""
    return {
        "provider": "compatible", "model": AUTO_KIMI,
        "base_url": LAB_NEW_API_BASE_URL, "api_key_env": KEY_ENV,
        "allow_lab_http": True, "max_output_tokens": 16384,
        "timeout_seconds": 300, "retries": 2, "json_mode": True,
    }


def validate_key(key: str) -> str:
    if not key or len(key) > 4096 or any(ord(c) < 33 or ord(c) > 126 for c in key):
        raise ValueError("API Key 不能为空，也不能包含空白、换行或非 ASCII 字符。")
    return key


def is_kimi_id(model: str) -> bool:
    # Recognise Kimi aliases advertised by the lab gateway and namespaced IDs,
    # but not arbitrary short k* names or unrelated Moonshot models.
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}", model):
        return False
    leaf = re.split(r"[/:]", model.lower())[-1]
    if not (leaf in {"kimi", "kimi-latest", "k3"} or re.match(r"kimi-k\d", leaf)):
        return False
    return not any(token in re.split(r"[-_.]", leaf) for token in ("base", "audio", "embedding", "rerank"))


def choose_kimi(model_ids: list[str], requested: str = AUTO_KIMI) -> str:
    """Select only an advertised ID, deterministically; never fall back to GPT/Claude.

    Prefer the literal aliases kimi / kimi-latest / k3 when advertised. Otherwise sort
    returned Kimi K-series by numeric version and then lexically. This is an
    installation policy, NOT a capability ranking. A persisted ID is never upgraded.
    """
    available = sorted(set(m for m in model_ids if isinstance(m, str) and is_kimi_id(m)))
    if requested != AUTO_KIMI:
        if not is_kimi_id(requested):
            raise ValueError("实验室默认接入只选择 Kimi；请管理员配置正确的 Kimi 模型 ID。")
        if requested not in available:
            raise ValueError("已固定的 Kimi 模型不在当前 Key 的 /v1/models 列表中；请联系管理员，不会自动换模型。")
        return requested
    if not available:
        raise ValueError("当前 Key 没有可识别的 Kimi 模型权限；请管理员开放 Kimi 或配置真实别名。没有调用其他模型。")
    for alias in ("kimi", "kimi-latest", "k3"):
        matches = [m for m in available if m.lower() == alias]
        if matches:
            return matches[0]
    def order(model: str):
        match = re.search(r"kimi-k(\d+)(?:\.(\d+))?", model, flags=re.IGNORECASE)
        version = (int(match[1]), int(match[2] or 0)) if match else (0, 0)
        return (-version[0], -version[1], model.lower(), model)
    return min(available, key=order)


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Bearer credentials must not cross a redirect boundary.
        return None


def discover_models(key: str, *, base_url: str = LAB_NEW_API_BASE_URL,
                    allow_lab_http: bool = True, opener=None) -> list[str]:
    """Return advertised model IDs from one bounded GET /models request."""
    validate_model_endpoint(base_url, allow_lab_http=allow_lab_http)
    validate_key(key)
    request = Request(base_url.rstrip("/") + "/models", headers={
        "Authorization": "Bearer " + key, "Accept": "application/json",
        "Accept-Encoding": "identity", "User-Agent": "Autoresearch-Classroom/0.6.1",
    })
    # The lab guide explicitly requires bypassing proxy settings. Never mutate
    # process-global proxy settings or log authorization headers.
    client = opener or build_opener(ProxyHandler({}), _NoRedirect())
    try:
        with client.open(request, timeout=20) as response:
            if response.status != 200:
                raise ValueError("模型列表没有返回 HTTP 200；未调用推理。")
            raw = response.read(MAX_MODELS_BYTES + 1)
        if len(raw) > MAX_MODELS_BYTES:
            raise ValueError("模型列表响应超过大小上限。")
        body = json.loads(raw)
    except HTTPError as exc:
        code = exc.code
        exc.close()
        if code in (401, 403):
            message = "Key 无效或无模型列表权限（401/403），请联系管理员。"
        elif 300 <= code < 400:
            message = "模型接口返回重定向；为保护 Key 已拒绝转发，请管理员确认直达地址。"
        elif code == 503:
            message = "实验室渠道暂不可用（503）；请管理员检查分组和渠道，不会高速重试。"
        else:
            message = f"模型列表请求失败（HTTP {code}）；未切换服务。"
        raise ValueError(message) from None
    except (URLError, TimeoutError, OSError):
        raise ValueError("无法直连实验室模型网关；本请求已禁用环境代理，请检查网络、隧道及服务状态。") from None
    except (UnicodeError, json.JSONDecodeError):
        raise ValueError("模型列表响应不是有效 JSON；未记录响应正文，也未调用推理。") from None
    if not isinstance(body, dict) or not isinstance(body.get("data"), list):
        raise ValueError("/v1/models 响应缺少 data 数组；请管理员检查接口映射。")
    return [item["id"] for item in body["data"]
            if isinstance(item, dict) and isinstance(item.get("id"), str)]


def discover_kimi(key: str, *, base_url: str = LAB_NEW_API_BASE_URL,
                  requested: str = AUTO_KIMI, allow_lab_http: bool = True,
                  opener=None) -> str:
    """Resolve a Kimi selector from one bounded GET /models request."""
    ids = discover_models(key, base_url=base_url, allow_lab_http=allow_lab_http, opener=opener)
    return choose_kimi(ids, requested)


def prepare_model(config):
    """Resolve a selector or recheck an exact lab ID; importing schema is deferred."""
    if config.model != AUTO_KIMI and not is_lab_gateway(config.base_url):
        return config
    if config.model == AUTO_KIMI and config.provider != "compatible":
        raise ValueError("Kimi 预设使用 Chat Completions，不使用 Codex Responses 或 Claude Messages。")
    key = os.environ.get(config.api_key_env, "")
    if config.provider == "responses":
        available = discover_models(key, base_url=config.base_url,
                                    allow_lab_http=config.allow_lab_http)
        if config.model not in available:
            raise ValueError("指定的 Responses 模型不在当前 Key 的 /v1/models 列表中。")
        resolved = config.model
    elif config.provider == "compatible":
        resolved = discover_kimi(key, base_url=config.base_url, requested=config.model,
                                 allow_lab_http=config.allow_lab_http)
    else:
        raise ValueError("实验室 Claude 模型请使用专用 Messages 配置，不经过 Kimi/Codex 预设。")
    return config.model_copy(update={"model": resolved})


def check_api(*, base_url: str = LAB_NEW_API_BASE_URL, model: str = AUTO_KIMI,
              provider: str = "compatible") -> int:
    """Pre-install visibility check for the selected model; never persist the key."""
    if is_lab_gateway(base_url):
        print(HTTP_NOTICE, flush=True)
    key = os.environ.get(KEY_ENV) or os.environ.get("ARLAB_API_KEY", "")
    if not key:
        if not sys.stdin.isatty():
            raise ValueError("请设置 NEW_API_KEY，或在交互终端执行 bash start.sh check-api。")
        key = getpass.getpass("实验室 New API Key（隐藏输入，不保存）：").strip()
    if provider == "compatible":
        selected = discover_kimi(key, base_url=base_url, requested=model,
                                 allow_lab_http=is_lab_gateway(base_url))
    elif provider == "responses":
        if model == AUTO_KIMI:
            raise ValueError("Responses 协议需要明确指定实际模型 ID。")
        available = discover_models(key, base_url=base_url,
                                    allow_lab_http=is_lab_gateway(base_url))
        if model not in available:
            raise ValueError("指定的 Responses 模型不在当前 Key 的 /v1/models 列表中。")
        selected = model
    else:
        raise ValueError("check-api 目前支持 compatible/Kimi 和 responses 模型检查。")
    print(f"Key 可读取模型列表；已配置模型：{selected}（{provider}）")
    print("这只验证模型可见性；没有调用模型推理、下载 D4RL 数据或启动训练。")
    return 0
