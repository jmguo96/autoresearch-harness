"""Independent HTTP model adapters; no coding CLI, SDK session, or hidden harness."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
from pydantic import BaseModel

from arlab.core.endpoints import AUTO_KIMI, is_lab_gateway, validate_model_endpoint
from arlab.core.io import canonical
from arlab.core.schema import ModelConfig


def schema_prompt(schema: type[BaseModel]) -> str:
    """Drop presentation-only Pydantic labels, not types/constraints/required fields.

    Local validation still uses the full model. The compact wire specification is
    sent once; tool arguments and recipe schemas are not repeated in user context.
    """
    def compact(value):
        if isinstance(value, dict):
            return {k: compact(v) for k, v in value.items() if k not in {"title", "default"}}
        if isinstance(value, list):
            return [compact(x) for x in value]
        return value
    return "\nReturn one JSON object conforming to this schema:\n" + canonical(compact(schema.model_json_schema()))


@dataclass(frozen=True)
class ModelReply:
    payload: dict[str, Any]
    input_tokens: int = 0
    output_tokens: int = 0
    attempts: int = 1


class ModelClient(Protocol):
    def complete(self, system: str, context: dict[str, Any], schema: type[BaseModel]) -> ModelReply: ...


class ModelBusy(RuntimeError):
    """The configured provider is temporarily saturated, not misconfigured."""


def _responses_sse(content: bytes) -> tuple[str, int, int]:
    """Extract public output text and usage from a bounded Responses SSE body."""
    chunks: list[str] = []
    tokens_in = tokens_out = 0
    for raw in content.splitlines():
        if not raw.startswith(b"data:"):
            continue
        data = raw[5:].strip()
        if not data or data == b"[DONE]":
            continue
        event = json.loads(data)
        kind = event.get("type")
        if kind == "response.output_text.delta":
            chunks.append(str(event.get("delta", "")))
        elif kind == "response.completed":
            usage = event.get("response", {}).get("usage", {})
            tokens_in = int(usage.get("input_tokens", 0) or 0)
            tokens_out = int(usage.get("output_tokens", 0) or 0)
        elif kind in ("error", "response.failed", "response.incomplete"):
            raise ValueError(f"Responses stream ended with {kind}")
    if not chunks:
        raise ValueError("Responses stream contained no output text")
    return "".join(chunks), tokens_in, tokens_out


class HTTPModel:
    """A bounded request; provider errors never silently fall back to the demo policy.

    Retries are explicit in the returned count. Callers reserve the worst-case
    token/call budget *before* entering this method. Never log request headers.
    """
    def __init__(self, config: ModelConfig, transport: httpx.BaseTransport | None = None):
        self.config = config
        self.transport = transport

    def complete(self, system: str, context: dict[str, Any], schema: type[BaseModel]) -> ModelReply:
        config = self.config
        key = os.environ.get(config.api_key_env, "")
        url = config.base_url.rstrip("/")
        validate_model_endpoint(url, allow_lab_http=config.allow_lab_http)
        if config.model == AUTO_KIMI:
            raise ValueError("Resolve the Kimi selector with start.sh before making a model call")
        schema_hint = schema_prompt(schema)
        if config.provider == "anthropic":
            headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
            payload = {"model": config.model, "max_tokens": config.max_output_tokens,
                       "system": system + schema_hint,
                       "messages": [{"role": "user", "content": canonical(context)}]}
            endpoint = url + "/messages"
        elif config.provider == "compatible":
            headers = {"Authorization": f"Bearer {key}"} if key else {}
            payload = {"model": config.model, "max_tokens": config.max_output_tokens, "stream": False,
                       "messages": [{"role": "system", "content": system + schema_hint},
                                    {"role": "user", "content": canonical(context)}]}
            if config.json_mode:
                payload["response_format"] = {"type": "json_object"}
            endpoint = url + "/chat/completions"
        elif config.provider == "responses":
            headers = {"Authorization": f"Bearer {key}"} if key else {}
            payload = {
                "model": config.model,
                "max_output_tokens": config.max_output_tokens,
                "input": [
                    {"role": "developer", "content": [
                        {"type": "input_text", "text": system + schema_hint},
                    ]},
                    {"role": "user", "content": [
                        {"type": "input_text", "text": canonical(context)},
                    ]},
                ],
                "stream": True,
            }
            endpoint = url + "/responses"
        else:
            raise ValueError("HTTPModel requires an HTTP provider")
        last_error = "No attempt made"
        with httpx.Client(timeout=config.timeout_seconds, transport=self.transport,
                          follow_redirects=False, trust_env=False) as client:
            for attempt in range(config.retries + 1):
                retry_delay = min(2**attempt, 4)
                try:
                    response = client.post(endpoint, headers=headers, json=payload)
                    response.raise_for_status()
                    if len(response.content) > 2_000_000:
                        raise ValueError("Model response exceeds size limit")
                    body = None if config.provider == "responses" else response.json()
                    if config.provider == "anthropic":
                        assert isinstance(body, dict)
                        text = "".join(x.get("text", "") for x in body["content"] if x.get("type") == "text")
                        usage = body.get("usage", {})
                        tokens_in, tokens_out = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
                    elif config.provider == "responses":
                        text, tokens_in, tokens_out = _responses_sse(response.content)
                    else:
                        assert isinstance(body, dict)
                        choice = body["choices"][0]
                        if choice.get("finish_reason") == "length":
                            raise ValueError("Model response was truncated")
                        text = choice["message"]["content"]
                        usage = body.get("usage", {})
                        tokens_in, tokens_out = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
                    obj = json.loads(text)
                    validated = schema.model_validate(obj)
                    return ModelReply(validated.model_dump(mode="json"), int(tokens_in), int(tokens_out), attempt + 1)
                except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
                    # Do not persist provider response bodies (may contain secrets or private inputs).
                    last_error = type(exc).__name__
                    if isinstance(exc, httpx.HTTPStatusError):
                        status = exc.response.status_code
                        # The numeric status is safe and materially more useful than
                        # the exception class. Never retain the response body here.
                        last_error = f"HTTP {status}"
                        if status in (401, 404):
                            break
                        if status == 403:
                            # The laboratory gateway has returned several different
                            # bodies (and occasionally no JSON body) for the same
                            # per-key/channel concurrency limit. The key and model
                            # were already resolved through /models at session start,
                            # so a 403 from this gateway's inference endpoint is a
                            # recoverable capacity signal. Do not depend on unstable
                            # provider text and never persist the response body.
                            lab_capacity = config.provider in ("compatible", "responses") and is_lab_gateway(url)
                            if not lab_capacity:
                                break
                            last_error = "HTTP 403 laboratory capacity limit"
                            retry_delay = min(60 * (2**attempt), 180)
                        if status == 400:
                            # The lab gateway can route the same Kimi alias through
                            # heterogeneous channels: an identical request has been
                            # observed to fail and then succeed. Retry only this exact
                            # approved gateway. On the retry, omit optional JSON mode;
                            # the local schema validator remains mandatory.
                            if not (config.provider == "compatible" and is_lab_gateway(url)):
                                break
                            payload.pop("response_format", None)
                    # A configured channel missing on New API is not fixed by repeated calls.
                    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 503:
                        try:
                            error = exc.response.json().get("error", {})
                            if isinstance(error, dict) and (
                                error.get("code") == "model_not_found"
                                or "No available channel" in str(error.get("message", ""))
                            ):
                                last_error = "model_not_found; ask the laboratory administrator"
                                break
                        except (ValueError, AttributeError):
                            pass
                    if attempt < config.retries:
                        time.sleep(retry_delay)
        if last_error == "HTTP 403 laboratory capacity limit":
            raise ModelBusy("Laboratory model capacity remained busy after bounded retries")
        raise RuntimeError(f"Model request failed ({last_error}); no fallback provider was used")
