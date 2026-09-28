"""Model endpoint validation, including one owner-supplied laboratory exception.

The lab guide specifies HTTP, not HTTPS. This is a narrow compatibility exception,
NOT a claim that HTTP protects credentials. Public research tools do not use it.
"""
from __future__ import annotations

from urllib.parse import urlsplit

LAB_NEW_API_BASE_URL = "http://43.143.241.66:18319/v1"
AUTO_KIMI = "auto-kimi"  # A local selector; never sent as a provider model ID.


def is_lab_gateway(value: str) -> bool:
    return value.rstrip("/") == LAB_NEW_API_BASE_URL


def validate_model_endpoint(value: str, *, allow_lab_http: bool = False) -> bool:
    """Return whether the endpoint is loopback; reject credentials and redirects.

    Only the *exact* supplied origin and /v1 prefix may opt into plaintext HTTP.
    A hostname, port or path changed by a candidate never inherits this grant.
    """
    try:
        endpoint = urlsplit(value)
        _ = endpoint.port  # Force malformed-port validation before any request.
        local = endpoint.hostname in {"localhost", "127.0.0.1", "::1"}
        invalid = (
            not isinstance(value, str) or any(ord(c) <= 32 or ord(c) == 127 for c in value)
            or endpoint.scheme not in {"https", "http"} or not endpoint.hostname
            or endpoint.username is not None or endpoint.password is not None
            or bool(endpoint.query) or bool(endpoint.fragment)
        )
        if invalid:
            raise ValueError
        if endpoint.scheme == "http" and not local:
            if not (allow_lab_http and is_lab_gateway(value)):
                raise ValueError
    except (ValueError, TypeError, AttributeError):
        raise ValueError(
            "Model endpoints require HTTPS and cannot contain credentials, query or fragment; "
            "HTTP is allowed only for loopback or the explicitly approved laboratory gateway."
        ) from None
    return local
