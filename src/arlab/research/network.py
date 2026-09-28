"""Bounded public-HTTPS transport with exact host grants and DNS-pinned connections.

No cookies, proxies, filesystem URLs, automatic redirects or candidate-supplied
headers. Credentials, when supplied by a provider adapter, never cross redirects.
Test transports can be injected by the host; they are not exposed as agent actions.
"""
from __future__ import annotations
import contextlib
import http.client
import ipaddress
import os
import socket
import ssl
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urljoin
from typing import Callable
from arlab.core.io import digest
from arlab.core.schema import ResearchAccess
from arlab.core.store import Store, BudgetExceeded


class ResearchError(RuntimeError):
    pass


@dataclass(frozen=True)
class Response:
    url: str
    status: int
    headers: dict[str, str]
    body: bytes


class _PinnedTLS(http.client.HTTPSConnection):
    def __init__(self, host: str, address: str, timeout: float):
        super().__init__(host, port=443, timeout=timeout, context=ssl.create_default_context())
        self.address = address

    def connect(self):
        raw = socket.create_connection((self.address, 443), self.timeout)
        try:
            self.sock = self._context.wrap_socket(raw, server_hostname=self.host)
        except BaseException:
            raw.close()
            raise


def validate_url(url: str, hosts: list[str], *, resolver=socket.getaddrinfo) -> tuple[str, str, str]:
    if len(url) > 8000 or any(ord(c) < 32 for c in url) or "\\" in url:
        raise PermissionError("Malformed research URL")
    parsed = urlsplit(url)
    try:
        port = parsed.port
        host = (parsed.hostname or "").encode("idna").decode("ascii").lower()
    except (ValueError, UnicodeError) as exc:
        raise PermissionError("Malformed host/port") from exc
    if (parsed.scheme != "https" or parsed.username is not None or parsed.password is not None
            or port not in (None, 443) or host not in hosts or not host):
        raise PermissionError("Research access requires HTTPS:443 and an exactly owner-approved host")
    addresses = {x[4][0] for x in resolver(host, 443, type=socket.SOCK_STREAM)}
    if not addresses:
        raise ResearchError("Research host did not resolve")
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if not ip.is_global or (getattr(ip, "ipv4_mapped", None) and not ip.ipv4_mapped.is_global):
            raise PermissionError("Research URL resolves to a private, reserved, or local address")
    target = parsed.path or "/"
    if parsed.query:
        target += "?" + parsed.query
    return host, sorted(addresses)[0], target


class PublicHTTP:
    def __init__(self, policy: ResearchAccess, store: Store, *, deadline: Callable[[], float | None] | None = None,
                 resolver=socket.getaddrinfo, exchange=None, limiter_dir: Path | None = None):
        self.policy, self.store, self.deadline = policy, store, deadline
        self.resolver, self.exchange = resolver, exchange
        self.limiter_dir = limiter_dir or Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "arlab/research-rate"

    def remaining(self) -> float:
        deadline = self.deadline() if self.deadline else None
        left = self.policy.timeout_seconds if deadline is None else min(self.policy.timeout_seconds, deadline - time.time())
        if left <= 0:
            raise BudgetExceeded("Campaign deadline reached before a research request")
        return left

    @contextlib.contextmanager
    def _rate_limit(self, host: str):
        # Managed launches share an arXiv lock within the project-local cache.
        # Multiple projects/users/hosts require owner-coordinated aggregate limiting.
        import fcntl
        group = "arxiv" if host in {"arxiv.org", "export.arxiv.org"} else host
        interval = 3.1 if group == "arxiv" else 1.0
        self.limiter_dir.mkdir(parents=True, exist_ok=True)
        with (self.limiter_dir / (digest(group)[:16] + ".lock")).open("a+") as lock:
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    self.remaining()
                    time.sleep(0.05)
            try:
                lock.seek(0)
                value = lock.read().strip()
                wait = max(0.0, float(value or "0") + interval - time.time())
                if wait > self.remaining():
                    raise BudgetExceeded("Insufficient remaining time for research rate limit")
                time.sleep(wait)
                lock.seek(0); lock.truncate(); lock.write(str(time.time())); lock.flush()
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _once(self, url: str, headers: dict[str, str]) -> Response:
        self.remaining()
        host, address, target = validate_url(url, self.policy.allowed_hosts, resolver=self.resolver)
        self.remaining()  # DNS resolution may have consumed the remaining time.
        rid = "net-" + uuid.uuid4().hex
        self.store.reserve(rid, {"research_requests": 1}, {"research_requests": self.policy.max_requests})
        remaining_bytes = self.policy.max_download_bytes - int(self.store.totals().get("research_bytes", 0))
        maximum = min(self.policy.response_bytes, remaining_bytes)
        if maximum <= 0:
            raise BudgetExceeded("Research download byte budget exhausted")
        request_headers = {"User-Agent": self.policy.user_agent, "Accept-Encoding": "identity", **headers}
        with self._rate_limit(host):
            if self.exchange is not None:
                response = self.exchange(url, request_headers, self.remaining(), maximum)
                if len(response.body) > maximum:
                    raise ResearchError("Research response exceeds byte limit")
                self.store.reserve(rid + "-body", {"research_bytes": len(response.body)},
                                   {"research_bytes": self.policy.max_download_bytes})
                return response
            connection = _PinnedTLS(host, address, self.remaining())
            collected = bytearray()
            try:
                connection.request("GET", target, headers=request_headers)
                response = connection.getresponse()
                result_headers = {k.lower(): v for k, v in response.getheaders()}
                if result_headers.get("content-encoding", "identity").lower() not in ("identity", ""):
                    raise ResearchError("Unexpected compressed response; unbounded decompression is disabled")
                length = result_headers.get("content-length")
                if length and int(length) > maximum:
                    raise ResearchError("Research response exceeds byte limit")
                block_index = 0
                while True:
                    if connection.sock:
                        connection.sock.settimeout(self.remaining())
                    block = response.read(min(65536, maximum - len(collected) + 1))
                    if not block:
                        break
                    self.store.reserve(rid + f"-bytes-{block_index}", {"research_bytes": len(block)},
                                       {"research_bytes": self.policy.max_download_bytes})
                    block_index += 1
                    collected.extend(block)
                    if len(collected) > maximum:
                        raise ResearchError("Research response exceeds byte limit")
                if length is not None and len(collected) != int(length):
                    raise ResearchError("Incomplete response body; source snapshot not accepted")
                return Response(url, response.status, result_headers, bytes(collected))
            finally:
                connection.close()

    def get(self, url: str, headers: dict[str, str] | None = None) -> Response:
        if not self.policy.enabled:
            raise PermissionError("Online research is disabled by the owner; cached sources can still be read")
        current, credentials = url, headers or {}
        for redirects in range(4):
            response = None
            for attempt in range(self.policy.retries + 1):
                try:
                    response = self._once(current, credentials)
                except PermissionError:
                    raise
                except (OSError, http.client.HTTPException) as exc:
                    if attempt == self.policy.retries:
                        raise ResearchError(f"Research network failed ({type(exc).__name__}); no fabricated fallback") from exc
                else:
                    if response.status not in (429, 500, 502, 503, 504):
                        break
                    if attempt == self.policy.retries:
                        raise ResearchError(f"Research provider unavailable (HTTP {response.status})")
                delay = min(2 ** attempt, 4)
                if response and response.headers.get("retry-after", "").isdigit():
                    delay = max(delay, int(response.headers["retry-after"]))
                if delay > 30 or delay > self.remaining():
                    raise ResearchError("Provider requested a longer retry delay; defer this tool call")
                time.sleep(delay)
            if response is None:
                raise ResearchError("Research request produced no response")
            if response.status in (301, 302, 303, 307, 308):
                if not response.headers.get("location"):
                    raise ResearchError("Redirect missing Location")
                current = urljoin(current, response.headers["location"])
                credentials = {}  # never forward auth on redirect
                continue
            if response.status != 200:
                raise ResearchError(f"Research source returned HTTP {response.status}; not an empty search")
            return response
        raise ResearchError("Too many research redirects")
