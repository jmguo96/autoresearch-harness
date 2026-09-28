"""HTTPS acquisition with atomic publication and validator-aware resume.

Only approved HTTPS hosts are followed. TLS verification is never disabled.
A locally computed hash records identity; without a trusted expected hash it is
not an independent attestation of the publisher's file.
"""
from __future__ import annotations

import http.client
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable

from .common import atomic_json, exclusive_lock, read_json, sha256


class DownloadError(RuntimeError):
    """Permanent policy/integrity failure; do not hide it with retries."""


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def validate_url(url: str, hosts: set[str]) -> None:
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme != "https" or parsed.hostname not in hosts or
            parsed.username is not None or parsed.password is not None or
            parsed.port not in (None, 443) or parsed.fragment):
        raise DownloadError("URL must use HTTPS on an explicitly approved host")


def _open(url: str, headers: dict[str, str], hosts: set[str], timeout: float):
    opener = urllib.request.build_opener(NoRedirect())
    for _ in range(5):
        validate_url(url, hosts)
        try:
            response = opener.open(urllib.request.Request(url, headers=headers), timeout=timeout)
            return response
        except urllib.error.HTTPError as exc:
            if exc.code not in (301, 302, 303, 307, 308):
                raise
            location = exc.headers.get("Location")
            exc.close()
            if not location:
                raise DownloadError("Redirect has no Location") from exc
            url = urllib.parse.urljoin(url, location)
    raise DownloadError("Too many redirects")


def fetch_https(url: str, destination: Path, *, expected_sha256: str = "",
                max_bytes: int = 2_000_000_000, attempts: int = 4,
                allowed_hosts: set[str] | None = None, timeout: float = 120,
                opener: Callable = _open, sleep: Callable = time.sleep) -> str:
    """Stream a file, resuming only when a validator or trusted hash is available.

    The opener/sleep parameters support transport fault tests without pretending
    that a simulated server is an actual successful internet download.
    """
    hosts = allowed_hosts or {urllib.parse.urlsplit(url).hostname or ""}
    validate_url(url, hosts)
    if expected_sha256 and not re.fullmatch(r"[a-fA-F0-9]{64}", expected_sha256):
        raise ValueError("expected_sha256 must be 64 hexadecimal characters")
    expected_sha256 = expected_sha256.lower()
    if not 1 <= attempts <= 20 or max_bytes < 1:
        raise ValueError("Invalid download budget")
    destination = Path(destination).absolute()
    if os.environ.get("ARLAB_PROJECT_ROOT"):
        from .local import contained_path
        contained_path(destination, Path(os.environ["ARLAB_PROJECT_ROOT"]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise DownloadError("Download destination cannot be a symlink")
    with exclusive_lock(destination.with_name(destination.name + ".lock")):
        return _fetch_locked(url, destination, hosts, expected_sha256, max_bytes,
                             attempts, timeout, opener, sleep)


def _fetch_locked(url, dest, hosts, expected, limit, attempts, timeout, opener, sleep):
    part = dest.with_name(dest.name + ".part")
    state_path = dest.with_name(dest.name + ".part.json")
    receipt = dest.with_name(dest.name + ".download.json")
    for path in (part, state_path, receipt):
        if path.is_symlink():
            raise DownloadError("Download sidecar cannot be a symlink")
    if dest.exists():
        if not dest.is_file() or not 0 < dest.stat().st_size <= limit:
            raise DownloadError("Invalid cached file size")
        actual = sha256(dest)
        if expected and actual != expected:
            raise DownloadError("Existing file checksum mismatch; refusing silent replacement")
        if receipt.exists():
            saved = read_json(receipt)
            if saved.get("sha256") != actual or saved.get("url") != url:
                raise DownloadError("Cached file differs from its acquisition receipt")
        return actual
    last_error = None
    for attempt in range(attempts):
        state = read_json(state_path) if state_path.exists() else {}
        valid_state = (state.get("url") == url and state.get("expected_sha256", "") == expected)
        validator = state.get("validator", "") if valid_state else ""
        offset = part.stat().st_size if part.exists() and valid_state and (validator or expected) else 0
        if offset > limit:
            raise DownloadError("Partial file exceeds download limit")
        headers = {"User-Agent": "Autoresearch-Lab/0.3", "Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
            if validator:
                headers["If-Range"] = validator
        try:
            with opener(url, headers, hosts, timeout) as response:
                status = response.status
                if status not in (200, 206):
                    raise DownloadError(f"Unexpected HTTP status {status}")
                encoding = response.headers.get("Content-Encoding", "identity")
                if encoding not in ("", "identity"):
                    raise DownloadError("Encoded responses are not safe for byte-range resume")
                etag = response.headers.get("ETag", "")
                new_validator = (etag if etag and not etag.startswith("W/")
                                 else response.headers.get("Last-Modified", ""))
                length = response.headers.get("Content-Length")
                response_bytes = int(length) if length is not None else None
                if response_bytes is not None and response_bytes < 0:
                    raise DownloadError("Negative Content-Length")
                if status == 206:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                    if not offset or not match:
                        raise DownloadError("Unsolicited or malformed range response")
                    start, end, total = map(int, match.groups())
                    if start != offset or not start <= end < total:
                        raise DownloadError("Server returned an incorrect resume range")
                    if validator and new_validator != validator:
                        raise DownloadError("Entity validator changed during resume")
                    if response_bytes is not None and response_bytes != end - start + 1:
                        raise DownloadError("Range and Content-Length disagree")
                    # Request is open-ended; a partial response must reach the end.
                    if end != total - 1:
                        raise DownloadError("Incomplete range response")
                    expected_size = total
                else:
                    offset = 0  # If-Range failed or server ignored Range: overwrite, never append.
                    expected_size = response_bytes
                if expected_size is not None and expected_size > limit:
                    raise DownloadError("Remote file exceeds download limit")
                atomic_json(state_path, {"url": url, "validator": new_validator,
                                        "expected_sha256": expected})
                size = offset
                with part.open("ab" if offset else "wb") as stream:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        size += len(block)
                        if size > limit:
                            raise DownloadError("Download exceeds byte budget")
                        stream.write(block)
                    stream.flush()
                    os.fsync(stream.fileno())
                if not size or expected_size is not None and size != expected_size:
                    raise http.client.IncompleteRead(b"", max(0, (expected_size or 1) - size))
            actual = sha256(part)
            if expected and actual != expected:
                # A mismatched partial cannot be reused on a subsequent invocation.
                part.unlink(missing_ok=True)
                state_path.unlink(missing_ok=True)
                raise DownloadError("Downloaded file checksum mismatch")
            os.replace(part, dest)
            atomic_json(receipt, {"url": url, "sha256": actual, "bytes": size,
                                  "expected_sha256": expected,
                                  "integrity": "expected_hash_matched" if expected else "https_and_local_identity"})
            state_path.unlink(missing_ok=True)
            return actual
        except urllib.error.HTTPError as exc:
            last_error = exc
            code = exc.code
            exc.close()
            if code == 416:
                # May have a complete-but-unpublished file, or an obsolete partial.
                # Restart rather than declaring it complete based on size alone.
                part.unlink(missing_ok=True)
                state_path.unlink(missing_ok=True)
            elif code not in (408, 429, 500, 502, 503, 504):
                raise DownloadError(f"HTTP {code}; no insecure or synthetic fallback") from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.IncompleteRead,
                http.client.RemoteDisconnected) as exc:
            last_error = exc
        if attempt + 1 < attempts:
            sleep(min(2 ** attempt, 8))
    raise DownloadError(f"Download failed after {attempts} attempts: {type(last_error).__name__}; partial retained") from last_error
