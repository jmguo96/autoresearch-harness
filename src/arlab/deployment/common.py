"""Auditable deployment I/O: atomic records, process locks and bounded commands."""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def atomic_text(path: Path, value: str, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def read_json(path: Path) -> Any:
    if path.stat().st_size > 8_000_000:
        raise ValueError("Deployment JSON exceeds size limit")
    return json.loads(path.read_text(encoding="utf-8"))


@contextlib.contextmanager
def exclusive_lock(path: Path) -> Iterator[None]:
    """Advisory POSIX lock; the inode is intentionally retained after unlock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("Lock must not be a symlink")
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another operation owns {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def public_environment(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Do not pass model credentials to pip/build hooks or environment checks."""
    allowed = {
        "HOME", "PATH", "LANG", "LC_ALL", "TERM", "TMPDIR", "SSL_CERT_FILE",
        "SSL_CERT_DIR", "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY", "https_proxy",
        "http_proxy", "no_proxy", "CUDA_VISIBLE_DEVICES",
    }
    env = {k: v for k, v in os.environ.items() if k in allowed}
    env.update(PYTHONUNBUFFERED="1", PYTHONNOUSERSITE="1", PIP_DISABLE_PIP_VERSION_CHECK="1",
               PIP_CONFIG_FILE=os.devnull, CUBLAS_WORKSPACE_CONFIG=":4096:8",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONHASHSEED="0")
    from .local import inherit_local_environment
    inherit_local_environment(env)
    env.update(extra or {})
    return env


class Runner:
    """No shell expansion; command output lives in a private, bounded log."""
    def __init__(self, root: Path, env: dict[str, str] | None = None):
        self.root = root
        self.env = public_environment(env)
        self.counter = 0

    def run(self, argv: list[str], label: str, *, cwd: Path | None = None,
            timeout: int = 3600, capture: bool = False) -> str:
        # This distribution never invokes privilege escalation, package managers or Docker.
        forbidden = {"sudo", "doas", "su", "apt", "apt-get", "dpkg", "yum", "dnf", "pacman", "brew", "docker", "podman"}
        if not argv or Path(argv[0]).name in forbidden:
            raise PermissionError("System package managers, privilege escalation and containers are disabled")
        return self._run_logged(argv, label, cwd=cwd, timeout=timeout, capture=capture)

    def _run_logged(self, argv: list[str], label: str, *, cwd: Path | None = None,
                    timeout: int = 3600, capture: bool = False) -> str:
        """Shared logging engine. NativeApt uses it ONLY after validating its fixed
        download/index/simulation operations; ordinary callers use run().
        """
        self.counter += 1
        log = self.root / "logs" / f"{self.counter:03d}-{label}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{label}] running; log: {log}", flush=True)
        start = time.monotonic()
        with log.open("wb") as stream:
            os.chmod(log, 0o600)
            process = subprocess.Popen(argv, cwd=cwd or self.root, env=self.env, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                while process.poll() is None:
                    if time.monotonic() - start > timeout or log.stat().st_size > 50_000_000:
                        raise TimeoutError(f"Command exceeded time/log budget: {label}")
                    time.sleep(0.1)
            except BaseException:
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        process.wait()
                raise
        if process.returncode:
            raise RuntimeError(f"{label} exited {process.returncode}; inspect {log}")
        return log.read_text(encoding="utf-8", errors="replace").strip() if capture else ""
