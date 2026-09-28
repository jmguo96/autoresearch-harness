"""Small, deterministic I/O primitives. No model-generated path is trusted."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def canonical(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    def encode(obj: Any) -> Any:
        if isinstance(obj, BaseModel):
            return obj.model_dump(mode="json")
        raise TypeError(f"Unsupported canonical object: {type(obj).__name__}")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False, default=encode)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_write(path: Path, data: str | bytes) -> None:
    """Replace only after a complete write; a crash never leaves half a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data.encode("utf-8") if isinstance(data, str) else data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, canonical(value) + "\n")


def read_json(path: Path, max_bytes: int = 8_000_000) -> Any:
    if path.stat().st_size > max_bytes:
        raise ValueError(f"JSON exceeds {max_bytes} bytes: {path.name}")
    def reject_constant(token: str) -> None:
        raise ValueError(f"Non-finite JSON constant: {token}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def within(root: Path, relative: str) -> Path:
    """Reject absolute paths, traversal, and symlink escapes."""
    if Path(relative).is_absolute():
        raise ValueError("Expected a relative path")
    result = (root / relative).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError("Path escapes its authorized workspace")
    return result


def source_digest() -> str:
    """Bind cached experiments to the installed implementation, not just its version."""
    root = Path(__file__).resolve().parents[1]
    return digest({str(p.relative_to(root)): file_digest(p) for p in sorted(root.rglob("*.py"))})


def directory_digest(path: Path) -> str:
    """Content identity of a portable artifact directory; code is never executed here."""
    if not path.is_dir() or path.is_symlink():
        raise ValueError("Artifact must be a real directory")
    manifest = {}
    for item in sorted(path.rglob("*")):
        if item.is_symlink():
            raise ValueError("Artifact symlinks are forbidden")
        if item.is_file():
            manifest[str(item.relative_to(path))] = file_digest(item)
    if not manifest:
        raise ValueError("Artifact directory is empty")
    return digest(manifest)


def runtime_info() -> dict[str, Any]:
    """Installed dependency identities for reproducibility, not a compatibility claim."""
    import importlib.metadata
    import platform
    versions = {}
    for name in ("torch", "numpy", "pydantic", "h5py", "safetensors", "gym", "D4RL", "mujoco-py"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": versions}
