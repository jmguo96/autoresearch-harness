"""Get a pinned uv binary *inside* the deployment, without pip/curl|sh/global install.

An existing matching uv is copied read-only from PATH. Otherwise a pinned PyPI
wheel is downloaded, SHA256-checked against registry metadata, and ONLY its uv
binary is extracted. Registry-provided checksums are not independent signatures.
"""
from __future__ import annotations

import os
import platform
import re
import shutil
import stat
import uuid
import zipfile
from pathlib import Path

from .common import Runner, atomic_json, read_json, sha256
from .download import fetch_https
from .local import contained_path

UV_VERSION = "0.10.0"


def _wheel_platform() -> str:
    machine = platform.machine().lower()
    if platform.system() == "Linux" and machine in ("x86_64", "amd64"):
        return "manylinux_2_17_x86_64"
    raise RuntimeError("Automatic uv bootstrap in this edition supports Linux x86_64")


def _extract_uv(wheel: Path, destination: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        members = [x for x in archive.infolist() if x.filename.endswith(".data/scripts/uv")]
        if len(members) != 1:
            raise ValueError("Pinned uv wheel has no unique executable")
        member = members[0]
        kind = stat.S_IFMT(member.external_attr >> 16)
        if kind not in (0, stat.S_IFREG) or not 0 < member.file_size <= 200_000_000:
            raise ValueError("Invalid uv executable in wheel")
        # No extractall(): archive paths never determine the output destination.
        with archive.open(member) as src, destination.open("xb") as dst:
            shutil.copyfileobj(src, dst, 1024 * 1024)


def ensure_uv(root: Path, runner: Runner, *, existing: Path | None = None) -> str:
    tool = contained_path(root / "tools/uv/bin/uv", root)
    marker = tool.with_suffix(".json")
    if marker.is_symlink():
        raise ValueError("uv receipt cannot be a symlink")
    if tool.exists():
        saved = read_json(marker) if marker.exists() else {}
        if saved.get("version") != UV_VERSION or saved.get("sha256") != sha256(tool):
            raise ValueError("Existing uv binary does not match its install receipt")
        return str(tool)
    tool.parent.mkdir(parents=True, exist_ok=True)
    staging = tool.with_name(".uv-" + uuid.uuid4().hex)
    acquired_from = ""
    try:
        found = existing or shutil.which("uv", path=runner.env.get("PATH"))
        if found:
            source = Path(found).resolve()
            if not source.is_file() or source.stat().st_size > 200_000_000:
                raise ValueError("Expected a regular uv executable")
            shutil.copyfile(source, staging)
            staging.chmod(0o700)
            version = runner.run([str(staging), "--version"], "uv-local-version", capture=True, timeout=30)
            if re.match(rf"^uv {re.escape(UV_VERSION)}(?:\s|$)", version):
                acquired_from = "copied_existing_binary"
            else:
                staging.unlink()
                if existing:
                    raise ValueError(f"Supplied uv must be version {UV_VERSION}; got {version}")
        if not acquired_from:
            metadata = root / "downloads" / f"uv-{UV_VERSION}.metadata.json"
            fetch_https(f"https://pypi.org/pypi/uv/{UV_VERSION}/json", metadata,
                        allowed_hosts={"pypi.org"}, max_bytes=2_000_000)
            records = read_json(metadata)
            tag = _wheel_platform()
            files = [x for x in records["urls"] if x["packagetype"] == "bdist_wheel"
                     and tag in x["filename"] and x["filename"].endswith(".whl") and not x.get("yanked")]
            if len(files) != 1 or records["info"]["version"] != UV_VERSION:
                raise ValueError("Cannot identify unique pinned uv wheel for this host")
            selected = files[0]
            filename = selected["filename"]
            if Path(filename).name != filename or not re.fullmatch(r"[A-Za-z0-9_.-]+", filename):
                raise ValueError("Unsafe uv wheel filename")
            wheel = root / "downloads" / filename
            fetch_https(selected["url"], wheel, allowed_hosts={"files.pythonhosted.org"},
                        expected_sha256=selected["digests"]["sha256"], max_bytes=100_000_000)
            _extract_uv(wheel, staging)
            staging.chmod(0o700)
            version = runner.run([str(staging), "--version"], "uv-downloaded-version", capture=True, timeout=30)
            if not re.match(rf"^uv {re.escape(UV_VERSION)}(?:\s|$)", version):
                raise ValueError("Downloaded uv version does not match the pin")
            acquired_from = "pypi_wheel_verified_against_registry_sha256"
        os.replace(staging, tool)
        atomic_json(marker, {"version": UV_VERSION, "sha256": sha256(tool), "acquired_from": acquired_from})
        return str(tool)
    finally:
        staging.unlink(missing_ok=True)
