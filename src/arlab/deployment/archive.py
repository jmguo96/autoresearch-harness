"""Extract the MuJoCo release without path traversal or escaping links."""
from __future__ import annotations

import os
import shutil
import tarfile
from pathlib import Path, PurePosixPath


def extract_mujoco(archive: Path, destination: Path) -> None:
    """Destination must be a fresh private staging directory, never an existing install."""
    destination.mkdir(parents=True, exist_ok=False)
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        if len(members) > 20000 or sum(m.size for m in members) > 1_000_000_000:
            raise ValueError("Archive exceeds extraction budget")
        seen = set()
        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "mujoco210":
                raise ValueError("Archive path escapes mujoco210")
            if member.name in seen or not (member.isdir() or member.isfile() or member.issym()):
                raise ValueError("Duplicate path or unsupported archive entry")
            seen.add(member.name)
            if member.issym():
                target = PurePosixPath(member.linkname)
                if target.is_absolute():
                    raise ValueError("Absolute symlink in archive")
                resolved = (root / member.name).parent.joinpath(member.linkname).resolve()
                if not resolved.is_relative_to(root / "mujoco210"):
                    raise ValueError("Archive symlink escapes install")
        # Never extract a regular file through an archive-created symlink.
        for member in members:
            if member.issym():
                continue
            target = root / member.name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                source = tar.extractfile(member)
                if source is None:
                    raise ValueError("Missing archive payload")
                with source, target.open("xb") as stream:
                    shutil.copyfileobj(source, stream)
                os.chmod(target, 0o755 if member.mode & 0o111 else 0o644)
        for member in members:
            if member.issym():
                target = root / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(member.linkname)
    if not (root / "mujoco210/bin/libmujoco210.so").is_file():
        raise ValueError("MuJoCo library is missing from release archive")
