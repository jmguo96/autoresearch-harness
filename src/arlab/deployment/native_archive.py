"""Extract .deb DATA without dpkg install, maintainer scripts or host writes.

The resulting sysroot is private. Only GL/KHR/OpenMP headers and shared libraries
are exposed to compilers/loader. In particular we never expose a downloaded libc
or ELF interpreter through LD_LIBRARY_PATH.
"""
from __future__ import annotations

import os
import posixpath
import re
import shutil
import subprocess
import tarfile
import threading
from pathlib import Path, PurePosixPath

from .common import Runner, atomic_json, read_json, sha256
from .local import contained_path

BASE_RUNTIME = re.compile(r'^(?:ld(?:-|\.so)|ld-linux|lib(?:c|m|dl|pthread|rt|resolv|nss_[^.]+|'
                          r'util|anl|BrokenLocale|thread_db)\.(?:so|a)|libc_nonshared)')


def archive_path(raw: str) -> PurePosixPath:
    path = PurePosixPath(raw)
    if path.is_absolute() or '..' in path.parts or '\\' in raw or '\x00' in raw:
        raise ValueError(f'Unsafe native archive path: {raw!r}')
    return path


def safe_extract_data(stream, destination: Path, *, max_bytes: int = 3_000_000_000, library_surface: bool = False) -> None:
    """Only ordinary files/directories and in-tree links; never tar.extractall."""
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    total, links = 0, []
    with tarfile.open(fileobj=stream, mode='r|') as archive:
        for index, member in enumerate(archive):
            if index > 150_000:
                raise ValueError('Native archive has too many members')
            rel = archive_path(member.name)
            if str(rel) == '.':
                continue
            if library_surface and not (rel.parts[0] == 'lib' or tuple(rel.parts[:2]) in
                                        (('usr', 'lib'), ('usr', 'include')) or str(rel) == 'usr'):
                # No /etc files, executables, doc links or maintainer state.
                continue
            target = contained_path(destination.joinpath(*rel.parts), destination)
            if member.isdev() or member.isfifo() or not (member.isdir() or member.isfile() or member.issym() or member.islnk()):
                raise ValueError('Unsupported native archive member')
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True, mode=0o700)
            elif member.isfile():
                if member.size < 0 or member.size > 800_000_000:
                    raise ValueError('Native archive file too large')
                total += member.size
                if total > max_bytes:
                    raise ValueError('Native extraction byte limit reached')
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                data = archive.extractfile(member)
                if data is None:
                    raise ValueError('Missing native archive data')
                if target.exists():
                    # Only identical overlaps may occur when merging dependency packages.
                    import hashlib
                    expected = hashlib.sha256()
                    for block in iter(lambda: data.read(1024 * 1024), b''):
                        expected.update(block)
                    if not target.is_file() or sha256(target) != expected.hexdigest():
                        raise ValueError(f'Conflicting package files: {rel}')
                else:
                    with target.open('xb') as output:
                        shutil.copyfileobj(data, output, 1024 * 1024)
                    if target.stat().st_size != member.size:
                        raise ValueError('Truncated native archive member')
                    # Do not restore ownership, setuid/setgid or world-writable modes.
                    target.chmod(0o700 if member.mode & 0o111 else 0o600)
            else:
                name = member.linkname
                if '\\' in name or '\x00' in name:
                    raise ValueError('Invalid native archive link')
                if member.islnk():
                    link_rel = archive_path(name)
                else:
                    normalized = posixpath.normpath(name.lstrip('/') if name.startswith('/')
                                                    else posixpath.join(str(rel.parent), name))
                    link_rel = archive_path(normalized)
                link_target = destination.joinpath(*link_rel.parts)
                if link_target == target:
                    raise ValueError('Self-referencing native link')
                links.append((target, link_target, member.islnk()))
    # No archive-created symlink may redirect a subsequent file write.
    for target, link_target, hard in links:
        contained_path(target.parent, destination, allow_parent=True)
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if target.is_symlink():
            if target.resolve() == link_target.resolve():
                continue
            raise ValueError('Conflicting native symlinks')
        if target.exists():
            raise ValueError('Native link conflicts with a file')
        if hard:
            contained_path(link_target, destination)
            if not link_target.is_file():
                raise ValueError('Missing native hardlink target')
            os.link(link_target, target)
        else:
            # Absolute package links are relocated INSIDE this extraction root.
            target.symlink_to(os.path.relpath(link_target, target.parent))
    for target, _, _ in links:
        if not target.resolve().is_relative_to(destination.resolve()):
            raise ValueError('Native link escapes the private sysroot')


def extract_deb(deb: Path, destination: Path, root: Path, runner: Runner, *, max_bytes: int = 3_000_000_000) -> None:
    tool = shutil.which('dpkg-deb', path=runner.env.get('PATH'))
    if not tool:
        raise RuntimeError('dpkg-deb (archive reader, NOT dpkg install) is required for native deb extraction')
    log = contained_path(root / 'native/logs' / (deb.stem + '.log'), root)
    log.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Only the data archive goes to the parser. preinst/postinst/triggers are not read or run.
    with log.open('wb') as error:
        proc = subprocess.Popen([tool, '--fsys-tarfile', str(deb)], stdout=subprocess.PIPE,
                                stderr=error, cwd=root, env=runner.env)
        watchdog = threading.Timer(180, proc.kill)
        watchdog.start()
        try:
            assert proc.stdout is not None
            safe_extract_data(proc.stdout, destination, library_surface=True, max_bytes=max_bytes)
            proc.stdout.close()
            if proc.wait(timeout=30) != 0:
                raise RuntimeError(f'dpkg-deb data extraction failed; see {log}')
        finally:
            watchdog.cancel()
            if proc.poll() is None:
                proc.kill()
            proc.wait()


def build_overlay(prefix: Path) -> None:
    """Link only the native build/runtime surface; host libc remains in charge."""
    sysroot = prefix / 'sysroot'
    include, libraries = prefix / 'include', prefix / 'lib'
    include.mkdir()
    libraries.mkdir()
    for name in ('GL', 'KHR'):
        source = sysroot / 'usr/include' / name
        if source.is_dir():
            (include / name).symlink_to(os.path.relpath(source, include), target_is_directory=True)
    omp_headers = list(sysroot.glob('usr/lib/gcc/*/*/include/omp.h'))
    if len(omp_headers) > 1:
        raise ValueError('Ambiguous GCC OpenMP headers')
    if omp_headers:
        (include / 'omp.h').symlink_to(os.path.relpath(omp_headers[0], include))
    locations = [sysroot / directory for directory in
                 ('usr/lib/x86_64-linux-gnu', 'lib/x86_64-linux-gnu', 'usr/lib', 'lib')]
    # GCC keeps its unversioned libgomp linker name in this directory.
    locations.extend(sorted(sysroot.glob('usr/lib/gcc/x86_64-linux-gnu/*')))
    for location in locations:
        if not location.is_dir():
            continue
        for source in sorted(location.glob('*.so*')):
            if BASE_RUNTIME.match(source.name):
                continue
            real = source.resolve()
            if not real.is_relative_to(sysroot.resolve()) or not real.is_file():
                raise ValueError(f'Unresolved native shared library: {source.name}')
            with real.open('rb') as stream:
                if stream.read(4) != b'\x7fELF':
                    raise ValueError(f'Not an ELF library; refusing absolute linker script: {source.name}')
            target = libraries / source.name
            if target.exists():
                if sha256(target) != sha256(real):
                    raise ValueError(f'Conflicting runtime libraries: {source.name}')
                continue
            # Resolve package symlinks, then link directly to an in-tree ordinary file.
            target.symlink_to(os.path.relpath(real, libraries))
    if not (include / 'GL/osmesa.h').is_file() or not (libraries / 'libOSMesa.so').is_file():
        raise ValueError('Resolved packages do not provide the OSMesa development interface')


def inventory(prefix: Path) -> dict:
    result = {}
    for path in sorted(prefix.rglob('*')):
        if path.name == '.arlab-native.json':
            continue
        name = str(path.relative_to(prefix))
        if path.is_symlink():
            if not path.resolve().is_relative_to(prefix.resolve()) or not path.exists():
                raise ValueError(f'Broken or escaping native link: {name}')
            result[name] = {'link': os.readlink(path)}
        elif path.is_file():
            result[name] = {'sha256': sha256(path), 'size': path.stat().st_size}
    return result


def validate_overlay(prefix: Path, lock_sha256: str) -> None:
    saved = read_json(prefix / '.arlab-native.json')
    if saved.get('lock_sha256') != lock_sha256 or saved.get('files') != inventory(prefix):
        raise ValueError('Project native libraries differ from their install receipt; refusing silent reuse')
