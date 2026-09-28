"""Signed Debian/Ubuntu packages in a private APT workspace, never system install.

APT is used as a repository verifier and dependency solver. Its configuration,
status (empty), lists, cache, logs, trust store and downloads belong to this
project. We only update that index, SIMULATE dependency resolution, inspect
metadata and print download URLs. dpkg execution is independently disabled.
The actual .deb downloads use our bounded SHA256-verifying HTTPS client.
"""
from __future__ import annotations

import os
import platform
import pwd
import re
import shlex
import shutil
from pathlib import Path
from urllib.parse import urlsplit

from .common import Runner, atomic_json, atomic_text, read_json, sha256
from .download import fetch_https
from .local import contained_path

DEFAULT_PACKAGES = ('libosmesa6-dev', 'libgl-dev', 'libgl1', 'libgomp1')
SUPPORTED = {'ubuntu': {'focal', 'jammy', 'noble'}, 'debian': {'bullseye', 'bookworm', 'trixie'}}
SPEC = re.compile(r'[a-z0-9][a-z0-9+.-]*(?::amd64)?(?:=[A-Za-z0-9.+:~_-]+)?\Z')
# Keep the host loader/libc pair. A relocatable GL overlay is not a whole OS.
HOST_PACKAGES = {'libc6', 'libc-bin', 'libc6-dev', 'libc-dev-bin', 'linux-libc-dev',
                 'gcc-10-base', 'gcc-11-base', 'gcc-12-base', 'gcc-13-base',
                 'gcc-14-base', 'gcc-15-base', 'gcc-16-base'}


def host_distribution(path: Path = Path('/etc/os-release')) -> dict[str, str]:
    fields = {}
    for line in path.read_text().splitlines():
        if '=' in line and not line.startswith('#'):
            key, raw = line.split('=', 1)
            values = shlex.split(raw)
            fields[key] = values[0] if values else ''
    family = fields.get('ID', '')
    suite = fields.get('VERSION_CODENAME', fields.get('UBUNTU_CODENAME', ''))
    if family not in SUPPORTED or suite not in SUPPORTED[family]:
        raise RuntimeError(f'Automatic native .deb provisioning supports exact Debian/Ubuntu releases '
                           f'{SUPPORTED}; host is {family}/{suite}. No cross-release guessing is allowed.')
    if platform.machine() not in ('x86_64', 'AMD64'):
        raise RuntimeError('Native .deb overlay currently supports amd64 only')
    return {'id': family, 'suite': suite, 'version_id': fields.get('VERSION_ID', ''),
            'architecture': 'amd64'}


def repository_lines(host: dict, keyring: Path) -> tuple[list[str], set[str]]:
    suite = host['suite']
    if host['id'] == 'ubuntu':
        origins = [('https://archive.ubuntu.com/ubuntu', suite, 'main universe'),
                   ('https://archive.ubuntu.com/ubuntu', suite + '-updates', 'main universe'),
                   ('https://security.ubuntu.com/ubuntu', suite + '-security', 'main universe')]
    else:
        origins = [('https://deb.debian.org/debian', suite, 'main'),
                   ('https://deb.debian.org/debian', suite + '-updates', 'main'),
                   ('https://security.debian.org/debian-security', suite + '-security', 'main')]
    # deb822 is not needed here. Quote a keyring path containing spaces.
    lines = [f'deb [arch=amd64 signed-by="{keyring}"] {url} {dist} {components}'
             for url, dist, components in origins]
    return lines, {urlsplit(url).hostname for url, _, _ in origins}


def parse_control(text: str) -> list[dict[str, str]]:
    records, item, key = [], {}, ''
    for line in text.splitlines() + ['']:
        if not line.strip():
            if item:
                records.append(item)
            item, key = {}, ''
        elif line[0].isspace() and key:
            item[key] += '\n' + line.strip()
        elif ':' in line:
            key, value = line.split(':', 1)
            item[key] = value.strip()
    return records


def resolution_specs(output: str) -> list[str]:
    result = []
    for line in output.splitlines():
        if line.startswith('Remv '):
            raise ValueError('Unexpected removal in private empty-status resolution')
        if not line.startswith('Inst '):
            continue
        match = re.match(r'^Inst (\S+) \((\S+)\s', line)
        if not match:
            raise ValueError(f'Unrecognized APT resolution: {line[:300]}')
        spec = match[1] + '=' + match[2]
        if not SPEC.fullmatch(spec):
            raise ValueError('Unsupported native package spec')
        result.append(spec)
    if not result or len(result) > 160:
        raise ValueError('Empty or unexpectedly large native dependency closure')
    return sorted(set(result))


class NativeApt:
    """One private resolver; no public arbitrary apt command interface."""

    def __init__(self, root: Path, runner: Runner, host: dict | None = None):
        self.root, self.runner = root, runner
        self.host = host or host_distribution()
        self.workspace = contained_path(root / 'native/apt', root)
        self.config = self.workspace / 'apt.conf'
        self.debs = contained_path(root / 'downloads/native/debs', root)
        self.apt_get = shutil.which('apt-get', path=runner.env.get('PATH'))
        self.apt_cache = shutil.which('apt-cache', path=runner.env.get('PATH'))
        if not self.apt_get or not self.apt_cache:
            raise RuntimeError('apt-get/apt-cache are needed only for private index/metadata '
                               'resolution, not installation. Reuse a native lock + downloaded debs '
                               'from a matching machine, or prepare reviewed native libraries locally.')
        self.allowed_hosts: set[str] = set()
        self.config_hash = ''

    def prepare(self) -> None:
        for relative in ('etc/parts', 'etc/sources.list.d', 'etc/trusted.gpg.d',
                         'etc/preferences.d', 'state/lists/partial', 'cache/archives/partial', 'log'):
            path = contained_path(self.workspace / relative, self.root)
            path.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.debs.mkdir(parents=True, exist_ok=True, mode=0o700)
        key_name = 'ubuntu-archive-keyring.gpg' if self.host['id'] == 'ubuntu' else 'debian-archive-keyring.gpg'
        upstream_key = Path('/usr/share/keyrings') / key_name
        if not upstream_key.is_file():
            raise RuntimeError(f'Missing trusted host archive keyring: {upstream_key}. '
                               'Will not disable signature verification or download an untrusted replacement.')
        keyring = contained_path(self.workspace / 'etc/archive-keyring.gpg', self.root)
        keyring.write_bytes(upstream_key.read_bytes())
        lines, self.allowed_hosts = repository_lines(self.host, keyring)
        atomic_text(self.workspace / 'etc/sources.list', '\n'.join(lines) + '\n')
        for relative in ('etc/main.conf', 'etc/preferences', 'state/status', 'state/extended_states'):
            atomic_text(self.workspace / relative, '')
        settings = {
            'Dir::Etc': self.workspace / 'etc', 'Dir::Etc::main': self.workspace / 'etc/main.conf',
            'Dir::Etc::parts': self.workspace / 'etc/parts',
            'Dir::Etc::sourcelist': self.workspace / 'etc/sources.list',
            'Dir::Etc::sourceparts': self.workspace / 'etc/sources.list.d',
            'Dir::Etc::trusted': keyring, 'Dir::Etc::trustedparts': self.workspace / 'etc/trusted.gpg.d',
            'Dir::Etc::preferences': self.workspace / 'etc/preferences',
            'Dir::Etc::preferencesparts': self.workspace / 'etc/preferences.d',
            'Dir::State': self.workspace / 'state', 'Dir::State::lists': self.workspace / 'state/lists',
            'Dir::State::status': self.workspace / 'state/status',
            'Dir::State::extended_states': self.workspace / 'state/extended_states',
            'Dir::Cache': self.workspace / 'cache', 'Dir::Cache::archives': self.workspace / 'cache/archives',
            'Dir::Cache::pkgcache': self.workspace / 'cache/pkgcache.bin',
            'Dir::Cache::srcpkgcache': self.workspace / 'cache/srcpkgcache.bin',
            'Dir::Log': self.workspace / 'log', 'Dir::Log::Terminal': self.workspace / 'log/term.log',
            'Dir::Log::History': self.workspace / 'log/history.log',
            'Dir::Bin::dpkg': '/bin/false',
            'APT::Architecture': 'amd64', 'APT::Sandbox::User': pwd.getpwuid(os.getuid()).pw_name,
            'APT::Install-Recommends': 'false', 'APT::Install-Suggests': 'false',
            'APT::Get::Download-Only': 'true', 'APT::Get::AllowUnauthenticated': 'false',
            'Acquire::AllowInsecureRepositories': 'false',
            'Acquire::AllowDowngradeToInsecureRepositories': 'false',
            'Acquire::Languages': 'none', 'Acquire::Retries': '2',
            'Acquire::http::Timeout': '30', 'Acquire::https::Timeout': '30',
            'Acquire::https::Verify-Peer': 'true', 'Acquire::https::Verify-Host': 'true',
        }
        if any(any(c in str(v) for c in ('"', '\\', '\n', '\r')) for v in settings.values()):
            raise ValueError('Path contains characters unsupported by APT configuration syntax')
        text = '\n'.join(f'{k} "{v}";' for k, v in settings.items())
        text += '\n#clear APT::Architectures;\nAPT::Architectures { "amd64"; };\n'
        # APT_CONFIG is read BEFORE the fragment paths are chosen. A plain -c alone
        # would be too late and could inherit system Post-Invoke hooks.
        text += '\n'.join('#clear ' + key + ';' for key in (
            'APT::Update::Pre-Invoke', 'APT::Update::Post-Invoke', 'APT::Update::Post-Invoke-Success',
            'DPkg::Pre-Invoke', 'DPkg::Post-Invoke', 'DPkg::Pre-Install-Pkgs', 'Binary')) + '\n'
        atomic_text(self.config, text)
        self.config_hash = sha256(self.config)

    def _command(self, operation: str, specs: list[str] | None = None) -> str:
        specs = specs or []
        if not self.config_hash or sha256(self.config) != self.config_hash:
            raise ValueError('Private APT configuration changed')
        if any(not SPEC.fullmatch(spec) for spec in specs):
            raise ValueError('Invalid native package input')
        if operation == 'index' and not specs:
            argv = [self.apt_get, '-o', 'APT::Update::Error-Mode=any', 'update']
        elif operation == 'resolve' and specs:
            argv = [self.apt_get, '--simulate', '--no-install-recommends', '--no-install-suggests', 'install', *specs]
        elif operation == 'metadata' and len(specs) == 1:
            argv = [self.apt_cache, 'show', '--no-all-versions', *specs]
        elif operation == 'uri' and len(specs) == 1:
            argv = [self.apt_get, '--print-uris', 'download', *specs]
        else:
            raise PermissionError('Only private-index, simulation, metadata and URI operations are allowed')
        previous = self.runner.env.copy()
        try:
            self.runner.env.update(APT_CONFIG=str(self.config), LC_ALL='C', LANG='C')
            # This deliberate internal entry is not the unrestricted Runner.run:
            # no apt install/remove operation can reach it without --simulate.
            return self.runner._run_logged(argv, 'native-apt-' + operation,
                                           cwd=self.debs, timeout=900, capture=True)
        finally:
            self.runner.env.clear()
            self.runner.env.update(previous)

    def resolve(self, packages: tuple[str, ...] = DEFAULT_PACKAGES) -> dict:
        self.prepare()
        self._command('index')
        specs = resolution_specs(self._command('resolve', list(packages)))
        records, excluded = [], []
        for spec in specs:
            package, version = spec.split('=', 1)
            name = package.split(':')[0]
            meta = next((m for m in parse_control(self._command('metadata', [spec]))
                         if m.get('Package') == name and m.get('Version') == version
                         and m.get('Architecture') in ('amd64', 'all')), None)
            if meta is None:
                raise ValueError(f'No exact package metadata for {spec}')
            source = meta.get('Source', name).split()[0]
            if name in HOST_PACKAGES or source == 'glibc':
                excluded.append({'package': name, 'version': version, 'reason': 'use_host_libc_or_base_toolchain'})
                continue
            digest = meta.get('SHA256', '')
            size = int(meta.get('Size', '0'))
            if not re.fullmatch(r'[a-fA-F0-9]{64}', digest) or not 0 < size <= 500_000_000:
                raise ValueError(f'Missing/unsafe SHA256 or size for {spec}')
            uris = [shlex.split(line)[0] for line in self._command('uri', [spec]).splitlines()
                    if line.startswith("'") and shlex.split(line)]
            if len(uris) != 1:
                raise ValueError(f'Expected one signed-index download URI for {spec}')
            uri = uris[0]
            parts = urlsplit(uri)
            if parts.scheme != 'https' or parts.hostname not in self.allowed_hosts or parts.username or parts.password:
                raise ValueError('APT package URI is outside the approved HTTPS repositories')
            records.append({'package': name, 'version': version, 'architecture': meta['Architecture'],
                            'url': uri, 'sha256': digest.lower(), 'size': size,
                            'depends': meta.get('Depends', ''), 'pre_depends': meta.get('Pre-Depends', ''),
                            'file': f'{name}_{digest[:16]}.deb'})
        if sum(r['size'] for r in records) > 1_500_000_000:
            raise ValueError('Native downloads exceed the 1.5 GB safety budget')
        return {'schema': 1, 'host': self.host, 'requested': list(packages), 'packages': records,
                'excluded': excluded, 'allowed_hosts': sorted(self.allowed_hosts),
                'authentication': 'APT signed InRelease/Release index then per-package SHA256',
                'keyring_sha256': sha256(self.workspace / 'etc/archive-keyring.gpg')}


def download_locked(root: Path, lock: dict, *, offline: bool = False) -> list[Path]:
    """Repeatable download. A new run cannot silently choose different packages."""
    allowed = set(lock['allowed_hosts'])
    paths = []
    for record in lock['packages']:
        if not re.fullmatch(r'[a-z0-9][a-z0-9+.-]*_[a-f0-9]{16}\.deb', record['file']):
            raise ValueError('Invalid locked filename')
        dest = contained_path(root / 'downloads/native/debs' / record['file'], root)
        if offline and not dest.is_file():
            raise RuntimeError(f'Offline native package missing: {dest}')
        fetch_https(record['url'], dest, expected_sha256=record['sha256'],
                    allowed_hosts=allowed, max_bytes=record['size'])
        if dest.stat().st_size != record['size']:
            raise ValueError('Native package size mismatch')
        paths.append(dest)
    return paths
