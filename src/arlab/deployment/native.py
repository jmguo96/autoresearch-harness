"""Check native libraries, or provision a verified project-local .deb overlay.

No sudo, system dpkg installation, maintainer scripts, ldconfig, drivers or
system package databases are modified. The host C/C++ toolchain is reused.
"""
from __future__ import annotations

import os
import re
import shutil
import uuid
from pathlib import Path

from .common import Runner, atomic_json, atomic_text, read_json, sha256, exclusive_lock
from .local import contained_path

PROBE = r'''#include <GL/osmesa.h>
#include <GL/gl.h>
#include <omp.h>
#include <stdio.h>
int main(void) {
    unsigned char pixels[4 * 4 * 4] = {0};
    OSMesaContext ctx = OSMesaCreateContextExt(OSMESA_RGBA, 16, 0, 0, NULL);
    if (!ctx) { fprintf(stderr,"OSMesa context creation failed\n"); return 2; }
    if (!OSMesaMakeCurrent(ctx, pixels, GL_UNSIGNED_BYTE, 4, 4)) return 3;
    glClearColor(1,0,0,1); glClear(GL_COLOR_BUFFER_BIT); glFinish();
    printf("renderer=%s; openmp_threads=%d\n", glGetString(GL_RENDERER), omp_get_max_threads());
    OSMesaDestroyContext(ctx);
    return omp_get_max_threads()<1;
}
'''


def check_native_prerequisites(root: Path, runner: Runner) -> dict:
    """Read-only with respect to host; compile/run probes live in root/tmp."""
    tools = {name: shutil.which(name, path=runner.env.get('PATH')) for name in ('git', 'cc', 'c++')}
    missing = [name for name, path in tools.items() if not path]
    result = {'tools': tools, 'system_modified': False, 'local_prefix': str(root / 'native')}
    if missing:
        result.update(status='missing_tools', missing=missing)
        atomic_json(root / 'native-check.json', result)
        raise RuntimeError('Missing existing native tools: ' + ', '.join(missing) +
                           '; local library downloads do not provide a complete C/C++ toolchain. '
                           'Provide a compatible toolchain under ' + str(root / 'native/bin') + '.')
    source, binary = root / 'tmp/native-probe.c', root / 'tmp/native-probe'
    atomic_text(source, PROBE)
    try:
        runner.run([tools['cc'], str(source), '-fopenmp', '-lOSMesa', '-lGL', '-o', str(binary)],
                   'native-headers-and-link', cwd=root, timeout=60)
        diagnostic = runner.run([str(binary)], 'native-library-load', cwd=root, timeout=60, capture=True)
    except (RuntimeError, OSError, TimeoutError) as exc:
        result.update(status='missing_headers_or_libraries', error=str(exc))
        atomic_json(root / 'native-check.json', result)
        raise RuntimeError('Native OpenMP/GL/OSMesa compile or context test failed; inspect native-* logs.') from exc
    result.update(status='passed', diagnostic=diagnostic)
    atomic_json(root / 'native-check.json', result)
    return result


def _required_packages(root: Path, runner: Runner) -> tuple[str, ...]:
    from .native_apt import DEFAULT_PACKAGES
    source, binary = root / 'tmp/openmp-probe.c', root / 'tmp/openmp-probe'
    atomic_text(source, '#include <omp.h>\nint main(void){return omp_get_max_threads()<1;}\n')
    try:
        runner.run(['cc', str(source), '-fopenmp', '-o', str(binary)], 'native-openmp', cwd=root, timeout=60)
        runner.run([str(binary)], 'native-openmp-load', cwd=root, timeout=30)
        return DEFAULT_PACKAGES
    except (OSError, RuntimeError, TimeoutError):
        # GCC's omp.h belongs to its version-matched libgcc-N-dev package. Never
        # put a downloaded glibc include tree ahead of the host compiler headers.
        macros = runner.run(['cc', '-dM', '-E', '-x', 'c', os.devnull],
                            'native-compiler-kind', cwd=root, capture=True)
        if '__clang__' in macros or '__GNUC__' not in macros:
            raise RuntimeError('OpenMP auto-repair currently requires a GCC toolchain. '
                               'Provide version-matched OpenMP headers/runtime in native/include and native/lib.')
        version = runner.run(['cc', '-dumpversion'], 'native-compiler-version', cwd=root, capture=True)
        if not re.fullmatch(r'\d+(?:\.\d+)*', version):
            raise RuntimeError('Cannot identify compatible OpenMP headers for the host compiler')
        return (*DEFAULT_PACKAGES, f'libgcc-{version.split(".")[0]}-dev')


def ensure_native_prerequisites(root: Path, runner: Runner, *, mode: str = 'auto', offline: bool = False) -> dict:
    """Missing libraries trigger local acquisition, not a request for sudo."""
    if mode not in ('auto', 'check'):
        raise ValueError('native mode must be auto or check')
    from .native_archive import build_overlay, extract_deb, inventory, validate_overlay
    from .native_apt import NativeApt, download_locked, host_distribution
    prefix = contained_path(root / 'native/auto', root)
    lock_path = contained_path(root / 'native/packages.lock.json', root)
    with exclusive_lock(root / 'native/.prepare.lock'):
        # Verify installed files BEFORE loading them into a probe process.
        if prefix.exists():
            if not lock_path.exists():
                raise ValueError('Existing automatic native prefix has no package lock')
            lock = read_json(lock_path)
            if lock['host'] != host_distribution():
                raise ValueError('Native package lock belongs to a different OS/architecture')
            validate_overlay(prefix, sha256(lock_path))
        try:
            result = check_native_prerequisites(root, runner)
            result['provisioning'] = 'reused_verified_local_overlay' if prefix.exists() else 'already_available'
            atomic_json(root / 'native-check.json', result)
            return result
        except RuntimeError:
            state = read_json(root / 'native-check.json')
            if state['status'] == 'missing_tools' or mode == 'check':
                raise
            if prefix.exists():
                raise RuntimeError('Verified local native files failed the compile/load probe. '
                                   'Inspect native logs for ABI/driver/compiler errors; do not replace system libc.')
        print('Native libraries missing: resolving signed packages into this project (no system installation).', flush=True)
        host = host_distribution()
        if lock_path.exists():
            lock = read_json(lock_path)
            if lock['host'] != host:
                raise ValueError('Native package lock belongs to a different OS/architecture')
        else:
            if offline:
                raise RuntimeError('Offline native setup needs an existing packages.lock.json and downloaded debs')
            lock = NativeApt(root, runner, host).resolve(_required_packages(root, runner))
            atomic_json(lock_path, lock)
        debs = download_locked(root, lock, offline=offline)
        stage = contained_path(root / 'native' / ('auto.pending-' + uuid.uuid4().hex[:10]), root)
        stage.mkdir(mode=0o700)
        previous = runner.env.copy()
        try:
            sysroot = stage / 'sysroot'
            for deb in debs:
                print('Extracting native data: ' + deb.name, flush=True)
                used = sum(p.stat().st_size for p in sysroot.rglob('*') if p.is_file() and not p.is_symlink()) if sysroot.exists() else 0
                extract_deb(deb, sysroot, root, runner, max_bytes=3_000_000_000 - used)
            build_overlay(stage)
            runner.env['CPATH'] = os.pathsep.join([str(stage / 'include'), str(root / 'native/include')])
            runner.env['LIBRARY_PATH'] = os.pathsep.join([str(stage / 'lib'), str(root / 'native/lib'), str(root / 'native/lib64')])
            runner.env['LD_LIBRARY_PATH'] = runner.env['LIBRARY_PATH']
            result = check_native_prerequisites(root, runner)
            atomic_json(stage / '.arlab-native.json', {'schema': 1, 'lock_sha256': sha256(lock_path),
                'files': inventory(stage), 'host': host, 'system_modified': False,
                'validation': 'C compile/link, OpenMP execution, OSMesa context creation and rendering'})
            os.replace(stage, prefix)
        finally:
            runner.env.clear()
            runner.env.update(previous)
            if stage.exists():
                shutil.rmtree(stage)
        # Paths have changed from staging to stable: verify the REAL final paths too.
        result = check_native_prerequisites(root, runner)
        result.update(provisioning='downloaded_and_extracted_local_debs', lock=str(lock_path),
                      packages=len(debs), local_prefix=str(prefix), system_modified=False)
        atomic_json(root / 'native-check.json', result)
        return result
