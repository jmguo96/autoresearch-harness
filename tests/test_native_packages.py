"""Rootless native provisioning tests.

Locally generated .debs/repository and a tiny compiled ABI fixture are explicitly
NOT official Mesa packages, not MuJoCo and not a D4RL benchmark execution.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

from arlab.deployment.common import Runner, atomic_json, atomic_text, read_json, sha256
from arlab.deployment.local import local_process, project_environment, prepare_local_directories
from arlab.deployment.native_apt import NativeApt, host_distribution, resolution_specs, parse_control, download_locked
from arlab.deployment.native_archive import safe_extract_data, build_overlay, BASE_RUNTIME, validate_overlay
from arlab.deployment.native import ensure_native_prerequisites
from arlab.deployment.manager import plan, parser


def test_default_auto_and_teacher_override():
    from arlab.classroom.bootstrap import parser as student_parser, installation_command
    options = student_parser(show_advanced=True).parse_args([])
    command = installation_command(Path('/tmp/source'), Path('/tmp/source/.arlab/root'), options)
    assert command[command.index('--native-deps') + 1] == 'auto'
    options = student_parser(show_advanced=True).parse_args(['--native-deps', 'check', '--native-offline'])
    command = installation_command(Path('/tmp/source'), Path('/tmp/source/.arlab/root'), options)
    assert command[command.index('--native-deps') + 1] == 'check'
    assert '--native-offline' in command


def test_plan_reports_local_packages_without_system_install(tmp_path):
    result = plan(parser().parse_args([]), tmp_path, tmp_path / '.arlab/root')
    assert result['native_prerequisites']['mode'] == 'auto'
    assert not result['native_prerequisites']['system_install']
    assert not result['native_prerequisites']['runs_maintainer_scripts']
    assert not result['uses_sudo'] and not result['uses_docker']
    assert not (tmp_path / '.arlab').exists()


@pytest.mark.parametrize('family,suite', [('ubuntu','focal'), ('ubuntu','jammy'), ('ubuntu','noble'),
                                          ('debian','bullseye'), ('debian','bookworm'), ('debian','trixie')])
def test_exact_distribution_supported(tmp_path, family, suite):
    file = tmp_path / 'os-release'
    file.write_text(f'ID={family}\nVERSION_CODENAME="{suite}"\nVERSION_ID="test"\n')
    assert host_distribution(file)['suite'] == suite


@pytest.mark.parametrize('content', ['ID=ubuntu\nVERSION_CODENAME=unknown', 'ID=rocky\nVERSION_ID=9',
                                      'ID=debian\nVERSION_CODENAME=testing'])
def test_no_cross_release_guessing(tmp_path, content):
    file=tmp_path/'os-release'; file.write_text(content)
    with pytest.raises(RuntimeError, match='No cross-release'):
        host_distribution(file)


def test_dependency_parser_and_exact_specs():
    assert resolution_specs('Inst libosmesa6 (25.0.7-2+deb13u1 Debian:13 [amd64])\nConf libosmesa6\n') == ['libosmesa6=25.0.7-2+deb13u1']
    assert parse_control('Package: foo\nDepends: bar,\n baz\n\n')[0]['Depends'] == 'bar,\nbaz'
    for output in ('', 'Inst ../../bad (1 test)', 'Remv libc6 [2.36]', 'Inst foo broken'):
        with pytest.raises(ValueError):
            resolution_specs(output)


def test_private_apt_configuration_no_system_config(tmp_path, monkeypatch):
    if not shutil.which('apt-get'):
        pytest.skip('real APT tools unavailable')
    root=tmp_path/'project';root.mkdir()
    prepare_local_directories(root)
    apt=NativeApt(root, Runner(root/'attempt', project_environment(root)))
    apt.prepare()
    config=apt.config.read_text()
    assert str(apt.workspace/'etc/parts') in config
    assert str(apt.workspace/'state/status') in config
    assert 'Dir::Bin::dpkg "/bin/false";' in config
    assert 'AllowUnauthenticated "false"' in config
    assert (apt.workspace/'state/status').read_text() == ''
    captured=[]
    def capture(argv, label, **kwargs):
        captured.append((argv,apt.runner.env.copy(),kwargs))
        return ''
    monkeypatch.setattr(apt.runner, '_run_logged', capture)
    apt._command('resolve', ['libosmesa6-dev'])
    assert '--simulate' in captured[-1][0]
    assert captured[-1][1]['APT_CONFIG']==str(apt.config)
    assert captured[-1][2]['cwd'].is_relative_to(root)
    with pytest.raises(PermissionError):
        apt._command('install', ['libosmesa6-dev'])
    with pytest.raises(ValueError):
        apt._command('resolve', ['--allow-unauthenticated'])
    apt.config.write_text(config+'\n')
    with pytest.raises(ValueError, match='configuration changed'):
        apt._command('index')


def tar_stream(items):
    output=io.BytesIO()
    with tarfile.open(fileobj=output, mode='w') as archive:
        for path,data,link in items:
            item=tarfile.TarInfo(path)
            if link is not None:
                item.type=tarfile.SYMTYPE;item.linkname=link
                archive.addfile(item)
            else:
                item.size=len(data);archive.addfile(item,io.BytesIO(data))
    output.seek(0)
    return output


@pytest.mark.parametrize('path', ['../outside', '/etc/evil', 'usr/lib/../../../../outside'])
def test_archive_traversal_rejected(tmp_path,path):
    with pytest.raises(ValueError,match='Unsafe'):
        safe_extract_data(tar_stream([(path,b'x',None)]),tmp_path/'root')


def test_archive_link_escape_rejected(tmp_path):
    with pytest.raises(ValueError,match='Unsafe'):
        safe_extract_data(tar_stream([('usr/lib/x',b'', '../../../../outside')]),tmp_path/'root')


def test_absolute_symlink_is_relocated_not_host_link(tmp_path):
    safe_extract_data(tar_stream([('lib/libtest.so.1',b'abc',None),
                                 ('usr/lib/libtest.so',b'','/lib/libtest.so.1')]),tmp_path/'root')
    assert (tmp_path/'root/usr/lib/libtest.so').read_bytes()==b'abc'
    assert (tmp_path/'root/usr/lib/libtest.so').resolve().is_relative_to(tmp_path/'root')


def test_surface_does_not_extract_etc_or_maintainer_data(tmp_path):
    safe_extract_data(tar_stream([('etc/ld.so.conf',b'ignored',None), ('usr/bin/evil',b'ignored',None),
                                 ('usr/share/doc/pkg',b'','missing-package'),
                                 ('usr/include/GL/osmesa.h',b'header',None)]),
                      tmp_path/'root',library_surface=True)
    assert not (tmp_path/'root/etc').exists() and not (tmp_path/'root/usr/bin').exists()
    assert (tmp_path/'root/usr/include/GL/osmesa.h').read_bytes()==b'header'


def test_archive_rejects_preexisting_symlink_parent(tmp_path):
    outside=tmp_path/'outside';outside.mkdir()
    root=tmp_path/'root';root.mkdir();(root/'usr').symlink_to(outside)
    with pytest.raises(ValueError,match='symlink'):
        safe_extract_data(tar_stream([('usr/lib/test.so',b'data',None)]),root)
    assert not list(outside.iterdir())


def test_archive_uncompressed_byte_bound(tmp_path):
    with pytest.raises(ValueError,match='byte limit'):
        safe_extract_data(tar_stream([('usr/lib/test',b'x'*32,None)]),tmp_path/'root',max_bytes=8)


@pytest.mark.parametrize('name', ['libc.so.6','libm.so.6','libdl.so.2','libpthread.so.0',
                                  'ld-linux-x86-64.so.2','libnss_dns.so.2'])
def test_do_not_expose_glibc_loader(name):
    assert BASE_RUNTIME.match(name)


@pytest.mark.parametrize('name', ['libOSMesa.so.8','libGL.so.1','libgomp.so.1','libstdc++.so.6','libLLVM.so.19.1'])
def test_expose_relocatable_native_libraries(name):
    assert not BASE_RUNTIME.match(name)


def test_no_auto_prefix_created_before_success(tmp_path):
    prepare_local_directories(tmp_path)
    assert (tmp_path/'native/lib').exists()
    assert not (tmp_path/'native/auto').exists()
    assert not any(':' in str(p) for p in tmp_path.rglob('*'))
    env=project_environment(tmp_path)
    for k in ('CPATH','LIBRARY_PATH','LD_LIBRARY_PATH'):
        assert all(Path(p).is_relative_to(tmp_path) for p in env[k].split(os.pathsep))


def test_check_only_does_not_fetch(tmp_path,monkeypatch):
    import arlab.deployment.native as module
    prepare_local_directories(tmp_path)
    def missing(*args):
        atomic_json(tmp_path/'native-check.json',{'status':'missing_headers_or_libraries'})
        raise RuntimeError('fixture failure')
    monkeypatch.setattr(module,'check_native_prerequisites',missing)
    monkeypatch.setattr('arlab.deployment.native_apt.NativeApt.resolve',lambda *a:pytest.fail('must not fetch'))
    with pytest.raises(RuntimeError,match='fixture failure'):
        ensure_native_prerequisites(tmp_path,Runner(tmp_path/'attempt'),mode='check')


def test_offline_requires_lock(tmp_path,monkeypatch):
    import arlab.deployment.native as module
    prepare_local_directories(tmp_path)
    def missing(*args):
        atomic_json(tmp_path/'native-check.json',{'status':'missing_headers_or_libraries'})
        raise RuntimeError('fixture failure')
    monkeypatch.setattr(module,'check_native_prerequisites',missing)
    with pytest.raises(RuntimeError,match='Offline native setup'):
        ensure_native_prerequisites(tmp_path,Runner(tmp_path/'attempt'),offline=True)


def build_fixture_deb(path: Path, name: str, files: dict, depends: str = '') -> Path:
    pkg=path/name
    (pkg/'DEBIAN').mkdir(parents=True)
    (pkg/'DEBIAN').chmod(0o755)  # Test workspaces can inherit setgid; dpkg-deb rejects it here.
    control=(f'Package: {name}\nVersion: 1.0\nArchitecture: amd64\nMaintainer: Test <test@example.invalid>\n'
             f'Description: explicit native test fixture, not Mesa\n')
    if depends:
        control+='Depends: '+depends+'\n'
    (pkg/'DEBIAN/control').write_text(control)
    # Must NEVER execute while extracting package data.
    post=pkg/'DEBIAN/postinst'
    post.write_text('#!/bin/sh\ntouch "'+str(path/'MAINTAINER_SCRIPT_WAS_EXECUTED')+'"\n')
    post.chmod(0o755)
    for name,data in files.items():
        file=pkg/name;file.parent.mkdir(parents=True,exist_ok=True)
        if isinstance(data,Path):shutil.copyfile(data,file)
        else:file.write_bytes(data)
    result=path/(pkg.name+'.deb')
    subprocess.run(['dpkg-deb','--build','--root-owner-group',str(pkg),str(result)],check=True,capture_output=True)
    return result


def test_real_deb_extract_compile_load_and_idempotent_receipt(tmp_path,monkeypatch):
    if not shutil.which('dpkg-deb') or not shutil.which('cc'):
        pytest.skip('host C compiler/deb archive reader unavailable')
    project=tmp_path/'project';project.mkdir()
    root=project/'.arlab/test';root.mkdir(parents=True)
    fixture=root/'fixture';fixture.mkdir()
    c=fixture/'fixture.c'
    c.write_text('''#include <stddef.h>
void *OSMesaCreateContextExt(unsigned int a,int b,int c,int d,void *e){return (void*)1;}
int OSMesaMakeCurrent(void*a,void*b,unsigned int c,int d,int e){return 1;}
void OSMesaDestroyContext(void*a){}
void glClearColor(float a,float b,float c,float d){}
void glClear(unsigned int a){}
void glFinish(void){}
const unsigned char *glGetString(unsigned int a){return (const unsigned char*)"TEST_ABI_FIXTURE_NOT_MESA";}
''')
    library=fixture/'libfixture.so'
    subprocess.run(['cc','-shared','-fPIC',str(c),'-o',str(library)],check=True)
    header=b'''#include <stddef.h>
typedef void *OSMesaContext;
#define OSMESA_RGBA 1
OSMesaContext OSMesaCreateContextExt(unsigned int,int,int,int,void*);
int OSMesaMakeCurrent(void*,void*,unsigned int,int,int);
void OSMesaDestroyContext(void*);
'''
    gl=b'''#define GL_UNSIGNED_BYTE 1
#define GL_COLOR_BUFFER_BIT 2
#define GL_RENDERER 3
void glClearColor(float,float,float,float);
void glClear(unsigned int);void glFinish(void);
const unsigned char *glGetString(unsigned int);
'''
    deb=build_fixture_deb(fixture,'arlab-native-test',{'usr/include/GL/osmesa.h':header,
        'usr/include/GL/gl.h':gl,'usr/lib/x86_64-linux-gnu/libOSMesa.so':library,
        'usr/lib/x86_64-linux-gnu/libGL.so':library})
    lock={'schema':1,'host':host_distribution(),'allowed_hosts':['packages.example.invalid'],
          'packages':[{'package':'arlab-native-test','version':'1.0','architecture':'amd64',
            'url':'https://packages.example.invalid/test.deb','sha256':sha256(deb),'size':deb.stat().st_size,
            'file':'arlab-native-test_'+sha256(deb)[:16]+'.deb'}],
          'authentication':'TEST_FIXTURE_NOT_OFFICIAL_REPOSITORY'}
    monkeypatch.setattr(NativeApt,'resolve',lambda *args:lock)
    def copy_fixture(url,dest,**kwargs):
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(deb,dest)
        assert sha256(dest)==kwargs['expected_sha256']
        return sha256(dest)
    monkeypatch.setattr('arlab.deployment.native_apt.fetch_https',copy_fixture)
    # Force acquisition even on a machine that already has real Mesa installed.
    import arlab.deployment.native as native
    original=native.check_native_prerequisites
    count=0
    def check(r,runner):
        nonlocal count
        count+=1
        if count==1:
            atomic_json(r/'native-check.json',{'status':'missing_headers_or_libraries'})
            raise RuntimeError('forced missing native fixture')
        return original(r,runner)
    monkeypatch.setattr(native,'check_native_prerequisites',check)
    with local_process(root,project):
        runner=Runner(root/'attempt',project_environment(root))
        result=ensure_native_prerequisites(root,runner)
        assert result['provisioning']=='downloaded_and_extracted_local_debs'
        assert 'TEST_ABI_FIXTURE_NOT_MESA' in result['diagnostic']
        assert not result['system_modified']
        assert not (fixture/'MAINTAINER_SCRIPT_WAS_EXECUTED').exists()
        result=ensure_native_prerequisites(root,runner,offline=True)
        assert result['provisioning']=='reused_verified_local_overlay'
        # A single changed installed byte must not be silently trusted.
        (root/'native/auto/sysroot/usr/include/GL/gl.h').write_text('tampered')
        with pytest.raises(ValueError,match='receipt'):
            ensure_native_prerequisites(root,runner)


def test_real_private_apt_signed_index_and_dependency_resolution(tmp_path):
    """Real apt + a signed local repo, not network or real distro packages."""
    if not all(shutil.which(x) for x in ('apt-get','apt-cache','dpkg-deb','gpg')):
        pytest.skip('APT/GPG tools unavailable')
    root=tmp_path/'project';root.mkdir()
    prepare_local_directories(root)
    runner=Runner(root/'attempt',project_environment(root))
    apt=NativeApt(root,runner);apt.prepare()
    repo=root/'fixture-repo';repo.mkdir()
    packages=[]
    for name,deps in [('arlab-leaf',''),('arlab-parent','arlab-leaf (>= 1.0)')]:
        deb=build_fixture_deb(repo,name,{'usr/lib/'+name+'.txt':b'fixture'},deps)
        packages.append(f'Package: {name}\nVersion: 1.0\nArchitecture: amd64\n'
                        +(f'Depends: {deps}\n' if deps else '')+
                        f'Filename: {deb.name}\nSize: {deb.stat().st_size}\nSHA256: {sha256(deb)}\n'
                        'Description: Explicit local test fixture\n')
    index=repo/'Packages';index.write_text('\n'.join(packages))
    import email.utils
    release=repo/'Release'
    release.write_text('Origin: ARLAB-TEST-FIXTURE\nLabel: ARLAB-TEST-FIXTURE\nSuite: fixture\n'
        'Architectures: amd64\nDate: '+email.utils.formatdate(usegmt=True)+'\n'
        f'SHA256:\n {sha256(index)} {index.stat().st_size} Packages\n')
    gnupg=root/'fixture-gnupg';gnupg.mkdir(mode=0o700)
    env={**runner.env,'GNUPGHOME':str(gnupg)}
    def gpg(*args):
        return subprocess.run(['gpg','--homedir',str(gnupg),'--batch','--pinentry-mode','loopback',
                               '--passphrase','',*args],env=env,capture_output=True,check=True,timeout=30)
    gpg('--quick-generate-key','ARLAB NATIVE TEST <test@example.invalid>','ed25519','sign','0')
    (apt.workspace/'etc/archive-keyring.gpg').write_bytes(gpg('--export').stdout)
    gpg('--yes','--clearsign','--output',str(repo/'InRelease'),str(release))
    # Test-only local transport; production prepare() always writes official HTTPS.
    (apt.workspace/'etc/sources.list').write_text(f'deb [signed-by={apt.workspace}/etc/archive-keyring.gpg] file:{repo} ./\n')
    host_status=sha256(Path('/var/lib/dpkg/status'))
    apt._command('index')
    specs=resolution_specs(apt._command('resolve',['arlab-parent']))
    assert specs==['arlab-leaf=1.0','arlab-parent=1.0']
    meta=parse_control(apt._command('metadata',['arlab-parent=1.0']))
    assert meta[0]['Depends']=='arlab-leaf (>= 1.0)'
    assert 'arlab-parent.deb' in apt._command('uri',['arlab-parent=1.0'])
    assert sha256(Path('/var/lib/dpkg/status'))==host_status
    assert (apt.workspace/'state/status').read_text()==''
    assert not (repo/'MAINTAINER_SCRIPT_WAS_EXECUTED').exists()
    # Destroy the signed index; APT must not accept an unsigned/changed Release.
    (repo/'InRelease').write_text('invalid signature fixture\n')
    with pytest.raises(RuntimeError):
        apt._command('index')
