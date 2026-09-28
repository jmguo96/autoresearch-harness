"""Offline real uv build/venv/install, with explicit scope in the output."""
from __future__ import annotations
import json, os, sys, hashlib, zipfile
from pathlib import Path
from arlab.deployment.local import local_process
from arlab.deployment.common import Runner, atomic_json
from arlab.deployment.uv_bootstrap import ensure_uv

source = Path(__file__).resolve().parents[2]
root = source / '.arlab' / 'package-validation'
result = {'scope': 'real_uv_offline_build_and_fresh_venv_install_using_existing_Python_NOT_target_legacy_stack',
          'source_python': sys.version, 'downloaded_dependencies': False,
          'reused_host_python_for_build': True}
with local_process(root, source):
    runner = Runner(root / 'execution')
    uv = ensure_uv(root, runner, existing=Path('/opt/pyvenv/bin/uv'))
    result['uv_version'] = runner.run([uv, '--version'], 'version', capture=True)
    runner.run([uv, 'build', '--wheel', '--no-build-isolation', '--no-index', '--python', sys.executable,
                '--out-dir', str(root/'dist'), str(source)], 'uv-build-wheel', cwd=source, timeout=120)
    wheel, = (root/'dist').glob('*.whl')
    env = root/'installed'
    runner.run([uv, 'venv', '--no-project', '--no-python-downloads', '--python', sys.executable, str(env)],
               'uv-new-venv', cwd=root)
    python = str(env/'bin/python')
    runner.run([uv, 'pip', 'install', '--no-index', '--no-deps', '--python', python, str(wheel)],
               'uv-install-package', cwd=root)
    check = runner.run([python, '-I', '-c',
        'import arlab,sys,site,json,importlib.metadata as m; '
        'assert sys.prefix != sys.base_prefix; '
        'assert m.version("autoresearch-lab") == "0.6.2"; '
        'print(json.dumps({"version":arlab.__version__,"prefix":sys.prefix,"base_prefix":sys.base_prefix,'
        '"module":arlab.__file__,"user_site_enabled":site.ENABLE_USER_SITE}))'],
        'installed-import', cwd=root, capture=True)
    result.update(installed=json.loads(check), wheel=str(wheel),
                  wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
                  include_system_site_packages=False)
    with zipfile.ZipFile(wheel) as archive:
        differences = []
        for path in (source/'src/arlab').rglob('*.py'):
            relative=path.relative_to(source/'src').as_posix()
            if archive.read(relative) != path.read_bytes(): differences.append(relative)
        result['wheel_code_mismatches']=differences
        assert not differences
    assert 'include-system-site-packages = false' in (env/'pyvenv.cfg').read_text().lower()
result['passed']=True
atomic_json(source/'evidence/uv-package-validation.json', result)
print(json.dumps(result, indent=2))
