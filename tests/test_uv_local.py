"""Local-install contract tests. Network fixtures are not a D4RL installation.

The uv integration test really creates an isolated venv and installs a locally
constructed wheel. It never installs packages into the test runner interpreter.
"""
from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import pytest

from arlab.deployment.common import Runner, public_environment, read_json
from arlab.deployment.local import (
    contained_path, local_process, prepare_local_directories, project_environment,
)
from arlab.deployment.manager import Deployer, parser, plan, validate_options
from arlab.deployment.uv_bootstrap import UV_VERSION, _extract_uv, ensure_uv

SOURCE = Path(__file__).resolve().parents[1]


def test_paths_are_pure_and_below_root(tmp_path):
    root = tmp_path / "project" / ".arlab" / "local"
    mapping = project_environment(root)
    assert not root.exists()
    for key in ("HOME", "UV_CACHE_DIR", "UV_PYTHON_INSTALL_DIR", "UV_TOOL_DIR", "XDG_CACHE_HOME",
                "TMPDIR", "PIP_CACHE_DIR", "TORCH_HOME", "CUDA_CACHE_PATH", "MPLCONFIGDIR", "HF_HOME"):
        assert Path(mapping[key]).is_relative_to(root)
    assert mapping["UV_LINK_MODE"] == "copy"
    assert mapping["UV_NO_CONFIG"] == "1"
    assert mapping["PIP_REQUIRE_VIRTUALENV"] == "1"


@pytest.mark.parametrize("fragment", ["../outside", "../../outside"])
def test_output_cannot_escape_project(tmp_path, fragment):
    project = tmp_path / "project"
    project.mkdir()
    with pytest.raises(ValueError):
        contained_path(project / fragment, project)


def test_deployer_rejects_external_root_before_writes(tmp_path):
    destination = tmp_path / "outside"
    args = parser().parse_args(["--root", str(destination)])
    with pytest.raises(ValueError, match="inside the project"):
        validate_options(args, SOURCE)
    assert not destination.exists()


@pytest.mark.parametrize("relative", ["cache", "tmp", "home", "python", "tools"])
def test_managed_directory_symlink_rejected(tmp_path, relative):
    root = tmp_path / "local"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / relative).symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        prepare_local_directories(root)
    assert not list(outside.iterdir())


def test_environment_scoped_restored_and_ignores_other_venv(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    root = project / ".arlab/local"
    other = tmp_path / "other-env"
    home = tmp_path / "original-home"
    home.mkdir()
    (home / ".bashrc").write_text("untouched")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("VIRTUAL_ENV", str(other))
    monkeypatch.setenv("PATH", str(other / "bin") + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("UV_SYSTEM_PYTHON", "true")
    monkeypatch.setenv("PIP_TARGET", str(other / "site-packages"))
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "global-cache"))
    monkeypatch.setenv("NEW_API_KEY", "fake-owner-key")
    before = dict(os.environ)
    # Force tempfile's global cache to demonstrate that it is reset in scope.
    tempfile.gettempdir()
    with local_process(root, project):
        assert "VIRTUAL_ENV" not in os.environ and "UV_SYSTEM_PYTHON" not in os.environ
        assert "PIP_TARGET" not in os.environ
        assert str(other / "bin") not in os.environ["PATH"].split(os.pathsep)
        assert os.environ["NEW_API_KEY"] == "fake-owner-key"
        assert "NEW_API_KEY" not in public_environment()
        with tempfile.NamedTemporaryFile() as file:
            assert Path(file.name).is_relative_to(root / "tmp")
        Path.home().joinpath("local-marker").write_text("local only")
    assert dict(os.environ) == before
    assert sorted(p.name for p in home.iterdir()) == [".bashrc"]
    assert (home / ".bashrc").read_text() == "untouched"
    assert not (tmp_path / "global-cache").exists()
    assert not other.exists()


@pytest.mark.parametrize("command", ["sudo", "doas", "su", "apt", "apt-get", "dpkg", "yum", "dnf", "pacman", "brew", "docker", "podman"])
def test_installer_rejects_system_and_container_commands(tmp_path, monkeypatch, command):
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("must reject before launching"))
    with pytest.raises(PermissionError):
        Runner(tmp_path).run([command, "anything"], "forbidden")


def test_default_plan_uv_local_only(tmp_path):
    args = parser().parse_args(["--suite9"])
    value = plan(args, SOURCE, tmp_path)
    assert value["system_packages"] == []
    assert not value["uses_sudo"] and not value["uses_docker"]
    assert value["installation"] == "uv_project_virtualenv"
    assert value["code_mode"] == "trusted_local"
    assert len(value["datasets"]) == 9
    with pytest.raises(SystemExit):
        parser().parse_args(["--code-mode", "docker"])


def test_native_tools_missing_fails_without_installing(tmp_path, monkeypatch):
    from arlab.deployment.native import check_native_prerequisites
    monkeypatch.setattr("arlab.deployment.native.shutil.which", lambda *a, **k: None)
    class NoCommands:
        env = {}
        def run(self, *a, **k):
            pytest.fail("no installation or compiler call should occur")
    with pytest.raises(RuntimeError, match="Missing existing native tools"):
        check_native_prerequisites(tmp_path, NoCommands())
    assert read_json(tmp_path / "native-check.json")["system_modified"] is False


def test_runtime_install_commands_all_target_uv_venv(tmp_path, monkeypatch):
    root = tmp_path / "managed"
    root.mkdir()
    args = parser().parse_args(["--profile", "smoke", "--python", sys.executable])
    deploy = Deployer(SOURCE, root, args)
    calls = []
    def fake_run(argv, label, **kwargs):
        calls.append((argv, label, kwargs))
        if "venv" in argv and "runtime-uv-venv" == label:
            (root / "venv/bin").mkdir(parents=True)
            (root / "venv/bin/python").write_text("test interpreter placeholder")
        return ""
    monkeypatch.setattr(deploy.runner, "run", fake_run)
    monkeypatch.setattr("arlab.deployment.manager.ensure_uv", lambda *a, **k: str(root / "tools/uv/bin/uv"))
    deploy.runtime()
    venv = [x for x in calls if x[1] == "runtime-uv-venv"][0][0]
    assert venv[1] == "venv" and venv[-1] == str(root / "venv")
    installs = [x[0] for x in calls if x[0][1:3] == ["pip", "install"]]
    assert len(installs) == 4
    for argv in installs:
        assert argv[0].endswith("/tools/uv/bin/uv")
        assert argv[argv.index("--python") + 1] == str(root / "venv/bin/python")
        assert "--system" not in argv and "--user" not in argv
    assert not any(argv[1:4] == ["-m", "pip", "install"] for argv, _, _ in calls)


def test_uv_extract_only_binary_never_archive_paths(tmp_path):
    wheel = tmp_path / "uv.whl"
    with zipfile.ZipFile(wheel, "w") as z:
        z.writestr("../../evil", "never extracted")
        z.writestr("uv-0.10.0.data/scripts/uv", b"safe-test-content")
    target = tmp_path / "uv"
    _extract_uv(wheel, target)
    assert target.read_bytes() == b"safe-test-content"
    assert not (tmp_path.parent / "evil").exists()


def test_uv_extract_rejects_link(tmp_path):
    wheel = tmp_path / "uv.whl"
    with zipfile.ZipFile(wheel, "w") as z:
        item = zipfile.ZipInfo("uv-0.10.0.data/scripts/uv")
        item.external_attr = (stat.S_IFLNK | 0o777) << 16
        z.writestr(item, "outside")
    with pytest.raises(ValueError, match="Invalid uv executable"):
        _extract_uv(wheel, tmp_path / "uv")


def test_scoped_worker_does_not_leak_caches_or_key(tmp_path, monkeypatch):
    from arlab.core.jobs import LocalBackend, DockerBackend
    from arlab.algorithms.policy import policy_process_environment
    project = tmp_path / "project"
    project.mkdir()
    root = project / ".arlab/test"
    monkeypatch.setenv("NEW_API_KEY", "fake-do-not-pass")
    monkeypatch.setenv("DOCKER_HOST", "unix:///elsewhere")
    with local_process(root, project):
        body = """import os,tempfile,json
from pathlib import Path
root=Path(os.environ['ARLAB_LOCAL_ROOT'])
assert 'NEW_API_KEY' not in os.environ
assert not any(k.startswith('DOCKER_') for k in os.environ)
for k in ['HOME','TMPDIR','XDG_CACHE_HOME','MPLCONFIGDIR','CUDA_CACHE_PATH','TORCH_HOME']:
    assert Path(os.environ[k]).is_relative_to(root), (k, os.environ[k])
with tempfile.NamedTemporaryFile() as f:
    assert Path(f.name).is_relative_to(root)
print('local worker passed')
"""
        result = LocalBackend().run([sys.executable, "-c", body], root / "job", root / "job/log", 20)
        assert result.exit_code == 0, (root / "job/log").read_text()
        env = policy_process_environment(str(root / "tmp/policy"), "trusted_local")
        assert env["TMPDIR"] == str(root / "tmp") and "NEW_API_KEY" not in env
        with pytest.raises(PermissionError, match="Docker"):
            DockerBackend("unused", [], root)
        with pytest.raises(PermissionError, match="Docker"):
            policy_process_environment(str(root / "tmp/policy"), "docker")


def test_lowlevel_download_cannot_write_elsewhere(tmp_path, monkeypatch):
    from arlab.deployment.download import fetch_https
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("ARLAB_PROJECT_ROOT", str(project))
    with pytest.raises(ValueError, match="inside the project"):
        fetch_https("https://example.org/data", tmp_path / "elsewhere/data",
                    opener=lambda *a, **k: pytest.fail("must fail before network"))
    assert not (tmp_path / "elsewhere").exists()


def test_research_rate_cache_local(tmp_path):
    from arlab.research.network import PublicHTTP
    from arlab.core.schema import ResearchAccess
    from arlab.core.store import Store
    project = tmp_path / "project"
    project.mkdir()
    root = project / "local"
    with local_process(root, project):
        store = Store(root / "research.sqlite3")
        try:
            http = PublicHTTP(ResearchAccess(), store)
            assert http.limiter_dir.is_relative_to(root)
        finally:
            store.close()


@pytest.mark.integration
def test_real_uv_venv_and_offline_wheel(tmp_path):
    original = shutil.which("uv")
    if not original:
        pytest.skip("uv binary unavailable: network install is not performed by tests")
    if not subprocess.check_output([original, "--version"], text=True).startswith("uv " + UV_VERSION + " "):
        # Releases may print no commit suffix.
        if subprocess.check_output([original, "--version"], text=True).strip() != "uv " + UV_VERSION:
            pytest.skip("this integration test requires the pinned uv version")
    project = tmp_path / "project"
    project.mkdir()
    root = project / "local"
    before = Path(original).read_bytes()
    with local_process(root, project):
        runner = Runner(root / "commands")
        uv = ensure_uv(root, runner, existing=Path(original))
        assert Path(uv).is_relative_to(root)
        venv = root / "venv"
        runner.run([uv, "venv", "--no-project", "--no-python-downloads", "--python", sys.executable, str(venv)], "test-uv-venv")
        wheel = root / "downloads" / "arlab_uv_probe-1.0-py3-none-any.whl"
        with zipfile.ZipFile(wheel, "w") as z:
            z.writestr("arlab_uv_probe.py", "VALUE = 'installed by uv into project venv'\n")
            z.writestr("arlab_uv_probe-1.0.dist-info/METADATA", "Metadata-Version: 2.1\nName: arlab-uv-probe\nVersion: 1.0\n")
            z.writestr("arlab_uv_probe-1.0.dist-info/WHEEL", "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
            z.writestr("arlab_uv_probe-1.0.dist-info/RECORD", "")
        python = str(venv / "bin/python")
        runner.run([uv, "pip", "install", "--python", python, "--no-index", str(wheel)], "test-uv-install")
        output = runner.run([python, "-c", "import arlab_uv_probe,sys; print(arlab_uv_probe.VALUE); print(sys.prefix)"],
                            "test-installed-import", capture=True)
        assert "installed by uv" in output and str(venv) in output
        assert "include-system-site-packages = false" in (venv / "pyvenv.cfg").read_text()
        assert ensure_uv(root, runner) == uv
        Path(uv).write_bytes(b"changed")
        with pytest.raises(ValueError, match="receipt"):
            ensure_uv(root, runner)
    assert Path(original).read_bytes() == before
