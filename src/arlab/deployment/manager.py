"""Project-local uv deployment: no Docker, no sudo, no system package changes.

Downloads, Python, venv, caches and temporary files stay under a source-tree
deployment root. Missing GL/OSMesa/OpenMP libraries are provisioned as private .deb data; no system install.
This is dependency isolation, not a sandbox for generated candidate Python.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from arlab.classroom.constants import D4RL_SUITE9
from .archive import extract_mujoco
from .common import Runner, atomic_json, atomic_text, exclusive_lock, read_json, sha256
from .download import fetch_https
from .local import (LOCAL_CODE_NOTICE, contained_path, project_environment, prepare_local_directories,
                    clean_path, local_process)
from .uv_bootstrap import ensure_uv
from .native import check_native_prerequisites, ensure_native_prerequisites

PYTHON_VERSION = "3.10.19"
TORCH_VERSION = "2.5.1"
D4RL_IDS = tuple(f"{env}-{quality}-v2" for env in ("halfcheetah", "hopper", "walker2d")
                for quality in ("random", "medium", "medium-replay", "medium-expert"))
REPOS = {"d4rl": "https://github.com/Farama-Foundation/D4RL.git",
         "mjrl": "https://github.com/aravindr93/mjrl.git"}
MUJOCO_URL = "https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz"
MUJOCO_HOSTS = {"mujoco.org", "github.com", "objects.githubusercontent.com",
                "release-assets.githubusercontent.com"}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--profile", choices=("d4rl", "smoke"), default="d4rl")
    p.add_argument("--root", type=Path, help="Project-local directory; must be below the source tree; default .arlab/PROFILE")
    selection = p.add_mutually_exclusive_group()
    selection.add_argument("--env", choices=D4RL_IDS, default="hopper-medium-v2")
    selection.add_argument("--suite9", action="store_true", help="Full textbook study: 9 tasks, not 12")
    selection.add_argument("--suite12", action="store_true", help="Download all 12 locomotion-v2 datasets")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--yes", action="store_true", help="Approve displayed installation/download actions")
    p.add_argument("--dry-run", action="store_true", help="Print plan; no subprocess, writes, network or sudo")
    p.add_argument("--skip-system-deps", action="store_true", help=argparse.SUPPRESS)  # Legacy no-op; never installs system packages.
    p.add_argument("--native-deps", choices=("auto", "check"), default="auto",
                   help="auto: fetch signed distro debs and unpack into project if native probe fails; check: no downloads")
    p.add_argument("--native-offline", action="store_true", help="Reuse locked native debs only; no native network access")
    p.add_argument("--uv", type=Path, help="Optional existing uv 0.10.0; copied into this project, never modified")
    p.add_argument("--skip-smoke", action="store_true", help="Prepare only; deployment remains unverified")
    p.add_argument("--python", type=Path, help="Existing Python 3.10 to create the private runtime from")
    p.add_argument("--reuse-runtime", type=Path,
                   help="Smoke profile only: use an already installed runtime, without pip/install/network")
    p.add_argument("--raw-dir", type=Path, help="Existing ENV.raw.hdf5 or official-name HDF5 files")
    p.add_argument("--offline-data", action="store_true", help="Never download datasets; imports/local cache only")
    p.add_argument("--checksums", type=Path, help="Optional JSON mapping of env IDs to trusted raw SHA256")
    p.add_argument("--mujoco-sha256", default="", help="Optional independently verified binary archive hash")
    p.add_argument("--d4rl-ref", default="master", help="Prefer an audited full commit; resolved once and recorded")
    p.add_argument("--mjrl-ref", default="master", help="Resolved once; reuse never git-pulls it forward")
    p.add_argument("--code-mode", choices=("disabled", "trusted_local"), default="trusted_local")
    p.add_argument("--acknowledge-local-code", action="store_true")
    p.add_argument("--research-online", action="store_true", help="Enable bounded public research APIs in generated contract")
    p.add_argument("--research-host", action="append", default=[], help="Additional exact approved HTTPS host")
    p.add_argument("--model", default="REPLACE_WITH_YOUR_MODEL")
    p.add_argument("--provider", choices=("compatible", "responses", "anthropic"), default="compatible")
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--smoke-steps", type=int, default=40)
    return p


def validate_options(args: argparse.Namespace, source: Path) -> Path:
    root = (args.root or source / ".arlab" / args.profile).expanduser().absolute()
    if root.is_symlink() or root.resolve() == source.resolve() or source.resolve().is_relative_to(root.resolve()):
        raise ValueError("Deployment root cannot be the source tree, its ancestor or a symlink")
    if any(ch in str(root) for ch in ("\n", "\r", ":", ",", "\0")):
        raise ValueError("Deployment path contains unsupported control or path-list delimiters")
    if not 1 <= args.smoke_steps <= 5000:
        raise ValueError("Smoke updates must be between 1 and 5000")
    if args.reuse_runtime and args.profile != "smoke":
        raise ValueError("--reuse-runtime is only for explicitly non-D4RL smoke checks")
    if args.profile == "smoke" and (args.suite12 or args.suite9):
        raise ValueError("--suite9/--suite12 are D4RL selections, not synthetic data")
    if args.code_mode not in ("disabled", "trusted_local"):
        raise ValueError("This installation supports only local uv execution; Docker is disabled")
    for ref in (args.d4rl_ref, args.mjrl_ref):
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_./-]{0,199}", ref) or ".." in ref:
            raise ValueError("Invalid Git reference")
    from urllib.parse import urlsplit
    endpoint = urlsplit(args.base_url)
    if endpoint.scheme not in ("http", "https") or not endpoint.hostname or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment:
        raise ValueError("Model endpoint must be a plain http(s) URL without embedded secrets")
    if endpoint.scheme == "http" and endpoint.hostname not in ("localhost", "127.0.0.1", "::1"):
        raise ValueError("Remote model endpoints must use HTTPS")
    if args.mujoco_sha256 and not re.fullmatch(r"[a-fA-F0-9]{64}", args.mujoco_sha256):
        raise ValueError("Invalid MuJoCo SHA256")
    return contained_path(root, source)


def plan(args: argparse.Namespace, source: Path, root: Path) -> dict[str, Any]:
    return {"profile": args.profile, "root": str(root), "python": str(args.reuse_runtime or args.python or PYTHON_VERSION),
            "datasets": list(D4RL_SUITE9 if args.suite9 else D4RL_IDS if args.suite12 else (args.env,)) if args.profile == "d4rl" else [],
            "device_request": args.device, "code_mode": args.code_mode,
            "system_packages": [], "uses_sudo": False, "uses_docker": False,
            "project_boundary": str(source.resolve()),
            "managed_paths": {k:v for k,v in project_environment(root).items() if str(v).startswith(str(root))},
            "native_prerequisites": {"mode": args.native_deps,
                "missing_libraries": "download signed matching-distribution debs and extract data under native/auto",
                "system_install": False, "runs_maintainer_scripts": False, "replaces_host_libc": False,
                "offline": args.native_offline},
            "execution_notice": LOCAL_CODE_NOTICE,
            "installation": "reuse_existing_runtime" if args.reuse_runtime else "uv_project_virtualenv",
            "actions": ["preflight", "runtime", "legacy_environment" if args.profile == "d4rl" else "synthetic_fixture",
                        "data", "configuration", "checks", "smoke" if not args.skip_smoke else "smoke_SKIPPED"],
            "starts_research": False, "calls_llm": False, "research_online": args.research_online, "modifies_gpu_drivers": False,
            "modifies_shell_profile": False,
            "note": "Git branch refs resolve once per deployment, not globally pinned releases; record/reuse sources.lock.json"}


def direct_pins(source: Path, profile: str) -> dict[str, str]:
    """Read our deliberately small direct-pin set, not a transitive lockfile."""
    names = ["runtime-compat.txt"] + (["d4rl-compat.txt"] if profile == "d4rl" else [])
    expected = {"torch": TORCH_VERSION}
    for name in names:
        for raw in (source / "requirements" / name).read_text().splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line or line.startswith("-r "):
                continue
            match = re.fullmatch(r"([a-zA-Z0-9_.-]+)==([a-zA-Z0-9_.-]+)", line)
            if not match:
                raise ValueError(f"Unsupported direct dependency pin: {line}")
            expected[match[1]] = match[2]
    return expected


def _source_identity(source: Path) -> str:
    import hashlib
    paths = list((source / "src").rglob("*.py")) + list((source / "requirements").glob("*.txt"))
    paths += [source / "pyproject.toml"]
    return hashlib.sha256(json.dumps({str(p.relative_to(source)): sha256(p) for p in sorted(paths)},
                                   sort_keys=True).encode()).hexdigest()


class Deployer:
    def __init__(self, source: Path, root: Path, args: argparse.Namespace):
        self.source, self.root, self.args = source, root, args
        self.ids = list(D4RL_SUITE9 if args.suite9 else D4RL_IDS if args.suite12 else (args.env,))
        self.run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        env = project_environment(root)
        env["ARLAB_PROJECT_ROOT"] = str(source.resolve())
        env["PATH"] = clean_path(root)
        self.runner = Runner(root / "attempts" / self.run_id, env)
        self.state: dict[str, Any] = {"schema": 1, "run_id": self.run_id, "status": "running",
                                     "profile": args.profile, "steps": [], "plan": plan(args, source, root),
                                     "source_identity": _source_identity(source)}
        self.python = str(root / "venv/bin/python")
        self.image = None  # Kept in settings schema; no image is built or used.
        self.uv = str(root / "tools/uv/bin/uv")
        self.device = "cpu"

    def save(self) -> None:
        atomic_json(self.root / "deployment.json", self.state)
        atomic_json(self.root / "attempts" / self.run_id / "deployment.json", self.state)

    def stage(self, name: str, fn) -> Any:
        record = {"name": name, "status": "running", "started": time.time()}
        self.state["steps"].append(record)
        self.save()
        print(f"\n== {name} ==", flush=True)
        try:
            result = fn()
            record.update(status="passed", finished=time.time())
            self.save()
            return result
        except BaseException as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}", finished=time.time())
            self.state["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            self.save()
            raise

    def preflight(self) -> None:
        prepare_local_directories(self.root)
        if self.args.profile == "d4rl":
            if platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64"):
                raise RuntimeError("Legacy D4RL local installation requires Linux x86_64 (including WSL2)")
        needed = 1 if self.args.reuse_runtime else (20 if self.args.suite12 or self.args.suite9 else 8)
        free = shutil.disk_usage(self.root).free
        self.state["disk_free_bytes"] = free
        if free < needed * 1024 ** 3:
            raise RuntimeError(f"Less than {needed} GiB free; move the entire project to a larger disk")
        if self.args.profile == "d4rl":
            # Check disk space before the optional native download/expansion.
            self.state["native_prerequisites"] = ensure_native_prerequisites(
                self.root, self.runner, mode=self.args.native_deps, offline=self.args.native_offline)
        self.device = self.args.device
        if self.device == "auto":
            visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            gpu = False
            if visible not in ("", "-1") and shutil.which("nvidia-smi"):
                check = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                       capture_output=True, text=True, timeout=20, check=False)
                gpu = check.returncode == 0 and bool(check.stdout.strip())
            self.device = "cuda" if gpu else "cpu"
        if self.args.reuse_runtime and self.args.device == "auto":
            self.device = "cpu"
        self.state["device"] = self.device
        self.runner.env["PATH"] = str(self.root / "venv/bin") + os.pathsep + self.runner.env.get("PATH", "")

    def runtime(self) -> None:
        if self.args.reuse_runtime:
            self.python = str(self.args.reuse_runtime.expanduser().absolute())
            if not Path(self.python).is_file():
                raise ValueError("Existing runtime Python not found")
            self.state["installation"] = "skipped_owner_supplied_runtime"
            # Needed only for this explicit source-tree, no-network self-check mode.
            self.runner.env["PYTHONPATH"] = str(self.source / "src")
            self.state["python"] = self.python
            return
        marker = self.root / "runtime-installed.json"
        signature = {"source_identity": self.state["source_identity"], "device": self.device,
                     "profile": self.args.profile, "d4rl_ref": self.args.d4rl_ref,
                     "mjrl_ref": self.args.mjrl_ref}
        if marker.exists():
            self.uv = ensure_uv(self.root, self.runner, existing=self.args.uv)
            if read_json(marker) != signature:
                raise ValueError("Existing runtime belongs to a different source/device/profile; choose a new --root")
            if not Path(self.python).exists():
                raise ValueError("Runtime marker exists but interpreter is missing; choose a new --root")
            self.state["installation"] = "reused_private_runtime"
            self.state["python"] = self.python
            return
        self.uv = ensure_uv(self.root, self.runner, existing=self.args.uv)
        interpreter = str(self.args.python.expanduser().absolute()) if self.args.python else None
        if not interpreter:
            self.runner.run([self.uv, "python", "install", "--no-bin", PYTHON_VERSION], "python-install", cwd=self.root)
            interpreter = self.runner.run([self.uv, "python", "find", "--managed-python", PYTHON_VERSION],
                                          "python-find", capture=True, cwd=self.root).splitlines()[-1]
            if not Path(interpreter).resolve().is_relative_to(self.root / "python"):
                raise ValueError("Managed Python unexpectedly resolved outside the project")
        self.runner.run([interpreter, "-c", "import sys; assert sys.version_info[:2] == (3,10), 'Use Python 3.10'"], "python-version")
        environment = self.root / "venv"
        contained_path(environment, self.root)
        # Interrupted installation may reuse only a venv with our matching receipt.
        environment_receipt = self.root / "venv-created.json"
        if environment.exists():
            if not environment_receipt.exists() or read_json(environment_receipt) != signature:
                raise ValueError("Refusing an unowned/incompatible venv; use a new project-local root")
        else:
            self.runner.run([self.uv, "venv", "--no-project", "--python", interpreter, str(environment)],
                            "runtime-uv-venv", cwd=self.root)
            atomic_json(environment_receipt, signature)
        self.runner.env["VIRTUAL_ENV"] = str(environment)
        self.uv_install("build-tools", "--index-url", "https://pypi.org/simple",
                        "pip==24.0", "setuptools==69.5.1", "wheel==0.43.0")
        channel = "cu121" if self.device == "cuda" else "cpu"
        self.uv_install("torch", f"torch=={TORCH_VERSION}",
                        "--index-url", f"https://download.pytorch.org/whl/{channel}")
        requirement = "d4rl-compat.txt" if self.args.profile == "d4rl" else "runtime-compat.txt"
        # Install build prerequisites before evaluating legacy source metadata.
        if self.args.profile == "d4rl":
            self.uv_install("legacy-build-prerequisites", "--index-url", "https://pypi.org/simple",
                            "numpy==1.26.4", "Cython==0.29.37", "patchelf==0.17.2.4")
        self.uv_install("runtime-dependencies", "--index-url", "https://pypi.org/simple",
                        "--no-build-isolation", "-r", str(self.source / "requirements" / requirement))
        self.uv_install("project-install", "--no-build-isolation", "--no-deps", "-e", str(self.source))
        # Do not mark complete here: legacy packages and checks have not succeeded yet.
        self.state["runtime_signature"] = signature
        self.state["python"] = self.python
        self.state["installation"] = "created_private_runtime"

    def uv_install(self, label: str, *arguments: str) -> None:
        """Always target the project venv explicitly; never --system or --user."""
        if Path(self.python).absolute() != (self.root / "venv/bin/python").absolute():
            raise ValueError("Installation target is not the project's virtual environment")
        self.runner.run([self.uv, "pip", "install", "--python", self.python, *arguments],
                        label, cwd=self.root)

    def legacy(self) -> None:
        archive = self.root / "downloads/mujoco210-linux-x86_64.tar.gz"
        archive_hash = fetch_https(MUJOCO_URL, archive, expected_sha256=self.args.mujoco_sha256,
                                   allowed_hosts=MUJOCO_HOSTS, max_bytes=250_000_000)
        target = self.root / "mujoco/mujoco210"
        marker = target / ".arlab-install.json"
        if target.exists():
            if not marker.exists() or read_json(marker).get("archive_sha256") != archive_hash:
                raise ValueError("Existing MuJoCo tree has no matching install receipt")
            recorded = read_json(marker)["files"]
            if {str(p.relative_to(target)): sha256(p) for p in sorted(target.rglob("*")) if p.is_file() and p != marker} != recorded:
                raise ValueError("MuJoCo files changed after installation")
        else:
            staging = self.root / "mujoco" / (".staging-" + uuid.uuid4().hex)
            try:
                extract_mujoco(archive, staging)
                target.parent.mkdir(parents=True, exist_ok=True)
                staged_tree = staging / "mujoco210"
                atomic_json(staged_tree / ".arlab-install.json", {"archive_sha256": archive_hash,
                    "files": {str(p.relative_to(staged_tree)): sha256(p)
                              for p in sorted(staged_tree.rglob("*")) if p.is_file()}})
                # The receipt and all files become visible in one rename.
                os.replace(staged_tree, target)
            finally:
                shutil.rmtree(staging, ignore_errors=True)
        self.runner.env.update(MUJOCO_PY_MUJOCO_PATH=str(target), MUJOCO_PY_FORCE_CPU="1",
                               MUJOCO_GL="osmesa", LD_LIBRARY_PATH=os.pathsep.join(
                                   [str(target / "bin"), project_environment(self.root)["LIBRARY_PATH"]]),
                               D4RL_SUPPRESS_IMPORT_ERROR="1", D4RL_DATASET_DIR=str(self.root / "data/raw"))
        # Force CPU is for headless MuJoCo rendering, NOT for PyTorch training.
        lock_path = self.root / "sources.lock.json"
        locks = read_json(lock_path) if lock_path.exists() else {}
        for name in ("mjrl", "d4rl"):
            ref = getattr(self.args, name + "_ref")
            repo = self.root / "sources" / name
            if name in locks and locks[name]["requested_ref"] != ref:
                raise ValueError(f"{name} reference conflicts with the persisted source lock; use a new root")
            if repo.exists():
                if name not in locks or locks[name]["requested_ref"] != ref:
                    raise ValueError(f"Unowned or changed {name} checkout; use a new deployment root")
                actual = self.runner.run(["git", "-C", str(repo), "rev-parse", "HEAD"], name + "-head", capture=True)
                dirty = self.runner.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], name + "-clean", capture=True)
                untracked_code = self.runner.run(["git", "-C", str(repo), "ls-files", "--others", "--exclude-standard",
                    "--", "*.py", "*.so", "*.pth"], name + "-untracked-code", capture=True)
                if actual != locks[name]["commit"] or dirty or untracked_code:
                    raise ValueError(f"{name} checkout differs from its recorded commit or has untracked executable code")
            else:
                repo.parent.mkdir(parents=True, exist_ok=True)
                # A staging checkout prevents interrupted git fetch from looking installed.
                staging = repo.with_name(repo.name + ".pending")
                if staging.exists():
                    shutil.rmtree(staging)
                self.runner.run(["git", "init", str(staging)], name + "-init")
                self.runner.run(["git", "-C", str(staging), "remote", "add", "origin", REPOS[name]], name + "-remote")
                pinned = locks.get(name, {}).get("commit", ref)
                self.runner.run(["git", "-C", str(staging), "fetch", "--depth=1", "origin", pinned], name + "-fetch")
                self.runner.run(["git", "-C", str(staging), "checkout", "--detach", "FETCH_HEAD"], name + "-checkout")
                actual = self.runner.run(["git", "-C", str(staging), "rev-parse", "HEAD"], name + "-commit", capture=True)
                if not re.fullmatch(r"[0-9a-f]{40}", actual):
                    raise ValueError("Git did not return an exact commit")
                locks[name] = {"url": REPOS[name], "requested_ref": ref, "commit": actual}
                atomic_json(lock_path, locks)
                os.replace(staging, repo)
            # D4RL's metadata references mjrl@master. Install our exact checkout first,
            # then --no-deps, so pip cannot quietly move it or replace legacy versions.
            self.uv_install(name + "-install", "--no-deps", "--no-build-isolation", "-e", str(repo))
        self.state["sources"] = locks
        self.state["mujoco_archive_sha256"] = archive_hash

    def python_task(self, task: str, *args: str) -> None:
        self.runner.run([self.python, "-m", "arlab.deployment.tasks", task, *args], task, timeout=7200)

    def data(self) -> None:
        args = ["--data-root", str(self.root / "data"), "--envs", *self.ids]
        if self.args.raw_dir:
            args += ["--raw-dir", str(self.args.raw_dir.expanduser().resolve())]
        if self.args.offline_data:
            args += ["--offline"]
        if self.args.checksums:
            args += ["--checksums", str(self.args.checksums.resolve())]
        self.python_task("datasets", *args)

    def configure(self) -> None:
        settings = {"root": str(self.root), "profile": self.args.profile, "envs": self.ids,
                    "suite12": self.args.suite12, "suite9": self.args.suite9, "device": self.device, "code_mode": self.args.code_mode,
                    "image": self.image, "model": self.args.model, "provider": self.args.provider,
                    "base_url": self.args.base_url, "smoke_steps": self.args.smoke_steps,
                    "research_online": self.args.research_online, "research_hosts": self.args.research_host,
                    "python": self.python, "source": str(self.source),
                    "reuse_runtime": bool(self.args.reuse_runtime), "installation_scope": "project_local_uv"}
        atomic_json(self.root / "settings.json", settings)
        self.python_task("configure", "--root", str(self.root))
        # Quote paths; no credentials are written, and .bashrc is not modified.
        exports = project_environment(self.root)
        exports["ARLAB_PROJECT_ROOT"] = str(self.source.resolve())
        exports.update({k:self.runner.env[k] for k in ("MUJOCO_PY_MUJOCO_PATH", "MUJOCO_PY_FORCE_CPU", "MUJOCO_GL",
                        "LD_LIBRARY_PATH", "CPATH", "LIBRARY_PATH", "D4RL_SUPPRESS_IMPORT_ERROR") if k in self.runner.env})
        if self.args.reuse_runtime:
            exports["PYTHONPATH"] = str(self.source / "src")
        env_script = ("#!/usr/bin/env bash\n# Private launcher environment; do not source into a daily shell.\n"
                      "unset PYTHONHOME PYTHONPATH PYTHONPYCACHEPREFIX VIRTUAL_ENV CONDA_PREFIX CONDA_DEFAULT_ENV BASH_ENV ENV LD_PRELOAD\n"
                      "for name in ${!UV_@} ${!PIP_@} ${!DOCKER_@} ${!GIT_CONFIG_@}; do unset \"$name\"; done\n")
        for key, value in exports.items():
            env_script += f"export {key}={shlex.quote(value)}\n"
        env_script += f"export PATH={shlex.quote(str(self.root / 'bin') + os.pathsep + self.runner.env['PATH'])}\n"
        if not self.args.reuse_runtime:
            env_script += f"export VIRTUAL_ENV={shlex.quote(str(self.root / 'venv'))}\n"
        # Limits damage from accidental bytecode/coredump/cache output; not a sandbox.
        env_script += "ulimit -c 0 || true\n"
        atomic_text(self.root / "env.sh", env_script)
        wrapper = "#!/usr/bin/env bash\nset -euo pipefail\n" + f"source {shlex.quote(str(self.root / 'env.sh'))}\n"
        wrapper += f"cd {shlex.quote(str(self.source))}\n"
        wrapper += f"exec {shlex.quote(self.python)} -m arlab \"$@\"\n"
        atomic_text(self.root / "bin/arlab", wrapper, 0o700)
        launcher = "#!/usr/bin/env bash\nset -euo pipefail\n" + f"source {shlex.quote(str(self.root / 'env.sh'))}\n"
        launcher += f"cd {shlex.quote(str(self.source))}\n"
        launcher += f"exec {shlex.quote(self.python)} -m arlab.deployment.launch --root {shlex.quote(str(self.root))} \"$@\"\n"
        atomic_text(self.root / "start-research.sh", launcher, 0o700)

    def checks(self) -> None:
        self.python_task("check", "--root", str(self.root))
        if not self.args.reuse_runtime:
            # Reuse must not silently bless a manually upgraded direct dependency.
            expected = direct_pins(self.source, self.args.profile)
            verify = ("import importlib.metadata as m,json; "
                      "expected=json.loads(" + repr(json.dumps(expected)) + "); "
                      "actual={k:m.version(k).split('+')[0] for k in expected}; "
                      "assert actual == expected, f'Direct dependency drift: {actual!r} != {expected!r}'; print(actual)")
            self.runner.run([self.python, "-c", verify], "dependency-pins")
            self.runner.run([self.uv, "pip", "check", "--python", self.python], "uv-pip-check", cwd=self.root)
            freeze = self.runner.run([self.uv, "pip", "freeze", "--python", self.python], "runtime-freeze", capture=True, cwd=self.root)
            atomic_text(self.root / "runtime-freeze.txt", freeze + "\n")
            inspect = self.runner.run([self.python, "-m", "pip", "inspect"], "pip-inspect", capture=True)
            atomic_text(self.root / "pip-inspect.json", inspect + "\n")
            if "runtime_signature" in self.state:
                atomic_json(self.root / "runtime-installed.json", self.state["runtime_signature"])

    def smoke(self) -> None:
        self.python_task("smoke", "--root", str(self.root), "--run-id", self.run_id)

    def run(self) -> None:
        self.stage("preflight", self.preflight)
        self.stage("runtime", self.runtime)
        if self.args.profile == "d4rl":
            self.stage("legacy-environment", self.legacy)
            self.stage("datasets", self.data)
        self.stage("configuration", self.configure)
        self.stage("environment-checks", self.checks)
        if not self.args.skip_smoke:
            self.stage("training-and-evaluation-smoke", self.smoke)
        self.state["status"] = "prepared_unverified" if self.args.skip_smoke else "ready"
        self.state["research_started"] = False
        self.state["llm_calls"] = 0
        self.save()
        print(f"\nDeployment: {self.state['status']}. Report: {self.root / 'deployment.json'}")
        print(f"CLI: {shlex.quote(str(self.root / 'bin/arlab'))}")
        print(f"Research requires model configuration and explicit budget approval: {self.root / 'start-research.sh'}")
        print(f"Source-code execution mode: {self.args.code_mode}; never silently upgraded.")


def main(argv: list[str] | None = None, *, source: Path | None = None) -> int:
    args = parser().parse_args(argv)
    source = (source or Path(__file__).resolve().parents[3]).resolve()
    try:
        root = validate_options(args, source)
        print(json.dumps(plan(args, source, root), ensure_ascii=False, indent=2))
        if args.dry_run:
            return 0
        if not args.yes:
            if not sys.stdin.isatty():
                raise ValueError("Non-interactive deployment requires --yes; use --dry-run to inspect first")
            print(LOCAL_CODE_NOTICE, flush=True)
            if input("Install into this project and run the small local self-check? [y/N] ").lower() != "y":
                return 2
        os.umask(0o077)
        if root.exists() and any(root.iterdir()) and not (root / "deployment.json").exists():
            raise ValueError("Refusing an existing non-empty unowned deployment directory")
        root.mkdir(parents=True, exist_ok=True)
        with exclusive_lock(root / ".deploy.lock"), local_process(root, source):
            deployer = Deployer(source, root, args)
            deployer.run()
        return 0
    except KeyboardInterrupt:
        print("Interrupted; completed downloads are retained. Repeat the same command to retry.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"DEPLOY ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
