"""Bounded process execution. Local mode is NOT an OS security sandbox.

Code recipes require separate owner permission. Local mode executes reviewed
Python with host-user privileges; a model still has no unrestricted shell tool. Container mode adds a network/filesystem boundary when available.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class ProcessResult:
    exit_code: int
    seconds: float
    timed_out: bool
    log_path: str


def terminate_group(process: subprocess.Popen, grace: float = 3.0) -> None:
    """Give cooperative workers a chance to close their own policy children."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=grace)
    except ProcessLookupError:
        pass
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


class LocalBackend:
    def run(self, command: Sequence[str], cwd: Path, log: Path, timeout: float) -> ProcessResult:
        allowed = ("PATH", "PYTHONPATH", "LD_LIBRARY_PATH", "MUJOCO_PY_MUJOCO_PATH", "MUJOCO_PY_MJKEY_PATH",
                   "MUJOCO_GL", "MUJOCO_PY_FORCE_CPU", "D4RL_SUPPRESS_IMPORT_ERROR", "D4RL_DATASET_DIR",
                   "CUDA_VISIBLE_DEVICES", "SYSTEMROOT", "HOME", "DOCKER_HOST", "DOCKER_CONTEXT",
                   "DOCKER_CONFIG", "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH", "XDG_RUNTIME_DIR")
        env = {k: os.environ[k] for k in allowed if k in os.environ}
        from arlab.deployment.local import inherit_local_environment
        inherit_local_environment(env)
        # Changing cwd must not break a source-tree launch such as PYTHONPATH=src.
        import_paths = [str(Path(__file__).resolve().parents[2])]
        import_paths += [str(Path(p or ".").resolve()) for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
        env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(import_paths))
        env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONHASHSEED="0",
                   CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONUNBUFFERED="1")
        cwd.mkdir(parents=True, exist_ok=True)
        log.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        timed_out = False
        with log.open("wb") as stream:
            process = subprocess.Popen(list(command), cwd=cwd, stdout=stream, stderr=subprocess.STDOUT,
                                       env=env, start_new_session=True)
            try:
                while process.poll() is None:
                    if time.monotonic() - started >= timeout or log.stat().st_size > 20_000_000:
                        timed_out = True
                        terminate_group(process)
                        break
                    time.sleep(0.05)
                code = process.wait()
            except BaseException:
                if process.poll() is None:
                    terminate_group(process)
                raise
        return ProcessResult(code, time.monotonic() - started, timed_out, str(log))


class DockerBackend(LocalBackend):
    """Build docker/worker.Dockerfile first. The image must include this package.

    Files are mounted at the SAME absolute paths as the host. Only the designated
    job output is writable; data/request/policy mounts are read-only. This backend
    is optional and is not claimed tested where no Docker daemon exists.
    """
    def __init__(self, image: str, read_only_paths: list[Path], writable: Path, *, code_policy=None):
        if os.environ.get("ARLAB_LOCAL_ROOT"):
            raise PermissionError("Docker is disabled in project-local uv installations")
        self.image = image
        self.code_policy = code_policy
        self.read_only_paths = read_only_paths
        self.writable = writable.resolve()

    def run(self, command: Sequence[str], cwd: Path, log: Path, timeout: float) -> ProcessResult:
        import uuid
        name = "arlab-" + uuid.uuid4().hex[:12]
        docker = ["docker", "run", "--rm", "--name", name, "--network=none", "--read-only",
                  "--cap-drop=ALL", "--security-opt=no-new-privileges", "--pids-limit=256",
                  "--memory=" + (self.code_policy.container_memory if self.code_policy else "8g"),
                  "--cpus=" + str(self.code_policy.container_cpus if self.code_policy else 2), "--user", f"{os.getuid()}:{os.getgid()}",
                  "--tmpfs", "/tmp:rw,noexec,nosuid,size=1g", "-e", "HOME=/tmp",
                  "-e", "OMP_NUM_THREADS=1", "-e", "MKL_NUM_THREADS=1", "--workdir", str(cwd)]
        if self.code_policy:
            docker += ["-e", "ARLAB_CODE_CONTAINER=1", "-e", "PYTHONDONTWRITEBYTECODE=1"]
            if self.code_policy.container_gpu:
                docker += ["--gpus", "all", "-e", "CUBLAS_WORKSPACE_CONFIG=:4096:8"]
        for path in self.read_only_paths:
            resolved = path.resolve()
            docker += ["--mount", f"type=bind,source={resolved},target={resolved},readonly"]
        docker += ["--mount", f"type=bind,source={self.writable},target={self.writable}", self.image]
        inner = list(command)
        if inner[0] == sys.executable:
            inner[0] = "python"
        try:
            return super().run(docker + inner, cwd, log, timeout)
        finally:
            # Killing the Docker client alone can orphan the actual container.
            subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=20, check=False)
