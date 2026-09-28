"""Parent-side policy transport. The evaluator never imports candidate Python.

Local mode gives process separation, NOT filesystem isolation. Docker mode mounts
only the artifact into the policy container: no dataset, environment, campaign,
scoring implementation output directory, credentials, or Docker socket.
"""
from __future__ import annotations

import json
import os
import selectors
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from pydantic import Field
from safetensors.torch import load_file

from arlab.algorithms.spec import CodeRecipe
from arlab.core.io import canonical, digest, directory_digest, file_digest, read_json, source_digest
from arlab.core.schema import CodeExecutionPolicy, StrictModel


class CodePolicyManifest(StrictModel):
    format_version: Literal[2] = Field(alias="schema")
    architecture: Literal["python-algorithm-v1"]
    observation_dim: int = Field(gt=0, le=2048)
    action_dim: int = Field(gt=0, le=256)
    weights_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    recipe_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_binding: str = Field(pattern=r"^[0-9a-f]{64}$")
    updates: int = Field(gt=0)
    total_ticks: int = Field(gt=0)
    extra_state: Any
    value_learning_audit: dict[str, Any]


def validate_artifact(directory: Path, expected: str, binding: str,
                      policy: CodeExecutionPolicy) -> tuple[CodePolicyManifest, CodeRecipe, dict[str, torch.Tensor]]:
    if directory_digest(directory) != expected:
        raise ValueError("Policy artifact content hash mismatch")
    if sum(p.stat().st_size for p in directory.rglob("*") if p.is_file()) > policy.max_artifact_bytes:
        raise ValueError("Policy artifact exceeds owner byte budget")
    manifest = CodePolicyManifest.model_validate(read_json(directory / "policy.json"))
    recipe = CodeRecipe.model_validate(read_json(directory / "algorithm.json"))
    if manifest.training_binding != binding or manifest.recipe_sha256 != digest(recipe):
        raise ValueError("Policy training/candidate binding mismatch")
    if manifest.source_sha256 != recipe.source_sha256:
        raise ValueError("Policy source binding mismatch")
    for name, text in recipe.files.items():
        if (directory / "source" / name).read_text(encoding="utf-8") != text:
            raise ValueError("Policy source differs from registered bundle")
    path = directory / "weights.safetensors"
    if file_digest(path) != manifest.weights_sha256:
        raise ValueError("Policy weight hash mismatch")
    tensors = load_file(str(path), device="cpu")
    if any(not torch.isfinite(t).all() for t in tensors.values()):
        raise ValueError("Non-finite policy tensors")
    return manifest, recipe, tensors


def policy_process_environment(temporary: str, mode: str) -> dict[str, str]:
    """Docker client configuration belongs to the owner, not the container env."""
    if mode == "docker" and os.environ.get("ARLAB_LOCAL_ROOT"):
        raise PermissionError("Docker is disabled in project-local uv installations")
    env = {k: os.environ[k] for k in ("PATH", "LD_LIBRARY_PATH", "SYSTEMROOT") if k in os.environ}
    from arlab.deployment.local import inherit_local_environment
    inherit_local_environment(env)
    env.update(HOME=temporary, PYTHONPATH=str(Path(__file__).resolve().parents[2]),
               PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               CUDA_VISIBLE_DEVICES="", PYTHONHASHSEED="0")
    if mode == "docker":
        for key in ("DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH", "XDG_RUNTIME_DIR"):
            if key in os.environ:
                env[key] = os.environ[key]
        env["DOCKER_CONFIG"] = os.environ.get("DOCKER_CONFIG", str(Path.home() / ".docker"))
    return env


class PythonPolicy:
    """A generic act(observation) client with independent normalization and bounds."""
    def __init__(self, directory: Path, expected_hash: str, binding: str, policy: CodeExecutionPolicy):
        if policy.mode == "disabled":
            raise PermissionError("Owner must authorize Python policy execution")
        self.manifest, _recipe, tensors = validate_artifact(directory, expected_hash, binding, policy)
        self.mean, self.std = tensors["observation_mean"].numpy(), tensors["observation_std"].numpy()
        self.low, self.high = tensors["action_low"].numpy(), tensors["action_high"].numpy()
        if (self.mean.shape != (self.manifest.observation_dim,) or self.std.shape != self.mean.shape
                or self.low.shape != (self.manifest.action_dim,) or self.high.shape != self.low.shape
                or not (self.std > 0).all() or not (self.high > self.low).all()):
            raise ValueError("Invalid policy normalization")
        self.policy, self.buffer = policy, b""
        self.calls, self.action_seconds, self.max_action_seconds = 0, 0.0, 0.0
        self.child_audit: dict[str, Any] = {}
        self.process: subprocess.Popen | None = None
        self.container_name: str | None = None
        tmp_base = (Path(os.environ["ARLAB_LOCAL_ROOT"]) / "tmp" if os.environ.get("ARLAB_LOCAL_ROOT") else None)
        self.temporary = tempfile.TemporaryDirectory(prefix="arlab-policy-", dir=tmp_base)
        command = [sys.executable, "-B", "-m", "arlab.algorithms.policy_worker",
                   "--artifact", str(directory.resolve()), "--expected-hash", expected_hash,
                   "--binding", binding, "--max-parameters", str(policy.max_parameters),
                   "--max-artifact-bytes", str(policy.max_artifact_bytes),
                   "--implementation-sha256", source_digest()]
        if policy.mode == "docker":
            self.container_name = "arlab-policy-" + uuid.uuid4().hex[:12]
            # The scoring process stays on the owner side of this boundary.
            command = ["docker", "run", "--rm", "-i", "--name", self.container_name,
                       "--network=none", "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges",
                       "--pids-limit=128", "--memory=" + policy.container_memory,
                       "--cpus=" + str(policy.container_cpus), "--user", f"{os.getuid()}:{os.getgid()}",
                       "--tmpfs", "/tmp:rw,noexec,nosuid,size=256m", "--workdir", "/tmp",
                       "--mount", f"type=bind,source={directory.resolve()},target=/policy,readonly",
                       "-e", "HOME=/tmp", "-e", "PYTHONDONTWRITEBYTECODE=1", policy.docker_image,
                       "python", "-B", "-m", "arlab.algorithms.policy_worker", "--artifact", "/policy",
                       "--expected-hash", expected_hash, "--binding", binding,
                       "--max-parameters", str(policy.max_parameters),
                       "--max-artifact-bytes", str(policy.max_artifact_bytes),
                       "--implementation-sha256", source_digest()]
        environment = policy_process_environment(self.temporary.name, policy.mode)
        started = time.monotonic()
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, cwd=self.temporary.name, env=environment,
                start_new_session=True, bufsize=0)
            os.set_blocking(self.process.stdout.fileno(), False)
            os.set_blocking(self.process.stdin.fileno(), False)
            ready = self._read(time.monotonic() + policy.startup_timeout_seconds)
            if ready.get("status") != "ready":
                raise ValueError(f"Policy startup failed: {ready.get('error', ready)}")
            self.startup_seconds = time.monotonic() - started
        except BaseException:
            self.close()
            raise

    def _wait(self, stream: Any, event: int, deadline: float) -> None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Policy worker deadline exceeded")
        with selectors.DefaultSelector() as selector:
            selector.register(stream, event)
            if not selector.select(remaining):
                raise TimeoutError("Policy worker deadline exceeded")

    def _read(self, deadline: float) -> dict[str, Any]:
        while b"\n" not in self.buffer:
            self._wait(self.process.stdout, selectors.EVENT_READ, deadline)
            chunk = os.read(self.process.stdout.fileno(), 65536)
            if not chunk:
                raise RuntimeError("Policy worker closed its protocol stream")
            self.buffer += chunk
            if len(self.buffer) > 1_000_000:
                raise ValueError("Policy protocol output exceeds byte budget")
        line, self.buffer = self.buffer.split(b"\n", 1)
        response = json.loads(line)
        if not isinstance(response, dict):
            raise ValueError("Invalid policy protocol response")
        return response

    def _exchange(self, message: dict[str, Any]) -> dict[str, Any]:
        deadline = time.monotonic() + self.policy.inference_timeout_seconds
        payload = (canonical(message) + "\n").encode()
        sent = 0
        try:
            while sent < len(payload):
                self._wait(self.process.stdin, selectors.EVENT_WRITE, deadline)
                sent += os.write(self.process.stdin.fileno(), payload[sent:])
            response = self._read(deadline)
            if "error" in response:
                raise ValueError("Policy worker: " + str(response["error"])[:4000])
            return response
        except BaseException:
            self.close()
            raise

    def reset(self, seed: int) -> None:
        self._exchange({"kind": "reset", "seed": int(seed)})

    def action(self, observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32)
        if obs.shape != self.mean.shape or not np.isfinite(obs).all():
            raise ValueError("Invalid policy observation")
        started = time.monotonic()
        response = self._exchange({"kind": "act", "observation": ((obs - self.mean) / self.std).tolist()})
        elapsed = time.monotonic() - started
        self.calls += 1
        self.action_seconds += elapsed
        self.max_action_seconds = max(self.max_action_seconds, elapsed)
        action = np.asarray(response.get("action"), dtype=np.float32)
        if action.shape != self.low.shape or not np.isfinite(action).all():
            raise ValueError("Policy returned an invalid action; no score is accepted")
        self.child_audit = response.get("audit", {})
        return self.low + (np.clip(action, -1, 1) + 1) * 0.5 * (self.high - self.low)

    def statistics(self) -> dict[str, Any]:
        return {"execution": self.policy.mode, "startup_seconds": self.startup_seconds,
                "action_calls": self.calls, "action_seconds": self.action_seconds,
                "max_action_seconds": self.max_action_seconds, "candidate_instrumentation": self.child_audit,
                "scoring_in_candidate_process": False}

    def close(self) -> None:
        if self.process is not None:
            if self.process.poll() is None:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            self.process.wait(timeout=10)
            for stream in (self.process.stdin, self.process.stdout):
                if stream is not None:
                    stream.close()
            self.process = None
        if self.container_name:
            subprocess.run(["docker", "rm", "-f", self.container_name], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=20, check=False)
            self.container_name = None
        self.temporary.cleanup()
