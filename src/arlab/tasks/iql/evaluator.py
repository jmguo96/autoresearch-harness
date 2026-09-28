"""Independent scoring. Open policies run across an observation/action-only process boundary."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from pydantic import Field, model_validator
from safetensors.torch import load_file

from arlab.core.io import digest, directory_digest, file_digest, read_json, runtime_info, write_json
from arlab.core.schema import CodeExecutionPolicy, StrictModel
from arlab.tasks.iql.environments import make_environment, normalized_score
from arlab.tasks.iql.networks import GaussianActor


class PolicyManifest(StrictModel):
    format_version: Literal[1] = Field(alias="schema")
    architecture: Literal["iql-gaussian-mean-v1"]
    observation_dim: int = Field(gt=0, le=2048)
    action_dim: int = Field(gt=0, le=256)
    hidden_dims: list[int] = Field(min_length=1, max_length=4)
    weights_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_binding: str = Field(pattern=r"^[0-9a-f]{64}$")
    updates: int = Field(gt=0)

    @model_validator(mode="after")
    def bounded_widths(self) -> PolicyManifest:
        if any(n < 8 or n > 1024 for n in self.hidden_dims):
            raise ValueError("Network exceeds evaluator architecture bounds")
        return self


class ExportedPolicy:
    def __init__(self, directory: Path, expected_hash: str, binding: str):
        manifest = PolicyManifest.model_validate(read_json(directory / "policy.json"))
        weights = directory / "weights.safetensors"
        if weights.stat().st_size > 128_000_000:
            raise ValueError("Policy exceeds artifact size limit")
        if directory_digest(directory) != expected_hash or file_digest(weights) != manifest.weights_sha256:
            raise ValueError("Policy content hash mismatch")
        if manifest.training_binding != binding:
            raise ValueError("Policy training binding mismatch")
        tensors = load_file(str(weights), device="cpu")
        if any(not torch.isfinite(t).all() for t in tensors.values()):
            raise ValueError("Policy contains non-finite values")
        self.actor = GaussianActor(manifest.observation_dim, manifest.action_dim, manifest.hidden_dims)
        expected_keys = {"actor." + k for k in self.actor.state_dict()} | {"observation_mean", "observation_std", "action_low", "action_high"}
        if tensors.keys() != expected_keys:
            raise ValueError("Unexpected policy tensors")
        self.actor.load_state_dict({k[6:]: t for k, t in tensors.items() if k.startswith("actor.")}, strict=True)
        self.actor.eval()
        self.mean, self.std = tensors["observation_mean"], tensors["observation_std"]
        self.low, self.high = tensors["action_low"], tensors["action_high"]
        if self.mean.shape != (manifest.observation_dim,) or self.std.shape != self.mean.shape:
            raise ValueError("Invalid observation normalization shapes")
        if self.low.shape != (manifest.action_dim,) or self.high.shape != self.low.shape:
            raise ValueError("Invalid action normalization shapes")
        if not (self.std > 0).all() or not (self.high > self.low).all():
            raise ValueError("Invalid normalization scales")
        self.manifest = manifest

    @torch.no_grad()
    def action(self, observation: np.ndarray) -> np.ndarray:
        obs = torch.as_tensor(np.asarray(observation, dtype=np.float32))
        if obs.shape != self.mean.shape or not torch.isfinite(obs).all():
            raise ValueError("Invalid evaluation observation")
        mean_action = self.actor(((obs - self.mean) / self.std).unsqueeze(0)).mean.squeeze(0)
        action = self.low + (mean_action.clamp(-1, 1) + 1) * 0.5 * (self.high - self.low)
        return action.numpy()


    def reset(self, seed: int) -> None:
        pass

    def close(self) -> None:
        pass


def evaluate_policy(request: dict[str, Any], output: Path) -> dict[str, Any]:
    started = time.monotonic()
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=True)
    binding = digest(request)
    cached = output / "evaluation.json"
    if cached.exists():
        result = read_json(cached)
        if result["evaluation_request_sha256"] != binding:
            raise ValueError("Evaluation result belongs to another request")
        return result
    seeds = request["eval_seeds"]
    if not isinstance(seeds, list) or not 1 <= len(seeds) <= 1000:
        raise ValueError("Invalid episode budget")
    artifact = Path(request["artifact"])
    manifest = read_json(artifact / "policy.json")
    if manifest.get("architecture") == "python-algorithm-v1":
        from arlab.algorithms.policy import PythonPolicy
        execution = CodeExecutionPolicy.model_validate(request.get("code_execution", {}))
        policy = PythonPolicy(artifact, request["artifact_sha256"], request["training_binding"], execution)
    else:
        policy = ExportedPolicy(artifact, request["artifact_sha256"], request["training_binding"])
    if policy.manifest.updates != request["steps"]:
        policy.close()
        raise ValueError("Policy update count differs from the requested horizon")
    env = None
    inference_calls, inference_seconds = 0, 0.0
    returns: list[float] = []
    episode_lengths: list[int] = []
    try:
        env = make_environment(request["environment"])
        for seed in seeds:
            policy.reset((int(seed) + 104729) % (2**31 - 1))
            try:
                reset_result = env.reset(seed=int(seed))
            except TypeError:
                env.seed(int(seed))
                reset_result = env.reset()
            obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            total = 0.0
            for step in range(10000):
                action_started = time.monotonic()
                action = policy.action(obs)
                inference_seconds += time.monotonic() - action_started
                inference_calls += 1
                result = env.step(action)
                if len(result) == 5:
                    obs, reward, terminated, truncated, _ = result
                    done = terminated or truncated
                else:
                    obs, reward, done, _ = result
                if not np.isfinite(reward):
                    raise ValueError("Non-finite environment reward")
                total += float(reward)
                if done:
                    break
            else:
                raise RuntimeError("Environment exceeded the independent episode-length limit")
            returns.append(total)
            episode_lengths.append(step + 1)
    finally:
        if env is not None:
            env.close()
        policy.close()
    kind = request["environment"]["kind"]
    raw_mean = float(np.mean(returns))
    score = normalized_score(request["environment"]["id"], raw_mean) if kind == "d4rl" else raw_mean
    result = {"returns": returns, "episode_lengths": episode_lengths, "score": score,
              "metric": "d4rl_normalized_return" if kind == "d4rl" else "synthetic_raw_return",
              "artifact_sha256": request["artifact_sha256"], "training_binding": request["training_binding"],
              "evaluation_request_sha256": binding, "evaluation_kind": kind,
              "evaluate_seconds": time.monotonic() - started, "runtime": runtime_info(),
              "inference": policy.statistics() if hasattr(policy, "statistics") else {
                  "execution": "trusted_fixed_iql", "action_calls": inference_calls,
                  "action_seconds": inference_seconds, "scoring_in_candidate_process": False}}
    write_json(cached, result)
    return result
