"""Trusted evaluators' environments and an explicitly non-D4RL teaching fixture."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from arlab.tasks.iql.data import save_dataset

# Frozen numeric constants from D4RL infos.py; see docs/references.md.
D4RL_REFERENCE = {
    "halfcheetah": (-280.178953, 12135.0, 17, 6),
    "hopper": (-20.272305, 3234.3, 11, 3),
    "walker2d": (1.629008, 4592.3, 17, 6),
}
QUALITIES = ("random", "medium", "medium-replay", "medium-expert")
D4RL_IDS = tuple(f"{env}-{quality}-v2" for env in D4RL_REFERENCE for quality in QUALITIES)


def reference(env_id: str) -> tuple[float, float, int, int]:
    if env_id not in D4RL_IDS:
        raise ValueError("This frozen example supports only the 12 locomotion D4RL-v2 tasks")
    return D4RL_REFERENCE[env_id.split("-")[0]]


def normalized_score(env_id: str, raw_return: float) -> float:
    lower, upper, _, _ = reference(env_id)
    return 100.0 * (float(raw_return) - lower) / (upper - lower)


class LinearQuadraticEnv:
    """Tiny continuous-control fixture. It is NOT D4RL, LQR optimal control, or a benchmark score."""
    observation_dim = 3
    action_dim = 2

    def __init__(self, dynamics_seed: int = 0, horizon: int = 32):
        if horizon < 1 or horizon > 10000:
            raise ValueError("Invalid horizon")
        self.horizon = horizon
        rng = np.random.default_rng(dynamics_seed)
        self.A = 0.85 * np.eye(3) + rng.normal(0, 0.025, (3, 3))
        self.B = rng.normal(0, 0.25, (3, 2))
        self.rng = rng
        self.state = np.zeros(3)
        self.steps = 0

    def reset(self, *, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
        self.rng = np.random.default_rng(seed)
        self.state = self.rng.normal(0, 0.7, 3)
        self.steps = 0
        return self.state.astype(np.float32), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.clip(np.asarray(action, dtype=float), -1, 1)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError("Invalid action")
        reward = -float(np.square(self.state).sum() + 0.05 * np.square(action).sum())
        self.state = self.A @ self.state + self.B @ action + self.rng.normal(0, 0.025, 3)
        self.steps += 1
        return self.state.astype(np.float32), reward, False, self.steps >= self.horizon, {}

    def close(self) -> None:
        return None


def make_environment(spec: dict[str, Any]) -> Any:
    kind = spec.get("kind")
    if kind == "synthetic_lq":
        allowed = {"kind", "dynamics_seed", "horizon"}
        if set(spec) - allowed:
            raise ValueError("Unknown synthetic environment options")
        return LinearQuadraticEnv(int(spec.get("dynamics_seed", 0)), int(spec.get("horizon", 32)))
    if kind == "d4rl":
        if set(spec) != {"kind", "id"}:
            raise ValueError("D4RL evaluator does not accept candidate environment overrides")
        env_id = spec["id"]
        reference(env_id)
        os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")
        try:
            import gym
            import d4rl  # noqa: F401 -- registers the legacy environments
        except ImportError as exc:
            raise RuntimeError("Legacy D4RL/Gym/MuJoCo is required. See docs/d4rl.md; no Gymnasium substitution is performed.") from exc
        env = gym.make(env_id)
        lower, upper, obs_dim, action_dim = reference(env_id)
        if env.observation_space.shape != (obs_dim,) or env.action_space.shape != (action_dim,):
            env.close()
            raise ValueError("Legacy environment dimensions disagree with frozen protocol")
        if not np.allclose(env.action_space.low, -1) or not np.allclose(env.action_space.high, 1):
            env.close()
            raise ValueError("Unexpected D4RL action bounds")
        # Fail closed when an installed fork uses different normalization constants.
        if not np.isclose(float(env.get_normalized_score(lower)), 0) or not np.isclose(float(env.get_normalized_score(upper)), 1):
            env.close()
            raise ValueError("Installed D4RL scoring constants differ")
        return env
    raise ValueError("Unknown evaluation environment")


def make_fixture(path: Path, *, dynamics_seed: int = 0, data_seed: int = 1,
                 episodes: int = 64, horizon: int = 32) -> str:
    env = LinearQuadraticEnv(dynamics_seed, horizon)
    rng = np.random.default_rng(data_seed)
    rows: dict[str, list[Any]] = {k: [] for k in ("observations", "actions", "rewards", "next_observations", "terminals")}
    for _ in range(episodes):
        obs, _ = env.reset(seed=int(rng.integers(2**31-1)))
        for _ in range(horizon):
            # Fixed behavior mixture generated BEFORE research; never appended online.
            action = np.clip(-0.9 * env.B.T @ obs + rng.normal(0, 0.45, 2), -1, 1)
            nxt, reward, terminated, _, _ = env.step(action)
            for key, value in zip(rows, (obs, action, reward, nxt, terminated), strict=True):
                rows[key].append(value)
            obs = nxt
    arrays = {k: np.asarray(v, dtype=np.float32) for k, v in rows.items()}
    successors = np.arange(1, episodes * horizon + 1, dtype=np.int64)
    successors[horizon-1::horizon] = -1
    arrays["next_indices"] = successors
    return save_dataset(path, arrays, {
        "kind": "synthetic_lq", "dynamics_seed": dynamics_seed, "horizon": horizon,
        "data_seed": data_seed, "action_low": [-1.0, -1.0], "action_high": [1.0, 1.0],
        "warning": "Synthetic teaching fixture, not D4RL; score is raw return, never D4RL-normalized.",
    })
