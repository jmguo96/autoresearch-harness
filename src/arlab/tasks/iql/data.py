"""Dataset import and an immutable replay buffer. There is no online append API."""
from __future__ import annotations

import json
import os
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from arlab.core.io import canonical, file_digest

FIELDS = ("observations", "actions", "rewards", "next_observations", "terminals")
OPTIONAL_FIELDS = ("next_indices",)


def validate_arrays(data: dict[str, np.ndarray]) -> None:
    if any(key not in data for key in FIELDS):
        raise ValueError("Missing transition fields")
    n = len(data["rewards"])
    if n < 2:
        raise ValueError("At least two transitions are required")
    for key in FIELDS:
        x = data[key]
        if len(x) != n or not np.issubdtype(x.dtype, np.number) and x.dtype != np.bool_:
            raise ValueError(f"Invalid shape or dtype: {key}")
        if not np.isfinite(x).all():
            raise ValueError(f"Non-finite dataset field: {key}")
    if data["observations"].ndim != 2 or data["actions"].ndim != 2:
        raise ValueError("Observations and actions must be matrices")
    if data["next_observations"].shape != data["observations"].shape:
        raise ValueError("Current and next observation shapes differ")
    if data["rewards"].shape != (n,) or data["terminals"].shape != (n,):
        raise ValueError("Rewards and terminals must be vectors")
    if not np.isin(data["terminals"], [0, 1]).all():
        raise ValueError("Terminal flags must be Boolean")
    if "next_indices" in data:
        nxt = data["next_indices"]
        if nxt.shape != (n,) or not np.issubdtype(nxt.dtype, np.integer):
            raise ValueError("next_indices must be an integer vector")
        if ((nxt < -1) | (nxt >= n)).any():
            raise ValueError("Invalid trajectory successor index")
        valid = nxt >= 0
        if ((data["terminals"] != 0) & valid).any():
            raise ValueError("Terminal transitions must not have successors")
        # Forward-only links rule out cycles; exact observations detect accidental joins.
        if (nxt[valid] <= np.flatnonzero(valid)).any():
            raise ValueError("Trajectory successor must move forward")
        if not np.array_equal(data["next_observations"][valid], data["observations"][nxt[valid]]):
            raise ValueError("Trajectory successor does not match next_observation")



def save_dataset(path: Path, arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> str:
    validate_arrays(arrays)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".dataset-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **{k: np.asarray(arrays[k], dtype=np.float32) for k in FIELDS},
                                **({"next_indices": np.asarray(arrays["next_indices"], dtype=np.int64)} if "next_indices" in arrays else {}),
                                metadata=np.asarray(canonical(metadata)))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    return file_digest(path)


def load_dataset(path: Path, expected_sha256: str = "") -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if expected_sha256 and file_digest(path) != expected_sha256:
        raise ValueError("Dataset content hash does not match its contract")
    # Numeric-only npz, bounded expansion; never unpickle an uploaded dataset.
    with zipfile.ZipFile(path) as archive:
        if sum(x.file_size for x in archive.infolist()) > 4_000_000_000:
            raise ValueError("Dataset exceeds uncompressed size limit")
    with np.load(path, allow_pickle=False) as z:
        if not {*FIELDS, "metadata"}.issubset(z.files) or set(z.files) - {*FIELDS, *OPTIONAL_FIELDS, "metadata"}:
            raise ValueError("Dataset must expose only the approved training fields")
        arrays = {key: z[key].astype(np.float32) for key in FIELDS}
        if "next_indices" in z.files:
            links = z["next_indices"]
            if not np.issubdtype(links.dtype, np.integer):
                raise ValueError("next_indices must be stored as integers")
            arrays["next_indices"] = links.astype(np.int64)
        metadata = json.loads(str(z["metadata"]))
    validate_arrays(arrays)
    low, high = np.asarray(metadata["action_low"]), np.asarray(metadata["action_high"])
    dim = arrays["actions"].shape[1]
    if low.shape != (dim,) or high.shape != (dim,) or not np.isfinite(low).all() or not np.isfinite(high).all() or not np.all(high > low):
        raise ValueError("Invalid action bounds")
    if (arrays["actions"] < low - 1e-4).any() or (arrays["actions"] > high + 1e-4).any():
        raise ValueError("Dataset actions violate declared bounds")
    return arrays, metadata


def convert_d4rl(raw: dict[str, np.ndarray], max_episode_steps: int = 1000) -> dict[str, np.ndarray]:
    """Match D4RL qlearning_dataset(terminate_on_end=False) for explicit timeouts.

    Only a_D is retained. Timeout rows are dropped before shifting, true terminals
    mask the bootstrap, and the last raw row is dropped. D4RL-v2 provides timeouts;
    without them this importer refuses to guess boundaries (an off-by-one-prone
    fallback would silently change the benchmark).
    """
    del max_episode_steps  # kept explicit in the caller's protocol, never guessed here
    required = {"observations", "actions", "rewards", "terminals", "timeouts"}
    if not required.issubset(raw):
        raise ValueError("D4RL-v2 import requires explicit timeouts")
    n = len(raw["rewards"])
    if any(len(raw[k]) != n for k in required):
        raise ValueError("Inconsistent raw dataset lengths")
    if not np.isin(raw["timeouts"], [0, 1]).all():
        raise ValueError("Invalid timeout flags")
    keep = np.flatnonzero(~np.asarray(raw["timeouts"][:-1], dtype=bool))
    arrays = {"observations": raw["observations"][keep],
              "actions": raw["actions"][keep], "rewards": raw["rewards"][keep],
              "next_observations": raw["observations"][keep + 1],
              "terminals": raw["terminals"][keep]}
    arrays = {k: np.asarray(x, dtype=np.float32) for k, x in arrays.items()}
    # Preserve true raw-row adjacency AFTER dropping timeout rows. Never infer
    # trajectories from proximity of states or join across episode boundaries.
    raw_to_kept = np.full(n, -1, dtype=np.int64)
    raw_to_kept[keep] = np.arange(len(keep))
    successors = raw_to_kept[keep + 1]
    successors[np.asarray(arrays["terminals"], dtype=bool)] = -1
    arrays["next_indices"] = successors
    validate_arrays(arrays)
    return arrays


def import_hdf5(source: Path, destination: Path, env_id: str, *, action_dim: int) -> str:
    """Import only whitelisted raw fields; discard simulator state and policy metadata."""
    with h5py.File(source, "r") as h5:
        raw = {k: np.asarray(h5[k]) for k in ("observations", "actions", "rewards", "terminals", "timeouts")}
    arrays = convert_d4rl(raw)
    if arrays["actions"].shape[1] != action_dim:
        raise ValueError("Environment and dataset action dimensions differ")
    return save_dataset(destination, arrays, {
        "kind": "d4rl", "environment": env_id, "raw_sha256": file_digest(source),
        "conversion": "d4rl-v2-qlearning-explicit-timeouts-drop-with-links-v2", "reward_transform": "identity",
        "action_low": [-1.0] * action_dim, "action_high": [1.0] * action_dim,
    })


@dataclass(frozen=True)
class Batch:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor
    terminals: torch.Tensor
    indices: torch.Tensor | None = None


class OfflineReplay:
    """Sampling is the only data operation available to the trainer."""
    def __init__(self, arrays: dict[str, np.ndarray], metadata: dict[str, Any], seed: int):
        validate_arrays(arrays)
        self.mean = arrays["observations"].mean(axis=0)
        self.std = arrays["observations"].std(axis=0) + 1e-3
        self.low = np.asarray(metadata["action_low"], dtype=np.float32)
        self.high = np.asarray(metadata["action_high"], dtype=np.float32)
        normalized = {
            "observations": (arrays["observations"] - self.mean) / self.std,
            "next_observations": (arrays["next_observations"] - self.mean) / self.std,
            "actions": np.clip(2 * (arrays["actions"] - self.low) / (self.high - self.low) - 1, -1, 1),
            "rewards": arrays["rewards"], "terminals": arrays["terminals"],
        }
        self._data = {key: torch.as_tensor(value.copy(), dtype=torch.float32) for key, value in normalized.items()}
        self.generator = torch.Generator().manual_seed(seed)
        self.size = len(arrays["rewards"])
        # Missing links in old files mean unknown adjacency, not an inferred chain.
        self.has_trajectory_links = "next_indices" in arrays
        self.next_indices = torch.as_tensor(arrays.get("next_indices", np.full(self.size, -1, dtype=np.int64))).clone()

    def sample(self, size: int, device: str) -> Batch:
        idx = torch.randint(self.size, (size,), generator=self.generator)
        return self.take(idx, device)

    def take(self, indices: torch.Tensor, device: str) -> Batch:
        idx = torch.as_tensor(indices, dtype=torch.long, device="cpu").reshape(-1)
        if not len(idx) or ((idx < 0) | (idx >= self.size)).any():
            raise ValueError("Dataset row index outside replay")
        return Batch(**{k: x[idx].to(device).clone() for k, x in self._data.items()}, indices=idx.clone())
