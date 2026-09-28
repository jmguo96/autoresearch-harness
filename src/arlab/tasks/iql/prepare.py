"""Owner-run dataset acquisition; never exposed as a model network/file tool."""
from __future__ import annotations

from pathlib import Path

from arlab.deployment.download import fetch_https
from arlab.tasks.iql.environments import reference


def download_raw(env_id: str, destination: Path, expected_sha256: str = "") -> str:
    reference(env_id)
    env, suffix = env_id.split("-", 1)
    quality = suffix.removesuffix("-v2").replace("-", "_")
    filename = f"{env}_{quality}-v2.hdf5"
    url = "https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/" + filename
    return fetch_https(url, destination, expected_sha256=expected_sha256,
                       allowed_hosts={"rail.eecs.berkeley.edu"})
