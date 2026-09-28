from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from arlab.tasks.iql.data import OfflineReplay, convert_d4rl, import_hdf5, load_dataset, save_dataset
from arlab.tasks.iql.environments import normalized_score, reference


def raw_dataset():
    return {"observations": np.arange(18, dtype=np.float32).reshape(6, 3),
            "actions": np.zeros((6, 2), dtype=np.float32), "rewards": np.arange(6, dtype=np.float32),
            "terminals": np.array([0, 0, 0, 1, 0, 0], dtype=bool),
            "timeouts": np.array([0, 1, 0, 0, 1, 0], dtype=bool)}


def test_timeout_conversion_matches_frozen_d4rl_semantics():
    raw = raw_dataset()
    converted = convert_d4rl(raw)
    np.testing.assert_array_equal(converted["observations"], raw["observations"][[0, 2, 3]])
    np.testing.assert_array_equal(converted["next_observations"], raw["observations"][[1, 3, 4]])
    np.testing.assert_array_equal(converted["terminals"], [0, 0, 1])
    # The terminal row's next state is masked by (1-terminal), not bootstrapped.
    assert not np.any(converted["observations"][:, 0] == raw["observations"][1, 0])


def test_missing_timeouts_are_not_guessed():
    raw = raw_dataset()
    raw.pop("timeouts")
    with pytest.raises(ValueError, match="timeouts"):
        convert_d4rl(raw)


def test_unequal_raw_lengths_rejected():
    raw = raw_dataset()
    raw["actions"] = raw["actions"][:-1]
    with pytest.raises(ValueError):
        convert_d4rl(raw)


def test_hdf_import_strips_oracle_metadata(tmp_path):
    raw = raw_dataset()
    source = tmp_path / "raw.hdf5"
    with h5py.File(source, "w") as stream:
        for key, value in raw.items():
            stream[key] = value
        stream["infos/qpos"] = np.ones((6, 20))
        stream["metadata/policy_weights"] = np.ones((100,))
    dest = tmp_path / "clean.npz"
    import_hdf5(source, dest, "test-fixture", action_dim=2)
    with np.load(dest, allow_pickle=False) as z:
        assert set(z.files) == {"observations", "actions", "next_observations", "rewards", "terminals", "next_indices", "metadata"}


def test_dataset_integrity(dataset):
    path, sha = dataset
    data, meta = load_dataset(path, sha)
    assert data["observations"].shape[1] == 3
    with pytest.raises(ValueError, match="hash"):
        load_dataset(path, "0" * 64)


def test_replay_has_no_online_mutation_api(dataset):
    path, _ = dataset
    data, meta = load_dataset(path)
    replay = OfflineReplay(data, meta, 0)
    assert not hasattr(replay, "add_transition") and not hasattr(replay, "append")
    first = replay.generator.get_state()
    batch = replay.sample(16, "cpu")
    replay.generator.set_state(first)
    other = replay.sample(16, "cpu")
    import torch
    torch.testing.assert_close(batch.actions, other.actions)


@pytest.mark.parametrize("env", ["hopper-medium-v2", "halfcheetah-random-v2", "walker2d-medium-expert-v2"])
def test_d4rl_normalization_is_percent_and_not_clipped(env):
    lo, hi, _, _ = reference(env)
    assert normalized_score(env, lo) == 0
    assert normalized_score(env, hi) == 100
    assert normalized_score(env, 2 * hi - lo) == pytest.approx(200)


def test_dataset_nonfinite_rejected(dataset, tmp_path):
    data, meta = load_dataset(dataset[0])
    data["rewards"][0] = np.nan
    with pytest.raises(ValueError, match="Non-finite"):
        save_dataset(tmp_path / "bad.npz", data, meta)


def test_npz_extra_fields_rejected(dataset, tmp_path):
    data, meta = load_dataset(dataset[0])
    path = tmp_path / "oracle.npz"
    np.savez(path, **data, metadata=json.dumps(meta), policy_weights=np.ones(3))
    with pytest.raises(ValueError, match="approved"):
        load_dataset(path)
