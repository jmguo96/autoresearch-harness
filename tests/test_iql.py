from __future__ import annotations

import copy

import pytest
import torch
from safetensors.torch import load_file

from arlab.core.io import digest, directory_digest, read_json, write_json
from arlab.tasks.iql.data import OfflineReplay, load_dataset
from arlab.tasks.iql.evaluator import ExportedPolicy, evaluate_policy
from arlab.tasks.iql.recipe import IQLRecipe
from arlab.tasks.iql.trainer import IQLTrainer, train


def tiny_request(dataset, steps=12):
    path, sha = dataset
    return {"recipe": IQLRecipe(hidden_dims=[16, 16], batch_size=16).model_dump(mode="json"),
            "data": str(path), "dataset_sha256": sha, "seed": 7, "steps": steps, "device": "cpu"}


def test_all_critic_queries_use_dataset_state_actions(dataset, monkeypatch):
    arrays, meta = load_dataset(dataset[0])
    replay = OfflineReplay(arrays, meta, 1)
    batch = replay.sample(16, "cpu")
    trainer = IQLTrainer(3, 2, IQLRecipe(hidden_dims=[16], batch_size=16), 10)
    queried = []
    def check(module, args):
        obs, act = args
        torch.testing.assert_close(obs, batch.observations, rtol=0, atol=0)
        torch.testing.assert_close(act, batch.actions, rtol=0, atol=0)
        queried.append(type(module).__name__)
    trainer.critic.register_forward_pre_hook(check)
    trainer.target.register_forward_pre_hook(check)
    def forbidden(*args, **kwargs):
        raise AssertionError("Training must not sample policy actions")
    monkeypatch.setattr(torch.distributions.Normal, "sample", forbidden)
    monkeypatch.setattr(torch.distributions.Normal, "rsample", forbidden)
    diagnostics = trainer.update(batch)
    assert len(queried) == 2
    assert 0 < diagnostics["ess_ratio"] <= 1.000001


def test_zero_weight_program_fails(dataset):
    arrays, meta = load_dataset(dataset[0])
    recipe = IQLRecipe(hidden_dims=[16], batch_size=16, actor_weight={"const": 0})
    trainer = IQLTrainer(3, 2, recipe, 2)
    with pytest.raises(ValueError, match="zero effective mass"):
        trainer.update(OfflineReplay(arrays, meta, 0).sample(16, "cpu"))


def test_negative_weight_program_fails(dataset):
    arrays, meta = load_dataset(dataset[0])
    recipe = IQLRecipe(hidden_dims=[16], batch_size=16, actor_weight={"const": -1})
    with pytest.raises(ValueError, match="negative"):
        IQLTrainer(3, 2, recipe, 2).update(OfflineReplay(arrays, meta, 0).sample(16, "cpu"))


def test_checkpoint_resume_matches_uninterrupted_training(dataset, tmp_path, monkeypatch):
    request = tiny_request(dataset, 12)
    train(request, tmp_path / "full")
    original_update = IQLTrainer.update
    def interrupt(self, batch):
        if self.step == 5:
            raise RuntimeError("injected crash")
        return original_update(self, batch)
    monkeypatch.setattr(IQLTrainer, "update", interrupt)
    with pytest.raises(RuntimeError, match="injected"):
        train(request, tmp_path / "resumed")
    monkeypatch.setattr(IQLTrainer, "update", original_update)
    result = train(request, tmp_path / "resumed")
    assert 0 < result["resumed_at"] <= 5
    full = load_file(str(tmp_path / "full/artifact/weights.safetensors"))
    resumed = load_file(str(tmp_path / "resumed/artifact/weights.safetensors"))
    for name in full:
        torch.testing.assert_close(full[name], resumed[name], rtol=0, atol=0)


def test_train_resume_rejects_changed_horizon(dataset, tmp_path):
    request = tiny_request(dataset)
    train(request, tmp_path / "out")
    request["steps"] += 1
    with pytest.raises(ValueError, match="another request"):
        train(request, tmp_path / "out")


def test_export_evaluate_and_idempotence(dataset, tmp_path):
    request = tiny_request(dataset)
    trained = train(request, tmp_path / "train")
    assert train(request, tmp_path / "train") == trained
    er = {"artifact": str(tmp_path / "train/artifact"), "artifact_sha256": trained["artifact_sha256"],
          "training_binding": digest(request), "steps": request["steps"], "eval_seeds": [10, 11],
          "environment": {"kind": "synthetic_lq", "dynamics_seed": 0, "horizon": 8}}
    scored = evaluate_policy(er, tmp_path / "eval")
    assert scored["metric"] == "synthetic_raw_return"
    assert len(scored["returns"]) == 2 and scored["episode_lengths"] == [8, 8]
    assert evaluate_policy(er, tmp_path / "eval") == scored
    assert directory_digest(tmp_path / "train/artifact") == trained["artifact_sha256"]


def test_tampered_artifact_is_rejected(dataset, tmp_path):
    request = tiny_request(dataset, 2)
    trained = train(request, tmp_path / "train")
    directory = tmp_path / "train/artifact"
    manifest = read_json(directory / "policy.json")
    manifest["updates"] = 1000000
    write_json(directory / "policy.json", manifest)
    with pytest.raises(ValueError, match="hash"):
        ExportedPolicy(directory, trained["artifact_sha256"], digest(request))


def test_unbounded_architecture_is_rejected_before_instantiation(dataset, tmp_path):
    request = tiny_request(dataset, 2)
    trained = train(request, tmp_path / "train")
    directory = tmp_path / "train/artifact"
    manifest = read_json(directory / "policy.json")
    manifest["hidden_dims"] = [1000000]
    write_json(directory / "policy.json", manifest)
    with pytest.raises(ValueError):
        ExportedPolicy(directory, trained["artifact_sha256"], digest(request))


def test_target_checkpoint_is_not_reinitialized(dataset):
    arrays, meta = load_dataset(dataset[0])
    replay = OfflineReplay(arrays, meta, 0)
    trainer = IQLTrainer(3, 2, IQLRecipe(hidden_dims=[16]), 10)
    trainer.update(replay.sample(16, "cpu"))
    saved = copy.deepcopy(trainer.state_dict())
    restored = IQLTrainer(3, 2, trainer.recipe, 10)
    restored.load_state_dict(saved)
    for k, x in trainer.target.state_dict().items():
        torch.testing.assert_close(x, restored.target.state_dict()[k], rtol=0, atol=0)
