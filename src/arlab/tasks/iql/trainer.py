"""IQL training: the entire network-query boundary is visible in update()."""
from __future__ import annotations

import copy
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from torch import nn

from arlab.core.expressions import evaluate, scalar
from arlab.core.checkpoints import training_binding, series_binding, schedule_horizon, checkpoint_source, validate_checkpoint
from arlab.core.io import digest, directory_digest, file_digest, read_json, runtime_info, write_json
from arlab.tasks.iql.data import Batch, OfflineReplay, load_dataset
from arlab.tasks.iql.networks import GaussianActor, TwinQ, Value
from arlab.tasks.iql.recipe import IQLRecipe


def checked_vector(value: Any, reference: torch.Tensor, label: str, *, nonnegative: bool) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        value = torch.full_like(reference, float(value))
    if value.ndim == 0:
        value = value.expand_as(reference)
    if value.shape != reference.shape or not torch.isfinite(value).all():
        raise ValueError(f"{label} must be a finite per-sample vector")
    if nonnegative and (value < 0).any():
        raise ValueError(f"{label} cannot be negative")
    return value


class IQLTrainer:
    def __init__(self, obs_dim: int, action_dim: int, recipe: IQLRecipe, total_steps: int,
                 device: str = "cpu"):
        self.recipe = recipe
        self.device = device
        self.actor = GaussianActor(obs_dim, action_dim, recipe.hidden_dims).to(device)
        self.critic = TwinQ(obs_dim, action_dim, recipe.hidden_dims).to(device)
        self.target = copy.deepcopy(self.critic).requires_grad_(False)
        self.value = Value(obs_dim, recipe.hidden_dims).to(device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=recipe.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=recipe.critic_lr)
        self.value_opt = torch.optim.Adam(self.value.parameters(), lr=recipe.value_lr)
        self.schedule = torch.optim.lr_scheduler.CosineAnnealingLR(self.actor_opt, total_steps)
        self.total_steps = total_steps
        self.step = 0
        self.beta = recipe.beta

    def _optimize(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer, network: nn.Module) -> None:
        if not loss.requires_grad or not torch.isfinite(loss):
            raise ValueError("Loss is non-finite or detached from its network")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in network.parameters()):
            raise ValueError("Non-finite gradient")
        if self.recipe.gradient_clip:
            nn.utils.clip_grad_norm_(network.parameters(), self.recipe.gradient_clip)
        optimizer.step()

    def update(self, batch: Batch) -> dict[str, float]:
        """Never queries Q(s, pi(s)), Q(s', pi(s')), random actions, or generated actions.

        The expression programs only see detached advantage / numeric residuals.
        They cannot access these networks. The only actor learning objective is
        nonnegative weighted likelihood of a_D, so all registered variants remain
        inside this declared in-sample family.
        """
        cfg = self.recipe
        with torch.no_grad():
            next_value = self.value(batch.next_observations)
            q1_target, q2_target = self.target(batch.observations, batch.actions)
            q_target = torch.minimum(q1_target, q2_target)
        value = self.value(batch.observations)
        advantage = q_target - value
        value_terms = checked_vector(evaluate(cfg.value_loss, {"delta": advantage, "expectile": cfg.expectile}),
                                     advantage, "value loss", nonnegative=False)
        value_loss = value_terms.mean()
        self._optimize(value_loss, self.value_opt, self.value)

        bellman = batch.rewards + cfg.discount * (1 - batch.terminals) * next_value
        q1, q2 = self.critic(batch.observations, batch.actions)
        q_terms = checked_vector(evaluate(cfg.q_loss, {"td1": q1 - bellman, "td2": q2 - bellman}),
                                 q1, "critic loss", nonnegative=False)
        q_loss = q_terms.mean()
        self._optimize(q_loss, self.critic_opt, self.critic)
        with torch.no_grad():
            for target_parameter, parameter in zip(self.target.parameters(), self.critic.parameters(), strict=True):
                target_parameter.lerp_(parameter, cfg.target_rate)
            weights = checked_vector(evaluate(cfg.actor_weight, {"adv": advantage.detach(),
                                      "beta": self.beta, "weight_cap": cfg.weight_cap}), advantage,
                                      "actor weights", nonnegative=True).clamp(max=cfg.weight_cap)
            if weights.sum() <= 1e-12:
                raise ValueError("Actor weights have zero effective mass")
            ess_ratio = float(weights.sum().square() / (weights.square().sum() * len(weights)))
            if cfg.normalize_weights:
                weights = weights / weights.mean().clamp_min(1e-12)
        # In-sample policy extraction: evaluate log pi(a_D|s), never Q(s, a_pi).
        log_prob = self.actor(batch.observations).log_prob(batch.actions).sum(-1)
        actor_loss = -(weights.detach() * log_prob).mean()
        self._optimize(actor_loss, self.actor_opt, self.actor)
        self.schedule.step()
        new_beta = scalar(cfg.beta_update, {"beta": self.beta, "ess_ratio": ess_ratio,
                          "controller_rate": cfg.controller_rate, "target_ess": cfg.target_ess})
        self.beta = max(cfg.beta_min, min(cfg.beta_max, new_beta))
        self.step += 1
        return {"value_loss": float(value_loss.detach()), "critic_loss": float(q_loss.detach()),
                "actor_loss": float(actor_loss.detach()), "advantage_mean": float(advantage.detach().mean()),
                "ess_ratio": ess_ratio, "beta": self.beta,
                "weight_max": float(weights.max()), "weight_mean": float(weights.mean())}

    def state_dict(self) -> dict[str, Any]:
        return {"actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                "target": self.target.state_dict(), "value": self.value.state_dict(),
                "actor_opt": self.actor_opt.state_dict(), "critic_opt": self.critic_opt.state_dict(),
                "value_opt": self.value_opt.state_dict(), "schedule": self.schedule.state_dict(),
                "step": self.step, "beta": self.beta, "total_steps": self.total_steps,
                "recipe_hash": digest(self.recipe)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state["recipe_hash"] != digest(self.recipe) or state["total_steps"] != self.total_steps:
            raise ValueError("Checkpoint recipe or training horizon mismatch")
        for name in ("actor", "critic", "target", "value", "actor_opt", "critic_opt", "value_opt", "schedule"):
            getattr(self, name).load_state_dict(state[name])
        self.step, self.beta = int(state["step"]), float(state["beta"])


def checkpoint(path: Path, trainer: IQLTrainer, replay: OfflineReplay, binding: str, *, series: str | None = None) -> None:
    state = {"trainer": trainer.state_dict(), "replay_rng": replay.generator.get_state(),
             "torch_rng": torch.get_rng_state(), "binding": binding,
             "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    if series is not None:
        state["series_sha256"] = series
    temporary = path.with_suffix(".pending")
    torch.save(state, temporary)
    os.replace(temporary, path)


def export_policy(path: Path, trainer: IQLTrainer, replay: OfflineReplay, obs_dim: int,
                  action_dim: int, binding: str) -> str:
    path.mkdir(parents=True, exist_ok=True)
    state = {"actor." + k: v.detach().cpu().contiguous() for k, v in trainer.actor.state_dict().items()}
    state.update({"observation_mean": torch.as_tensor(replay.mean), "observation_std": torch.as_tensor(replay.std),
                  "action_low": torch.as_tensor(replay.low), "action_high": torch.as_tensor(replay.high)})
    temp = path / ".weights.pending"
    save_file(state, str(temp))
    os.replace(temp, path / "weights.safetensors")
    sha = file_digest(path / "weights.safetensors")
    write_json(path / "policy.json", {"schema": 1, "architecture": "iql-gaussian-mean-v1",
               "observation_dim": obs_dim, "action_dim": action_dim, "hidden_dims": trainer.recipe.hidden_dims,
               "weights_sha256": sha, "training_binding": binding, "updates": trainer.step})
    return directory_digest(path)


def train(request: dict[str, Any], output: Path) -> dict[str, Any]:
    """Run or resume a bounded experiment. Environment APIs are not imported here."""
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    binding = training_binding(request)
    complete_path = output / "training.json"
    if complete_path.exists():
        done = read_json(complete_path)
        if done["request_sha256"] != binding:
            raise ValueError("Completed training artifact belongs to another request")
        if directory_digest(output / "artifact") != done["artifact_sha256"]:
            raise ValueError("Completed policy artifact was altered")
        return done
    recipe = IQLRecipe.model_validate(request["recipe"])
    device = request.get("device", "cpu")
    seed, steps = int(request["seed"]), int(request["steps"])
    if steps < 1:
        raise ValueError("Training steps must be positive")
    torch.set_num_threads(int(request.get("threads", 1)))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    arrays, metadata = load_dataset(Path(request["data"]), request.get("dataset_sha256", ""))
    replay = OfflineReplay(arrays, metadata, seed)
    obs_dim, action_dim = arrays["observations"].shape[1], arrays["actions"].shape[1]
    horizon = schedule_horizon(request)
    trainer = IQLTrainer(obs_dim, action_dim, recipe, horizon, device)
    checkpoint_path = output / "checkpoint.pt"
    restore = checkpoint_source(request, checkpoint_path)
    if restore:
        path, expected = restore
        state = torch.load(path, map_location=device, weights_only=True)
        validate_checkpoint(state, request, expected, int(state["trainer"]["step"]))
        trainer.load_state_dict(state["trainer"])
        replay.generator.set_state(state["replay_rng"].cpu())
        torch.set_rng_state(state["torch_rng"].cpu())
        if device.startswith("cuda") and state["cuda_rng"]:
            torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda_rng"]])
    resumed_at = trainer.step
    diagnostics: dict[str, float] = {}
    period = min(1000, max(1, steps // 10))
    with (output / "metrics.jsonl").open("a", encoding="utf-8") as log:
        from arlab.core.io import canonical
        while trainer.step < steps:
            diagnostics = trainer.update(replay.sample(recipe.batch_size, device))
            if trainer.step % period == 0 or trainer.step == steps:
                log.write(canonical({"step": trainer.step, **diagnostics}) + "\n")
                log.flush()
                checkpoint(checkpoint_path, trainer, replay, binding, series=series_binding(request))
    if not diagnostics:
        rows = (output / "metrics.jsonl").read_text().splitlines()
        import json
        diagnostics = {k: v for k, v in json.loads(rows[-1]).items() if k != "step"}
    if not all(math.isfinite(v) for v in diagnostics.values()):
        raise ValueError("Non-finite final diagnostics")
    sha = export_policy(output / "artifact", trainer, replay, obs_dim, action_dim, binding)
    result = {"request_sha256": binding, "artifact_sha256": sha, "steps": trainer.step,
              "resumed_at": resumed_at, "schedule_steps": horizon,
              "series_sha256": series_binding(request), "checkpoint_sha256": file_digest(checkpoint_path),
              "train_seconds": time.monotonic() - started,
              "diagnostics": diagnostics, "dataset_kind": metadata["kind"],
              "in_sample": True, "optimizer_steps": trainer.step * 3,
              "value_learning_audit": {"mode": "fixed_iql", "value_actions": "recorded_pairs",
                  "policy_generated_q_queries": 0, "formal_verification": False}, "recipe_sha256": digest(recipe), "runtime": runtime_info()}
    write_json(complete_path, result)
    return result
