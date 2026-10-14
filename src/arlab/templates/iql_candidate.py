"""Editable IQL source: same objectives and networks as the fixed baseline.

The open runtime separates value/policy phases and samples a context for each
phase, so trajectories need not be bitwise identical to the expression trainer.
Edit actor_weights for a new extraction mechanism while retaining Q/V learning.
"""
from __future__ import annotations

import copy

import torch

from arlab.algorithms.protocol import Algorithm, BuildSpec
from arlab.core.expressions import evaluate, scalar
from .networks import GaussianActor, TwinQ, Value


def vector(value, reference, label, *, nonnegative=False):
    if not isinstance(value, torch.Tensor):
        value = torch.full_like(reference, float(value))
    if value.ndim == 0:
        value = value.expand_as(reference)
    if value.shape != reference.shape or not torch.isfinite(value).all():
        raise ValueError(f"{label} must be a finite per-sample vector")
    if nonnegative and (value < 0).any():
        raise ValueError(f"{label} cannot be negative")
    return value


class EditableIQL(Algorithm):
    def __init__(self, spec: BuildSpec):
        super().__init__()
        self.cfg = spec.config
        cfg = self.cfg
        self.batch_size = cfg["batch_size"]
        actor = GaussianActor(spec.observation_dim, spec.action_dim, cfg["hidden_dims"]).to(spec.device)
        q = TwinQ(spec.observation_dim, spec.action_dim, cfg["hidden_dims"]).to(spec.device)
        target = copy.deepcopy(q).requires_grad_(False)
        v = Value(spec.observation_dim, cfg["hidden_dims"]).to(spec.device)
        self.policy_modules = {"actor": actor}
        self.value_modules = {"q": q, "target_q": target, "v": v}
        self.action_value_names = ["q", "target_q"]
        self.optimizers = {
            "actor": torch.optim.Adam(actor.parameters(), lr=cfg["actor_lr"]),
            "q": torch.optim.Adam(q.parameters(), lr=cfg["critic_lr"]),
            "v": torch.optim.Adam(v.parameters(), lr=cfg["value_lr"]),
        }
        self.schedulers = {"actor": torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizers["actor"], spec.total_ticks)}
        self.beta = cfg["beta"]

    def actor_weights(self, advantage):
        """Replace this method to implement rank weighting or another mechanism."""
        return evaluate(self.cfg["actor_weight"], {
            "adv": advantage, "beta": self.beta, "weight_cap": self.cfg["weight_cap"]})

    def value_step(self, ctx, name):
        b, cfg = ctx.batch, self.cfg
        q, target, v = (self.value_modules[k] for k in ("q", "target_q", "v"))
        with torch.no_grad():
            next_v = v(b.next_observations)
            tq1, tq2 = target(b.observations, b.actions)
            target_q = torch.minimum(tq1, tq2)
        advantage = target_q - v(b.observations)
        value_loss = vector(evaluate(cfg["value_loss"], {
            "delta": advantage, "expectile": cfg["expectile"]}), advantage, "value loss").mean()
        ctx.optimize(value_loss, "v", gradient_clip=cfg["gradient_clip"])
        bellman = b.rewards + cfg["discount"] * (1 - b.terminals) * next_v
        q1, q2 = q(b.observations, b.actions)
        q_loss = vector(evaluate(cfg["q_loss"], {
            "td1": q1 - bellman, "td2": q2 - bellman}), q1, "critic loss").mean()
        ctx.optimize(q_loss, "q", gradient_clip=cfg["gradient_clip"])
        ctx.soft_update("q", "target_q", cfg["target_rate"])
        self._batch = b
        self._advantage = advantage.detach()
        return {"value_loss": value_loss.detach().item(), "critic_loss": q_loss.detach().item()}

    def policy_step(self, ctx, name):
        b, advantage, cfg = self._batch, self._advantage, self.cfg
        with torch.no_grad():
            weights = vector(self.actor_weights(advantage), advantage,
                             "actor weights", nonnegative=True).clamp(max=cfg["weight_cap"])
            if weights.sum() <= 1e-12:
                raise ValueError("Actor weights have zero effective mass")
            ess = float(weights.sum().square() / (weights.square().sum() * len(weights)))
            if cfg["normalize_weights"]:
                weights = weights / weights.mean().clamp_min(1e-12)
        log_prob = self.policy_modules["actor"](b.observations).log_prob(b.actions).sum(-1)
        loss = -(weights * log_prob).mean()
        ctx.optimize(loss, "actor", gradient_clip=cfg["gradient_clip"])
        self.schedulers["actor"].step()
        beta = scalar(cfg["beta_update"], {
            "beta": self.beta, "ess_ratio": ess,
            "controller_rate": cfg["controller_rate"], "target_ess": cfg["target_ess"]})
        self.beta = max(cfg["beta_min"], min(cfg["beta_max"], beta))
        return {"actor_loss": loss.detach().item(), "ess_ratio": ess,
                "advantage_mean": advantage.mean().item(), "weight_mean": weights.mean().item()}

    def act(self, observations):
        return self.policy_modules["actor"](observations).mean

    def extra_state_dict(self):
        return {"beta": self.beta}

    def load_extra_state_dict(self, state):
        self.beta = float(state["beta"])


def build(spec: BuildSpec):
    return EditableIQL(spec)
