"""Editable in-sample-value learner; IQL is the baseline, not a structural ceiling.

Q/V updates use witnessed dataset pairs, with optional real-trajectory n-step
returns. Policy learning/extraction may sample new actions and query a frozen Q.
Change this SOURCE to research other targets/networks/schedules, not only config.
"""
from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from arlab.algorithms.protocol import Algorithm, BuildSpec, Update
from .networks import DiffusionActor, MixtureActor, QEnsemble, mlp


class OpenValueLearner(Algorithm):
    def __init__(self, spec: BuildSpec):
        super().__init__()
        cfg = self.cfg = spec.config
        widths = cfg.get("hidden_dims", [128, 128])
        self.batch_size = int(cfg.get("batch_size", 256))
        q = QEnsemble(spec.observation_dim, spec.action_dim, widths, int(cfg.get("ensemble", 3))).to(spec.device)
        self.value_modules = {
            "q": q, "target_q": copy.deepcopy(q).requires_grad_(False),
            "v": mlp(spec.observation_dim, widths, 1, True).to(spec.device),
        }
        self.action_value_names = ["q", "target_q"]
        actor_class = DiffusionActor if cfg.get("actor", "mixture") == "diffusion" else MixtureActor
        count = int(cfg.get("diffusion_steps", 8) if actor_class is DiffusionActor else cfg.get("mixtures", 4))
        actor = actor_class(spec.observation_dim, spec.action_dim, widths, count).to(spec.device)
        self.policy_modules = {"actor": actor}
        self.optimizers = {
            "q": torch.optim.Adam(q.parameters(), lr=cfg.get("q_lr", 3e-4)),
            "v": torch.optim.Adam(self.value_modules["v"].parameters(), lr=cfg.get("v_lr", 3e-4)),
            "actor": torch.optim.Adam(actor.parameters(), lr=cfg.get("actor_lr", 3e-4)),
        }
        self.schedulers = {}
        self.beta = float(cfg.get("beta", 3))

    def schedule(self, tick: int):
        updates = [Update("value", "fit")]
        # The generic runtime does not fix update order, frequency, or warmup.
        if tick >= int(self.cfg.get("value_warmup", 0)) and tick % int(self.cfg.get("policy_every", 1)) == 0:
            updates.append(Update("policy", "extract"))
        return updates

    def value_step(self, ctx, name):
        b = ctx.batch
        q, target, v = (self.value_modules[k] for k in ("q", "target_q", "v"))
        multi = ctx.data.n_step(b, int(self.cfg.get("n_step", 1)), float(self.cfg.get("discount", 0.99)))
        with torch.no_grad():
            next_v = v(multi.bootstrap_observations).squeeze(-1)
            data_q = target(b.observations, b.actions).min(0).values
        residual = data_q - v(b.observations).squeeze(-1)
        tau = float(self.cfg.get("expectile", 0.7))
        # This entire objective is editable. Signed offsets are intentionally legal.
        asymmetric = torch.where(residual > 0, tau, 1 - tau)
        v_loss = (asymmetric * residual.square()).mean() - float(self.cfg.get("loss_offset", 0))
        ctx.optimize(v_loss, "v")
        bellman = multi.rewards + multi.bootstrap_discount * next_v
        predicted = q(b.observations, b.actions)
        target_values = bellman[None].expand_as(predicted)
        q_loss = F.smooth_l1_loss(predicted, target_values) - float(self.cfg.get("loss_offset", 0))
        ctx.optimize(q_loss, "q")
        ctx.soft_update("q", "target_q", float(self.cfg.get("target_rate", 0.005)))
        return {"value_loss": v_loss.detach().item(), "q_loss": q_loss.detach().item(),
                "effective_n_step": multi.lengths.float().mean().item()}

    def policy_step(self, ctx, name):
        b = ctx.batch
        actor, q, v = self.policy_modules["actor"], self.value_modules["q"], self.value_modules["v"]
        with torch.no_grad():
            advantage = q(b.observations, b.actions).min(0).values - v(b.observations).squeeze(-1)
            weights = (self.beta * advantage).clamp(-20, 4.6).exp()
            weights = weights / weights.mean().clamp_min(1e-8)
        loss = (weights * actor.loss_per_sample(b.observations, b.actions)).mean()
        guidance = float(self.cfg.get("actor_q_guidance", 0))
        if guidance:
            generated = actor.sample(b.observations)
            # Legal: gradients to actor/actions, but frozen value parameters/buffers.
            loss = loss - guidance * q(b.observations, generated).mean()
        ctx.optimize(loss, "actor", gradient_clip=10)
        return {"actor_loss": loss.detach().item(), "weight_mean": weights.mean().item()}

    def act(self, observations):
        count = int(self.cfg.get("action_candidates", 8))
        with torch.no_grad():
            repeated = observations.repeat_interleave(count, 0)
            actions = self.policy_modules["actor"].sample(repeated)
            scores = self.value_modules["q"](repeated, actions).min(0).values.reshape(len(observations), count)
            proposals = actions.reshape(len(observations), count, -1)
            return proposals[torch.arange(len(observations), device=observations.device), scores.argmax(-1)]

    def extra_state_dict(self):
        return {"beta": self.beta}

    def load_extra_state_dict(self, state):
        self.beta = float(state["beta"])


def build(spec: BuildSpec):
    return OpenValueLearner(spec)
