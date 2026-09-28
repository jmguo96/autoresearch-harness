"""Data provenance and checked optimization, without specifying the learning objective."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from arlab.algorithms.guard import BoundaryViolation, ValueBoundary
from arlab.tasks.iql.data import Batch, OfflineReplay


@dataclass(frozen=True)
class MultiStep:
    rewards: torch.Tensor
    bootstrap_observations: torch.Tensor
    bootstrap_discount: torch.Tensor
    lengths: torch.Tensor


class RecordedData:
    """Only this adapter registers new observed state-action witnesses with the guard."""
    def __init__(self, replay: OfflineReplay, guard: ValueBoundary, device: str):
        self._replay, self._guard, self.device = replay, guard, device

    @property
    def size(self) -> int:
        return self._replay.size

    def take(self, indices: Any) -> Batch:
        batch = self._replay.take(torch.as_tensor(indices, dtype=torch.long, device="cpu"), self.device)
        self._guard.witness(batch)
        return batch

    def successors(self, batch: Batch) -> tuple[torch.Tensor, Batch | None]:
        """Positions and actual next recorded pairs; never synthesize next actions."""
        if batch.indices is None or not self._replay.has_trajectory_links:
            raise ValueError("Recorded successors require explicit trajectory links")
        current = self._replay.take(batch.indices, "cpu")
        links = self._replay.next_indices[current.indices]
        positions = torch.nonzero((links >= 0) & (current.terminals == 0)).squeeze(-1)
        return positions, self.take(links[positions]) if len(positions) else None

    def sample(self, size: int) -> Batch:
        batch = self._replay.sample(size, self.device)
        self._guard.witness(batch)
        return batch

    def n_step(self, batch: Batch, horizon: int, discount: float) -> MultiStep:
        """Follow explicit successors only; missing links truncate, never fabricate.

        At a dataset/time-limit cut bootstrap from V(s_next); at true terminal use
        zero bootstrap. Old files without links must be reimported for multi-step.
        """
        if batch.indices is None or not 1 <= horizon <= 1024 or not 0 <= discount <= 1:
            raise ValueError("Invalid multi-step request")
        if horizon > 1 and not self._replay.has_trajectory_links:
            raise ValueError("Multi-step targets require explicit trajectory links; reimport this dataset")
        # Fetch again so a mutated batch cannot alter rewards or successor identities.
        current = self.take(batch.indices)
        count = len(current.rewards)
        rewards = torch.zeros_like(current.rewards)
        factors = torch.ones_like(current.rewards)
        lengths = torch.zeros(count, dtype=torch.long, device=self.device)
        next_obs = current.next_observations.clone()
        slots = torch.arange(count, device=self.device)
        for _ in range(horizon):
            rewards[slots] += factors[slots] * current.rewards
            factors[slots] *= discount * (1 - current.terminals)
            lengths[slots] += 1
            next_obs[slots] = current.next_observations
            links = self._replay.next_indices[current.indices]
            valid = (links >= 0) & (current.terminals.detach().cpu() == 0)
            if not valid.any():
                break
            slots = slots[valid.to(self.device)]
            current = self.take(links[valid])
        return MultiStep(rewards, next_obs, factors, lengths)


class UpdateContext:
    def __init__(self, learner: Any, guard: ValueBoundary, replay: OfflineReplay,
                 *, tick: int, device: str, remaining_optimizations: int):
        self.learner, self.guard, self.tick = learner, guard, tick
        self.data = RecordedData(replay, guard, device)
        self.batch = self.data.sample(int(learner.batch_size))
        self.remaining_optimizations = remaining_optimizations
        self.optimizations = 0

    def optimize(self, loss: torch.Tensor, optimizer: str, *, gradient_clip: float = 0,
                 retain_graph: bool = False) -> None:
        """Signed finite scalar losses are valid; no nonnegative-loss restriction.

        Optimizer ownership is checked by actual Parameter identity, including
        shared encoders. Frozen value parameters still allow gradients to actions.
        """
        if self.optimizations >= self.remaining_optimizations:
            raise BoundaryViolation("Candidate exceeded its optimizer-step reservation")
        from arlab.algorithms.provenance import PolicyOnlyValue
        if self.guard.phase == "value" and isinstance(loss, PolicyOnlyValue):
            raise BoundaryViolation("Policy-only generated-action value leaked into a Q/V objective")
        if loss.ndim != 0 or not loss.requires_grad or not torch.isfinite(loss):
            raise ValueError("Objective must be a finite, differentiable scalar")
        if optimizer not in self.learner.optimizers:
            raise ValueError("Unregistered optimizer")
        opt = self.learner.optimizers[optimizer]
        parameters = [p for group in opt.param_groups for p in group["params"]]
        if not parameters or any(id(p) not in self.guard.allowed_parameter_ids() for p in parameters):
            raise BoundaryViolation("Optimizer touches parameters owned by another phase")
        opt.zero_grad(set_to_none=True)
        loss.backward(retain_graph=retain_graph)
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
            raise ValueError("Non-finite gradient")
        if not any(p.grad is not None for p in parameters):
            raise ValueError("Objective is disconnected from the selected optimizer")
        if gradient_clip < 0:
            raise ValueError("gradient_clip cannot be negative")
        if gradient_clip:
            torch.nn.utils.clip_grad_norm_(parameters, gradient_clip)
        opt.step()
        self.optimizations += 1

    def soft_update(self, source: str, target: str, rate: float) -> None:
        if self.guard.phase != "value" or not 0 <= rate <= 1:
            raise BoundaryViolation("Value targets can only change during value updates")
        left, right = self.learner.value_modules[source], self.learner.value_modules[target]
        source_state, target_state = left.state_dict(), right.state_dict()
        if source_state.keys() != target_state.keys():
            raise ValueError("Target/source state structures differ")
        with torch.no_grad():
            for key, value in target_state.items():
                other = source_state[key]
                value.lerp_(other, rate) if value.is_floating_point() else value.copy_(other)
