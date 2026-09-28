"""Minimal IQL networks. Evaluator imports only this trusted architecture."""
from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal


def mlp(input_dim: int, hidden: list[int], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for width in hidden:
        layers.extend([nn.Linear(input_dim, width), nn.ReLU()])
        input_dim = width
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


class GaussianActor(nn.Module):
    """Tanh mean, global diagonal Gaussian std; likelihood is over dataset actions.

    This is an unsquashed Normal with a bounded mean, not a tanh-transformed
    distribution. Deterministic evaluation uses the mean. No Jacobian correction
    is appropriate for this particular parameterization.
    """
    def __init__(self, obs_dim: int, action_dim: int, hidden: list[int]):
        super().__init__()
        self.net = mlp(obs_dim, hidden, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, observations: torch.Tensor) -> Normal:
        return Normal(torch.tanh(self.net(observations)), self.log_std.clamp(-20, 2).exp())


class TwinQ(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: list[int]):
        super().__init__()
        self.q1 = mlp(obs_dim + action_dim, hidden, 1)
        self.q2 = mlp(obs_dim + action_dim, hidden, 1)

    def forward(self, observations: torch.Tensor, dataset_actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat((observations, dataset_actions), dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


class Value(nn.Module):
    def __init__(self, obs_dim: int, hidden: list[int]):
        super().__init__()
        self.net = mlp(obs_dim, hidden, 1)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations).squeeze(-1)
