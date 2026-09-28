"""Editable example networks; none of these classes are evaluator requirements."""
from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical, Normal


def mlp(inputs: int, widths: list[int], outputs: int, layer_norm: bool = False) -> nn.Sequential:
    layers = []
    for width in widths:
        layers.append(nn.Linear(inputs, width))
        if layer_norm:
            layers.append(nn.LayerNorm(width))
        layers.append(nn.SiLU())
        inputs = width
    layers.append(nn.Linear(inputs, outputs))
    return nn.Sequential(*layers)


class QEnsemble(nn.Module):
    def __init__(self, obs: int, act: int, widths: list[int], count: int):
        super().__init__()
        self.members = nn.ModuleList([mlp(obs + act, widths, 1, True) for _ in range(count)])

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat((observations, actions), -1)
        return torch.stack([m(inputs).squeeze(-1) for m in self.members])


class MixtureActor(nn.Module):
    def __init__(self, obs: int, act: int, widths: list[int], components: int):
        super().__init__()
        self.action_dim, self.components = act, components
        self.net = mlp(obs, widths, components * (2 * act + 1))

    def distribution(self, observations: torch.Tensor):
        raw = self.net(observations).reshape(-1, self.components, 2 * self.action_dim + 1)
        logits = raw[:, :, 0]
        mean = raw[:, :, 1:1+self.action_dim].tanh()
        std = raw[:, :, 1+self.action_dim:].clamp(-5, 1).exp()
        return logits, Normal(mean, std)

    def loss_per_sample(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        logits, normal = self.distribution(observations)
        return -torch.logsumexp(logits.log_softmax(-1) + normal.log_prob(actions[:, None]).sum(-1), -1)

    def sample(self, observations: torch.Tensor) -> torch.Tensor:
        logits, normal = self.distribution(observations)
        component = Categorical(logits=logits).sample()
        # Reparameterize within the sampled component, useful for optional actor guidance.
        proposals = normal.rsample()
        return proposals[torch.arange(len(observations), device=observations.device), component].clamp(-1, 1)


class DiffusionActor(nn.Module):
    """A compact conditional DDPM used to exercise the open interface.

    This teaching implementation is NOT a reproduction of an IDQL paper result.
    The candidate can edit the entire denoiser, noise schedule, loss, and sampler.
    """
    def __init__(self, obs: int, act: int, widths: list[int], steps: int):
        super().__init__()
        self.action_dim, self.steps = act, steps
        self.denoiser = mlp(obs + act + 2, widths, act)
        # Cosine cumulative signal variance reaches an approximately Gaussian
        # terminal distribution even for the short teaching noise schedule.
        time = torch.linspace(0, 1, steps + 1)
        cumulative = torch.cos((time + 0.008) / 1.008 * torch.pi / 2).square()
        cumulative = cumulative / cumulative[0]
        beta = (1 - cumulative[1:] / cumulative[:-1]).clamp(1e-5, 0.999)
        alpha = 1 - beta
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha.cumprod(0))

    def predict_noise(self, observations: torch.Tensor, noisy: torch.Tensor, t: torch.Tensor):
        phase = t.to(observations.dtype) / max(1, self.steps - 1)
        encoded = torch.stack((phase.sin(), phase.cos()), -1)
        return self.denoiser(torch.cat((observations, noisy, encoded), -1))

    def loss_per_sample(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        t = torch.randint(self.steps, (len(actions),), device=actions.device)
        noise = torch.randn_like(actions)
        a = self.alpha_bar[t, None]
        noisy = a.sqrt() * actions + (1 - a).sqrt() * noise
        prediction = self.predict_noise(observations, noisy, t)
        return (prediction - noise).square().mean(-1)

    def sample(self, observations: torch.Tensor) -> torch.Tensor:
        x = torch.randn(len(observations), self.action_dim, device=observations.device)
        for index in range(self.steps - 1, -1, -1):
            t = torch.full((len(x),), index, dtype=torch.long, device=x.device)
            epsilon = self.predict_noise(observations, x, t)
            x0 = ((x - (1 - self.alpha_bar[index]).sqrt() * epsilon)
                  / self.alpha_bar[index].sqrt()).clamp(-1, 1)
            if index == 0:
                x = x0
            else:
                previous = self.alpha_bar[index - 1]
                denominator = 1 - self.alpha_bar[index]
                mean = (previous.sqrt() * self.beta[index] / denominator * x0
                        + self.alpha[index].sqrt() * (1 - previous) / denominator * x)
                variance = self.beta[index] * (1 - previous) / denominator
                x = mean + variance.sqrt() * torch.randn_like(x)
        return x.clamp(-1, 1)
