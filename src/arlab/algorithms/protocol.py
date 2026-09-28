"""Small Python interface for open algorithms; no Gaussian/twin-Q/TD-loss assumptions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Mapping, Protocol, Sequence

import torch
from torch import nn

if TYPE_CHECKING:
    from arlab.algorithms.context import UpdateContext


@dataclass(frozen=True)
class BuildSpec:
    observation_dim: int
    action_dim: int
    total_ticks: int
    device: str
    config: dict[str, Any]
    inference: bool = False


@dataclass(frozen=True)
class Update:
    """One named update phase. An algorithm may choose a different schedule each tick."""
    phase: Literal["value", "policy"]
    name: str = "default"


class Learner(Protocol):
    # Names are unique across groups. Include targets and shared value encoders.
    value_modules: Mapping[str, nn.Module]
    policy_modules: Mapping[str, nn.Module]
    # Every root accepting (observations, actions, ...) to estimate action values.
    action_value_names: Sequence[str]
    optimizers: Mapping[str, torch.optim.Optimizer]
    schedulers: Mapping[str, Any]
    batch_size: int

    def schedule(self, tick: int) -> Sequence[Update]: ...
    def value_step(self, context: UpdateContext, name: str) -> dict[str, float]: ...
    def policy_step(self, context: UpdateContext, name: str) -> dict[str, float]: ...
    def act(self, observations: torch.Tensor) -> torch.Tensor: ...
    def extra_state_dict(self) -> dict[str, Any]: ...
    def load_extra_state_dict(self, state: dict[str, Any]) -> None: ...


class Algorithm:
    """Optional convenience base; subclassing is not required by the loader.

    Implement value_step / policy_step / act and register all modules/optimizers.
    Stateful schedules/EMA controllers belong in extra_state_dict so resume is exact.
    """
    def __init__(self) -> None:
        self.value_modules: dict[str, nn.Module] = {}
        self.policy_modules: dict[str, nn.Module] = {}
        self.action_value_names: list[str] = []
        self.optimizers: dict[str, torch.optim.Optimizer] = {}
        self.schedulers: dict[str, Any] = {}
        self.batch_size = 256

    def schedule(self, tick: int) -> Sequence[Update]:
        return (Update("value"), Update("policy"))

    def extra_state_dict(self) -> dict[str, Any]:
        return {}

    def load_extra_state_dict(self, state: dict[str, Any]) -> None:
        if state:
            raise ValueError("Algorithm did not implement restore for its extra state")
