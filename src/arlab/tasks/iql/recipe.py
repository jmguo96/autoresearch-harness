"""Editable in-sample research objects: typed parameters and pure tensor formulas."""
from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator

from arlab.core.expressions import constant as c
from arlab.core.expressions import operation as op
from arlab.core.expressions import validate_expr
from arlab.core.expressions import variable as v
from arlab.core.schema import StrictModel


def value_formula() -> dict[str, Any]:
    return op("mul", op("where", op("lt", v("delta"), c(0)), op("sub", c(1), v("expectile")),
                         v("expectile")), op("square", v("delta")))


def q_formula() -> dict[str, Any]:
    return op("mul", c(0.5), op("add", op("square", v("td1")), op("square", v("td2"))))


def weight_formula() -> dict[str, Any]:
    # Clip the exponent BEFORE exp to avoid overflow; -60 is a numerical floor.
    return op("exp", op("clip", op("mul", v("beta"), v("adv")), c(-60), op("log", v("weight_cap"))))


def ess_controller() -> dict[str, Any]:
    # ESS below target -> lower inverse temperature -> less concentrated weights.
    return op("mul", v("beta"), op("exp", op("mul", v("controller_rate"),
              op("sub", v("ess_ratio"), v("target_ess")))))


class IQLRecipe(StrictModel):
    name: str = Field(default="iql", min_length=1, max_length=120)
    expectile: float = Field(default=0.7, gt=0, lt=1)
    beta: float = Field(default=3.0, ge=0.01, le=50)
    discount: float = Field(default=0.99, ge=0, le=1)
    target_rate: float = Field(default=0.005, gt=0, le=1)
    actor_lr: float = Field(default=3e-4, ge=1e-6, le=0.01)
    critic_lr: float = Field(default=3e-4, ge=1e-6, le=0.01)
    value_lr: float = Field(default=3e-4, ge=1e-6, le=0.01)
    hidden_dims: list[int] = Field(default_factory=lambda: [256, 256], min_length=1, max_length=4)
    batch_size: int = Field(default=256, ge=8, le=4096)
    weight_cap: float = Field(default=100, ge=1, le=1000)
    normalize_weights: bool = False
    gradient_clip: float = Field(default=0, ge=0, le=100)
    value_loss: dict[str, Any] = Field(default_factory=value_formula)
    q_loss: dict[str, Any] = Field(default_factory=q_formula)
    actor_weight: dict[str, Any] = Field(default_factory=weight_formula)
    beta_update: dict[str, Any] = Field(default_factory=lambda: v("beta"))
    controller_rate: float = Field(default=0.02, ge=0, le=0.1)
    target_ess: float = Field(default=0.5, gt=0, le=1)
    beta_min: float = Field(default=0.05, gt=0, le=50)
    beta_max: float = Field(default=20, gt=0, le=50)

    @model_validator(mode="after")
    def validate_program(self) -> IQLRecipe:
        if any(isinstance(n, bool) or n < 8 or n > 1024 for n in self.hidden_dims):
            raise ValueError("Hidden widths must be in [8,1024]")
        if self.beta_min > self.beta_max or not self.beta_min <= self.beta <= self.beta_max:
            raise ValueError("beta must lie inside controller bounds")
        validate_expr(self.value_loss, {"delta", "expectile"})
        validate_expr(self.q_loss, {"td1", "td2"})
        validate_expr(self.actor_weight, {"adv", "beta", "weight_cap"})
        validate_expr(self.beta_update, {"beta", "ess_ratio", "controller_rate", "target_ess"})
        return self
