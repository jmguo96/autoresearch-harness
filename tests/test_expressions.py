from __future__ import annotations

import copy

import pytest
import torch

from arlab.core.expressions import constant as c
from arlab.core.expressions import evaluate, operation as op, scalar, validate_expr, variable as v
from arlab.tasks.iql.recipe import IQLRecipe, ess_controller


@pytest.mark.parametrize("node", [
    {"var": "critic"}, {"var": "sample_action"}, {"op": "__import__", "args": []},
    {"const": float("nan")}, {"const": float("inf")}, {"const": True},
    {"const": 1e9}, {"op": "add", "args": [{"const": 1}]},
    {"var": "adv", "python": "open('/etc/passwd')"}, "__import__('os')", 3,
])
def test_rejects_non_programs(node):
    with pytest.raises(ValueError):
        validate_expr(node, {"adv"})


def test_complexity_is_bounded():
    expr = v("x")
    for _ in range(14):
        expr = op("square", expr)
    with pytest.raises(ValueError, match="complexity"):
        validate_expr(expr, {"x"})


def test_iql_formulas_match_equations():
    recipe = IQLRecipe()
    delta = torch.tensor([-2.0, -0.2, 0, 1.0, 3.0], requires_grad=True)
    values = evaluate(recipe.value_loss, {"delta": delta, "expectile": 0.7})
    expected = torch.where(delta < 0, 0.3, 0.7) * delta.square()
    torch.testing.assert_close(values, expected)
    values.mean().backward()
    assert delta.grad is not None and torch.isfinite(delta.grad).all()
    q = evaluate(recipe.q_loss, {"td1": delta.detach(), "td2": delta.detach() * 2})
    torch.testing.assert_close(q, 2.5 * delta.detach().square())


def test_exponential_does_not_overflow_at_extreme_advantage():
    result = evaluate(IQLRecipe().actor_weight, {"adv": torch.tensor([-1e9, 0, 1e9]), "beta": 3, "weight_cap": 100})
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result[1:], torch.tensor([1.0, 100.0]))


def test_ess_controller_reduces_temperature_when_concentrated():
    lower = scalar(ess_controller(), {"beta": 3, "ess_ratio": 0.1, "target_ess": 0.5, "controller_rate": 0.02})
    assert lower < 3


@pytest.mark.parametrize("key,value", [("expectile", 1), ("expectile", 0), ("beta", -1),
    ("hidden_dims", [1]), ("hidden_dims", [2048]), ("batch_size", 0), ("allow_ood", True),
    ("value_loss", {"var": "policy_action"})])
def test_recipe_contract_rejects_out_of_bounds(key, value):
    raw = IQLRecipe().model_dump(mode="json")
    raw[key] = value
    with pytest.raises(ValueError):
        IQLRecipe.model_validate(raw)


@pytest.mark.parametrize("name,args,expected", [
    ("add", [2, 3], 5), ("sub", [2, 3], -1), ("mul", [2, 3], 6),
    ("div", [6, 3], 2), ("min", [2, 3], 2), ("max", [2, 3], 3),
    ("neg", [2], -2), ("sqrt", [4], 2), ("abs", [-4], 4),
    ("where", [1, 2, 3], 2), ("clip", [10, 0, 2], 2),
])
def test_scalar_operations(name, args, expected):
    expr = op(name, *(c(x) for x in args))
    validate_expr(expr, set())
    assert scalar(expr, {}) == expected
