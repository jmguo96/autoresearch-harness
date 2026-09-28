"""A small tensor/scalar expression language; never Python eval or executable plugins.

Expressions can only transform supplied values. In the IQL pack no expression has
an actor, critic, dataset, environment, file handle, or network capability.
This makes the in-sample boundary structural instead of a prompt convention.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

ARITY = {"add": 2, "sub": 2, "mul": 2, "div": 2, "min": 2, "max": 2,
         "lt": 2, "gt": 2, "where": 3, "clip": 3, "neg": 1, "abs": 1,
         "square": 1, "sqrt": 1, "exp": 1, "log": 1, "mean": 1}


def validate_expr(expr: Any, variables: set[str], *, max_nodes: int = 96) -> None:
    count = 0
    def visit(node: Any, depth: int) -> None:
        nonlocal count
        count += 1
        if depth > 12 or count > max_nodes:
            raise ValueError("Expression exceeds complexity budget")
        if not isinstance(node, dict):
            raise ValueError("Expression nodes must be objects")
        if set(node) == {"const"}:
            value = node["const"]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("Constant must be numeric")
            if not math.isfinite(value) or abs(value) > 1e6:
                raise ValueError("Constant is not finite or exceeds the numeric bound")
        elif set(node) == {"var"}:
            if not isinstance(node["var"], str) or node["var"] not in variables:
                raise ValueError(f"Unknown variable: {node['var']!r}")
        elif set(node) == {"op", "args"}:
            op, args = node["op"], node["args"]
            if not isinstance(op, str) or op not in ARITY or not isinstance(args, list):
                raise ValueError("Unknown operation")
            if len(args) != ARITY[op]:
                raise ValueError(f"Incorrect arity for {op}")
            for arg in args:
                visit(arg, depth + 1)
        else:
            raise ValueError("Invalid expression fields")
    visit(expr, 0)


def constant(value: float) -> dict[str, Any]:
    return {"const": value}


def variable(name: str) -> dict[str, Any]:
    return {"var": name}


def operation(name: str, *args: dict[str, Any]) -> dict[str, Any]:
    return {"op": name, "args": list(args)}


def evaluate(expr: dict[str, Any], values: dict[str, Any]) -> Any:
    """Evaluate a previously validated AST using NumPy scalars or PyTorch tensors."""
    if "const" in expr:
        return float(expr["const"])
    if "var" in expr:
        return values[expr["var"]]
    args = [evaluate(arg, values) for arg in expr["args"]]
    op = expr["op"]
    # Torch is imported lazily: the generic harness has no torch import requirement.
    is_tensor = any(type(x).__module__.startswith("torch") for x in args)
    if is_tensor:
        import torch
        anchor = next(x for x in args if isinstance(x, torch.Tensor))
        args = [x if isinstance(x, torch.Tensor) else torch.as_tensor(x, device=anchor.device,
                dtype=anchor.dtype if anchor.dtype != torch.bool else torch.float32) for x in args]
        lib = torch
    else:
        lib = np
    a = args[0]
    if op == "add": return a + args[1]
    if op == "sub": return a - args[1]
    if op == "mul": return a * args[1]
    if op == "div": return a / args[1]
    if op == "neg": return -a
    if op == "abs": return lib.abs(a)
    if op == "square": return a * a
    if op == "sqrt": return lib.sqrt(a)
    if op == "exp": return lib.exp(a)
    if op == "log": return lib.log(a)
    if op == "mean": return lib.mean(a) if is_tensor else np.mean(a)
    if op == "lt": return a < args[1]
    if op == "gt": return a > args[1]
    if op == "min": return lib.minimum(a, args[1])
    if op == "max": return lib.maximum(a, args[1])
    if op == "where": return lib.where(a, args[1], args[2])
    if op == "clip": return lib.minimum(lib.maximum(a, args[1]), args[2])
    raise ValueError(f"Unknown operation: {op}")


def scalar(expr: dict[str, Any], values: dict[str, float]) -> float:
    with np.errstate(all="raise"):
        value = float(evaluate(expr, values))
    if not math.isfinite(value):
        raise ValueError("Non-finite program output")
    return value
