"""Candidate loading: call only inside explicitly authorized training/policy workers."""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

from arlab.algorithms.protocol import BuildSpec, Learner
from arlab.algorithms.spec import CodeRecipe


def load_algorithm(recipe: CodeRecipe, source: Path, spec: BuildSpec) -> Learner:
    """Python is deliberately open here. This function is NOT a security sandbox."""
    sys.dont_write_bytecode = True
    module_path, factory_name = recipe.entrypoint.split(":")
    for name, text in recipe.files.items():
        path = source / name
        if path.is_symlink() or path.read_text(encoding="utf-8") != text:
            raise ValueError("Materialized source differs from its registered candidate")
    package_name = "arlab_candidate_" + recipe.source_sha256
    package = types.ModuleType(package_name)
    package.__path__ = [str(source)]
    sys.modules[package_name] = package
    module_name = package_name + "." + module_path[:-3].replace("/", ".")
    module_spec = importlib.util.spec_from_file_location(module_name, source / module_path)
    if module_spec is None or module_spec.loader is None:
        raise ValueError("Cannot construct candidate module")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    module_spec.loader.exec_module(module)
    learner = getattr(module, factory_name)(spec)
    for name in ("schedule", "value_step", "policy_step", "act", "extra_state_dict", "load_extra_state_dict"):
        if not callable(getattr(learner, name, None)):
            raise ValueError(f"Candidate lacks required method {name}")
    if not isinstance(learner.batch_size, int) or not 1 <= learner.batch_size <= 1_000_000:
        raise ValueError("Invalid candidate batch size")
    return learner


def encode_state(value: Any, tensors: dict[str, Any], prefix: str = "extra") -> Any:
    """Tensor + JSON serialization, preserving tuples; never arbitrary pickle."""
    import torch
    if isinstance(value, torch.Tensor):
        key = f"{prefix}.{len(tensors)}"
        tensors[key] = value.detach().cpu().contiguous().clone()
        return ["tensor", key]
    if value is None or isinstance(value, (str, int, float, bool)):
        return ["scalar", value]
    if isinstance(value, dict):
        if any(not isinstance(k, str) for k in value):
            raise ValueError("Extra-state dictionaries must have string keys")
        return ["dict", {k: encode_state(v, tensors, prefix) for k, v in value.items()}]
    if isinstance(value, (tuple, list)):
        return ["tuple" if isinstance(value, tuple) else "list", [encode_state(v, tensors, prefix) for v in value]]
    raise ValueError(f"Unsupported extra-state type: {type(value).__name__}")


def decode_state(node: Any, tensors: dict[str, Any]) -> Any:
    tag, payload = node
    if tag == "tensor":
        return tensors[payload]
    if tag == "scalar":
        return payload
    if tag == "dict":
        return {k: decode_state(v, tensors) for k, v in payload.items()}
    if tag in ("list", "tuple"):
        seq = [decode_state(v, tensors) for v in payload]
        return tuple(seq) if tag == "tuple" else seq
    raise ValueError("Unknown extra-state encoding")
