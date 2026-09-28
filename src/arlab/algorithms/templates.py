"""Copyable code starting points, not a closed algorithm registry."""
from __future__ import annotations

from importlib.resources import files
from typing import Any

from arlab.algorithms.spec import CodeRecipe

TEMPLATES = ("ensemble-mixture", "diffusion-extraction")


def template(name: str, *, tiny: bool = False) -> dict[str, Any]:
    if name not in TEMPLATES:
        raise ValueError(f"Unknown template; choose from {TEMPLATES}")
    directory = files("arlab").joinpath("templates")
    sources = {p: directory.joinpath(p).read_text(encoding="utf-8") for p in ("candidate.py", "networks.py")}
    config = {"hidden_dims": [32, 32] if tiny else [256, 256],
              "batch_size": 32 if tiny else 256, "ensemble": 3,
              "actor": "mixture" if name == "ensemble-mixture" else "diffusion",
              "mixtures": 3, "diffusion_steps": 4 if tiny else 12,
              "n_step": 2, "policy_every": 2 if name == "ensemble-mixture" else 1,
              "action_candidates": 4 if tiny else 16,
              "actor_q_guidance": 0.02 if name == "diffusion-extraction" else 0.0}
    return CodeRecipe(name=name, files=sources, config=config,
                      max_optimizer_steps_per_tick=3).model_dump(mode="json")
