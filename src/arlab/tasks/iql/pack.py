"""The only bridge between the general research runtime and the IQL domain."""
from __future__ import annotations

import copy
from importlib.resources import files
from pathlib import Path
from typing import Any

from arlab.core.schema import Case
from arlab.tasks.iql.data import load_dataset
from arlab.tasks.iql.environments import reference
from arlab.tasks.iql.recipe import IQLRecipe, ess_controller


class IQLPack:
    name = "iql"

    def validate_recipe(self, recipe: dict[str, Any]) -> dict[str, Any]:
        from arlab.algorithms.spec import CodeRecipe, is_code_recipe
        if is_code_recipe(recipe):
            return CodeRecipe.model_validate(recipe).model_dump(mode="json")
        return IQLRecipe.model_validate(recipe).model_dump(mode="json")

    def recipe_schema(self) -> dict[str, Any]:
        from pydantic import TypeAdapter
        from arlab.algorithms.spec import CodeRecipe
        return TypeAdapter(IQLRecipe | CodeRecipe).json_schema()

    def audit_case(self, case: Case) -> dict[str, Any]:
        arrays, metadata = load_dataset(Path(case.data), case.dataset_sha256)
        if metadata["kind"] != case.environment["kind"]:
            raise ValueError("Dataset and evaluation environment families differ")
        if metadata["kind"] == "d4rl":
            env_id = case.environment["id"]
            _, _, obs, actions = reference(env_id)
            if metadata.get("environment") != env_id:
                raise ValueError("Wrong D4RL dataset for evaluation environment")
            if arrays["observations"].shape[1] != obs or arrays["actions"].shape[1] != actions:
                raise ValueError("Wrong D4RL dataset dimensions")
        else:
            if metadata["dynamics_seed"] != case.environment.get("dynamics_seed", 0):
                raise ValueError("Synthetic dataset dynamics differ from evaluation environment")
            if metadata["horizon"] != case.environment.get("horizon", 32):
                raise ValueError("Synthetic dataset horizon differs from evaluation environment")
        return {"transitions": len(arrays["rewards"]), "observation_dim": arrays["observations"].shape[1],
                "action_dim": arrays["actions"].shape[1], "kind": metadata["kind"]}

    def train(self, request: dict[str, Any], output: Path) -> dict[str, Any]:
        from arlab.algorithms.spec import is_code_recipe
        if is_code_recipe(request["recipe"]):
            from arlab.algorithms.runtime import train_code
            return train_code(request, output)
        from arlab.tasks.iql.trainer import train
        return train(request, output)

    def evaluate(self, request: dict[str, Any], output: Path) -> dict[str, Any]:
        from arlab.tasks.iql.evaluator import evaluate_policy
        return evaluate_policy(request, output)

    def teaching_variants(self, baseline: dict[str, Any]) -> list[dict[str, Any]]:
        adaptive = copy.deepcopy(baseline)
        adaptive.update(name="iql-ess-controller", beta_update=ess_controller())
        normalized = copy.deepcopy(baseline)
        normalized.update(name="iql-normalized-weights", normalize_weights=True)
        temperature = copy.deepcopy(baseline)
        temperature.update(name="iql-lower-fixed-temperature", beta=max(0.1, baseline["beta"] * 0.5))
        return [self.validate_recipe(r) for r in (adaptive, normalized, temperature)]

    def template(self, name: str) -> dict[str, Any]:
        from arlab.algorithms.templates import template
        return template(name)

    def code_variants(self, baseline: dict[str, Any]) -> list[dict[str, Any]]:
        from arlab.algorithms.templates import TEMPLATES, template
        tiny = max(baseline.get("hidden_dims", [256])) <= 64
        return [template(name, tiny=tiny) for name in TEMPLATES]

    def skill(self) -> str:
        return files("arlab").joinpath("skills/iql.md").read_text(encoding="utf-8")
