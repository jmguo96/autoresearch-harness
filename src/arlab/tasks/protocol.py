"""A task pack contains domain knowledge; the generic harness only calls this contract."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from arlab.core.schema import Case


class TaskPack(Protocol):
    name: str
    def validate_recipe(self, recipe: dict[str, Any]) -> dict[str, Any]: ...
    def recipe_schema(self) -> dict[str, Any]: ...
    def audit_case(self, case: Case) -> dict[str, Any]: ...
    def train(self, request: dict[str, Any], output: Path) -> dict[str, Any]: ...
    def evaluate(self, request: dict[str, Any], output: Path) -> dict[str, Any]: ...
    def teaching_variants(self, baseline: dict[str, Any]) -> list[dict[str, Any]]: ...
    def skill(self) -> str: ...


def get_task(name: str) -> TaskPack:
    if name == "iql":
        from arlab.tasks.iql.pack import IQLPack
        return IQLPack()
    # Extensions are explicit trusted installations, not model-supplied import paths.
    from importlib.metadata import entry_points
    matches = list(entry_points(group="arlab.tasks", name=name))
    if len(matches) != 1:
        raise ValueError(f"Unknown or ambiguous task pack: {name}")
    return matches[0].load()()
