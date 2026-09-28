"""Serializable source bundles. Validation here NEVER imports candidate code."""
from __future__ import annotations

import ast
import re
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import Field, model_validator

from arlab.core.io import atomic_write, canonical, digest
from arlab.core.schema import StrictModel


class CodeRecipe(StrictModel):
    """An algorithm is source + configuration, not one of a closed set of losses.

    Limits bound artifacts/resources rather than selecting an algorithm family.
    Actual import/execution occurs only in an authorized worker.
    """
    kind: Literal["python"] = "python"
    api_version: Literal[1] = 1
    name: str = Field(min_length=1, max_length=120)
    files: dict[str, str] = Field(min_length=1, max_length=64)
    entrypoint: str = "candidate.py:build"
    config: dict[str, Any] = Field(default_factory=dict)
    # Reservations use this bound; every ctx.optimize call is counted at runtime.
    max_optimizer_steps_per_tick: int = Field(default=8, ge=1, le=256)

    @model_validator(mode="after")
    def validate_bundle(self) -> CodeRecipe:
        if sum(len(s.encode("utf-8")) for s in self.files.values()) > 1_000_000:
            raise ValueError("Candidate source bundle exceeds 1 MB")
        for name, text in self.files.items():
            p = PurePosixPath(name)
            if (p.is_absolute() or ".." in p.parts or "\\" in name or str(p) != name
                    or p.suffix != ".py" or any(not part.isidentifier() for part in p.with_suffix("").parts)):
                raise ValueError(f"Source path must be a relative Python module path: {name}")
            tree = ast.parse(text, filename=name)  # inert syntax check, not an import sandbox
            if sum(1 for _ in ast.walk(tree)) > 100000:
                raise ValueError("Candidate syntax tree exceeds resource limit")
        parts = self.entrypoint.split(":")
        if len(parts) != 2 or parts[0] not in self.files or not re.fullmatch(r"[A-Za-z_]\w*", parts[1]):
            raise ValueError("Entrypoint must be an included source file and a top-level factory")
        if len(canonical(self.config)) > 100000:
            raise ValueError("Candidate configuration exceeds resource limit")
        return self

    @property
    def source_sha256(self) -> str:
        return digest({"files": self.files, "entrypoint": self.entrypoint, "api_version": 1})

    def materialize(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for name, text in self.files.items():
            path = directory / name
            if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
                raise ValueError("Source destination escaped its bundle directory")
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write(path, text)


def is_code_recipe(recipe: dict[str, Any]) -> bool:
    return recipe.get("kind") == "python"



def apply_source_edits(base: dict[str, Any], action: Any) -> dict[str, Any]:
    """Apply a content-bound change to a copied recipe, then fully revalidate it.

    replace requires exactly one occurrence, avoiding fuzzy or stale patches.
    Each successful edit creates a NEW immutable candidate with its own identity.
    """
    import copy
    raw = copy.deepcopy(CodeRecipe.model_validate(base).model_dump(mode="json"))
    raw["name"] = action.name
    for change in action.edits:
        if change.kind == "write":
            raw["files"][change.path] = change.new
        elif change.kind == "delete":
            if change.path not in raw["files"]:
                raise ValueError("Cannot delete a missing source file")
            del raw["files"][change.path]
        else:
            text = raw["files"].get(change.path)
            if text is None or not change.old or text.count(change.old) != 1:
                raise ValueError("Source replacement needs exactly one matching old fragment")
            raw["files"][change.path] = text.replace(change.old, change.new, 1)
    raw["config"].update(action.config_updates)
    for key in action.config_remove:
        raw["config"].pop(key, None)
    if action.entrypoint is not None:
        raw["entrypoint"] = action.entrypoint
    if action.max_optimizer_steps_per_tick is not None:
        raw["max_optimizer_steps_per_tick"] = action.max_optimizer_steps_per_tick
    return CodeRecipe.model_validate(raw).model_dump(mode="json")
