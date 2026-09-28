"""Construct an executable in-sample candidate without adding new network queries."""
from pathlib import Path

from arlab.core.io import write_json
from arlab.tasks.iql.recipe import IQLRecipe, ess_controller


def main() -> None:
    # This is a hypothesis, not a claimed improvement over IQL.
    recipe = IQLRecipe(
        name="iql-ess-hypothesis",
        beta_update=ess_controller(),
        controller_rate=0.02,
        target_ess=0.5,
    )
    write_json(Path("runs/iql-ess-hypothesis.json"), recipe)


if __name__ == "__main__":
    main()
