"""Measure serialized context size on explicit ledger fixtures, never RL scores."""
from __future__ import annotations
import argparse
import copy
import tempfile
from pathlib import Path
from arlab.cli import demo_contract
from arlab.core.harness import Harness
from arlab.core.io import canonical, digest, write_json
from arlab.core.models import schema_prompt
from arlab.core.schema import Decision, ResearchContract


def measure(base: Path, count: int) -> dict:
    raw = demo_contract(base, steps=2, candidates=count).model_dump(mode="json")
    raw["model"]["provider"] = "compatible"
    raw["cases"] = [{**raw["cases"][0], "name": f"task-{i}"} for i in range(6)]
    raw["train_seeds"], raw["rungs"] = [0, 1, 2], [2, 4, 8]
    agent = Harness(ResearchContract.model_validate(raw), base / "campaign")
    try:
        agent._initialize()
        ids = [agent.store.get("baseline")]
        for i in range(count - 1):
            recipe = copy.deepcopy(raw["seed_recipe"])
            recipe["name"] = f"fixture-{i}"
            ids.append(agent.store.add_candidate(recipe, {"question": "Synthetic ledger-size fixture"}, ids[0]))
        for cid in ids:
            for case in raw["cases"]:
                for seed in raw["train_seeds"]:
                    for steps in raw["rungs"]:
                        tid = "fixture-" + digest([cid, case["name"], seed, steps])[:24]
                        row = {"trial_id": tid, "candidate": cid, "case": case["name"],
                               "train_seed": seed, "steps": steps, "score": 1.0,
                               "metric": "NOT_D4RL_fixture", "returns": [1.0] * 10,
                               "purpose": "experiment", "diagnostics": {"fixture": "do_not_replay" * 100}}
                        agent.store.save_trial(tid, cid, "complete", row)
        context = agent._context()
        rows = agent.store.trials()
        old_trials = [{k: v for k, v in row.items() if k not in ("returns", "attempt")} for row in rows]
        return {"candidate_count": count, "trial_rows": len(rows), "cases": 6, "seeds": 3, "rungs": 3,
                "state_context_characters": len(canonical(context)),
                "old_style_trial_projection_characters": len(canonical(old_trials)),
                "old_comparison_scope": "trial-only projection, NOT a reconstructed full v0.3 prompt",
                "state_limit": agent.config.context_characters,
                "instruction_and_action_schema_characters": len(agent.config.instruction + schema_prompt(Decision)),
                "total_model_text_characters": len(canonical(context)) + len(agent.config.instruction + schema_prompt(Decision)),
                "context_control": context["context_control"],
                "note": "Character counts, not tokens, performance gains or real experiments."}
    finally:
        agent.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="arlab-context-measure-") as tmp:
        rows = [measure(Path(tmp) / str(count), count) for count in (4, 20)]
    write_json(args.output, {"scope": "deterministic synthetic ledger fixture", "measurements": rows})
    print(canonical(rows))


if __name__ == "__main__":
    main()
