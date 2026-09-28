"""Executable teaching trace: partial experiment -> continuation -> breadth -> freeze.

The controller is explicitly scripted, not a real LLM discovery. Training and
independent rollout are real, on small synthetic tasks, NOT D4RL results. Real
students run start.sh with their model for the full protocol.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from arlab.audit import audit_campaign
from arlab.classroom.reporting import study_report
from arlab.core.harness import Harness
from arlab.core.io import file_digest, write_json
from arlab.core.models import ModelReply
from arlab.core.schema import Budget, Case, ExplorationPolicy, FinalEvaluation, ResearchContract
from arlab.tasks.iql.environments import make_fixture
from arlab.tasks.iql.recipe import IQLRecipe


class DemonstrationController:
    """A reproducible action trace used to inspect the actual tool loop."""
    def __init__(self):
        self.index = 0
        self.trials = {}
        self.actions = []

    def complete(self, system, context, schema):
        del system, schema
        baseline = context["baseline"]
        candidates = [c["id"] for c in context["candidates"] if c["id"] != baseline]
        candidate = candidates[0] if candidates else baseline
        details = (context.get("last_feedback") or {}).get("details", {})
        if details.get("trial_id"):
            key = details["candidate"], details["case"], details["train_seed"], details["steps"]
            self.trials[key] = details["trial_id"]
        def run(cid, case=0, seed=0, steps=4, episodes=1):
            return {"kind": "run", "candidate": cid, "case": f"synthetic-{case}",
                    "train_seed": seed, "steps": steps, "eval_episodes": episodes}
        def extend(cid):
            return {"kind": "continue_trial", "trial": self.trials[(cid, "synthetic-0", 0, 4)],
                    "steps": 8, "eval_episodes": 1}
        sequence = [
            lambda: {"kind": "plan", "reason": "Demonstrate adaptive tool semantics without claiming research gains",
                     "items": [{"id": "explore", "objective": "Cheap probes, read feedback, extend, expand", "depends_on": []}]},
            lambda: {"kind": "propose", "parent": baseline,
                     "recipe": IQLRecipe(name="teaching-temperature-variant", hidden_dims=[16], batch_size=16, beta=1.0).model_dump(mode="json"),
                     "hypothesis": {"question": "Does this temperature help?", "prediction": "Unknown",
                         "rationale": "Scripted method for protocol demonstration", "falsification": "No matched improvement", "evidence_ids": []}},
            lambda: run(baseline), lambda: run(candidate),
            lambda: {"kind": "compare", "candidate": candidate, "steps": 4,
                     "cases": ["synthetic-0"], "train_seeds": [0], "eval_episodes": 1},
            lambda: {"kind": "read_trial_log", "trial": self.trials[(candidate, "synthetic-0", 0, 4)],
                     "stream": "metrics", "fields": ["critic_loss", "actor_loss"]},
            lambda: extend(baseline), lambda: extend(candidate),
            lambda: run(baseline, case=1), lambda: run(candidate, case=1),
            lambda: run(baseline, seed=1), lambda: run(candidate, seed=1),
            lambda: {"kind": "compare", "candidate": candidate, "steps": 4,
                     "cases": ["synthetic-0", "synthetic-1"], "train_seeds": [0], "eval_episodes": 1},
        ]
        # This trace conservatively retains IQL. Not a test-set-driven choice.
        for case in range(3):
            for seed in range(3):
                sequence.append(lambda c=case, s=seed: run(baseline, c, s, 12, 3))
        sequence.append(lambda: {"kind": "finish", "candidate": baseline,
            "conclusion": "Teaching trace retained the fixed baseline; tiny synthetic results establish no algorithm improvement."})
        action = sequence[self.index]()
        self.index += 1
        self.actions.append(action)
        print(f"[scripted teaching action {self.index}] {action['kind']}", flush=True)
        return ModelReply({"summary": "Explicit teaching trace; the operation and observed result are real.", "action": action})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    if (root / "development/state.sqlite3").exists():
        raise ValueError("Use a new output directory for this scripted demonstration")
    cases = []
    for i in range(9):
        data = root / "data" / f"synthetic-{i}.npz"
        sha = make_fixture(data, dynamics_seed=i, episodes=4, horizon=4)
        cases.append(Case(name=f"synthetic-{i}", split="dev" if i < 3 else "test", data=str(data),
            dataset_sha256=sha, environment={"kind": "synthetic_lq", "dynamics_seed": i, "horizon": 4}))
    contract = ResearchContract(name="adaptive-teaching-trace", goal="Observe a real adaptive experiment protocol; not benchmark performance",
        cases=cases, seed_recipe=IQLRecipe(hidden_dims=[16], batch_size=16).model_dump(mode="json"),
        train_seeds=[0, 1, 2, 3], eval_seeds=[1000, 1001, 1002], rungs=[4, 8, 12],
        exploration=ExplorationPolicy(adaptive=True, minimum_steps=1, initial_steps=4,
            initial_eval_episodes=1, confirmation_seeds=[0, 1, 2]),
        budget=Budget(max_decisions=100, max_trials=100, max_train_steps=10000,
                      max_optimizer_steps=100000, max_eval_episodes=1000, max_wall_seconds=3600),
        final_evaluation=FinalEvaluation(cases=[c.name for c in cases], train_seeds=[41],
            eval_seeds=[9000], steps=12))
    model = DemonstrationController()
    agent = Harness(contract, root / "development", model=model)
    try:
        development = agent.run()
    finally:
        agent.close()
    write_json(root / "scripted-action-trace.json", model.actions)
    if development["status"] != "completed":
        raise RuntimeError(f"Teaching development stopped: {development['status']}")
    final = audit_campaign(root / "development", root / "final", split="final")
    report = study_report(root, contract)
    summary = {"controller": "scripted_NOT_LLM", "task": "synthetic_NOT_D4RL",
               "development_status": development["status"], "action_count": len(model.actions),
               "development_reserved": development["reserved"], "development_actual": development["actual_known_usage"],
               "final_status": final["status"], "final_trials": len(final["trials"]),
               "report_scope": report["scope"], "algorithm_improvement_proven": False}
    write_json(root / "validation-summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
