"""Explicit deterministic lesson policy, not an LLM or an autonomous discovery claim.

The real model uses the same action schema. Candidate code is fetched from trusted
installed templates only for this lesson; it is never silently substituted for a
failed model or a failed network search.
"""
from __future__ import annotations
from typing import Any
from pydantic import BaseModel
from arlab.core.models import ModelReply
from arlab.core.io import digest


class TeachingModel:
    def __init__(self, task=None, seed_recipe=None, open_code=False):
        self.task, self.seed_recipe, self.open_code = task, seed_recipe, open_code

    def complete(self, system: str, context: dict[str, Any], schema: type[BaseModel]) -> ModelReply:
        del system, schema
        return ModelReply(self._research(context))

    def _research(self, context: dict[str, Any]) -> dict[str, Any]:
        t = context["teaching"]
        counts, baseline = t["action_counts"], context["baseline"]
        def decision(action, summary):
            return {"summary": summary, "action": action}
        if not counts.get("plan"):
            return decision({"kind": "plan", "reason": "Measure a fixed baseline, then compare candidates.",
                "items": [{"id": "baseline", "objective": "Measure fixed IQL", "depends_on": []},
                          {"id": "compare", "objective": "Run matched candidate experiments", "depends_on": ["baseline"]}]},
                "Create a research plan")
        if not counts.get("inspect"):
            return decision({"kind": "inspect", "target": "skill", "id": context["task"]}, "Load domain guidance")
        if not counts.get("recall"):
            return decision({"kind": "recall", "query": context["goal"][:1000]}, "Read relevant scoped notes")
        rungs = context["protocol"]["rungs"]
        def next_run(ids, steps):
            for cid in ids:
                for row in t["next_runs"]:
                    if row["candidate"] == cid and row["steps"] == steps:
                        return {"kind": "run", **{k: row[k] for k in ("candidate", "case", "train_seed", "steps")}}
            return None
        run = next_run([baseline], rungs[0])
        if run:
            return decision(run, "Obtain real baseline evidence")
        if context["candidate_count"] < context["protocol"]["max_candidates"]:
            variants = (self.task.code_variants(self.seed_recipe) if self.open_code
                        else self.task.teaching_variants(self.seed_recipe))
            order = [i for i in t["proposal_order"] if 0 <= i < len(variants)]
            order += [i for i in range(len(variants)) if i not in order]
            for i in order:
                if digest(variants[i]) not in t["recipe_hashes"]:
                    return decision({"kind": "propose", "parent": baseline, "recipe": variants[i],
                        "hypothesis": {"question": "Does this teaching candidate improve the matched result?",
                            "prediction": "Unknown until tested", "rationale": "Explicitly scripted teaching candidate",
                            "falsification": "No replicated gain or candidate violates the protocol", "evidence_ids": []}},
                        "Register an editable candidate; this is not a model discovery")
        run = next_run(t["candidate_ids"], rungs[0])
        if run:
            return decision(run, "Screen on matched cases and seeds")
        for i, steps in enumerate(rungs[1:], 1):
            # Each rung uses the preceding rung's evidence, not a forever-fixed
            # shortlist from the lowest fidelity. Baseline always stays in scope.
            ranked = t["rankings_by_rung"].get(str(rungs[i - 1]), [])
            promoted = [baseline] + [r["candidate"] for r in ranked if r["candidate"] != baseline][:t["promotion_count"]]
            run = next_run(promoted, steps)
            if run:
                return decision(run, "Use current comparable evidence for the next fidelity")
        if not counts.get("remember") and t.get("last_trial"):
            return decision({"kind": "remember", "category": "procedure", "evidence_ids": [t["last_trial"]],
                             "text": "These short teaching trials demonstrate execution; they do not establish D4RL performance."},
                            "Save a conditional observation")
        finalists = context["eligible_finalists"]
        selected = finalists[0]["candidate"] if finalists else baseline
        return decision({"kind": "finish", "candidate": selected,
            "conclusion": "Selected from visible matched development trials; independent evaluation is still required."},
            "Freeze the method, not a claim of scientific superiority")
