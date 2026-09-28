"""Explicit partial evidence for exploration; full matched panels for freezing.

Multiple checkpoints of a seed, and repeated evaluations of the same checkpoint,
are not independent seeds. Only a single matching episode protocol per cell is
used. Arbitrary exploratory subsets are named as such and cannot finish research.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from arlab.core.schema import ResearchContract
from arlab.core.stats import paired_comparison


class PanelIndex:
    def __init__(self, contract: ResearchContract, records: list[dict[str, Any]]):
        self.contract = contract
        self.cases = [c.name for c in contract.cases if c.split == "dev"]
        self.seeds = contract.exploration.confirmation_seeds or contract.train_seeds
        self.required = {(case, seed) for case in self.cases for seed in self.seeds}
        self.records = [r for r in records if r.get("purpose", "experiment") == "experiment"]
        self.groups: dict[tuple[str, int], list[dict]] = {}
        for row in self.records:
            self.groups.setdefault((row["candidate"], row["steps"]), []).append(row)

    def legal_steps(self, steps: int) -> bool:
        policy = self.contract.exploration
        return (policy.minimum_steps <= steps <= self.contract.rungs[-1]
                if policy.adaptive else steps in self.contract.rungs)

    def episodes(self, row: dict) -> int:
        # Compatibility for pre-existing fixture rows without episode metadata.
        return len(row["eval_seeds"]) if row.get("eval_seeds") else len(self.contract.eval_seeds)

    def completed(self, candidate: str, steps: int, *, cases=None, seeds=None, episodes=None) -> list[dict]:
        cases = self.cases if cases is None else cases
        seeds = self.seeds if seeds is None else seeds
        episodes = len(self.contract.eval_seeds) if episodes is None else episodes
        by_cell = {}
        for row in self.groups.get((candidate, steps), []):
            if (row["status"] != "complete" or row["case"] not in cases or row["train_seed"] not in seeds
                    or self.episodes(row) != episodes):
                continue
            if row.get("eval_seeds") and row["eval_seeds"] != self.contract.eval_seeds[:episodes]:
                continue
            # In adaptive mode every scored checkpoint must belong to the common schedule.
            if (self.contract.exploration.adaptive and row.get("schedule_steps") is not None
                    and row["schedule_steps"] != self.contract.rungs[-1]):
                continue
            key = row["case"], row["train_seed"]
            if key in by_cell:
                raise ValueError("Duplicate independent training cell; do not inflate seed count")
            by_cell[key] = row
        return list(by_cell.values())

    def missing(self, candidate: str, steps: int) -> list[dict]:
        complete = {(r["case"], r["train_seed"]) for r in self.completed(candidate, steps)}
        rows = self.groups.get((candidate, steps), [])
        statuses = {(r["case"], r["train_seed"]): r["status"] for r in rows
                    if self.episodes(r) == len(self.contract.eval_seeds)}
        return [{"candidate": candidate, "case": case, "train_seed": seed, "steps": steps,
                 "status": statuses.get((case, seed), "not_run"),
                 "eval_episodes": len(self.contract.eval_seeds)}
                for case, seed in sorted(self.required) if (case, seed) not in complete]

    def panel(self, candidate: str, steps: int) -> dict:
        rows = self.groups.get((candidate, steps), [])
        done = self.completed(candidate, steps)
        actual = {(r["case"], r["train_seed"]) for r in done}
        complete = actual == self.required and len(done) == len(self.required)
        scores = [float(r["score"]) for r in done]
        # Partial experiments remain visible without becoming a misleading global score.
        observations = [r for r in rows if r["status"] == "complete"]
        return {"candidate": candidate, "steps": steps,
                "complete": complete, "completed": len(actual & self.required),
                "expected": len(self.required), "missing_count": len(self.required - actual),
                "failed": sum(r["status"] == "failed" for r in rows),
                "observed_trials": len(observations),
                "observed_cases": sorted({r["case"] for r in observations}),
                "observed_seeds": sorted({r["train_seed"] for r in observations}),
                "mean": float(np.mean(scores)) if scores else None,
                "score_scope": "complete_panel" if complete else "partial_not_rankable",
                "train_seconds": sum(r.get("train_seconds", 0) for r in done),
                "optimizer_steps": sum(r.get("optimizer_steps", 0) for r in done)}

    def compare(self, baseline: str, candidate: str, steps: int, *, cases=None, seeds=None, episodes=None) -> dict:
        if not self.legal_steps(steps):
            raise PermissionError("Comparison horizon is outside the research contract")
        chosen_cases = self.cases if cases is None else cases
        chosen_seeds = self.seeds if seeds is None else seeds
        episodes = len(self.contract.eval_seeds) if episodes is None else episodes
        if (not chosen_cases or not set(chosen_cases).issubset(self.cases) or
                not chosen_seeds or not set(chosen_seeds).issubset(self.contract.train_seeds) or
                not 1 <= episodes <= len(self.contract.eval_seeds)):
            raise PermissionError("Comparison subset is outside the approved development pool")
        if len(set(chosen_cases)) != len(chosen_cases) or len(set(chosen_seeds)) != len(chosen_seeds):
            raise ValueError("Comparison cases/seeds must be unique")
        expected = {(c, s, steps) for c in chosen_cases for s in chosen_seeds}
        result = paired_comparison(
            self.completed(baseline, steps, cases=chosen_cases, seeds=chosen_seeds, episodes=episodes),
            self.completed(candidate, steps, cases=chosen_cases, seeds=chosen_seeds, episodes=episodes),
            expected_cells=expected)
        full = (set(chosen_cases) == set(self.cases) and set(chosen_seeds) == set(self.seeds)
                and episodes == len(self.contract.eval_seeds))
        result["selected_panel_complete"] = result["contract_panel_complete"]
        result["contract_panel_complete"] = bool(full and result["contract_panel_complete"])
        result["complete"] = result["contract_panel_complete"]
        result["scope"] = "full_development_panel" if full else "exploratory_subset_NOT_final_evidence"
        result["selection"] = {"cases": chosen_cases, "train_seeds": chosen_seeds,
                               "steps": steps, "eval_episodes": episodes}
        result["freeze_eligible"] = bool(result["contract_panel_complete"] and steps == self.contract.rungs[-1])
        return result

    def ranking(self, candidates: list[str], steps: int) -> list[dict]:
        rows = [self.panel(cid, steps) for cid in candidates]
        return sorted([r for r in rows if r["complete"]], key=lambda r: (-r["mean"], r["candidate"]))

    def latest_comparable_rung(self, candidates: list[str], baseline: str) -> int:
        endpoints = sorted(set(self.contract.rungs) | {r["steps"] for r in self.records}, reverse=True)
        for steps in endpoints:
            complete = {r["candidate"] for r in self.ranking(candidates, steps)}
            if baseline in complete and (len(complete) >= 2 or len(candidates) == 1):
                return steps
        return self.contract.rungs[0]

    def totals(self) -> dict:
        return dict(Counter(r["status"] for r in self.records))
