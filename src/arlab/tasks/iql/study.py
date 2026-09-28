"""Textbook study: open research, three development tasks, nine-task confirmation.

The nine-task selection is our experiment protocol, not an official D4RL split.
Dataset construction and IQL math are unchanged.
"""
from __future__ import annotations

from arlab.core.schema import FinalEvaluation, ResearchContract
from arlab.core.presets import research_budget, adaptive_exploration

from arlab.classroom.constants import D4RL_SUITE9, EXPLORATION_TASKS


def apply_nine_task_protocol(raw: dict) -> ResearchContract:
    """Turn a prepared nine-task contract into the frozen study definition."""
    raw = __import__("copy").deepcopy(raw)
    if {c["name"] for c in raw["cases"]} != set(D4RL_SUITE9) or len(raw["cases"]) != 9:
        raise ValueError("The textbook study requires exactly the nine declared D4RL-v2 tasks")
    for case in raw["cases"]:
        if case["environment"] != {"kind": "d4rl", "id": case["name"]}:
            raise ValueError("A D4RL study cannot silently use synthetic environments")
        case["split"] = "dev" if case["name"] in EXPLORATION_TASKS else "test"
    raw.update(name="iql-nine-task-research", train_seeds=list(range(10)),
               eval_seeds=list(range(1000, 1010)), rungs=[5000, 20000, 50000, 100000, 250000, 1000000],
               max_candidates=128, max_revisions=10000,
               budget=research_budget().model_dump(mode="json"),
               exploration=adaptive_exploration().model_dump(mode="json"))
    raw["goal"] = (
        "Conduct a complete literature- and code-informed autoresearch study using fixed IQL as baseline. "
        f"Develop on {', '.join(EXPLORATION_TASKS)}. "
        "These three fixed tasks cover three environments and medium, medium-replay, medium-expert "
        "data regimes, but environment and data-regime effects are confounded in this exploratory subset. "
        "Start experiments cheaply: one of the three dev tasks, one seed, about 5000 cumulative updates "
        "and 3 evaluation episodes. Partial scored experiments are allowed and exploratory, NOT final conclusions. "
        "Choose based on evidence whether to continue the SAME checkpoint, try another dev environment, add seeds, "
        "increase evaluation episodes, revise source or stop a direction. Read logs and metric curves. "
        "Do not mechanically fill all tasks/seeds/rungs for every new idea. Schedule horizon stays at one million; "
        "run auto-reuses compatible checkpoints; continue_trial targets are cumulative. New environments/seeds/code start fresh. "
        "Only method selection requires complete, matched three-task panels on confirmation seeds 0/1/2. "
        "Compare task-normalized scores at "
        "the same horizon with equal task weight, inspect per-task differences rather than only the "
        "best environment, make falsifiable hypotheses, retain negative results, "
        "and freeze ONE reproducible method (source + configuration) before confirmation. "
        "Keep Q/V learning in-sample, while networks, actor extraction and update schedules are open. "
        "The owner then independently retrains IQL and the frozen method on all NINE declared D4RL tasks "
        "with five new training seeds. Development completion is NOT the final research conclusion. "
        "Use available no-key literature and public-code tools when needed; no general web search."
    )
    raw["final_evaluation"] = FinalEvaluation(cases=list(D4RL_SUITE9)).model_dump(mode="json")
    raw["research_access"]["enabled"] = True
    raw["research_access"]["max_requests"] = 2000
    raw["research_access"]["max_download_bytes"] = 1_000_000_000
    raw["research_access"]["web_search_enabled"] = False
    # Six non-development tasks are not accessible through agent run tools.
    return ResearchContract.model_validate(raw)


def require_nine_task_protocol(contract: ResearchContract) -> None:
    final = contract.final_evaluation
    if not final or set(final.cases) != set(D4RL_SUITE9) or len(final.cases) != 9:
        raise ValueError("This entry requires a nine-task final panel; run start.sh setup in a new root")
    if final.steps < 1_000_000 or len(final.train_seeds) < 5 or len(final.eval_seeds) < 10:
        raise ValueError("The full study cannot be reduced to smoke-test training or a single seed")
    dev = [c.name for c in contract.cases if c.split == "dev"]
    if (set(dev) != set(EXPLORATION_TASKS) or len(dev) != 3
            or {c.name for c in contract.cases} != set(D4RL_SUITE9)):
        raise ValueError("Three-task development must use the fixed mixed-distribution panel "
                         "(one medium, one medium-replay, one medium-expert across three environments); "
                         "preserve old runs and create a new deployment root")
    if len(contract.exploration.confirmation_seeds or contract.train_seeds) < 3 or contract.rungs[-1] < 1_000_000:
        raise ValueError("Full development requires at least three training seeds and a million-update final rung")
    for case in contract.cases:
        if case.environment != {"kind": "d4rl", "id": case.name}:
            raise ValueError("The classroom research path must use real D4RL, not a fixture")
