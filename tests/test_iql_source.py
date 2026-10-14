from pathlib import Path

import torch

from arlab.cli import demo_contract
from arlab.algorithms.templates import template
from arlab.algorithms.spec import CodeRecipe
from arlab.algorithms.loading import load_algorithm
from arlab.algorithms.protocol import BuildSpec
from arlab.algorithms.context import UpdateContext
from arlab.algorithms.guard import ValueBoundary
from arlab.core.harness import Harness
from arlab.core.schema import Decision, Hypothesis, ResearchContract
from arlab.core.reading import inspect_artifact
from arlab.core.schema import InspectAction
from arlab.tasks.iql.trainer import IQLTrainer
from arlab.tasks.iql.recipe import IQLRecipe
from arlab.tasks.iql.data import OfflineReplay, load_dataset


def test_editable_iql_preserves_baseline_one_update(tmp_path):
    contract = demo_contract(tmp_path, steps=3)
    recipe = CodeRecipe.model_validate(template("iql-source", tiny=True))
    cfg = IQLRecipe.model_validate(recipe.config)
    arrays, metadata = load_dataset(Path(contract.cases[0].data))
    obs, act = arrays["observations"].shape[1], arrays["actions"].shape[1]
    torch.manual_seed(7)
    baseline = IQLTrainer(obs, act, cfg, 3)
    torch.manual_seed(7)
    recipe.materialize(tmp_path / "source")
    learner = load_algorithm(recipe, tmp_path / "source", BuildSpec(obs, act, 3, "cpu", recipe.config))
    replay = OfflineReplay(arrays, metadata, 7)
    baseline.update(OfflineReplay(arrays, metadata, 7).sample(cfg.batch_size, "cpu"))
    guard = ValueBoundary(learner)
    guard.tick_remaining = 3
    try:
        for phase in ("value", "policy"):
            with guard.enter(phase):
                ctx = UpdateContext(learner, guard, replay, tick=0, device="cpu", remaining_optimizations=3)
                getattr(learner, phase + "_step")(ctx, "default")
        for expected, actual in ((baseline.actor, learner.policy_modules["actor"]),
                                 (baseline.critic, learner.value_modules["q"]),
                                 (baseline.target, learner.value_modules["target_q"]),
                                 (baseline.value, learner.value_modules["v"])):
            for key, value in expected.state_dict().items():
                torch.testing.assert_close(value, actual.state_dict()[key], rtol=0, atol=1e-7)
    finally:
        guard.close()


def test_expression_parent_can_be_read_edited_trained_and_evaluated(tmp_path):
    raw = demo_contract(tmp_path, steps=3, open_code=True).model_dump(mode="json")
    raw["model"].update(provider="responses", model="test", base_url="https://example.com/v1")
    agent = Harness(ResearchContract.model_validate(raw), tmp_path / "run")
    try:
        agent._initialize()
        parent = agent.store.get("baseline")
        source = inspect_artifact(InspectAction(kind="inspect", target="candidate", id=parent,
                                               path="candidate.py", max_lines=150),
                                  agent.store, agent.task, contract=agent.contract)
        assert source["source_origin"] == "iql-source"
        assert "def actor_weights" in source["text"]
        for schema in ("edit_algorithm", "expression"):
            assert inspect_artifact(InspectAction(kind="inspect", target="schema", id=schema),
                                    agent.store, agent.task, contract=agent.contract)["text"]
        listing = inspect_artifact(InspectAction(kind="inspect", target="template"),
                                   agent.store, agent.task, contract=agent.contract)
        assert "iql-source" in listing["text"]
        hypothesis = Hypothesis(question="Rank weighting?", prediction="Unknown", rationale="Change mechanism",
                                falsification="No matched gain")
        review = agent._execute(Decision(summary="review", action={
            "kind": "review_proposal", "parent": parent, "hypothesis": hypothesis.model_dump(),
            "method_change": "Rank weighting", "why_not_tuning": "Order statistics replace magnitude",
            "feasibility": "Source edit", "failure_mode": "Lose cardinal information", "verdict": "proceed"}))
        text = template("iql-source")["files"]["candidate.py"]
        old = text[text.index("    def actor_weights"):text.index("    def value_step")]
        new = '''    def actor_weights(self, advantage):
        _, inverse, counts = torch.unique(advantage, sorted=True, return_inverse=True, return_counts=True)
        ends = counts.cumsum(0).to(advantage.dtype) - 1
        ranks = (ends - (counts.to(advantage.dtype) - 1) / 2)[inverse]
        weights = 2 * ranks / max(len(advantage) - 1, 1)
        return weights / weights.mean().clamp_min(1e-12)

'''
        edited = agent._execute(Decision(summary="implement", action={
            "kind": "edit_algorithm", "parent": parent, "name": "rank-iql",
            "hypothesis": hypothesis.model_dump(), "review_id": review.details["review_id"],
            "edits": [{"kind": "replace", "path": "candidate.py", "old": old, "new": new}]}))
        cid = edited.details["candidate"]
        assert agent.store.candidate(cid)["recipe"]["config"] == agent.store.candidate(parent)["recipe"]
        checked = agent._execute(Decision(summary="smoke", action={
            "kind": "check_candidate", "candidate": cid, "case": "lq-dev", "train_seed": 0, "steps": 3}))
        assert checked.status == "ok"
        assert checked.details["optimizer_steps"] == 9
        assert checked.details["purpose"] == "check"
    finally:
        agent.close()
