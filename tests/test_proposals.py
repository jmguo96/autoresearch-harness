from __future__ import annotations

import pytest

from arlab.cli import demo_contract
from arlab.core.harness import Harness
from arlab.core.proposals import method_changed
from arlab.core.schema import Decision, Hypothesis, ProposeAction, ResearchContract, ReviewProposalAction


def test_scalar_iql_tuning_is_not_a_method_change(tmp_path):
    parent = demo_contract(tmp_path).seed_recipe
    tuned = {**parent, "weight_cap": 20.0, "name": "better-iql"}
    assert not method_changed(parent, tuned)
    changed = {**parent, "actor_weight": {"op": "const", "value": 1}}
    assert method_changed(parent, changed)


def test_real_model_requires_matching_approved_review_before_candidate(tmp_path):
    raw = demo_contract(tmp_path).model_dump(mode="json")
    raw["model"].update(provider="responses", model="gpt-5.6-sol",
                        base_url="https://example.com/v1")
    contract = ResearchContract.model_validate(raw)
    agent = Harness(contract, tmp_path / "run")
    try:
        agent._initialize()
        baseline = agent.store.get("baseline")
        hypothesis = Hypothesis(question="New actor objective?", prediction="May help",
                                rationale="Reduce extrapolation", falsification="No matched gain")
        recipe = {**raw["seed_recipe"], "actor_weight": {"op": "exp", "args": [{"var": "adv"}]}}
        def propose(review_id=""):
            return Decision(summary="candidate", action=ProposeAction(
                kind="propose", parent=baseline, hypothesis=hypothesis,
                recipe=recipe, review_id=review_id))
        with pytest.raises(ValueError, match="approved review_proposal"):
            agent._execute(propose())
        rejected = agent._execute(Decision(summary="review", action=ReviewProposalAction(
            kind="review_proposal", parent=baseline, hypothesis=hypothesis,
            method_change="New actor objective", why_not_tuning="Objective changes",
            feasibility="Simple expression", failure_mode="Underfit", verdict="reject")))
        with pytest.raises(ValueError, match="approved review_proposal"):
            agent._execute(propose(rejected.details["review_id"]))
        approved = agent._execute(Decision(summary="review", action=ReviewProposalAction(
            kind="review_proposal", parent=baseline, hypothesis=hypothesis,
            method_change="New actor objective", why_not_tuning="Objective changes",
            feasibility="Simple expression", failure_mode="Underfit", verdict="proceed")))
        assert agent._context()["proposal_review"]["id"] == approved.details["review_id"]
        scalar = {**raw["seed_recipe"], "weight_cap": 20.0}
        with pytest.raises(ValueError, match="Only scalar hyperparameters"):
            agent._execute(Decision(summary="tuning", action=ProposeAction(
                kind="propose", parent=baseline, hypothesis=hypothesis,
                recipe=scalar, review_id=approved.details["review_id"])))
        result = agent._execute(propose(approved.details["review_id"]))
        assert result.details["candidate"]
        assert agent.store.get("proposal_review") is None
        with pytest.raises(ValueError, match="approved review_proposal"):
            agent._execute(propose(approved.details["review_id"]))
    finally:
        agent.close()
